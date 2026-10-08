"""NPUCudaGraphBackend — Ascend NPU full-graph capture (torch.npu.NPUGraph).

Mirrors FullCudaGraphBackend with two differences:
  - Captures via torch.npu.graph(...) into torch.npu.NPUGraph.
  - replay_with_input_update(shape_key, seq_lens, attr_name) rebinds
    the recorded graph's input bindings for variable seq_lens at replay
    time (NPU's NPUGraph.update(...) API).

torch.npu is imported lazily inside methods so the module loads on
non-NPU hosts.
"""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager, contextmanager
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import numpy as np
import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    set_graph_pool_id,
)
from sglang.srt.model_executor.runner.shape_key import ShapeKey
from sglang.srt.model_executor.runner_backend.base_cuda_graph_backend import (
    BaseCudaGraphBackend,
)
from sglang.srt.utils import empty_context, get_bool_env_var
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.runner.base_cuda_graph_runner import (
        BaseCudaGraphRunner,
    )


class NPUCudaGraphBackend(BaseCudaGraphBackend):
    """One torch.npu.NPUGraph per shape; attention metadata captured
    inside the graph. replay_with_input_update substitutes fresh
    seq_lens without re-recording."""

    def __init__(
        self,
        cuda_graph_runner: BaseCudaGraphRunner,
        *,
        enable_memory_saver: bool = False,
    ) -> None:
        self._graphs: Dict[Any, Any] = {}
        self._outputs: Dict[Any, Any] = {}
        self._pool = None
        self._device_module = cuda_graph_runner.device_module
        self._device_id = self._device_module.current_device()
        self._tp_group = cuda_graph_runner.model_runner.tp_group
        self._capture_stream = None
        self._memory_saver_adapter: Optional[Any] = TorchMemorySaverAdapter.create(
            enable=enable_memory_saver
            and get_bool_env_var("SGLANG_MEMORY_SAVER_CUDA_GRAPH")
        )
        self._enable_torch_compile = getattr(
            cuda_graph_runner, "enable_torch_compile", False
        )

    @contextmanager
    def capture_session(self, stream):
        if self._pool is None:
            self._pool = self._device_module.graph_pool_handle()
        set_graph_pool_id(self._pool)
        self._capture_stream = stream
        try:
            yield
        finally:
            self._capture_stream = None

    def capture_one(
        self,
        shape_key: ShapeKey,
        forward_fn: Callable[[], Any],
        capture_inputs: Optional[Any] = None,
        post_warmup_hook: Optional[Callable[[], None]] = None,
    ) -> None:
        import torch_npu  # noqa: F401  (verifies NPU availability)

        # Two warmups so kernels are loaded and one-time setup is paid before capture.
        # post_warmup_hook lets the attention backend reset state that warmup mutated.
        for _ in range(2):
            self._device_module.synchronize()
            self._tp_group.barrier()
            forward_fn()
            if post_warmup_hook is not None:
                post_warmup_hook()

        graph = torch.npu.NPUGraph()

        if self._enable_torch_compile:
            skip_guard_context = torch.compiler.set_stance(skip_guard_eval_unsafe=True)
        else:
            skip_guard_context = empty_context()

        graph_ctx: Callable[..., AbstractContextManager]
        if (
            self._memory_saver_adapter is not None
            and self._memory_saver_adapter.enabled
        ):
            graph_ctx = partial(
                self._memory_saver_adapter.cuda_graph,
                tag=GPU_MEMORY_TYPE_CUDA_GRAPH,
            )
        else:
            graph_ctx = torch.npu.graph

        with (
            skip_guard_context,
            graph_ctx(
                graph,
                pool=self._pool,
                stream=self._capture_stream,
                auto_dispatch_capture=True,
            ),
        ):
            out = forward_fn()

        # Updates must not wait for the ordinary launch queue to drain:
        # a queued synchronizing op can itself be waiting for this graph's
        # update event. Keep updates off that queue while preserving the
        # graph's existing event ordering.
        graph.graph_dispatch_mode.update_stream = torch.npu.SyncLaunchStream(
            device=self._device_id
        )

        self._graphs[shape_key] = graph
        self._outputs[shape_key] = out

    def can_run(self, forward_batch: ForwardBatch, shape_key: ShapeKey) -> bool:
        return shape_key in self._graphs

    @contextmanager
    def replay_session(self):
        yield

    def replay(
        self,
        shape_key: ShapeKey,
        static_forward_batch: ForwardBatch,
        **kwargs,
    ) -> Any:
        self._graphs[shape_key].replay()
        return self._outputs[shape_key]

    def replay_with_input_update(
        self,
        shape_key: ShapeKey,
        seq_lens: Any,
        attr_name: str = None,
        attr_type: Any = None,
        cpu_update_input: list = None,
    ) -> Any:
        """Rebind seq_lens on the recorded NPU graph in a background
        thread, then replay. Used when the model is not deepseek-nsa.

        Two calling conventions:
        1. (legacy) seq_lens + attr_name + attr_type:
           Constructs cpu_update_input=[{attr_name: seq_lens}] internally.
        2. cpu_update_input: A list of {attr_name: seq_lens} dicts,
           one per speculative step.  Used by EAGLE draft runners.
        """
        if cpu_update_input is None:
            if isinstance(attr_type, torch.Tensor):
                seq_lens = torch.from_numpy(np.array(seq_lens).astype(np.int32))
            cpu_update_input = [{attr_name: seq_lens}]

        cpu_update_input = [
            (
                {
                    **update,
                    "context_lens": torch.tensor(
                        update["actual_seq_lengths_kv"], dtype=torch.int32, device="cpu"
                    ),
                }
                if "actual_seq_lengths_kv" in update
                else update
            )
            for update in cpu_update_input
        ]
        graph = self._graphs[shape_key]

        def _tensor_geometry(value, torch_npu):
            # PA scratch sizing depends on logical geometry, not a layer slice's
            # position inside the joint KV backing storage.
            return (
                str(value.device),
                str(value.dtype),
                str(value.layout),
                tuple(value.shape),
                tuple(value.stride()),
                (
                    torch_npu.get_npu_format(value)
                    if value.device.type == "npu"
                    else None
                ),
            )

        def _workspace_signature(args, kwargs, schema_names, torch_npu):
            parameter_names = (
                "query",
                "key_cache",
                "value_cache",
                "num_kv_heads",
                "num_heads",
                "scale_value",
                "block_table",
                "context_lens",
                "out",
            )
            assert set(schema_names) == {
                *parameter_names,
                "workspace",
            }, f"Unexpected PagedAttention schema: {schema_names}"

            def get_argument(name):
                index = schema_names.index(name)
                if index < len(args):
                    return args[index]
                return kwargs[name]

            query, key_cache, value_cache = (
                get_argument(name) for name in parameter_names[:3]
            )
            block_table = get_argument("block_table")
            context_lens = get_argument("context_lens")
            output = get_argument("out")
            return (
                _tensor_geometry(query, torch_npu),
                _tensor_geometry(key_cache, torch_npu),
                _tensor_geometry(value_cache, torch_npu),
                get_argument("num_kv_heads"),
                get_argument("num_heads"),
                get_argument("scale_value"),
                _tensor_geometry(block_table, torch_npu)
                + (block_table.storage_offset(), block_table.data_ptr()),
                _tensor_geometry(context_lens, torch_npu)
                + (context_lens.storage_offset(), tuple(context_lens.tolist())),
                _tensor_geometry(output, torch_npu),
            )

        def _prepare_update_inputs():
            import torch_npu

            records = graph.graph_dispatch_mode.graph_dispatch_records
            if len(cpu_update_input) == 1:
                update_inputs = [cpu_update_input[0].copy() for _ in records]
            else:
                update_inputs = [update.copy() for update in cpu_update_input]

            if len(update_inputs) != len(records):
                return update_inputs

            workspace_cache = {}
            for record, update_input in zip(records, update_inputs):
                if record.op_cache_entry.__name__ != "_npu_paged_attention.default":
                    continue
                assert (
                    "context_lens" in update_input
                ), "PagedAttention graph records require a context_lens update"
                assert (
                    "workspace" in record.kwargs
                ), "PagedAttention graph records must capture workspace as a keyword"

                args = list(record.args)
                kwargs = dict(record.kwargs)
                schema_names = [
                    argument.name
                    for argument in record.op_cache_entry._schema.arguments
                ]
                context_lens_index = schema_names.index("context_lens")
                if context_lens_index < len(args):
                    args[context_lens_index] = update_input["context_lens"]
                else:
                    kwargs["context_lens"] = update_input["context_lens"]
                kwargs.pop("workspace")
                signature = _workspace_signature(args, kwargs, schema_names, torch_npu)
                if signature not in workspace_cache:
                    workspace_cache[signature] = (
                        torch_npu._npu_paged_attention_get_workspace(*args, **kwargs)
                    )
                update_input["workspace"] = workspace_cache[signature]
            return update_inputs

        self._device_module.set_device(self._device_id)
        with self._device_module.stream(graph.graph_dispatch_mode.update_stream):
            # PagedAttention workspace size is not monotonic in context length.
            # Prepare it before replay so schema, allocation, or OOM failures are
            # raised on this calling thread instead of stranding graph.replay().
            update_inputs = _prepare_update_inputs()

        def _update():
            self._device_module.set_device(self._device_id)
            graph.update(cpu_update_input=update_inputs)

        thread = threading.Thread(target=_update)
        thread.start()
        graph.replay()
        thread.join()
        return self._outputs[shape_key]

    def cleanup(self) -> None:
        self._graphs.clear()
        self._outputs.clear()
        self._pool = None
