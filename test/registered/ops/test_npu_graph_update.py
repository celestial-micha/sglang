"""Regression for concurrent dynamic NPU graph updates and synchronizing ops."""

import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.utils.common import is_npu
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=45, suite="base-b-test-1-npu-a3")


def exercise_graph_update():
    import torch
    import torch_npu
    import torch_npu.npu.graphs as graph_api

    from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
        NPUCudaGraphBackend,
    )

    torch.npu.set_device(0)
    torch.manual_seed(1234)
    q = torch.randn(1, 4, 1, 64, device="npu", dtype=torch.float16)
    k = torch.randn(1, 4, 128, 64, device="npu", dtype=torch.float16)
    v = torch.randn_like(k)

    def attention(length):
        return torch_npu.npu_fused_infer_attention_score(
            q,
            k,
            v,
            num_heads=4,
            num_key_value_heads=4,
            input_layout="BNSD",
            scale=0.125,
            actual_seq_lengths_kv=[length],
        )[0]

    runner = SimpleNamespace(
        device_module=torch.npu,
        model_runner=SimpleNamespace(tp_group=SimpleNamespace(barrier=lambda: None)),
    )
    backend = NPUCudaGraphBackend(runner)
    backend.capture_one("test", lambda: attention(128))
    graph = backend._graphs["test"]
    original_end = graph_api.graph_task_update_end
    original_replay = graph.replay
    for length in (64, 96, 128):
        reached = threading.Event()
        release = threading.Event()
        submitted = threading.Event()
        errors = []
        outputs = []

        def delayed_end(stream):
            reached.set()
            if not release.wait(10):
                raise TimeoutError("Test did not release the graph update")
            original_end(stream)

        def replay():
            original_replay()
            submitted.set()

        def run_graph():
            try:
                torch.npu.set_device(0)
                outputs.append(
                    backend.replay_with_input_update(
                        "test", [length], attr_name="actual_seq_lengths_kv"
                    )
                )
            except BaseException as exc:
                errors.append(exc)

        def synchronizing_op():
            try:
                torch.npu.set_device(0)
                # Nonzero synchronizes inside the launch-queue consumer.
                indices = q.nonzero()
                assert indices.shape[0] == q.numel()
            except BaseException as exc:
                errors.append(exc)

        graph_api.graph_task_update_end = delayed_end
        graph.replay = replay
        replay_thread = threading.Thread(target=run_graph, daemon=True)
        replay_thread.start()
        assert reached.wait(10), "Graph update did not start"
        assert submitted.wait(10), "Graph replay was not submitted"
        copy_thread = threading.Thread(target=synchronizing_op, daemon=True)
        copy_thread.start()
        time.sleep(0.1)
        release.set()
        replay_thread.join(10)
        copy_thread.join(10)
        assert not replay_thread.is_alive() and not copy_thread.is_alive()
        assert not errors, errors
        graph_api.graph_task_update_end = original_end
        graph.replay = original_replay
        torch.npu.synchronize()
        torch.testing.assert_close(outputs[0], attention(length), rtol=1e-3, atol=1e-3)
    backend.cleanup()


def exercise_paged_graph_update() -> None:
    import torch
    import torch_npu

    from sglang.srt.hardware_backend.npu.graph_runner.npu_cudagraph_backend import (
        NPUCudaGraphBackend,
    )

    torch.npu.set_device(0)
    generator = torch.Generator(device="cpu").manual_seed(19)
    batch, blocks, page, dim = 2, 7, 32, 128
    for heads, kv_heads in ((4, 4), (32, 8)):
        query = torch.randn(batch, heads, dim, generator=generator).bfloat16().to("npu")
        key_backing = (
            torch.randn(2, batch * blocks, page, kv_heads, dim, generator=generator)
            .bfloat16()
            .to("npu")
        )
        value_backing = (
            torch.randn(2, batch * blocks, page, kv_heads, dim, generator=generator)
            .bfloat16()
            .to("npu")
        )
        key, shared_key = key_backing[0], key_backing[1]
        value, shared_value = value_backing[0], value_backing[1]
        assert key.storage_offset() != shared_key.storage_offset()
        assert value.storage_offset() != shared_value.storage_offset()
        assert key.shape == shared_key.shape and key.stride() == shared_key.stride()
        assert (
            value.shape == shared_value.shape
            and value.stride() == shared_value.stride()
        )
        assert torch_npu.get_npu_format(key) == torch_npu.get_npu_format(shared_key)
        assert torch_npu.get_npu_format(value) == torch_npu.get_npu_format(shared_value)
        shared_query = (
            torch.randn(batch, heads, dim, generator=generator).bfloat16().to("npu")
        )
        distinct_query = (
            torch.randn(batch, heads, dim, generator=generator).bfloat16().to("npu")
        )
        distinct_key = (
            torch.randn(batch * blocks, page, kv_heads, dim, generator=generator)
            .bfloat16()
            .to("npu")
        )
        distinct_value = (
            torch.randn(batch * blocks, page, kv_heads, dim, generator=generator)
            .bfloat16()
            .to("npu")
        )
        table = torch.arange(batch * blocks, dtype=torch.int32, device="npu").reshape(
            batch, blocks
        )
        captured_lengths = torch.ones(batch, dtype=torch.int32, device="cpu")
        output = torch.empty_like(query)
        params = dict(
            query=query,
            key_cache=key,
            value_cache=value,
            num_heads=heads,
            num_kv_heads=kv_heads,
            scale_value=dim**-0.5,
            block_table=table,
            context_lens=captured_lengths,
            out=output,
        )
        workspace = torch_npu._npu_paged_attention_get_workspace(**params)
        shared_output = torch.empty_like(shared_query)
        shared_params = {
            **params,
            "query": shared_query,
            "key_cache": shared_key,
            "value_cache": shared_value,
            "out": shared_output,
        }
        shared_workspace = torch_npu._npu_paged_attention_get_workspace(**shared_params)
        assert workspace.numel() == shared_workspace.numel()
        assert workspace.element_size() == shared_workspace.element_size()
        distinct_output = torch.empty_like(distinct_query)
        distinct_params = {
            **params,
            "query": distinct_query,
            "key_cache": distinct_key,
            "value_cache": distinct_value,
            "scale_value": (dim**-0.5) / 2,
            "out": distinct_output,
        }
        distinct_workspace = torch_npu._npu_paged_attention_get_workspace(
            **distinct_params
        )
        fia_query = query.view(batch, 1, heads * dim)
        fia_output = torch.empty_like(fia_query)
        fia_lse = torch.empty(1, dtype=query.dtype, device=query.device)
        fia_params = dict(
            block_table=table,
            block_size=page,
            num_heads=heads,
            num_key_value_heads=kv_heads,
            input_layout="BSH",
            scale=dim**-0.5,
            actual_seq_lengths_kv=captured_lengths.tolist(),
        )
        fia_workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
            fia_query,
            key.view(-1, page, kv_heads * dim),
            value.view(-1, page, kv_heads * dim),
            **fia_params,
        )

        def fia(lengths: list[int]) -> torch.Tensor:
            return torch_npu.npu_fused_infer_attention_score(
                fia_query,
                key.view(-1, page, kv_heads * dim),
                value.view(-1, page, kv_heads * dim),
                block_table=table,
                block_size=page,
                num_heads=heads,
                num_key_value_heads=kv_heads,
                input_layout="BSH",
                scale=dim**-0.5,
                actual_seq_lengths_kv=lengths,
            )[0]

        def attention() -> tuple[torch.Tensor, ...]:
            torch_npu._npu_paged_attention(**params, workspace=workspace)
            torch_npu._npu_paged_attention(**shared_params, workspace=workspace)
            torch_npu._npu_paged_attention(
                **distinct_params, workspace=distinct_workspace
            )
            torch_npu.npu_fused_infer_attention_score.out(
                fia_query,
                key.view(-1, page, kv_heads * dim),
                value.view(-1, page, kv_heads * dim),
                **fia_params,
                workspace=fia_workspace,
                out=[fia_output, fia_lse],
            )
            return output, shared_output, distinct_output, fia_output

        runner = SimpleNamespace(
            device_module=torch.npu,
            model_runner=SimpleNamespace(
                tp_group=SimpleNamespace(barrier=lambda: None)
            ),
        )
        backend = NPUCudaGraphBackend(runner)
        print(f"capture heads={heads} kv_heads={kv_heads}", flush=True)
        backend.capture_one("mixed", attention)
        print("capture complete", flush=True)
        records = backend._graphs["mixed"].graph_dispatch_mode.graph_dispatch_records
        assert len(records) == 4
        for update_index, lengths in enumerate(
            ([1, 31], [32, 33], [193, 0], [129, 193])
        ):
            query.add_(0.125)
            key.add_(0.03125)
            value.mul_(1.0625)
            shared_query.sub_(0.0625)
            shared_key.mul_(0.9375)
            shared_value.add_(0.015625)
            distinct_query.mul_(1.03125)
            distinct_key.sub_(0.0078125)
            distinct_value.add_(0.046875)
            table.copy_(table.flip(0))
            torch.npu.synchronize()
            print(f"update lengths={lengths}", flush=True)
            get_workspace = torch_npu._npu_paged_attention_get_workspace
            workspace_calls = 0

            def counted_get_workspace(*args, **kwargs):
                nonlocal workspace_calls
                workspace_calls += 1
                return get_workspace(*args, **kwargs)

            torch_npu._npu_paged_attention_get_workspace = counted_get_workspace
            try:
                if update_index % 2 == 0:
                    captured = backend.replay_with_input_update(
                        "mixed",
                        lengths,
                        attr_name="actual_seq_lengths_kv",
                        attr_type=[],
                    )
                else:
                    captured = backend.replay_with_input_update(
                        "mixed",
                        seq_lens=None,
                        cpu_update_input=[
                            {"actual_seq_lengths_kv": lengths} for _ in records
                        ],
                    )
            finally:
                torch_npu._npu_paged_attention_get_workspace = get_workspace
            assert workspace_calls == 2, (
                "Equal PA signatures must share one workspace while a different "
                f"scale must not share it, got {workspace_calls} Setup calls"
            )
            actual = tuple(tensor.detach().cpu().clone() for tensor in captured)
            print("graph copied", flush=True)
            expected_lengths = torch.tensor(lengths, dtype=torch.int32, device="cpu")
            expected_params = {**params, "context_lens": expected_lengths}
            expected_workspace = torch_npu._npu_paged_attention_get_workspace(
                **expected_params
            )
            torch_npu._npu_paged_attention(
                **expected_params, workspace=expected_workspace
            )
            expected_shared_params = {
                **shared_params,
                "context_lens": expected_lengths,
            }
            torch_npu._npu_paged_attention(
                **expected_shared_params, workspace=expected_workspace
            )
            expected_distinct_params = {
                **distinct_params,
                "context_lens": expected_lengths,
            }
            expected_distinct_workspace = torch_npu._npu_paged_attention_get_workspace(
                **expected_distinct_params
            )
            torch_npu._npu_paged_attention(
                **expected_distinct_params, workspace=expected_distinct_workspace
            )
            expected = (
                output.cpu().clone(),
                shared_output.cpu().clone(),
                distinct_output.cpu().clone(),
                fia(lengths).cpu(),
            )
            for graph_output, eager_output in zip(actual, expected):
                assert torch.isfinite(graph_output).all()
                torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)
            assert (
                captured_lengths == 1
            ).all(), "Captured lengths must remain immutable"

        graph = backend._graphs["mixed"]
        original_replay = graph.replay
        original_get_workspace = torch_npu._npu_paged_attention_get_workspace
        replay_called = False

        def replay():
            nonlocal replay_called
            replay_called = True
            return original_replay()

        def fail_get_workspace(*args, **kwargs):
            raise RuntimeError("workspace preparation failed")

        graph.replay = replay
        torch_npu._npu_paged_attention_get_workspace = fail_get_workspace
        try:
            try:
                backend.replay_with_input_update(
                    "mixed",
                    [32, 33],
                    attr_name="actual_seq_lengths_kv",
                    attr_type=[],
                )
            except RuntimeError as error:
                assert str(error) == "workspace preparation failed"
            else:
                raise AssertionError("Workspace preparation failure was not raised")
        finally:
            graph.replay = original_replay
            torch_npu._npu_paged_attention_get_workspace = original_get_workspace
        assert (
            not replay_called
        ), "Graph replay started after workspace preparation failed"
        backend.cleanup()


@unittest.skipUnless(is_npu(), "requires Ascend NPU")
class TestNPUGraphUpdate(unittest.TestCase):
    def test_paged_and_fia_graphs_update_lengths(self):
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--exercise-paged"],
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_synchronizing_op_does_not_deadlock_update(self):
        # A deadlocked native thread can hold the GIL; isolate it so the test
        # times out and fails instead of hanging the complete CI worker.
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--exercise"],
            capture_output=True,
            text=True,
            timeout=90,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    if "--exercise-paged" in sys.argv:
        exercise_paged_graph_update()
    elif "--exercise" in sys.argv:
        exercise_graph_update()
    else:
        unittest.main()
