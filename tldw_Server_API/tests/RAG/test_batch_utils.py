"""
Tests for batch_utils module (semaphore-based batch execution).

These tests cover:
- run_batch with all successes, partial failures, fail_fast, empty input, serial execution
- run_batch with progress callback
- run_batch_indexed verifying (index, item) is passed to the callable
- BatchResult properties: success_rate and has_errors
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

import tldw_Server_API.app.core.RAG.rag_service.batch_utils as batch_utils
from tldw_Server_API.app.core.RAG.rag_service.batch_utils import (
    BatchResult,
    run_batch,
    run_batch_indexed,
)


def _capture_batch_logs(level: str = "WARNING") -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = batch_utils.logger.add(
        lambda message: messages.append(str(message.record.get("message") or "")),
        level=level,
    )
    return messages, sink_id


# ---------------------------------------------------------------------------
# BatchResult dataclass tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBatchResult:
    """Tests for the BatchResult dataclass and its properties."""

    def test_success_rate_all_completed(self):
        """success_rate should be 1.0 when all items complete."""
        result = BatchResult(results=[1, 2, 3], errors=[], total=3, completed=3)
        assert result.success_rate == pytest.approx(1.0)

    def test_success_rate_partial_completion(self):
        """success_rate should reflect partial completion."""
        err = ValueError("boom")
        result = BatchResult(
            results=[1, 2],
            errors=[(2, err)],
            total=3,
            completed=2,
        )
        assert result.success_rate == pytest.approx(2.0 / 3.0)

    def test_success_rate_zero_total(self):
        """success_rate should be 0.0 when total is 0 (avoid division by zero)."""
        result = BatchResult(total=0, completed=0)
        assert result.success_rate == 0.0

    def test_has_errors_true(self):
        """has_errors should be True when the errors list is non-empty."""
        result = BatchResult(errors=[(0, ValueError("x"))], total=1, completed=0)
        assert result.has_errors is True

    def test_has_errors_false(self):
        """has_errors should be False when the errors list is empty."""
        result = BatchResult(results=[42], errors=[], total=1, completed=1)
        assert result.has_errors is False

    def test_default_fields(self):
        """Default BatchResult should have empty lists and zeroed counters."""
        result = BatchResult()
        assert result.results == []
        assert result.results_by_index == {}
        assert result.errors == []
        assert result.total == 0
        assert result.completed == 0
        assert result.cancelled is False

    def test_ordered_results_with_errors_prefers_index_map(self):
        """ordered_results_with_errors should use results_by_index when provided."""
        err = ValueError("boom")
        result = BatchResult(
            results=[],
            results_by_index={0: "a", 2: "c"},
            errors=[(1, err)],
            total=3,
            completed=2,
        )
        ordered = result.ordered_results_with_errors(default=None)
        assert ordered[0] == "a"
        assert isinstance(ordered[1], ValueError)
        assert ordered[2] == "c"


# ---------------------------------------------------------------------------
# run_batch tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
class TestRunBatch:
    """Tests for run_batch."""

    async def test_all_successes_preserves_order(self):
        """Results should appear in the same order as input items."""
        items = [10, 20, 30, 40, 50]

        async def double(x: int) -> int:
            await asyncio.sleep(0)  # yield to event loop
            return x * 2

        result = await run_batch(items, double, max_concurrency=3)

        assert result.results == [20, 40, 60, 80, 100]
        assert result.results_by_index == {0: 20, 1: 40, 2: 60, 3: 80, 4: 100}
        assert result.total == 5
        assert result.completed == 5
        assert result.cancelled is False
        assert result.has_errors is False
        assert result.success_rate == pytest.approx(1.0)

    async def test_some_failures_records_errors(self):
        """Errors list should contain (index, exception) for failed items."""
        items = [1, 2, 3, 4, 5]

        async def maybe_fail(x: int) -> int:
            if x % 2 == 0:
                raise ValueError(f"even: {x}")
            return x

        result = await run_batch(items, maybe_fail, max_concurrency=5, fail_fast=False)

        # Successful results: items at indices 0, 2, 4 (values 1, 3, 5)
        assert result.results == [1, 3, 5]
        assert result.results_by_index == {0: 1, 2: 3, 4: 5}
        assert result.completed == 3
        assert result.total == 5
        assert result.cancelled is False
        assert result.has_errors is True

        error_indices = sorted(idx for idx, _ in result.errors)
        assert error_indices == [1, 3]
        for idx, exc in result.errors:
            assert isinstance(exc, ValueError)

    async def test_failure_log_sanitizes_exception_details(self):
        """Failure logs should not expose raw exception details."""
        secret_path = "/private/rag-batch.db?token=secret-batch-token"
        exc = RuntimeError(f"batch backend failed at {secret_path}")
        messages, sink_id = _capture_batch_logs()

        async def fail(_item: int) -> int:
            raise exc

        try:
            result = await run_batch([1], fail, max_concurrency=1, fail_fast=False)
        finally:
            batch_utils.logger.remove(sink_id)

        assert result.errors[0][1] is exc
        assert secret_path in str(result.errors[0][1])
        joined = "\n".join(messages)
        assert "Batch item 0 failed" in joined
        assert "rag-batch.db" not in joined
        assert "secret-batch-token" not in joined

    async def test_fail_fast_log_sanitizes_exception_details(self):
        """fail_fast cancellation logs should not expose raw exception details."""
        secret_path = "/private/rag-batch-fail-fast.db?token=secret-fail-fast-token"
        exc = RuntimeError(f"fail_fast backend failed at {secret_path}")
        messages, sink_id = _capture_batch_logs()

        async def fail(_item: int) -> int:
            raise exc

        try:
            result = await run_batch([1, 2], fail, max_concurrency=1, fail_fast=True)
        finally:
            batch_utils.logger.remove(sink_id)

        assert result.cancelled is True
        assert result.errors[0][1] is exc
        assert secret_path in str(result.errors[0][1])
        joined = "\n".join(messages)
        assert "Batch item 0 failed (fail_fast=True), cancelling remaining" in joined
        assert "rag-batch-fail-fast.db" not in joined
        assert "secret-fail-fast-token" not in joined

    async def test_fail_fast_cancels_remaining(self):
        """With fail_fast=True the batch should set cancelled and stop early."""
        call_log: list[int] = []

        async def slow_then_fail(x: int) -> int:
            call_log.append(x)
            if x == 0:
                # First item fails immediately
                raise RuntimeError("fail-fast trigger")
            # Remaining items sleep long enough to still be pending
            await asyncio.sleep(5)
            return x

        items = list(range(5))
        result = await run_batch(items, slow_then_fail, max_concurrency=1, fail_fast=True)

        assert result.cancelled is True
        assert result.has_errors is True
        # With concurrency=1, the first item (index 0) should fail and cancel the rest.
        # completed should be 0 because the only item that ran raised an error.
        assert result.completed == 0
        assert len(result.errors) >= 1
        # The first error should be index 0
        first_err_idx, first_err_exc = result.errors[0]
        assert first_err_idx == 0
        assert isinstance(first_err_exc, RuntimeError)

    async def test_empty_items(self):
        """Passing an empty list should return a zero-count BatchResult immediately."""
        async def should_not_be_called(x: int) -> int:
            raise AssertionError("should not be called")

        result = await run_batch([], should_not_be_called)

        assert result.results == []
        assert result.results_by_index == {}
        assert result.errors == []
        assert result.total == 0
        assert result.completed == 0
        assert result.cancelled is False

    async def test_serial_execution_max_concurrency_one(self):
        """max_concurrency=1 should process items one at a time."""
        execution_order: list[int] = []

        async def record_order(x: int) -> int:
            execution_order.append(x)
            await asyncio.sleep(0)
            return x

        items = [0, 1, 2, 3, 4]
        result = await run_batch(items, record_order, max_concurrency=1)

        assert result.results == [0, 1, 2, 3, 4]
        assert result.results_by_index == {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}
        assert result.completed == 5
        # With concurrency=1, items should execute in order
        assert execution_order == [0, 1, 2, 3, 4]

    async def test_progress_callback_invoked(self):
        """on_progress should be called with (completed_count, total) after each success."""
        progress_calls: list[tuple[int, int]] = []

        def on_progress(completed: int, total: int) -> None:
            progress_calls.append((completed, total))

        items = ["a", "b", "c"]

        async def identity(x: str) -> str:
            return x

        result = await run_batch(
            items, identity, max_concurrency=1, on_progress=on_progress,
        )

        assert result.completed == 3
        assert result.total == 3
        # Callback should have been called once per successful item
        assert len(progress_calls) == 3
        # All calls should report total=3
        for _, total in progress_calls:
            assert total == 3
        # The completed values should include 1, 2, 3 (order depends on execution)
        completed_values = sorted(c for c, _ in progress_calls)
        assert completed_values == [1, 2, 3]

    async def test_async_progress_callback(self):
        """on_progress can be an async-compatible coroutine and should still work."""
        progress_calls: list[tuple[int, int]] = []

        async def on_progress_async(completed: int, total: int) -> None:
            progress_calls.append((completed, total))

        items = [1, 2]

        async def identity(x: int) -> int:
            return x

        result = await run_batch(
            items, identity, max_concurrency=1, on_progress=on_progress_async,
        )

        assert result.completed == 2
        assert len(progress_calls) == 2

    async def test_max_concurrency_clamped_to_one(self):
        """max_concurrency < 1 should be clamped to 1 and not raise."""
        items = [10, 20]

        async def identity(x: int) -> int:
            return x

        result = await run_batch(items, identity, max_concurrency=0)
        assert result.results == [10, 20]
        assert result.completed == 2

        result2 = await run_batch(items, identity, max_concurrency=-5)
        assert result2.results == [10, 20]
        assert result2.completed == 2


# ---------------------------------------------------------------------------
# run_batch_indexed tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
@pytest.mark.asyncio
class TestRunBatchIndexed:
    """Tests for run_batch_indexed."""

    async def test_func_receives_index_and_item(self):
        """The callable should receive (index, item) pairs."""
        received: list[tuple[int, str]] = []

        async def track(index: int, item: str) -> str:
            received.append((index, item))
            return f"{index}:{item}"

        items = ["a", "b", "c"]
        result = await run_batch_indexed(items, track, max_concurrency=1)

        assert result.results == ["0:a", "1:b", "2:c"]
        assert result.results_by_index == {0: "0:a", 1: "1:b", 2: "2:c"}
        assert result.completed == 3
        assert result.total == 3
        assert result.has_errors is False
        # Verify every (index, item) pair was received
        assert sorted(received) == [(0, "a"), (1, "b"), (2, "c")]

    async def test_indexed_with_failures(self):
        """Errors in run_batch_indexed should report the correct index."""
        async def fail_on_odd_index(index: int, item: int) -> int:
            if index % 2 == 1:
                raise ValueError(f"odd index {index}")
            return item * 10

        items = [100, 200, 300, 400]
        result = await run_batch_indexed(
            items, fail_on_odd_index, max_concurrency=5, fail_fast=False,
        )

        assert result.results == [1000, 3000]  # indices 0 and 2 succeed
        assert result.results_by_index == {0: 1000, 2: 3000}
        assert result.completed == 2
        assert result.total == 4
        error_indices = sorted(idx for idx, _ in result.errors)
        assert error_indices == [1, 3]

    async def test_indexed_failure_log_sanitizes_exception_details(self):
        """Indexed failure logs should not expose raw exception details."""
        secret_path = "/private/rag-indexed.db?token=secret-indexed-token"
        exc = RuntimeError(f"indexed backend failed at {secret_path}")
        messages, sink_id = _capture_batch_logs()

        async def fail(index: int, item: int) -> int:
            _ = (index, item)
            raise exc

        try:
            result = await run_batch_indexed([10], fail, max_concurrency=1, fail_fast=False)
        finally:
            batch_utils.logger.remove(sink_id)

        assert result.errors[0][1] is exc
        assert secret_path in str(result.errors[0][1])
        joined = "\n".join(messages)
        assert "Batch item 0 failed" in joined
        assert "rag-indexed.db" not in joined
        assert "secret-indexed-token" not in joined

    async def test_indexed_empty_items(self):
        """Empty items list should return zero-count BatchResult."""
        async def noop(index: int, item: int) -> int:
            raise AssertionError("should not be called")

        result = await run_batch_indexed([], noop)
        assert result.total == 0
        assert result.completed == 0
        assert result.results == []
        assert result.results_by_index == {}
        assert result.errors == []

    async def test_indexed_fail_fast(self):
        """fail_fast should cancel the batch on the first indexed failure."""
        async def fail_immediately(index: int, item: int) -> int:
            if index == 0:
                raise RuntimeError("stop")
            await asyncio.sleep(5)
            return item

        items = [1, 2, 3, 4, 5]
        result = await run_batch_indexed(
            items, fail_immediately, max_concurrency=1, fail_fast=True,
        )

        assert result.cancelled is True
        assert result.has_errors is True
        assert result.completed == 0

    async def test_indexed_progress_callback(self):
        """Progress callback should fire for each successful indexed item."""
        progress_calls: list[tuple[int, int]] = []

        def on_progress(completed: int, total: int) -> None:
            progress_calls.append((completed, total))

        async def identity(index: int, item: str) -> str:
            return item

        items = ["x", "y"]
        result = await run_batch_indexed(
            items, identity, max_concurrency=1, on_progress=on_progress,
        )

        assert result.completed == 2
        assert len(progress_calls) == 2
        completed_values = sorted(c for c, _ in progress_calls)
        assert completed_values == [1, 2]
