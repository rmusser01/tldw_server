"""Tests for control flow adapters.

This module tests all 11 control adapters:
1. run_prompt_adapter - Render a prompt template
2. run_delay_adapter - Delay execution
3. run_log_adapter - Log output
4. run_branch_adapter - Conditional branching
5. run_map_adapter - Map over array items
6. run_parallel_adapter - Parallel execution marker
7. run_batch_adapter - Batch processing
8. run_cache_result_adapter - Cache a result
9. run_retry_adapter - Retry logic
10. run_checkpoint_adapter - Workflow checkpoint
11. run_workflow_call_adapter - Call another workflow
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tldw_Server_API.app.core.Workflows.adapters import (
    run_prompt_adapter,
    run_delay_adapter,
    run_log_adapter,
    run_branch_adapter,
    run_map_adapter,
    run_parallel_adapter,
    run_batch_adapter,
    run_cache_result_adapter,
    run_retry_adapter,
    run_checkpoint_adapter,
    run_workflow_call_adapter,
)

pytestmark = pytest.mark.unit


# =============================================================================
# Prompt Adapter Tests
# =============================================================================


class TestPromptAdapter:
    """Tests for run_prompt_adapter."""

    @pytest.mark.asyncio
    async def test_renders_simple_template(self):
        """Test prompt adapter renders a simple Jinja template."""
        config = {"template": "Hello {{ inputs.name }}!"}
        context = {"inputs": {"name": "World"}}

        result = await run_prompt_adapter(config, context)

        assert result["text"] == "Hello World!"

    @pytest.mark.asyncio
    async def test_renders_template_with_prompt_alias(self):
        """Test prompt adapter accepts 'prompt' as alias for 'template'."""
        config = {"prompt": "Greetings, {{ inputs.user }}!"}
        context = {"inputs": {"user": "Alice"}}

        result = await run_prompt_adapter(config, context)

        assert result["text"] == "Greetings, Alice!"

    @pytest.mark.asyncio
    async def test_renders_template_with_variables(self):
        """Test prompt adapter merges variables into context."""
        config = {
            "template": "{{ greeting }}, {{ inputs.name }}!",
            "variables": {"greeting": "Hello"},
        }
        context = {"inputs": {"name": "Bob"}}

        result = await run_prompt_adapter(config, context)

        assert "Hello" in result["text"]
        assert "Bob" in result["text"]

    @pytest.mark.asyncio
    async def test_handles_empty_template(self):
        """Test prompt adapter handles empty template gracefully."""
        config = {"template": ""}
        context = {"inputs": {}}

        result = await run_prompt_adapter(config, context)

        assert result["text"] == ""

    @pytest.mark.asyncio
    async def test_handles_missing_input_key(self):
        """Test prompt adapter handles missing input key with fallback."""
        config = {"template": "Hello {{ inputs.name || '' }}!"}
        context = {"inputs": {}}

        result = await run_prompt_adapter(config, context)

        # Should use fallback empty string
        assert "Hello" in result["text"]

    @pytest.mark.asyncio
    async def test_complex_nested_template(self):
        """Test prompt adapter handles nested template variables."""
        config = {"template": "User: {{ inputs.user }}, Items: {{ inputs.count }}"}
        context = {"inputs": {"user": "Charlie", "count": 42}}

        result = await run_prompt_adapter(config, context)

        assert "Charlie" in result["text"]
        assert "42" in result["text"]

    @pytest.mark.asyncio
    async def test_simulated_delay(self):
        """Test prompt adapter simulated delay for testing."""
        config = {"template": "Test", "simulate_delay_ms": 50}
        context = {"inputs": {}}

        start = time.time()
        result = await run_prompt_adapter(config, context)
        elapsed = time.time() - start

        assert result["text"] == "Test"
        assert elapsed >= 0.04  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_force_error(self):
        """Test prompt adapter force_error raises AdapterError."""
        from tldw_Server_API.app.core.exceptions import AdapterError

        config = {"template": "Test", "force_error": True}
        context = {"inputs": {}}

        with pytest.raises(AdapterError, match="forced_error"):
            await run_prompt_adapter(config, context)

    @pytest.mark.asyncio
    async def test_cancellation_during_delay(self):
        """Test prompt adapter respects cancellation during simulated delay."""
        cancelled = False

        def is_cancelled():
            return cancelled

        config = {"template": "Test", "simulate_delay_ms": 500}
        context = {"inputs": {}, "is_cancelled": is_cancelled}

        async def cancel_after():
            nonlocal cancelled
            await asyncio.sleep(0.05)
            cancelled = True

        task = asyncio.create_task(run_prompt_adapter(config, context))
        await asyncio.gather(task, cancel_after())

        result = task.result()
        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_merge_fallback_log_sanitizes_backend_error(self):
        """Test prompt fallback logs hide raw backend exception details."""
        from tldw_Server_API.app.core.Workflows.adapters.control import flow

        class BrokenVariables:
            def keys(self):
                raise RuntimeError("merge exploded at /private/flow.env")

        messages = []
        sink_id = flow.logger.add(lambda message: messages.append(str(message)), level="DEBUG")
        try:
            result = await run_prompt_adapter(
                {"template": "OK", "variables": BrokenVariables()},
                {"inputs": {}},
            )
        finally:
            flow.logger.remove(sink_id)

        assert result == {"text": "OK"}
        joined = "\n".join(messages)
        assert "Prompt adapter: failed to merge variables into context" in joined
        assert "merge exploded at /private/flow.env" not in joined


# =============================================================================
# Delay Adapter Tests
# =============================================================================


class TestDelayAdapter:
    """Tests for run_delay_adapter."""

    @pytest.mark.asyncio
    async def test_delays_specified_milliseconds(self):
        """Test delay adapter waits for specified milliseconds."""
        config = {"milliseconds": 100}
        context = {}

        start = time.time()
        result = await run_delay_adapter(config, context)
        elapsed = time.time() - start

        assert result["delayed_ms"] == 100
        assert elapsed >= 0.09  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_default_delay(self):
        """Test delay adapter uses default 1000ms if not specified."""
        config = {}
        context = {}

        # Use short delay for test
        config["milliseconds"] = 50

        result = await run_delay_adapter(config, context)

        assert result["delayed_ms"] == 50

    @pytest.mark.asyncio
    async def test_zero_delay(self):
        """Test delay adapter handles zero milliseconds."""
        config = {"milliseconds": 0}
        context = {}

        start = time.time()
        result = await run_delay_adapter(config, context)
        elapsed = time.time() - start

        assert result["delayed_ms"] == 0
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_cancellation_during_delay(self):
        """Test delay adapter respects cancellation."""
        cancelled = False

        def is_cancelled():
            return cancelled

        config = {"milliseconds": 500}
        context = {"is_cancelled": is_cancelled}

        async def cancel_after():
            nonlocal cancelled
            await asyncio.sleep(0.05)
            cancelled = True

        task = asyncio.create_task(run_delay_adapter(config, context))
        await asyncio.gather(task, cancel_after())

        result = task.result()
        assert result.get("__status__") == "cancelled"


# =============================================================================
# Log Adapter Tests
# =============================================================================


class TestLogAdapter:
    """Tests for run_log_adapter."""

    @pytest.mark.asyncio
    async def test_logs_simple_message(self):
        """Test log adapter logs a simple message."""
        config = {"message": "Hello, World!", "level": "info"}
        context = {"inputs": {}}

        result = await run_log_adapter(config, context)

        assert result["logged"] is True
        assert result["message"] == "Hello, World!"
        assert result["level"] == "info"

    @pytest.mark.asyncio
    async def test_logs_templated_message(self):
        """Test log adapter renders template in message."""
        config = {"message": "User {{ inputs.name }} logged in", "level": "info"}
        context = {"inputs": {"name": "Alice"}}

        result = await run_log_adapter(config, context)

        assert result["logged"] is True
        assert "Alice" in result["message"]

    @pytest.mark.asyncio
    async def test_log_levels(self):
        """Test log adapter supports different log levels."""
        context = {"inputs": {}}

        for level in ["debug", "info", "warning", "error"]:
            config = {"message": f"Test {level}", "level": level}
            result = await run_log_adapter(config, context)

            assert result["level"] == level
            assert result["logged"] is True

    @pytest.mark.asyncio
    async def test_default_log_level(self):
        """Test log adapter defaults to info level."""
        config = {"message": "Test message"}
        context = {"inputs": {}}

        result = await run_log_adapter(config, context)

        assert result["level"] == "info"

    @pytest.mark.asyncio
    async def test_handles_template_with_fallback(self):
        """Test log adapter handles template with || '' fallback."""
        config = {"message": "Value: {{ inputs.value || '' }}", "level": "info"}
        context = {"inputs": {}}

        result = await run_log_adapter(config, context)

        assert result["logged"] is True
        assert "Value:" in result["message"]


# =============================================================================
# Branch Adapter Tests
# =============================================================================


class TestBranchAdapter:
    """Tests for run_branch_adapter."""

    @pytest.mark.asyncio
    async def test_true_condition(self):
        """Test branch adapter with true condition."""
        config = {
            "condition": "true",
            "true_next": "step_a",
            "false_next": "step_b",
        }
        context = {"inputs": {}}

        result = await run_branch_adapter(config, context)

        assert result["branch"] == "true"
        assert result["__next__"] == "step_a"

    @pytest.mark.asyncio
    async def test_false_condition(self):
        """Test branch adapter with false condition."""
        config = {
            "condition": "false",
            "true_next": "step_a",
            "false_next": "step_b",
        }
        context = {"inputs": {}}

        result = await run_branch_adapter(config, context)

        assert result["branch"] == "false"
        assert result["__next__"] == "step_b"

    @pytest.mark.asyncio
    async def test_templated_condition_true(self):
        """Test branch adapter evaluates templated condition as true."""
        config = {
            "condition": "{{ inputs.enabled }}",
            "true_next": "enabled_step",
            "false_next": "disabled_step",
        }
        context = {"inputs": {"enabled": "true"}}

        result = await run_branch_adapter(config, context)

        assert result["branch"] == "true"
        assert result["__next__"] == "enabled_step"

    @pytest.mark.asyncio
    async def test_templated_condition_false(self):
        """Test branch adapter evaluates templated condition as false."""
        config = {
            "condition": "{{ inputs.enabled }}",
            "true_next": "enabled_step",
            "false_next": "disabled_step",
        }
        context = {"inputs": {"enabled": "no"}}

        result = await run_branch_adapter(config, context)

        assert result["branch"] == "false"
        assert result["__next__"] == "disabled_step"

    @pytest.mark.asyncio
    async def test_condition_truthy_values(self):
        """Test branch adapter recognizes truthy values."""
        truthy_values = ["1", "true", "yes", "y", "on", "TRUE", "Yes", "Y", "ON"]
        context = {"inputs": {}}

        for val in truthy_values:
            config = {"condition": val, "true_next": "t", "false_next": "f"}
            result = await run_branch_adapter(config, context)
            assert result["branch"] == "true", f"Expected '{val}' to be truthy"

    @pytest.mark.asyncio
    async def test_condition_falsy_values(self):
        """Test branch adapter recognizes falsy values."""
        falsy_values = ["0", "false", "no", "off", "", "random"]
        context = {"inputs": {}}

        for val in falsy_values:
            config = {"condition": val, "true_next": "t", "false_next": "f"}
            result = await run_branch_adapter(config, context)
            assert result["branch"] == "false", f"Expected '{val}' to be falsy"

    @pytest.mark.asyncio
    async def test_no_next_step_provided(self):
        """Test branch adapter when no next step is provided."""
        config = {"condition": "true", "true_next": "", "false_next": ""}
        context = {"inputs": {}}

        result = await run_branch_adapter(config, context)

        assert result["branch"] == "true"
        # When both true_next and false_next are empty strings, __next__ should not be added
        assert "__next__" not in result or result.get("__next__") == ""


# =============================================================================
# Map Adapter Tests
# =============================================================================


class TestMapAdapter:
    """Tests for run_map_adapter."""

    @pytest.mark.asyncio
    async def test_maps_over_list(self):
        """Test map adapter iterates over a list of items."""
        config = {
            "items": ["a", "b", "c"],
            "step": {"type": "prompt", "config": {"template": "Item: {{ item }}"}},
            "concurrency": 2,
        }
        context = {"inputs": {}}

        result = await run_map_adapter(config, context)

        assert result["count"] == 3
        assert len(result["results"]) == 3
        # Each result should contain the rendered text
        for r in result["results"]:
            assert "Item:" in r.get("text", "")

    @pytest.mark.asyncio
    async def test_map_with_concurrency(self):
        """Test map adapter respects concurrency limit."""
        items = list(range(10))
        config = {
            "items": items,
            "step": {"type": "delay", "config": {"milliseconds": 50}},
            "concurrency": 5,
        }
        context = {"inputs": {}}

        start = time.time()
        result = await run_map_adapter(config, context)
        elapsed = time.time() - start

        assert result["count"] == 10
        # With concurrency 5 and 10 items of 50ms each, should take ~100ms minimum
        # (2 batches * 50ms), not 500ms (10 * 50ms sequential)
        assert elapsed < 0.4  # Allow tolerance

    @pytest.mark.asyncio
    async def test_map_with_templated_items(self):
        """Test map adapter resolves templated items from context."""
        config = {
            "items": '["x", "y", "z"]',  # JSON string
            "step": {"type": "prompt", "config": {"template": "{{ item }}"}},
        }
        context = {"inputs": {}}

        result = await run_map_adapter(config, context)

        assert result["count"] == 3

    @pytest.mark.asyncio
    async def test_map_with_comma_separated_items(self):
        """Test map adapter handles comma-separated string items."""
        config = {
            "items": "apple, banana, cherry",
            "step": {"type": "prompt", "config": {"template": "{{ item }}"}},
        }
        context = {"inputs": {}}

        result = await run_map_adapter(config, context)

        assert result["count"] == 3

    @pytest.mark.asyncio
    async def test_map_missing_step_type(self):
        """Test map adapter raises error for missing step type."""
        from tldw_Server_API.app.core.exceptions import AdapterError

        config = {
            "items": ["a", "b"],
            "step": {"config": {}},  # Missing 'type'
        }
        context = {"inputs": {}}

        with pytest.raises(AdapterError, match="missing_substep_type"):
            await run_map_adapter(config, context)

    @pytest.mark.asyncio
    async def test_map_unsupported_step_type(self):
        """Test map adapter raises error for non-parallelizable step type."""
        from tldw_Server_API.app.core.exceptions import AdapterError

        config = {
            "items": ["a"],
            "step": {"type": "nonexistent_type", "config": {}},
        }
        context = {"inputs": {}}

        with pytest.raises(AdapterError, match="unsupported_substep_type"):
            await run_map_adapter(config, context)

    @pytest.mark.asyncio
    async def test_map_cancellation(self):
        """Test map adapter respects cancellation."""
        cancelled = False

        def is_cancelled():
            return cancelled

        config = {
            "items": list(range(20)),
            "step": {"type": "delay", "config": {"milliseconds": 100}},
            "concurrency": 2,
        }
        context = {"inputs": {}, "is_cancelled": is_cancelled}

        async def cancel_after():
            nonlocal cancelled
            await asyncio.sleep(0.1)
            cancelled = True

        task = asyncio.create_task(run_map_adapter(config, context))
        await asyncio.gather(task, cancel_after())

        result = task.result()
        # Some results should be cancelled
        cancelled_count = sum(
            1 for r in result.get("results", []) if r.get("__status__") == "cancelled"
        )
        assert cancelled_count > 0 or result["count"] < 20


# =============================================================================
# Parallel Adapter Tests
# =============================================================================


class TestParallelAdapter:
    """Tests for run_parallel_adapter."""

    @pytest.mark.asyncio
    async def test_executes_steps_in_parallel(self):
        """Test parallel adapter executes multiple steps concurrently."""
        config = {
            "steps": [
                {"type": "prompt", "config": {"template": "Step 1"}},
                {"type": "prompt", "config": {"template": "Step 2"}},
                {"type": "prompt", "config": {"template": "Step 3"}},
            ],
            "max_concurrency": 3,
        }
        context = {"inputs": {}}

        result = await run_parallel_adapter(config, context)

        assert result["count"] == 3
        assert len(result["results"]) == 3
        assert result["results"][0]["text"] == "Step 1"
        assert result["results"][1]["text"] == "Step 2"
        assert result["results"][2]["text"] == "Step 3"

    @pytest.mark.asyncio
    async def test_parallel_with_delays(self):
        """Test parallel adapter runs steps concurrently."""
        config = {
            "steps": [
                {"type": "delay", "config": {"milliseconds": 100}},
                {"type": "delay", "config": {"milliseconds": 100}},
                {"type": "delay", "config": {"milliseconds": 100}},
            ],
            "max_concurrency": 3,
        }
        context = {"inputs": {}}

        start = time.time()
        result = await run_parallel_adapter(config, context)
        elapsed = time.time() - start

        assert result["count"] == 3
        # All 3 should run in parallel, so ~100ms total, not 300ms
        assert elapsed < 0.25

    @pytest.mark.asyncio
    async def test_parallel_respects_max_concurrency(self):
        """Test parallel adapter respects max_concurrency limit."""
        config = {
            "steps": [
                {"type": "delay", "config": {"milliseconds": 50}},
                {"type": "delay", "config": {"milliseconds": 50}},
                {"type": "delay", "config": {"milliseconds": 50}},
                {"type": "delay", "config": {"milliseconds": 50}},
            ],
            "max_concurrency": 2,
        }
        context = {"inputs": {}}

        start = time.time()
        result = await run_parallel_adapter(config, context)
        elapsed = time.time() - start

        assert result["count"] == 4
        # With max_concurrency=2 and 4 steps of 50ms, should take ~100ms (2 batches)
        assert elapsed >= 0.09
        assert elapsed < 0.3

    @pytest.mark.asyncio
    async def test_parallel_missing_steps(self):
        """Test parallel adapter handles missing steps."""
        config = {"steps": []}
        context = {"inputs": {}}

        result = await run_parallel_adapter(config, context)

        assert result["error"] == "missing_steps"
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_parallel_unknown_step_type(self):
        """Test parallel adapter handles unknown step types gracefully."""
        config = {
            "steps": [
                {"type": "unknown_type_xyz", "config": {}},
            ],
            "max_concurrency": 1,
        }
        context = {"inputs": {}}

        result = await run_parallel_adapter(config, context)

        assert result["count"] == 1
        assert "error" in result["results"][0]

    @pytest.mark.asyncio
    async def test_parallel_cancellation(self):
        """Test parallel adapter respects cancellation."""
        cancelled = False

        def is_cancelled():
            return cancelled

        config = {
            "steps": [
                {"type": "delay", "config": {"milliseconds": 200}},
                {"type": "delay", "config": {"milliseconds": 200}},
            ],
            "max_concurrency": 2,
        }
        context = {"inputs": {}, "is_cancelled": is_cancelled}

        # Start with cancelled=True
        cancelled = True
        result = await run_parallel_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_parallel_sanitizes_substep_errors(self):
        """Test parallel adapter hides raw substep exception details."""
        config = {
            "steps": [
                {"type": "prompt", "config": {"template": "Test", "force_error": True}},
            ],
            "max_concurrency": 1,
            "fail_fast": True,
        }

        result = await run_parallel_adapter(config, {"inputs": {}})

        assert result["results"] == [{"error": "parallel_step_error"}]
        assert result["errors"] == ["parallel_step_error"]


# =============================================================================
# Batch Adapter Tests
# =============================================================================


class TestBatchAdapter:
    """Tests for run_batch_adapter."""

    @pytest.mark.asyncio
    async def test_batches_items(self):
        """Test batch adapter splits items into batches."""
        config = {
            "items": list(range(25)),
            "batch_size": 10,
        }
        context = {}

        result = await run_batch_adapter(config, context)

        assert result["batch_count"] == 3
        assert result["total_items"] == 25
        assert result["batch_size"] == 10
        assert len(result["batches"]) == 3
        assert len(result["batches"][0]) == 10
        assert len(result["batches"][1]) == 10
        assert len(result["batches"][2]) == 5

    @pytest.mark.asyncio
    async def test_batch_default_size(self):
        """Test batch adapter uses default batch size of 10."""
        config = {
            "items": list(range(15)),
        }
        context = {}

        result = await run_batch_adapter(config, context)

        assert result["batch_size"] == 10
        assert result["batch_count"] == 2

    @pytest.mark.asyncio
    async def test_batch_from_prev_context(self):
        """Test batch adapter gets items from previous step output."""
        config = {}
        context = {
            "prev": {"items": ["a", "b", "c", "d", "e"]},
        }

        result = await run_batch_adapter(config, context)

        assert result["total_items"] == 5

    @pytest.mark.asyncio
    async def test_batch_from_documents_in_prev(self):
        """Test batch adapter gets items from 'documents' in prev."""
        config = {}
        context = {
            "prev": {"documents": [{"id": 1}, {"id": 2}, {"id": 3}]},
        }

        result = await run_batch_adapter(config, context)

        assert result["total_items"] == 3

    @pytest.mark.asyncio
    async def test_batch_missing_items(self):
        """Test batch adapter handles missing items."""
        config = {}
        context = {}

        result = await run_batch_adapter(config, context)

        assert result["error"] == "missing_items"
        assert result["batch_count"] == 0

    @pytest.mark.asyncio
    async def test_batch_min_size(self):
        """Test batch adapter enforces minimum batch size of 1."""
        config = {
            "items": [1, 2, 3],
            "batch_size": 0,  # Should become 1
        }
        context = {}

        result = await run_batch_adapter(config, context)

        assert result["batch_count"] == 3  # 3 batches of size 1

    @pytest.mark.asyncio
    async def test_batch_cancellation(self):
        """Test batch adapter respects cancellation."""

        def is_cancelled():
            return True

        config = {"items": [1, 2, 3], "batch_size": 1}
        context = {"is_cancelled": is_cancelled}

        result = await run_batch_adapter(config, context)

        assert result.get("__status__") == "cancelled"


# =============================================================================
# Cache Result Adapter Tests
# =============================================================================


class TestCacheResultAdapter:
    """Tests for run_cache_result_adapter."""

    @pytest.mark.asyncio
    async def test_cache_missing_key(self):
        """Test cache adapter returns error for missing key."""
        config = {}
        context = {}

        result = await run_cache_result_adapter(config, context)

        assert result["error"] == "missing_cache_key"
        assert result["cached"] is False

    @pytest.mark.asyncio
    async def test_cache_get_or_set_fallback(self, monkeypatch):
        """Test cache adapter fallback when cache unavailable."""
        # Mock collection resolver to indicate cache unavailable
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.control.state._get_workflow_cache_collection",
            lambda _name: None,
        )

        config = {
            "key": "test_key",
            "action": "get_or_set",
        }
        context = {"prev": {"value": 42}}

        result = await run_cache_result_adapter(config, context)

        assert result["cached"] is False
        assert result["error"] == "cache_unavailable"

    @pytest.mark.asyncio
    async def test_cache_invalidate(self, monkeypatch):
        """Test cache adapter invalidate action."""
        # Mock ChromaDB client
        mock_collection = MagicMock()
        mock_collection.delete = MagicMock()

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.control.state._get_workflow_cache_collection",
            lambda _name: mock_collection,
        )

        config = {
            "key": "test_key",
            "action": "invalidate",
        }
        context = {}

        result = await run_cache_result_adapter(config, context)

        assert result["invalidated"] is True
        assert result["key"] == "test_key"
        mock_collection.delete.assert_called_once_with(ids=["test_key"])

    @pytest.mark.asyncio
    async def test_cache_store_sanitizes_backend_errors(self, monkeypatch):
        """Test cache store failures hide raw backend exception details."""
        mock_collection = MagicMock()
        mock_collection.upsert.side_effect = RuntimeError(
            "cache store exploded at /private/cache"
        )

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.control.state._get_workflow_cache_collection",
            lambda _name: mock_collection,
        )

        result = await run_cache_result_adapter(
            {"key": "test_key", "action": "set", "data": {"value": 42}},
            {},
        )

        assert result == {
            "cached": False,
            "data": {"value": 42},
            "error": "cache_store_error",
        }

    @pytest.mark.asyncio
    async def test_cache_result_sanitizes_backend_errors(self, monkeypatch):
        """Test outer cache failures hide raw backend exception details."""

        class BadCollection:
            def __bool__(self):
                raise RuntimeError("cache collection exploded at /private/cache")

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters.control.state._get_workflow_cache_collection",
            lambda _name: BadCollection(),
        )

        result = await run_cache_result_adapter({"key": "test_key"}, {})

        assert result == {"cached": False, "data": {}, "error": "cache_result_error"}

    @pytest.mark.asyncio
    async def test_cache_cancellation(self):
        """Test cache adapter respects cancellation."""

        def is_cancelled():
            return True

        config = {"key": "test_key"}
        context = {"is_cancelled": is_cancelled}

        result = await run_cache_result_adapter(config, context)

        assert result.get("__status__") == "cancelled"


# =============================================================================
# Retry Adapter Tests
# =============================================================================


class TestRetryAdapter:
    """Tests for run_retry_adapter."""

    @pytest.mark.asyncio
    async def test_retry_success_first_attempt(self, monkeypatch):
        """Test retry adapter succeeds on first attempt."""
        # Set TEST_MODE to avoid real LLM calls
        monkeypatch.setenv("TEST_MODE", "1")

        config = {
            "step_type": "prompt",
            "step_config": {"template": "Hello"},
            "max_retries": 3,
        }
        context = {"inputs": {}}

        result = await run_retry_adapter(config, context)

        assert result["success"] is True
        assert result["attempts"] == 1
        assert result["result"]["text"] == "Hello"

    @pytest.mark.asyncio
    async def test_retry_missing_step_type(self):
        """Test retry adapter returns error for missing step_type."""
        config = {
            "step_config": {},
        }
        context = {}

        result = await run_retry_adapter(config, context)

        assert result["error"] == "missing_step_type"
        assert result["result"] is None

    @pytest.mark.asyncio
    async def test_retry_unknown_step_type(self):
        """Test retry adapter handles unknown step type."""
        config = {
            "step_type": "unknown_adapter_xyz",
            "step_config": {},
        }
        context = {}

        result = await run_retry_adapter(config, context)

        assert "unknown_step_type" in result["error"]

    @pytest.mark.asyncio
    async def test_retry_on_error_with_retries(self, monkeypatch):
        """Test retry adapter retries on errors."""
        attempt_count = 0

        async def failing_adapter(config, context):
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                return {"error": "temporary_failure"}
            return {"text": "Success"}

        # Register a test adapter
        from tldw_Server_API.app.core.Workflows.adapters._registry import registry

        # Use monkeypatch to mock get_adapter
        original_get_adapter = registry.get_adapter

        def mock_get_adapter(name):
            if name == "test_retry_step":
                return failing_adapter
            return original_get_adapter(name)

        monkeypatch.setattr(registry, "get_adapter", mock_get_adapter)
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters._registry.get_adapter",
            mock_get_adapter,
        )

        config = {
            "step_type": "test_retry_step",
            "step_config": {},
            "max_retries": 5,
            "backoff_base": 0.01,  # Fast backoff for testing
        }
        context = {"inputs": {}}

        result = await run_retry_adapter(config, context)

        assert result["success"] is True
        assert result["attempts"] == 3
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted(self, monkeypatch):
        """Test retry adapter fails after exhausting retries without raw backend details."""

        async def always_failing_adapter(config, context):
            raise RuntimeError("persistent failure at /private/retry-cache")

        from tldw_Server_API.app.core.Workflows.adapters._registry import registry

        original_get_adapter = registry.get_adapter

        def mock_get_adapter(name):
            if name == "always_fail":
                return always_failing_adapter
            return original_get_adapter(name)

        monkeypatch.setattr(registry, "get_adapter", mock_get_adapter)
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Workflows.adapters._registry.get_adapter",
            mock_get_adapter,
        )

        config = {
            "step_type": "always_fail",
            "step_config": {},
            "max_retries": 2,
            "backoff_base": 0.01,
        }
        context = {"inputs": {}}

        result = await run_retry_adapter(config, context)

        assert result["success"] is False
        assert result["attempts"] == 3  # initial + 2 retries
        assert result["error"] == "retry_step_error"

    @pytest.mark.asyncio
    async def test_retry_cancellation(self):
        """Test retry adapter respects cancellation."""

        def is_cancelled():
            return True

        config = {
            "step_type": "prompt",
            "step_config": {"template": "Test"},
        }
        context = {"is_cancelled": is_cancelled}

        result = await run_retry_adapter(config, context)

        assert result.get("__status__") == "cancelled"


# =============================================================================
# Checkpoint Adapter Tests
# =============================================================================


class TestCheckpointAdapter:
    """Tests for run_checkpoint_adapter."""

    @pytest.mark.asyncio
    async def test_checkpoint_creates_id(self):
        """Test checkpoint adapter generates checkpoint ID."""
        config = {}
        context = {"inputs": {"key": "value"}, "prev": {"result": 42}}

        result = await run_checkpoint_adapter(config, context)

        assert result["saved"] is True
        assert "checkpoint_" in result["checkpoint_id"]

    @pytest.mark.asyncio
    async def test_checkpoint_custom_id(self):
        """Test checkpoint adapter uses custom checkpoint ID."""
        config = {"checkpoint_id": "my_checkpoint_001"}
        context = {"inputs": {}}

        result = await run_checkpoint_adapter(config, context)

        assert result["checkpoint_id"] == "my_checkpoint_001"

    @pytest.mark.asyncio
    async def test_checkpoint_with_run_id(self):
        """Test checkpoint adapter includes run_id in result."""
        config = {}
        context = {"inputs": {}, "run_id": "run_12345"}

        result = await run_checkpoint_adapter(config, context)

        assert result["run_id"] == "run_12345"

    @pytest.mark.asyncio
    async def test_checkpoint_calls_append_event(self):
        """Test checkpoint adapter calls append_event callback."""
        events = []

        def append_event(event_type, data):
            events.append((event_type, data))

        config = {"checkpoint_id": "ckpt_test"}
        context = {"inputs": {"x": 1}, "append_event": append_event}

        result = await run_checkpoint_adapter(config, context)

        assert result["saved"] is True
        assert len(events) == 1
        assert events[0][0] == "checkpoint"
        assert events[0][1]["checkpoint_id"] == "ckpt_test"

    @pytest.mark.asyncio
    async def test_checkpoint_with_custom_data(self):
        """Test checkpoint adapter saves custom data."""
        events = []

        def append_event(event_type, data):
            events.append((event_type, data))

        config = {"checkpoint_id": "ckpt_data", "data": {"custom": "payload"}}
        context = {"inputs": {}, "append_event": append_event}

        result = await run_checkpoint_adapter(config, context)

        assert result["saved"] is True
        assert events[0][1]["data"] == {"custom": "payload"}

    @pytest.mark.asyncio
    async def test_checkpoint_sanitizes_backend_errors(self):
        """Test checkpoint failures hide raw backend exception details."""

        def append_event(_event_type, _data):
            raise RuntimeError("checkpoint backend exploded at /private/checkpoints")

        config = {"checkpoint_id": "ckpt_error"}
        context = {"inputs": {}, "append_event": append_event}

        result = await run_checkpoint_adapter(config, context)

        assert result == {
            "error": "checkpoint_error",
            "checkpoint_id": "ckpt_error",
            "saved": False,
        }

    @pytest.mark.asyncio
    async def test_checkpoint_cancellation(self):
        """Test checkpoint adapter respects cancellation."""

        def is_cancelled():
            return True

        config = {}
        context = {"is_cancelled": is_cancelled}

        result = await run_checkpoint_adapter(config, context)

        assert result.get("__status__") == "cancelled"


# =============================================================================
# Workflow Call Adapter Tests
# =============================================================================


class TestWorkflowCallAdapter:
    """Tests for run_workflow_call_adapter."""

    @pytest.mark.asyncio
    async def test_workflow_call_missing_id(self):
        """Test workflow call adapter returns error for missing workflow_id."""
        config = {}
        context = {}

        result = await run_workflow_call_adapter(config, context)

        assert result["error"] == "missing_workflow_id"

    @pytest.mark.asyncio
    async def test_workflow_call_not_found(self, monkeypatch):
        """Test workflow call adapter handles workflow not found."""
        mock_db = MagicMock()
        mock_db.get_workflow = MagicMock(return_value=None)

        # The adapter imports workflows_db inside the function, so we need to
        # create a mock module and patch it at the orchestration module level
        mock_module = MagicMock()
        mock_module.get_workflows_db = lambda: mock_db

        import sys
        monkeypatch.setitem(sys.modules, "tldw_Server_API.app.core.Workflows.workflows_db", mock_module)

        config = {"workflow_id": "nonexistent_workflow"}
        context = {}

        result = await run_workflow_call_adapter(config, context)

        assert "workflow_not_found" in result["error"]

    @pytest.mark.asyncio
    async def test_workflow_call_async(self, monkeypatch):
        """Test workflow call adapter in async (non-waiting) mode."""
        mock_workflow = MagicMock()
        mock_workflow.id = "wf_123"

        mock_db = MagicMock()
        mock_db.get_workflow = MagicMock(return_value=mock_workflow)
        mock_db.create_run = MagicMock()

        mock_engine = MagicMock()
        mock_engine.submit = MagicMock()

        # Create mock module for workflows_db
        mock_db_module = MagicMock()
        mock_db_module.get_workflows_db = lambda: mock_db

        # Create mock module for engine
        mock_engine_module = MagicMock()
        mock_engine_module.WorkflowEngine = MagicMock(return_value=mock_engine)
        mock_engine_module.EngineConfig = MagicMock()

        import sys
        monkeypatch.setitem(sys.modules, "tldw_Server_API.app.core.Workflows.workflows_db", mock_db_module)
        monkeypatch.setitem(sys.modules, "tldw_Server_API.app.core.Workflows.engine", mock_engine_module)

        config = {
            "workflow_id": "wf_123",
            "inputs": {"param": "value"},
            "wait": False,
        }
        context = {"tenant_id": "test_tenant", "user_id": "user_1"}

        result = await run_workflow_call_adapter(config, context)

        assert result["status"] == "submitted"
        assert result["async"] is True
        assert "run_id" in result
        mock_engine.submit.assert_called_once()

    @pytest.mark.asyncio
    async def test_workflow_call_cancellation(self):
        """Test workflow call adapter respects cancellation."""

        def is_cancelled():
            return True

        config = {"workflow_id": "wf_123"}
        context = {"is_cancelled": is_cancelled}

        result = await run_workflow_call_adapter(config, context)

        assert result.get("__status__") == "cancelled"

    @pytest.mark.asyncio
    async def test_workflow_call_with_timeout(self, monkeypatch):
        """Test workflow call adapter handles timeout."""
        mock_workflow = MagicMock()
        mock_workflow.id = "wf_slow"

        mock_db = MagicMock()
        mock_db.get_workflow = MagicMock(return_value=mock_workflow)
        mock_db.create_run = MagicMock()
        mock_db.get_run = MagicMock(return_value=None)

        # Mock WorkflowEngine that times out
        async def slow_start(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than timeout

        mock_engine = MagicMock()
        mock_engine.start_run = slow_start

        # Create mock module for workflows_db
        mock_db_module = MagicMock()
        mock_db_module.get_workflows_db = lambda: mock_db

        # Create mock module for engine
        mock_engine_module = MagicMock()
        mock_engine_module.WorkflowEngine = MagicMock(return_value=mock_engine)
        mock_engine_module.EngineConfig = MagicMock()

        import sys
        monkeypatch.setitem(sys.modules, "tldw_Server_API.app.core.Workflows.workflows_db", mock_db_module)
        monkeypatch.setitem(sys.modules, "tldw_Server_API.app.core.Workflows.engine", mock_engine_module)

        config = {
            "workflow_id": "wf_slow",
            "inputs": {},
            "wait": True,
            "timeout_seconds": 1,  # Short timeout for test
        }
        context = {"tenant_id": "default"}

        result = await run_workflow_call_adapter(config, context)

        assert result["error"] == "workflow_timeout"

    @pytest.mark.asyncio
    async def test_workflow_call_sanitizes_backend_errors(self, monkeypatch):
        """Test workflow call failures hide raw backend exception details."""
        mock_db_module = MagicMock()
        mock_db_module.get_workflows_db.side_effect = RuntimeError(
            "workflow DB exploded at /private/workflows.db"
        )

        import sys
        monkeypatch.setitem(sys.modules, "tldw_Server_API.app.core.Workflows.workflows_db", mock_db_module)

        result = await run_workflow_call_adapter({"workflow_id": "wf_error"}, {})

        assert result == {"error": "workflow_call_error", "result": None}


# =============================================================================
# Integration Tests
# =============================================================================


class TestControlAdaptersIntegration:
    """Integration tests for control adapters working together."""

    @pytest.mark.asyncio
    async def test_prompt_to_branch_flow(self):
        """Test prompt output used in branch condition."""
        # First render a prompt
        prompt_result = await run_prompt_adapter(
            {"template": "true"},
            {"inputs": {}},
        )

        # Use result in branch
        branch_result = await run_branch_adapter(
            {
                "condition": prompt_result["text"],
                "true_next": "success",
                "false_next": "fail",
            },
            {"inputs": {}},
        )

        assert branch_result["branch"] == "true"
        assert branch_result["__next__"] == "success"

    @pytest.mark.asyncio
    async def test_batch_then_map(self):
        """Test batching items then mapping over batches."""
        # First batch items
        batch_result = await run_batch_adapter(
            {"items": list(range(10)), "batch_size": 3},
            {},
        )

        assert batch_result["batch_count"] == 4

        # Map over first batch
        map_result = await run_map_adapter(
            {
                "items": batch_result["batches"][0],
                "step": {"type": "prompt", "config": {"template": "Item {{ item }}"}},
            },
            {"inputs": {}},
        )

        assert map_result["count"] == 3

    @pytest.mark.asyncio
    async def test_parallel_with_mixed_steps(self):
        """Test parallel execution with different step types."""
        config = {
            "steps": [
                {"type": "prompt", "config": {"template": "Hello"}},
                {"type": "delay", "config": {"milliseconds": 10}},
                {"type": "log", "config": {"message": "Test log", "level": "debug"}},
            ],
            "max_concurrency": 3,
        }
        context = {"inputs": {}}

        result = await run_parallel_adapter(config, context)

        assert result["count"] == 3
        assert result["results"][0]["text"] == "Hello"
        assert result["results"][1]["delayed_ms"] == 10
        assert result["results"][2]["logged"] is True
