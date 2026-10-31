"""Stable tenant identity requirements for MCTS durable caching."""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.PromptStudioDatabase import (
    PromptStudioDatabase,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.mcts_optimizer import (
    MCTSOptimizer,
)
from tldw_Server_API.app.core.Prompt_Management.prompt_studio.types_common import (
    MetricType,
)

pytestmark = pytest.mark.unit


class _Cursor:
    def fetchone(self) -> dict[str, Any]:
        return {
            "payload": {
                "value": "foreign-tenant-value",
                "expires_at": "2099-01-01T00:00:00",
            }
        }


class _IdentitylessCacheDb:
    """Cache facade deliberately lacking every stable tenant identifier."""

    def __init__(self) -> None:
        self.reads = 0
        self.writes: list[dict[str, Any]] = []

    @contextmanager
    def transaction(self) -> Iterator[object]:
        self.reads += 1
        yield object()

    def _cursor_exec(
        self,
        _conn: object,
        _sql: str,
        _params: tuple[Any, ...],
    ) -> _Cursor:
        return _Cursor()

    def _log_sync_event(self, **event: Any) -> None:
        self.writes.append(event)


class _PostgresCacheDb:
    """PostgreSQL facade where the SQLite-owned sync_log must never be touched."""

    backend_type = "postgresql"
    tenant_user_id = "tenant-42"

    @contextmanager
    def transaction(self) -> Iterator[object]:
        raise AssertionError("PostgreSQL MCTS cache must not open a sync_log transaction")
        yield object()  # pragma: no cover

    def _log_sync_event(self, **_event: Any) -> None:
        raise AssertionError("PostgreSQL MCTS cache must not write to sync_log")


def _optimizer_with_db(db: object) -> MCTSOptimizer:
    optimizer = object.__new__(MCTSOptimizer)
    optimizer.db = db  # type: ignore[assignment]
    return optimizer


def test_mcts_durable_cache_get_misses_without_stable_tenant_identity() -> None:
    db = _IdentitylessCacheDb()

    result = _optimizer_with_db(db)._db_cache_get("shared-key")

    assert result is None
    assert db.reads == 0


def test_mcts_durable_cache_set_is_noop_without_stable_tenant_identity() -> None:
    db = _IdentitylessCacheDb()

    _optimizer_with_db(db)._db_cache_set("shared-key", "tenant-value")

    assert db.writes == []


def test_mcts_durable_cache_skips_unsupported_postgres_boundary_concurrently() -> None:
    optimizer = _optimizer_with_db(_PostgresCacheDb())

    with ThreadPoolExecutor(max_workers=8) as executor:
        misses = list(
            executor.map(
                lambda index: optimizer._db_cache_get(f"shared-key-{index}"),
                range(32),
            )
        )
        list(
            executor.map(
                lambda index: optimizer._db_cache_set(
                    f"shared-key-{index}",
                    "tenant-value",
                ),
                range(32),
            )
        )

    assert misses == [None] * 32


def test_sqlite_prompt_studio_database_preserves_tenant_identity(tmp_path) -> None:
    db = PromptStudioDatabase(
        tmp_path / "prompt-studio.sqlite",
        client_id="request-audit-client",
        tenant_user_id="tenant-42",
    )

    try:
        assert db.tenant_user_id == "tenant-42"
        assert db._impl.tenant_user_id == "tenant-42"
    finally:
        db.close()


def test_mcts_durable_cache_round_trips_on_real_sqlite(tmp_path) -> None:
    db = PromptStudioDatabase(
        tmp_path / "prompt-studio-cache.sqlite",
        client_id="cache-audit-client",
        tenant_user_id="tenant-42",
    )
    optimizer = _optimizer_with_db(db)

    try:
        optimizer._db_cache_set("real-round-trip", {"score": 0.75})

        assert optimizer._db_cache_get("real-round-trip") == {"score": 0.75}
    finally:
        db.close()


@pytest.mark.asyncio
async def test_mcts_eval_cache_invalidates_when_test_case_changes(tmp_path) -> None:
    class _Runner:
        def __init__(self) -> None:
            self.score = 0.91
            self.calls = 0

        async def run_single_test(self, **kwargs: Any) -> dict[str, Any]:
            self.calls += 1
            callback = kwargs.get("on_provider_success")
            if callback is not None:
                await callback()
            return {
                "success": True,
                "scores": {
                    "accuracy": self.score,
                    "aggregate_score": self.score,
                },
            }

    db = PromptStudioDatabase(
        tmp_path / "prompt-studio-revision.sqlite",
        client_id="revision-audit-client",
        tenant_user_id="tenant-42",
    )
    project = db.create_project("Revision project", "")
    test_case = db.create_test_case(
        project_id=int(project["id"]),
        name="Mutable case",
        inputs={"question": "before"},
        expected_outputs={"answer": "before"},
    )
    runner = _Runner()
    optimizer = MCTSOptimizer(db, runner)  # type: ignore[arg-type]
    optimizer._create_ephemeral_prompt_version = lambda **_kwargs: 101  # type: ignore[method-assign]
    marks: list[str] = []

    async def _mark() -> None:
        marks.append("used")

    config = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "parameters": {},
        "api_key": "runtime-secret",
        "app_config": {"openai_api": {"base_url": "https://api.example/v1"}},
        "credentials_resolved": True,
    }
    call_kwargs = {
        "base_prompt": {
            "id": 1,
            "project_id": int(project["id"]),
            "name": "Revision prompt",
            "version_number": 1,
        },
        "system_text": "Shared system prompt",
        "user_text": "Shared user prompt",
        "test_case_ids": [int(test_case["id"])],
        "model_config": config,
        "target_metric": MetricType.ACCURACY,
        "feedback_enabled": False,
        "feedback_threshold": 10.0,
        "feedback_max_retries": 0,
        "on_provider_success": _mark,
        "strict_provider_errors": True,
    }

    try:
        first, _ = await optimizer._evaluate_with_feedback(**call_kwargs)
        db.update_test_case(
            int(test_case["id"]),
            {
                "inputs": {"question": "after"},
                "expected_outputs": {"answer": "after"},
            },
        )
        runner.score = 0.27
        second, _ = await optimizer._evaluate_with_feedback(**call_kwargs)

        assert first == pytest.approx(0.91)
        assert second == pytest.approx(0.27)
        assert runner.calls == 2
        assert marks == ["used", "used"]
    finally:
        db.close()
