"""Operational logging contracts for recipient source-status fallbacks."""
from __future__ import annotations

from types import SimpleNamespace, TracebackType
from typing import NoReturn
from unittest.mock import MagicMock, call

import pytest

from tldw_Server_API.app.api.v1.API_Deps import DB_Deps, jobs_deps
from tldw_Server_API.app.api.v1.endpoints import sharing
from tldw_Server_API.app.core.Workspaces import job_status, status_projection


class _FailingMediaContext:
    """Context manager that simulates an unavailable owner media database."""

    def __enter__(self) -> NoReturn:
        """Raise the configured media dependency failure."""
        raise RuntimeError("media database unavailable")

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Leave dependency failures unsuppressed."""
        return False


@pytest.mark.asyncio
async def test_source_status_fallbacks_log_workspace_context(monkeypatch) -> None:
    """Both degraded dependencies should remain visible in structured logs."""
    context = SimpleNamespace(
        owner_user_id=7,
        workspace_id="workspace-alpha",
    )
    fake_logger = MagicMock()

    def fail_job_manager() -> NoReturn:
        """Simulate unavailable Jobs infrastructure."""
        raise RuntimeError("jobs unavailable")

    monkeypatch.setattr(sharing, "logger", fake_logger)
    monkeypatch.setattr(jobs_deps, "try_get_job_manager", fail_job_manager)
    monkeypatch.setattr(
        DB_Deps,
        "managed_media_db_for_owner",
        lambda _owner_user_id: _FailingMediaContext(),
    )
    monkeypatch.setattr(
        status_projection,
        "build_source_status_projection",
        lambda **_kwargs: {
            "sources": [],
            "summary": {
                "total": 0,
                "queryable": 0,
                "processing": 0,
                "failed": 0,
            },
        },
    )
    monkeypatch.setattr(
        job_status,
        "list_recent_workspace_source_ingest_jobs",
        lambda *_args, **_kwargs: [],
    )

    result = await sharing._project_recipient_source_status(context, [])

    assert [error["code"] for error in result["partial_errors"]] == [
        "jobs_status_unavailable",
        "source_readiness_unavailable",
    ]
    assert fake_logger.bind.call_args_list == [
        call(owner_user_id=7, workspace_id="workspace-alpha"),
        call(owner_user_id=7, workspace_id="workspace-alpha"),
    ]
    assert fake_logger.bind.return_value.warning.call_count == 2
