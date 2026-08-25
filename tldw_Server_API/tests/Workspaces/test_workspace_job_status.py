"""Tests for bounded, fail-open workspace source Jobs lookup."""
from __future__ import annotations

from typing import Any

import pytest

from tldw_Server_API.app.core.Workspaces.job_status import (
    list_recent_workspace_source_ingest_jobs,
)

pytestmark = pytest.mark.unit


class _JobManager:
    def __init__(
        self,
        *,
        workspace_jobs: list[dict[str, Any]] | None = None,
        legacy_jobs: list[dict[str, Any]] | None = None,
        fail_workspace: bool = False,
        fail_legacy: bool = False,
    ) -> None:
        self.workspace_jobs = workspace_jobs or []
        self.legacy_jobs = legacy_jobs or []
        self.fail_workspace = fail_workspace
        self.fail_legacy = fail_legacy
        self.calls: list[dict[str, Any]] = []

    def list_jobs(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append(dict(kwargs))
        is_workspace_query = kwargs.get("job_type") == "workspace_source_ingest"
        if is_workspace_query and self.fail_workspace:
            raise RuntimeError("workspace Jobs query failed")
        if not is_workspace_query and self.fail_legacy:
            raise RuntimeError("legacy Jobs query failed")
        return list(self.workspace_jobs if is_workspace_query else self.legacy_jobs)


def test_workspace_job_lookup_is_empty_without_optional_jobs_manager() -> None:
    assert list_recent_workspace_source_ingest_jobs(None, owner_user_id=7) == []


def test_workspace_job_lookup_preserves_both_bounded_query_families() -> None:
    manager = _JobManager(
        workspace_jobs=[{"id": 1, "job_type": "workspace_source_ingest"}],
        legacy_jobs=[{"id": 2, "job_type": "media_ingest_item"}],
    )

    jobs = list_recent_workspace_source_ingest_jobs(manager, owner_user_id=7)

    assert [job["id"] for job in jobs] == [1, 2]
    assert manager.calls == [
        {
            "domain": "media_ingest",
            "queue": "default",
            "owner_user_id": "7",
            "job_type": "workspace_source_ingest",
            "limit": 500,
            "sort_by": "created_at",
            "sort_order": "desc",
        },
        {
            "domain": "media_ingest",
            "owner_user_id": "7",
            "limit": 500,
            "sort_by": "created_at",
            "sort_order": "desc",
        },
    ]


def test_workspace_job_lookup_dedupes_by_id_uuid_and_fallback_identity() -> None:
    fallback = {
        "domain": "media_ingest",
        "queue": "default",
        "job_type": "media_ingest_item",
        "created_at": "2026-08-21T01:02:03Z",
        "payload": {"media_id": 9},
    }
    manager = _JobManager(
        workspace_jobs=[
            {"id": 1, "uuid": "workspace-uuid"},
            {"uuid": "uuid-only"},
            fallback,
        ],
        legacy_jobs=[
            {"id": 1, "uuid": "different-uuid"},
            {"uuid": "uuid-only"},
            dict(fallback),
            {"id": 2},
        ],
    )

    jobs = list_recent_workspace_source_ingest_jobs(manager, owner_user_id="7")

    assert jobs == [
        {"id": 1, "uuid": "workspace-uuid"},
        {"uuid": "uuid-only"},
        fallback,
        {"id": 2},
    ]


@pytest.mark.parametrize(
    ("fail_workspace", "fail_legacy", "expected_ids"),
    [(True, False, [2]), (False, True, [1]), (True, True, [])],
)
def test_workspace_job_lookup_fails_open_per_optional_query_family(
    fail_workspace: bool,
    fail_legacy: bool,
    expected_ids: list[int],
) -> None:
    manager = _JobManager(
        workspace_jobs=[{"id": 1}],
        legacy_jobs=[{"id": 2}],
        fail_workspace=fail_workspace,
        fail_legacy=fail_legacy,
    )

    jobs = list_recent_workspace_source_ingest_jobs(manager, owner_user_id=7)

    assert [job["id"] for job in jobs] == expected_ids
    assert len(manager.calls) == 2
