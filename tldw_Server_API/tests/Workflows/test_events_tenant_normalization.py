"""Regression tests for normalized tenant IDs on workflow audit events."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tldw_Server_API.app.api.v1.endpoints import workflows as workflows_ep
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.Workflows_DB import WorkflowsDatabase

pytestmark = pytest.mark.unit


def _user_with_tenant(tenant_id: str | None) -> User:
    """Build an admin user whose absent or blank tenant must normalize."""
    return User(
        id=1,
        username="tester",
        email="t@e.com",
        is_active=True,
        roles=["admin"],
        tenant_id=tenant_id,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("tenant_id", [None, "   "])
async def test_control_run_normalizes_admin_impersonation_event_tenant(
    tmp_path: Path,
    tenant_id: str | None,
) -> None:
    """Persist impersonation audit events under the normalized tenant."""
    db = WorkflowsDatabase(str(tmp_path / "workflow-control.db"))
    run_id = "run-admin-impersonation"
    db.create_run(
        run_id=run_id,
        tenant_id="default",
        user_id="1",
        inputs={},
        workflow_id=None,
        definition_version=1,
        definition_snapshot={"name": "control", "version": 1, "steps": []},
    )
    request = SimpleNamespace(headers={"x-impersonate-user": "2"})

    try:
        response = await workflows_ep.control_run(
            run_id=run_id,
            action="cancel",
            request=request,
            current_user=_user_with_tenant(tenant_id),
            db=db,
        )

        events = db.get_events(run_id, types=["admin_impersonation"])
        assert response == {"ok": True, "result": "applied"}
        assert [event["tenant_id"] for event in events] == ["default"]
    finally:
        db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("tenant_id", [None, "   "])
@pytest.mark.parametrize("use_backend", [False, True])
async def test_reject_step_normalizes_event_tenant_for_each_storage_path(
    tenant_id: str | None,
    use_backend: bool,
) -> None:
    """Use the normalized tenant in both rejection persistence branches."""
    connection = object()
    db = MagicMock()
    db.backend = SimpleNamespace(transaction=lambda: nullcontext(connection)) if use_backend else None
    db.get_run.return_value = SimpleNamespace(
        tenant_id="default",
        definition_snapshot_json='{"steps": [{"id": "review"}]}',
    )
    db.get_latest_step_run.return_value = {"assigned_to": "1"}

    response = await workflows_ep.reject_step(
        run_id="run-review",
        step_id="review",
        payload=workflows_ep.HumanReviewPayload(comment="reject"),
        current_user=_user_with_tenant(tenant_id),
        db=db,
    )

    event_kwargs = {"connection": connection} if use_backend else {}
    db.append_event.assert_called_once_with(
        "default",
        "run-review",
        "human_rejected",
        {"step_id": "review"},
        **event_kwargs,
    )
    assert response == {"ok": True}
