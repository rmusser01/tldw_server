from __future__ import annotations

import pytest

from tldw_Server_API.app.api.v1.endpoints import workflows as workflows_api
from tldw_Server_API.app.core.Security import egress

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_workflows_config_reports_default_egress_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(egress.ALLOWED_PORTS_ENV, raising=False)
    monkeypatch.setattr(workflows_api, "get_content_backend_instance", lambda: None)

    cfg = await workflows_api.get_workflows_config(_current_user=object())

    assert cfg["egress"]["allowed_ports"] == ["80", "443", "8080"]


@pytest.mark.asyncio
async def test_workflows_tests_do_not_touch_authnz_daily_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The conftest keeps runs out of the checkout's users.db ledger (TASK-13367)."""
    from tldw_Server_API.app.core.Workflows import daily_ledger

    monkeypatch.setattr(workflows_api, "get_content_backend_instance", lambda: None)
    cfg = await workflows_api.get_workflows_config(_current_user=object())

    assert cfg["rate_limits"]["quotas_disabled"] is True
    assert await daily_ledger.record_workflow_run(entity_scope="user", entity_value="1", run_id="r") is False
