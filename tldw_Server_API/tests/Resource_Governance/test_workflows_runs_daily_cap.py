import pytest

from tldw_Server_API.app.core.Resource_Governance import MemoryResourceGovernor, RGRequest
from tldw_Server_API.app.core.Resource_Governance.policy_loader import (
    PolicyLoader,
    PolicyReloadConfig,
)
from tldw_Server_API.app.core.Workflows.daily_ledger import record_workflow_run


pytestmark = pytest.mark.rate_limit


async def _init_authnz_sqlite(db_path, monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    try:
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

        reset_settings()
    except Exception:
        _ = None
    try:
        from tldw_Server_API.app.core.AuthNZ.initialize import ensure_authnz_schema_ready_once

        await ensure_authnz_schema_ready_once()
    except Exception:
        _ = None
    # Reset cached RG daily ledger between tests when DATABASE_URL changes.
    try:
        import tldw_Server_API.app.core.Resource_Governance.daily_caps as _dc

        _dc._daily_ledger = None  # type: ignore[attr-defined]
    except Exception:
        _ = None
    # Reset cached ledger inside workflows daily_ledger between tests.
    try:
        import tldw_Server_API.app.core.Workflows.daily_ledger as _dl

        _dl._workflows_daily_ledger = None  # type: ignore[attr-defined]
        _dl._workflows_backfill_done = set()  # type: ignore[attr-defined]
    except Exception:
        _ = None


@pytest.mark.asyncio
async def test_workflows_runs_daily_cap_denies_when_exceeded(tmp_path, monkeypatch, fresh_services):
    db_path = tmp_path / "authnz_workflows.db"
    await _init_authnz_sqlite(db_path, monkeypatch)

    policy_yaml = tmp_path / "rg_wf_daily.yaml"
    policy_yaml.write_text(
        """
schema_version: 1
policies:
  workflows.test:
    requests: { rpm: 100000, burst: 1.0 }
    workflows_runs: { daily_cap: 2 }
    scopes: [user]
route_map: {}
""".lstrip()
    )

    loader = PolicyLoader(str(policy_yaml), PolicyReloadConfig(enabled=False, interval_sec=0))
    await loader.load_once()
    gov = MemoryResourceGovernor(policy_loader=loader)

    # Record two runs so far for user 1.
    await record_workflow_run(entity_scope="user", entity_value="1", run_id="run-a", units=1)
    await record_workflow_run(entity_scope="user", entity_value="1", run_id="run-b", units=1)

    req = RGRequest(
        entity="user:1",
        categories={"workflows_runs": {"units": 1}},
        tags={"policy_id": "workflows.test"},
    )
    dec, _ = await gov.reserve(req, op_id="op-wf-1")
    assert dec.allowed is False


@pytest.mark.asyncio
async def test_record_workflow_run_is_idempotent(tmp_path, monkeypatch, fresh_services):
    db_path = tmp_path / "authnz_workflows_idem.db"
    await _init_authnz_sqlite(db_path, monkeypatch)

    from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import ResourceDailyLedger

    ledger = ResourceDailyLedger()
    await ledger.initialize()

    before = await ledger.total_for_day("user", "1", "workflows_runs")

    await record_workflow_run(entity_scope="user", entity_value="1", run_id="run-dup", units=1)
    await record_workflow_run(entity_scope="user", entity_value="1", run_id="run-dup", units=1)

    after = await ledger.total_for_day("user", "1", "workflows_runs")
    assert after == before + 1
