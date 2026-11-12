import os
import time

import pytest
pytestmark = pytest.mark.rate_limit

from tldw_Server_API.app.core.Resource_Governance.authnz_policy_store import AuthNZPolicyStore
from tldw_Server_API.app.core.Resource_Governance.policy_loader import PolicyLoader, PolicyReloadConfig
from tldw_Server_API.app.core.Resource_Governance.seed_helpers import seed_rg_policies_postgres


@pytest.mark.asyncio
async def test_db_policy_loader_merges_route_map_from_file_on_postgres(monkeypatch, test_db_pool):
    # Point RG to the stub file for route_map merge
    from pathlib import Path
    # Use the in-repo tldw_Server_API policy YAML that includes route_map
    base = Path(__file__).resolve().parents[4]
    stub = base / "tldw_Server_API" / "Config_Files" / "resource_governor_policies.yaml"
    monkeypatch.setenv("RG_POLICY_PATH", str(stub))

    # Seed minimal policies into Postgres
    now = time.time()
    await seed_rg_policies_postgres(
        test_db_pool,
        [
            {"id": "chat.default", "payload": {"requests": {"rpm": 10}}, "version": 1, "updated_at": now},
        ],
    )

    store = AuthNZPolicyStore()
    loader = PolicyLoader(str(stub), PolicyReloadConfig(enabled=False), store=store)
    snap = await loader.load_once()

    # Validate we have policies from DB and route_map from file
    assert "chat.default" in snap.policies
    assert isinstance(snap.route_map, dict) and snap.route_map
    assert snap.route_map.get("by_tag", {}).get("health") == "health.default"
