import time

import pytest
pytestmark = pytest.mark.rate_limit

from tldw_Server_API.app.core.Resource_Governance.policy_loader import PolicyLoader, PolicyReloadConfig


class _DummyStore:
    async def get_latest_policy(self):
        # Return minimal valid tuple (version, policies, tenant, updated_at)
        return 2, {"chat.default": {"requests": {"rpm": 120}}}, {"enabled": True}, time.time()


@pytest.mark.asyncio
async def test_db_policy_loader_includes_route_map_from_file(monkeypatch):
    # Use the repo stub policy YAML for route_map
    from pathlib import Path
    base = Path(__file__).resolve().parents[3]
    # Use the tldw_Server_API stub (includes route_map)
    path = base / "tldw_Server_API" / "Config_Files" / "resource_governor_policies.yaml"

    loader = PolicyLoader(str(path), PolicyReloadConfig(enabled=False), store=_DummyStore())
    snap = await loader.load_once()
    # Should merge route_map from file even when using DB-backed store
    assert isinstance(snap.route_map, dict) and snap.route_map
    assert "by_tag" in snap.route_map
    assert snap.route_map["by_tag"].get("chat") == "chat.default"
    assert snap.route_map["by_tag"].get("health") == "health.default"
    assert snap.route_map["by_tag"].get("flashcards") == "flashcards.default"
    assert snap.route_map["by_tag"].get("quizzes") == "quizzes.default"
    assert snap.route_map.get("by_path", {}).get("/api/v1/chatbooks/export") == "chatbooks.export"
    assert snap.route_map.get("by_path", {}).get("/api/v1/health*") == "health.default"
    assert snap.route_map.get("by_path", {}).get("/api/v1/flashcards*") == "flashcards.default"
    assert snap.route_map.get("by_path", {}).get("/api/v1/quizzes*") == "quizzes.default"
    assert snap.route_map.get("by_path", {}).get("/api/v1/watchlists/*") == "watchlists.default"
