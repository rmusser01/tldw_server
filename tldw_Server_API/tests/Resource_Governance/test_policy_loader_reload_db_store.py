import asyncio
import os
import time
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Resource_Governance.policy_loader import PolicyLoader, PolicyReloadConfig


class _BumpStore:
    def __init__(self):
        self._ver = 1
        self._pol = {"chat.default": {"requests": {"rpm": 60}}}
        self._tenant = {"enabled": True}
        self._ts = time.time()

    async def get_latest_policy(self):
        # Return (version, policies, tenant, updated_at_ts)
        return self._ver, dict(self._pol), dict(self._tenant), float(self._ts)

    def bump(self, *, rpm: int = 120):
        self._ver += 1
        self._pol = {"chat.default": {"requests": {"rpm": rpm}}}
        self._ts = time.time()


@pytest.mark.asyncio
async def test_db_store_reload_ttl_and_route_map_precedence(tmp_path):
    # Create a small YAML with a route_map
    yaml_path = tmp_path / "rg.yaml"
    yaml_path.write_text(
        """
version: 1
policies:
  chat.default:
    requests: { rpm: 60 }
route_map:
  by_path:
    /api/v1/chat/*: chat.default
    /api/v1/alt/*: file.policy
        """.strip(),
        encoding="utf-8",
    )
    future_mtime = time.time() + 3600
    os.utime(yaml_path, (future_mtime, future_mtime))

    store = _BumpStore()
    loader = PolicyLoader(str(yaml_path), PolicyReloadConfig(enabled=False), store=store)
    snap1 = await loader.load_once()
    assert snap1.version == 1
    assert snap1.route_map.get("by_path", {}).get("/api/v1/chat/*") == "chat.default"

    # Bump store version & rpm; then maybe_reload should pull v2 even when the
    # file route_map mtime is newer than the DB updated_at timestamp.
    store.bump(rpm=200)
    await loader._maybe_reload()  # type: ignore[attr-defined]
    snap2 = loader.get_snapshot()
    assert snap2.version >= 2
    # Route map precedence: file route_map survives DB policy changes
    assert snap2.route_map.get("by_path", {}).get("/api/v1/alt/*") == "file.policy"
