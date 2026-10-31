"""One person's WebUI session plus an extension stream never hits a governor 429.

The shipped policy YAML resolves by_path and default against a catch-all app.
Tag-only routes therefore resolve to `default` here (same or looser limits than
their tag policy), which the spec's goal 1 tolerates.
"""

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Resource_Governance.governor import MemoryResourceGovernor
from tldw_Server_API.app.core.Resource_Governance.middleware_simple import RGSimpleMiddleware
from tldw_Server_API.app.core.Resource_Governance.policy_loader import PolicyLoader, PolicyReloadConfig

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit, pytest.mark.asyncio]

FIXTURE = Path(__file__).parent / "fixtures" / "webui_session_requests.json"
YAML = Path(__file__).resolve().parents[2] / "Config_Files" / "resource_governor_policies.yaml"


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


async def test_webui_session_and_extension_stream_see_no_429(monkeypatch):
    monkeypatch.delenv("RG_POLICY_PATH", raising=False)
    loader = PolicyLoader(YAML, PolicyReloadConfig(enabled=False))
    await loader.load_once()
    clock = Clock()
    app = FastAPI()

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    def anything(path: str) -> dict:
        return {}

    app.add_middleware(RGSimpleMiddleware)
    app.state.rg_policy_loader = loader
    app.state.rg_governor = MemoryResourceGovernor(policy_loader=loader, time_source=clock)

    async def one_user(self, request):
        return "user:1"

    monkeypatch.setattr(RGSimpleMiddleware, "_principal_entity", one_user)
    client = TestClient(app)

    session = json.loads(FIXTURE.read_text(encoding="utf-8"))
    duration = session[-1][0] if session else 0.0
    timeline = [(t, m, p) for t, m, p in session]
    timeline += [(t + duration, m, p) for t, m, p in session]  # second pass: steady state
    timeline += [(float(s), "GET", "/api/v1/notes/") for s in range(int(2 * duration) + 1)]  # extension, 1/s
    timeline.sort(key=lambda row: row[0])

    denied = []
    for t, method, path in timeline:
        clock.t = 1000.0 + t
        if client.request(method, path).status_code == 429:
            denied.append((t, method, path))
    assert denied == [], f"{len(denied)} governor 429s, first: {denied[:5]}"
