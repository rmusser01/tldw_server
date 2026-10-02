"""One person's WebUI session plus an extension stream never hits a governor 429.

The shipped policy YAML resolves by_path and default against a catch-all app.
Tag-only routes therefore resolve to `default` here (same or looser limits than
their tag policy), which the spec's goal 1 tolerates.

Runs against both the memory and Redis governor backends (TASK-13404 AC2): the
Redis path uses the in-process InMemoryAsyncRedis stub (see
test_governor_safety_net.py::_gov), never a real Redis server.
"""

import itertools
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Resource_Governance import (
    MemoryResourceGovernor,
    RedisResourceGovernor,
    ResourceGovernor,
)
from tldw_Server_API.app.core.Resource_Governance.middleware_simple import RGSimpleMiddleware
from tldw_Server_API.app.core.Resource_Governance.policy_loader import PolicyLoader, PolicyReloadConfig

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit, pytest.mark.asyncio]

FIXTURE = Path(__file__).parent / "fixtures" / "webui_session_requests.json"
YAML = Path(__file__).resolve().parents[2] / "Config_Files" / "resource_governor_policies.yaml"

BACKENDS = ["memory", "redis"]
_ns = itertools.count()


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


def _governor(backend: str, loader: PolicyLoader, clock: Clock) -> ResourceGovernor:
    """Build the governor for ``backend``; the Redis path never touches a real Redis server."""
    if backend == "memory":
        return MemoryResourceGovernor(policy_loader=loader, time_source=clock)

    from tldw_Server_API.app.core.Infrastructure.redis_factory import InMemoryAsyncRedis

    gov = RedisResourceGovernor(policy_loader=loader, time_source=clock, ns=f"rg_t_webui_replay_{next(_ns)}")
    # Inject the in-process stub directly so this test never touches a real Redis
    # on 127.0.0.1:6379 if one happens to be running.
    gov._client = InMemoryAsyncRedis()
    return gov


@pytest.mark.parametrize("backend", BACKENDS)
async def test_webui_session_and_extension_stream_see_no_429(monkeypatch, backend):
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
    app.state.rg_governor = _governor(backend, loader, clock)

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
        resp = client.request(method, path)
        if resp.status_code == 429:
            policy_id = None
            try:
                policy_id = resp.json().get("policy_id")
            except Exception:  # noqa: BLE001 - diagnostics only
                policy_id = None
            denied.append((t, method, path, policy_id))
    assert denied == [], f"{len(denied)} governor 429s on {backend} backend, first: {denied[:5]}"
