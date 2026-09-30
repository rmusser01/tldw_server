from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.endpoints import auth as auth_ep
from tldw_Server_API.app.core.Resource_Governance.governor import RGDecision

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit, pytest.mark.asyncio]


class _SpyGov:
    def __init__(self):
        self.entities = []

    async def reserve(self, req, op_id=None):
        self.entities.append(req.entity)
        return RGDecision(allowed=True, retry_after=None, details={}), None


def _request(policy_id=None):
    state = SimpleNamespace(rg_policy_id=policy_id) if policy_id else SimpleNamespace()
    return SimpleNamespace(state=state, url=SimpleNamespace(path="/api/v1/auth/forgot-password"), app=SimpleNamespace(state=SimpleNamespace()), client=SimpleNamespace(host="203.0.113.9"), headers={})


@pytest.fixture
def spy(monkeypatch):
    gov = _SpyGov()

    async def _get(_request):
        return gov

    monkeypatch.setattr(auth_ep, "_get_auth_endpoint_rg_governor", _get)
    monkeypatch.setattr(auth_ep, "_auth_rg_policy_defined", lambda *_a: True)
    monkeypatch.setattr(auth_ep, "_auth_request_client_ip", lambda _request: "203.0.113.9")
    return gov


async def test_ingress_charged_same_policy_skips_ip_reservation(spy):
    allowed, _ = await auth_ep._reserve_auth_rg_requests(_request("authnz.forgot_password"), policy_id="authnz.forgot_password")
    assert allowed and spy.entities == []


async def test_per_email_throttle_still_applies(spy):
    await auth_ep._reserve_auth_rg_requests(_request("authnz.forgot_password"), policy_id="authnz.forgot_password", entity="email:abc")
    assert spy.entities == ["email:abc"]


async def test_no_ingress_charge_reserves_by_ip(spy):
    await auth_ep._reserve_auth_rg_requests(_request(), policy_id="authnz.forgot_password")
    assert len(spy.entities) == 1 and spy.entities[0].startswith("ip:")
