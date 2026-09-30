"""Auth endpoints charge their per-IP RG bucket once per request.

When RG ingress already charged this request's IP to the same policy, the auth
handler skips its own IP reservation. It must not skip when ingress charged some
other entity (e.g. a rotating fake API key), when ingress failed open and charged
nothing, or when the auth reservation is keyed on email or user.
"""

from types import SimpleNamespace

import pytest

from tldw_Server_API.app.api.v1.endpoints import auth as auth_ep
from tldw_Server_API.app.core.Resource_Governance.governor import RGDecision

pytestmark = [pytest.mark.unit, pytest.mark.rate_limit, pytest.mark.asyncio]

_POLICY = "authnz.forgot_password"
_IP = "ip:203.0.113.9"


class _SpyGov:
    """Governor stub that records the entity of every reservation and allows it."""

    def __init__(self):
        self.entities = []

    async def reserve(self, req, op_id=None):
        """Record the reserved entity and allow the request."""
        self.entities.append(req.entity)
        return RGDecision(allowed=True, retry_after=None, details={}), None


def _request(policy_id=None, ingress_entity=None):
    """Build a minimal request whose state mimics what RG ingress left behind."""
    state = SimpleNamespace()
    if policy_id:
        state.rg_policy_id = policy_id
    if ingress_entity:
        state.rg_ingress_entity = ingress_entity
    return SimpleNamespace(state=state, url=SimpleNamespace(path="/api/v1/auth/forgot-password"), app=SimpleNamespace(state=SimpleNamespace()), client=SimpleNamespace(host="203.0.113.9"), headers={})


@pytest.fixture
def spy(monkeypatch):
    """Route auth's RG reservations to a _SpyGov and return it."""
    gov = _SpyGov()

    async def _get(_request):
        return gov

    monkeypatch.setattr(auth_ep, "_get_auth_endpoint_rg_governor", _get)
    monkeypatch.setattr(auth_ep, "_auth_rg_policy_defined", lambda *_a: True)
    monkeypatch.setattr(auth_ep, "_auth_request_client_ip", lambda _request: "203.0.113.9")
    return gov


async def test_ingress_charged_same_policy_skips_ip_reservation(spy):
    allowed, _ = await auth_ep._reserve_auth_rg_requests(_request(_POLICY, _IP), policy_id=_POLICY, entity=_IP)
    assert allowed and spy.entities == []


async def test_ingress_charged_api_key_entity_still_reserves_ip(spy):
    # A rotating fake X-API-KEY gets a fresh ingress bucket every time; the IP bucket is the real limit.
    await auth_ep._reserve_auth_rg_requests(_request(_POLICY, "api_key:deadbeef"), policy_id=_POLICY, entity=_IP)
    assert spy.entities == [_IP]


async def test_ingress_failed_open_still_reserves_ip(spy):
    # rg_policy_id is set before ingress reserves; without rg_ingress_entity nothing was charged.
    await auth_ep._reserve_auth_rg_requests(_request(_POLICY), policy_id=_POLICY, entity=_IP)
    assert spy.entities == [_IP]


async def test_per_email_throttle_still_applies(spy):
    await auth_ep._reserve_auth_rg_requests(_request(_POLICY, _IP), policy_id=_POLICY, entity="email:abc")
    assert spy.entities == ["email:abc"]


async def test_per_user_throttle_still_applies(spy):
    await auth_ep._reserve_auth_rg_requests(_request(_POLICY, _IP), policy_id=_POLICY, entity="user:7")
    assert spy.entities == ["user:7"]


async def test_no_ingress_charge_reserves_by_ip(spy):
    await auth_ep._reserve_auth_rg_requests(_request(), policy_id=_POLICY, entity=_IP)
    assert len(spy.entities) == 1 and spy.entities[0].startswith("ip:")
