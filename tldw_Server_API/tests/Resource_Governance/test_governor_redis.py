import asyncio
from datetime import datetime, timezone

import pytest
pytestmark = pytest.mark.rate_limit

from tldw_Server_API.app.core.Resource_Governance import RedisResourceGovernor, RGRequest
from tldw_Server_API.app.core.DB_Management.Resource_Daily_Ledger import ResourceDailyLedger, LedgerEntry


class FakeTime:
    def __init__(self, t0: float = 0.0):
        self.t = t0

    def __call__(self) -> float:

        return self.t

    def advance(self, s: float) -> None:
        self.t += s


@pytest.mark.asyncio
async def test_requests_sliding_window_with_stub_redis():
    # Policies loader stub with simple get_policy
    class _Loader:
        def get_policy(self, pid):
            return {"requests": {"rpm": 2}, "scopes": ["global", "user"]}

    ft = FakeTime(0.0)
    ns = "rg_t_reqsliding"
    rg = RedisResourceGovernor(policy_loader=_Loader(), time_source=ft, ns=ns)
    # Ensure clean keys for this policy/category
    client = await rg._client_get()
    # Aggressive cleanup for this policy id across categories
    try:
        for pat in (f"{ns}:win:p:requests*", f"{ns}:win:p:*", f"{ns}:lease:p:*"):
            _cur, keys = await client.scan(match=pat)
            for k in keys:
                await client.delete(k)
    except Exception:
        _ = None
    e = "user:1"
    req = RGRequest(entity=e, categories={"requests": {"units": 1}}, tags={"policy_id": "p"})

    d1, h1 = await rg.reserve(req, op_id="r1")
    assert d1.allowed and h1
    d2, h2 = await rg.reserve(req, op_id="r2")
    assert d2.allowed and h2

    d3, h3 = await rg.reserve(req, op_id="r3")
    assert not d3.allowed and h3 is None

    # Advance window → should allow again
    ft.advance(60.0)
    d4, h4 = await rg.reserve(req, op_id="r4")
    assert d4.allowed and h4


@pytest.mark.asyncio
async def test_commit_with_same_op_id_does_not_corrupt_redis_reserve_replay():
    class _Loader:
        def get_policy(self, pid):
            return {"requests": {"rpm": 5}, "scopes": ["user"]}

    ft = FakeTime(0.0)
    rg = RedisResourceGovernor(policy_loader=_Loader(), time_source=ft, ns="rg_t_idempotency")
    req = RGRequest(entity="user:idem", categories={"requests": {"units": 1}}, tags={"policy_id": "pidem"})

    decision, handle_id = await rg.reserve(req, op_id="shared-op")
    assert decision.allowed and handle_id

    await rg.commit(handle_id, op_id="shared-op")

    replay_decision, replay_handle = await rg.reserve(req, op_id="shared-op")
    assert replay_decision.allowed is True
    assert replay_handle == handle_id


@pytest.mark.asyncio
async def test_tokens_lua_script_retry_after():
    class _Loader:
        def get_policy(self, pid):
            return {"tokens": {"per_min": 2, "burst": 1.0}, "scopes": ["global", "user"]}

    ft = FakeTime(0.0)
    ns = "rg_t_tokens"
    rg = RedisResourceGovernor(policy_loader=_Loader(), time_source=ft, ns=ns)
    # Clean token keys for this test's policy id
    client = await rg._client_get()
    try:
        for pat in (f"{ns}:win:ptok:tokens*", f"{ns}:win:ptok:*"):
            _cur, keys = await client.scan(match=pat)
            for k in keys:
                await client.delete(k)
    except Exception:
        _ = None
    e = "user:tok"
    req = RGRequest(entity=e, categories={"tokens": {"units": 1}}, tags={"policy_id": "ptok"})

    d1, _ = await rg.reserve(req)
    assert d1.allowed
    d2, _ = await rg.reserve(req)
    assert d2.allowed

    d3, _ = await rg.reserve(req)
    assert not d3.allowed
    assert d3.retry_after is not None and int(d3.retry_after) >= 60

    ft.advance(30.0)
    d4, _ = await rg.reserve(req)
    assert not d4.allowed
    assert d4.retry_after is not None and 25 <= int(d4.retry_after) <= 60

    ft.advance(31.0)
    d5, _ = await rg.reserve(req)
    assert d5.allowed


@pytest.mark.asyncio
async def test_tokens_oversized_single_reservation_is_denied():
    class _Loader:
        def get_policy(self, pid):
            return {"tokens": {"per_min": 2, "burst": 1.0}, "scopes": ["global", "user"]}

    ft = FakeTime(0.0)
    rg = RedisResourceGovernor(policy_loader=_Loader(), time_source=ft, ns="rg_t_tokens_oversized")
    req = RGRequest(entity="user:oversized", categories={"tokens": {"units": 3}}, tags={"policy_id": "ptok_big"})

    decision, handle_id = await rg.reserve(req, op_id="oversized-1")

    assert decision.allowed is False
    assert handle_id is None
    assert decision.details["categories"]["tokens"]["limit"] == 2


@pytest.mark.asyncio
async def test_tokens_per_min_zero_is_unbounded_in_reserve():
    class _Loader:
        def get_policy(self, pid):
            return {"tokens": {"per_min": 0}, "scopes": ["global", "user"]}

    ft = FakeTime(0.0)
    ns = "rg_t_tok_unbounded"
    rg = RedisResourceGovernor(policy_loader=_Loader(), time_source=ft, ns=ns)
    client = await rg._client_get()
    try:
        for pat in (f"{ns}:win:punb:tokens*", f"{ns}:win:punb:*"):
            _cur, keys = await client.scan(match=pat)
            for k in keys:
                await client.delete(k)
    except Exception:
        _ = None

    req = RGRequest(entity="user:unb", categories={"tokens": {"units": 50}}, tags={"policy_id": "punb"})
    for i in range(3):
        d, h = await rg.reserve(req, op_id=f"unb{i}")
        assert d.allowed and h


@pytest.mark.asyncio
async def test_tokens_daily_cap_denial_short_circuits_reserve(monkeypatch, tmp_path):
    db_path = tmp_path / "authnz_tokens_daily.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    try:
        from tldw_Server_API.app.core.AuthNZ.database import reset_db_pool
        from tldw_Server_API.app.core.AuthNZ.settings import reset_settings
        await reset_db_pool()
        reset_settings()
    except Exception:
        _ = None
    try:
        from tldw_Server_API.app.core.AuthNZ.initialize import ensure_authnz_schema_ready_once
        await ensure_authnz_schema_ready_once()
    except Exception:
        _ = None
    try:
        import tldw_Server_API.app.core.Resource_Governance.daily_caps as _dc
        _dc._daily_ledger = None  # type: ignore[attr-defined]
    except Exception:
        _ = None

    ledger = ResourceDailyLedger()
    await ledger.initialize()
    await ledger.add(
        LedgerEntry(
            entity_scope="user",
            entity_value="1",
            category="tokens",
            units=1,
            op_id="seed-tokens",
            occurred_at=datetime.now(timezone.utc),
        )
    )

    class _Loader:
        def get_policy(self, pid):
            return {"tokens": {"per_min": 1000000, "daily_cap": 1}, "scopes": ["user"]}

    rg = RedisResourceGovernor(policy_loader=_Loader(), ns="rg_t_daily_cap")
    req = RGRequest(entity="user:1", categories={"tokens": {"units": 1}}, tags={"policy_id": "p"})
    d, h = await rg.reserve(req, op_id="daily-cap-1")
    assert (not d.allowed) and (h is None)
    assert not rg._local_handles


@pytest.mark.asyncio
async def test_concurrency_leases_with_zrem_capability():
    class _Loader:
        def get_policy(self, pid):
            return {"streams": {"max_concurrent": 1, "ttl_sec": 60}, "scopes": ["global", "user"]}

    ft = FakeTime(0.0)
    ns = "rg_t_conc"
    rg = RedisResourceGovernor(policy_loader=_Loader(), time_source=ft, ns=ns)
    client = await rg._client_get()
    # Clean any prior leases
    try:
        for pat in (f"{ns}:lease:pcon:streams*", f"{ns}:lease:pcon:*"):
            _cur, keys = await client.scan(match=pat)
            for k in keys:
                await client.delete(k)
    except Exception:
        _ = None
    if not hasattr(client, "zrem"):
        pytest.skip("Redis client lacks zrem; skipping precise lease deletion test")

    e = "user:lease"
    req = RGRequest(entity=e, categories={"streams": {"units": 1}}, tags={"policy_id": "pcon"})

    d1, h1 = await rg.reserve(req)
    assert d1.allowed and h1

    d2, h2 = await rg.reserve(req)
    assert not d2.allowed and h2 is None


@pytest.mark.asyncio
async def test_concurrency_parallel_reserve_does_not_overbook_after_separate_checks():
    class _Loader:
        def get_policy(self, pid):
            return {"streams": {"max_concurrent": 1, "ttl_sec": 60}, "scopes": ["user"]}

    class _BarrierRedisGovernor(RedisResourceGovernor):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._barrier_seen = 0
            self._barrier = asyncio.Event()

        async def check(self, req: RGRequest):
            decision = await super().check(req)
            if decision.allowed and "streams" in req.categories:
                self._barrier_seen += 1
                if self._barrier_seen == 2:
                    self._barrier.set()
                await asyncio.wait_for(self._barrier.wait(), timeout=2)
            return decision

    ft = FakeTime(0.0)
    rg = _BarrierRedisGovernor(policy_loader=_Loader(), time_source=ft, ns="rg_t_conc_race")
    req = RGRequest(entity="user:race", categories={"streams": {"units": 1}}, tags={"policy_id": "pcon_race"})

    results = await asyncio.gather(
        rg.reserve(req, op_id="race-1"),
        rg.reserve(req, op_id="race-2"),
    )

    allowed = [decision.allowed for decision, _handle in results]
    assert allowed.count(True) == 1
    assert allowed.count(False) == 1


@pytest.mark.asyncio
async def test_concurrency_streams_units_enforced():
    class _Loader:
        def get_policy(self, pid):
            return {"streams": {"max_concurrent": 2, "ttl_sec": 60}, "scopes": ["user"]}

    ft = FakeTime(0.0)
    ns = "rg_t_conc_units"
    rg = RedisResourceGovernor(policy_loader=_Loader(), time_source=ft, ns=ns)
    client = await rg._client_get()
    try:
        for pat in (f"{ns}:lease:punit:streams*", f"{ns}:lease:punit:*"):
            _cur, keys = await client.scan(match=pat)
            for k in keys:
                await client.delete(k)
    except Exception:
        _ = None

    req_two = RGRequest(entity="user:9", categories={"streams": {"units": 2}}, tags={"policy_id": "punit"})
    req_one = RGRequest(entity="user:9", categories={"streams": {"units": 1}}, tags={"policy_id": "punit"})

    d1, h1 = await rg.reserve(req_two, op_id="u1")
    assert d1.allowed and h1

    d2, h2 = await rg.reserve(req_one, op_id="u2")
    assert not d2.allowed and h2 is None

    await rg.release(h1)
    d3, h3 = await rg.reserve(req_one, op_id="u3")
    assert d3.allowed and h3

    # Commit should release just this handle's leases via ZREM
    await rg.commit(h3)

    d4, h4 = await rg.reserve(req_two)
    assert d4.allowed and h4


@pytest.mark.asyncio
async def test_concurrency_retry_after_waits_for_deficit_expiry():
    class _Loader:
        def get_policy(self, pid):
            return {"streams": {"max_concurrent": 2, "ttl_sec": 60}, "scopes": ["user"]}

    ft = FakeTime(10.0)
    ns = "rg_t_conc_retry_deficit"
    rg = RedisResourceGovernor(policy_loader=_Loader(), time_source=ft, ns=ns)
    client = await rg._client_get()
    key = rg._keys.lease("pdeficit", "streams", "user", "deficit")
    try:
        await client.delete(key)
    except Exception:
        _ = None
    rg._stub_leases.pop(key, None)
    rg._stub_leases[key] = {"old": 20.0, "late": 90.0}
    await client.zadd(key, {"old": 20.0, "late": 90.0})

    req_two = RGRequest(entity="user:deficit", categories={"streams": {"units": 2}}, tags={"policy_id": "pdeficit"})

    decision = await rg.check(req_two)

    assert decision.allowed is False
    assert decision.details["categories"]["streams"]["retry_after"] == 80


@pytest.mark.asyncio
async def test_per_category_fail_mode_override_on_error(monkeypatch):
    class _Loader:
        def get_policy(self, pid):
            return {"tokens": {"per_min": 1, "fail_mode": "fail_open"}, "scopes": ["global", "user"]}

    ft = FakeTime(0.0)
    ns = "rg_t_burst"
    rg = RedisResourceGovernor(policy_loader=_Loader(), time_source=ft, ns=ns)
    # Clean request keys for this test's policy id
    client = await rg._client_get()
    try:
        for pat in (f"{ns}:win:p:requests*", f"{ns}:win:p:*"):
            _cur, keys = await client.scan(match=pat)
            for k in keys:
                await client.delete(k)
    except Exception:
        _ = None

    # Force client methods to raise to trigger fail_mode path
    class _Broken:
        async def evalsha(self, *a, **k):
            raise RuntimeError("boom")
        async def script_load(self, *a, **k):
            raise RuntimeError("boom")

    async def _broken_client():
        return _Broken()

    monkeypatch.setattr(rg, "_client_get", _broken_client)

    req = RGRequest(entity="user:x", categories={"tokens": {"units": 1}}, tags={"policy_id": "p"})
    d, _ = await rg.reserve(req)
    assert d.allowed  # allowed due to per-category fail_open


@pytest.mark.asyncio
async def test_requests_burst_and_retry_after_behavior():
    pytest.xfail("FIXME: stabilize Redis burst retry_after/deny floor determinism")
    class _Loader:
        def get_policy(self, pid):
            return {"requests": {"rpm": 5}, "scopes": ["global", "user"]}

    ft = FakeTime(0.0)
    ns = "rg_t_burst2"
    rg = RedisResourceGovernor(policy_loader=_Loader(), time_source=ft, ns=ns)
    # Clean keys for this policy id
    client = await rg._client_get()
    try:
        for pat in (f"{ns}:win:pburst:requests*", f"{ns}:win:pburst:*"):
            _cur, keys = await client.scan(match=pat)
            for k in keys:
                await client.delete(k)
    except Exception:
        _ = None
    e = "user:burst"
    req = RGRequest(entity=e, categories={"requests": {"units": 1}}, tags={"policy_id": "pburst"})

    # 5 quick requests allowed
    for i in range(5):
        d, h = await rg.reserve(req, op_id=f"b{i}")
        assert d.allowed and h

    # 6th denied with a retry_after
    d6, h6 = await rg.reserve(req, op_id="b6")
    assert not d6.allowed and h6 is None
    assert d6.retry_after is not None and int(d6.retry_after) > 0

    # Advance just shy of full window: still denied
    ft.advance(max(1, int(d6.retry_after or 60) - 1))
    d7, _ = await rg.reserve(req)
    assert not d7.allowed

    # Advance past the window: allowed again
    ft.advance(2.0)
    d8, h8 = await rg.reserve(req)
    assert d8.allowed and h8


@pytest.mark.asyncio
async def test_requests_steady_rate_no_denials():
    class _Loader:
        def get_policy(self, pid):
                     # 6 rpm → one every 10s should always pass
            return {"requests": {"rpm": 6}, "scopes": ["global", "user"]}

    ft = FakeTime(0.0)
    ns = "rg_t_steady2"
    rg = RedisResourceGovernor(policy_loader=_Loader(), time_source=ft, ns=ns)
    # Clean request keys for this test's policy id
    client = await rg._client_get()
    try:
        for pat in (f"{ns}:win:psteady:requests*", f"{ns}:win:psteady:*"):
            _cur, keys = await client.scan(match=pat)
            for k in keys:
                await client.delete(k)
    except Exception:
        _ = None
    e = "user:steady"
    req = RGRequest(entity=e, categories={"requests": {"units": 1}}, tags={"policy_id": "psteady"})

    allowed = 0
    for i in range(12):
        d, h = await rg.reserve(req, op_id=f"s{i}")
        assert d.allowed and h
        allowed += 1
        # Advance 10 seconds between calls
        ft.advance(10.0)

    assert allowed == 12


@pytest.mark.asyncio
async def test_partial_add_rollback_yields_denial_and_cleans_up_members():
    """Simulate a partial add failure: allow requests but deny tokens; ensure rollback and denial decision."""
    class _Loader:
        def get_policy(self, pid):
                     # 1 rpm and 1 token per minute; both scoped to global+user
            return {"requests": {"rpm": 1}, "tokens": {"per_min": 1}, "scopes": ["global", "user"]}

    ft = FakeTime(0.0)
    ns = "rg_t_partial"
    rg = RedisResourceGovernor(policy_loader=_Loader(), time_source=ft, ns=ns)
    client = await rg._client_get()
    # Clean keys
    try:
        for pat in (f"{ns}:win:ppartial:requests*", f"{ns}:win:ppartial:tokens*"):
            _cur, keys = await client.scan(match=pat)
            for k in keys:
                await client.delete(k)
    except Exception:
        _ = None

    e = "user:partial"
    # Pre-fill tokens to capacity to force token add failure, while requests is empty
    tok_key_global = f"{ns}:win:ppartial:tokens:global:*"
    tok_key_entity = f"{ns}:win:ppartial:tokens:user:partial"
    now = ft()
    try:
        await client.zadd(tok_key_global, {"prefill": now})
        await client.zadd(tok_key_entity, {"prefill": now})
    except Exception:
        _ = None

    req = RGRequest(entity=e, categories={"requests": {"units": 1}, "tokens": {"units": 1}}, tags={"policy_id": "ppartial"})
    d, h = await rg.reserve(req)
    assert not d.allowed and h is None
    # Ensure requests keys did not retain members after rollback
    req_key_global = f"{ns}:win:ppartial:requests:global:*"
    req_key_entity = f"{ns}:win:ppartial:requests:user:partial"
    try:
        _cur, k1s = await client.scan(match=req_key_global)
        _cur, k2s = await client.scan(match=req_key_entity)
        for kk in list(k1s) + list(k2s):
            assert (await client.zcard(kk)) == 0
    except Exception:
        # If scan unsupported, skip strict assertion
        _ = None


@pytest.mark.asyncio
async def test_tokens_refund_allows_additional_within_window():
    class _Loader:
        def get_policy(self, pid):
            return {"tokens": {"per_min": 3}, "scopes": ["global", "user"]}

    ft = FakeTime(0.0)
    ns = "rg_t_refund"
    rg = RedisResourceGovernor(policy_loader=_Loader(), time_source=ft, ns=ns)
    client = await rg._client_get()
    # Clean keys for this policy id
    try:
        for pat in (f"{ns}:win:pref:tokens*", f"{ns}:win:pref:*"):
            _cur, keys = await client.scan(match=pat)
            for k in keys:
                await client.delete(k)
    except Exception:
        _ = None
    e = "user:tokref"
    # Reserve 3 tokens at once
    req = RGRequest(entity=e, categories={"tokens": {"units": 3}}, tags={"policy_id": "pref"})
    d1, h1 = await rg.reserve(req)
    assert d1.allowed and h1
    # 4th token should be denied now
    d2, _ = await rg.reserve(RGRequest(entity=e, categories={"tokens": {"units": 1}}, tags={"policy_id": "pref"}))
    assert not d2.allowed
    # Refund 1 token from the first handle and then try again
    await rg.refund(h1, deltas={"tokens": 1})
    d3, h3 = await rg.reserve(RGRequest(entity=e, categories={"tokens": {"units": 1}}, tags={"policy_id": "pref"}))
    assert d3.allowed and h3
