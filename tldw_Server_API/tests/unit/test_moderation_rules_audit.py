"""Rules configuration writes retain authenticated, payload-safe audit attribution."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import Audit_DB_Deps, auth_deps
from tldw_Server_API.app.api.v1.endpoints import moderation
from tldw_Server_API.app.core.Audit.unified_audit_service import AuditContext, AuditEventType, UnifiedAuditService
from tldw_Server_API.app.core.AuthNZ.permissions import SYSTEM_CONFIGURE
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.Moderation import moderation_service

PRIVATE_PATTERN = "S05-PRIVATE-RULE-DO-NOT-AUDIT"


@pytest_asyncio.fixture
async def rule_case(tmp_path, monkeypatch):
    configuration = {
        "moderation": {
            "enabled": "true",
            "input_enabled": "true",
            "input_action": "block",
            "blocklist_file": str(tmp_path / "blocklist.txt"),
            "user_overrides_file": str(tmp_path / "users.json"),
            "runtime_overrides_file": str(tmp_path / "runtime.json"),
        }
    }
    monkeypatch.setattr(moderation_service, "load_and_log_configs", lambda: configuration)
    service = moderation_service.ModerationService()
    monkeypatch.setattr(moderation, "get_moderation_service", lambda: service)
    audit = UnifiedAuditService(db_path=str(tmp_path / "audit.db"), storage_mode="per_user")
    await audit.initialize(start_background_tasks=False)

    async def for_actor(user_id=None):
        assert user_id == 41  # nosec B101 -- Verify the audit tenant comes from the authenticated actor.
        return audit

    monkeypatch.setattr(Audit_DB_Deps, "get_or_create_audit_service_for_user_id_optional", for_actor)
    actor = AuthPrincipal(  # nosec B106 -- 'access' is the token-kind enum, not a credential.
        kind="user",
        user_id=41,
        api_key_id=None,
        subject="user:41",
        token_type="access",
        jti=None,
        roles=["admin"],
        permissions=[SYSTEM_CONFIGURE],
        is_admin=True,
        org_ids=[],
        team_ids=[],
    )
    app = FastAPI()
    app.include_router(moderation.router, prefix="/api/v1")

    async def principal(request: Request):
        request.state.auth = AuthContext(principal=actor, ip=None, user_agent=None, request_id=None)
        return actor

    app.dependency_overrides[auth_deps.get_auth_principal] = principal
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            yield SimpleNamespace(client=client, service=service, audit=audit, actor=actor)
    finally:
        await audit.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation", ["append", "delete", "replace", "override_set", "override_delete", "settings", "reload"]
)
async def test_successful_rules_write_has_persisted_actor_audit(rule_case, operation):
    case = rule_case
    headers = {}
    body = None
    if operation in {"delete", "replace"}:
        assert case.service.set_blocklist_lines([PRIVATE_PATTERN])  # nosec B101 -- Real private storage precondition.
    if operation == "override_delete":
        assert case.service.set_user_override("9", {"enabled": True})["ok"]  # nosec B101 -- Real private storage precondition.
    if operation in {"append", "delete"}:
        headers["If-Match"] = '"' + case.service.get_blocklist_state()["version"] + '"'
    method, path, action = {
        "append": ("POST", "/blocklist/append", "moderation.blocklist.append"),
        "delete": ("DELETE", "/blocklist/0", "moderation.blocklist.delete"),
        "replace": ("PUT", "/blocklist", "moderation.blocklist.replace"),
        "override_set": ("PUT", "/users/9", "moderation.override.set"),
        "override_delete": ("DELETE", "/users/9", "moderation.override.delete"),
        "settings": ("PUT", "/settings", "moderation.settings.update"),
        "reload": ("POST", "/reload", "moderation.reload"),
    }[operation]
    if operation == "append":
        body = {"line": PRIVATE_PATTERN}
    elif operation == "replace":
        body = {"lines": ["S05 replacement"]}
    elif operation == "override_set":
        body = {"enabled": True}
    elif operation == "settings":
        body = {"pii_enabled": True, "persist": True}
    response = await case.client.request(method, "/api/v1/moderation" + path, headers=headers, json=body)
    assert response.status_code == 200  # nosec B101 -- The actual mutation must succeed before checking audit.
    if operation == "append":
        assert case.service.get_blocklist_lines() == [PRIVATE_PATTERN]  # nosec B101 -- Prove real persisted mutation.
    events = await case.audit.query_events(user_id="41", event_types=[AuditEventType.CONFIG_CHANGED])
    assert len(events) == 1  # nosec B101 -- Exactly one durable attributed event per successful write.
    event = events[0]
    assert event["action"] == action and event["context_user_id"] == "41"  # nosec B101 -- Actor/action are part of the contract.
    assert json.loads(event["metadata"])["actor_id"] == case.actor.principal_id  # nosec B101 -- Use authenticated principal identity.
    assert PRIVATE_PATTERN not in json.dumps(event)  # nosec B101 -- Rule text must not leak into audit.


@pytest.mark.asyncio
async def test_stale_rule_write_has_no_mutation_or_success_audit(rule_case):
    case = rule_case
    before = case.service.get_blocklist_state()
    response = await case.client.post(
        "/api/v1/moderation/blocklist/append", headers={"If-Match": '"stale"'}, json={"line": PRIVATE_PATTERN}
    )
    assert response.status_code == 412  # nosec B101 -- Existing optimistic concurrency refusal.
    assert case.service.get_blocklist_state() == before  # nosec B101 -- A refused write cannot change the policy.
    assert await case.audit.query_events(user_id="41") == []  # nosec B101 -- Refusal is not a successful rules change.


@pytest.mark.asyncio
async def test_denied_rules_write_has_no_mutation_or_success_audit(rule_case):
    case = rule_case
    case.actor.roles = []
    case.actor.permissions = []
    case.actor.is_admin = False
    response = await case.client.put("/api/v1/moderation/blocklist", json={"lines": [PRIVATE_PATTERN]})
    assert response.status_code == 403  # nosec B101 -- Existing admin and permission boundary stays enforced.
    assert case.service.get_blocklist_lines() == []  # nosec B101 -- Unauthorized input cannot change rules.
    assert await case.audit.query_events(user_id="41") == []  # nosec B101 -- No success record for denied mutation.


@pytest.mark.asyncio
@pytest.mark.parametrize("flagged", [False, True])
async def test_rules_test_audits_verdict_not_sample_or_target_identity(rule_case, flagged):
    case = rule_case
    assert case.service.set_blocklist_lines([PRIVATE_PATTERN])  # nosec B101 -- Real private test policy.
    response = await case.client.post(
        "/api/v1/moderation/test",
        json={"text": PRIVATE_PATTERN if flagged else "Welcome to Rowan Observatory.", "user_id": "999"},
    )
    assert response.status_code == 200 and response.json()["flagged"] is flagged  # nosec B101 -- Actual policy outcome.
    events = await case.audit.query_events(user_id="41", event_types=[AuditEventType.SECURITY_SCAN])
    assert len(events) == 1  # nosec B101 -- Testing a rule also needs attributed outcome.
    event = events[0]
    details = json.loads(event["metadata"])
    assert event["action"] == "moderation.test" and event["context_user_id"] == "41"  # nosec B101 -- Simulation target is not actor.
    assert details["flagged"] is flagged and details["verdict"] == ("block" if flagged else "pass")  # nosec B101 -- Fixed outcome.
    assert details["actor_id"] == case.actor.principal_id and len(details["policy_fingerprint"]) == 64  # nosec B101 -- Authenticated attribution and immutable policy link.
    assert PRIVATE_PATTERN not in json.dumps(event)  # nosec B101 -- Raw sample and pattern are private.


@pytest.mark.asyncio
async def test_payload_and_headers_cannot_spoof_audit_actor(rule_case):
    case = rule_case
    response = await case.client.put(
        "/api/v1/moderation/blocklist",
        headers={"X-Actor-ID": "999"},
        json={"lines": [PRIVATE_PATTERN], "actor_id": "999"},
    )
    assert response.status_code == 200  # nosec B101 -- Accepted payload still uses authenticated actor.
    events = await case.audit.query_events(user_id="41", event_types=[AuditEventType.CONFIG_CHANGED])
    assert len(events) == 1 and json.loads(events[0]["metadata"])["actor_id"] == case.actor.principal_id  # nosec B101 -- No client actor override.


@pytest.mark.asyncio
async def test_failed_rule_storage_has_no_success_audit(rule_case, monkeypatch):
    case = rule_case
    monkeypatch.setattr(case.service, "set_blocklist_lines", lambda _lines: False)
    response = await case.client.put("/api/v1/moderation/blocklist", json={"lines": [PRIVATE_PATTERN]})
    assert response.status_code == 500  # nosec B101 -- Preserve existing persistence failure.
    assert case.service.get_blocklist_lines() == []  # nosec B101 -- No stored policy change.
    assert await case.audit.query_events(user_id="41") == []  # nosec B101 -- No successful audit for failed storage.


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["service", "log_event", "flush"])
async def test_audit_failure_reports_uncertain_applied_change_without_replay(rule_case, monkeypatch, failure_stage):
    case = rule_case

    async def unavailable(*_args, **_kwargs):
        raise OSError("private audit location or credentials must not appear")

    with monkeypatch.context() as injection:
        if failure_stage == "service":
            injection.setattr(Audit_DB_Deps, "get_or_create_audit_service_for_user_id_optional", unavailable)
        else:
            injection.setattr(case.audit, failure_stage, unavailable)
        response = await case.client.put("/api/v1/moderation/blocklist", json={"lines": [PRIVATE_PATTERN]})
    assert response.status_code == 503  # nosec B101 -- No false fully audited success.
    assert case.service.get_blocklist_lines() == [PRIVATE_PATTERN]  # nosec B101 -- Never pretend file mutation rolled back.
    assert "read the current rules/settings before retrying" in response.json()["detail"].lower()  # nosec B101 -- Explicit uncertain-write recovery.
    assert "private audit location" not in response.text  # nosec B101 -- No backend error leakage.


@pytest.mark.asyncio
async def test_rules_response_waits_for_its_real_audit_commit(rule_case, monkeypatch):
    case = rule_case
    entered, release, logged = asyncio.Event(), asyncio.Event(), asyncio.Event()
    transaction = case.audit._immediate_write_transaction
    log_event = case.audit.log_event
    tasks = []
    event_ids = []

    @asynccontextmanager
    async def held_transaction(db):
        async with transaction(db):
            entered.set()
            await release.wait()
            yield

    async def log_then_background_flush(*args, **kwargs):
        event_id = await log_event(*args, **kwargs)
        event_ids.append(event_id)
        tasks.append(asyncio.create_task(case.audit.flush()))
        await entered.wait()
        logged.set()
        return event_id

    try:
        with monkeypatch.context() as injection:
            injection.setattr(case.audit, "_immediate_write_transaction", held_transaction)
            injection.setattr(case.audit, "log_event", log_then_background_flush)
            request = asyncio.create_task(
                case.client.put("/api/v1/moderation/blocklist", json={"lines": [PRIVATE_PATTERN]})
            )
            tasks.append(request)
            await asyncio.wait_for(logged.wait(), 5)
            assert not request.done()  # nosec B101 -- The public success response must await durable audit.
            async with case.audit._read_db() as db:
                assert await case.audit._fetch_existing_event_ids(db, event_ids) == set()  # nosec B101 -- No auto-flush in the witness.
            release.set()
            response = await asyncio.wait_for(request, 5)
            assert response.status_code == 200  # nosec B101 -- Success follows commit.
            async with case.audit._read_db() as db:
                assert await case.audit._fetch_existing_event_ids(db, event_ids) == set(event_ids)  # nosec B101 -- Exact event is already durable at response.
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 10)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["cancel", "failed_spill"])
async def test_rules_write_never_acknowledges_a_lost_overflow_event(rule_case, monkeypatch, outcome):
    case = rule_case
    assert case.audit._test_mode  # nosec B101 -- Use the service's normal synchronous test flushing.
    case.audit.buffer_size = 1
    entered, release = asyncio.Event(), asyncio.Event()
    transaction = case.audit._immediate_write_transaction
    attempts = 0
    tasks = []
    request_event_ids = []

    @asynccontextmanager
    async def held_first_transaction(db):
        nonlocal attempts
        attempts += 1
        first = attempts == 1
        async with transaction(db):
            if first:
                entered.set()
                await release.wait()
                if outcome == "failed_spill":
                    raise OSError("controlled transaction failure")
            yield

    def failed_spill(*_args):
        raise OSError("controlled fallback failure")

    async def queued_request_event():
        while True:
            async with case.audit.buffer_lock:
                matching = [
                    event.event_id
                    for event in case.audit.event_buffer
                    if event.action == "moderation.blocklist.replace"
                ]
                if matching:
                    assert len(matching) == 1  # nosec B101 -- Capture the exact one-shot request event.
                    request_event_ids.extend(matching)
                    assert len(case.audit.event_buffer) == 3  # nosec B101 -- Normal arrivals exceed the two-event recovery cap.
                    return
            await asyncio.sleep(0)

    try:
        with monkeypatch.context() as injection:
            injection.setattr(case.audit, "_immediate_write_transaction", held_first_transaction)
            injection.setattr(case.audit, "_append_events_to_fallback", failed_spill)
            first = asyncio.create_task(
                case.audit.log_event(
                    AuditEventType.CONFIG_CHANGED, context=AuditContext(user_id="41"), action="controlled.first"
                )
            )
            tasks.append(first)
            await asyncio.wait_for(entered.wait(), 5)
            for index in range(2):
                tasks.append(
                    asyncio.create_task(
                        case.audit.log_event(
                            AuditEventType.CONFIG_CHANGED,
                            context=AuditContext(user_id="41"),
                            action="controlled.arrival." + str(index),
                        )
                    )
                )
            request = asyncio.create_task(
                case.client.put("/api/v1/moderation/blocklist", json={"lines": [PRIVATE_PATTERN]})
            )
            tasks.append(request)
            await asyncio.wait_for(queued_request_event(), 5)
            if outcome == "cancel":
                first.cancel()
            release.set()
            response = await asyncio.wait_for(request, 5)
            assert response.status_code == 503  # nosec B101 -- Missing mandatory event cannot be acknowledged.
            async with case.audit._read_db() as db:
                assert await case.audit._fetch_existing_event_ids(db, request_event_ids) == set()  # nosec B101 -- Exact request event really is missing, without auto-flush.
            assert case.service.get_blocklist_lines() == [PRIVATE_PATTERN]  # nosec B101 -- Do not falsely roll back the applied change.
            assert "read the current rules/settings before retrying" in response.json()["detail"].lower()  # nosec B101 -- Truthful uncertain-write recovery.
    finally:
        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 10)
