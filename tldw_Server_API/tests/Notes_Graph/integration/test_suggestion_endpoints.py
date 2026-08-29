from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from threading import Event, get_ident
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.requests import Request

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.api.v1.endpoints import notes_graph_suggestions as endpoint
from tldw_Server_API.app.api.v1.schemas.notes_graph_suggestions import (
    SuggestionDecisionRequest,
    SuggestionResetRequest,
    SuggestionRunCreateRequest,
)
from tldw_Server_API.app.core.AuthNZ.permissions import (
    KEYWORDS_CREATE,
    NOTES_GRAPH_READ,
    NOTES_GRAPH_SUGGEST,
    NOTES_GRAPH_WRITE,
    NOTES_LINK_KEYWORD,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.Notes_Graph.suggestion_api import SuggestionAPIError
from tldw_Server_API.app.core.Sync.v2.notes_link_coordinator import (
    NotesLinkDatasetConflictError,
)

pytestmark = pytest.mark.integration

NOTE_ID = "00000000-0000-4000-8000-000000000001"
REVISION = f"sha256:{'4' * 64}"
FINGERPRINT = f"sha256:{'5' * 64}"
NOW = datetime(2026, 8, 27, 20, 0, tzinfo=timezone.utc)


def _principal(
    permissions: tuple[str, ...],
    *,
    roles: tuple[str, ...] = ("user",),
    is_admin: bool = False,
) -> AuthPrincipal:
    return AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject=None,
        token_type="access",
        jti=None,
        roles=list(roles),
        permissions=list(permissions),
        is_admin=is_admin,
        org_ids=[],
        team_ids=[],
    )


class FakeAPI:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.accept_permissions = (NOTES_GRAPH_WRITE,)
        self.error: SuggestionAPIError | None = None
        self.suggestion_items: tuple[object, ...] = ()

    def _record(self, name: str, kwargs: dict[str, object]):
        self.calls.append((name, kwargs))
        if self.error is not None:
            raise self.error

    def get_capabilities(self, **kwargs):
        self._record("capabilities", kwargs)
        return SimpleNamespace(
            provider="openai",
            model="model-a",
            endpoint_origin_revision=f"sha256:{'3' * 64}",
            data_boundary="remote",
            disclosure_external=True,
            outbound_data_categories=("selected_note_excerpts",),
            generation_available=True,
            unavailable_reason=None,
            limits=SimpleNamespace(
                max_candidates=30,
                max_relationships=5,
                max_tags=5,
                max_new_tags=2,
                max_tag_catalog=100,
                max_estimated_input_tokens=24_000,
                max_output_tokens=2_000,
                provider_timeout_seconds=120,
                response_candidates=1,
            ),
            allowed_actions=("generate", "cancel", "accept", "reject", "reset_rejections"),
            revision=REVISION,
        )

    def admit_run(self, **kwargs):
        self._record("admit", kwargs)
        return SimpleNamespace(
            disposition="created",
            run=SimpleNamespace(
                id="run-1",
                provider="openai",
                model="model-a",
                state=SimpleNamespace(value="queued"),
                revision=2,
                created_at=NOW.isoformat(),
                started_at=None,
                completed_at=None,
                suggestion_count=0,
                related_note_count=0,
                tag_count=0,
                invalid_item_count=0,
                error_code=None,
                guidance_key=None,
            ),
            job={"uuid": "private-job-id"},
        )

    def list_runs(self, **kwargs):
        self._record("list_runs", kwargs)
        return SimpleNamespace(items=(), next_cursor=None)

    def get_run(self, **kwargs):
        self._record("get_run", kwargs)
        return self.admit_run(**kwargs).run

    def cancel_run(self, **kwargs):
        self._record("cancel", kwargs)
        return SimpleNamespace(
            cancellation=SimpleNamespace(replay_envelope={"run_id": "run-1", "state": "cancelling", "revision": 3}),
            accepted=True,
        )

    def list_suggestions(self, **kwargs):
        self._record("list_suggestions", kwargs)
        return SimpleNamespace(
            items=self.suggestion_items,
            next_cursor=None,
            current_source_fingerprint=FINGERPRINT,
            rejection_set_revision=0,
            rejection_count=0,
        )

    def reset_rejections(self, **kwargs):
        self._record("reset", kwargs)
        return SimpleNamespace(
            envelope={"source_note_id": NOTE_ID, "state": "reset", "revision": 1, "cleared_count": 0}
        )

    def reject_suggestion(self, **kwargs):
        self._record("reject", kwargs)
        return SimpleNamespace(envelope={"suggestion_id": "suggestion-1", "state": "rejected", "revision": 2})

    def accept_permission_requirements(self, **kwargs):
        self._record("accept_permissions", kwargs)
        return self.accept_permissions

    def accept_suggestion(self, **kwargs):
        self._record("accept", kwargs)
        return SimpleNamespace(envelope={"suggestion_id": "suggestion-1", "state": "accepted", "revision": 2})


def _app(
    fake: FakeAPI,
    permissions: tuple[str, ...],
    *,
    principal: AuthPrincipal | None = None,
    override_rate_limit: bool = True,
) -> FastAPI:
    app = FastAPI()
    app.include_router(endpoint.router, prefix="/api/v1/notes")
    principal = principal or _principal(permissions)

    async def auth_principal(request: Request) -> AuthPrincipal:
        request.state.auth = AuthContext(
            principal=principal,
            ip=None,
            user_agent=None,
            request_id=None,
        )
        return principal

    async def request_user():
        return SimpleNamespace(id=1, id_str="1", roles=["user"], permissions=list(permissions))

    app.dependency_overrides[auth_deps.get_auth_principal] = auth_principal
    app.dependency_overrides[endpoint.get_request_user] = request_user
    app.dependency_overrides[endpoint.get_chacha_db_for_user] = lambda: SimpleNamespace()
    app.dependency_overrides[endpoint.try_get_job_manager] = lambda: SimpleNamespace()
    app.dependency_overrides[auth_deps.get_db_pool] = lambda: SimpleNamespace()
    endpoint.build_notes_graph_suggestions_api = lambda **_kwargs: fake

    async def allow() -> None:
        return None

    for route in app.routes:
        dependant = getattr(route, "dependant", None)
        if dependant is None:
            continue
        for dependency in dependant.dependencies:
            call = dependency.call
            if getattr(call, "_tldw_token_scope", False):
                app.dependency_overrides[call] = allow
            if override_rate_limit and getattr(call, "_tldw_rate_limit_resource", None) is not None:
                app.dependency_overrides[call] = allow
    return app


def _base_permissions() -> tuple[str, ...]:
    return (NOTES_GRAPH_READ, NOTES_GRAPH_SUGGEST)


def test_public_request_schemas_are_bounded_and_forbid_provider_authority_fields() -> None:
    request = SuggestionRunCreateRequest(provider="openai", model="model-a")
    assert request.model_dump() == {"provider": "openai", "model": "model-a"}
    for forbidden in ("endpoint", "endpoint_url", "api_key", "credential", "prompt", "candidate_count"):
        with pytest.raises(ValidationError):
            SuggestionRunCreateRequest.model_validate(
                {"provider": "openai", "model": "model-a", forbidden: "not-allowed"}
            )
    with pytest.raises(ValidationError):
        SuggestionDecisionRequest(
            expected_revision=0,
            expected_source_fingerprint=FINGERPRINT,
            expected_target_fingerprint=None,
        )
    with pytest.raises(ValidationError):
        SuggestionResetRequest(
            expected_rejection_revision=0,
            source_fingerprint=FINGERPRINT,
            confirm=False,
        )


@pytest.mark.asyncio
async def test_sync_facade_releases_database_connection_on_its_worker_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_threads: list[int] = []
    close_threads: list[int] = []
    db = SimpleNamespace(release_context_connection=lambda: close_threads.append(get_ident()))

    class API:
        @staticmethod
        def read() -> str:
            call_threads.append(get_ident())
            return "ok"

    monkeypatch.setattr(
        endpoint,
        "build_notes_graph_suggestions_api",
        lambda **_kwargs: API(),
    )

    result = await endpoint._call_api(
        user=SimpleNamespace(id=1, id_str="1"),
        db=db,
        jobs=SimpleNamespace(),
        dataset_id=None,
        operation=lambda api: api.read(),
    )

    assert result == "ok"
    assert call_threads == close_threads
    assert call_threads != [get_ident()]


def test_suggestion_list_serializes_the_server_authoritative_target_title() -> None:
    fake = FakeAPI()
    fake.suggestion_items = (
        SimpleNamespace(
            suggestion=SimpleNamespace(
                id="suggestion-1",
                run_id="run-1",
                kind=SimpleNamespace(value="related_note"),
                state=SimpleNamespace(value="pending"),
                revision=1,
                source_note_id=NOTE_ID,
                source_fingerprint=FINGERPRINT,
                target_note_id="00000000-0000-4000-8000-000000000002",
                target_fingerprint=FINGERPRINT,
                normalized_tag=None,
                display_tag=None,
                keyword_sync_id=None,
                match_strength="strong",
                rationale="Grounded",
                updated_at=NOW,
            ),
            target_title="Authoritative target",
            evidence=(),
        ),
    )

    with TestClient(_app(fake, _base_permissions())) as client:
        response = client.get(
            f"/api/v1/notes/{NOTE_ID}/graph/suggestions",
        )

    assert response.status_code == 200, response.text
    assert response.json()["items"][0]["target_title"] == "Authoritative target"


def test_capability_sets_etag_and_run_admission_is_durable_202_without_jobs_internals() -> None:
    fake = FakeAPI()
    with TestClient(_app(fake, _base_permissions())) as client:
        capability = client.get(
            f"/api/v1/notes/{NOTE_ID}/graph/suggestions/capabilities",
            params={"provider": "openai", "model": "model-a"},
        )
        admitted = client.post(
            f"/api/v1/notes/{NOTE_ID}/graph/suggestions/runs",
            json={"provider": "openai", "model": "model-a"},
            headers={"If-Match": f'"{REVISION}"', "Idempotency-Key": "run-key"},
        )
        detail = client.get(f"/api/v1/notes/{NOTE_ID}/graph/suggestions/runs/run-1")

    assert capability.status_code == 200
    assert capability.headers["etag"] == f'"{REVISION}"'
    assert admitted.status_code == 202
    assert admitted.json()["id"] == "run-1"
    assert "job_id" not in admitted.json()
    assert detail.status_code == 200
    assert detail.json()["state"] == "queued"


@pytest.mark.asyncio
async def test_run_admission_yields_the_event_loop_during_sync_facade_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = Event()
    release = Event()

    class BlockingAPI(FakeAPI):
        def admit_run(self, **kwargs):
            started.set()
            if not release.wait(timeout=1):
                raise AssertionError("event loop could not release synchronous admission")
            return super().admit_run(**kwargs)

    fake = BlockingAPI()
    monkeypatch.setattr(
        endpoint,
        "build_notes_graph_suggestions_api",
        lambda **_kwargs: fake,
    )
    task = asyncio.create_task(
        endpoint.create_suggestion_run(
            note_id=NOTE_ID,
            body=SuggestionRunCreateRequest(provider="openai", model="model-a"),
            dataset_id=None,
            if_match=f'"{REVISION}"',
            idempotency_key="run-key",
            user=SimpleNamespace(id=1, id_str="1"),
            db=SimpleNamespace(),
            jobs=SimpleNamespace(),
            _principal=_principal(_base_permissions()),
            _rate=None,
            _scope=None,
        )
    )

    assert await asyncio.to_thread(started.wait, 0.5) is True
    release.set()
    response = await task

    assert response.id == "run-1"


@pytest.mark.asyncio
async def test_run_listing_yields_the_event_loop_during_sync_facade_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = Event()
    release = Event()

    class BlockingAPI(FakeAPI):
        def list_runs(self, **kwargs):
            started.set()
            if not release.wait(timeout=1):
                raise AssertionError("event loop could not release synchronous run listing")
            return super().list_runs(**kwargs)

    fake = BlockingAPI()
    monkeypatch.setattr(
        endpoint,
        "build_notes_graph_suggestions_api",
        lambda **_kwargs: fake,
    )
    task = asyncio.create_task(
        endpoint.list_suggestion_runs(
            note_id=NOTE_ID,
            state=None,
            limit=20,
            cursor=None,
            dataset_id=None,
            user=SimpleNamespace(id=1, id_str="1"),
            db=SimpleNamespace(),
            jobs=SimpleNamespace(),
            _principal=_principal(_base_permissions()),
            _rate=None,
            _scope=None,
        )
    )

    assert await asyncio.to_thread(started.wait, 0.5) is True
    release.set()
    response = await task

    assert response.items == ()


@pytest.mark.asyncio
async def test_suggestion_acceptance_yields_the_event_loop_during_sync_facade_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = Event()
    release = Event()

    class BlockingAPI(FakeAPI):
        def accept_suggestion(self, **kwargs):
            started.set()
            if not release.wait(timeout=1):
                raise AssertionError("event loop could not release synchronous suggestion acceptance")
            return super().accept_suggestion(**kwargs)

    fake = BlockingAPI()
    monkeypatch.setattr(
        endpoint,
        "build_notes_graph_suggestions_api",
        lambda **_kwargs: fake,
    )
    principal = _principal((*_base_permissions(), NOTES_GRAPH_WRITE))
    task = asyncio.create_task(
        endpoint.accept_suggestion(
            note_id=NOTE_ID,
            suggestion_id="suggestion-1",
            body=SuggestionDecisionRequest(
                expected_revision=1,
                expected_source_fingerprint=FINGERPRINT,
                expected_target_fingerprint=None,
            ),
            dataset_id=None,
            idempotency_key="accept-key",
            user=SimpleNamespace(id=1, id_str="1"),
            db=SimpleNamespace(),
            jobs=SimpleNamespace(),
            principal=principal,
            _rate=None,
            _scope=None,
        )
    )

    assert await asyncio.to_thread(started.wait, 0.5) is True
    release.set()
    response = await task

    assert response.resource_id == "suggestion-1"


def test_in_progress_cancellation_returns_authoritative_cancelling_run() -> None:
    fake = FakeAPI()
    fake.cancel_run = lambda **_kwargs: SimpleNamespace(
        cancellation=SimpleNamespace(
            replay_envelope=None,
            run=SimpleNamespace(id="run-1", state=SimpleNamespace(value="cancelling"), revision=3),
        ),
        accepted=False,
    )
    with TestClient(_app(fake, _base_permissions())) as client:
        response = client.post(
            f"/api/v1/notes/{NOTE_ID}/graph/suggestions/runs/run-1/cancel",
            json={"expected_revision": 2},
            headers={"Idempotency-Key": "cancel-key"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "resource_id": "run-1",
        "state": "cancelling",
        "revision": 3,
        "cleared_count": None,
    }


def test_terminal_admission_replay_returns_the_original_public_run_envelope() -> None:
    fake = FakeAPI()
    envelope = {
        "run_id": "run-replay",
        "provider": "openai",
        "model": "model-original",
        "state": "queued",
        "revision": 7,
        "created_at": NOW.isoformat().replace("+00:00", "Z"),
        "started_at": None,
        "completed_at": None,
        "suggestion_count": 4,
        "related_note_count": 3,
        "tag_count": 1,
        "invalid_item_count": 2,
        "cancellation_available": True,
        "error_code": None,
        "guidance_key": None,
    }
    fake.admit_run = lambda **_kwargs: SimpleNamespace(
        disposition="terminal_replay",
        run=None,
        job=None,
        replay_envelope=envelope,
    )
    with TestClient(_app(fake, _base_permissions()), raise_server_exceptions=False) as client:
        response = client.post(
            f"/api/v1/notes/{NOTE_ID}/graph/suggestions/runs",
            json={"provider": "openai", "model": "model-original"},
            headers={"If-Match": f'"{REVISION}"', "Idempotency-Key": "run-key"},
        )

    assert response.status_code == 202
    assert response.json() == {"id": "run-replay", **{k: v for k, v in envelope.items() if k != "run_id"}}


def test_run_admission_rejects_a_non_hex_capability_etag_before_calling_the_facade() -> None:
    fake = FakeAPI()
    with TestClient(_app(fake, _base_permissions())) as client:
        response = client.post(
            f"/api/v1/notes/{NOTE_ID}/graph/suggestions/runs",
            json={"provider": "openai", "model": "model-a"},
            headers={"If-Match": f'"sha256:{"g" * 64}"', "Idempotency-Key": "run-key"},
        )

    assert response.status_code == 422
    assert response.json()["detail"]["error_code"] == "notes_graph_invalid_request"
    assert fake.calls == []


def test_run_admission_rejects_bare_unquoted_if_match() -> None:
    fake = FakeAPI()
    with TestClient(_app(fake, _base_permissions())) as client:
        response = client.post(
            f"/api/v1/notes/{NOTE_ID}/graph/suggestions/runs",
            json={"provider": "openai", "model": "model-a"},
            headers={"If-Match": REVISION, "Idempotency-Key": "run-key"},
        )

    assert response.status_code == 422
    assert response.json()["detail"]["error_code"] == "notes_graph_invalid_request"
    assert fake.calls == []


def test_suggestion_list_defaults_to_pending_and_accepting_and_enforces_limit() -> None:
    fake = FakeAPI()
    with TestClient(_app(fake, _base_permissions())) as client:
        response = client.get(f"/api/v1/notes/{NOTE_ID}/graph/suggestions")
        too_large = client.get(
            f"/api/v1/notes/{NOTE_ID}/graph/suggestions",
            params={"limit": 101},
        )
    assert response.status_code == 200
    assert fake.calls[-1][0] == "list_suggestions"
    assert fake.calls[-1][1]["states"] == ("pending", "accepting")
    assert too_large.status_code == 422


@pytest.mark.parametrize(
    ("status_code", "code"),
    [
        (404, "notes_graph_suggestion_not_found"),
        (409, "notes_graph_suggestion_conflict"),
        (412, "notes_graph_capabilities_changed"),
        (422, "notes_graph_provider_disallowed"),
        (429, "notes_graph_admission_rate_limited"),
        (503, "notes_graph_provider_call_policy_unsupported"),
        (503, "notes_graph_provider_not_configured"),
        (503, "notes_graph_provider_retry_policy_unsupported"),
        (503, "notes_graph_provider_unavailable"),
    ],
)
def test_typed_domain_errors_map_to_stable_sanitized_http_contract(
    status_code: int,
    code: str,
) -> None:
    fake = FakeAPI()
    fake.error = SuggestionAPIError(status_code, code)
    with TestClient(_app(fake, _base_permissions())) as client:
        response = client.get(f"/api/v1/notes/{NOTE_ID}/graph/suggestions/capabilities")
    assert response.status_code == status_code
    assert response.json()["detail"] == {
        "error_code": code,
        "message": endpoint.SUGGESTION_ERROR_MESSAGES[code],
    }
    assert "private" not in response.text.lower()


def test_base_suggestion_permission_is_required_by_authoritative_principal() -> None:
    fake = FakeAPI()
    with TestClient(_app(fake, (NOTES_GRAPH_READ,))) as client:
        response = client.get(f"/api/v1/notes/{NOTE_ID}/graph/suggestions/capabilities")
    assert response.status_code == 403
    assert NOTES_GRAPH_SUGGEST in response.text
    assert fake.calls == []


@pytest.mark.parametrize(
    ("required", "granted", "expected"),
    [
        ((NOTES_GRAPH_WRITE,), (), 403),
        ((NOTES_GRAPH_WRITE,), (NOTES_GRAPH_WRITE,), 200),
        ((NOTES_LINK_KEYWORD,), (), 403),
        ((NOTES_LINK_KEYWORD,), (NOTES_LINK_KEYWORD,), 200),
        ((NOTES_LINK_KEYWORD, KEYWORDS_CREATE), (NOTES_LINK_KEYWORD,), 403),
        (
            (NOTES_LINK_KEYWORD, KEYWORDS_CREATE),
            (NOTES_LINK_KEYWORD, KEYWORDS_CREATE),
            200,
        ),
    ],
)
def test_accept_enforces_kind_specific_permissions_from_authoritative_principal(
    required: tuple[str, ...],
    granted: tuple[str, ...],
    expected: int,
) -> None:
    fake = FakeAPI()
    fake.accept_permissions = required
    permissions = (*_base_permissions(), *granted)
    with TestClient(_app(fake, permissions)) as client:
        response = client.post(
            f"/api/v1/notes/{NOTE_ID}/graph/suggestions/suggestion-1/accept",
            json={
                "expected_revision": 1,
                "expected_source_fingerprint": FINGERPRINT,
                "expected_target_fingerprint": None,
            },
            headers={"Idempotency-Key": "decision-key"},
        )
    assert response.status_code == expected


def test_accept_does_not_trust_forged_is_admin_boolean() -> None:
    fake = FakeAPI()
    forged = _principal(_base_permissions(), is_admin=True)
    with TestClient(_app(fake, _base_permissions(), principal=forged)) as client:
        response = client.post(
            f"/api/v1/notes/{NOTE_ID}/graph/suggestions/suggestion-1/accept",
            json={
                "expected_revision": 1,
                "expected_source_fingerprint": FINGERPRINT,
                "expected_target_fingerprint": None,
            },
            headers={"Idempotency-Key": "decision-key"},
        )

    assert response.status_code == 403
    assert all(name != "accept" for name, _kwargs in fake.calls)


@pytest.mark.parametrize(
    "required",
    [
        (NOTES_GRAPH_WRITE,),
        (NOTES_LINK_KEYWORD,),
        (NOTES_LINK_KEYWORD, KEYWORDS_CREATE),
    ],
)
def test_accept_allows_verified_admin_claims_for_each_suggestion_kind(
    required: tuple[str, ...],
) -> None:
    fake = FakeAPI()
    fake.accept_permissions = required
    verified_admin = _principal(_base_permissions(), roles=("admin",), is_admin=False)
    with TestClient(_app(fake, _base_permissions(), principal=verified_admin)) as client:
        response = client.post(
            f"/api/v1/notes/{NOTE_ID}/graph/suggestions/suggestion-1/accept",
            json={
                "expected_revision": 1,
                "expected_source_fingerprint": FINGERPRINT,
                "expected_target_fingerprint": None,
            },
            headers={"Idempotency-Key": "decision-key"},
        )

    assert response.status_code == 200


_ROUTE_CASES = (
    ("get", f"/api/v1/notes/{NOTE_ID}/graph/suggestions/capabilities", None, {}),
    (
        "post",
        f"/api/v1/notes/{NOTE_ID}/graph/suggestions/runs",
        {"provider": "openai", "model": "model-a"},
        {"If-Match": f'"{REVISION}"', "Idempotency-Key": "run-key"},
    ),
    ("get", f"/api/v1/notes/{NOTE_ID}/graph/suggestions/runs", None, {}),
    ("get", f"/api/v1/notes/{NOTE_ID}/graph/suggestions/runs/run-1", None, {}),
    (
        "post",
        f"/api/v1/notes/{NOTE_ID}/graph/suggestions/runs/run-1/cancel",
        {"expected_revision": 2},
        {"Idempotency-Key": "cancel-key"},
    ),
    ("get", f"/api/v1/notes/{NOTE_ID}/graph/suggestions", None, {}),
    (
        "post",
        f"/api/v1/notes/{NOTE_ID}/graph/suggestions/rejections/reset",
        {
            "expected_rejection_revision": 1,
            "source_fingerprint": FINGERPRINT,
            "confirm": True,
        },
        {"Idempotency-Key": "reset-key"},
    ),
    (
        "post",
        f"/api/v1/notes/{NOTE_ID}/graph/suggestions/suggestion-1/accept",
        {
            "expected_revision": 1,
            "expected_source_fingerprint": FINGERPRINT,
            "expected_target_fingerprint": None,
        },
        {"Idempotency-Key": "accept-key"},
    ),
    (
        "post",
        f"/api/v1/notes/{NOTE_ID}/graph/suggestions/suggestion-1/reject",
        {
            "expected_revision": 1,
            "expected_source_fingerprint": FINGERPRINT,
            "expected_target_fingerprint": None,
        },
        {"Idempotency-Key": "reject-key"},
    ),
)


@pytest.mark.parametrize(("method", "path", "body", "headers"), _ROUTE_CASES)
def test_invalid_or_cross_owner_dataset_is_sanitized_404_on_every_route(
    monkeypatch,
    method: str,
    path: str,
    body: dict[str, object] | None,
    headers: dict[str, str],
) -> None:
    def reject_dataset(**_kwargs):
        raise NotesLinkDatasetConflictError()

    monkeypatch.setattr(endpoint, "resolve_notes_link_dataset_authority", reject_dataset)
    with TestClient(
        _app(FakeAPI(), _base_permissions()),
        raise_server_exceptions=False,
    ) as client:
        response = client.request(
            method,
            path,
            params={"dataset_id": "other-owner-dataset"},
            json=body,
            headers=headers,
        )

    assert response.status_code == 404
    assert response.json()["detail"] == {
        "error_code": "notes_graph_suggestion_not_found",
        "message": endpoint.SUGGESTION_ERROR_MESSAGES["notes_graph_suggestion_not_found"],
    }


@pytest.mark.parametrize(("method", "path", "body", "headers"), _ROUTE_CASES)
def test_rbac_rate_limit_is_stable_structured_429_on_every_route(
    monkeypatch,
    method: str,
    path: str,
    body: dict[str, object] | None,
    headers: dict[str, str],
) -> None:
    async def deny_rate(*_args, **_kwargs):
        raise HTTPException(status_code=429, detail="generic RBAC detail")

    monkeypatch.setattr(auth_deps, "enforce_rbac_rate_limit", deny_rate)
    with TestClient(
        _app(FakeAPI(), _base_permissions(), override_rate_limit=False),
    ) as client:
        response = client.request(method, path, json=body, headers=headers)

    assert response.status_code == 429
    assert response.json()["detail"] == {
        "error_code": "notes_graph_admission_rate_limited",
        "message": endpoint.SUGGESTION_ERROR_MESSAGES["notes_graph_admission_rate_limited"],
    }


def test_openapi_declares_bounded_headers_and_structured_suggestion_errors() -> None:
    schema = _app(FakeAPI(), _base_permissions()).openapi()
    run_operation = schema["paths"]["/api/v1/notes/{note_id}/graph/suggestions/runs"]["post"]
    parameters = {item["name"]: item["schema"] for item in run_operation["parameters"]}

    def string_branch(name: str) -> dict[str, object]:
        return next(
            branch for branch in parameters[name].get("anyOf", (parameters[name],)) if branch.get("type") == "string"
        )

    assert string_branch("Idempotency-Key")["minLength"] == 1
    assert string_branch("Idempotency-Key")["maxLength"] == 256
    assert string_branch("If-Match")["minLength"] == 1
    assert string_branch("If-Match")["maxLength"] == 128
    for operation_path in schema["paths"].values():
        for operation in operation_path.values():
            if "notes-graph-suggestions" not in operation.get("tags", []):
                continue
            for status_code in ("404", "409", "412", "422", "429", "503"):
                response_schema = operation["responses"][status_code]["content"]["application/json"]["schema"]
                assert response_schema["$ref"].endswith("/SuggestionHTTPErrorResponse")
