from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError, ResponseValidationError
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import (
    get_chacha_db_for_user,
)
from tldw_Server_API.app.api.v1.API_Deps.Collections_DB_Deps import (
    get_collections_db_for_user,
)
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.Slides_DB_Deps import (
    get_slides_db_for_user,
)
from tldw_Server_API.app.api.v1.endpoints import slides_standalone_html
from tldw_Server_API.app.api.v1.endpoints.slides import (
    _load_version_payload,
    _slides_lifespan,
)
from tldw_Server_API.app.api.v1.endpoints.slides import (
    router as slides_router,
)
from tldw_Server_API.app.api.v1.schemas.slides_schemas import (
    ExportFormat,
    PresentationPatchRequest,
    SlidesCapabilitiesResponse,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.Security.standalone_html_request_guard import (
    is_standalone_sensitive_route,
    standalone_request_validation_response,
    standalone_response_invalid_response,
)
from tldw_Server_API.app.core.Slides.slides_db import SlidesDatabase
from tldw_Server_API.app.core.Slides.standalone_html_config import (
    ResolvedExecutionTarget,
    ResolvedPrompt,
    SlidesStandaloneHtmlConfig,
    StandaloneHtmlInputLimits,
    StandaloneHtmlOutputLimits,
    StandaloneHtmlProviderLimits,
)
from tldw_Server_API.app.core.Slides.standalone_html_registry import (
    DigestKeyAvailability,
    DigestKeyMetadata,
    DigestKeySnapshot,
    DigestKeyState,
    StandaloneHtmlHmacKeyring,
)
from tldw_Server_API.app.core.Slides.standalone_html_service import (
    StandaloneHtmlGenerationError,
    StandaloneHtmlGenerationSubmission,
)
from tldw_Server_API.app.core.Slides.standalone_html_validator import (
    validate_standalone_html,
)
from tldw_Server_API.app.services.lifecycle_worker_specs import WorkerLifecycleContext

_ACCEPT = "X-Slides-Accept-Content-Kinds"
_BOTH = {_ACCEPT: "structured_slides,standalone_html"}
_REVISION = "sha256:" + "a" * 64
_GENERATION_REQUEST = {
    "generation_mode": "standalone_html",
    "generation_config_revision": _REVISION,
    "source": {"kind": "prompt", "prompt": "Build a safe systems deck"},
    "html_options": {
        "presentation_type": "tech-sharing",
        "audience": "backend engineers",
        "slide_count": 10,
        "visual_direction": "dark-technical",
        "delivery_style": "speaker-led",
    },
}


def _runtime_config(
    *,
    enabled: bool = True,
    reason: str | None = None,
    revision: str = _REVISION,
):
    target = (
        SimpleNamespace(
            provider="openai",
            model="allowed-model",
            adapter_id="openai_official_chat_v1",
            endpoint_identity="https://api.openai.com:443/v1/chat/completions",
        )
        if enabled
        else None
    )
    return SlidesStandaloneHtmlConfig(
        feature_enabled=enabled,
        egress_enabled=enabled,
        enabled=enabled,
        disabled_reason=reason,
        target=target,
        prompt=(
            SimpleNamespace(
                text="Build a standalone presentation.",
                sha256="b" * 64,
                contract_version="slides.standalone_html.v1",
                byte_count=32,
            )
            if enabled
            else None
        ),
        allowed_targets=((target,) if target is not None else ()),
        generation_config_revision=revision if enabled else None,
        input_limits=SimpleNamespace(
            max_request_bytes=4_194_304,
            max_source_chars=200_000,
            max_source_tokens=50_000,
            max_audience_chars=500,
            max_source_identifier_bytes=256,
            max_note_ids=100,
            max_rag_query_chars=20_000,
            max_rag_top_k=100,
        ),
        output_limits=SimpleNamespace(
            max_provider_response_bytes=8_388_608,
            max_document_bytes=1_048_576,
        ),
        provider_limits=SimpleNamespace(
            connect_timeout_seconds=10.0,
            read_timeout_seconds=120.0,
            overall_timeout_seconds=180.0,
            max_output_tokens=16_384,
        ),
        _revision_manifest="test-manifest" if enabled else "",
    )


def _lifecycle_runtime_config() -> SlidesStandaloneHtmlConfig:
    target = ResolvedExecutionTarget(
        provider="openai",
        model="allowed-model",
        adapter_id="openai_official_chat_v1",
        endpoint_identity="https://api.openai.com:443/v1/chat/completions",
    )
    return SlidesStandaloneHtmlConfig(
        feature_enabled=True,
        egress_enabled=True,
        enabled=True,
        disabled_reason=None,
        target=target,
        prompt=ResolvedPrompt(
            text="Build a standalone presentation.",
            sha256="b" * 64,
            contract_version="slides.standalone_html.v1",
            byte_count=32,
        ),
        allowed_targets=(target,),
        generation_config_revision=_REVISION,
        input_limits=StandaloneHtmlInputLimits(
            max_request_bytes=4_194_304,
            max_source_chars=200_000,
            max_source_tokens=50_000,
            max_audience_chars=500,
            max_source_identifier_bytes=256,
            max_note_ids=100,
            max_rag_query_chars=20_000,
            max_rag_top_k=100,
        ),
        output_limits=StandaloneHtmlOutputLimits(
            max_provider_response_bytes=8_388_608,
            max_document_bytes=1_048_576,
        ),
        provider_limits=StandaloneHtmlProviderLimits(
            connect_timeout_seconds=10.0,
            read_timeout_seconds=120.0,
            overall_timeout_seconds=180.0,
            max_output_tokens=16_384,
        ),
        _revision_manifest="test-manifest",
    )


@pytest.mark.asyncio
async def test_generation_http_uses_lifecycle_owned_transport_and_cleans_up(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    from tldw_Server_API.app.core import config as config_module
    from tldw_Server_API.app.core.DB_Management import db_path_utils
    from tldw_Server_API.app.core.Slides import (
        standalone_html_config,
        standalone_html_reconciler,
        standalone_html_registry,
    )
    from tldw_Server_API.app.services import startup_content_jobs_pollers

    app = FastAPI()
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="1")
    config = _lifecycle_runtime_config()
    shared_job_manager = object()
    shared_keyring = StandaloneHtmlHmacKeyring(
        secrets={"key-v1": b"k" * 32},
        current_key_id="key-v1",
    )
    stop_event = asyncio.Event()
    handler_started = asyncio.Event()
    allocations = {"job_manager": 0, "keyring": 0, "registry": 0}
    registries = []
    reconcilers = []
    handler_runtimes = []

    class _Registry:
        def __init__(self, *, store, keyring) -> None:
            del store
            assert keyring is shared_keyring
            allocations["registry"] += 1
            self.snapshot_calls = 0
            self.snapshot_value = DigestKeySnapshot(
                records=(),
                config_epoch=None,
                configured_current_key_id="key-v1",
                availability=DigestKeyAvailability(missing_key_ids=()),
            )
            registries.append(self)

        async def snapshot(self):
            self.snapshot_calls += 1
            return self.snapshot_value

        async def activate_configured_current(self, **kwargs):
            self.snapshot_value = DigestKeySnapshot(
                records=(
                    DigestKeyMetadata(
                        key_id="key-v1",
                        state=DigestKeyState.CURRENT,
                        activated_at=datetime(2026, 8, 21, tzinfo=timezone.utc),
                        retired_at=None,
                    ),
                ),
                config_epoch=kwargs["new_config_epoch"],
                configured_current_key_id="key-v1",
                availability=DigestKeyAvailability(missing_key_ids=()),
            )
            return self.snapshot_value

    class _Reconciler:
        def __init__(self, **_kwargs) -> None:
            self.released = False
            reconcilers.append(self)

        def admission_ready(self) -> bool:
            return True

        def run_batch(self):
            return SimpleNamespace(
                startup_ready=True,
                leader=True,
                completed_pass=True,
                jobs_available=True,
                local_sweep_state="not_run",
            )

        def release(self) -> bool:
            self.released = True
            return True

    def _job_manager_factory():
        allocations["job_manager"] += 1
        return shared_job_manager

    class _KeyringFactory:
        @classmethod
        def from_env(cls):
            allocations["keyring"] += 1
            return shared_keyring

    async def _handler(runtime, worker_stop_event):
        handler_runtimes.append(runtime)
        handler_started.set()
        await worker_stop_event.wait()

    async def _override_user():
        return User(
            id=1,
            username="tester",
            email=None,
            is_active=True,
            is_admin=True,
        )

    async def _override_principal(request=None):
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="test-user",
            token_type="single_user",  # nosec B106 - test principal type
            jti=None,
            roles=["admin"],
            permissions=["media.create"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        if request is not None:
            request.state.auth = AuthContext(
                principal=principal,
                ip=None,
                user_agent=None,
                request_id=None,
            )
        return principal

    async def _override_db():
        yield db

    monkeypatch.setattr(
        startup_content_jobs_pollers,
        "_standalone_html_jobs_manager",
        _job_manager_factory,
    )
    monkeypatch.setattr(
        startup_content_jobs_pollers,
        "_get_worker_owned_validation_pool",
        lambda _app: asyncio.sleep(0, result=object()),
    )
    monkeypatch.setattr(
        startup_content_jobs_pollers,
        "_run_standalone_html_generation_handler",
        _handler,
    )
    monkeypatch.setattr(
        db_path_utils.DatabasePaths,
        "resolve_user_db_base_dir",
        lambda: tmp_path / "users",
    )
    monkeypatch.setattr(config_module, "load_comprehensive_config", lambda: {})
    monkeypatch.setattr(config_module, "refresh_config_cache", lambda: None)
    monkeypatch.setattr(
        standalone_html_config,
        "load_standalone_html_config",
        lambda _raw, *, availability: config,
    )
    monkeypatch.setattr(
        standalone_html_registry,
        "StandaloneHtmlHmacKeyring",
        _KeyringFactory,
    )
    monkeypatch.setattr(
        standalone_html_registry,
        "JobManagerDigestKeyRegistryStore",
        lambda manager: manager,
    )
    monkeypatch.setattr(
        standalone_html_registry,
        "StandaloneHtmlKeyRegistry",
        _Registry,
    )
    monkeypatch.setattr(
        standalone_html_reconciler,
        "FencedStandaloneHtmlReconciler",
        _Reconciler,
    )

    app.include_router(slides_router, prefix="/api/v1", tags=["slides"])
    app.dependency_overrides[get_request_user] = _override_user
    app.dependency_overrides[get_auth_principal] = _override_principal
    app.dependency_overrides[get_slides_db_for_user] = _override_db
    lifecycle_context = WorkerLifecycleContext(
        app=app,
        settings={},
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )
    lifecycle_task = asyncio.create_task(
        startup_content_jobs_pollers._run_standalone_html_generation_jobs_service(
            lifecycle_context,
            stop_event,
        )
    )

    try:
        await asyncio.wait_for(handler_started.wait(), timeout=1)
        await asyncio.sleep(0)
        published = app.state.standalone_html_transport_context
        assert not hasattr(app.state, "standalone_html_api_runtime")
        assert handler_runtimes == [published]
        assert published.job_manager is shared_job_manager
        assert published.admission_gate.open is True
        snapshot_calls_before_request = registries[0].snapshot_calls

        stale_request = {
            **_GENERATION_REQUEST,
            "generation_config_revision": "sha256:" + "c" * 64,
        }
        async with AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                "/api/v1/slides/generations",
                headers={"Idempotency-Key": "lifecycle-owned-http-request"},
                json=stale_request,
            )

        assert response.status_code == 409
        assert response.json() == {"detail": "generation_configuration_changed"}
        assert registries[0].snapshot_calls == snapshot_calls_before_request + 1
        assert allocations == {"job_manager": 1, "keyring": 1, "registry": 1}
        assert app.state.standalone_html_transport_context is published
    finally:
        stop_event.set()
        await asyncio.wait_for(lifecycle_task, timeout=1)
        db.close_connection()

    assert published.admission_gate.open is False
    assert reconcilers[0].released is True
    assert not hasattr(app.state, "standalone_html_transport_context")
    assert app.state.standalone_html_generation_worker_registered is False
    assert app.state.standalone_html_reconciler_admission_ready is False


@pytest.mark.asyncio
async def test_request_runtime_reuses_lifecycle_owned_fenced_transport(
    tmp_path,
):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="1")
    app = FastAPI()
    shared_jobs = object()
    shared_keyring = object()

    async def _digest_snapshot_loader():
        return object()

    context = SimpleNamespace(
        local_only=False,
        job_manager=shared_jobs,
        keyring=shared_keyring,
        digest_snapshot_loader=_digest_snapshot_loader,
        current_config_loader=lambda: _runtime_config(),
        reconciler=object(),
        admission_gate=SimpleNamespace(open=True),
        validator_available=True,
    )
    app.state.standalone_html_transport_context = context
    runtime = await slides_standalone_html._build_runtime(SimpleNamespace(app=app), db)

    assert runtime.job_manager is shared_jobs
    assert runtime.generation_service.slides_db is db
    assert runtime.generation_service.job_manager is shared_jobs
    assert runtime.generation_service.keyring is shared_keyring
    db.close_connection()


@pytest.mark.asyncio
@pytest.mark.parametrize("local_only", [None, True], ids=["absent", "local-only"])
async def test_request_runtime_without_full_lifecycle_context_fails_closed_without_resources(
    tmp_path,
    local_only,
):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="1")
    app = FastAPI()
    if local_only is not None:
        app.state.standalone_html_transport_context = SimpleNamespace(
            local_only=local_only,
            job_manager=object(),
        )

    try:
        runtime = await slides_standalone_html._build_runtime(
            SimpleNamespace(app=app),
            db,
        )
        config = runtime.config_loader()
        assert runtime.job_manager is None
        assert runtime.generation_service.job_manager is None
        assert runtime.generation_service.keyring is None
        assert config.enabled is False
        assert config.disabled_reason in {
            "feature_disabled",
            "digest_key_unavailable",
            "generation_worker_unavailable",
        }
    finally:
        db.close_connection()


@pytest.mark.asyncio
async def test_lifecycle_runtime_checks_digest_fence_before_stale_config_and_source(tmp_path):
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="1")
    app = FastAPI()
    events: list[str] = []
    keyring = StandaloneHtmlHmacKeyring(
        secrets={"key-v1": b"k" * 32},
        current_key_id="key-v1",
    )
    snapshot = DigestKeySnapshot(
        records=(
            DigestKeyMetadata(
                key_id="key-v1",
                state=DigestKeyState.CURRENT,
                activated_at=datetime(2026, 8, 20, tzinfo=timezone.utc),
                retired_at=None,
            ),
        ),
        config_epoch="expected-epoch",
        configured_current_key_id="key-v1",
        availability=DigestKeyAvailability(missing_key_ids=()),
    )

    async def _digest_snapshot_loader():
        events.append("digest")
        return snapshot

    def _current_config_loader():
        events.append("config")
        return _runtime_config(revision="sha256:" + "b" * 64)

    context = SimpleNamespace(
        local_only=False,
        job_manager=object(),
        keyring=keyring,
        digest_snapshot_loader=_digest_snapshot_loader,
        current_config_loader=_current_config_loader,
        reconciler=object(),
        admission_gate=SimpleNamespace(open=True),
        validator_available=True,
        config_epoch="expected-epoch",
    )
    app.state.standalone_html_transport_context = context
    runtime = await slides_standalone_html._build_runtime(SimpleNamespace(app=app), db)

    async def _source_resolver(_source, _limits):
        events.append("source")
        raise AssertionError("stale configuration must not resolve source")

    try:
        with pytest.raises(StandaloneHtmlGenerationError) as exc_info:
            await runtime.generation_service.submit(
                owner_user_id="1",
                idempotency_key="task11-stale-config-key",
                request=_GENERATION_REQUEST,
                config_loader=runtime.config_loader,
                source_resolver=_source_resolver,
            )
        assert exc_info.value.code == "generation_configuration_changed"
        assert events == ["digest", "config"]
    finally:
        db.close_connection()


class _FakeGenerationService:
    def __init__(self) -> None:
        self.owner_user_id = "1"
        self.submission = StandaloneHtmlGenerationSubmission(
            receipt_id="018f2f4a-6f79-7a27-a1aa-7bb60777d9f1",
            status="queued",
            job_uuid="018f2f4a-6f79-7a27-a1aa-7bb60777d9f2",
            presentation_id=None,
            replayed=False,
        )
        self.submit_calls: list[dict[str, object]] = []
        self.status_calls: list[tuple[str, str]] = []
        self.submit_error: StandaloneHtmlGenerationError | None = None

    async def submit(self, **kwargs):
        self.submit_calls.append(kwargs)
        if self.submit_error is not None:
            raise self.submit_error
        return self.submission

    def get_generation(self, *, owner_user_id: str, receipt_id: str):
        self.status_calls.append((owner_user_id, receipt_id))
        if owner_user_id != self.owner_user_id or receipt_id != self.submission.receipt_id:
            raise StandaloneHtmlGenerationError("generation_not_found", status_code=404)
        return self.submission


class _FakeStandaloneRuntime:
    def __init__(self) -> None:
        self.config = _runtime_config()
        self.validator_available = True
        self.generation_service = _FakeGenerationService()
        self.reconcile_calls: list[str] = []
        self.receipt_error: tuple[str | None, str | None] = (None, None)
        self.job = {"progress_message": "Resolving source", "progress_percent": 25.0}

    def config_loader(self):
        return self.config

    def reconcile_owner(self, owner_user_id: str):
        self.reconcile_calls.append(owner_user_id)
        return SimpleNamespace(jobs_available=True)

    def receipt_error_fields(self, owner_user_id: str, receipt_id: str):
        assert owner_user_id == "1"
        assert receipt_id == self.generation_service.submission.receipt_id
        return self.receipt_error

    def job_progress(self, job_uuid: str, owner_user_id: str):
        assert job_uuid == self.generation_service.submission.job_uuid
        assert owner_user_id == "1"
        return self.job


class _InlineValidationPool:
    def __init__(self, db: SlidesDatabase) -> None:
        self.db = db
        self.calls: list[str | bytes] = []
        self.closed = False

    async def validate(self, document: str | bytes):
        assert not self.db.get_connection().in_transaction
        self.calls.append(document)
        return validate_standalone_html(document)

    async def close(self) -> None:
        self.closed = True


def _assert_operation_error(response, *, operation: str, content_kind: str) -> None:
    assert response.status_code == 409
    assert response.json() == {
        "detail": "operation_not_supported_for_content_kind",
        "operation": operation,
        "content_kind": content_kind,
    }


def _document(*, title: str = "HTML Deck", text: str = "Visible HTML text") -> str:
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{title}</title><style>.slide{{color:#111}}</style></head>"
        f'<body><section class="slide"><h1>{text}</h1>'
        '<aside class="notes">Hidden note</aside></section>'
        "<script>document.addEventListener('keydown', () => {});</script>"
        "</body></html>"
    )


def _provenance_json() -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "source_kind": "prompt",
            "source_ref": None,
            "source_snapshot_hmac_sha256": "a" * 64,
            "digest_key_id": "slides-generation-v1",
            "source_bytes": 10,
            "provider": "openai",
            "model": "test-model",
            "adapter_id": "openai_official_chat_v1",
            "endpoint_identity": "https://api.openai.com:443/v1/chat/completions",
            "prompt_sha256": "b" * 64,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _create_html(db: SlidesDatabase, *, presentation_id: str = "html"):
    document = _document()
    derived = validate_standalone_html(document)
    return db.create_presentation(
        presentation_id=presentation_id,
        title=derived.title,
        description=None,
        theme="black",
        marp_theme=None,
        settings=None,
        studio_data=None,
        slides="[]",
        slides_text=derived.indexable_text,
        source_type="prompt",
        source_ref=None,
        source_query=None,
        custom_css=None,
        content_kind="standalone_html",
        html_document=document,
        html_sha256=derived.html_sha256,
        html_bytes=derived.html_bytes,
        html_slide_count=derived.slide_count,
        generation_job_uuid=f"job-{presentation_id}",
        generation_provenance_json=_provenance_json(),
    )


def _create_structured(db: SlidesDatabase, *, presentation_id: str = "structured"):
    slides = [
        {
            "order": 0,
            "layout": "title",
            "title": "Structured Deck",
            "content": "",
            "speaker_notes": None,
            "metadata": {},
        }
    ]
    return db.create_presentation(
        presentation_id=presentation_id,
        title="Structured Deck",
        description=None,
        theme="black",
        marp_theme=None,
        settings=None,
        studio_data=None,
        slides=json.dumps(slides),
        slides_text="Structured Deck",
        source_type="manual",
        source_ref=None,
        source_query=None,
        custom_css=None,
    )


class _Collections:
    list_calls = 0

    def get_output_artifact(self, output_id: int):
        return {"id": output_id}

    def resolve_output_storage_path(self, path_value):
        return str(path_value)

    def list_output_artifacts(self, **_kwargs):
        type(self).list_calls += 1
        return [], 0


class _SourceDbCalls:
    media = 0
    chacha = 0

    @classmethod
    def reset(cls) -> None:
        cls.media = 0
        cls.chacha = 0


@pytest.fixture()
def html_client(tmp_path):
    _Collections.list_calls = 0
    _SourceDbCalls.reset()
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="1")
    structured = _create_structured(db)
    html = _create_html(db)
    app = FastAPI()
    validation_pool = _InlineValidationPool(db)
    standalone_runtime = _FakeStandaloneRuntime()
    app.state.standalone_html_validation_pool = validation_pool
    app.state.standalone_html_api_runtime = standalone_runtime
    app.include_router(slides_router, prefix="/api/v1", tags=["slides"])

    @app.exception_handler(RequestValidationError)
    async def _request_validation_handler(request: Request, exc: RequestValidationError):
        if is_standalone_sensitive_route(request.method, request.url.path):
            return standalone_request_validation_response(request, exc)
        raise exc

    @app.exception_handler(ResponseValidationError)
    async def _response_validation_handler(request: Request, exc: ResponseValidationError):
        if is_standalone_sensitive_route(request.method, request.url.path):
            return standalone_response_invalid_response(exc)
        raise exc

    @app.exception_handler(Exception)
    async def _sensitive_exception_handler(request: Request, exc: Exception):
        if is_standalone_sensitive_route(request.method, request.url.path):
            return standalone_response_invalid_response(exc)
        raise exc

    async def _override_user():
        return User(
            id=1,
            username="tester",
            email=None,
            is_active=True,
            is_admin=True,
        )

    async def _override_principal(request=None):
        principal = AuthPrincipal(
            kind="user",
            user_id=1,
            api_key_id=None,
            subject="test-user",
            token_type="single_user",  # nosec B106 - test principal type
            jti=None,
            roles=["admin"],
            permissions=[
                "media.create",
                "media.read",
                "media.update",
                "media.delete",
            ],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        if request is not None:
            request.state.auth = AuthContext(
                principal=principal,
                ip=None,
                user_agent=None,
                request_id=None,
            )
        return principal

    async def _override_db():
        yield db

    async def _override_collections():
        return _Collections()

    async def _override_media_db():
        _SourceDbCalls.media += 1
        yield SimpleNamespace(source_db="media")

    async def _override_chacha_db():
        _SourceDbCalls.chacha += 1
        return SimpleNamespace(source_db="chacha")

    app.dependency_overrides[get_request_user] = _override_user
    app.dependency_overrides[get_auth_principal] = _override_principal
    app.dependency_overrides[get_slides_db_for_user] = _override_db
    app.dependency_overrides[get_collections_db_for_user] = _override_collections
    app.dependency_overrides[get_media_db_for_user] = _override_media_db
    app.dependency_overrides[get_chacha_db_for_user] = _override_chacha_db

    with TestClient(app) as client:
        yield client, db, structured, html

    assert validation_pool.closed
    assert getattr(app.state, "standalone_html_validation_pool", None) is None
    assert getattr(app.state, "standalone_html_validation_pool_lock", None) is None
    app.dependency_overrides.clear()
    db.close_connection()


@pytest.mark.parametrize(
    "value",
    ["", " ", ",", "structured_slides,", "bad token", "future_kind"],
)
def test_malformed_or_unknown_only_negotiation_is_fixed_400(html_client, value):
    client, _db, _structured, _html = html_client

    response = client.get(
        "/api/v1/slides/presentations",
        headers={_ACCEPT: value},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid_content_kind_header"
    assert _ACCEPT.lower() in response.headers["Vary"].lower()


def test_slides_lifespan_without_html_pool_shuts_down_cleanly():
    app = FastAPI()
    app.include_router(slides_router, prefix="/api/v1")

    with TestClient(app):
        assert getattr(app.state, "standalone_html_validation_pool", None) is None

    assert getattr(app.state, "standalone_html_validation_pool", None) is None
    assert getattr(app.state, "standalone_html_validation_pool_lock", None) is None


def test_slides_lifespan_defers_worker_owned_pool_cleanup_until_composite_shutdown(tmp_path):
    app = FastAPI()
    db = SlidesDatabase(db_path=tmp_path / "Slides.db", client_id="1")
    validation_pool = _InlineValidationPool(db)
    app.state.standalone_html_validation_pool = validation_pool
    app.state.standalone_html_validation_pool_worker_owned = True
    app.include_router(slides_router, prefix="/api/v1")

    with TestClient(app):
        pass

    assert validation_pool.closed is False
    assert app.state.standalone_html_validation_pool is validation_pool
    asyncio.run(validation_pool.close())
    db.close_connection()


@pytest.mark.asyncio
async def test_worker_owned_slides_lifespan_does_not_suppress_endpoint_errors():
    app = FastAPI()
    app.state.standalone_html_validation_pool_worker_owned = True

    with pytest.raises(RuntimeError, match="endpoint failed"):
        async with _slides_lifespan(app):
            raise RuntimeError("endpoint failed")


def test_negotiated_validation_errors_match_fastapi_body_and_add_vary(html_client):
    client, _db, structured, _html = html_client
    baseline_app = FastAPI()

    @baseline_app.get("/presentations")
    async def baseline_list(limit: int = Query(50, ge=1, le=200)):
        return {"limit": limit}

    @baseline_app.patch("/presentations/{presentation_id}")
    async def baseline_patch(
        presentation_id: str,
        request: PresentationPatchRequest,
    ):
        return {"id": presentation_id, "request": request.model_dump()}

    with TestClient(baseline_app) as baseline:
        baseline_query = baseline.get("/presentations?limit=0")
        baseline_body = baseline.patch("/presentations/structured", json={"title": []})

    query = client.get("/api/v1/slides/presentations?limit=0")
    body = client.patch(
        f"/api/v1/slides/presentations/{structured.id}",
        json={"title": []},
        headers={"If-Match": 'W/"v1"'},
    )

    assert query.status_code == baseline_query.status_code == 422
    assert body.status_code == baseline_body.status_code == 422
    assert query.json() == baseline_query.json()
    assert body.json() == baseline_body.json()
    assert _ACCEPT.lower() in query.headers["Vary"].lower()
    assert _ACCEPT.lower() in body.headers["Vary"].lower()


def _html_mutation_requests(html_id: str):
    return [
        (
            "PUT",
            f"/api/v1/slides/presentations/{html_id}/html-source",
            {
                "content": _document().encode("utf-8"),
                "headers": {"Content-Type": "application/octet-stream"},
            },
        ),
        ("PUT", f"/api/v1/slides/presentations/{html_id}", {"json": {"title": "No"}}),
        ("PATCH", f"/api/v1/slides/presentations/{html_id}", {"json": {"title": "No"}}),
        ("POST", f"/api/v1/slides/presentations/{html_id}/reorder", {"json": {"order": [0]}}),
        ("DELETE", f"/api/v1/slides/presentations/{html_id}", {}),
        ("POST", f"/api/v1/slides/presentations/{html_id}/restore", {}),
        ("POST", f"/api/v1/slides/presentations/{html_id}/versions/1/restore", {}),
    ]


def test_negotiated_mutations_reject_malformed_kind_before_missing_if_match(html_client):
    client, _db, _structured, html = html_client

    for method, path, kwargs in _html_mutation_requests(html.id):
        kwargs["headers"] = {**kwargs.get("headers", {}), _ACCEPT: "bad token"}
        response = client.request(method, path, **kwargs)

        assert response.status_code == 400, (method, path, response.text)
        assert response.json()["detail"] == "invalid_content_kind_header"
        assert _ACCEPT.lower() in response.headers["Vary"].lower()


def test_negotiated_mutations_reject_unaccepted_html_before_missing_if_match(html_client):
    client, _db, _structured, html = html_client

    for method, path, kwargs in _html_mutation_requests(html.id):
        response = client.request(method, path, **kwargs)

        assert response.status_code == 406, (method, path, response.text)
        assert response.json()["detail"] == "content_kind_not_accepted"
        assert _ACCEPT.lower() in response.headers["Vary"].lower()


def test_list_negotiation_filters_before_pagination_and_returns_source_free_unions(
    html_client,
):
    client, _db, _structured, _html = html_client

    legacy = client.get("/api/v1/slides/presentations?limit=1&offset=0")
    html_only = client.get(
        "/api/v1/slides/presentations?limit=1&offset=0",
        headers={_ACCEPT: " standalone_html ,standalone_html "},
    )
    structured_only = client.get(
        "/api/v1/slides/presentations?limit=1&offset=0",
        headers={_ACCEPT: "structured_slides,future_kind"},
    )
    dual = client.get(
        "/api/v1/slides/presentations?limit=1&offset=0",
        headers={_ACCEPT: "structured_slides, future_kind, standalone_html"},
    )

    assert legacy.status_code == structured_only.status_code == html_only.status_code == dual.status_code == 200
    assert legacy.json()["total"] == 1
    assert set(legacy.json()["presentations"][0]) == {
        "id",
        "title",
        "description",
        "theme",
        "created_at",
        "last_modified",
        "deleted",
        "version",
    }
    assert structured_only.json()["presentations"] == legacy.json()["presentations"]
    assert html_only.json()["total"] == 1
    html_summary = html_only.json()["presentations"][0]
    assert html_summary["content_kind"] == "standalone_html"
    assert html_summary["html_slide_count"] == 1
    assert html_summary["html_bytes"] == len(_document().encode("utf-8"))
    assert html_summary["provenance"] == {
        "source_kind": "prompt",
        "provider": "openai",
        "model": "test-model",
    }
    assert "html_document" not in html_summary and "slides" not in html_summary
    assert dual.json()["total"] == 2
    assert len(dual.json()["presentations"]) == 1
    for response in (legacy, structured_only, html_only, dual):
        assert _ACCEPT.lower() in {item.strip().lower() for item in response.headers["Vary"].split(",")}


def test_presentation_metadata_route_is_source_free_without_content_opt_in(html_client):
    client, db, _structured, html = html_client
    statements: list[str] = []
    db.get_connection().set_trace_callback(statements.append)

    response = client.get(f"/api/v1/slides/presentations/{html.id}/metadata")

    assert response.status_code == 200, response.text
    assert response.json() == {
        "id": html.id,
        "title": "HTML Deck",
        "description": None,
        "theme": "black",
        "created_at": html.created_at.replace("+00:00", "Z"),
        "last_modified": html.last_modified.replace("+00:00", "Z"),
        "deleted": False,
        "version": 1,
        "provenance": {
            "source_kind": "prompt",
            "provider": "openai",
            "model": "test-model",
        },
        "content_kind": "standalone_html",
        "html_slide_count": 1,
        "html_bytes": len(_document().encode("utf-8")),
    }
    assert response.headers["Cache-Control"] == "private, no-store"
    assert response.headers["ETag"] == '"v1"'
    assert response.headers["Last-Modified"] == html.last_modified
    assert "authorization" in response.headers["Vary"].lower()
    selected = "\n".join(
        statement for statement in statements if statement.lstrip().upper().startswith("SELECT")
    ).lower()
    assert "html_document" not in selected
    assert "payload_json" not in selected
    assert not any(column in selected for column in (" slides ", " slides,", ".slides "))


def test_structured_version_list_preserves_legacy_title_and_deleted_values(html_client):
    client, db, structured, _html = html_client
    renamed = db.update_presentation(
        presentation_id=structured.id,
        update_fields={"title": "Renamed Structured Deck"},
        expected_version=structured.version,
    )
    deleted = db.soft_delete_presentation(structured.id, renamed.version)

    response = client.get(f"/api/v1/slides/presentations/{structured.id}/versions")

    assert response.status_code == 200, response.text
    assert [(version["version"], version["title"], version["deleted"]) for version in response.json()["versions"]] == [
        (deleted.version, "Renamed Structured Deck", True),
        (renamed.version, "Renamed Structured Deck", False),
        (structured.version, "Structured Deck", False),
    ]


def test_targeted_html_requires_opt_in_before_source_projection(html_client):
    client, db, _structured, html = html_client
    statements: list[str] = []
    db.get_connection().set_trace_callback(statements.append)

    response = client.get(f"/api/v1/slides/presentations/{html.id}")

    assert response.status_code == 406
    assert response.json()["detail"] == "content_kind_not_accepted"
    assert _ACCEPT.lower() in response.headers["Vary"].lower()
    selected = "\n".join(
        statement for statement in statements if statement.lstrip().upper().startswith("SELECT")
    ).lower()
    assert "html_document" not in selected


def test_opted_in_html_detail_is_discriminated_json_with_strong_etag(html_client):
    client, _db, _structured, html = html_client

    response = client.get(f"/api/v1/slides/presentations/{html.id}", headers=_BOTH)

    assert response.status_code == 200
    payload = response.json()
    assert response.headers["content-type"].startswith("application/json")
    assert response.headers["ETag"] == '"v1"'
    assert payload["content_kind"] == "standalone_html"
    assert payload["html_document"] == _document()
    assert payload["html_sha256"] == hashlib.sha256(_document().encode("utf-8")).hexdigest()
    assert payload["html_slide_count"] == 1
    assert "slides" not in payload
    assert _ACCEPT.lower() in response.headers["Vary"].lower()


def test_generic_create_and_mutation_reject_standalone_kind(html_client):
    client, _db, structured, html = html_client

    create = client.post(
        "/api/v1/slides/presentations",
        json={
            "title": "No",
            "content_kind": "standalone_html",
            "html_document": _document(),
            "slides": [],
        },
    )
    html_patch = client.patch(
        f"/api/v1/slides/presentations/{html.id}",
        json={"title": "No"},
        headers={**_BOTH, "If-Match": '"v1"'},
    )
    wrong_accept = client.patch(
        f"/api/v1/slides/presentations/{structured.id}",
        json={"title": "No"},
        headers={_ACCEPT: "standalone_html", "If-Match": 'W/"v1"'},
    )

    assert create.status_code == 409
    assert create.json()["detail"] == "standalone_html_creation_requires_generation"
    _assert_operation_error(
        html_patch,
        operation="update",
        content_kind="standalone_html",
    )
    assert wrong_accept.status_code == 406
    assert wrong_accept.json()["detail"] == "content_kind_not_accepted"


def test_html_source_save_validates_derives_and_noops_with_strong_etag(html_client):
    client, _db, _structured, html = html_client
    validation_pool = client.app.state.standalone_html_validation_pool
    changed_document = _document(title="Renamed", text="New searchable content")

    changed = client.put(
        f"/api/v1/slides/presentations/{html.id}/html-source",
        content=changed_document.encode("utf-8"),
        headers={
            **_BOTH,
            "If-Match": '"v1"',
            "Content-Type": "application/octet-stream",
        },
    )
    same = client.put(
        f"/api/v1/slides/presentations/{html.id}/html-source",
        content=changed_document.encode("utf-8"),
        headers={
            **_BOTH,
            "If-Match": '"v2"',
            "Content-Type": "application/octet-stream",
        },
    )

    assert changed.status_code == 200, changed.text
    assert changed.headers["ETag"] == '"v2"'
    assert changed.json()["title"] == "Renamed"
    assert changed.json()["html_bytes"] == len(changed_document.encode("utf-8"))
    assert same.status_code == 200, same.text
    assert same.headers["ETag"] == '"v2"'
    assert same.json()["version"] == 2
    assert validation_pool.calls == [
        changed_document.encode("utf-8"),
        changed_document.encode("utf-8"),
    ]


def test_html_source_errors_preserve_negotiation_vary(html_client):
    client, _db, _structured, html = html_client
    path = f"/api/v1/slides/presentations/{html.id}/html-source"

    wrong_media = client.put(
        path,
        content=_document(),
        headers={**_BOTH, "If-Match": '"v1"', "Content-Type": "text/html"},
    )
    invalid_source = client.put(
        path,
        content=b"not a complete document",
        headers={
            **_BOTH,
            "If-Match": '"v1"',
            "Content-Type": "application/octet-stream",
        },
    )
    stale = client.put(
        path,
        content=_document(title="Stale candidate"),
        headers={
            **_BOTH,
            "If-Match": '"v0"',
            "Content-Type": "application/octet-stream",
        },
    )

    assert wrong_media.status_code == 415
    assert invalid_source.status_code == 422
    assert stale.status_code == 412
    for response in (wrong_media, invalid_source, stale):
        assert _ACCEPT.lower() in response.headers["Vary"].lower()


def test_html_version_list_and_delete_are_source_free(html_client):
    client, db, _structured, html = html_client
    statements: list[str] = []
    db.get_connection().set_trace_callback(statements.append)

    versions = client.get(f"/api/v1/slides/presentations/{html.id}/versions", headers=_BOTH)
    deleted = client.delete(
        f"/api/v1/slides/presentations/{html.id}",
        headers={**_BOTH, "If-Match": '"v1"'},
    )

    assert versions.status_code == 200, versions.text
    assert versions.json()["total"] == 1
    assert "html_document" not in json.dumps(versions.json())
    assert deleted.status_code == 200, deleted.text
    assert set(deleted.json()) == {"id", "content_kind", "deleted_at"}
    assert deleted.json()["content_kind"] == "standalone_html"
    selected = "\n".join(
        statement for statement in statements if statement.lstrip().upper().startswith("SELECT")
    ).lower()
    assert "payload_json" not in selected
    assert "html_document" not in selected


def test_html_reveal_and_render_reject_before_source_or_dispatch(html_client):
    client, db, _structured, html = html_client
    statements: list[str] = []
    db.get_connection().set_trace_callback(statements.append)

    export = client.get(
        f"/api/v1/slides/presentations/{html.id}/export?format=revealjs",
        headers=_BOTH,
    )
    render = client.post(
        f"/api/v1/slides/presentations/{html.id}/render-jobs",
        json={"format": "mp4"},
        headers={**_BOTH, "If-Match": '"v1"'},
    )

    _assert_operation_error(
        export,
        operation="export",
        content_kind="standalone_html",
    )
    _assert_operation_error(
        render,
        operation="render",
        content_kind="standalone_html",
    )
    selected = "\n".join(
        statement for statement in statements if statement.lstrip().upper().startswith("SELECT")
    ).lower()
    assert "html_document" not in selected


def test_html_is_an_explicit_export_format_but_transport_is_deferred():
    assert ExportFormat.HTML.value == "html"


def test_search_negotiation_filters_before_count_and_preserves_legacy_shape(
    html_client,
):
    client, _db, _structured, _html = html_client

    legacy = client.get("/api/v1/slides/presentations/search?q=Deck&limit=1")
    html_only = client.get(
        "/api/v1/slides/presentations/search?q=Deck&limit=1",
        headers={_ACCEPT: "standalone_html"},
    )
    dual = client.get(
        "/api/v1/slides/presentations/search?q=Deck&limit=1",
        headers=_BOTH,
    )

    assert legacy.status_code == html_only.status_code == dual.status_code == 200
    assert legacy.json()["total"] == 1
    assert "content_kind" not in legacy.json()["presentations"][0]
    assert html_only.json()["total"] == 1
    assert html_only.json()["presentations"][0]["content_kind"] == "standalone_html"
    assert dual.json()["total"] == 2
    assert len(dual.json()["presentations"]) == 1
    for response in (legacy, html_only, dual):
        assert _ACCEPT.lower() in response.headers["Vary"].lower()


def test_html_render_artifacts_rejects_before_collection_dispatch(html_client):
    client, _db, _structured, html = html_client

    response = client.get(
        f"/api/v1/slides/presentations/{html.id}/render-artifacts",
        headers=_BOTH,
    )

    _assert_operation_error(
        response,
        operation="render",
        content_kind="standalone_html",
    )
    assert _Collections.list_calls == 0


def test_explicit_null_standalone_fields_and_kind_are_rejected_by_presence(
    html_client,
):
    client, _db, structured, _html = html_client

    create = client.post(
        "/api/v1/slides/presentations",
        json={"title": "No", "slides": [], "html_document": None},
    )
    null_source = client.patch(
        f"/api/v1/slides/presentations/{structured.id}",
        json={"html_document": None},
        headers={**_BOTH, "If-Match": 'W/"v1"'},
    )
    null_kind = client.patch(
        f"/api/v1/slides/presentations/{structured.id}",
        json={"content_kind": None},
        headers={**_BOTH, "If-Match": 'W/"v1"'},
    )

    _assert_operation_error(
        create,
        operation="create",
        content_kind="structured_slides",
    )
    _assert_operation_error(
        null_source,
        operation="update",
        content_kind="structured_slides",
    )
    assert null_kind.status_code == 409
    assert null_kind.json()["detail"] == "content_kind_immutable"


@pytest.mark.parametrize(
    ("method", "payload"),
    [
        ("PUT", {"title": "No", "content_kind": "structured_slides"}),
        ("PATCH", {"content_kind": "structured_slides"}),
    ],
)
def test_html_kind_change_has_stable_immutable_error(html_client, method, payload):
    client, _db, _structured, html = html_client

    response = client.request(
        method,
        f"/api/v1/slides/presentations/{html.id}",
        json=payload,
        headers={**_BOTH, "If-Match": '"v1"'},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "content_kind_immutable"


def test_json_export_is_explicit_and_discriminated_for_opted_in_kinds(html_client):
    client, _db, structured, html = html_client
    validation_pool = client.app.state.standalone_html_validation_pool

    html_export = client.get(
        f"/api/v1/slides/presentations/{html.id}/export?format=json",
        headers=_BOTH,
    )
    structured_export = client.get(
        f"/api/v1/slides/presentations/{structured.id}/export?format=json",
        headers=_BOTH,
    )

    assert html_export.status_code == 200, html_export.text
    assert html_export.headers["content-type"].startswith("application/json")
    assert html_export.json()["content_kind"] == "standalone_html"
    assert html_export.json()["html_document"] == _document()
    assert "slides" not in html_export.json()
    assert structured_export.status_code == 200, structured_export.text
    assert structured_export.json()["content_kind"] == "structured_slides"
    assert "slides" in structured_export.json()
    assert validation_pool.calls == [_document()]


def test_json_export_rejects_corrupt_stored_derived_metadata_after_pool_validation(html_client):
    client, db, _structured, html = html_client
    with db.transaction(immediate=True) as conn:
        conn.execute(
            "UPDATE presentations SET slides_text = ? WHERE id = ?",
            ("forged", html.id),
        )

    response = client.get(
        f"/api/v1/slides/presentations/{html.id}/export?format=json",
        headers=_BOTH,
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "standalone_html_response_invalid"
    assert _ACCEPT.lower() in response.headers["Vary"].lower()
    assert client.app.state.standalone_html_validation_pool.calls == [_document()]


@pytest.mark.parametrize(
    ("column", "corrupt_value"),
    [
        pytest.param("slides", "not-json", id="slides-invalid-json"),
        pytest.param("slides", "{}", id="slides-nonlist"),
        pytest.param("slides", '[{"title":"forged"}]', id="slides-nonempty"),
        pytest.param("slides", "[" + " " * 8192 + "]", id="slides-oversize-empty"),
        pytest.param("generation_job_uuid", None, id="job-uuid-missing"),
        pytest.param("generation_job_uuid", "   ", id="job-uuid-blank"),
        pytest.param("generation_provenance_json", None, id="provenance-missing"),
        pytest.param(
            "generation_provenance_json",
            '{"private":"SECRET-MALFORMED-PROVENANCE"',
            id="provenance-malformed",
        ),
        pytest.param("generation_provenance_json", "[]", id="provenance-nonobject"),
        pytest.param("generation_provenance_json", "{}", id="provenance-empty-object"),
        pytest.param(
            "generation_provenance_json",
            "[" * 1100 + "0" + "]" * 1100,
            id="provenance-recursive",
        ),
        pytest.param(
            "generation_provenance_json",
            json.dumps({"private": "SECRET-OVERSIZE-PROVENANCE" + "x" * 4096}),
            id="provenance-oversize",
        ),
    ],
)
def test_json_export_rejects_corrupt_complete_row_invariant_before_pool_validation(
    html_client,
    column,
    corrupt_value,
):
    client, db, _structured, html = html_client
    with db.transaction(immediate=True) as conn:
        conn.execute(f"UPDATE presentations SET {column} = ? WHERE id = ?", (corrupt_value, html.id))
    validation_pool = client.app.state.standalone_html_validation_pool
    validation_pool.calls.clear()

    response = client.get(
        f"/api/v1/slides/presentations/{html.id}/export?format=json",
        headers=_BOTH,
    )

    assert response.status_code == 500
    assert response.json() == {"detail": "standalone_html_response_invalid"}
    assert "SECRET-" not in response.text
    assert validation_pool.calls == []


def test_negotiated_downstream_http_error_keeps_vary(html_client):
    client, _db, structured, _html = html_client

    response = client.patch(
        f"/api/v1/slides/presentations/{structured.id}",
        json={},
        headers={"If-Match": 'W/"v1"'},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "no_fields_to_update"
    assert _ACCEPT.lower() in response.headers["Vary"].lower()


def test_html_version_restore_uses_interactive_pool_before_atomic_write(html_client):
    client, _db, _structured, html = html_client
    changed_document = _document(title="Changed", text="second")
    changed = client.put(
        f"/api/v1/slides/presentations/{html.id}/html-source",
        content=changed_document.encode("utf-8"),
        headers={
            **_BOTH,
            "If-Match": '"v1"',
            "Content-Type": "application/octet-stream",
        },
    )
    assert changed.status_code == 200, changed.text
    validation_pool = client.app.state.standalone_html_validation_pool
    validation_pool.calls.clear()

    restored = client.post(
        f"/api/v1/slides/presentations/{html.id}/versions/1/restore",
        headers={**_BOTH, "If-Match": '"v2"'},
    )

    assert restored.status_code == 200, restored.text
    assert restored.json()["html_document"] == _document()
    assert validation_pool.calls == [_document()]


def test_html_soft_restore_revalidates_cross_field_consistent_stored_source(html_client):
    client, db, _structured, html = html_client
    deleted = client.delete(
        f"/api/v1/slides/presentations/{html.id}",
        headers={**_BOTH, "If-Match": '"v1"'},
    )
    assert deleted.status_code == 200, deleted.text
    corrupt = _document().replace(
        "document.addEventListener('keydown', () => {});",
        "fetch('/private-data');",
    )
    encoded = corrupt.encode("utf-8")
    with db.transaction(immediate=True) as conn:
        conn.execute(
            """
            UPDATE presentations
            SET html_document = ?, html_sha256 = ?, html_bytes = ?
            WHERE id = ?
            """,
            (corrupt, hashlib.sha256(encoded).hexdigest(), len(encoded), html.id),
        )
    validation_pool = client.app.state.standalone_html_validation_pool
    validation_pool.calls.clear()

    restored = client.post(
        f"/api/v1/slides/presentations/{html.id}/restore",
        headers={**_BOTH, "If-Match": '"v2"'},
    )

    assert restored.status_code == 422, restored.text
    assert validation_pool.calls == [corrupt]
    current = db.get_presentation_by_id(html.id, include_deleted=True)
    assert current.deleted == 1
    assert current.version == 2


def test_html_soft_restore_source_response_sets_private_no_store_headers(html_client):
    client, _db, _structured, html = html_client
    deleted = client.delete(
        f"/api/v1/slides/presentations/{html.id}",
        headers={**_BOTH, "If-Match": '"v1"'},
    )
    assert deleted.status_code == 200, deleted.text

    restored = client.post(
        f"/api/v1/slides/presentations/{html.id}/restore",
        headers={**_BOTH, "If-Match": '"v2"'},
    )

    assert restored.status_code == 200, restored.text
    assert restored.json()["content_kind"] == "standalone_html"
    assert restored.json()["html_document"] == _document()
    assert restored.headers["Cache-Control"] == "private, no-store"
    assert restored.headers["X-Content-Type-Options"] == "nosniff"


def test_structured_soft_restore_keeps_legacy_path_without_pool_validation(html_client):
    client, _db, structured, _html = html_client
    validation_pool = client.app.state.standalone_html_validation_pool
    validation_pool.calls.clear()
    deleted = client.delete(
        f"/api/v1/slides/presentations/{structured.id}",
        headers={"If-Match": 'W/"v1"'},
    )
    assert deleted.status_code == 200, deleted.text

    restored = client.post(
        f"/api/v1/slides/presentations/{structured.id}/restore",
        headers={"If-Match": 'W/"v2"'},
    )

    assert restored.status_code == 200, restored.text
    assert restored.json()["deleted"] is False
    assert "Cache-Control" not in restored.headers
    assert "X-Content-Type-Options" not in restored.headers
    assert validation_pool.calls == []


def test_endpoint_snapshot_decoder_retains_no_source_exception_context():
    sentinel = "SECRET-ENDPOINT-SNAPSHOT"

    with pytest.raises(HTTPException) as exc_info:
        _load_version_payload('{"html_document":"' + sentinel)

    chain = [exc_info.value]
    while chain[-1].__cause__ is not None or chain[-1].__context__ is not None:
        chain.append(chain[-1].__cause__ or chain[-1].__context__)
    assert not any(isinstance(exc, json.JSONDecodeError) for exc in chain)
    assert sentinel not in " ".join(repr(exc) for exc in chain)


@pytest.mark.parametrize(
    ("method", "suffix", "extra_headers"),
    [
        pytest.param("GET", "", {}, id="get"),
        pytest.param("POST", "/restore", {"If-Match": '"v1"'}, id="restore"),
    ],
)
def test_recursive_snapshot_matches_fixed_malformed_payload_mapping(
    html_client,
    method,
    suffix,
    extra_headers,
):
    client, db, _structured, html = html_client
    path = f"/api/v1/slides/presentations/{html.id}/versions/1{suffix}"

    def _replace_snapshot(payload_json: str) -> None:
        with db.transaction(immediate=True) as conn:
            conn.execute(
                "UPDATE presentations_versions SET payload_json = ? " "WHERE presentation_id = ? AND version = 1",
                (payload_json, html.id),
            )

    _replace_snapshot('{"html_document":"malformed')
    baseline = client.request(method, path, headers={**_BOTH, **extra_headers})

    sentinel = "SECRET-RECURSIVE-SNAPSHOT"
    recursive = '{"html_document":"' + sentinel + '","nested":' + "[" * 1100 + "0" + "]" * 1100 + "}"
    assert len(recursive.encode("utf-8")) < 4096
    _replace_snapshot(recursive)

    response = client.request(method, path, headers={**_BOTH, **extra_headers})

    assert baseline.json() == response.json() == {"detail": "version_payload_invalid"}
    assert response.status_code == baseline.status_code
    assert sentinel not in response.text


def test_structured_restore_recomputes_legacy_slide_text_with_image_alt(html_client):
    client, db, _structured, _html = html_client
    created = client.post(
        "/api/v1/slides/presentations",
        json={
            "title": "Legacy image deck",
            "slides": [
                {
                    "order": 0,
                    "layout": "content",
                    "title": "Image slide",
                    "content": "Body",
                    "speaker_notes": "Narration",
                    "metadata": {"images": [{"asset_ref": "output:123", "alt": "Restored cover"}]},
                }
            ],
        },
    )
    assert created.status_code == 201, created.text
    presentation_id = created.json()["id"]

    with db.transaction(immediate=True) as conn:
        version_row = conn.execute(
            """
            SELECT payload_json FROM presentations_versions
            WHERE presentation_id = ? AND version = 1
            """,
            (presentation_id,),
        ).fetchone()
        payload = json.loads(version_row["payload_json"])
        payload.pop("slides_text", None)
        conn.execute(
            """
            UPDATE presentations_versions SET payload_json = ?
            WHERE presentation_id = ? AND version = 1
            """,
            (json.dumps(payload), presentation_id),
        )

    updated = client.patch(
        f"/api/v1/slides/presentations/{presentation_id}",
        json={"title": "Changed"},
        headers={"If-Match": created.headers["ETag"]},
    )
    assert updated.status_code == 200, updated.text

    restored = client.post(
        f"/api/v1/slides/presentations/{presentation_id}/versions/1/restore",
        headers={"If-Match": updated.headers["ETag"]},
    )

    assert restored.status_code == 200, restored.text
    assert "Restored cover" in db.get_presentation_by_id(presentation_id).slides_text


def test_structured_restore_keeps_legacy_missing_version_precedence(html_client):
    client, _db, structured, _html = html_client

    response = client.post(
        f"/api/v1/slides/presentations/{structured.id}/versions/999/restore",
        headers={"If-Match": 'W/"v0"'},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "presentation_version_not_found"


def test_standalone_capabilities_exact_enabled_shape_and_private_cache(html_client):
    client, _db, _structured, _html = html_client

    response = client.get("/api/v1/slides/capabilities")

    assert response.status_code == 200, response.text
    assert response.json() == {
        "schema_version": 1,
        "content_kind_request_header": "X-Slides-Accept-Content-Kinds",
        "content_kinds": {
            "structured_slides": {"read": True, "edit": True},
            "standalone_html": {
                "read": True,
                "edit": True,
                "export_attachment": True,
                "draft_attachment": True,
                "reason": None,
                "limits": {
                    "max_document_bytes": 1_048_576,
                    "max_source_write_bytes": 1_048_576,
                    "max_draft_attachment_bytes": 1_048_576,
                    "max_slides": 30,
                    "max_nesting_depth": 128,
                },
            },
        },
        "generation_modes": {
            "structured_slides": {
                "enabled": True,
                "transport": "existing_source_endpoints",
            },
            "standalone_html": {
                "enabled": True,
                "reason": None,
                "transport": "slides_generation_job",
                "source_kinds": ["prompt", "chat", "media", "notes", "rag"],
                "provider": "openai",
                "model": "allowed-model",
                "adapter_id": "openai_official_chat_v1",
                "endpoint_identity": "https://api.openai.com:443/v1/chat/completions",
                "generation_config_revision": _REVISION,
                "input_limits": {
                    "max_request_bytes": 4_194_304,
                    "max_source_chars": 200_000,
                    "max_source_tokens": 50_000,
                    "max_audience_chars": 500,
                    "max_source_identifier_bytes": 256,
                    "max_note_ids": 100,
                    "max_rag_query_chars": 20_000,
                    "max_rag_top_k": 100,
                },
                "output_limits": {
                    "max_provider_response_bytes": 8_388_608,
                    "max_document_bytes": 1_048_576,
                },
            },
        },
    }
    assert response.headers["Cache-Control"] == "private, no-store"
    vary = response.headers["Vary"].lower()
    assert all(token in vary for token in ("authorization", "x-api-key", "cookie"))


def test_validator_unavailable_capability_keeps_read_and_draft_only(html_client):
    client, _db, _structured, _html = html_client
    runtime = client.app.state.standalone_html_api_runtime
    runtime.config = _runtime_config(enabled=False, reason="validator_unavailable")
    runtime.validator_available = False

    response = client.get("/api/v1/slides/capabilities")

    assert response.status_code == 200, response.text
    html_kind = response.json()["content_kinds"]["standalone_html"]
    generation = response.json()["generation_modes"]["standalone_html"]
    assert html_kind == {
        "read": True,
        "edit": False,
        "export_attachment": False,
        "draft_attachment": True,
        "reason": "validator_unavailable",
        "limits": html_kind["limits"],
    }
    assert generation["enabled"] is False
    assert generation["reason"] == "validator_unavailable"
    assert generation["generation_config_revision"] is None
    assert all(generation[field] is None for field in ("provider", "model", "adapter_id", "endpoint_identity"))


def test_validator_unavailable_is_independent_from_generation_disabled_reason(html_client):
    client, _db, _structured, _html = html_client
    runtime = client.app.state.standalone_html_api_runtime
    runtime.config = _runtime_config(enabled=False, reason="feature_disabled")
    runtime.validator_available = False

    response = client.get("/api/v1/slides/capabilities")

    assert response.status_code == 200, response.text
    html_kind = response.json()["content_kinds"]["standalone_html"]
    generation = response.json()["generation_modes"]["standalone_html"]
    assert html_kind["edit"] is False
    assert html_kind["export_attachment"] is False
    assert html_kind["reason"] == "validator_unavailable"
    assert generation["enabled"] is False
    assert generation["reason"] == "feature_disabled"


@pytest.mark.parametrize(
    "reason",
    [
        "feature_disabled",
        "egress_disabled",
        "default_model_not_configured",
        "default_model_not_allowed",
        "default_endpoint_not_allowed",
        "prompt_asset_unavailable",
        "digest_key_unavailable",
        "generation_worker_unavailable",
        "generation_reconciler_overloaded",
        "validator_unavailable",
    ],
)
def test_capabilities_project_each_approved_disabled_reason_safely(html_client, reason):
    client, _db, _structured, _html = html_client
    runtime = client.app.state.standalone_html_api_runtime
    runtime.config = _runtime_config(enabled=False, reason=reason)
    runtime.validator_available = reason != "validator_unavailable"

    response = client.get("/api/v1/slides/capabilities")

    assert response.status_code == 200, response.text
    html_kind = response.json()["content_kinds"]["standalone_html"]
    generation = response.json()["generation_modes"]["standalone_html"]
    assert generation["enabled"] is False
    assert generation["reason"] == reason
    assert generation["generation_config_revision"] is None
    assert all(generation[field] is None for field in ("provider", "model", "adapter_id", "endpoint_identity"))
    assert html_kind["reason"] == ("validator_unavailable" if reason == "validator_unavailable" else None)
    assert html_kind["edit"] is (reason != "validator_unavailable")
    assert html_kind["export_attachment"] is (reason != "validator_unavailable")


@pytest.mark.parametrize(
    ("section", "reason"),
    [
        pytest.param("content", "feature_disabled", id="content-reason"),
        pytest.param("generation", "not_an_approved_reason", id="generation-reason"),
    ],
)
def test_capability_schemas_reject_unapproved_reason_literals(section, reason):
    payload = slides_standalone_html._capability_payload(
        _runtime_config(enabled=False, reason="feature_disabled"),
        validator_available=True,
    )
    if section == "content":
        payload["content_kinds"]["standalone_html"]["reason"] = reason
    else:
        payload["generation_modes"]["standalone_html"]["reason"] = reason

    with pytest.raises(ValidationError):
        SlidesCapabilitiesResponse.model_validate(payload)


@pytest.mark.parametrize(
    "source",
    [
        pytest.param({"kind": "prompt", "prompt": "Material"}, id="prompt"),
        pytest.param({"kind": "chat", "conversation_id": "chat-1"}, id="chat"),
        pytest.param({"kind": "media", "media_id": 1}, id="media"),
        pytest.param({"kind": "notes", "note_ids": ["note-1"]}, id="notes"),
        pytest.param({"kind": "rag", "query": "bounded query", "top_k": 8}, id="rag"),
    ],
)
def test_generation_accepts_each_closed_source_variant(html_client, source):
    client, _db, _structured, _html = html_client
    payload = {**_GENERATION_REQUEST, "source": source}

    response = client.post(
        "/api/v1/slides/generations",
        json=payload,
        headers={"Idempotency-Key": "task11-generation-key"},
    )

    assert response.status_code == 202, response.text
    assert response.json() == {
        "generation_id": "018f2f4a-6f79-7a27-a1aa-7bb60777d9f1",
        "status": "queued",
        "status_url": "/api/v1/slides/generations/018f2f4a-6f79-7a27-a1aa-7bb60777d9f1",
        "presentation_id": None,
    }
    call = client.app.state.standalone_html_api_runtime.generation_service.submit_calls[-1]
    assert call["owner_user_id"] == "1"
    assert call["request"] == payload


def test_generation_replay_does_not_acquire_unrelated_source_databases(html_client):
    client, _db, _structured, _html = html_client

    response = client.post(
        "/api/v1/slides/generations",
        json=_GENERATION_REQUEST,
        headers={"Idempotency-Key": "task11-generation-replay"},
    )

    assert response.status_code == 202, response.text
    assert (_SourceDbCalls.media, _SourceDbCalls.chacha) == (0, 0)


@pytest.mark.parametrize(
    ("source", "expected_calls"),
    [
        pytest.param({"kind": "prompt", "prompt": "Material"}, (0, 0), id="prompt"),
        pytest.param({"kind": "chat", "conversation_id": "chat-1"}, (0, 1), id="chat"),
        pytest.param({"kind": "media", "media_id": 1}, (1, 0), id="media"),
        pytest.param({"kind": "notes", "note_ids": ["note-1"]}, (0, 1), id="notes"),
        pytest.param({"kind": "rag", "query": "bounded query", "top_k": 8}, (1, 1), id="rag"),
    ],
)
def test_new_generation_acquires_only_selected_source_databases(
    html_client,
    monkeypatch,
    source,
    expected_calls,
):
    client, _db, _structured, _html = html_client
    runtime = client.app.state.standalone_html_api_runtime
    service = runtime.generation_service

    async def _resolve_submit(**kwargs):
        await kwargs["source_resolver"](kwargs["request"]["source"], SimpleNamespace())
        return service.submission

    async def _resolved_source(
        _source,
        *,
        owner_user_id,
        limits,
        media_db,
        chacha_db,
    ):
        del owner_user_id, limits, media_db, chacha_db
        return SimpleNamespace()

    monkeypatch.setattr(service, "submit", _resolve_submit)
    monkeypatch.setattr(slides_standalone_html, "resolve_standalone_html_source", _resolved_source)

    response = client.post(
        "/api/v1/slides/generations",
        json={**_GENERATION_REQUEST, "source": source},
        headers={"Idempotency-Key": "task11-generation-new"},
    )

    assert response.status_code == 202, response.text
    assert (_SourceDbCalls.media, _SourceDbCalls.chacha) == expected_calls


@pytest.mark.parametrize(
    "headers",
    [
        pytest.param({}, id="missing"),
        pytest.param({"Idempotency-Key": "short"}, id="short"),
        pytest.param({"Idempotency-Key": "contains spaces 123"}, id="invalid-character"),
    ],
)
def test_generation_requires_one_valid_idempotency_header(html_client, headers):
    client, _db, _structured, _html = html_client

    response = client.post("/api/v1/slides/generations", json=_GENERATION_REQUEST, headers=headers)

    assert response.status_code == 400
    assert response.json()["detail"] in {
        "generation_idempotency_key_required",
        "generation_idempotency_key_invalid",
    }
    assert client.app.state.standalone_html_api_runtime.generation_service.submit_calls == []


def test_generation_closed_schema_rejects_provider_override_before_service(html_client):
    client, _db, _structured, _html = html_client
    payload = {**_GENERATION_REQUEST, "provider": "attacker-provider"}

    response = client.post(
        "/api/v1/slides/generations",
        json=payload,
        headers={"Idempotency-Key": "task11-generation-key"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "standalone_html_request_invalid"
    assert "attacker-provider" not in response.text
    assert client.app.state.standalone_html_api_runtime.generation_service.submit_calls == []


def test_generation_maps_retryable_service_error_without_echoing_key(html_client):
    client, _db, _structured, _html = html_client
    runtime = client.app.state.standalone_html_api_runtime
    runtime.generation_service.submit_error = StandaloneHtmlGenerationError(
        "generation_receipt_unresolved",
        status_code=503,
        retry_after=1,
    )

    response = client.post(
        "/api/v1/slides/generations",
        json=_GENERATION_REQUEST,
        headers={"Idempotency-Key": "task11-secret-key"},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "generation_receipt_unresolved"}
    assert response.headers["Retry-After"] == "1"
    assert "task11-secret-key" not in response.text


def test_terminal_generation_replay_returns_closed_200_variant(html_client):
    client, _db, _structured, _html = html_client
    service = client.app.state.standalone_html_api_runtime.generation_service
    service.submission = StandaloneHtmlGenerationSubmission(
        receipt_id="018f2f4a-6f79-7a27-a1aa-7bb60777d9f1",
        status="completed",
        job_uuid="018f2f4a-6f79-7a27-a1aa-7bb60777d9f2",
        presentation_id="presentation-1",
        replayed=True,
    )

    response = client.post(
        "/api/v1/slides/generations",
        json=_GENERATION_REQUEST,
        headers={"Idempotency-Key": "task11-generation-key"},
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "generation_id": "018f2f4a-6f79-7a27-a1aa-7bb60777d9f1",
        "status": "completed",
        "status_url": "/api/v1/slides/generations/018f2f4a-6f79-7a27-a1aa-7bb60777d9f1",
        "presentation_id": "presentation-1",
        "content_kind": "standalone_html",
    }


@pytest.mark.parametrize("generation_id", ["not-a-uuid", "018f2f4a-6f79-7a27-a1aa-7bb60777d900"])
def test_generation_status_malformed_and_unknown_are_equivalent_404(html_client, generation_id):
    client, _db, _structured, _html = html_client

    response = client.get(f"/api/v1/slides/generations/{generation_id}")

    assert response.status_code == 404
    assert response.json() == {"detail": "generation_not_found"}


def test_generation_status_unknown_remains_404_without_lifecycle_context(html_client):
    client, _db, _structured, _html = html_client
    del client.app.state.standalone_html_api_runtime

    response = client.get("/api/v1/slides/generations/018f2f4a-6f79-7a27-a1aa-7bb60777d900")

    assert response.status_code == 404
    assert response.json() == {"detail": "generation_not_found"}


def test_generation_status_reconciles_and_returns_bounded_progress_without_html(html_client):
    client, _db, _structured, _html = html_client
    runtime = client.app.state.standalone_html_api_runtime
    generation_id = runtime.generation_service.submission.receipt_id

    response = client.get(f"/api/v1/slides/generations/{generation_id}")

    assert response.status_code == 200, response.text
    assert response.json() == {
        "generation_id": generation_id,
        "status": "queued",
        "status_url": f"/api/v1/slides/generations/{generation_id}",
        "presentation_id": None,
        "progress_text": "Resolving source",
    }
    assert runtime.reconcile_calls == ["1"]
    assert "html" not in json.dumps(response.json()).lower()


def test_generation_status_unresolved_claim_is_retryable_and_source_free(html_client):
    client, _db, _structured, _html = html_client
    runtime = client.app.state.standalone_html_api_runtime
    runtime.generation_service.submission = StandaloneHtmlGenerationSubmission(
        receipt_id="018f2f4a-6f79-7a27-a1aa-7bb60777d9f1",
        status="claimed",
        job_uuid=None,
        presentation_id=None,
        replayed=True,
    )

    response = client.get(f"/api/v1/slides/generations/{runtime.generation_service.submission.receipt_id}")

    assert response.status_code == 503
    assert response.json() == {"detail": "generation_receipt_unresolved"}
    assert response.headers["Retry-After"] == "1"
    assert "html" not in response.text.lower()


def test_generation_status_jobs_failure_is_retryable_and_redacted(html_client):
    client, _db, _structured, _html = html_client
    runtime = client.app.state.standalone_html_api_runtime

    def unavailable(_owner_user_id: str):
        raise RuntimeError("SECRET-JOBS-OUTAGE")

    runtime.reconcile_owner = unavailable

    response = client.get(f"/api/v1/slides/generations/{runtime.generation_service.submission.receipt_id}")

    assert response.status_code == 503
    assert response.json() == {"detail": "generation_receipt_unresolved"}
    assert response.headers["Retry-After"] == "1"
    assert "SECRET-JOBS-OUTAGE" not in response.text


def test_generation_status_other_owner_is_indistinguishable_from_unknown(html_client):
    client, _db, _structured, _html = html_client
    generation_id = client.app.state.standalone_html_api_runtime.generation_service.submission.receipt_id

    async def _other_owner():
        return User(
            id=2,
            username="other-owner",
            email=None,
            is_active=True,
            is_admin=False,
        )

    client.app.dependency_overrides[get_request_user] = _other_owner
    response = client.get(f"/api/v1/slides/generations/{generation_id}")

    assert response.status_code == 404
    assert response.json() == {"detail": "generation_not_found"}


@pytest.mark.parametrize(
    ("status", "receipt_error", "expected"),
    [
        pytest.param(
            "completed",
            (None, None),
            {
                "presentation_id": "presentation-1",
                "content_kind": "standalone_html",
            },
            id="completed",
        ),
        pytest.param(
            "failed",
            ("provider_timeout", "The provider timed out."),
            {
                "presentation_id": None,
                "error_code": "provider_timeout",
                "error_message": "The provider timed out.",
            },
            id="failed",
        ),
        pytest.param(
            "failed",
            ("generation_quarantined", "Generation was quarantined."),
            {
                "presentation_id": None,
                "error_code": "generation_quarantined",
                "error_message": "Generation was quarantined.",
            },
            id="quarantined",
        ),
        pytest.param(
            "cancelled",
            (None, None),
            {
                "presentation_id": None,
                "error_code": "generation_cancelled",
            },
            id="cancelled",
        ),
    ],
)
def test_generation_status_returns_each_closed_terminal_variant(
    html_client,
    status,
    receipt_error,
    expected,
):
    client, _db, _structured, _html = html_client
    runtime = client.app.state.standalone_html_api_runtime
    generation_id = runtime.generation_service.submission.receipt_id
    runtime.generation_service.submission = StandaloneHtmlGenerationSubmission(
        receipt_id=generation_id,
        status=status,
        job_uuid=None,
        presentation_id="presentation-1" if status == "completed" else None,
        replayed=True,
    )
    runtime.receipt_error = receipt_error

    response = client.get(f"/api/v1/slides/generations/{generation_id}")

    assert response.status_code == 200, response.text
    assert response.json() == {
        "generation_id": generation_id,
        "status": status,
        "status_url": f"/api/v1/slides/generations/{generation_id}",
        **expected,
    }


def test_generation_status_bounds_terminal_error_fields(html_client):
    client, _db, _structured, _html = html_client
    runtime = client.app.state.standalone_html_api_runtime
    generation_id = runtime.generation_service.submission.receipt_id
    runtime.generation_service.submission = StandaloneHtmlGenerationSubmission(
        receipt_id=generation_id,
        status="failed",
        job_uuid=None,
        presentation_id=None,
        replayed=True,
    )
    runtime.receipt_error = ("Unsafe Secret Value", "X" * 257)

    response = client.get(f"/api/v1/slides/generations/{generation_id}")

    assert response.status_code == 200, response.text
    assert response.json()["error_code"] == "generation_failed"
    assert response.json()["error_message"] == "Generation failed."
    assert "Unsafe Secret Value" not in response.text
    assert "X" * 257 not in response.text


def test_generation_status_omits_unbounded_progress_text(html_client):
    client, _db, _structured, _html = html_client
    runtime = client.app.state.standalone_html_api_runtime
    generation_id = runtime.generation_service.submission.receipt_id
    runtime.job = {"progress_message": "X" * 257, "progress_percent": 25.0}

    response = client.get(f"/api/v1/slides/generations/{generation_id}")

    assert response.status_code == 200, response.text
    assert "progress_text" not in response.json()


def _assert_html_attachment_headers(response) -> None:
    assert response.headers["Content-Type"] == "application/octet-stream"
    assert response.headers["Content-Disposition"] == 'attachment; filename="presentation.html"'
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Download-Options"] == "noopen"
    assert response.headers["Cache-Control"] == "private, no-store"
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Cross-Origin-Resource-Policy"] == "same-origin"


def test_draft_attachment_echoes_exact_invalid_deck_without_validation_or_persistence(html_client):
    client, db, _structured, html = html_client
    draft = b"not a valid deck, but exact UTF-8 recovery bytes \xe2\x98\x83"
    validation_pool = client.app.state.standalone_html_validation_pool
    validation_pool.calls.clear()

    response = client.post(
        f"/api/v1/slides/presentations/{html.id}/draft-attachment",
        content=draft,
        headers={**_BOTH, "Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 200, response.text
    assert response.content == draft
    _assert_html_attachment_headers(response)
    assert db.get_presentation_by_id(html.id).version == 1
    assert validation_pool.calls == []


def test_draft_attachment_rejects_structured_kind_with_flat_operation_error(html_client):
    client, _db, structured, _html = html_client

    response = client.post(
        f"/api/v1/slides/presentations/{structured.id}/draft-attachment",
        content=b"not read for a structured target",
        headers={**_BOTH, "Content-Type": "application/octet-stream"},
    )

    _assert_operation_error(
        response,
        operation="draft_attachment",
        content_kind="structured_slides",
    )


def test_saved_html_and_json_exports_use_fixed_names_and_security_headers(html_client):
    client, _db, _structured, html = html_client

    html_export = client.get(
        f"/api/v1/slides/presentations/{html.id}/export?format=html",
        headers=_BOTH,
    )
    json_export = client.get(
        f"/api/v1/slides/presentations/{html.id}/export?format=json",
        headers=_BOTH,
    )

    assert html_export.status_code == 200, html_export.text
    assert html_export.content == _document().encode("utf-8")
    _assert_html_attachment_headers(html_export)
    assert json_export.status_code == 200, json_export.text
    assert json_export.headers["Content-Type"] == "application/json"
    assert json_export.headers["Content-Disposition"] == 'attachment; filename="presentation.json"'
    assert json_export.headers["X-Content-Type-Options"] == "nosniff"
    assert json_export.headers["Cache-Control"] == "private, no-store"
    assert json_export.headers["Referrer-Policy"] == "no-referrer"
    assert json_export.headers["Cross-Origin-Resource-Policy"] == "same-origin"


def test_html_version_content_has_strong_etag_last_modified_and_source_headers(html_client):
    client, _db, _structured, html = html_client

    response = client.get(
        f"/api/v1/slides/presentations/{html.id}/versions/1",
        headers=_BOTH,
    )

    assert response.status_code == 200, response.text
    assert response.headers["Content-Type"].startswith("application/json")
    assert response.headers["ETag"] == '"v1"'
    assert response.headers["Last-Modified"]
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["Cache-Control"] == "private, no-store"
    vary = response.headers["Vary"].lower()
    assert all(token in vary for token in ("x-slides-accept-content-kinds", "authorization", "x-api-key", "cookie"))
    assert response.json()["html_document"] == _document()


def test_lost_save_response_reconciles_same_source_but_stale_different_source_is_bounded(html_client):
    client, _db, _structured, html = html_client
    path = f"/api/v1/slides/presentations/{html.id}/html-source"
    changed_document = _document(title="Changed after response loss", text="new")
    headers = {**_BOTH, "If-Match": '"v1"', "Content-Type": "application/octet-stream"}
    changed = client.put(path, content=changed_document.encode(), headers=headers)
    assert changed.status_code == 200, changed.text
    validation_pool = client.app.state.standalone_html_validation_pool
    validation_pool.calls.clear()

    reconciled = client.put(path, content=changed_document.encode(), headers=headers)
    stale = client.put(path, content=_document(title="Other").encode(), headers=headers)

    assert reconciled.status_code == 200, reconciled.text
    assert reconciled.headers["ETag"] == '"v2"'
    assert reconciled.json()["version"] == 2
    assert stale.status_code == 412
    assert stale.json() == {
        "detail": "presentation_version_conflict",
        "current_version": 2,
        "etag": '"v2"',
    }
    assert "html_document" not in stale.text
    assert changed_document not in stale.text
    assert validation_pool.calls == []
