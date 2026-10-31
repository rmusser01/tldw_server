"""End-to-end owner lifecycle for standalone HTML presentations."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient

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
from tldw_Server_API.app.api.v1.endpoints.slides import router as slides_router
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthContext, AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.Jobs.manager import JobManager
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
    StandaloneHtmlGenerationService,
)
from tldw_Server_API.app.core.Slides.standalone_html_sources import (
    StandaloneHtmlSourceProvenance,
    StandaloneHtmlSourceSnapshot,
)
from tldw_Server_API.app.core.Slides.standalone_html_validation_pool import (
    StandaloneHtmlValidationPool,
)
from tldw_Server_API.app.core.Slides.standalone_html_validator import (
    validate_standalone_html,
)
from tldw_Server_API.app.services.standalone_html_generation_jobs_worker import (
    process_standalone_html_generation_job,
)

_ACCEPT = "X-Slides-Accept-Content-Kinds"
_BOTH = {_ACCEPT: "structured_slides,standalone_html"}
_REVISION = "sha256:" + "7" * 64
_NOW = datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc)


def _config(*, enabled: bool = True) -> SlidesStandaloneHtmlConfig:
    target = ResolvedExecutionTarget(
        provider="openai",
        model="task17-model",
        adapter_id="openai_official_chat_v1",
        endpoint_identity="https://api.openai.com:443/v1/chat/completions",
    )
    prompt_text = "Build one self-contained standalone presentation."
    prompt = ResolvedPrompt(
        text=prompt_text,
        sha256=hashlib.sha256(prompt_text.encode()).hexdigest(),
        contract_version="slides.standalone_html.v1",
        byte_count=len(prompt_text.encode()),
    )
    return SlidesStandaloneHtmlConfig(
        feature_enabled=enabled,
        egress_enabled=enabled,
        enabled=enabled,
        disabled_reason=None if enabled else "feature_disabled",
        target=target if enabled else None,
        prompt=prompt if enabled else None,
        allowed_targets=(target,) if enabled else (),
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
        generation_config_revision=_REVISION if enabled else None,
        _revision_manifest="task17-integration" if enabled else "",
    )


def _digest_material() -> tuple[StandaloneHtmlHmacKeyring, DigestKeySnapshot]:
    keyring = StandaloneHtmlHmacKeyring(
        secrets={"task17-key": b"k" * 32},
        current_key_id="task17-key",
    )
    snapshot = DigestKeySnapshot(
        records=(
            DigestKeyMetadata(
                key_id="task17-key",
                state=DigestKeyState.CURRENT,
                activated_at=_NOW - timedelta(days=1),
                retired_at=None,
            ),
        ),
        config_epoch="task17-config",
        configured_current_key_id="task17-key",
        availability=DigestKeyAvailability(missing_key_ids=()),
    )
    return keyring, snapshot


def _document(index: int) -> bytes:
    return (
        '<!doctype html><html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>Task17 Deck {index}</title><style>.slide{{color:#111}}</style></head>"
        f'<body><section class="slide"><h1>Task17 source {index}</h1>'
        '<aside class="notes">Speaker context</aside></section>'
        f'<script>globalThis.__TASK17_GENERATED_SENTINEL__="task17-global-mutation-{index}";'
        "document.addEventListener('keydown', () => {});</script>"
        "</body></html>"
    ).encode()


def _structured_seed(db: SlidesDatabase) -> None:
    db.create_presentation(
        presentation_id="structured-task17",
        title="Structured Task17",
        description=None,
        theme="black",
        marp_theme=None,
        settings=None,
        studio_data=None,
        slides=json.dumps(
            [
                {
                    "order": 0,
                    "layout": "title",
                    "title": "Structured Task17",
                    "content": "",
                    "speaker_notes": None,
                    "metadata": {},
                }
            ]
        ),
        slides_text="Structured Task17",
        source_type="manual",
        source_ref=None,
        source_query=None,
        custom_css=None,
    )


def _generation_request(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "generation_mode": "standalone_html",
        "generation_config_revision": _REVISION,
        "source": source,
        "html_options": {
            "presentation_type": "tech-sharing",
            "audience": "release reviewers",
            "slide_count": 1,
            "visual_direction": "minimal-light",
            "delivery_style": "speaker-led",
        },
    }


@pytest.mark.asyncio
async def test_owner_generation_jobs_and_presentation_http_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise five source adapters through submit, worker, persistence, and HTTP."""
    monkeypatch.setenv("JOBS_DISABLE_LEASE_ENFORCEMENT", "0")
    monkeypatch.setenv("JOBS_SECRET_REJECT", "1")
    monkeypatch.delenv("JOBS_JSON_TRUNCATE", raising=False)
    monkeypatch.delenv("JOBS_MAX_JSON_BYTES", raising=False)

    generated_fixture = validate_standalone_html(_document(0))
    assert generated_fixture.slide_count == 1
    assert "task17-global-mutation-0" not in generated_fixture.indexable_text
    assert generated_fixture.html_sha256 == hashlib.sha256(_document(0)).hexdigest()

    slides_path = tmp_path / "Slides.db"
    db_holder = [SlidesDatabase(slides_path, client_id="1")]
    other_owner_db = SlidesDatabase(tmp_path / "Slides-owner-2.db", client_id="2")
    jobs = JobManager(db_path=tmp_path / "jobs.db")
    _structured_seed(db_holder[0])
    keyring, digest_snapshot = _digest_material()

    async def load_digest_snapshot() -> DigestKeySnapshot:
        return digest_snapshot

    enabled = {"value": True}

    def load_config() -> SlidesStandaloneHtmlConfig:
        return _config(enabled=enabled["value"])

    def runtime_factory(*, request: Request, slides_db: SlidesDatabase):
        del request
        generation_service = StandaloneHtmlGenerationService(
            slides_db=slides_db,
            job_manager=jobs,
            keyring=keyring,
            digest_snapshot_loader=load_digest_snapshot,
            now=lambda: _NOW,
            receipt_id_factory=lambda: str(uuid.uuid4()),
        )
        return slides_standalone_html.StandaloneHtmlApiRuntime(
            slides_db=slides_db,
            job_manager=jobs,
            generation_service=generation_service,
            config_loader=load_config,
            validator_available=True,
        )
    validation_pool = StandaloneHtmlValidationPool(
        max_workers=1,
        watchdog_seconds=10,
        mp_start_method="spawn",
    )
    app = FastAPI()
    app.state.standalone_html_api_runtime_factory = runtime_factory
    app.state.standalone_html_validation_pool = validation_pool
    app.include_router(slides_router, prefix="/api/v1", tags=["slides"])

    active_owner = {"id": 1}

    async def override_user() -> User:
        return User(
            id=active_owner["id"],
            username=f"task17-owner-{active_owner['id']}",
            email=None,
            is_active=True,
            is_admin=True,
        )

    async def override_principal(request: Request) -> AuthPrincipal:
        principal = AuthPrincipal(
            kind="user",
            user_id=active_owner["id"],
            api_key_id=None,
            subject=f"task17-owner-{active_owner['id']}",
            token_type="single_user",  # nosec B106 - fixed nonsecret test principal
            jti=None,
            roles=["admin"],
            permissions=["media.create", "media.read", "media.update", "media.delete"],
            is_admin=True,
            org_ids=[],
            team_ids=[],
        )
        request.state.auth = AuthContext(
            principal=principal,
            ip=None,
            user_agent=None,
            request_id=None,
        )
        return principal

    async def override_slides_db():
        yield db_holder[0] if active_owner["id"] == 1 else other_owner_db

    async def override_media_db():
        yield SimpleNamespace(adapter="media")

    async def override_chacha_db():
        return SimpleNamespace(adapter="chacha")

    async def override_collections_db():
        return SimpleNamespace(adapter="collections")

    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_auth_principal] = override_principal
    app.dependency_overrides[get_slides_db_for_user] = override_slides_db
    app.dependency_overrides[get_media_db_for_user] = override_media_db
    app.dependency_overrides[get_chacha_db_for_user] = override_chacha_db
    app.dependency_overrides[get_collections_db_for_user] = override_collections_db

    source_requests = {
        "prompt": {"kind": "prompt", "prompt": "Direct source"},
        "chat": {"kind": "chat", "conversation_id": "conversation-task17"},
        "media": {"kind": "media", "media_id": 17},
        "notes": {"kind": "notes", "note_ids": ["note-a", "note-b"]},
        "rag": {"kind": "rag", "query": "release evidence", "top_k": 2},
    }
    source_calls: list[tuple[str, bool, bool]] = []

    async def resolve_source(
        source: dict[str, Any],
        *,
        owner_user_id: str,
        limits: Any,
        media_db: Any,
        chacha_db: Any,
    ) -> StandaloneHtmlSourceSnapshot:
        del limits
        kind = str(source["kind"])
        assert owner_user_id == "1"
        assert (media_db is not None) is (kind in {"media", "rag"})
        assert (chacha_db is not None) is (kind in {"chat", "notes", "rag"})
        text = f"Resolved private {kind} source"
        if kind == "prompt":
            provenance = StandaloneHtmlSourceProvenance("prompt", None)
        elif kind == "chat":
            provenance = StandaloneHtmlSourceProvenance(
                "chat", str(source["conversation_id"])
            )
        elif kind == "media":
            provenance = StandaloneHtmlSourceProvenance("media", str(source["media_id"]))
        elif kind == "notes":
            reference = json.dumps(
                {"note_ids": source["note_ids"]},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            provenance = StandaloneHtmlSourceProvenance(
                "notes", None, reference_hmac_input=reference
            )
        else:
            reference = json.dumps(
                {"query": source["query"], "top_k": source["top_k"]},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            provenance = StandaloneHtmlSourceProvenance(
                "rag", None, reference_hmac_input=reference
            )
        source_calls.append((kind, media_db is not None, chacha_db is not None))
        return StandaloneHtmlSourceSnapshot(
            source_kind=kind,  # type: ignore[arg-type]
            text=text,
            char_count=len(text),
            byte_count=len(text.encode()),
            token_count=4,
            provenance=provenance,
        )

    monkeypatch.setattr(slides_standalone_html, "resolve_standalone_html_source", resolve_source)

    provider_calls: list[str] = []

    async def provider_generate(**kwargs: Any) -> bytes:
        assert kwargs["stored_target"].adapter_id == "openai_official_chat_v1"
        assert "Resolved private" in kwargs["user_content"]
        provider_calls.append(kwargs["user_content"])
        return _document(len(provider_calls))

    created: list[str] = []
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            for index, (kind, source) in enumerate(source_requests.items(), start=1):
                idempotency_key = f"task17-{kind}-request-0001"
                request_body = _generation_request(source)
                submitted = await client.post(
                    "/api/v1/slides/generations",
                    headers={"Idempotency-Key": idempotency_key},
                    json=request_body,
                )
                assert submitted.status_code == 202, submitted.text
                generation_id = submitted.json()["generation_id"]

                job = jobs.acquire_next_job(
                    domain="slides",
                    queue="default",
                    job_type="presentation.generate",
                    lease_seconds=600,
                    worker_id=f"task17-worker-{kind}",
                )
                assert job is not None
                assert job["owner_user_id"] == "1"
                assert job["payload"] == {"receipt_id": generation_id}

                result = await process_standalone_html_generation_job(
                    job,
                    job_manager=jobs,
                    slides_db_factory=lambda owner: db_holder[0] if owner == "1" else None,
                    keyring=keyring,
                    digest_snapshot_loader=load_digest_snapshot,
                    validation_pool=validation_pool,
                    current_config_loader=load_config,
                    provider_api_key_loader=lambda _target: None,
                    provider_generate=provider_generate,
                    now=lambda index=index: _NOW + timedelta(minutes=index),
                )
                assert isinstance(result, dict)
                assert result["content_kind"] == "standalone_html"
                lease_id = str(job["lease_id"])
                assert jobs.complete_job(
                    int(job["id"]),
                    result=result,
                    worker_id=str(job["worker_id"]),
                    lease_id=lease_id,
                    completion_token=lease_id,
                    enforce=True,
                )

                status_response = await client.get(
                    f"/api/v1/slides/generations/{generation_id}"
                )
                assert status_response.status_code == 200, status_response.text
                assert status_response.json()["status"] == "completed"
                assert status_response.json()["presentation_id"] == generation_id

                replay = await client.post(
                    "/api/v1/slides/generations",
                    headers={"Idempotency-Key": idempotency_key},
                    json=request_body,
                )
                assert replay.status_code == 200, replay.text
                assert replay.json()["generation_id"] == generation_id
                assert replay.json()["status"] == "completed"
                assert len(provider_calls) == index
                created.append(generation_id)

            assert [call[0] for call in source_calls] == list(source_requests)
            assert len(provider_calls) == len(source_requests)

            legacy_list = await client.get("/api/v1/slides/presentations")
            assert legacy_list.status_code == 200, legacy_list.text
            assert legacy_list.json()["total"] == 1
            assert legacy_list.json()["presentations"][0]["id"] == "structured-task17"

            structured_detail = await client.get(
                "/api/v1/slides/presentations/structured-task17"
            )
            assert structured_detail.status_code == 200, structured_detail.text
            assert structured_detail.headers["ETag"] == 'W/"v1"'
            structured_saved = await client.patch(
                "/api/v1/slides/presentations/structured-task17",
                headers={"If-Match": structured_detail.headers["ETag"]},
                json={"title": "Structured Task17 Saved"},
            )
            assert structured_saved.status_code == 200, structured_saved.text
            assert structured_saved.headers["ETag"] == 'W/"v2"'
            assert structured_saved.json()["title"] == "Structured Task17 Saved"

            html_list = await client.get(
                "/api/v1/slides/presentations?limit=20",
                headers={_ACCEPT: "standalone_html"},
            )
            assert html_list.status_code == 200, html_list.text
            assert html_list.json()["total"] == 5
            assert all(
                item["content_kind"] == "standalone_html"
                and "html_document" not in item
                for item in html_list.json()["presentations"]
            )

            legacy_search = await client.get(
                "/api/v1/slides/presentations/search?q=Task17&limit=20"
            )
            assert legacy_search.status_code == 200, legacy_search.text
            assert legacy_search.json()["total"] == 1
            assert [
                item["id"] for item in legacy_search.json()["presentations"]
            ] == ["structured-task17"]
            assert all(presentation_id not in legacy_search.text for presentation_id in created)

            searched = await client.get(
                "/api/v1/slides/presentations/search?q=Task17&limit=20",
                headers={_ACCEPT: "standalone_html"},
            )
            assert searched.status_code == 200, searched.text
            assert searched.json()["total"] == 5
            assert all(
                item["content_kind"] == "standalone_html"
                and "html_document" not in item
                for item in searched.json()["presentations"]
            )
            assert "Task17 source" not in searched.text

            presentation_id = created[0]
            detail = await client.get(
                f"/api/v1/slides/presentations/{presentation_id}", headers=_BOTH
            )
            assert detail.status_code == 200, detail.text
            assert detail.headers["ETag"] == '"v1"'
            assert detail.json()["html_document"] == _document(1).decode()

            changed_document = _document(1).decode().replace(
                "Task17 Deck 1", "Task17 Saved Deck"
            )
            saved = await client.put(
                f"/api/v1/slides/presentations/{presentation_id}/html-source",
                content=changed_document.encode(),
                headers={
                    **_BOTH,
                    "If-Match": detail.headers["ETag"],
                    "Content-Type": "application/octet-stream",
                },
            )
            assert saved.status_code == 200, saved.text
            assert saved.headers["ETag"] == '"v2"'

            versions = await client.get(
                f"/api/v1/slides/presentations/{presentation_id}/versions",
                headers=_BOTH,
            )
            assert versions.status_code == 200, versions.text
            assert versions.json()["total"] == 2
            assert "html_document" not in versions.text

            first_version = await client.get(
                f"/api/v1/slides/presentations/{presentation_id}/versions/1",
                headers=_BOTH,
            )
            assert first_version.status_code == 200, first_version.text
            assert first_version.headers["ETag"] == '"v1"'
            assert first_version.json()["html_document"] == _document(1).decode()

            exported = await client.get(
                f"/api/v1/slides/presentations/{presentation_id}/export?format=html",
                headers=_BOTH,
            )
            assert exported.status_code == 200, exported.text
            assert exported.content == changed_document.encode()
            assert exported.headers["Content-Type"] == "application/octet-stream"
            assert exported.headers["Content-Disposition"] == (
                'attachment; filename="presentation.html"'
            )
            assert exported.headers["X-Content-Type-Options"] == "nosniff"
            assert exported.headers["X-Download-Options"] == "noopen"
            assert exported.headers["Cache-Control"] == "private, no-store"
            assert exported.headers["Referrer-Policy"] == "no-referrer"
            assert exported.headers["Cross-Origin-Resource-Policy"] == "same-origin"
            assert all(
                not response.headers.get("Content-Type", "").startswith("text/html")
                for response in (detail, saved, first_version, exported)
            )

            enabled["value"] = False
            disabled_submit = await client.post(
                "/api/v1/slides/generations",
                headers={"Idempotency-Key": "task17-disabled-request-0001"},
                json=_generation_request({"kind": "prompt", "prompt": "blocked"}),
            )
            assert disabled_submit.status_code == 503, disabled_submit.text
            assert disabled_submit.json() == {"detail": "feature_disabled"}

            still_readable = await client.get(
                f"/api/v1/slides/presentations/{presentation_id}", headers=_BOTH
            )
            assert still_readable.status_code == 200, still_readable.text
            assert still_readable.json()["html_document"] == changed_document

            calls_before_other_owner = (len(source_calls), len(provider_calls))
            active_owner["id"] = 2
            cross_owner_responses = [
                await client.get(
                    f"/api/v1/slides/presentations/{presentation_id}", headers=_BOTH
                ),
                await client.get(
                    f"/api/v1/slides/presentations/{presentation_id}/versions",
                    headers=_BOTH,
                ),
                await client.get(
                    f"/api/v1/slides/presentations/{presentation_id}/versions/1",
                    headers=_BOTH,
                ),
                await client.get(
                    f"/api/v1/slides/presentations/{presentation_id}/export?format=html",
                    headers=_BOTH,
                ),
                await client.get(f"/api/v1/slides/generations/{presentation_id}"),
            ]
            assert [response.status_code for response in cross_owner_responses] == [
                404,
                404,
                404,
                404,
                404,
            ]
            assert all(
                response.headers["Content-Type"].startswith("application/json")
                and len(response.content) <= 128
                and changed_document not in response.text
                for response in cross_owner_responses
            )
            assert (len(source_calls), len(provider_calls)) == calls_before_other_owner
            active_owner["id"] = 1

            db_holder[0].close_connection()
            reopened = SlidesDatabase(slides_path, client_id="1")
            db_holder[0] = reopened
            reopened_detail = await client.get(
                f"/api/v1/slides/presentations/{presentation_id}", headers=_BOTH
            )
            assert reopened_detail.status_code == 200, reopened_detail.text
            assert reopened_detail.headers["ETag"] == '"v2"'
            assert reopened_detail.json()["html_document"] == changed_document
    finally:
        await validation_pool.close()
        db_holder[0].close_connection()
        other_owner_db.close_connection()
        app.dependency_overrides.clear()
