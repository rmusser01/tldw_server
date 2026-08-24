"""Receipt-backed standalone HTML generation Jobs contracts."""

from __future__ import annotations

import asyncio
import base64
import gzip
import hmac
import importlib
import inspect
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier
from types import ModuleType
from typing import Any

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_sdk import (
    WorkerConfig,
    WorkerSDK,
    WorkerTerminalizationConflict,
    WorkerTerminalOutcome,
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
from tldw_Server_API.app.core.Slides.standalone_html_contracts import (
    StandaloneHtmlValidationError,
    StandaloneHtmlValidationResult,
)
from tldw_Server_API.app.core.Slides.standalone_html_provider import (
    StandaloneHtmlProviderError,
)
from tldw_Server_API.app.core.Slides.standalone_html_registry import (
    DigestKeyAvailability,
    DigestKeyMetadata,
    DigestKeySnapshot,
    DigestKeyState,
    StandaloneHtmlHmacKeyring,
)
from tldw_Server_API.app.core.Slides.standalone_html_sources import (
    StandaloneHtmlSourceProvenance,
    StandaloneHtmlSourceSnapshot,
)

_FIXED_NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
_RECEIPT_ID = "0198b65f-a600-7000-8000-000000000001"
_IDEMPOTENCY_KEY = "receipt-test-key-0001"


def _module(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pytest.fail(f"required Task 8 module is missing: {name}", pytrace=False)


def _service_module() -> ModuleType:
    return _module("tldw_Server_API.app.core.Slides.standalone_html_service")


def _worker_module() -> ModuleType:
    return _module("tldw_Server_API.app.services.standalone_html_generation_jobs_worker")


def _request(*, prompt: str = "Explain café locks") -> dict[str, Any]:
    return {
        "generation_mode": "standalone_html",
        "generation_config_revision": "sha256:" + "a" * 64,
        "source": {"kind": "prompt", "prompt": prompt},
        "html_options": {
            "presentation_type": "tech-sharing",
            "audience": " backend engineers ",
            "slide_count": 3,
            "visual_direction": "dark-technical",
            "delivery_style": "speaker-led",
        },
    }


def _config() -> SlidesStandaloneHtmlConfig:
    target = ResolvedExecutionTarget(
        provider="openai",
        model="gpt-test",
        adapter_id="openai_official_chat_v1",
        endpoint_identity="https://api.openai.com:443/v1/chat/completions",
    )
    prompt_text = "Build one self-contained presentation."
    prompt = ResolvedPrompt(
        text=prompt_text,
        sha256=__import__("hashlib").sha256(prompt_text.encode()).hexdigest(),
        contract_version="slides.standalone_html.v1",
        byte_count=len(prompt_text.encode()),
    )
    input_limits = StandaloneHtmlInputLimits(
        max_request_bytes=4_194_304,
        max_source_chars=200_000,
        max_source_tokens=50_000,
        max_audience_chars=500,
        max_source_identifier_bytes=256,
        max_note_ids=100,
        max_rag_query_chars=20_000,
        max_rag_top_k=100,
    )
    output_limits = StandaloneHtmlOutputLimits(
        max_provider_response_bytes=8_388_608,
        max_document_bytes=1_048_576,
    )
    provider_limits = StandaloneHtmlProviderLimits(
        connect_timeout_seconds=10.0,
        read_timeout_seconds=120.0,
        overall_timeout_seconds=180.0,
        max_output_tokens=16_384,
    )
    return SlidesStandaloneHtmlConfig(
        feature_enabled=True,
        egress_enabled=True,
        enabled=True,
        disabled_reason=None,
        target=target,
        prompt=prompt,
        allowed_targets=(target,),
        input_limits=input_limits,
        output_limits=output_limits,
        provider_limits=provider_limits,
        generation_config_revision="sha256:" + "a" * 64,
        _revision_manifest="test",
    )


def _digest_material() -> tuple[StandaloneHtmlHmacKeyring, DigestKeySnapshot]:
    keyring = StandaloneHtmlHmacKeyring(
        secrets={"key-v1": b"k" * 32},
        current_key_id="key-v1",
    )
    snapshot = DigestKeySnapshot(
        records=(
            DigestKeyMetadata(
                key_id="key-v1",
                state=DigestKeyState.CURRENT,
                activated_at=_FIXED_NOW - timedelta(days=1),
                retired_at=None,
            ),
        ),
        config_epoch="config-v1",
        configured_current_key_id="key-v1",
        availability=DigestKeyAvailability(missing_key_ids=()),
    )
    return keyring, snapshot


def _digest_snapshot_loader(snapshot: DigestKeySnapshot):
    async def load() -> DigestKeySnapshot:
        return snapshot

    return load


def _source_snapshot() -> StandaloneHtmlSourceSnapshot:
    text = "Exact café source"
    return StandaloneHtmlSourceSnapshot(
        source_kind="prompt",
        text=text,
        char_count=len(text),
        byte_count=len(text.encode("utf-8")),
        token_count=3,
        provenance=StandaloneHtmlSourceProvenance(
            source_kind="prompt",
            source_ref=None,
        ),
    )


@pytest.fixture
def stores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("JOBS_DISABLE_LEASE_ENFORCEMENT", "0")
    monkeypatch.setenv("JOBS_SECRET_REJECT", "1")
    monkeypatch.delenv("JOBS_JSON_TRUNCATE", raising=False)
    monkeypatch.delenv("JOBS_MAX_JSON_BYTES", raising=False)
    slides = SlidesDatabase(tmp_path / "slides.db", client_id="owner-1")
    jobs = JobManager(db_path=tmp_path / "jobs.db")
    try:
        yield slides, jobs
    finally:
        slides.close_connection()


def _service(
    slides: SlidesDatabase,
    jobs: JobManager,
    *,
    keyring: StandaloneHtmlHmacKeyring | None = None,
    digest_snapshot: DigestKeySnapshot | None = None,
    digest_snapshot_loader: Any | None = None,
    receipt_id_factory: Any | None = None,
):
    module = _service_module()
    default_keyring, default_snapshot = _digest_material()
    kwargs = {
        "slides_db": slides,
        "job_manager": jobs,
        "keyring": keyring or default_keyring,
        "now": lambda: _FIXED_NOW,
        "receipt_id_factory": receipt_id_factory or (lambda: _RECEIPT_ID),
    }
    assert "digest_snapshot_loader" in inspect.signature(module.StandaloneHtmlGenerationService).parameters
    kwargs["digest_snapshot_loader"] = digest_snapshot_loader or _digest_snapshot_loader(
        digest_snapshot or default_snapshot
    )
    return module.StandaloneHtmlGenerationService(
        **kwargs,
    )


async def _submit(service: Any, *, request: dict[str, Any] | None = None):
    async def resolve_source(_source: dict[str, Any], _limits: Any):
        return _source_snapshot()

    return await service.submit(
        owner_user_id="owner-1",
        idempotency_key=_IDEMPOTENCY_KEY,
        request=request or _request(),
        config_loader=_config,
        source_resolver=resolve_source,
    )


@pytest.mark.parametrize(
    "value",
    [
        None,
        "short",
        "x" * 201,
        " surrounding-space-0001",
        "contains/slash-0001",
        "contains:☃:0000001",
    ],
)
def test_idempotency_key_uses_the_closed_url_safe_syntax(value: object):
    module = _service_module()
    with pytest.raises(module.StandaloneHtmlGenerationError) as exc:
        module.validate_idempotency_key(value)
    assert exc.value.code == "generation_idempotency_key_invalid"


def test_canonical_request_is_closed_integer_only_and_utf8():
    module = _service_module()
    canonical = module.canonicalize_generation_request(_request())
    assert canonical.manifest_bytes == json.dumps(
        {
            **_request(),
            "html_options": {
                **_request()["html_options"],
                "audience": "backend engineers",
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert b"caf\xc3\xa9" in canonical.manifest_bytes
    assert b"\\u00e9" not in canonical.manifest_bytes

    bad = _request()
    bad["html_options"]["slide_count"] = 3.0
    with pytest.raises(module.StandaloneHtmlGenerationError):
        module.canonicalize_generation_request(bad)


def test_rag_query_limit_applies_before_trimming():
    module = _service_module()
    request = _request()
    request["source"] = {"kind": "rag", "query": " x "}
    assert module.canonicalize_generation_request(request).source == {
        "kind": "rag",
        "query": "x",
        "top_k": 8,
    }

    request["source"] = {"kind": "rag", "query": (" " * 20_000) + "x"}
    with pytest.raises(module.StandaloneHtmlGenerationError) as exc:
        module.canonicalize_generation_request(request)
    assert exc.value.code == "generation_request_invalid"
    bad = _request()
    bad["provider"] = "openai"
    with pytest.raises(module.StandaloneHtmlGenerationError):
        module.canonicalize_generation_request(bad)


@pytest.mark.asyncio
async def test_atomic_claim_uses_receipt_only_job_payload_and_exact_replay(stores):
    slides, jobs = stores
    service = _service(slides, jobs)

    first = await _submit(service)
    replay = await _submit(service)

    assert first.receipt_id == replay.receipt_id == _RECEIPT_ID
    assert first.job_uuid == replay.job_uuid
    assert first.status == replay.status == "queued"
    assert first.replayed is False
    assert replay.replayed is True
    receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    generation_input = slides.get_generation_input(
        _RECEIPT_ID,
        owner_user_id="owner-1",
    )
    job = jobs.get_job_by_uuid(first.job_uuid)
    assert job is not None
    assert job["domain"] == "slides"
    assert job["queue"] == "default"
    assert job["job_type"] == "presentation.generate"
    assert job["owner_user_id"] == "owner-1"
    assert job["payload"] == {"receipt_id": _RECEIPT_ID}
    assert receipt.job_uuid == job["uuid"]
    assert receipt.job_id == job["id"]
    assert generation_input.source_text == "Exact café source"
    assert generation_input.system_prompt == _config().prompt.text
    assert generation_input.provider == _config().target.provider
    assert datetime.fromisoformat(generation_input.input_expires_at) == (_FIXED_NOW + timedelta(hours=24))

    changed = _request(prompt="A different canonical request")
    module = _service_module()
    with pytest.raises(module.StandaloneHtmlGenerationError) as exc:
        await _submit(service, request=changed)
    assert exc.value.code == "generation_idempotency_conflict"


def test_atomic_claim_serializes_two_connection_race(tmp_path: Path):
    slides_path = tmp_path / "atomic-claim-race.db"
    bootstrap = SlidesDatabase(slides_path, client_id="owner-1")
    bootstrap.close_connection()
    barrier = Barrier(2)
    created_at = _FIXED_NOW.isoformat()

    def claim(index: int):
        slides = SlidesDatabase(slides_path, client_id="owner-1")
        receipt_id = f"0198b65f-a600-7000-8000-{index:012d}"
        receipt = {
            "id": receipt_id,
            "owner_user_id": "owner-1",
            "digest_key_id": "key-v1",
            "idempotency_key_hmac_sha256": "a" * 64,
            "jobs_idempotency_key": "slides:v1:" + "b" * 64,
            "client_request_hmac_sha256": "c" * 64,
            "execution_hmac_sha256": "d" * 64,
            "created_at": created_at,
            "updated_at": created_at,
        }
        generation_input = {
            "receipt_id": receipt_id,
            "source_kind": "prompt",
            "source_text": "source",
            "source_hmac_sha256": "e" * 64,
            "source_bytes": 6,
            "provenance_json": "{}",
            "html_options_json": "{}",
            "provider": "openai",
            "model": "gpt-test",
            "adapter_id": "openai_official_chat_v1",
            "endpoint_identity": "https://api.openai.com:443/v1/chat/completions",
            "system_prompt": "prompt",
            "prompt_sha256": "f" * 64,
            "prompt_contract_version": "slides.standalone_html.v1",
            "input_expires_at": (_FIXED_NOW + timedelta(hours=24)).isoformat(),
            "created_at": created_at,
        }
        try:
            barrier.wait(timeout=5)
            return slides.claim_generation_receipt_input(
                receipt=receipt,
                generation_input=generation_input,
                replay_digest_candidates=("a" * 64,),
            )
        finally:
            slides.close_connection()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(claim, (1, 2)))

    assert sorted(result.created for result in results) == [False, True]
    winner_ids = {result.receipt.id for result in results}
    assert len(winner_ids) == 1
    assert all(result.generation_input is not None for result in results)
    assert {
        result.generation_input.receipt_id for result in results if result.generation_input is not None
    } == winner_ids
    slides = SlidesDatabase(slides_path, client_id="owner-1")
    try:
        connection = slides.get_connection()
        assert connection.execute("SELECT COUNT(*) FROM slides_generation_receipts").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM slides_generation_inputs").fetchone()[0] == 1
    finally:
        slides.close_connection()


@pytest.mark.asyncio
async def test_replay_precedes_current_config_and_source_resolution(stores):
    slides, jobs = stores
    service = _service(slides, jobs)
    first = await _submit(service)

    def stale_config():
        raise AssertionError("replay must not resolve current config")

    async def stale_source(_source: dict[str, Any], _limits: Any):
        raise AssertionError("replay must not reread source")

    replay = await service.submit(
        owner_user_id="owner-1",
        idempotency_key=_IDEMPOTENCY_KEY,
        request=_request(),
        config_loader=stale_config,
        source_resolver=stale_source,
    )
    assert replay.receipt_id == first.receipt_id
    assert replay.replayed is True


@pytest.mark.asyncio
async def test_truncating_jobs_policy_is_rejected_without_mutated_row(
    stores,
    monkeypatch: pytest.MonkeyPatch,
):
    slides, jobs = stores
    monkeypatch.setenv("JOBS_JSON_TRUNCATE", "true")
    monkeypatch.setenv("JOBS_MAX_JSON_BYTES", "8")
    service = _service(slides, jobs)
    module = _service_module()

    with pytest.raises(module.StandaloneHtmlGenerationError) as exc:
        await _submit(service)

    assert (exc.value.code, exc.value.status_code) == (
        "generation_job_payload_too_large",
        413,
    )
    assert (
        jobs.lookup_slides_generation_job(
            owner_user_id="owner-1",
            idempotency_key=module.derive_jobs_idempotency_key(
                owner_user_id="owner-1",
                idempotency_key=_IDEMPOTENCY_KEY,
                keyring=_digest_material()[0],
                digest_snapshot=_digest_material()[1],
            ),
        )
        is None
    )
    with pytest.raises(KeyError):
        slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert jobs.list_jobs(domain="slides", queue="default") == []


@pytest.mark.parametrize("invalid_shape", ["redacted", "nonexact"])
@pytest.mark.asyncio
async def test_invalid_jobs_payload_construction_is_deterministic_422(
    stores,
    monkeypatch: pytest.MonkeyPatch,
    invalid_shape: str,
):
    slides, jobs = stores
    monkeypatch.setenv("JOBS_MAX_JSON_BYTES", "1")
    if invalid_shape == "redacted":
        monkeypatch.setattr(
            jobs,
            "_scan_and_redact_secrets",
            lambda _payload: ({"receipt_id": "[REDACTED]"}, True, "payload"),
        )
    else:
        monkeypatch.setattr(
            jobs,
            "_maybe_encrypt_json",
            lambda _payload, _domain: {"unexpected": "envelope"},
        )

    module = _service_module()
    with pytest.raises(module.StandaloneHtmlGenerationError) as exc:
        await _submit(_service(slides, jobs))
    assert (exc.value.code, exc.value.status_code) == (
        "generation_job_payload_invalid",
        422,
    )
    with pytest.raises(KeyError):
        slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert jobs.list_jobs(domain="slides", queue="default") == []


@pytest.mark.parametrize(
    ("failure_phase", "expected_code", "expected_status"),
    [
        ("encrypt", "generation_job_payload_unavailable", 503),
        ("decrypt", "generation_job_payload_invalid", 422),
    ],
)
@pytest.mark.asyncio
async def test_jobs_payload_crypto_preflight_fails_closed_without_a_jobs_row(
    stores,
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
    expected_code: str,
    expected_status: int,
):
    slides, jobs = stores

    def unavailable(*_args: object, **_kwargs: object):
        raise RuntimeError("source-bearing payload policy failure")

    monkeypatch.setattr(
        jobs,
        "_maybe_encrypt_json" if failure_phase == "encrypt" else "_maybe_decrypt_json",
        unavailable,
    )
    module = _service_module()
    with pytest.raises(module.StandaloneHtmlGenerationError) as exc:
        await _submit(_service(slides, jobs))
    assert (exc.value.code, exc.value.status_code, str(exc.value)) == (
        expected_code,
        expected_status,
        expected_code,
    )
    with pytest.raises(KeyError):
        slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert jobs.list_jobs(domain="slides", queue="default") == []


@pytest.mark.asyncio
async def test_jobs_payload_policy_failure_is_the_only_transient_preflight_error(
    stores,
    monkeypatch: pytest.MonkeyPatch,
):
    slides, jobs = stores

    def unavailable(_payload: object):
        raise RuntimeError("source-bearing jobs policy failure")

    monkeypatch.setattr(jobs, "_scan_and_redact_secrets", unavailable)
    module = _service_module()
    with pytest.raises(module.StandaloneHtmlGenerationError) as exc:
        await _submit(_service(slides, jobs))
    assert (exc.value.code, exc.value.status_code, str(exc.value)) == (
        "generation_job_payload_unavailable",
        503,
        "generation_job_payload_unavailable",
    )
    with pytest.raises(KeyError):
        slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert jobs.list_jobs(domain="slides", queue="default") == []


class _Reservation:
    def __init__(
        self,
        validation: StandaloneHtmlValidationResult,
        *,
        on_validate: Any | None = None,
        validation_error: StandaloneHtmlValidationError | None = None,
    ):
        self.validation = validation
        self.on_validate = on_validate
        self.validation_error = validation_error
        self.released = False
        self.validate_calls: list[tuple[str | bytes, str | None]] = []

    async def validate(
        self,
        document: str | bytes,
        *,
        delivery_style: str | None = None,
    ):
        self.validate_calls.append((document, delivery_style))
        if self.on_validate is not None:
            self.on_validate()
        if self.validation_error is not None:
            raise self.validation_error
        return self.validation

    async def release(self):
        self.released = True


class _ValidationPool:
    def __init__(
        self,
        validation: StandaloneHtmlValidationResult,
        *,
        on_validate: Any | None = None,
        validation_error: StandaloneHtmlValidationError | None = None,
        acquire_error: StandaloneHtmlValidationError | None = None,
        on_acquire: Any | None = None,
    ):
        self.reservation = _Reservation(
            validation,
            on_validate=on_validate,
            validation_error=validation_error,
        )
        self.acquire_error = acquire_error
        self.on_acquire = on_acquire

    async def acquire_generation_reservation(self):
        if self.acquire_error is not None:
            raise self.acquire_error
        if self.on_acquire is not None:
            self.on_acquire()
        return self.reservation


_HTML = b"<!doctype html><html><head><title>Caf\xc3\xa9</title></head><body></body></html>"


def _validation() -> StandaloneHtmlValidationResult:
    return StandaloneHtmlValidationResult(
        title="Café",
        slide_count=3,
        html_bytes=len(_HTML),
        html_sha256=__import__("hashlib").sha256(_HTML).hexdigest(),
        indexable_text="Café locks",
    )


async def _submitted_and_acquired(
    slides: SlidesDatabase,
    jobs: JobManager,
) -> tuple[Any, dict[str, Any]]:
    submitted = await _submit(_service(slides, jobs))
    job = jobs.acquire_next_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        lease_seconds=600,
        worker_id="slides-worker-1",
    )
    assert job is not None
    return submitted, job


def _update_job(jobs: JobManager, job_id: int, assignment: str, parameters: tuple[Any, ...] = ()) -> None:
    connection = jobs._connect()
    try:
        with connection:
            connection.execute(
                f"UPDATE jobs SET {assignment} WHERE id = ?",  # nosec B608 - fixed test SQL
                (*parameters, job_id),
            )
    finally:
        connection.close()


async def _process(
    slides: SlidesDatabase,
    jobs: JobManager,
    job: dict[str, Any],
    *,
    pool: Any | None = None,
    provider_generate: Any | None = None,
    slides_db_factory: Any | None = None,
    now: Any | None = None,
    keyring: StandaloneHtmlHmacKeyring | None = None,
    digest_snapshot: DigestKeySnapshot | None = None,
    digest_snapshot_loader: Any | None = None,
    current_config_loader: Any | None = None,
    provider_api_key_loader: Any | None = None,
):
    worker = _worker_module()

    async def default_provider(**_kwargs: Any) -> bytes:
        return _HTML

    kwargs = {
        "job_manager": jobs,
        "slides_db_factory": slides_db_factory or (lambda owner: slides if owner == "owner-1" else slides),
        "keyring": keyring or _digest_material()[0],
        "validation_pool": pool or _ValidationPool(_validation()),
        "current_config_loader": current_config_loader or _config,
        "provider_api_key_loader": provider_api_key_loader or (lambda _target: None),
        "provider_generate": provider_generate or default_provider,
        "now": now or (lambda: _FIXED_NOW + timedelta(minutes=1)),
    }
    assert "digest_snapshot_loader" in inspect.signature(worker.process_standalone_html_generation_job).parameters
    kwargs["digest_snapshot_loader"] = digest_snapshot_loader or _digest_snapshot_loader(
        digest_snapshot or _digest_material()[1]
    )
    return await worker.process_standalone_html_generation_job(
        job,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_worker_commits_presentation_receipt_and_input_once(stores):
    slides, jobs = stores
    submitted = await _submit(_service(slides, jobs))
    job = jobs.acquire_next_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        lease_seconds=60,
        worker_id="slides-worker-1",
    )
    assert job is not None
    html = b"<!doctype html><html><head><title>Caf\xc3\xa9</title></head><body></body></html>"
    validation = StandaloneHtmlValidationResult(
        title="Café",
        slide_count=3,
        html_bytes=len(html),
        html_sha256=__import__("hashlib").sha256(html).hexdigest(),
        indexable_text="Café locks",
    )
    pool = _ValidationPool(validation)
    provider_calls = 0

    async def provider_generate(**_kwargs: Any) -> bytes:
        nonlocal provider_calls
        provider_calls += 1
        return html

    worker = _worker_module()
    result = await worker.process_standalone_html_generation_job(
        job,
        job_manager=jobs,
        slides_db_factory=lambda owner: slides if owner == "owner-1" else None,
        keyring=_digest_material()[0],
        digest_snapshot_loader=_digest_snapshot_loader(_digest_material()[1]),
        validation_pool=pool,
        current_config_loader=_config,
        provider_api_key_loader=lambda _target: None,
        provider_generate=provider_generate,
        now=lambda: _FIXED_NOW + timedelta(minutes=1),
    )

    assert provider_calls == 1
    assert result == {
        "presentation_id": _RECEIPT_ID,
        "content_kind": "standalone_html",
        "html_bytes": len(html),
        "html_slide_count": 3,
        "validation_status": "accepted",
    }
    receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    presentation = slides.get_presentation_by_id(_RECEIPT_ID)
    assert receipt.receipt_status == "completed"
    assert receipt.presentation_id == presentation.id == submitted.receipt_id
    assert receipt.expires_at == (_FIXED_NOW + timedelta(minutes=1, days=30)).isoformat()
    assert presentation.generation_job_uuid == job["uuid"]
    assert presentation.html_document == html.decode("utf-8")
    with pytest.raises(KeyError):
        slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")


def _rotation_material(
    current_key_id: str,
) -> tuple[StandaloneHtmlHmacKeyring, DigestKeySnapshot]:
    keyring = StandaloneHtmlHmacKeyring(
        secrets={"key-old": b"o" * 32, "key-new": b"n" * 32},
        current_key_id=current_key_id,
    )
    records = []
    for key_id in ("key-old", "key-new"):
        state = DigestKeyState.CURRENT if key_id == current_key_id else DigestKeyState.RETIRING
        records.append(
            DigestKeyMetadata(
                key_id=key_id,
                state=state,
                activated_at=_FIXED_NOW - timedelta(days=2),
                retired_at=None if state is DigestKeyState.CURRENT else _FIXED_NOW - timedelta(days=1),
            )
        )
    return keyring, DigestKeySnapshot(
        records=tuple(records),
        config_epoch="rotation-v1",
        configured_current_key_id=current_key_id,
        availability=DigestKeyAvailability(missing_key_ids=()),
    )


def test_hmac_domains_are_distinct_for_identical_utf8_payload():
    from tldw_Server_API.app.core.Slides.standalone_html_registry import HmacDomain

    keyring, snapshot = _digest_material()
    payload = "café".encode()
    digests = {
        keyring.digest_current(snapshot=snapshot, domain=domain, payload=payload).digest_hex for domain in HmacDomain
    }
    assert len(digests) == len(HmacDomain) == 5


@pytest.mark.asyncio
async def test_historical_key_replay_uses_constant_time_digest_comparison(
    stores,
    monkeypatch: pytest.MonkeyPatch,
):
    slides, jobs = stores
    old_keyring, old_snapshot = _rotation_material("key-old")
    await _submit(
        _service(
            slides,
            jobs,
            keyring=old_keyring,
            digest_snapshot=old_snapshot,
        )
    )
    receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    new_keyring, new_snapshot = _rotation_material("key-new")
    module = _service_module()
    service = _service(
        slides,
        jobs,
        keyring=new_keyring,
        digest_snapshot=new_snapshot,
    )
    canonical = module.canonicalize_generation_request(_request())
    candidates = service._candidate_hmacs(
        _IDEMPOTENCY_KEY,
        digest_snapshot=new_snapshot,
    )
    candidate_digest = next(digest for key_id, digest in candidates if key_id == receipt.digest_key_id)
    incoming_request_hmac = service._request_hmac(
        receipt.digest_key_id,
        canonical.manifest_bytes,
        digest_snapshot=new_snapshot,
    )
    real_compare = module.hmac.compare_digest
    comparisons: list[tuple[object, object]] = []

    def record_compare(left: object, right: object) -> bool:
        comparisons.append((left, right))
        return real_compare(left, right)

    monkeypatch.setattr(module.hmac, "compare_digest", record_compare)
    replay = await _submit(service)
    assert replay.replayed is True
    assert (
        candidate_digest,
        receipt.idempotency_key_hmac_sha256,
    ) in comparisons
    assert (
        incoming_request_hmac,
        receipt.client_request_hmac_sha256,
    ) in comparisons
    assert all(isinstance(left, str) and isinstance(right, str) for left, right in comparisons)


@pytest.mark.asyncio
async def test_persisted_provenance_source_hmac_uses_strict_constant_time_comparison(stores, monkeypatch):
    slides, jobs = stores
    await _submit(_service(slides, jobs))
    receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    generation_input = slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")
    module = _service_module()
    comparisons: list[tuple[object, object]] = []
    real_compare = hmac.compare_digest

    def record_compare(left: object, right: object) -> bool:
        comparisons.append((left, right))
        return real_compare(left, right)

    monkeypatch.setattr(module.hmac, "compare_digest", record_compare)
    module._validated_provenance(receipt, generation_input)
    assert (
        generation_input.source_hmac_sha256,
        generation_input.source_hmac_sha256,
    ) in comparisons

    different_source_hmac = "0" * 64 if generation_input.source_hmac_sha256 != "0" * 64 else "1" * 64
    provenance = json.loads(generation_input.provenance_json)
    provenance["source_snapshot_hmac_sha256"] = different_source_hmac
    mismatched = replace(
        generation_input,
        provenance_json=json.dumps(provenance, sort_keys=True, separators=(",", ":")),
    )
    with pytest.raises(module.StandaloneHtmlGenerationError):
        module._validated_provenance(receipt, mismatched)
    assert (different_source_hmac, generation_input.source_hmac_sha256) in comparisons

    provenance = json.loads(generation_input.provenance_json)
    provenance["source_snapshot_hmac_sha256"] = generation_input.source_hmac_sha256.upper()
    malformed = replace(
        generation_input,
        source_hmac_sha256=generation_input.source_hmac_sha256.upper(),
        provenance_json=json.dumps(provenance, sort_keys=True, separators=(",", ":")),
    )
    with pytest.raises(module.StandaloneHtmlGenerationError):
        module._validated_provenance(receipt, malformed)


@pytest.mark.asyncio
async def test_jobs_hmac_derived_idempotency_key_uses_strict_constant_time_comparison(
    stores,
    monkeypatch: pytest.MonkeyPatch,
):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    service = _service(slides, jobs)
    comparisons: list[tuple[object, object]] = []
    real_compare = hmac.compare_digest

    def record_compare(left: object, right: object) -> bool:
        comparisons.append((left, right))
        return real_compare(left, right)

    monkeypatch.setattr(hmac, "compare_digest", record_compare)
    service._validate_job(job, receipt=receipt)
    assert (receipt.jobs_idempotency_key, receipt.jobs_idempotency_key) in comparisons

    different_jobs_key = "slides:v1:" + ("0" * 64 if not receipt.jobs_idempotency_key.endswith("0" * 64) else "1" * 64)
    with pytest.raises(_service_module().StandaloneHtmlGenerationError):
        service._validate_job(
            {**job, "idempotency_key": different_jobs_key},
            receipt=receipt,
        )
    assert (different_jobs_key, receipt.jobs_idempotency_key) in comparisons
    assert (
        _worker_module()._job_identity_is_exact(
            {**job, "idempotency_key": different_jobs_key},
            job,
        )
        is False
    )
    assert (different_jobs_key, job["idempotency_key"]) in comparisons

    malformed_receipt = replace(receipt, jobs_idempotency_key="slides:v1:not-a-digest")
    malformed_job = {**job, "idempotency_key": "slides:v1:not-a-digest"}
    with pytest.raises(_service_module().StandaloneHtmlGenerationError):
        service._validate_job(malformed_job, receipt=malformed_receipt)


@pytest.mark.parametrize("bool_field", ["schema_version", "source_bytes"])
@pytest.mark.asyncio
async def test_provenance_rejects_bool_as_integer_fields(stores, bool_field: str):
    slides, jobs = stores
    await _submit(_service(slides, jobs))
    receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    generation_input = slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")
    provenance = json.loads(generation_input.provenance_json)
    provenance[bool_field] = True
    if bool_field == "source_bytes":
        generation_input = replace(generation_input, source_bytes=1)
    generation_input = replace(
        generation_input,
        provenance_json=json.dumps(provenance, sort_keys=True, separators=(",", ":")),
    )

    with pytest.raises(_service_module().StandaloneHtmlGenerationError):
        _service_module()._validated_provenance(receipt, generation_input)


@pytest.mark.asyncio
async def test_missing_digest_key_has_one_stable_admission_error_before_lookup(stores):
    slides, jobs = stores
    keyring, snapshot = _digest_material()
    missing = replace(
        snapshot,
        availability=DigestKeyAvailability(missing_key_ids=("key-v1",)),
    )
    service = _service(slides, jobs, keyring=keyring, digest_snapshot=missing)
    config_calls = 0
    source_calls = 0

    def config_loader():
        nonlocal config_calls
        config_calls += 1
        return _config()

    async def source_resolver(_source: dict[str, Any], _limits: Any):
        nonlocal source_calls
        source_calls += 1
        return _source_snapshot()

    module = _service_module()
    with pytest.raises(module.StandaloneHtmlGenerationError) as exc:
        await service.submit(
            owner_user_id="owner-1",
            idempotency_key=_IDEMPOTENCY_KEY,
            request=_request(),
            config_loader=config_loader,
            source_resolver=source_resolver,
        )
    assert (exc.value.code, exc.value.status_code, str(exc.value)) == (
        "generation_digest_key_unavailable",
        503,
        "generation_digest_key_unavailable",
    )
    assert config_calls == source_calls == 0


@pytest.mark.asyncio
async def test_admission_reloads_digest_snapshot_after_source_before_claim(stores):
    slides, jobs = stores
    keyring, ready = _digest_material()
    missing = replace(
        ready,
        availability=DigestKeyAvailability(missing_key_ids=("key-v1",)),
    )
    state = {"snapshot": ready}

    async def digest_snapshot_loader():
        return state["snapshot"]

    service = _service(
        slides,
        jobs,
        keyring=keyring,
        digest_snapshot_loader=digest_snapshot_loader,
    )

    async def source_resolver(_source: dict[str, Any], _limits: Any):
        state["snapshot"] = missing
        return _source_snapshot()

    module = _service_module()
    with pytest.raises(module.StandaloneHtmlGenerationError) as exc:
        await service.submit(
            owner_user_id="owner-1",
            idempotency_key=_IDEMPOTENCY_KEY,
            request=_request(),
            config_loader=_config,
            source_resolver=source_resolver,
        )
    assert (exc.value.code, exc.value.status_code) == (
        "generation_digest_key_unavailable",
        503,
    )
    with pytest.raises(KeyError):
        slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert (
        jobs.lookup_slides_generation_job(
            owner_user_id="owner-1",
            idempotency_key=module.derive_jobs_idempotency_key(
                owner_user_id="owner-1",
                idempotency_key=_IDEMPOTENCY_KEY,
                keyring=keyring,
                digest_snapshot=ready,
            ),
        )
        is None
    )


@pytest.mark.asyncio
async def test_admission_rechecks_live_digest_state_before_idempotency_lookup(
    stores,
    monkeypatch: pytest.MonkeyPatch,
):
    slides, jobs = stores
    keyring, ready = _digest_material()
    state = {"snapshot": ready}
    events: list[str] = []
    real_lookup = slides.find_generation_receipt_by_idempotency_digests

    async def digest_snapshot_loader():
        events.append("snapshot")
        return state["snapshot"]

    def receipt_lookup(**kwargs: Any):
        events.append("lookup")
        assert events[-2:] == ["snapshot", "lookup"]
        return real_lookup(**kwargs)

    monkeypatch.setattr(
        slides,
        "find_generation_receipt_by_idempotency_digests",
        receipt_lookup,
    )

    service = _service(
        slides,
        jobs,
        keyring=keyring,
        digest_snapshot_loader=digest_snapshot_loader,
    )
    first = await _submit(service)
    assert events == ["snapshot", "lookup", "snapshot", "lookup"]
    events.clear()
    state["snapshot"] = replace(
        ready,
        availability=DigestKeyAvailability(missing_key_ids=("key-v1",)),
    )

    module = _service_module()
    with pytest.raises(module.StandaloneHtmlGenerationError) as exc:
        await _submit(service)
    assert (exc.value.code, exc.value.status_code) == (
        "generation_digest_key_unavailable",
        503,
    )
    assert events == ["snapshot"]
    assert first.receipt_id == _RECEIPT_ID


@pytest.mark.asyncio
async def test_reference_hmac_is_framed_inside_the_existing_source_domain(stores):
    slides, jobs = stores
    request = _request()
    request["source"] = {"kind": "notes", "note_ids": ["note-1"]}
    same_bytes = b'["note-1"]'

    async def resolve_source(_source: dict[str, Any], _limits: Any):
        text = same_bytes.decode("utf-8")
        return StandaloneHtmlSourceSnapshot(
            source_kind="notes",
            text=text,
            char_count=len(text),
            byte_count=len(same_bytes),
            token_count=3,
            provenance=StandaloneHtmlSourceProvenance(
                source_kind="notes",
                source_ref=None,
                reference_hmac_input=same_bytes,
            ),
        )

    await _service(slides, jobs).submit(
        owner_user_id="owner-1",
        idempotency_key=_IDEMPOTENCY_KEY,
        request=request,
        config_loader=_config,
        source_resolver=resolve_source,
    )
    generation_input = slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")
    provenance = json.loads(generation_input.provenance_json)
    assert provenance["source_ref"] != generation_input.source_hmac_sha256
    assert len(provenance["source_ref"]) == 64


@pytest.mark.parametrize(
    (
        "source_selector",
        "source_kind",
        "source_ref",
        "reference_input",
        "replacement_ref",
    ),
    (
        (
            {"kind": "notes", "note_ids": ["note-1"]},
            "notes",
            None,
            b'["note-1"]',
            "f" * 64,
        ),
        (
            {"kind": "media", "media_id": 7},
            "media",
            "7",
            None,
            "8",
        ),
    ),
)
@pytest.mark.asyncio
async def test_execution_hmac_rejects_valid_shape_provenance_reference_tamper_before_egress(
    stores,
    source_selector: dict[str, Any],
    source_kind: str,
    source_ref: str | None,
    reference_input: bytes | None,
    replacement_ref: str,
):
    slides, jobs = stores
    request = _request()
    request["source"] = source_selector

    async def resolve_source(_source: dict[str, Any], _limits: Any):
        text = "Bounded note source"
        return StandaloneHtmlSourceSnapshot(
            source_kind=source_kind,
            text=text,
            char_count=len(text),
            byte_count=len(text.encode("utf-8")),
            token_count=3,
            provenance=StandaloneHtmlSourceProvenance(
                source_kind=source_kind,
                source_ref=source_ref,
                reference_hmac_input=reference_input,
            ),
        )

    submitted = await _service(slides, jobs).submit(
        owner_user_id="owner-1",
        idempotency_key=_IDEMPOTENCY_KEY,
        request=request,
        config_loader=_config,
        source_resolver=resolve_source,
    )
    job = jobs.acquire_next_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        lease_seconds=600,
        worker_id="slides-worker-1",
    )
    assert job is not None
    generation_input = slides.get_generation_input(
        submitted.receipt_id,
        owner_user_id="owner-1",
    )
    provenance = json.loads(generation_input.provenance_json)
    assert provenance["source_ref"] != replacement_ref
    provenance["source_ref"] = replacement_ref
    with slides.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE slides_generation_inputs SET provenance_json=? WHERE receipt_id=?",
            (
                json.dumps(provenance, sort_keys=True, separators=(",", ":")),
                submitted.receipt_id,
            ),
        )
    provider_calls = 0

    async def provider_generate(**_kwargs: Any) -> bytes:
        nonlocal provider_calls
        provider_calls += 1
        return _HTML

    outcome = await _process(
        slides,
        jobs,
        job,
        provider_generate=provider_generate,
    )

    assert outcome == WorkerTerminalOutcome(
        status="failed",
        error_code="generation_correlation_mismatch",
        message="Generation correlation failed.",
    )
    assert provider_calls == 0


@pytest.mark.parametrize("receipt_status", ["running", "failed", "cancelled"])
@pytest.mark.asyncio
async def test_exact_replay_returns_running_failed_and_cancelled_without_rereading(
    stores,
    receipt_status: str,
):
    slides, jobs = stores
    service = _service(slides, jobs)
    submitted = await _submit(service)
    receipt = slides.get_generation_receipt(submitted.receipt_id, owner_user_id="owner-1")
    assert receipt.job_uuid is not None
    if receipt_status == "running":
        slides.set_generation_receipt_running(
            receipt_id=receipt.id,
            owner_user_id=receipt.owner_user_id,
            job_uuid=receipt.job_uuid,
            updated_at=_FIXED_NOW.isoformat(),
        )
    else:
        assert service.terminalize(
            receipt=receipt,
            status=receipt_status,
            error_code=("generation_cancelled" if receipt_status == "cancelled" else "generation_failed"),
            error_message="Generation did not complete.",
        )

    def fail_config():
        raise AssertionError("terminal replay must not load current configuration")

    async def fail_source(_source: dict[str, Any], _limits: Any):
        raise AssertionError("terminal replay must not load source")

    replay = await service.submit(
        owner_user_id="owner-1",
        idempotency_key=_IDEMPOTENCY_KEY,
        request=_request(),
        config_loader=fail_config,
        source_resolver=fail_source,
    )
    assert replay.status == receipt_status
    assert replay.receipt_id == submitted.receipt_id
    assert replay.replayed is True
    if receipt_status == "running":
        assert slides.get_generation_input(receipt.id, owner_user_id="owner-1")
    else:
        with pytest.raises(KeyError):
            slides.get_generation_input(receipt.id, owner_user_id="owner-1")


@pytest.mark.asyncio
async def test_owner_scoped_generation_lookup_makes_missing_and_cross_owner_identical(stores):
    slides, jobs = stores
    service = _service(slides, jobs)
    submitted = await _submit(service)
    module = _service_module()

    failures = []
    for owner, receipt_id in (
        ("other-owner", submitted.receipt_id),
        ("owner-1", str(uuid.uuid4())),
    ):
        with pytest.raises(module.StandaloneHtmlGenerationError) as exc:
            service.get_generation(owner_user_id=owner, receipt_id=receipt_id)
        failures.append((exc.value.code, exc.value.status_code, str(exc.value)))

    assert failures == [
        ("generation_not_found", 404, "generation_not_found"),
        ("generation_not_found", 404, "generation_not_found"),
    ]


@pytest.mark.asyncio
async def test_terminal_input_deletion_occurs_only_when_receipt_cas_wins(stores):
    slides, jobs = stores
    service = _service(slides, jobs)
    submitted = await _submit(service)
    receipt = slides.get_generation_receipt(submitted.receipt_id, owner_user_id="owner-1")
    losing_receipt = replace(receipt, job_uuid=str(uuid.uuid4()))

    assert (
        service.terminalize(
            receipt=losing_receipt,
            status="failed",
            error_code="generation_failed",
            error_message="Generation failed.",
        )
        is False
    )
    assert slides.get_generation_input(receipt.id, owner_user_id="owner-1")

    assert (
        service.terminalize(
            receipt=receipt,
            status="failed",
            error_code="generation_failed",
            error_message="Generation failed.",
        )
        is True
    )
    terminal = slides.get_generation_receipt(receipt.id, owner_user_id="owner-1")
    assert terminal.updated_at == _FIXED_NOW.isoformat()
    assert terminal.expires_at == (_FIXED_NOW + timedelta(days=30)).isoformat()
    with pytest.raises(KeyError):
        slides.get_generation_input(receipt.id, owner_user_id="owner-1")


@pytest.mark.asyncio
async def test_stale_unbound_terminal_cas_cannot_delete_concurrently_bound_input(stores):
    slides, jobs = stores
    await _submit(_service(slides, jobs))
    bound = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    stale_unbound = replace(bound, job_id=None, job_uuid=None, receipt_status="claimed")

    assert (
        _service(slides, jobs).terminalize(
            receipt=stale_unbound,
            status="failed",
            error_code="generation_correlation_mismatch",
            error_message="Generation correlation failed.",
        )
        is False
    )
    winner = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert winner.job_uuid == bound.job_uuid
    assert winner.receipt_status == "queued"
    assert slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")


@pytest.mark.asyncio
async def test_truncation_enabled_with_sufficient_limit_keeps_exact_receipt_payload(
    stores,
    monkeypatch: pytest.MonkeyPatch,
):
    slides, jobs = stores
    monkeypatch.setenv("JOBS_JSON_TRUNCATE", "true")
    monkeypatch.setenv("JOBS_MAX_JSON_BYTES", "4096")
    submitted = await _submit(_service(slides, jobs))
    job = jobs.get_job_by_uuid(submitted.job_uuid)
    assert job is not None
    assert job["payload"] == {"receipt_id": _RECEIPT_ID}


@pytest.mark.asyncio
async def test_completed_replay_precedes_current_config_and_source_resolution(stores):
    slides, jobs = stores
    await _submitted_and_acquired(slides, jobs)
    job = jobs.get_job_by_uuid(slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1").job_uuid)
    assert job is not None
    result = await _process(slides, jobs, job)

    def fail_config():
        raise AssertionError("completed replay must not load current configuration")

    async def fail_source(_source: dict[str, Any], _limits: Any):
        raise AssertionError("completed replay must not load source")

    replay = await _service(slides, jobs).submit(
        owner_user_id="owner-1",
        idempotency_key=_IDEMPOTENCY_KEY,
        request=_request(),
        config_loader=fail_config,
        source_resolver=fail_source,
    )
    assert replay.status == "completed"
    assert replay.presentation_id == result["presentation_id"]
    assert replay.replayed is True


@pytest.mark.asyncio
async def test_api_first_and_worker_first_binding_use_the_same_immutable_uuid(stores):
    slides, jobs = stores
    submitted = await _submit(_service(slides, jobs))
    api_bound = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert api_bound.job_uuid == submitted.job_uuid
    with slides.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE slides_generation_receipts SET job_id = NULL, job_uuid = NULL, "
            "receipt_status = 'claimed' WHERE id = ?",
            (_RECEIPT_ID,),
        )

    job = jobs.acquire_next_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        lease_seconds=600,
        worker_id="slides-worker-1",
    )
    assert job is not None
    await _process(slides, jobs, job)
    worker_bound = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert worker_bound.job_uuid == job["uuid"] == api_bound.job_uuid
    assert worker_bound.job_id == job["id"]


@pytest.mark.parametrize("archived", [False, True])
@pytest.mark.asyncio
async def test_claim_recovers_active_or_archived_job_by_uuid_authority(stores, archived: bool):
    slides, jobs = stores
    module = _service_module()
    keyring, snapshot = _digest_material()
    jobs_key = module.derive_jobs_idempotency_key(
        owner_user_id="owner-1",
        idempotency_key=_IDEMPOTENCY_KEY,
        keyring=keyring,
        digest_snapshot=snapshot,
    )
    existing = jobs.create_job(
        domain="slides",
        queue="default",
        job_type="presentation.generate",
        payload={"receipt_id": _RECEIPT_ID},
        owner_user_id="owner-1",
        idempotency_key=jobs_key,
    )
    if archived:
        connection = jobs._connect()
        try:
            with connection:
                active_columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
                copied_columns = [
                    row[1] for row in connection.execute("PRAGMA table_info(jobs_archive)") if row[1] in active_columns
                ]
                columns_sql = ", ".join(f'"{column}"' for column in copied_columns)
                connection.execute(
                    f"INSERT INTO jobs_archive ({columns_sql}) "  # nosec B608 - trusted schema metadata
                    f"SELECT {columns_sql} FROM jobs WHERE id = ?",  # nosec B608 - trusted schema metadata
                    (existing["id"],),
                )
                connection.execute("DELETE FROM jobs WHERE id = ?", (existing["id"],))
        finally:
            connection.close()

    submitted = await _submit(_service(slides, jobs, keyring=keyring, digest_snapshot=snapshot))
    receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert submitted.job_uuid == receipt.job_uuid == existing["uuid"]
    assert receipt.job_id == existing["id"]
    recovered = jobs.lookup_slides_generation_job(
        owner_user_id="owner-1",
        idempotency_key=jobs_key,
    )
    assert recovered is not None
    assert recovered["uuid"] == existing["uuid"]
    assert recovered["archived"] is archived


@pytest.mark.asyncio
async def test_bound_claim_replay_uses_uuid_authority_for_compressed_archive_with_null_id(
    stores,
):
    slides, jobs = stores
    submitted = await _submit(_service(slides, jobs))
    job = jobs.get_job_by_uuid(submitted.job_uuid)
    assert job is not None
    connection = jobs._connect()
    try:
        with connection:
            stored_payload = connection.execute(
                "SELECT payload FROM jobs WHERE uuid=?",
                (submitted.job_uuid,),
            ).fetchone()[0]
            active_columns = {row[1] for row in connection.execute("PRAGMA table_info(jobs)")}
            copied_columns = [
                row[1] for row in connection.execute("PRAGMA table_info(jobs_archive)") if row[1] in active_columns
            ]
            columns_sql = ", ".join(f'"{column}"' for column in copied_columns)
            connection.execute(
                f"INSERT INTO jobs_archive ({columns_sql}) "  # nosec B608 - trusted schema metadata
                f"SELECT {columns_sql} FROM jobs WHERE uuid = ?",  # nosec B608 - trusted schema metadata
                (submitted.job_uuid,),
            )
            connection.execute("DELETE FROM jobs WHERE uuid=?", (submitted.job_uuid,))
            compressed_payload = "gzip64:" + base64.b64encode(gzip.compress(stored_payload.encode("utf-8"))).decode(
                "ascii"
            )
            connection.execute(
                "UPDATE jobs_archive SET id=NULL, payload=NULL, payload_compressed=? WHERE uuid=?",
                (compressed_payload, submitted.job_uuid),
            )
    finally:
        connection.close()
    with slides.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE slides_generation_receipts SET receipt_status='claimed' WHERE id=?",
            (_RECEIPT_ID,),
        )

    replay = await _submit(_service(slides, jobs))
    assert replay.replayed is True
    assert replay.job_uuid == submitted.job_uuid
    rebound = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert rebound.receipt_status == "queued"
    assert rebound.job_uuid == submitted.job_uuid
    assert rebound.job_id == job["id"]
    archived = jobs.lookup_slides_generation_job(
        owner_user_id="owner-1",
        idempotency_key=rebound.jobs_idempotency_key,
        expected_job_uuid=submitted.job_uuid,
    )
    assert archived is not None
    assert archived["id"] is None
    assert archived["payload"] == {"receipt_id": _RECEIPT_ID}


@pytest.mark.parametrize("recovered_job", [False, True])
@pytest.mark.asyncio
async def test_post_enqueue_bind_storage_failure_retains_claim_and_returns_bounded_503(
    stores,
    monkeypatch: pytest.MonkeyPatch,
    recovered_job: bool,
):
    slides, jobs = stores
    module = _service_module()
    keyring, snapshot = _digest_material()
    if recovered_job:
        jobs.create_job(
            domain="slides",
            queue="default",
            job_type="presentation.generate",
            payload={"receipt_id": _RECEIPT_ID},
            owner_user_id="owner-1",
            idempotency_key=module.derive_jobs_idempotency_key(
                owner_user_id="owner-1",
                idempotency_key=_IDEMPOTENCY_KEY,
                keyring=keyring,
                digest_snapshot=snapshot,
            ),
        )

    def bind_failure(**_kwargs: Any):
        raise RuntimeError("source-bearing storage failure")

    monkeypatch.setattr(slides, "bind_generation_job", bind_failure)
    with pytest.raises(module.StandaloneHtmlGenerationError) as exc:
        await _submit(
            _service(
                slides,
                jobs,
                keyring=keyring,
                digest_snapshot=snapshot,
            )
        )
    assert (exc.value.code, exc.value.status_code, str(exc.value)) == (
        "generation_receipt_unresolved",
        503,
        "generation_receipt_unresolved",
    )
    assert exc.value.__cause__ is None
    receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert receipt.receipt_status == "claimed"
    assert receipt.job_uuid is None
    assert slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")
    retained_job = jobs.lookup_slides_generation_job(
        owner_user_id="owner-1",
        idempotency_key=module.derive_jobs_idempotency_key(
            owner_user_id="owner-1",
            idempotency_key=_IDEMPOTENCY_KEY,
            keyring=keyring,
            digest_snapshot=snapshot,
        ),
    )
    assert retained_job is not None
    assert retained_job["payload"] == {"receipt_id": _RECEIPT_ID}


@pytest.mark.asyncio
async def test_unbound_null_job_uuid_terminal_cas_deletes_input(
    stores,
    monkeypatch: pytest.MonkeyPatch,
):
    slides, jobs = stores

    def bind_failure(**_kwargs: Any):
        raise RuntimeError("source-bearing storage failure")

    monkeypatch.setattr(slides, "bind_generation_job", bind_failure)
    service = _service(slides, jobs)
    with pytest.raises(_service_module().StandaloneHtmlGenerationError):
        await _submit(service)
    receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert receipt.job_uuid is None
    assert service.terminalize(
        receipt=receipt,
        status="failed",
        error_code="generation_correlation_mismatch",
        error_message="Generation correlation failed.",
    )
    terminal = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert terminal.receipt_status == "failed"
    with pytest.raises(KeyError):
        slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")


@pytest.mark.parametrize("mutation", ["numeric_id", "uuid"])
@pytest.mark.asyncio
async def test_bound_job_rejects_numeric_id_or_uuid_mismatch_with_zero_provider_calls(
    stores,
    mutation: str,
):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    bad_job = dict(job)
    if mutation == "numeric_id":
        bad_job["id"] = int(job["id"]) + 1
    else:
        bad_job["uuid"] = str(uuid.uuid4())
    provider_calls = 0

    async def provider_generate(**_kwargs: Any) -> bytes:
        nonlocal provider_calls
        provider_calls += 1
        return _HTML

    outcome = await _process(slides, jobs, bad_job, provider_generate=provider_generate)
    assert outcome == WorkerTerminalOutcome(
        status="failed",
        error_code="generation_correlation_mismatch",
        message="Generation correlation failed.",
    )
    assert provider_calls == 0
    receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert receipt.receipt_status == "failed"
    with pytest.raises(KeyError):
        slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")


@pytest.mark.parametrize("lookup", ["other_owner", "missing"])
@pytest.mark.asyncio
async def test_worker_cross_owner_and_missing_receipts_are_indistinguishable(
    stores,
    lookup: str,
):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    bad_job = dict(job)
    if lookup == "other_owner":
        bad_job["owner_user_id"] = "other-owner"
    else:
        bad_job["payload"] = {"receipt_id": str(uuid.uuid4())}
    provider_calls = 0

    async def provider_generate(**_kwargs: Any) -> bytes:
        nonlocal provider_calls
        provider_calls += 1
        return _HTML

    module = _service_module()
    with pytest.raises(module.StandaloneHtmlGenerationError) as exc:
        await _process(slides, jobs, bad_job, provider_generate=provider_generate)
    assert (exc.value.code, exc.value.status_code, str(exc.value)) == (
        "generation_correlation_mismatch",
        409,
        "generation_correlation_mismatch",
    )
    assert provider_calls == 0


@pytest.mark.parametrize(
    ("column", "tampered_value"),
    [
        ("input_expires_at", (_FIXED_NOW + timedelta(days=2)).isoformat()),
        ("created_at", (_FIXED_NOW + timedelta(seconds=1)).isoformat()),
        ("input_expires_at", "2026-07-19T12:00:00"),
    ],
)
@pytest.mark.asyncio
async def test_verified_input_rejects_tampered_timestamps_before_egress(
    stores,
    column: str,
    tampered_value: str,
):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    with slides.transaction(immediate=True) as connection:
        connection.execute(
            f"UPDATE slides_generation_inputs SET {column} = ? WHERE receipt_id = ?",  # nosec B608 - closed parametrization
            (tampered_value, _RECEIPT_ID),
        )
    provider_calls = 0

    async def provider_generate(**_kwargs: Any) -> bytes:
        nonlocal provider_calls
        provider_calls += 1
        return _HTML

    outcome = await _process(slides, jobs, job, provider_generate=provider_generate)
    assert outcome == WorkerTerminalOutcome(
        status="failed",
        error_code="generation_correlation_mismatch",
        message="Generation correlation failed.",
    )
    assert provider_calls == 0
    with pytest.raises(KeyError):
        slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")


@pytest.mark.parametrize(
    "provenance_update",
    [
        {"source_kind": "chat"},
        {"source_snapshot_hmac_sha256": "f" * 64},
        {"source_bytes": 999},
        {"digest_key_id": "other-key"},
        {"model": "other-model"},
        {"prompt_sha256": "f" * 64},
        {"source_ref": "should-be-null-for-prompt"},
        {"extra": "not-closed"},
    ],
)
@pytest.mark.asyncio
async def test_verified_input_rejects_inconsistent_or_open_provenance_before_egress(
    stores,
    provenance_update: dict[str, Any],
):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    generation_input = slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")
    provenance = json.loads(generation_input.provenance_json)
    provenance.update(provenance_update)
    with slides.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE slides_generation_inputs SET provenance_json = ? WHERE receipt_id = ?",
            (json.dumps(provenance, sort_keys=True, separators=(",", ":")), _RECEIPT_ID),
        )
    provider_calls = 0

    async def provider_generate(**_kwargs: Any) -> bytes:
        nonlocal provider_calls
        provider_calls += 1
        return _HTML

    outcome = await _process(slides, jobs, job, provider_generate=provider_generate)
    assert outcome == WorkerTerminalOutcome(
        status="failed",
        error_code="generation_correlation_mismatch",
        message="Generation correlation failed.",
    )
    assert provider_calls == 0
    with pytest.raises(KeyError):
        slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")


@pytest.mark.asyncio
async def test_nonterminal_replay_verifies_retained_input_before_returning(stores):
    slides, jobs = stores
    await _submit(_service(slides, jobs))
    with slides.transaction(immediate=True) as connection:
        connection.execute(
            "UPDATE slides_generation_inputs SET source_text = ? WHERE receipt_id = ?",
            ("tampered source", _RECEIPT_ID),
        )

    def fail_config():
        raise AssertionError("replay must not load current configuration")

    async def fail_source(_source: dict[str, Any], _limits: Any):
        raise AssertionError("replay must not reread source")

    module = _service_module()
    with pytest.raises(module.StandaloneHtmlGenerationError) as exc:
        await _service(slides, jobs).submit(
            owner_user_id="owner-1",
            idempotency_key=_IDEMPOTENCY_KEY,
            request=_request(),
            config_loader=fail_config,
            source_resolver=fail_source,
        )
    assert exc.value.code == "generation_correlation_mismatch"


@pytest.mark.asyncio
async def test_worker_reconstructs_exact_bounded_user_content_and_delivery_style(stores):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    provider_kwargs: dict[str, Any] = {}
    pool = _ValidationPool(_validation())

    async def provider_generate(**kwargs: Any) -> bytes:
        provider_kwargs.update(kwargs)
        return _HTML

    await _process(
        slides,
        jobs,
        job,
        pool=pool,
        provider_generate=provider_generate,
    )
    assert provider_kwargs["system_prompt"] == _config().prompt.text
    assert provider_kwargs["user_content"] == json.dumps(
        {
            "schema_version": 1,
            "source": {"kind": "prompt", "text": "Exact café source"},
            "html_options": {
                "presentation_type": "tech-sharing",
                "audience": "backend engineers",
                "slide_count": 3,
                "visual_direction": "dark-technical",
                "delivery_style": "speaker-led",
            },
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    assert pool.reservation.validate_calls == [(_HTML, "speaker-led")]


@pytest.mark.asyncio
async def test_validator_saturation_retries_without_provider_or_input_deletion(stores):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    pool = _ValidationPool(
        _validation(),
        acquire_error=StandaloneHtmlValidationError(
            "standalone_html_validator_busy",
            status_code=503,
            retry_after=1,
        ),
    )
    provider_calls = 0

    async def provider_generate(**_kwargs: Any) -> bytes:
        nonlocal provider_calls
        provider_calls += 1
        return _HTML

    worker = _worker_module()
    with pytest.raises(worker.StandaloneHtmlGenerationRetry) as exc:
        await _process(
            slides,
            jobs,
            job,
            pool=pool,
            provider_generate=provider_generate,
        )
    assert exc.value.failure_code == "standalone_html_validator_busy"
    assert provider_calls == 0
    receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert receipt.receipt_status == "queued"
    assert slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")


@pytest.mark.asyncio
async def test_retryable_provider_failure_resets_receipt_and_retains_input(stores):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)

    async def provider_generate(**_kwargs: Any) -> bytes:
        raise StandaloneHtmlProviderError("standalone_html_provider_timeout")

    worker = _worker_module()
    with pytest.raises(worker.StandaloneHtmlGenerationRetry) as exc:
        await _process(slides, jobs, job, provider_generate=provider_generate)
    assert exc.value.failure_code == "standalone_html_provider_timeout"
    receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert receipt.receipt_status == "queued"
    assert receipt.error_code == "standalone_html_provider_timeout"
    assert slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")


@pytest.mark.asyncio
async def test_exhausted_budget_terminalizes_before_provider(stores):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    job = {**job, "retry_count": job["max_retries"]}
    provider_calls = 0

    async def provider_generate(**_kwargs: Any) -> bytes:
        nonlocal provider_calls
        provider_calls += 1
        return _HTML

    outcome = await _process(slides, jobs, job, provider_generate=provider_generate)
    assert outcome == WorkerTerminalOutcome(
        status="failed",
        error_code="generation_retry_exhausted",
        message="Generation retry budget was exhausted.",
    )
    assert provider_calls == 0
    receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert receipt.expires_at == (_FIXED_NOW + timedelta(minutes=1, days=30)).isoformat()
    with pytest.raises(KeyError):
        slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")


@pytest.mark.asyncio
async def test_poison_threshold_is_anticipated_before_worker_sdk_quarantine(
    stores,
    monkeypatch: pytest.MonkeyPatch,
):
    slides, jobs = stores
    monkeypatch.setenv("JOBS_QUARANTINE_THRESHOLD", "2")
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    job = {
        **job,
        "failure_streak_code": "standalone_html_provider_timeout",
        "failure_streak_count": 1,
    }

    async def provider_generate(**_kwargs: Any) -> bytes:
        raise StandaloneHtmlProviderError("standalone_html_provider_timeout")

    outcome = await _process(slides, jobs, job, provider_generate=provider_generate)
    assert outcome == WorkerTerminalOutcome(
        status="failed",
        error_code="generation_quarantined",
        message="Generation was quarantined.",
    )
    receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert receipt.receipt_status == "failed"
    with pytest.raises(KeyError):
        slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")


@pytest.mark.asyncio
async def test_nonretryable_validation_terminalizes_and_deletes_input(stores):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    pool = _ValidationPool(
        _validation(),
        validation_error=StandaloneHtmlValidationError(
            "standalone_html_invalid_document",
            status_code=422,
        ),
    )
    outcome = await _process(slides, jobs, job, pool=pool)
    assert outcome == WorkerTerminalOutcome(
        status="failed",
        error_code="standalone_html_invalid_document",
        message="Generated HTML did not pass validation.",
    )
    receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert receipt.receipt_status == "failed"
    assert receipt.expires_at == (_FIXED_NOW + timedelta(minutes=1, days=30)).isoformat()
    with pytest.raises(KeyError):
        slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")


@pytest.mark.asyncio
async def test_missing_digest_key_releases_without_retry_or_provider_burn(stores):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    keyring, snapshot = _digest_material()
    missing = replace(
        snapshot,
        availability=DigestKeyAvailability(missing_key_ids=("key-v1",)),
    )
    provider_calls = 0

    async def provider_generate(**_kwargs: Any) -> bytes:
        nonlocal provider_calls
        provider_calls += 1
        return _HTML

    worker = _worker_module()
    with pytest.raises(worker.StandaloneHtmlGenerationRetry) as exc:
        await _process(
            slides,
            jobs,
            job,
            keyring=keyring,
            digest_snapshot=missing,
            provider_generate=provider_generate,
        )
    assert exc.value.failure_code == "generation_digest_key_unavailable"
    assert provider_calls == 0
    stored_job = jobs.get_job(int(job["id"]))
    assert stored_job["status"] == "queued"
    assert stored_job["retry_count"] == job["retry_count"]
    assert slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")
    guard = worker.make_generation_acquire_guard(_digest_snapshot_loader(missing))
    assert await guard(job) is False


@pytest.mark.parametrize("loss_stage", ["pre_provider", "post_provider", "pre_commit"])
@pytest.mark.asyncio
async def test_live_digest_key_loss_releases_and_discards_uncommitted_output(
    stores,
    loss_stage: str,
):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    keyring, ready = _digest_material()
    missing = replace(
        ready,
        availability=DigestKeyAvailability(missing_key_ids=("key-v1",)),
    )
    state = {"snapshot": ready}

    async def digest_snapshot_loader():
        return state["snapshot"]

    pool = _ValidationPool(
        _validation(),
        on_acquire=((lambda: state.update(snapshot=missing)) if loss_stage == "pre_provider" else None),
        on_validate=((lambda: state.update(snapshot=missing)) if loss_stage == "pre_commit" else None),
    )
    provider_calls = 0

    async def provider_generate(**_kwargs: Any) -> bytes:
        nonlocal provider_calls
        provider_calls += 1
        if loss_stage == "post_provider":
            state["snapshot"] = missing
        return _HTML

    worker = _worker_module()
    with pytest.raises(worker.StandaloneHtmlGenerationRetry) as exc:
        await _process(
            slides,
            jobs,
            job,
            pool=pool,
            keyring=keyring,
            digest_snapshot_loader=digest_snapshot_loader,
            provider_generate=provider_generate,
        )
    assert exc.value.failure_code == "generation_digest_key_unavailable"
    assert provider_calls == (0 if loss_stage == "pre_provider" else 1)
    assert len(pool.reservation.validate_calls) == (1 if loss_stage == "pre_commit" else 0)
    receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert receipt.receipt_status == "queued"
    assert slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")
    with pytest.raises(KeyError):
        slides.get_presentation_by_id(_RECEIPT_ID)
    stored_job = jobs.get_job(int(job["id"]))
    assert stored_job["status"] == "queued"
    assert stored_job["retry_count"] == job["retry_count"]


@pytest.mark.parametrize("loader_failure", ["raises", "invalid"])
@pytest.mark.asyncio
async def test_digest_snapshot_loader_failures_close_guard_and_handler(
    stores,
    loader_failure: str,
):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)

    async def digest_snapshot_loader():
        if loader_failure == "raises":
            raise RuntimeError("source-bearing registry failure")
        return object()

    worker = _worker_module()
    guard = worker.make_generation_acquire_guard(digest_snapshot_loader)
    assert await guard(job) is False
    with pytest.raises(worker.StandaloneHtmlGenerationRetry) as exc:
        await _process(
            slides,
            jobs,
            job,
            digest_snapshot_loader=digest_snapshot_loader,
        )
    assert exc.value.failure_code == "generation_digest_key_unavailable"
    assert slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")
    assert jobs.get_job(int(job["id"]))["status"] == "queued"


@pytest.mark.asyncio
async def test_input_expiry_is_rechecked_after_reservation_wait_before_egress(stores):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    clock = {"value": _FIXED_NOW + timedelta(minutes=1)}
    pool = _ValidationPool(
        _validation(),
        on_acquire=lambda: clock.update(value=_FIXED_NOW + timedelta(hours=24)),
    )
    provider_calls = 0

    async def provider_generate(**_kwargs: Any) -> bytes:
        nonlocal provider_calls
        provider_calls += 1
        return _HTML

    outcome = await _process(
        slides,
        jobs,
        job,
        pool=pool,
        provider_generate=provider_generate,
        now=lambda: clock["value"],
    )
    assert outcome == WorkerTerminalOutcome(
        status="failed",
        error_code="generation_expired",
        message="Generation input expired.",
    )
    assert provider_calls == 0
    receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert receipt.updated_at == (_FIXED_NOW + timedelta(hours=24)).isoformat()
    assert receipt.expires_at == (_FIXED_NOW + timedelta(days=31)).isoformat()


@pytest.mark.asyncio
async def test_input_expiry_during_pre_provider_digest_load_blocks_egress(stores):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    _keyring, ready = _digest_material()
    input_deadline = _FIXED_NOW + timedelta(hours=24)
    clock = {"value": _FIXED_NOW + timedelta(minutes=1)}
    snapshot_loads = 0
    provider_calls = 0

    async def digest_snapshot_loader():
        nonlocal snapshot_loads
        snapshot_loads += 1
        if snapshot_loads == 2:
            await asyncio.sleep(0)
            clock["value"] = input_deadline
        return ready

    async def provider_generate(**_kwargs: Any) -> bytes:
        nonlocal provider_calls
        provider_calls += 1
        return _HTML

    outcome = await _process(
        slides,
        jobs,
        job,
        digest_snapshot_loader=digest_snapshot_loader,
        provider_generate=provider_generate,
        now=lambda: clock["value"],
    )

    assert outcome == WorkerTerminalOutcome(
        status="failed",
        error_code="generation_expired",
        message="Generation input expired.",
    )
    assert snapshot_loads == 2
    assert provider_calls == 0
    with pytest.raises(KeyError):
        slides.get_presentation_by_id(_RECEIPT_ID)


@pytest.mark.asyncio
async def test_input_expiry_during_pre_provider_jobs_lookup_blocks_egress(
    stores,
    monkeypatch: pytest.MonkeyPatch,
):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    input_deadline = _FIXED_NOW + timedelta(hours=24)
    clock = {"value": _FIXED_NOW + timedelta(minutes=1)}
    _update_job(
        jobs,
        int(job["id"]),
        "leased_until = ?",
        ((input_deadline + timedelta(hours=1)).isoformat(),),
    )
    original_get_job_by_uuid = jobs.get_job_by_uuid
    lookups = 0
    provider_calls = 0

    def delayed_get_job_by_uuid(job_uuid: str):
        nonlocal lookups
        candidate = original_get_job_by_uuid(job_uuid)
        lookups += 1
        if lookups == 1:
            clock["value"] = input_deadline
        return candidate

    async def provider_generate(**_kwargs: Any) -> bytes:
        nonlocal provider_calls
        provider_calls += 1
        return _HTML

    monkeypatch.setattr(jobs, "get_job_by_uuid", delayed_get_job_by_uuid)
    outcome = await _process(
        slides,
        jobs,
        job,
        provider_generate=provider_generate,
        now=lambda: clock["value"],
    )

    assert outcome == WorkerTerminalOutcome(
        status="failed",
        error_code="generation_expired",
        message="Generation input expired.",
    )
    assert lookups == 1
    assert provider_calls == 0
    with pytest.raises(KeyError):
        slides.get_presentation_by_id(_RECEIPT_ID)


@pytest.mark.asyncio
async def test_changed_default_target_does_not_replace_allowed_stored_target(stores):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    stored = _config().target
    changed_default = replace(stored, model="gpt-new-default")
    current = replace(
        _config(),
        target=changed_default,
        allowed_targets=(stored, changed_default),
    )
    provider_calls = 0

    async def provider_generate(**kwargs: Any) -> bytes:
        nonlocal provider_calls
        provider_calls += 1
        assert kwargs["stored_target"] == stored
        return _HTML

    result = await _process(
        slides,
        jobs,
        job,
        current_config_loader=lambda: current,
        provider_generate=provider_generate,
    )
    assert result["content_kind"] == "standalone_html"
    assert provider_calls == 1


@pytest.mark.parametrize(
    ("allowed_target", "expected_code"),
    [
        (
            replace(_config().target, model="other-model"),
            "standalone_html_model_not_allowed",
        ),
        (
            replace(
                _config().target,
                endpoint_identity="https://api.openai.com:443/v2/chat/completions",
            ),
            "standalone_html_endpoint_not_allowed",
        ),
    ],
)
@pytest.mark.asyncio
async def test_worker_distinguishes_stored_model_and_endpoint_allowlist_removal(
    stores,
    allowed_target: ResolvedExecutionTarget,
    expected_code: str,
):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    current = replace(
        _config(),
        target=allowed_target,
        allowed_targets=(allowed_target,),
    )
    provider_calls = 0

    async def provider_generate(**_kwargs: Any) -> bytes:
        nonlocal provider_calls
        provider_calls += 1
        return _HTML

    outcome = await _process(
        slides,
        jobs,
        job,
        current_config_loader=lambda: current,
        provider_generate=provider_generate,
    )
    assert outcome.status == "failed"
    assert outcome.error_code == expected_code
    assert provider_calls == 0


@pytest.mark.parametrize("disabled_field", ["feature_enabled", "egress_enabled"])
@pytest.mark.asyncio
async def test_worker_kill_controls_fail_closed_before_provider(
    stores,
    disabled_field: str,
):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    current = replace(_config(), **{disabled_field: False})
    provider_calls = 0

    async def provider_generate(**_kwargs: Any) -> bytes:
        nonlocal provider_calls
        provider_calls += 1
        return _HTML

    outcome = await _process(
        slides,
        jobs,
        job,
        current_config_loader=lambda: current,
        provider_generate=provider_generate,
    )
    assert outcome.error_code == "standalone_html_egress_disabled"
    assert provider_calls == 0


@pytest.mark.asyncio
async def test_cancel_requested_before_provider_is_fenced_with_zero_egress(stores):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    _update_job(
        jobs,
        int(job["id"]),
        "cancel_requested_at = ?",
        ((_FIXED_NOW + timedelta(seconds=1)).isoformat(),),
    )
    provider_calls = 0

    async def provider_generate(**_kwargs: Any) -> bytes:
        nonlocal provider_calls
        provider_calls += 1
        return _HTML

    outcome = await _process(slides, jobs, job, provider_generate=provider_generate)
    assert outcome == WorkerTerminalOutcome(
        status="cancelled",
        error_code="generation_cancelled",
        message="Generation was cancelled.",
    )
    assert provider_calls == 0


@pytest.mark.asyncio
async def test_cancel_requested_after_provider_discards_late_result(stores):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)

    async def provider_generate(**_kwargs: Any) -> bytes:
        _update_job(
            jobs,
            int(job["id"]),
            "cancel_requested_at = ?",
            ((_FIXED_NOW + timedelta(seconds=2)).isoformat(),),
        )
        return _HTML

    outcome = await _process(slides, jobs, job, provider_generate=provider_generate)
    assert outcome.status == "cancelled"
    assert outcome.error_code == "generation_cancelled"
    with pytest.raises(KeyError):
        slides.get_presentation_by_id(_RECEIPT_ID)
    with pytest.raises(KeyError):
        slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")


@pytest.mark.asyncio
async def test_cancel_requested_during_precommit_digest_load_is_fenced(stores):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    keyring, ready = _digest_material()
    snapshot_loads = 0

    async def digest_snapshot_loader():
        nonlocal snapshot_loads
        snapshot_loads += 1
        if snapshot_loads == 4:
            await asyncio.sleep(0)
            _update_job(
                jobs,
                int(job["id"]),
                "cancel_requested_at = ?",
                ((_FIXED_NOW + timedelta(seconds=3)).isoformat(),),
            )
        return ready

    outcome = await _process(
        slides,
        jobs,
        job,
        keyring=keyring,
        digest_snapshot_loader=digest_snapshot_loader,
    )
    assert snapshot_loads == 4
    assert outcome == WorkerTerminalOutcome(
        status="cancelled",
        error_code="generation_cancelled",
        message="Generation was cancelled.",
    )
    assert (
        slides.get_generation_receipt(
            _RECEIPT_ID,
            owner_user_id="owner-1",
        ).receipt_status
        == "cancelled"
    )
    with pytest.raises(KeyError):
        slides.get_presentation_by_id(_RECEIPT_ID)
    with pytest.raises(KeyError):
        slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")


@pytest.mark.asyncio
async def test_lease_loss_during_precommit_digest_load_is_fenced(stores):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    keyring, ready = _digest_material()
    snapshot_loads = 0

    async def digest_snapshot_loader():
        nonlocal snapshot_loads
        snapshot_loads += 1
        if snapshot_loads == 4:
            await asyncio.sleep(0)
            _update_job(
                jobs,
                int(job["id"]),
                "leased_until = ?",
                ((_FIXED_NOW - timedelta(seconds=1)).isoformat(),),
            )
        return ready

    worker = _worker_module()
    with pytest.raises(worker.StandaloneHtmlGenerationRetry) as exc:
        await _process(
            slides,
            jobs,
            job,
            keyring=keyring,
            digest_snapshot_loader=digest_snapshot_loader,
        )
    assert exc.value.failure_code == "generation_job_state_changed"
    assert snapshot_loads == 4
    assert (
        slides.get_generation_receipt(
            _RECEIPT_ID,
            owner_user_id="owner-1",
        ).receipt_status
        == "queued"
    )
    assert slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")
    with pytest.raises(KeyError):
        slides.get_presentation_by_id(_RECEIPT_ID)


@pytest.mark.asyncio
async def test_lease_expiry_during_final_jobs_lookup_is_fenced(
    stores,
    monkeypatch: pytest.MonkeyPatch,
):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    clock = {"value": _FIXED_NOW + timedelta(minutes=1)}
    lease_deadline = _FIXED_NOW + timedelta(minutes=2)
    _update_job(
        jobs,
        int(job["id"]),
        "leased_until = ?",
        (lease_deadline.isoformat(),),
    )
    original_get_job_by_uuid = jobs.get_job_by_uuid
    lookups = 0
    provider_calls = 0

    def delayed_get_job_by_uuid(job_uuid: str):
        nonlocal lookups
        candidate = original_get_job_by_uuid(job_uuid)
        lookups += 1
        if lookups == 2:
            clock["value"] = lease_deadline
        return candidate

    async def provider_generate(**_kwargs: Any) -> bytes:
        nonlocal provider_calls
        provider_calls += 1
        return _HTML

    monkeypatch.setattr(jobs, "get_job_by_uuid", delayed_get_job_by_uuid)
    worker = _worker_module()
    with pytest.raises(worker.StandaloneHtmlGenerationRetry) as exc:
        await _process(
            slides,
            jobs,
            job,
            provider_generate=provider_generate,
            now=lambda: clock["value"],
        )

    assert exc.value.failure_code == "generation_job_state_changed"
    assert lookups == 2
    assert provider_calls == 1
    assert (
        slides.get_generation_receipt(
            _RECEIPT_ID,
            owner_user_id="owner-1",
        ).receipt_status
        == "queued"
    )
    assert slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")
    with pytest.raises(KeyError):
        slides.get_presentation_by_id(_RECEIPT_ID)


@pytest.mark.parametrize(
    ("final_status", "expected"),
    [
        ("queued", "retry"),
        ("failed", "terminal"),
        ("quarantined", "terminal"),
    ],
)
@pytest.mark.asyncio
async def test_final_job_state_change_discards_output_without_commit(
    stores,
    final_status: str,
    expected: str,
):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    pool = _ValidationPool(
        _validation(),
        on_validate=lambda: _update_job(
            jobs,
            int(job["id"]),
            "status = ?, leased_until = NULL, worker_id = NULL, lease_id = NULL",
            (final_status,),
        ),
    )
    worker = _worker_module()
    if expected == "retry":
        with pytest.raises(worker.StandaloneHtmlGenerationRetry) as exc:
            await _process(slides, jobs, job, pool=pool)
        assert exc.value.failure_code == "generation_job_state_changed"
        assert slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")
    else:
        outcome = await _process(slides, jobs, job, pool=pool)
        assert outcome.status == "failed"
        assert outcome.error_code in {
            "generation_job_terminal",
            "generation_quarantined",
        }
        with pytest.raises(KeyError):
            slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")
    with pytest.raises(KeyError):
        slides.get_presentation_by_id(_RECEIPT_ID)


@pytest.mark.parametrize(
    ("assignment", "parameters", "expected_code", "terminal"),
    [
        (
            "leased_until = ?",
            ((_FIXED_NOW - timedelta(seconds=1)).isoformat(),),
            "generation_job_state_changed",
            False,
        ),
        ("worker_id = ?", ("other-worker",), "generation_job_state_changed", False),
        ("lease_id = ?", (str(uuid.uuid4()),), "generation_job_state_changed", False),
        (
            "payload = ?",
            (json.dumps({"receipt_id": str(uuid.uuid4())}),),
            "generation_correlation_mismatch",
            True,
        ),
    ],
)
@pytest.mark.asyncio
async def test_final_lease_worker_and_payload_mutations_are_fenced(
    stores,
    assignment: str,
    parameters: tuple[Any, ...],
    expected_code: str,
    terminal: bool,
):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    pool = _ValidationPool(
        _validation(),
        on_validate=lambda: _update_job(
            jobs,
            int(job["id"]),
            assignment,
            parameters,
        ),
    )
    worker = _worker_module()
    if terminal:
        outcome = await _process(slides, jobs, job, pool=pool)
        assert outcome.status == "failed"
        assert outcome.error_code == expected_code
        with pytest.raises(KeyError):
            slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")
    else:
        with pytest.raises(worker.StandaloneHtmlGenerationRetry) as exc:
            await _process(slides, jobs, job, pool=pool)
        assert exc.value.failure_code == expected_code
        assert slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")
    with pytest.raises(KeyError):
        slides.get_presentation_by_id(_RECEIPT_ID)


@pytest.mark.asyncio
async def test_final_job_payload_is_compared_after_jobs_normalization(stores, monkeypatch):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    real_get = jobs.get_job_by_uuid

    def serialized_get(job_uuid: str):
        row = real_get(job_uuid)
        assert row is not None
        return {**row, "payload": json.dumps(row["payload"])}

    monkeypatch.setattr(jobs, "get_job_by_uuid", serialized_get)
    result = await _process(slides, jobs, job)
    assert result["presentation_id"] == _RECEIPT_ID


@pytest.mark.asyncio
async def test_commit_before_jobs_completion_is_idempotent_without_second_provider(stores):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    first_calls = 0

    async def first_provider(**_kwargs: Any) -> bytes:
        nonlocal first_calls
        first_calls += 1
        return _HTML

    first = await _process(slides, jobs, job, provider_generate=first_provider)
    second_calls = 0

    async def second_provider(**_kwargs: Any) -> bytes:
        nonlocal second_calls
        second_calls += 1
        return _HTML

    second = await _process(slides, jobs, job, provider_generate=second_provider)
    assert first == second
    assert first_calls == 1
    assert second_calls == 0
    count = (
        slides.get_connection()
        .execute(
            "SELECT COUNT(*) FROM presentations WHERE generation_job_uuid = ?",
            (job["uuid"],),
        )
        .fetchone()[0]
    )
    assert count == 1


@pytest.mark.asyncio
async def test_completed_presentation_precedes_later_jobs_cancellation(stores):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    completed = await _process(slides, jobs, job)
    cancelled_job = {
        **job,
        "status": "cancelled",
        "cancelled_at": (_FIXED_NOW + timedelta(minutes=2)).isoformat(),
    }
    provider_calls = 0

    async def provider_generate(**_kwargs: Any) -> bytes:
        nonlocal provider_calls
        provider_calls += 1
        return _HTML

    recovered = await _process(
        slides,
        jobs,
        cancelled_job,
        provider_generate=provider_generate,
    )
    assert recovered == completed
    assert provider_calls == 0


@pytest.mark.asyncio
async def test_cancellation_winning_commit_cas_returns_cancelled_not_raw_conflict(
    stores,
    monkeypatch: pytest.MonkeyPatch,
):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    original_commit = slides.commit_generation_presentation

    def cancellation_wins(**kwargs: Any):
        receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
        assert _service(slides, jobs).terminalize(
            receipt=receipt,
            status="cancelled",
            error_code="generation_cancelled",
            error_message="Generation was cancelled.",
            terminal_at=_FIXED_NOW + timedelta(minutes=1),
        )
        return original_commit(**kwargs)

    monkeypatch.setattr(slides, "commit_generation_presentation", cancellation_wins)
    outcome = await _process(slides, jobs, job)
    assert outcome.status == "cancelled"
    assert outcome.error_code == "generation_cancelled"
    with pytest.raises(KeyError):
        slides.get_presentation_by_id(_RECEIPT_ID)


@pytest.mark.asyncio
async def test_commit_transaction_rechecks_expiry_and_terminalizes_at_logical_deadline(
    stores,
    monkeypatch: pytest.MonkeyPatch,
):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    clock = {"value": _FIXED_NOW + timedelta(minutes=1)}
    input_deadline = _FIXED_NOW + timedelta(hours=24)
    original_commit = slides.commit_generation_presentation
    provider_calls = 0

    def expiry_wins_inside_commit(**kwargs: Any):
        original_now = kwargs["now"]

        def advance_after_begin() -> datetime:
            assert slides.get_connection().in_transaction
            clock["value"] = input_deadline
            return original_now()

        kwargs["now"] = advance_after_begin
        return original_commit(**kwargs)

    async def provider_generate(**_kwargs: Any) -> bytes:
        nonlocal provider_calls
        provider_calls += 1
        return _HTML

    monkeypatch.setattr(slides, "commit_generation_presentation", expiry_wins_inside_commit)
    outcome = await _process(
        slides,
        jobs,
        job,
        provider_generate=provider_generate,
        now=lambda: clock["value"],
    )
    assert outcome == WorkerTerminalOutcome(
        status="failed",
        error_code="generation_expired",
        message="Generation input expired.",
    )
    assert provider_calls == 1
    receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert receipt.receipt_status == "failed"
    assert receipt.updated_at == input_deadline.isoformat()
    assert receipt.expires_at == (input_deadline + timedelta(days=30)).isoformat()
    with pytest.raises(KeyError):
        slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")
    with pytest.raises(KeyError):
        slides.get_presentation_by_id(_RECEIPT_ID)


@pytest.mark.parametrize("tampered_column", ["created_at", "input_expires_at"])
@pytest.mark.asyncio
async def test_commit_transaction_independently_rejects_concurrent_input_time_tamper(
    stores,
    monkeypatch: pytest.MonkeyPatch,
    tampered_column: str,
):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    original_commit = slides.commit_generation_presentation
    tampered_value = (
        (_FIXED_NOW + timedelta(seconds=1)).isoformat()
        if tampered_column == "created_at"
        else (_FIXED_NOW + timedelta(hours=48)).isoformat()
    )

    def tamper_after_service_verification(**kwargs: Any):
        with slides.transaction(immediate=True) as connection:
            connection.execute(
                f"UPDATE slides_generation_inputs SET {tampered_column} = ? WHERE receipt_id = ?",  # nosec B608 - closed parametrization
                (tampered_value, _RECEIPT_ID),
            )
        return original_commit(**kwargs)

    monkeypatch.setattr(slides, "commit_generation_presentation", tamper_after_service_verification)
    outcome = await _process(slides, jobs, job)
    assert outcome == WorkerTerminalOutcome(
        status="failed",
        error_code="generation_correlation_mismatch",
        message="Generation correlation failed.",
    )
    receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert receipt.receipt_status == "failed"
    with pytest.raises(KeyError):
        slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")
    with pytest.raises(KeyError):
        slides.get_presentation_by_id(_RECEIPT_ID)


@pytest.mark.parametrize(
    ("provider_code", "quarantine", "expected_code"),
    [
        ("standalone_html_provider_response_invalid", False, "standalone_html_provider_response_invalid"),
        ("standalone_html_provider_timeout", True, "generation_quarantined"),
    ],
)
@pytest.mark.asyncio
async def test_provider_terminal_retention_starts_when_failure_occurs(
    stores,
    monkeypatch: pytest.MonkeyPatch,
    provider_code: str,
    quarantine: bool,
    expected_code: str,
):
    slides, jobs = stores
    monkeypatch.setenv("JOBS_QUARANTINE_THRESHOLD", "2")
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    if quarantine:
        job = {
            **job,
            "failure_streak_code": provider_code,
            "failure_streak_count": 1,
        }
    clock = {"value": _FIXED_NOW + timedelta(minutes=1)}
    terminal_time = _FIXED_NOW + timedelta(hours=2)

    async def provider_generate(**_kwargs: Any) -> bytes:
        clock["value"] = terminal_time
        raise StandaloneHtmlProviderError(provider_code)

    outcome = await _process(
        slides,
        jobs,
        job,
        provider_generate=provider_generate,
        now=lambda: clock["value"],
    )
    assert outcome.status == "failed"
    assert outcome.error_code == expected_code
    receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert receipt.updated_at == terminal_time.isoformat()
    assert receipt.expires_at == (terminal_time + timedelta(days=30)).isoformat()


@pytest.mark.asyncio
async def test_lost_terminal_cas_reloads_completed_winner(stores, monkeypatch):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    original_terminalize = slides.terminalize_generation_receipt

    def completed_wins(**_kwargs: Any) -> bool:
        generation_input = slides.get_generation_input(
            _RECEIPT_ID,
            owner_user_id="owner-1",
        )
        slides.commit_generation_presentation(
            receipt_id=_RECEIPT_ID,
            owner_user_id="owner-1",
            job_uuid=job["uuid"],
            html_document=_HTML,
            validation_result=_validation(),
            generation_provenance_json=generation_input.provenance_json,
            committed_at=(_FIXED_NOW + timedelta(minutes=1)).isoformat(),
            expires_at=(_FIXED_NOW + timedelta(days=30, minutes=1)).isoformat(),
        )
        return False

    monkeypatch.setattr(slides, "terminalize_generation_receipt", completed_wins)
    _update_job(
        jobs,
        int(job["id"]),
        "cancel_requested_at = ?",
        ((_FIXED_NOW + timedelta(seconds=1)).isoformat(),),
    )
    result = await _process(slides, jobs, job)
    assert result["presentation_id"] == _RECEIPT_ID
    monkeypatch.setattr(slides, "terminalize_generation_receipt", original_terminalize)


@pytest.mark.parametrize("winner_status", ["completed", "failed"])
@pytest.mark.asyncio
async def test_lost_retry_reset_cas_reloads_completed_or_terminal_winner(
    stores,
    monkeypatch: pytest.MonkeyPatch,
    winner_status: str,
):
    slides, jobs = stores
    _submitted, job = await _submitted_and_acquired(slides, jobs)
    generation_input = slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")

    def winner_takes_reset(**_kwargs: Any) -> bool:
        if winner_status == "completed":
            slides.commit_generation_presentation(
                receipt_id=_RECEIPT_ID,
                owner_user_id="owner-1",
                job_uuid=job["uuid"],
                html_document=_HTML,
                validation_result=_validation(),
                generation_provenance_json=generation_input.provenance_json,
                committed_at=(_FIXED_NOW + timedelta(minutes=2)).isoformat(),
                expires_at=(_FIXED_NOW + timedelta(days=30, minutes=2)).isoformat(),
            )
        else:
            current = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
            assert _service(slides, jobs).terminalize(
                receipt=current,
                status="failed",
                error_code="generation_failed_elsewhere",
                error_message="Generation failed.",
                terminal_at=_FIXED_NOW + timedelta(minutes=2),
            )
        return False

    monkeypatch.setattr(slides, "reset_generation_receipt_queued", winner_takes_reset)

    async def provider_generate(**_kwargs: Any) -> bytes:
        raise StandaloneHtmlProviderError("standalone_html_provider_timeout")

    outcome = await _process(
        slides,
        jobs,
        job,
        provider_generate=provider_generate,
    )
    if winner_status == "completed":
        assert outcome["presentation_id"] == _RECEIPT_ID
    else:
        assert outcome == WorkerTerminalOutcome(
            status="failed",
            error_code="generation_failed_elsewhere",
            message="Generation failed.",
        )


async def _run_one_worker_sdk_job(
    slides: SlidesDatabase,
    jobs: JobManager,
    *,
    provider_generate: Any,
    provider_api_key_loader: Any | None = None,
) -> None:
    sdk = WorkerSDK(
        jobs,
        WorkerConfig(
            domain="slides",
            queue="default",
            worker_id="slides-worker-sdk",
            lease_seconds=600,
            renew_threshold_seconds=10,
            renew_jitter_seconds=0,
            retry_backoff_seconds=0,
        ),
    )

    async def handler(job: dict[str, Any]):
        try:
            return await _process(
                slides,
                jobs,
                job,
                provider_generate=provider_generate,
                provider_api_key_loader=provider_api_key_loader,
            )
        finally:
            sdk.stop()

    await asyncio.wait_for(
        sdk.run(handler=handler, job_type="presentation.generate"),
        timeout=2,
    )


@pytest.mark.asyncio
async def test_real_worker_sdk_retryable_failure_succeeds_on_next_attempt(stores):
    slides, jobs = stores
    submitted = await _submit(_service(slides, jobs))
    provider_calls = 0

    async def provider_generate(**_kwargs: Any) -> bytes:
        nonlocal provider_calls
        provider_calls += 1
        if provider_calls == 1:
            raise StandaloneHtmlProviderError("standalone_html_provider_timeout")
        return _HTML

    await _run_one_worker_sdk_job(
        slides,
        jobs,
        provider_generate=provider_generate,
    )
    retry_job = jobs.get_job_by_uuid(submitted.job_uuid)
    assert retry_job is not None
    assert retry_job["status"] == "queued"
    retry_receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert retry_receipt.receipt_status == "queued"
    assert slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")

    await _run_one_worker_sdk_job(
        slides,
        jobs,
        provider_generate=provider_generate,
    )

    assert provider_calls == 2
    completed_job = jobs.get_job_by_uuid(submitted.job_uuid)
    assert completed_job is not None
    assert completed_job["status"] == "completed"
    completed_receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert completed_receipt.receipt_status == "completed"
    assert completed_receipt.presentation_id == _RECEIPT_ID
    assert slides.get_presentation_by_id(_RECEIPT_ID).generation_job_uuid == submitted.job_uuid
    with pytest.raises(KeyError):
        slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")


@pytest.mark.parametrize("terminal_first", ["cancelled", "failed"])
@pytest.mark.asyncio
async def test_real_worker_sdk_observes_terminal_first_job_without_second_finalize(
    stores,
    monkeypatch: pytest.MonkeyPatch,
    terminal_first: str,
):
    slides, jobs = stores
    submitted = await _submit(_service(slides, jobs))
    calls = {"terminalize": 0, "complete": 0}
    original_terminalize = jobs.terminalize_job_from_worker
    original_complete = jobs.complete_job

    def record_terminalize(**kwargs: Any):
        calls["terminalize"] += 1
        return original_terminalize(**kwargs)

    def record_complete(*args: Any, **kwargs: Any):
        calls["complete"] += 1
        return original_complete(*args, **kwargs)

    monkeypatch.setattr(jobs, "terminalize_job_from_worker", record_terminalize)
    monkeypatch.setattr(jobs, "complete_job", record_complete)

    async def provider_generate(**_kwargs: Any) -> bytes:
        live = jobs.get_job_by_uuid(submitted.job_uuid)
        assert live is not None
        if terminal_first == "cancelled":
            assert jobs.cancel_job(int(live["id"]), reason="terminal-first")
        else:
            assert jobs.fail_job(
                int(live["id"]),
                error="terminal-first",
                retryable=False,
                worker_id=str(live["worker_id"]),
                lease_id=str(live["lease_id"]),
                enforce=True,
                error_code="terminal_first_failure",
                error_class="TerminalFirstFailure",
            )
        return _HTML

    await _run_one_worker_sdk_job(
        slides,
        jobs,
        provider_generate=provider_generate,
    )

    assert calls == {"terminalize": 0, "complete": 0}
    terminal_job = jobs.get_job_by_uuid(submitted.job_uuid)
    assert terminal_job is not None
    assert terminal_job["status"] == terminal_first
    receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert receipt.receipt_status == terminal_first
    with pytest.raises(KeyError):
        slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")
    with pytest.raises(KeyError):
        slides.get_presentation_by_id(_RECEIPT_ID)


@pytest.mark.asyncio
async def test_real_worker_sdk_rejects_completed_first_job_as_terminal_conflict(stores):
    slides, jobs = stores
    submitted = await _submit(_service(slides, jobs))

    async def provider_generate(**_kwargs: Any) -> bytes:
        live = jobs.get_job_by_uuid(submitted.job_uuid)
        assert live is not None
        lease_id = str(live["lease_id"])
        assert jobs.complete_job(
            int(live["id"]),
            result={"winner": "completed"},
            worker_id=str(live["worker_id"]),
            lease_id=lease_id,
            completion_token=lease_id,
            enforce=True,
        )
        return _HTML

    with pytest.raises(WorkerTerminalizationConflict):
        await _run_one_worker_sdk_job(
            slides,
            jobs,
            provider_generate=provider_generate,
        )

    completed_job = jobs.get_job_by_uuid(submitted.job_uuid)
    assert completed_job is not None
    assert completed_job["status"] == "completed"
    receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert receipt.receipt_status == "failed"
    assert receipt.error_code == "generation_job_terminal"
    with pytest.raises(KeyError):
        slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")


@pytest.mark.parametrize(
    ("failure_phase", "expected_code"),
    [
        ("api_key", "standalone_html_provider_credentials_unavailable"),
        ("provider", "standalone_html_provider_unavailable"),
    ],
)
@pytest.mark.asyncio
async def test_unexpected_provider_boundaries_redact_exception_logs_and_jobs_state(
    stores,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure_phase: str,
    expected_code: str,
):
    slides, jobs = stores
    monkeypatch.setenv("JOBS_QUARANTINE_THRESHOLD", "99")
    submitted = await _submit(_service(slides, jobs))
    marker = "Exact café source::provider-secret::raw-html"

    def api_key_loader(_target: ResolvedExecutionTarget) -> None:
        if failure_phase == "api_key":
            raise RuntimeError(marker)
        return None

    async def provider_generate(**_kwargs: Any) -> bytes:
        if failure_phase == "provider":
            raise RuntimeError(marker)
        return _HTML

    await _run_one_worker_sdk_job(
        slides,
        jobs,
        provider_generate=provider_generate,
        provider_api_key_loader=api_key_loader,
    )
    job = jobs.get_job_by_uuid(submitted.job_uuid)
    assert job is not None
    assert job["status"] == "queued"
    assert job["error_code"] == expected_code
    assert marker not in str(job.get("last_error"))
    assert marker not in str(job.get("error_message"))
    receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert receipt.receipt_status == "queued"
    assert receipt.error_code == expected_code
    assert marker not in str(receipt.error_message)
    captured = capsys.readouterr()
    assert marker not in captured.out
    assert marker not in captured.err


@pytest.mark.parametrize("terminal_mode", ["quarantine", "exhausted"])
@pytest.mark.asyncio
async def test_real_worker_sdk_terminal_cleanup_is_safe_and_source_free(
    stores,
    monkeypatch: pytest.MonkeyPatch,
    terminal_mode: str,
):
    slides, jobs = stores
    monkeypatch.setenv("JOBS_QUARANTINE_THRESHOLD", "2")
    submitted = await _submit(_service(slides, jobs))
    job = jobs.get_job_by_uuid(submitted.job_uuid)
    assert job is not None
    if terminal_mode == "quarantine":
        _update_job(
            jobs,
            int(job["id"]),
            "failure_streak_code = ?, failure_streak_count = ?",
            ("standalone_html_provider_timeout", 1),
        )
    else:
        _update_job(
            jobs,
            int(job["id"]),
            "retry_count = max_retries",
        )
    provider_calls = 0

    async def provider_generate(**_kwargs: Any) -> bytes:
        nonlocal provider_calls
        provider_calls += 1
        if terminal_mode == "quarantine":
            raise StandaloneHtmlProviderError("standalone_html_provider_timeout")
        return _HTML

    await _run_one_worker_sdk_job(
        slides,
        jobs,
        provider_generate=provider_generate,
    )
    terminal_job = jobs.get_job_by_uuid(submitted.job_uuid)
    assert terminal_job is not None
    assert terminal_job["status"] == "failed"
    assert terminal_job["error_code"] in {
        "generation_quarantined",
        "generation_retry_exhausted",
    }
    assert "Exact café source" not in str(terminal_job.get("error_message"))
    assert provider_calls == (1 if terminal_mode == "quarantine" else 0)
    receipt = slides.get_generation_receipt(_RECEIPT_ID, owner_user_id="owner-1")
    assert receipt.receipt_status == "failed"
    with pytest.raises(KeyError):
        slides.get_generation_input(_RECEIPT_ID, owner_user_id="owner-1")
