"""Receipt-backed admission and persistence for standalone HTML generation."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Slides.slides_db import (
    ConflictError,
    PresentationRow,
    SlidesDatabase,
    SlidesGenerationInputRow,
    SlidesGenerationReceiptRow,
)
from tldw_Server_API.app.core.Slides.standalone_html_config import (
    SlidesStandaloneHtmlConfig,
)
from tldw_Server_API.app.core.Slides.standalone_html_contracts import (
    StandaloneHtmlValidationResult,
)
from tldw_Server_API.app.core.Slides.standalone_html_registry import (
    DigestKeySnapshot,
    DigestKeyUnavailableError,
    HmacDomain,
    StandaloneHtmlHmacKeyring,
)
from tldw_Server_API.app.core.Slides.standalone_html_sources import (
    StandaloneHtmlSourceSnapshot,
)

JOB_DOMAIN = "slides"
JOB_QUEUE = "default"
JOB_TYPE = "presentation.generate"
_IDEMPOTENCY_KEY_RE = re.compile(r"[A-Za-z0-9._~-]{16,200}\Z")
_REVISION_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_LOWER_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SAFE_ERROR_CODE_RE = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")
_TERMINAL_RETENTION = timedelta(days=30)
_INPUT_RETENTION = timedelta(hours=24)
_REFERENCE_HMAC_PREFIX = b"slides-source-reference:v1\x00"
_PRESENTATION_TYPES = frozenset(
    {
        "pitch-deck",
        "tech-sharing",
        "product-launch",
        "weekly-report",
        "course-module",
        "keynote",
        "data-report",
        "training",
        "social-media",
        "case-study",
        "comparison",
        "roadmap",
    }
)
_VISUAL_DIRECTIONS = frozenset(
    {
        "auto",
        "dark-technical",
        "minimal-light",
        "editorial",
        "corporate",
        "soft-pastel",
        "bold-creative",
        "neo-brutalist",
    }
)
_DELIVERY_STYLES = frozenset({"speaker-led", "self-guided"})


class StandaloneHtmlGenerationError(RuntimeError):
    """Bounded source-free generation failure."""

    __slots__ = ("code", "status_code", "retry_after", "retryable")

    def __init__(
        self,
        code: str,
        *,
        status_code: int = 422,
        retry_after: int | None = None,
    ) -> None:
        self.code = code
        self.status_code = status_code
        self.retry_after = retry_after
        self.retryable = status_code >= 500
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class CanonicalGenerationRequest:
    """Closed normalized client request and its canonical UTF-8 bytes."""

    source: dict[str, Any]
    html_options: dict[str, Any]
    generation_config_revision: str
    manifest_bytes: bytes = field(repr=False)


@dataclass(frozen=True, slots=True)
class StandaloneHtmlGenerationSubmission:
    """Source-free generation submission/replay result."""

    receipt_id: str
    status: str
    job_uuid: str | None
    presentation_id: str | None
    replayed: bool


def _fail(code: str, *, status_code: int = 422, retry_after: int | None = None) -> None:
    raise StandaloneHtmlGenerationError(
        code,
        status_code=status_code,
        retry_after=retry_after,
    ) from None


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, UnicodeEncodeError, ValueError):
        _fail("generation_request_invalid")


def _scalar_text(value: object, *, trim: bool = False) -> str:
    if not isinstance(value, str):
        _fail("generation_request_invalid")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        _fail("generation_request_invalid")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        _fail("generation_request_invalid")
    return value.strip() if trim else value


def _exact_mapping(value: object, fields: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        _fail("generation_request_invalid")
    return value


def validate_idempotency_key(value: object) -> str:
    """Validate the nonsecret transport key without trimming or echoing it."""
    if not isinstance(value, str) or _IDEMPOTENCY_KEY_RE.fullmatch(value) is None:
        _fail("generation_idempotency_key_invalid")
    return value


def _canonical_source(value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("generation_request_invalid")
    kind = value.get("kind")
    if kind == "prompt":
        source = _exact_mapping(value, frozenset({"kind", "prompt"}))
        prompt = _scalar_text(source["prompt"])
        if not prompt.strip():
            _fail("generation_request_invalid")
        return {"kind": "prompt", "prompt": prompt}
    if kind == "chat":
        source = _exact_mapping(value, frozenset({"kind", "conversation_id"}))
        conversation_id = _scalar_text(source["conversation_id"], trim=True)
        if not conversation_id or len(conversation_id.encode("utf-8")) > 256:
            _fail("generation_request_invalid")
        return {"kind": "chat", "conversation_id": conversation_id}
    if kind == "media":
        source = _exact_mapping(value, frozenset({"kind", "media_id"}))
        media_id = source["media_id"]
        if (
            isinstance(media_id, bool)
            or not isinstance(media_id, int)
            or not 1 <= media_id <= 9_223_372_036_854_775_807
        ):
            _fail("generation_request_invalid")
        return {"kind": "media", "media_id": media_id}
    if kind == "notes":
        source = _exact_mapping(value, frozenset({"kind", "note_ids"}))
        values = source["note_ids"]
        if not isinstance(values, list) or not 1 <= len(values) <= 100:
            _fail("generation_request_invalid")
        note_ids = [_scalar_text(item) for item in values]
        if any(not item.strip() or len(item.encode("utf-8")) > 256 for item in note_ids):
            _fail("generation_request_invalid")
        if len(set(note_ids)) != len(note_ids):
            _fail("generation_request_invalid")
        return {"kind": "notes", "note_ids": note_ids}
    if kind == "rag":
        fields = set(value)
        if fields not in ({"kind", "query"}, {"kind", "query", "top_k"}):
            _fail("generation_request_invalid")
        raw_query = _scalar_text(value.get("query"))
        query = raw_query.strip()
        top_k = value.get("top_k", 8)
        if (
            not query
            or len(raw_query) > 20_000
            or isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or not 1 <= top_k <= 100
        ):
            _fail("generation_request_invalid")
        return {"kind": "rag", "query": query, "top_k": top_k}
    _fail("generation_request_invalid")


def _canonical_html_options(value: object) -> dict[str, Any]:
    options = _exact_mapping(
        value,
        frozenset(
            {
                "presentation_type",
                "audience",
                "slide_count",
                "visual_direction",
                "delivery_style",
            }
        ),
    )
    presentation_type = _scalar_text(options["presentation_type"])
    audience = _scalar_text(options["audience"], trim=True)
    visual_direction = _scalar_text(options["visual_direction"])
    delivery_style = _scalar_text(options["delivery_style"])
    slide_count = options["slide_count"]
    if (
        presentation_type not in _PRESENTATION_TYPES
        or not audience
        or len(audience) > 500
        or visual_direction not in _VISUAL_DIRECTIONS
        or delivery_style not in _DELIVERY_STYLES
        or isinstance(slide_count, bool)
        or not isinstance(slide_count, int)
        or not 1 <= slide_count <= 30
    ):
        _fail("generation_request_invalid")
    return {
        "presentation_type": presentation_type,
        "audience": audience,
        "slide_count": slide_count,
        "visual_direction": visual_direction,
        "delivery_style": delivery_style,
    }


def canonicalize_generation_request(value: object) -> CanonicalGenerationRequest:
    """Validate and canonicalize the complete V1 generation request."""
    request = _exact_mapping(
        value,
        frozenset(
            {
                "generation_mode",
                "generation_config_revision",
                "source",
                "html_options",
            }
        ),
    )
    if request["generation_mode"] != "standalone_html":
        _fail("generation_request_invalid")
    revision = _scalar_text(request["generation_config_revision"])
    if _REVISION_RE.fullmatch(revision) is None:
        _fail("generation_request_invalid")
    source = _canonical_source(request["source"])
    html_options = _canonical_html_options(request["html_options"])
    normalized = {
        "generation_mode": "standalone_html",
        "generation_config_revision": revision,
        "source": source,
        "html_options": html_options,
    }
    return CanonicalGenerationRequest(
        source=source,
        html_options=html_options,
        generation_config_revision=revision,
        manifest_bytes=_canonical_json_bytes(normalized),
    )


def _stored_html_options(generation_input: SlidesGenerationInputRow) -> dict[str, Any]:
    try:
        options = json.loads(generation_input.html_options_json)
    except (json.JSONDecodeError, RecursionError, TypeError):
        _fail("generation_correlation_mismatch", status_code=409)
    if not isinstance(options, dict):
        _fail("generation_correlation_mismatch", status_code=409)
    try:
        normalized = _canonical_html_options(options)
    except StandaloneHtmlGenerationError:
        _fail("generation_correlation_mismatch", status_code=409)
    if normalized != options or _canonical_json_bytes(options).decode("utf-8") != generation_input.html_options_json:
        _fail("generation_correlation_mismatch", status_code=409)
    return normalized


def build_generation_user_content(
    generation_input: SlidesGenerationInputRow,
) -> str:
    """Rebuild the exact delimited user message from the immutable snapshot."""
    options = _stored_html_options(generation_input)
    return _canonical_json_bytes(
        {
            "schema_version": 1,
            "source": {
                "kind": generation_input.source_kind,
                "text": generation_input.source_text,
            },
            "html_options": options,
        }
    ).decode("utf-8")


def derive_jobs_idempotency_key(
    *,
    owner_user_id: str,
    idempotency_key: str,
    keyring: StandaloneHtmlHmacKeyring,
    digest_snapshot: DigestKeySnapshot,
) -> str:
    """Derive the global Jobs key without exposing the raw client key."""
    owner = _scalar_text(owner_user_id, trim=True)
    key = validate_idempotency_key(idempotency_key)
    if not owner:
        _fail("generation_request_invalid")
    digest = keyring.digest_current(
        snapshot=digest_snapshot,
        domain=HmacDomain.JOBS_IDEMPOTENCY_KEY,
        payload=owner.encode("utf-8") + b"\x00" + key.encode("ascii"),
    )
    return "slides:v1:" + digest.digest_hex


def _iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("generation clock must return aware UTC")
    return value.replace(microsecond=0).isoformat()


def _parse_canonical_utc(value: object) -> datetime:
    if not isinstance(value, str):
        _fail("generation_correlation_mismatch", status_code=409)
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        _fail("generation_correlation_mismatch", status_code=409)
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0) or _iso(parsed) != value:
        _fail("generation_correlation_mismatch", status_code=409)
    return parsed


def _validated_provenance(
    receipt: SlidesGenerationReceiptRow,
    generation_input: SlidesGenerationInputRow,
) -> dict[str, Any]:
    try:
        provenance = json.loads(generation_input.provenance_json)
    except (json.JSONDecodeError, RecursionError, TypeError):
        _fail("generation_correlation_mismatch", status_code=409)
    fields = {
        "schema_version",
        "source_kind",
        "source_ref",
        "source_snapshot_hmac_sha256",
        "digest_key_id",
        "source_bytes",
        "provider",
        "model",
        "adapter_id",
        "endpoint_identity",
        "prompt_sha256",
    }
    if (
        not isinstance(provenance, dict)
        or set(provenance) != fields
        or _canonical_json_bytes(provenance).decode("utf-8") != generation_input.provenance_json
        or provenance["schema_version"] != 1
        or provenance["source_kind"] != generation_input.source_kind
        or provenance["source_snapshot_hmac_sha256"] != generation_input.source_hmac_sha256
        or provenance["digest_key_id"] != receipt.digest_key_id
        or provenance["source_bytes"] != generation_input.source_bytes
        or provenance["provider"] != generation_input.provider
        or provenance["model"] != generation_input.model
        or provenance["adapter_id"] != generation_input.adapter_id
        or provenance["endpoint_identity"] != generation_input.endpoint_identity
        or provenance["prompt_sha256"] != generation_input.prompt_sha256
    ):
        _fail("generation_correlation_mismatch", status_code=409)
    source_ref = provenance["source_ref"]
    if generation_input.source_kind == "prompt":
        valid_reference = source_ref is None
    elif generation_input.source_kind == "chat":
        try:
            reference_bytes = source_ref.encode("utf-8") if isinstance(source_ref, str) else b""
        except UnicodeEncodeError:
            reference_bytes = b""
        valid_reference = (
            isinstance(source_ref, str)
            and bool(source_ref.strip())
            and source_ref == source_ref.strip()
            and 0 < len(reference_bytes) <= 256
        )
    elif generation_input.source_kind == "media":
        try:
            media_id = int(source_ref) if isinstance(source_ref, str) else 0
        except (TypeError, ValueError):
            media_id = 0
        valid_reference = bool(
            isinstance(source_ref, str)
            and source_ref.isascii()
            and source_ref.isdigit()
            and str(media_id) == source_ref
            and media_id > 0
            and len(source_ref) <= 256
        )
    elif generation_input.source_kind in {"notes", "rag"}:
        valid_reference = isinstance(source_ref, str) and _LOWER_SHA256_RE.fullmatch(source_ref) is not None
    else:
        valid_reference = False
    if not valid_reference:
        _fail("generation_correlation_mismatch", status_code=409)
    return provenance


def _valid_uuid(value: object) -> str:
    if not isinstance(value, str):
        _fail("generation_correlation_mismatch", status_code=409)
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError):
        _fail("generation_correlation_mismatch", status_code=409)
    if str(parsed) != value.lower():
        _fail("generation_correlation_mismatch", status_code=409)
    return value


def expected_job_payload(receipt_id: str) -> dict[str, str]:
    """Return the only allowed Jobs payload for standalone generation."""
    return {"receipt_id": _valid_uuid(receipt_id)}


class StandaloneHtmlGenerationService:
    """Claim, correlate, and atomically commit standalone generation receipts."""

    def __init__(
        self,
        *,
        slides_db: SlidesDatabase,
        job_manager: JobManager,
        keyring: StandaloneHtmlHmacKeyring,
        digest_snapshot: DigestKeySnapshot,
        now: Callable[[], datetime] | None = None,
        receipt_id_factory: Callable[[], str] | None = None,
    ) -> None:
        self.slides_db = slides_db
        self.job_manager = job_manager
        self.keyring = keyring
        self.digest_snapshot = digest_snapshot
        self._now = now or (lambda: datetime.now(timezone.utc))
        self._receipt_id_factory = receipt_id_factory or (lambda: str(uuid.uuid4()))

    def _clock(self) -> datetime:
        value = self._now()
        _iso(value)
        return value.replace(microsecond=0)

    def _candidate_hmacs(
        self,
        idempotency_key: str,
    ) -> tuple[tuple[str, str], ...]:
        candidates = self.keyring.digest_candidates(
            snapshot=self.digest_snapshot,
            domain=HmacDomain.CLIENT_IDEMPOTENCY_KEY,
            payload=idempotency_key.encode("ascii"),
        )
        return tuple((candidate.key_id, candidate.digest_hex) for candidate in candidates)

    def _request_hmac(self, key_id: str, request_bytes: bytes) -> str:
        return self.keyring.digest_for_key(
            snapshot=self.digest_snapshot,
            key_id=key_id,
            domain=HmacDomain.CLIENT_REQUEST,
            payload=request_bytes,
        ).digest_hex

    def _find_replay(
        self,
        *,
        owner_user_id: str,
        idempotency_candidates: tuple[tuple[str, str], ...],
        request_bytes: bytes,
    ) -> SlidesGenerationReceiptRow | None:
        receipt = self.slides_db.find_generation_receipt_by_idempotency_digests(
            owner_user_id=owner_user_id,
            digest_candidates=(digest for _key_id, digest in idempotency_candidates),
        )
        if receipt is None:
            return None
        matching = next(
            (
                digest
                for key_id, digest in idempotency_candidates
                if key_id == receipt.digest_key_id and hmac.compare_digest(digest, receipt.idempotency_key_hmac_sha256)
            ),
            None,
        )
        if matching is None:
            _fail("generation_correlation_mismatch", status_code=409)
        incoming_request_hmac = self._request_hmac(
            receipt.digest_key_id,
            request_bytes,
        )
        if not hmac.compare_digest(
            incoming_request_hmac,
            receipt.client_request_hmac_sha256,
        ):
            _fail("generation_idempotency_conflict", status_code=409)
        return receipt

    @staticmethod
    def _submission(
        receipt: SlidesGenerationReceiptRow,
        *,
        replayed: bool,
    ) -> StandaloneHtmlGenerationSubmission:
        return StandaloneHtmlGenerationSubmission(
            receipt_id=receipt.id,
            status=receipt.receipt_status,
            job_uuid=receipt.job_uuid,
            presentation_id=receipt.presentation_id,
            replayed=replayed,
        )

    def _validate_job(
        self,
        job: Mapping[str, Any],
        *,
        receipt: SlidesGenerationReceiptRow,
    ) -> tuple[int, str]:
        try:
            job_id = job["id"]
            job_uuid = _valid_uuid(job["uuid"])
        except (KeyError, TypeError):
            _fail("generation_correlation_mismatch", status_code=409)
        if isinstance(job_id, bool) or not isinstance(job_id, int) or job_id < 0:
            _fail("generation_correlation_mismatch", status_code=409)
        payload = self.job_manager._maybe_decrypt_json(self.job_manager._parse_json_value(job.get("payload")))
        if (
            job.get("domain") != JOB_DOMAIN
            or job.get("queue") != JOB_QUEUE
            or job.get("job_type") != JOB_TYPE
            or job.get("owner_user_id") != receipt.owner_user_id
            or job.get("idempotency_key") != receipt.jobs_idempotency_key
            or payload != expected_job_payload(receipt.id)
        ):
            _fail("generation_correlation_mismatch", status_code=409)
        if receipt.job_uuid is not None and receipt.job_uuid != job_uuid:
            _fail("generation_correlation_mismatch", status_code=409)
        if receipt.job_id is not None and receipt.job_id != job_id:
            _fail("generation_correlation_mismatch", status_code=409)
        return job_id, job_uuid

    @staticmethod
    def _execution_manifest(
        *,
        receipt: SlidesGenerationReceiptRow,
        generation_input: SlidesGenerationInputRow,
        html_options: Mapping[str, Any],
    ) -> bytes:
        return _canonical_json_bytes(
            {
                "schema_version": 1,
                "client_request_hmac_sha256": receipt.client_request_hmac_sha256,
                "source": {
                    "kind": generation_input.source_kind,
                    "hmac_sha256": generation_input.source_hmac_sha256,
                    "bytes": generation_input.source_bytes,
                },
                "html_options": dict(html_options),
                "target": {
                    "provider": generation_input.provider,
                    "model": generation_input.model,
                    "adapter_id": generation_input.adapter_id,
                    "endpoint_identity": generation_input.endpoint_identity,
                },
                "prompt": {
                    "sha256": generation_input.prompt_sha256,
                    "contract_version": generation_input.prompt_contract_version,
                },
            }
        )

    def verified_input(
        self,
        receipt: SlidesGenerationReceiptRow,
    ) -> SlidesGenerationInputRow:
        """Load immutable input and recompute every worker-available correlation."""
        try:
            generation_input = self.slides_db.get_generation_input(
                receipt.id,
                owner_user_id=receipt.owner_user_id,
            )
        except KeyError:
            _fail("generation_correlation_mismatch", status_code=409)
        digests = (
            receipt.idempotency_key_hmac_sha256,
            receipt.client_request_hmac_sha256,
            receipt.execution_hmac_sha256,
            generation_input.source_hmac_sha256,
            generation_input.prompt_sha256,
        )
        if any(_LOWER_SHA256_RE.fullmatch(value) is None for value in digests):
            _fail("generation_correlation_mismatch", status_code=409)
        try:
            source_bytes = generation_input.source_text.encode("utf-8")
            prompt_bytes = generation_input.system_prompt.encode("utf-8")
        except UnicodeEncodeError:
            _fail("generation_correlation_mismatch", status_code=409)
        receipt_created = _parse_canonical_utc(receipt.created_at)
        input_created = _parse_canonical_utc(generation_input.created_at)
        input_expires = _parse_canonical_utc(generation_input.input_expires_at)
        try:
            source_verified = self.keyring.verify(
                snapshot=self.digest_snapshot,
                key_id=receipt.digest_key_id,
                domain=HmacDomain.SOURCE_SNAPSHOT,
                payload=source_bytes,
                expected_digest_hex=generation_input.source_hmac_sha256,
            )
        except Exception:  # noqa: BLE001 - collapse key/correlation detail
            _fail("generation_correlation_mismatch", status_code=409)
        if (
            input_created != receipt_created
            or input_expires != receipt_created + _INPUT_RETENTION
            or len(source_bytes) != generation_input.source_bytes
            or hashlib.sha256(prompt_bytes).hexdigest() != generation_input.prompt_sha256
            or not source_verified
        ):
            _fail("generation_correlation_mismatch", status_code=409)
        options = _stored_html_options(generation_input)
        _validated_provenance(receipt, generation_input)
        try:
            execution_verified = self.keyring.verify(
                snapshot=self.digest_snapshot,
                key_id=receipt.digest_key_id,
                domain=HmacDomain.EXECUTION_MANIFEST,
                payload=self._execution_manifest(
                    receipt=receipt,
                    generation_input=generation_input,
                    html_options=options,
                ),
                expected_digest_hex=receipt.execution_hmac_sha256,
            )
        except Exception:  # noqa: BLE001 - collapse key/correlation detail
            _fail("generation_correlation_mismatch", status_code=409)
        if not execution_verified:
            _fail("generation_correlation_mismatch", status_code=409)
        return generation_input

    def correlate_job(
        self,
        job: Mapping[str, Any],
        *,
        owner_user_id: str,
        receipt_id: str,
    ) -> SlidesGenerationReceiptRow:
        """Validate exact Jobs correlation and support API-first or worker-first binding."""
        try:
            receipt = self.slides_db.get_generation_receipt(
                receipt_id,
                owner_user_id=owner_user_id,
            )
        except KeyError:
            _fail("generation_correlation_mismatch", status_code=409)
        job_id, job_uuid = self._validate_job(job, receipt=receipt)
        if receipt.receipt_status == "completed":
            if not receipt.presentation_id or receipt.job_uuid != job_uuid:
                _fail("generation_correlation_mismatch", status_code=409)
            try:
                presentation = self.slides_db.get_presentation_by_id(
                    receipt.presentation_id,
                    include_deleted=True,
                )
            except KeyError:
                _fail("generation_correlation_mismatch", status_code=409)
            if presentation.generation_job_uuid != job_uuid:
                _fail("generation_correlation_mismatch", status_code=409)
            return receipt
        if receipt.receipt_status in {"failed", "cancelled"}:
            return receipt
        self.verified_input(receipt)
        try:
            return self.slides_db.bind_generation_job(
                receipt_id=receipt.id,
                owner_user_id=receipt.owner_user_id,
                job_id=job_id,
                job_uuid=job_uuid,
                updated_at=_iso(self._clock()),
            )
        except (ConflictError, KeyError):
            _fail("generation_correlation_mismatch", status_code=409)

    def _replay(self, receipt: SlidesGenerationReceiptRow) -> StandaloneHtmlGenerationSubmission:
        if receipt.receipt_status == "completed":
            if not receipt.presentation_id or not receipt.job_uuid:
                _fail("generation_correlation_mismatch", status_code=409)
            try:
                presentation = self.slides_db.get_presentation_by_id(
                    receipt.presentation_id,
                    include_deleted=True,
                )
            except KeyError:
                _fail("generation_correlation_mismatch", status_code=409)
            if presentation.generation_job_uuid != receipt.job_uuid:
                _fail("generation_correlation_mismatch", status_code=409)
            return self._submission(receipt, replayed=True)
        if receipt.receipt_status != "claimed":
            if receipt.receipt_status in {"queued", "running"}:
                self.verified_input(receipt)
            return self._submission(receipt, replayed=True)
        try:
            job = self.job_manager.lookup_slides_generation_job(
                owner_user_id=receipt.owner_user_id,
                idempotency_key=receipt.jobs_idempotency_key,
                expected_job_uuid=receipt.job_uuid,
                expected_job_id=receipt.job_id,
            )
        except Exception as exc:  # noqa: BLE001 - Jobs errors cross a bounded boundary
            raise StandaloneHtmlGenerationError(
                "generation_receipt_unresolved",
                status_code=503,
                retry_after=1,
            ) from exc
        if job is None:
            _fail("generation_receipt_unresolved", status_code=503, retry_after=1)
        bound = self.correlate_job(
            job,
            owner_user_id=receipt.owner_user_id,
            receipt_id=receipt.id,
        )
        return self._submission(bound, replayed=True)

    def get_generation(
        self,
        *,
        owner_user_id: str,
        receipt_id: str,
    ) -> StandaloneHtmlGenerationSubmission:
        """Return one owner-scoped source-free receipt or a uniform miss."""
        try:
            owner = _scalar_text(owner_user_id, trim=True)
            if not owner or not isinstance(receipt_id, str):
                raise KeyError
            receipt = self.slides_db.get_generation_receipt(
                receipt_id,
                owner_user_id=owner,
            )
        except (KeyError, StandaloneHtmlGenerationError):
            _fail("generation_not_found", status_code=404)
        return self._submission(receipt, replayed=True)

    def _preflight_payload(self, payload: dict[str, str]) -> None:
        try:
            cleaned, _found, _where = self.job_manager._scan_and_redact_secrets(payload)
            if cleaned != payload:
                _fail("generation_job_payload_invalid", status_code=503)
            stored = self.job_manager._maybe_encrypt_json(payload, JOB_DOMAIN)
            encoded = json.dumps(stored).encode("utf-8")
            max_bytes = int(os.getenv("JOBS_MAX_JSON_BYTES", "1048576") or "1048576")
        except StandaloneHtmlGenerationError:
            raise
        except Exception:  # noqa: BLE001 - existing Jobs policy has varied failures
            _fail("generation_job_payload_invalid", status_code=503)
        if len(encoded) > max_bytes:
            _fail("generation_job_payload_invalid", status_code=503)

    async def submit(
        self,
        *,
        owner_user_id: str,
        idempotency_key: str,
        request: object,
        config_loader: Callable[[], SlidesStandaloneHtmlConfig],
        source_resolver: Callable[
            [dict[str, Any], Any],
            Awaitable[StandaloneHtmlSourceSnapshot],
        ],
    ) -> StandaloneHtmlGenerationSubmission:
        """Replay or atomically claim input before creating one exact Jobs row."""
        owner = _scalar_text(owner_user_id, trim=True)
        if not owner:
            _fail("generation_request_invalid")
        key = validate_idempotency_key(idempotency_key)
        canonical = canonicalize_generation_request(request)
        try:
            self.digest_snapshot.require_generation_ready()
        except DigestKeyUnavailableError:
            _fail("generation_digest_key_unavailable", status_code=503, retry_after=1)
        candidates = self._candidate_hmacs(key)
        replay = self._find_replay(
            owner_user_id=owner,
            idempotency_candidates=candidates,
            request_bytes=canonical.manifest_bytes,
        )
        if replay is not None:
            return self._replay(replay)

        config = config_loader()
        if not isinstance(config, SlidesStandaloneHtmlConfig) or not config.enabled:
            reason = getattr(config, "disabled_reason", None)
            _fail(str(reason or "generation_unavailable"), status_code=503)
        if canonical.generation_config_revision != config.generation_config_revision:
            _fail("generation_configuration_changed", status_code=409)
        if config.target is None or config.prompt is None:
            _fail("generation_unavailable", status_code=503)
        source_snapshot = await source_resolver(canonical.source, config.input_limits)
        if (
            not isinstance(source_snapshot, StandaloneHtmlSourceSnapshot)
            or source_snapshot.source_kind != canonical.source["kind"]
        ):
            _fail("source_invalid")

        current_key_id, current_idempotency_hmac = candidates[0]
        client_request_hmac = self._request_hmac(
            current_key_id,
            canonical.manifest_bytes,
        )
        source_bytes = source_snapshot.text.encode("utf-8")
        source_hmac = self.keyring.digest_for_key(
            snapshot=self.digest_snapshot,
            key_id=current_key_id,
            domain=HmacDomain.SOURCE_SNAPSHOT,
            payload=source_bytes,
        ).digest_hex
        source_ref = source_snapshot.provenance.source_ref
        if source_snapshot.provenance.reference_hmac_input is not None:
            source_ref = self.keyring.digest_for_key(
                snapshot=self.digest_snapshot,
                key_id=current_key_id,
                domain=HmacDomain.SOURCE_SNAPSHOT,
                payload=(_REFERENCE_HMAC_PREFIX + source_snapshot.provenance.reference_hmac_input),
            ).digest_hex
        provenance = {
            "schema_version": 1,
            "source_kind": source_snapshot.source_kind,
            "source_ref": source_ref,
            "source_snapshot_hmac_sha256": source_hmac,
            "digest_key_id": current_key_id,
            "source_bytes": source_snapshot.byte_count,
            "provider": config.target.provider,
            "model": config.target.model,
            "adapter_id": config.target.adapter_id,
            "endpoint_identity": config.target.endpoint_identity,
            "prompt_sha256": config.prompt.sha256,
        }
        provenance_json = _canonical_json_bytes(provenance).decode("utf-8")
        options_json = _canonical_json_bytes(canonical.html_options).decode("utf-8")
        receipt_id = _valid_uuid(self._receipt_id_factory())
        created_at = self._clock()
        input_expires_at = created_at + _INPUT_RETENTION
        jobs_key = derive_jobs_idempotency_key(
            owner_user_id=owner,
            idempotency_key=key,
            keyring=self.keyring,
            digest_snapshot=self.digest_snapshot,
        )
        provisional_receipt = SlidesGenerationReceiptRow(
            id=receipt_id,
            owner_user_id=owner,
            digest_key_id=current_key_id,
            idempotency_key_hmac_sha256=current_idempotency_hmac,
            jobs_idempotency_key=jobs_key,
            client_request_hmac_sha256=client_request_hmac,
            execution_hmac_sha256="0" * 64,
            job_id=None,
            job_uuid=None,
            presentation_id=None,
            receipt_status="claimed",
            error_code=None,
            error_message=None,
            created_at=_iso(created_at),
            updated_at=_iso(created_at),
            expires_at=None,
        )
        provisional_input = SlidesGenerationInputRow(
            receipt_id=receipt_id,
            source_kind=source_snapshot.source_kind,
            source_text=source_snapshot.text,
            source_hmac_sha256=source_hmac,
            source_bytes=source_snapshot.byte_count,
            provenance_json=provenance_json,
            html_options_json=options_json,
            provider=config.target.provider,
            model=config.target.model,
            adapter_id=config.target.adapter_id,
            endpoint_identity=config.target.endpoint_identity,
            system_prompt=config.prompt.text,
            prompt_sha256=config.prompt.sha256,
            prompt_contract_version=config.prompt.contract_version,
            input_expires_at=_iso(input_expires_at),
            created_at=_iso(created_at),
        )
        execution_hmac = self.keyring.digest_for_key(
            snapshot=self.digest_snapshot,
            key_id=current_key_id,
            domain=HmacDomain.EXECUTION_MANIFEST,
            payload=self._execution_manifest(
                receipt=provisional_receipt,
                generation_input=provisional_input,
                html_options=canonical.html_options,
            ),
        ).digest_hex
        receipt_values = {
            **asdict(provisional_receipt),
            "execution_hmac_sha256": execution_hmac,
        }
        generation_input_values = asdict(provisional_input)
        claim = self.slides_db.claim_generation_receipt_input(
            receipt=receipt_values,
            generation_input=generation_input_values,
            replay_digest_candidates=(digest for _key_id, digest in candidates),
        )
        if not claim.created:
            winner = self._find_replay(
                owner_user_id=owner,
                idempotency_candidates=candidates,
                request_bytes=canonical.manifest_bytes,
            )
            if winner is None:
                _fail("generation_correlation_mismatch", status_code=409)
            return self._replay(winner)

        payload = expected_job_payload(receipt_id)
        try:
            self._preflight_payload(payload)
        except StandaloneHtmlGenerationError:
            self.slides_db.delete_unbound_generation_claim(
                receipt_id=receipt_id,
                owner_user_id=owner,
            )
            raise
        try:
            job = self.job_manager.lookup_slides_generation_job(
                owner_user_id=owner,
                idempotency_key=jobs_key,
            )
            if job is None:
                job = self.job_manager.create_job(
                    domain=JOB_DOMAIN,
                    queue=JOB_QUEUE,
                    job_type=JOB_TYPE,
                    payload=payload,
                    owner_user_id=owner,
                    idempotency_key=jobs_key,
                )
        except Exception as exc:  # noqa: BLE001 - deterministic miss permits cleanup
            try:
                recovered = self.job_manager.lookup_slides_generation_job(
                    owner_user_id=owner,
                    idempotency_key=jobs_key,
                )
            except Exception:  # noqa: BLE001 - preserve the original bounded error
                raise StandaloneHtmlGenerationError(
                    "generation_receipt_unresolved",
                    status_code=503,
                    retry_after=1,
                ) from exc
            if recovered is not None:
                job = recovered
            else:
                self.slides_db.delete_unbound_generation_claim(
                    receipt_id=receipt_id,
                    owner_user_id=owner,
                )
                raise StandaloneHtmlGenerationError(
                    "generation_job_enqueue_rejected",
                    status_code=503,
                ) from exc
        bound = self.correlate_job(
            job,
            owner_user_id=owner,
            receipt_id=receipt_id,
        )
        return self._submission(bound, replayed=False)

    def terminalize(
        self,
        *,
        receipt: SlidesGenerationReceiptRow,
        status: str,
        error_code: str,
        error_message: str,
        terminal_at: datetime | None = None,
    ) -> bool:
        """Terminalize one receipt with deterministic retention."""
        if _SAFE_ERROR_CODE_RE.fullmatch(error_code) is None or len(error_message) > 1024:
            raise ValueError("generation terminal error is invalid")
        when = (terminal_at or self._clock()).replace(microsecond=0)
        return self.slides_db.terminalize_generation_receipt(
            receipt_id=receipt.id,
            owner_user_id=receipt.owner_user_id,
            job_uuid=receipt.job_uuid,
            status=status,
            error_code=error_code,
            error_message=error_message,
            terminal_at=_iso(when),
            expires_at=_iso(when + _TERMINAL_RETENTION),
        )

    def commit(
        self,
        *,
        receipt: SlidesGenerationReceiptRow,
        html_document: str | bytes,
        validation_result: StandaloneHtmlValidationResult,
    ) -> PresentationRow:
        """Atomically commit validated output using immutable input provenance."""
        if receipt.job_uuid is None:
            _fail("generation_correlation_mismatch", status_code=409)
        try:
            current = self.slides_db.get_generation_receipt(
                receipt.id,
                owner_user_id=receipt.owner_user_id,
            )
        except KeyError:
            _fail("generation_correlation_mismatch", status_code=409)
        if current.job_uuid != receipt.job_uuid:
            _fail("generation_correlation_mismatch", status_code=409)
        if current.receipt_status == "completed":
            if not current.presentation_id:
                _fail("generation_correlation_mismatch", status_code=409)
            try:
                presentation = self.slides_db.get_presentation_by_id(
                    current.presentation_id,
                    include_deleted=True,
                )
            except KeyError:
                _fail("generation_correlation_mismatch", status_code=409)
            if presentation.generation_job_uuid != current.job_uuid:
                _fail("generation_correlation_mismatch", status_code=409)
            return presentation
        if current.receipt_status in {"failed", "cancelled"}:
            _fail("generation_correlation_mismatch", status_code=409)
        receipt = current
        try:
            generation_input = self.verified_input(receipt)
        except StandaloneHtmlGenerationError as original_error:
            try:
                winner = self.slides_db.get_generation_receipt(
                    receipt.id,
                    owner_user_id=receipt.owner_user_id,
                )
            except KeyError:
                raise original_error from None
            if winner.receipt_status != "completed" or not winner.presentation_id:
                raise original_error from None
            try:
                presentation = self.slides_db.get_presentation_by_id(
                    winner.presentation_id,
                    include_deleted=True,
                )
            except KeyError:
                raise original_error from None
            if presentation.generation_job_uuid != receipt.job_uuid:
                raise original_error from None
            return presentation
        when = self._clock()
        result = self.slides_db.commit_generation_presentation(
            receipt_id=receipt.id,
            owner_user_id=receipt.owner_user_id,
            job_uuid=receipt.job_uuid,
            html_document=html_document,
            validation_result=validation_result,
            generation_provenance_json=generation_input.provenance_json,
            committed_at=_iso(when),
            expires_at=_iso(when + _TERMINAL_RETENTION),
        )
        return result.presentation


__all__ = [
    "CanonicalGenerationRequest",
    "JOB_DOMAIN",
    "JOB_QUEUE",
    "JOB_TYPE",
    "StandaloneHtmlGenerationError",
    "StandaloneHtmlGenerationService",
    "StandaloneHtmlGenerationSubmission",
    "build_generation_user_content",
    "canonicalize_generation_request",
    "derive_jobs_idempotency_key",
    "expected_job_payload",
    "validate_idempotency_key",
]
