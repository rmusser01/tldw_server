"""Thin REST transport for standalone HTML presentation capabilities and jobs."""

from __future__ import annotations

import inspect
import re
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequirePermission,
    User,
    get_request_user,
    rbac_rate_limit,
)
from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.Slides_DB_Deps import get_slides_db_for_user
from tldw_Server_API.app.api.v1.schemas.slides_schemas import (
    SlidesCapabilitiesResponse,
    StandaloneHtmlGenerationRequest,
    StandaloneHtmlGenerationResponse,
)
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_CREATE, MEDIA_READ
from tldw_Server_API.app.core.Slides import standalone_html_validator
from tldw_Server_API.app.core.Slides.presentation_service import (
    CONTENT_KIND_HEADER,
    PresentationService,
    PresentationServiceError,
    merge_auth_vary_header,
    merge_vary_header,
    parse_accepted_content_kinds,
)
from tldw_Server_API.app.core.Slides.slides_db import SlidesDatabase
from tldw_Server_API.app.core.Slides.standalone_html_config import (
    SlidesStandaloneHtmlConfig,
    StandaloneHtmlGenerationAvailability,
    load_standalone_html_config,
)
from tldw_Server_API.app.core.Slides.standalone_html_reconciler import (
    reconcile_owner_generation_receipts,
)
from tldw_Server_API.app.core.Slides.standalone_html_service import (
    StandaloneHtmlGenerationError,
    StandaloneHtmlGenerationService,
    StandaloneHtmlGenerationSubmission,
    validate_idempotency_key,
)
from tldw_Server_API.app.core.Slides.standalone_html_sources import (
    StandaloneHtmlSourceError,
    resolve_standalone_html_source,
)

router = APIRouter()

_RUNTIME_STATE_ATTR = "standalone_html_api_runtime"
_TRANSPORT_CONTEXT_STATE_ATTR = "standalone_html_transport_context"
_SAFE_ERROR_CODE_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_MAX_PROGRESS_TEXT = 256
_MAX_ERROR_MESSAGE = 256


def _bounded_public_text(value: object, *, maximum: int) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text or len(text) > maximum:
        return None
    if any(ord(character) < 32 and character not in "\t\n\r" for character in text):
        return None
    return text


@dataclass(slots=True)
class StandaloneHtmlApiRuntime:
    """Source-free request runtime composed from the existing producers."""

    slides_db: SlidesDatabase
    job_manager: Any | None
    generation_service: StandaloneHtmlGenerationService
    config_loader: Any
    validator_available: bool

    def reconcile_owner(self, owner_user_id: str):
        if self.job_manager is None:
            raise StandaloneHtmlGenerationError(
                "generation_receipt_unresolved",
                status_code=503,
                retry_after=1,
            )
        return reconcile_owner_generation_receipts(
            self.slides_db,
            self.job_manager,
            owner_user_id=owner_user_id,
            now=datetime.now(timezone.utc),
            after_receipt_id=None,
            limit=100,
        )

    def receipt_error_fields(self, owner_user_id: str, receipt_id: str) -> tuple[str | None, str | None]:
        row = self.slides_db.get_generation_receipt(receipt_id, owner_user_id=owner_user_id)
        code = row.error_code if _SAFE_ERROR_CODE_RE.fullmatch(row.error_code or "") else None
        message = _bounded_public_text(row.error_message, maximum=_MAX_ERROR_MESSAGE)
        return code, message

    def job_progress(self, job_uuid: str, owner_user_id: str) -> dict[str, Any] | None:
        if self.job_manager is None:
            raise StandaloneHtmlGenerationError(
                "generation_receipt_unresolved",
                status_code=503,
                retry_after=1,
            )
        try:
            job = self.job_manager.get_job_by_uuid(job_uuid)
        except Exception:  # noqa: BLE001 - Jobs failures cross a bounded boundary
            raise StandaloneHtmlGenerationError(
                "generation_receipt_unresolved",
                status_code=503,
                retry_after=1,
            ) from None
        if not isinstance(job, dict) or job.get("uuid") != job_uuid or job.get("owner_user_id") != owner_user_id:
            return None
        return job


@dataclass(slots=True)
class _LazySourceDatabases:
    """Acquire only the selected request's source stores after replay checks."""

    request: Request
    current_user: User

    @asynccontextmanager
    async def _dependency(self, dependency: Any, **kwargs: Any):
        provider = self.request.app.dependency_overrides.get(dependency, dependency)
        resolved = provider(**kwargs) if provider is dependency else provider()
        if inspect.isawaitable(resolved):
            resolved = await resolved
        if inspect.isasyncgen(resolved):
            try:
                yield await resolved.__anext__()
            finally:
                await resolved.aclose()
            return
        if inspect.isgenerator(resolved):
            try:
                yield next(resolved)
            finally:
                resolved.close()
            return
        yield resolved

    def media(self):
        return self._dependency(
            get_media_db_for_user,
            request=self.request,
            current_user=self.current_user,
        )

    def chacha(self):
        return self._dependency(
            get_chacha_db_for_user,
            current_user=self.current_user,
        )


def _closed_config_loader(*, validator_available: bool):
    def load() -> SlidesStandaloneHtmlConfig:
        from tldw_Server_API.app.core.config import load_comprehensive_config, refresh_config_cache

        refresh_config_cache()
        return load_standalone_html_config(
            load_comprehensive_config(),
            availability=StandaloneHtmlGenerationAvailability(
                digest_key_available=False,
                worker_handler_registered=False,
                reconciler_admission_ready=False,
                validator_available=validator_available,
            ),
        )

    return load


async def _closed_digest_snapshot_loader():
    raise RuntimeError("standalone generation transport unavailable")


async def _build_runtime(request: Request, slides_db: SlidesDatabase) -> StandaloneHtmlApiRuntime:
    existing = getattr(request.app.state, _RUNTIME_STATE_ATTR, None)
    if existing is not None:
        return existing

    validator_available = bool(
        standalone_html_validator.html5lib is not None and standalone_html_validator.tinycss2 is not None
    )
    context = getattr(request.app.state, _TRANSPORT_CONTEXT_STATE_ATTR, None)
    required = (
        getattr(context, "job_manager", None),
        getattr(context, "keyring", None),
        getattr(context, "digest_snapshot_loader", None),
        getattr(context, "current_config_loader", None),
    )
    if context is None or getattr(context, "local_only", True) or any(value is None for value in required):
        service = StandaloneHtmlGenerationService(
            slides_db=slides_db,
            job_manager=None,
            keyring=None,
            digest_snapshot_loader=_closed_digest_snapshot_loader,
        )
        return StandaloneHtmlApiRuntime(
            slides_db=slides_db,
            job_manager=None,
            generation_service=service,
            config_loader=_closed_config_loader(validator_available=validator_available),
            validator_available=validator_available,
        )
    service = StandaloneHtmlGenerationService(
        slides_db=slides_db,
        job_manager=context.job_manager,
        keyring=context.keyring,
        digest_snapshot_loader=context.digest_snapshot_loader,
    )
    return StandaloneHtmlApiRuntime(
        slides_db=slides_db,
        job_manager=context.job_manager,
        generation_service=service,
        config_loader=context.current_config_loader,
        validator_available=bool(getattr(context, "validator_available", False)),
    )


def _auth_private_headers(response: Response) -> None:
    response.headers["Cache-Control"] = "private, no-store"
    merge_auth_vary_header(response.headers)


def _limits(value: object, names: tuple[str, ...]) -> dict[str, int]:
    return {name: int(getattr(value, name)) for name in names}


def _capability_payload(config: Any, *, validator_available: bool) -> dict[str, Any]:
    target = config.target
    return {
        "schema_version": 1,
        "content_kind_request_header": CONTENT_KIND_HEADER,
        "content_kinds": {
            "structured_slides": {"read": True, "edit": True},
            "standalone_html": {
                "read": True,
                "edit": validator_available,
                "export_attachment": validator_available,
                "draft_attachment": True,
                "reason": None if validator_available else "validator_unavailable",
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
                "enabled": bool(config.enabled),
                "reason": config.disabled_reason,
                "transport": "slides_generation_job",
                "source_kinds": ["prompt", "chat", "media", "notes", "rag"],
                "provider": target.provider if target is not None else None,
                "model": target.model if target is not None else None,
                "adapter_id": target.adapter_id if target is not None else None,
                "endpoint_identity": target.endpoint_identity if target is not None else None,
                "generation_config_revision": config.generation_config_revision,
                "input_limits": _limits(
                    config.input_limits,
                    (
                        "max_request_bytes",
                        "max_source_chars",
                        "max_source_tokens",
                        "max_audience_chars",
                        "max_source_identifier_bytes",
                        "max_note_ids",
                        "max_rag_query_chars",
                        "max_rag_top_k",
                    ),
                ),
                "output_limits": _limits(
                    config.output_limits,
                    ("max_provider_response_bytes", "max_document_bytes"),
                ),
            },
        },
    }


@router.get(
    "/capabilities",
    response_model=SlidesCapabilitiesResponse,
    dependencies=[Depends(RequirePermission(MEDIA_READ)), Depends(rbac_rate_limit("slides.capabilities"))],
)
async def get_slides_capabilities(
    request: Request,
    response: Response,
    slides_db: SlidesDatabase = Depends(get_slides_db_for_user),
) -> SlidesCapabilitiesResponse:
    runtime = await _build_runtime(request, slides_db)
    config = runtime.config_loader()
    _auth_private_headers(response)
    return SlidesCapabilitiesResponse.model_validate(
        _capability_payload(config, validator_available=runtime.validator_available)
    )


def _idempotency_key(request: Request) -> str:
    raw_values = [value for name, value in request.scope.get("headers", ()) if name.lower() == b"idempotency-key"]
    if not raw_values:
        raise HTTPException(status_code=400, detail="generation_idempotency_key_required")
    if len(raw_values) != 1:
        raise HTTPException(status_code=400, detail="generation_idempotency_key_invalid")
    try:
        value = raw_values[0].decode("ascii")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="generation_idempotency_key_invalid") from None
    try:
        return validate_idempotency_key(value)
    except StandaloneHtmlGenerationError:
        raise HTTPException(status_code=400, detail="generation_idempotency_key_invalid") from None


def _generation_error(exc: StandaloneHtmlGenerationError | StandaloneHtmlSourceError) -> HTTPException:
    headers: dict[str, str] = {}
    retry_after = getattr(exc, "retry_after", None)
    if isinstance(retry_after, int) and 1 <= retry_after <= 5:
        headers["Retry-After"] = str(retry_after)
    return HTTPException(
        status_code=exc.status_code,
        detail=exc.code,
        headers=headers or None,
    )


def _status_url(receipt_id: str) -> str:
    return f"/api/v1/slides/generations/{receipt_id}"


def _generation_payload(
    submission: StandaloneHtmlGenerationSubmission,
    *,
    runtime: Any,
    owner_user_id: str,
    include_progress: bool,
) -> dict[str, Any]:
    common = {
        "generation_id": submission.receipt_id,
        "status": submission.status,
        "status_url": _status_url(submission.receipt_id),
        "presentation_id": submission.presentation_id,
    }
    if submission.status in {"queued", "running"}:
        if include_progress and submission.job_uuid:
            job = runtime.job_progress(submission.job_uuid, owner_user_id)
            progress = _bounded_public_text(
                job.get("progress_message") if isinstance(job, dict) else None,
                maximum=_MAX_PROGRESS_TEXT,
            )
            if progress is not None:
                common["progress_text"] = progress
        return common
    if submission.status == "completed":
        return {**common, "content_kind": "standalone_html"}
    if submission.status == "cancelled":
        return {**common, "error_code": "generation_cancelled"}
    if submission.status == "failed":
        code, message = runtime.receipt_error_fields(owner_user_id, submission.receipt_id)
        return {
            **common,
            "error_code": (
                code if isinstance(code, str) and _SAFE_ERROR_CODE_RE.fullmatch(code) else "generation_failed"
            ),
            "error_message": (_bounded_public_text(message, maximum=_MAX_ERROR_MESSAGE) or "Generation failed."),
        }
    raise StandaloneHtmlGenerationError("generation_correlation_mismatch", status_code=409)


@router.post(
    "/generations",
    response_model=StandaloneHtmlGenerationResponse,
    response_model_exclude_unset=True,
    dependencies=[Depends(RequirePermission(MEDIA_CREATE)), Depends(rbac_rate_limit("slides.generate"))],
)
async def submit_standalone_html_generation(
    payload: StandaloneHtmlGenerationRequest,
    request: Request,
    response: Response,
    current_user: User = Depends(get_request_user),
    slides_db: SlidesDatabase = Depends(get_slides_db_for_user),
) -> dict[str, Any]:
    key = _idempotency_key(request)
    runtime = await _build_runtime(request, slides_db)
    owner_user_id = str(current_user.id)
    source_databases = _LazySourceDatabases(request, current_user)

    async def source_resolver(source: dict[str, Any], limits: Any):
        kind = source.get("kind")
        async with AsyncExitStack() as stack:
            media_db = None
            chacha_db = None
            if kind in {"media", "rag"}:
                media_db = await stack.enter_async_context(source_databases.media())
            if kind in {"chat", "notes", "rag"}:
                chacha_db = await stack.enter_async_context(source_databases.chacha())
            return await resolve_standalone_html_source(
                source,
                owner_user_id=owner_user_id,
                limits=limits,
                media_db=media_db,
                chacha_db=chacha_db,
            )

    try:
        submission = await runtime.generation_service.submit(
            owner_user_id=owner_user_id,
            idempotency_key=key,
            request=payload.model_dump(mode="python"),
            config_loader=runtime.config_loader,
            source_resolver=source_resolver,
        )
        body = _generation_payload(
            submission,
            runtime=runtime,
            owner_user_id=owner_user_id,
            include_progress=False,
        )
    except (StandaloneHtmlGenerationError, StandaloneHtmlSourceError) as exc:
        raise _generation_error(exc) from None
    response.status_code = (
        status.HTTP_202_ACCEPTED if submission.status in {"queued", "running"} else status.HTTP_200_OK
    )
    return body


def _canonical_generation_id(value: str) -> str:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError):
        raise HTTPException(status_code=404, detail="generation_not_found") from None
    if str(parsed) != value.lower():
        raise HTTPException(status_code=404, detail="generation_not_found")
    return str(parsed)


@router.get(
    "/generations/{generation_id}",
    response_model=StandaloneHtmlGenerationResponse,
    response_model_exclude_unset=True,
    dependencies=[Depends(RequirePermission(MEDIA_READ)), Depends(rbac_rate_limit("slides.generate.status"))],
)
async def get_standalone_html_generation(
    generation_id: str,
    request: Request,
    current_user: User = Depends(get_request_user),
    slides_db: SlidesDatabase = Depends(get_slides_db_for_user),
) -> dict[str, Any]:
    receipt_id = _canonical_generation_id(generation_id)
    owner_user_id = str(current_user.id)
    runtime = await _build_runtime(request, slides_db)
    try:
        submission = runtime.generation_service.get_generation(
            owner_user_id=owner_user_id,
            receipt_id=receipt_id,
        )
        if submission.status in {"claimed", "queued", "running"}:
            try:
                reconciliation = runtime.reconcile_owner(owner_user_id)
            except Exception:  # noqa: BLE001 - Jobs failures cross a bounded boundary
                raise StandaloneHtmlGenerationError(
                    "generation_receipt_unresolved",
                    status_code=503,
                    retry_after=1,
                ) from None
            if not reconciliation.jobs_available:
                raise StandaloneHtmlGenerationError(
                    "generation_receipt_unresolved",
                    status_code=503,
                    retry_after=1,
                )
            submission = runtime.generation_service.get_generation(
                owner_user_id=owner_user_id,
                receipt_id=receipt_id,
            )
            if submission.status == "claimed":
                raise StandaloneHtmlGenerationError(
                    "generation_receipt_unresolved",
                    status_code=503,
                    retry_after=1,
                )
        return _generation_payload(
            submission,
            runtime=runtime,
            owner_user_id=owner_user_id,
            include_progress=True,
        )
    except (KeyError, StandaloneHtmlGenerationError) as exc:
        if isinstance(exc, StandaloneHtmlGenerationError):
            raise _generation_error(exc) from None
        raise HTTPException(status_code=404, detail="generation_not_found") from None


def _accepted_kinds(raw: str | None, response: Response) -> frozenset[str]:
    merge_vary_header(response.headers)
    merge_auth_vary_header(response.headers)
    try:
        return parse_accepted_content_kinds(raw)
    except PresentationServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.code,
            headers={"Vary": response.headers["Vary"]},
        ) from exc


def _html_attachment_headers() -> dict[str, str]:
    headers = {
        "Content-Disposition": 'attachment; filename="presentation.html"',
        "X-Content-Type-Options": "nosniff",
        "X-Download-Options": "noopen",
        "Cache-Control": "private, no-store",
        "Referrer-Policy": "no-referrer",
        "Cross-Origin-Resource-Policy": "same-origin",
    }
    merge_vary_header(headers)
    merge_auth_vary_header(headers)
    return headers


@router.post(
    "/presentations/{presentation_id}/draft-attachment",
    response_class=Response,
    dependencies=[Depends(RequirePermission(MEDIA_READ)), Depends(rbac_rate_limit("slides.draft.download"))],
)
async def download_standalone_html_draft(
    presentation_id: str,
    request: Request,
    response: Response,
    content_type: str | None = Header(None, alias="Content-Type"),
    accept_content_kinds: str | None = Header(None, alias=CONTENT_KIND_HEADER),
    slides_db: SlidesDatabase = Depends(get_slides_db_for_user),
) -> Response:
    accepted = _accepted_kinds(accept_content_kinds, response)
    service = PresentationService(slides_db)
    try:
        kind = service.guard_target(presentation_id, accepted)
        service.require_operation(kind.content_kind, "draft_attachment")
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail="presentation_not_found",
            headers={"Vary": response.headers["Vary"]},
        ) from None
    except PresentationServiceError as exc:
        if exc.operation and exc.content_kind:
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "detail": exc.code,
                    "operation": exc.operation,
                    "content_kind": exc.content_kind,
                },
                headers={"Vary": response.headers["Vary"]},
            )
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.code,
            headers={"Vary": response.headers["Vary"]},
        ) from exc
    if (content_type or "").split(";", 1)[0].strip().lower() != "application/octet-stream":
        raise HTTPException(
            status_code=415,
            detail="unsupported_media_type",
            headers={"Vary": response.headers["Vary"]},
        )
    source = await request.body()
    try:
        body = service.prepare_draft_attachment(
            presentation_id=presentation_id,
            html_document=source,
        )
    except PresentationServiceError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail=exc.code,
            headers={"Vary": response.headers["Vary"]},
        ) from None
    return Response(
        content=body,
        media_type="application/octet-stream",
        headers=_html_attachment_headers(),
    )


__all__ = ["StandaloneHtmlApiRuntime", "router"]
