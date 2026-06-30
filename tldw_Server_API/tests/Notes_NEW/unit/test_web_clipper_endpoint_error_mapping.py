import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import web_clipper as web_clipper_endpoint
from tldw_Server_API.app.api.v1.endpoints.web_clipper import (
    get_web_clip_status,
    persist_web_clip_enrichment,
    save_web_clip,
)
from tldw_Server_API.app.api.v1.schemas.web_clipper_schemas import (
    WebClipperEnrichmentPayload,
    WebClipperSaveRequest,
    WebClipperSaveResponse,
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import (
    CharactersRAGDBError,
    ConflictError,
    InputError,
)

pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def error(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.errors.append(message)


class _FakeRequest:
    headers = {}


class _NoopRateLimiter:
    async def check_user_rate_limit(self, *_args, **_kwargs):
        return True, {}


class _FailingRateLimiter:
    async def check_user_rate_limit(self, *_args, **_kwargs):
        raise RuntimeError("rate limiter exploded at /private/rate-limiter.db")


async def _run_inline(fn, *args, **kwargs):
    return fn(*args, **kwargs)


def _current_user() -> User:
    return User(id=1, username="tester", email=None, is_active=True)


def _save_payload() -> WebClipperSaveRequest:
    return WebClipperSaveRequest(
        clip_id="clip-123",
        clip_type="article",
        source_url="https://example.com/story",
        source_title="Example Story",
        destination_mode="note",
        note={
            "title": "Example Story",
            "comment": "Saved from browser",
            "keywords": ["example"],
        },
        content={
            "visible_body": "Alpha paragraph.",
            "full_extract": "Alpha paragraph.\n\nBeta paragraph.",
            "selected_text": "Alpha paragraph.",
        },
        attachments=[],
        enhancements={"run_ocr": False, "run_vlm": False},
        capture_metadata={"fallback_path": ["article"]},
    )


def _enrichment_payload() -> WebClipperEnrichmentPayload:
    return WebClipperEnrichmentPayload(
        clip_id="clip-123",
        enrichment_type="ocr",
        status="complete",
        inline_summary="Captured text summary.",
        structured_payload={"raw_text": "Captured text summary."},
        source_note_version=1,
    )


@pytest.mark.asyncio
async def test_check_rate_limit_backend_failure_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(web_clipper_endpoint, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await web_clipper_endpoint._check_rate_limit(
            rate_limiter=_FailingRateLimiter(),
            current_user=_current_user(),
            scope="web_clipper.save",
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Rate limiter unavailable"
    assert logger_stub.errors == ["Web clipper rate limiter unavailable"]
    rendered_logs = " ".join(logger_stub.errors)
    assert "web_clipper.save" not in rendered_logs
    assert "/private/" not in rendered_logs
    assert "exploded" not in rendered_logs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raised_exc", "expected_status", "expected_detail"),
    [
        (InputError("invalid clip payload"), 400, "invalid clip payload"),
        (ConflictError("duplicate clip"), 409, "duplicate clip"),
        (CharactersRAGDBError("write failed"), 500, "Internal server error"),
    ],
)
async def test_save_web_clip_maps_db_errors(
    monkeypatch,
    raised_exc,
    expected_status,
    expected_detail,
):
    monkeypatch.setattr(web_clipper_endpoint.asyncio, "to_thread", _run_inline)

    def _raise_save(self, payload):
        raise raised_exc

    monkeypatch.setattr(web_clipper_endpoint.WebClipperService, "save_clip", _raise_save)

    with pytest.raises(HTTPException) as exc_info:
        await save_web_clip(
            request=_FakeRequest(),
            payload=_save_payload(),
            db=object(),
            rate_limiter=_NoopRateLimiter(),
            current_user=_current_user(),
        )

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == expected_detail


@pytest.mark.asyncio
async def test_save_web_clip_canonical_failure_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(web_clipper_endpoint, "logger", logger_stub)
    monkeypatch.setattr(web_clipper_endpoint.asyncio, "to_thread", _run_inline)

    def _failed_save(self, payload):
        return WebClipperSaveResponse(
            clip_id="clip-123",
            status="failed",
            note=None,
            workspace_placement=None,
            attachments=[],
            warnings=["Canonical note save failed at /private/clipper.db"],
            note_id="clip-123",
            workspace_placement_saved=False,
            workspace_placement_count=0,
        )

    monkeypatch.setattr(web_clipper_endpoint.WebClipperService, "save_clip", _failed_save)

    with pytest.raises(HTTPException) as exc_info:
        await save_web_clip(
            request=_FakeRequest(),
            payload=_save_payload(),
            db=object(),
            rate_limiter=_NoopRateLimiter(),
            current_user=_current_user(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Canonical note save failed."
    assert logger_stub.errors == ["Web clipper canonical save failed"]
    rendered_logs = " ".join(logger_stub.errors)
    assert "clip-123" not in rendered_logs
    assert "/private/" not in rendered_logs
    assert "Canonical note save failed at" not in rendered_logs


@pytest.mark.asyncio
async def test_save_web_clip_db_error_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(web_clipper_endpoint, "logger", logger_stub)
    monkeypatch.setattr(web_clipper_endpoint.asyncio, "to_thread", _run_inline)

    def _raise_save(self, payload):
        raise CharactersRAGDBError("save backend exploded for clip-123 at /private/clipper.db")

    monkeypatch.setattr(web_clipper_endpoint.WebClipperService, "save_clip", _raise_save)

    with pytest.raises(HTTPException) as exc_info:
        await save_web_clip(
            request=_FakeRequest(),
            payload=_save_payload(),
            db=object(),
            rate_limiter=_NoopRateLimiter(),
            current_user=_current_user(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Internal server error"
    assert logger_stub.errors == ["Web clipper save failed"]
    rendered_logs = " ".join(logger_stub.errors)
    assert "clip-123" not in rendered_logs
    assert "/private/" not in rendered_logs
    assert "exploded" not in rendered_logs


@pytest.mark.asyncio
async def test_save_web_clip_generic_fallback_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(web_clipper_endpoint, "logger", logger_stub)
    monkeypatch.setattr(web_clipper_endpoint.asyncio, "to_thread", _run_inline)

    def _raise_save(self, payload):
        raise RuntimeError("save backend exploded for clip-123 at /private/clipper.db")

    monkeypatch.setattr(web_clipper_endpoint.WebClipperService, "save_clip", _raise_save)

    with pytest.raises(HTTPException) as exc_info:
        await save_web_clip(
            request=_FakeRequest(),
            payload=_save_payload(),
            db=object(),
            rate_limiter=_NoopRateLimiter(),
            current_user=_current_user(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Internal server error"
    assert logger_stub.errors == ["Web clipper save failed"]
    rendered_logs = " ".join(logger_stub.errors)
    assert "clip-123" not in rendered_logs
    assert "/private/" not in rendered_logs
    assert "exploded" not in rendered_logs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raised_exc", "expected_status", "expected_detail"),
    [
        (ConflictError("clip not found"), 404, "clip not found"),
        (InputError("invalid clip id"), 400, "invalid clip id"),
        (CharactersRAGDBError("read failed"), 500, "Internal server error"),
    ],
)
async def test_get_web_clip_status_maps_db_errors(
    monkeypatch,
    raised_exc,
    expected_status,
    expected_detail,
):
    monkeypatch.setattr(web_clipper_endpoint.asyncio, "to_thread", _run_inline)

    def _raise_status(self, clip_id):
        raise raised_exc

    monkeypatch.setattr(web_clipper_endpoint.WebClipperService, "get_clip_status", _raise_status)

    with pytest.raises(HTTPException) as exc_info:
        await get_web_clip_status(
            clip_id="clip-123",
            request=_FakeRequest(),
            db=object(),
            rate_limiter=_NoopRateLimiter(),
            current_user=_current_user(),
        )

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == expected_detail


@pytest.mark.asyncio
async def test_get_web_clip_status_db_error_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(web_clipper_endpoint, "logger", logger_stub)
    monkeypatch.setattr(web_clipper_endpoint.asyncio, "to_thread", _run_inline)

    def _raise_status(self, clip_id):
        raise CharactersRAGDBError("status backend exploded for clip-123 at /private/clipper.db")

    monkeypatch.setattr(web_clipper_endpoint.WebClipperService, "get_clip_status", _raise_status)

    with pytest.raises(HTTPException) as exc_info:
        await get_web_clip_status(
            clip_id="clip-123",
            request=_FakeRequest(),
            db=object(),
            rate_limiter=_NoopRateLimiter(),
            current_user=_current_user(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Internal server error"
    assert logger_stub.errors == ["Web clipper status failed"]
    rendered_logs = " ".join(logger_stub.errors)
    assert "clip-123" not in rendered_logs
    assert "/private/" not in rendered_logs
    assert "exploded" not in rendered_logs


@pytest.mark.asyncio
async def test_get_web_clip_status_generic_fallback_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(web_clipper_endpoint, "logger", logger_stub)
    monkeypatch.setattr(web_clipper_endpoint.asyncio, "to_thread", _run_inline)

    def _raise_status(self, clip_id):
        raise RuntimeError("status backend exploded for clip-123 at /private/clipper.db")

    monkeypatch.setattr(web_clipper_endpoint.WebClipperService, "get_clip_status", _raise_status)

    with pytest.raises(HTTPException) as exc_info:
        await get_web_clip_status(
            clip_id="clip-123",
            request=_FakeRequest(),
            db=object(),
            rate_limiter=_NoopRateLimiter(),
            current_user=_current_user(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Internal server error"
    assert logger_stub.errors == ["Web clipper status failed"]
    rendered_logs = " ".join(logger_stub.errors)
    assert "clip-123" not in rendered_logs
    assert "/private/" not in rendered_logs
    assert "exploded" not in rendered_logs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raised_exc", "expected_status", "expected_detail"),
    [
        (ConflictError("clip not found"), 404, "clip not found"),
        (InputError("invalid enrichment payload"), 400, "invalid enrichment payload"),
        (CharactersRAGDBError("write failed"), 500, "Internal server error"),
    ],
)
async def test_persist_web_clip_enrichment_maps_db_errors(
    monkeypatch,
    raised_exc,
    expected_status,
    expected_detail,
):
    monkeypatch.setattr(web_clipper_endpoint.asyncio, "to_thread", _run_inline)

    def _raise_enrichment(self, clip_id, payload):
        raise raised_exc

    monkeypatch.setattr(
        web_clipper_endpoint.WebClipperService,
        "persist_enrichment",
        _raise_enrichment,
    )

    with pytest.raises(HTTPException) as exc_info:
        await persist_web_clip_enrichment(
            clip_id="clip-123",
            request=_FakeRequest(),
            payload=_enrichment_payload(),
            db=object(),
            rate_limiter=_NoopRateLimiter(),
            current_user=_current_user(),
        )

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == expected_detail


@pytest.mark.asyncio
async def test_persist_web_clip_enrichment_db_error_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(web_clipper_endpoint, "logger", logger_stub)
    monkeypatch.setattr(web_clipper_endpoint.asyncio, "to_thread", _run_inline)

    def _raise_enrichment(self, clip_id, payload):
        raise CharactersRAGDBError("enrichment backend exploded for clip-123 at /private/clipper.db")

    monkeypatch.setattr(
        web_clipper_endpoint.WebClipperService,
        "persist_enrichment",
        _raise_enrichment,
    )

    with pytest.raises(HTTPException) as exc_info:
        await persist_web_clip_enrichment(
            clip_id="clip-123",
            request=_FakeRequest(),
            payload=_enrichment_payload(),
            db=object(),
            rate_limiter=_NoopRateLimiter(),
            current_user=_current_user(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Internal server error"
    assert logger_stub.errors == ["Web clipper enrichment failed"]
    rendered_logs = " ".join(logger_stub.errors)
    assert "clip-123" not in rendered_logs
    assert "/private/" not in rendered_logs
    assert "exploded" not in rendered_logs


@pytest.mark.asyncio
async def test_persist_web_clip_enrichment_generic_fallback_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(web_clipper_endpoint, "logger", logger_stub)
    monkeypatch.setattr(web_clipper_endpoint.asyncio, "to_thread", _run_inline)

    def _raise_enrichment(self, clip_id, payload):
        raise RuntimeError("enrichment backend exploded for clip-123 at /private/clipper.db")

    monkeypatch.setattr(
        web_clipper_endpoint.WebClipperService,
        "persist_enrichment",
        _raise_enrichment,
    )

    with pytest.raises(HTTPException) as exc_info:
        await persist_web_clip_enrichment(
            clip_id="clip-123",
            request=_FakeRequest(),
            payload=_enrichment_payload(),
            db=object(),
            rate_limiter=_NoopRateLimiter(),
            current_user=_current_user(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Internal server error"
    assert logger_stub.errors == ["Web clipper enrichment failed"]
    rendered_logs = " ".join(logger_stub.errors)
    assert "clip-123" not in rendered_logs
    assert "/private/" not in rendered_logs
    assert "exploded" not in rendered_logs
