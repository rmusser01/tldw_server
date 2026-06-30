import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import feedback as feedback_endpoint
from tldw_Server_API.app.api.v1.schemas.feedback_schemas import ExplicitFeedbackRequest
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError


pytestmark = pytest.mark.unit


_SENSITIVE_MARKERS = (
    "feedback-sensitive",
    "sensitive-idem",
    "feedback merge leaked",
    "feedback submit leaked",
    "feedback service leaked",
    "/private/feedback.db",
    "/private/feedback-submit.db",
)


class _LoggerStub:
    def __init__(self):
        self.exceptions: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.warnings: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def exception(self, message: str, *args: object, **kwargs: object) -> None:
        self.exceptions.append((message, args, kwargs))

    def warning(self, message: str, *args: object, **kwargs: object) -> None:
        self.warnings.append((message, args, kwargs))


class _FailingUserFeedback:
    async def merge_feedback_update(self, *args, **kwargs):
        raise CharactersRAGDBError("feedback merge leaked /private/feedback.db")


class _FeedbackSystemStub:
    def __init__(self, chacha_db):
        self.user_feedback = _FailingUserFeedback()

    async def submit_feedback(self, *args, **kwargs):
        return {"feedback_id": "feedback-sensitive"}


class _SubmitFailureFeedbackSystemStub:
    def __init__(self, chacha_db):
        self.user_feedback = None

    async def submit_feedback(self, *args, **kwargs):
        raise OSError("feedback submit leaked /private/feedback-submit.db")


class _SubmitErrorsFeedbackSystemStub:
    def __init__(self, chacha_db):
        self.user_feedback = None

    async def submit_feedback(self, *args, **kwargs):
        return {
            "feedback_id": "feedback-sensitive",
            "errors": ["feedback service leaked /private/feedback-submit.db"],
        }


def _request() -> ExplicitFeedbackRequest:
    return ExplicitFeedbackRequest(
        query="hello",
        feedback_type="helpful",
        helpful=True,
        issues=["other"],
        user_notes="note",
        idempotency_key="sensitive-idem",
    )


def _user() -> User:
    return User(id=1, username="alice", email=None, is_active=True)


def _assert_sanitized_exception_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.exceptions == [(expected_message, (), {})]
    rendered = repr(logger_stub.exceptions)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered


def _assert_sanitized_warning_log(logger_stub: _LoggerStub, expected_message: str) -> None:
    assert logger_stub.warnings == [(expected_message, (), {})]
    rendered = repr(logger_stub.warnings)
    for marker in _SENSITIVE_MARKERS:
        assert marker not in rendered


@pytest.mark.asyncio
async def test_submit_explicit_feedback_sanitizes_duplicate_merge_failure_log(monkeypatch):
    logger_stub = _LoggerStub()

    async def _reserve_duplicate(*args, **kwargs):
        return False, feedback_endpoint._IdempotencyRecord(
            feedback_id="feedback-sensitive",
            issues=["missing_details"],
            user_notes=None,
            pending=False,
        )

    monkeypatch.setattr(feedback_endpoint, "logger", logger_stub)
    monkeypatch.setattr(feedback_endpoint, "_reserve_idempotency_record", _reserve_duplicate)
    monkeypatch.setattr(feedback_endpoint, "UnifiedFeedbackSystem", _FeedbackSystemStub)

    with pytest.raises(HTTPException) as exc_info:
        await feedback_endpoint.submit_explicit_feedback(
            payload=_request(),
            current_user=_user(),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "merge_feedback_update_failed"
    _assert_sanitized_exception_log(logger_stub, "Failed to merge feedback update")


@pytest.mark.asyncio
async def test_submit_explicit_feedback_sanitizes_finalize_merge_failure_log(monkeypatch):
    logger_stub = _LoggerStub()

    async def _reserve_new(*args, **kwargs):
        return True, feedback_endpoint._IdempotencyRecord(
            feedback_id=None,
            issues=[],
            user_notes=None,
            pending=True,
        )

    async def _finalize_pending_merge(*args, **kwargs):
        return ["other"], "note", True

    monkeypatch.setattr(feedback_endpoint, "logger", logger_stub)
    monkeypatch.setattr(feedback_endpoint, "_reserve_idempotency_record", _reserve_new)
    monkeypatch.setattr(feedback_endpoint, "_finalize_idempotency_record", _finalize_pending_merge)
    monkeypatch.setattr(feedback_endpoint, "UnifiedFeedbackSystem", _FeedbackSystemStub)

    with pytest.raises(HTTPException) as exc_info:
        await feedback_endpoint.submit_explicit_feedback(
            payload=_request(),
            current_user=_user(),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "merge_feedback_update_failed"
    _assert_sanitized_exception_log(logger_stub, "Failed to finalize idempotency merge")


@pytest.mark.asyncio
async def test_submit_explicit_feedback_sanitizes_unexpected_submit_failure_log(monkeypatch):
    logger_stub = _LoggerStub()

    async def _reserve_new(*args, **kwargs):
        return True, feedback_endpoint._IdempotencyRecord(
            feedback_id=None,
            issues=[],
            user_notes=None,
            pending=True,
        )

    async def _clear_record(*args, **kwargs):
        return None

    monkeypatch.setattr(feedback_endpoint, "logger", logger_stub)
    monkeypatch.setattr(feedback_endpoint, "_reserve_idempotency_record", _reserve_new)
    monkeypatch.setattr(feedback_endpoint, "_clear_idempotency_record", _clear_record)
    monkeypatch.setattr(feedback_endpoint, "UnifiedFeedbackSystem", _SubmitFailureFeedbackSystemStub)

    with pytest.raises(HTTPException) as exc_info:
        await feedback_endpoint.submit_explicit_feedback(
            payload=_request(),
            current_user=_user(),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Internal server error"
    _assert_sanitized_exception_log(logger_stub, "Unexpected error in submit_feedback")


@pytest.mark.asyncio
async def test_submit_explicit_feedback_sanitizes_successful_submit_errors_log(monkeypatch):
    logger_stub = _LoggerStub()

    async def _reserve_new(*args, **kwargs):
        return True, feedback_endpoint._IdempotencyRecord(
            feedback_id=None,
            issues=[],
            user_notes=None,
            pending=True,
        )

    async def _finalize_record(*args, **kwargs):
        return ["other"], "note", False

    monkeypatch.setattr(feedback_endpoint, "logger", logger_stub)
    monkeypatch.setattr(feedback_endpoint, "_reserve_idempotency_record", _reserve_new)
    monkeypatch.setattr(feedback_endpoint, "_finalize_idempotency_record", _finalize_record)
    monkeypatch.setattr(feedback_endpoint, "UnifiedFeedbackSystem", _SubmitErrorsFeedbackSystemStub)

    response = await feedback_endpoint.submit_explicit_feedback(
        payload=_request(),
        current_user=_user(),
        db=object(),
    )

    assert response.ok is True
    assert response.feedback_id == "feedback-sensitive"
    _assert_sanitized_warning_log(logger_stub, "Explicit feedback completed with errors")
