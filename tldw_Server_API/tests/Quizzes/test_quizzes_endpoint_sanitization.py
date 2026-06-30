from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from tldw_Server_API.app.api.v1.endpoints import quizzes
from tldw_Server_API.app.api.v1.schemas.flashcards import StudyAssistantRespondRequest


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.errors: list[str] = []

    def debug(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.debugs.append(message)

    def error(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.errors.append(message)


def test_study_suggestions_refresh_no_job_manager_log_is_sanitized(monkeypatch):
    logger_stub = _LoggerStub()
    monkeypatch.setattr(quizzes, "logger", logger_stub)

    quizzes._enqueue_study_suggestions_refresh(
        jm=None,
        current_user=SimpleNamespace(id=7),
        anchor_type="quiz_attempt",
        anchor_id=123,
    )

    assert logger_stub.debugs == ["Study-suggestions refresh enqueue skipped (no JobManager)"]
    assert "quiz_attempt" not in str(logger_stub.debugs)
    assert "123" not in str(logger_stub.debugs)


@pytest.mark.asyncio
async def test_quiz_question_assistant_unexpected_failure_log_is_sanitized(monkeypatch):
    def fail_build_quiz_attempt_question_context(*_args, **_kwargs):
        raise RuntimeError("study assistant exploded at /private/study-assistant.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(quizzes, "build_quiz_attempt_question_context", fail_build_quiz_attempt_question_context)
    monkeypatch.setattr(quizzes, "logger", logger_stub)

    with pytest.raises(HTTPException) as exc_info:
        await quizzes.respond_quiz_attempt_question_assistant(
            attempt_id=1,
            question_id=2,
            payload=StudyAssistantRespondRequest(action="explain"),
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Failed to generate study assistant response"
    assert logger_stub.errors == ["Unexpected quiz question assistant failure"]
    assert "study assistant exploded" not in str(logger_stub.errors)
    assert "/private/study-assistant.db" not in str(logger_stub.errors)
