from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations import quizzes_module
from tldw_Server_API.app.core.MCP_unified.modules.implementations.quizzes_module import (
    QuizzesModule,
)


_SENSITIVE_CLOSE_ERROR = "db close leaked /private/quizzes.db with sk-quiz-close"
_SENSITIVE_GENERATION_ERROR = "quiz generation leaked /private/generation.txt with sk-quiz-gen"


@dataclass
class _Context:
    client_id: str = "mcp-test-client"
    db_paths: dict[str, str] = field(default_factory=lambda: {"chacha": "/tmp/chacha.db"})


class _CloseFailsDB:
    def close_all_connections(self) -> None:
        raise RuntimeError(_SENSITIVE_CLOSE_ERROR)

    def list_quizzes(self, **_kwargs: Any) -> dict[str, Any]:
        return {"items": [], "count": 0}

    def get_quiz(self, quiz_id: int, **_kwargs: Any) -> dict[str, Any]:
        return {"id": quiz_id, "name": "Quiz"}

    def create_quiz(self, **_kwargs: Any) -> int:
        return 101

    def update_quiz(self, **_kwargs: Any) -> bool:
        return True

    def delete_quiz(self, **_kwargs: Any) -> bool:
        return True

    def list_questions(self, **_kwargs: Any) -> dict[str, Any]:
        return {"items": [], "count": 0}

    def get_question(self, question_id: int, **_kwargs: Any) -> dict[str, Any]:
        return {"id": question_id, "question_text": "Question?"}

    def create_question(self, **_kwargs: Any) -> int:
        return 202

    def update_question(self, **_kwargs: Any) -> bool:
        return True

    def delete_question(self, **_kwargs: Any) -> bool:
        return True

    def start_attempt(self, **_kwargs: Any) -> dict[str, Any]:
        return {"id": 303}

    def submit_attempt(self, **_kwargs: Any) -> dict[str, Any]:
        return {"score": 100}

    def list_attempts(self, **_kwargs: Any) -> dict[str, Any]:
        return {"items": [], "count": 0}

    def get_attempt(self, attempt_id: int, **_kwargs: Any) -> dict[str, Any]:
        return {"id": attempt_id}


class _CleanupFailsDB:
    def delete_quiz(self, *_args: Any, **_kwargs: Any) -> bool:
        raise RuntimeError(_SENSITIVE_GENERATION_ERROR)


class _GeneratedQuizDB:
    def __init__(
        self,
        create_question: Callable[..., int] | None = None,
    ) -> None:
        self._create_question = create_question

    def close_all_connections(self) -> None:
        return None

    def create_quiz(self, **_kwargs: Any) -> int:
        return 505

    def get_quiz(self, quiz_id: int, **_kwargs: Any) -> dict[str, Any]:
        return {"id": quiz_id, "name": "Generated Quiz"}

    def create_question(self, **kwargs: Any) -> int:
        if self._create_question is not None:
            return self._create_question(**kwargs)
        return 606


_QuizInvocation = Callable[[QuizzesModule, _Context], dict[str, Any]]


@pytest.fixture
def captured_quiz_logs() -> list[str]:
    messages: list[str] = []
    sink_id = quizzes_module.logger.add(
        lambda message: messages.append(str(message)),
        level="DEBUG",
    )
    try:
        yield messages
    finally:
        quizzes_module.logger.remove(sink_id)


@pytest.mark.parametrize(
    "invoke",
    [
        pytest.param(
            lambda module, context: module._list_quizzes_sync(context, None, None, None, 10, 0),
            id="list-quizzes",
        ),
        pytest.param(lambda module, context: module._get_quiz_sync(context, 101), id="get-quiz"),
        pytest.param(
            lambda module, context: module._create_quiz_sync(context, {"name": "Quiz"}),
            id="create-quiz",
        ),
        pytest.param(
            lambda module, context: module._update_quiz_sync(
                context,
                {"quiz_id": 101, "updates": {"name": "Updated"}},
            ),
            id="update-quiz",
        ),
        pytest.param(
            lambda module, context: module._delete_quiz_sync(context, {"quiz_id": 101}),
            id="delete-quiz",
        ),
        pytest.param(
            lambda module, context: module._list_questions_sync(context, {"quiz_id": 101}),
            id="list-questions",
        ),
        pytest.param(
            lambda module, context: module._create_question_sync(
                context,
                {
                    "quiz_id": 101,
                    "question_type": "true_false",
                    "question_text": "Question?",
                    "correct_answer": True,
                },
            ),
            id="create-question",
        ),
        pytest.param(
            lambda module, context: module._update_question_sync(
                context,
                {"question_id": 202, "updates": {"question_text": "Updated?"}},
            ),
            id="update-question",
        ),
        pytest.param(
            lambda module, context: module._delete_question_sync(context, {"question_id": 202}),
            id="delete-question",
        ),
        pytest.param(
            lambda module, context: module._start_attempt_sync(context, {"quiz_id": 101}),
            id="start-attempt",
        ),
        pytest.param(
            lambda module, context: module._submit_attempt_sync(
                context,
                {"attempt_id": 303, "answers": []},
            ),
            id="submit-attempt",
        ),
        pytest.param(
            lambda module, context: module._list_attempts_sync(context, {"quiz_id": 101}),
            id="list-attempts",
        ),
        pytest.param(
            lambda module, context: module._get_attempt_sync(context, {"attempt_id": 303}),
            id="get-attempt",
        ),
        pytest.param(
            lambda module, context: module._create_generated_quiz_sync(
                context,
                "Generated Quiz",
                404,
                [
                    {
                        "question_type": "true_false",
                        "question_text": "Generated?",
                        "correct_answer": True,
                    }
                ],
            ),
            id="create-generated-quiz",
        ),
    ],
)
def test_quizzes_module_db_close_failure_logs_are_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    invoke: _QuizInvocation,
) -> None:
    module = QuizzesModule(ModuleConfig(name="quizzes", description="Quizzes module"))
    context = _Context()
    monkeypatch.setattr(module, "_open_db", lambda _context: _CloseFailsDB())

    messages: list[str] = []
    sink_id = quizzes_module.logger.add(
        lambda message: messages.append(str(message.record.get("message") or "")),
        level="DEBUG",
    )
    try:
        invoke(module, context)
    finally:
        quizzes_module.logger.remove(sink_id)

    rendered_logs = "\n".join(messages)
    assert "Failed to close DB" in rendered_logs
    assert _SENSITIVE_CLOSE_ERROR not in rendered_logs
    assert "/private/quizzes.db" not in rendered_logs
    assert "sk-quiz-close" not in rendered_logs


def test_cleanup_generated_quiz_exception_log_is_sanitized(
    captured_quiz_logs: list[str],
) -> None:
    module = QuizzesModule(ModuleConfig(name="quizzes", description="Quizzes module"))

    result = module._cleanup_generated_quiz(
        _CleanupFailsDB(),
        505,
        reason=_SENSITIVE_GENERATION_ERROR,
    )

    assert result is False
    rendered_logs = "\n".join(captured_quiz_logs)
    assert "Exception during cleanup of generated quiz" in rendered_logs
    assert _SENSITIVE_GENERATION_ERROR not in rendered_logs
    assert "/private/generation.txt" not in rendered_logs
    assert "sk-quiz-gen" not in rendered_logs


def test_quiz_generation_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    captured_quiz_logs: list[str],
) -> None:
    module = QuizzesModule(ModuleConfig(name="quizzes", description="Quizzes module"))
    context = _Context(db_paths={"chacha": "/tmp/chacha.db", "media": "/tmp/media.db"})

    async def fail_llm(*_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError(_SENSITIVE_GENERATION_ERROR)

    monkeypatch.setattr(module, "_get_media_content", lambda *_args: "media content")
    monkeypatch.setattr(module, "_call_llm", fail_llm)

    with pytest.raises(ValueError) as exc_info:
        import asyncio

        asyncio.run(
            module._generate_quiz(
                {"media_id": 505, "num_questions": 1},
                context,
            )
        )

    assert _SENSITIVE_GENERATION_ERROR in str(exc_info.value)
    rendered_logs = "\n".join(captured_quiz_logs)
    assert "Quiz generation failed" in rendered_logs
    assert _SENSITIVE_GENERATION_ERROR not in rendered_logs
    assert "/private/generation.txt" not in rendered_logs
    assert "sk-quiz-gen" not in rendered_logs


def test_media_content_lookup_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    captured_quiz_logs: list[str],
) -> None:
    module = QuizzesModule(ModuleConfig(name="quizzes", description="Quizzes module"))

    def fail_managed_media_database(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(_SENSITIVE_GENERATION_ERROR)

    monkeypatch.setattr(
        quizzes_module,
        "managed_media_database",
        fail_managed_media_database,
    )

    assert module._get_media_content("/tmp/media.db", 505) is None
    rendered_logs = "\n".join(captured_quiz_logs)
    assert "Failed to get media content" in rendered_logs
    assert _SENSITIVE_GENERATION_ERROR not in rendered_logs
    assert "/private/generation.txt" not in rendered_logs
    assert "sk-quiz-gen" not in rendered_logs


def test_generated_question_json_parse_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    captured_quiz_logs: list[str],
) -> None:
    module = QuizzesModule(ModuleConfig(name="quizzes", description="Quizzes module"))

    def fail_json_loads(*_args: Any, **_kwargs: Any) -> Any:
        raise quizzes_module.json.JSONDecodeError(
            _SENSITIVE_GENERATION_ERROR,
            _SENSITIVE_GENERATION_ERROR,
            0,
        )

    monkeypatch.setattr(quizzes_module.json, "loads", fail_json_loads)

    with pytest.raises(ValueError, match="Failed to parse generated questions from LLM response"):
        module._parse_generated_questions("not json")

    rendered_logs = "\n".join(captured_quiz_logs)
    assert "Failed to parse generated questions" in rendered_logs
    assert _SENSITIVE_GENERATION_ERROR not in rendered_logs
    assert "/private/generation.txt" not in rendered_logs
    assert "sk-quiz-gen" not in rendered_logs


def test_generated_question_validation_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    captured_quiz_logs: list[str],
) -> None:
    module = QuizzesModule(ModuleConfig(name="quizzes", description="Quizzes module"))
    context = _Context()

    calls = 0

    def validate_question(question: dict[str, Any], **_kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError(_SENSITIVE_GENERATION_ERROR)
        return None

    monkeypatch.setattr(module, "_validate_question_payload", validate_question)
    monkeypatch.setattr(module, "_open_db", lambda _context: _GeneratedQuizDB())

    result = module._create_generated_quiz_sync(
        context,
        "Generated Quiz",
        505,
        [
            {"question_type": "true_false", "question_text": "Invalid?", "correct_answer": True},
            {"question_type": "true_false", "question_text": "Valid?", "correct_answer": True},
        ],
    )

    assert result["success"] is True
    assert result["questions_created"] == 1
    rendered_logs = "\n".join(captured_quiz_logs)
    assert "Failed to validate generated question" in rendered_logs
    assert _SENSITIVE_GENERATION_ERROR not in rendered_logs
    assert "/private/generation.txt" not in rendered_logs
    assert "sk-quiz-gen" not in rendered_logs


def test_generated_question_creation_failure_log_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
    captured_quiz_logs: list[str],
) -> None:
    module = QuizzesModule(ModuleConfig(name="quizzes", description="Quizzes module"))
    context = _Context()
    calls = 0

    def create_question(**_kwargs: Any) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError(_SENSITIVE_GENERATION_ERROR)
        return 606

    monkeypatch.setattr(module, "_open_db", lambda _context: _GeneratedQuizDB(create_question))

    result = module._create_generated_quiz_sync(
        context,
        "Generated Quiz",
        505,
        [
            {"question_type": "true_false", "question_text": "Skipped?", "correct_answer": True},
            {"question_type": "true_false", "question_text": "Created?", "correct_answer": True},
        ],
    )

    assert result["success"] is True
    assert result["questions_created"] == 1
    rendered_logs = "\n".join(captured_quiz_logs)
    assert "Failed to create question" in rendered_logs
    assert _SENSITIVE_GENERATION_ERROR not in rendered_logs
    assert "/private/generation.txt" not in rendered_logs
    assert "sk-quiz-gen" not in rendered_logs
