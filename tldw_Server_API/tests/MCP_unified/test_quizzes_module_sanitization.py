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


_QuizInvocation = Callable[[QuizzesModule, _Context], dict[str, Any]]


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
