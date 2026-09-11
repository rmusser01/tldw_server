from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations import quizzes_module
from tldw_Server_API.app.core.MCP_unified.modules.implementations.quizzes_module import (
    QuizzesModule,
)

_SENSITIVE_CLOSE_ERROR = "db close leaked /private/quizzes.db with sk-quiz-close"
_SENSITIVE_GENERATION_ERROR = "quiz generation leaked /private/generation.txt with sk-quiz-gen"
_UNSUPPORTED_ACTIVITY_ERROR = "This MCP quiz operation supports question quizzes only"


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


class _OsceTargetDB(_CloseFailsDB):
    def __init__(self) -> None:
        self.mutations: list[str] = []

    def get_quiz(self, quiz_id: int, **_kwargs: Any) -> dict[str, Any]:
        return {
            "id": quiz_id,
            "name": "Private OSCE",
            "activity_type": "osce",
            "total_questions": 0,
            "total_stations": 1,
            "stations": [
                {
                    "candidate_notes": "private candidate notes",
                    "checklist_items": [{"rationale": "private marking guide"}],
                }
            ],
        }

    def get_question(self, question_id: int, **_kwargs: Any) -> dict[str, Any]:
        return {"id": question_id, "quiz_id": 101, "question_text": "Impossible legacy row"}

    def get_attempt(self, attempt_id: int, **_kwargs: Any) -> dict[str, Any]:
        return {"id": attempt_id, "quiz_id": 101}

    def create_question(self, **_kwargs: Any) -> int:
        self.mutations.append("create_question")
        return 202

    def update_question(self, **_kwargs: Any) -> bool:
        self.mutations.append("update_question")
        return True

    def delete_question(self, **_kwargs: Any) -> bool:
        self.mutations.append("delete_question")
        return True

    def start_attempt(self, **_kwargs: Any) -> dict[str, Any]:
        self.mutations.append("start_attempt")
        return {"id": 303}

    def submit_attempt(self, **_kwargs: Any) -> dict[str, Any]:
        self.mutations.append("submit_attempt")
        return {"score": 100}


class _MixedQuizListDB(_CloseFailsDB):
    def __init__(self) -> None:
        self.activity_type: str | None = None

    def list_quizzes(self, **kwargs: Any) -> dict[str, Any]:
        self.activity_type = kwargs.get("activity_type")
        return {
            "items": [
                {
                    "id": 10,
                    "name": "Questions",
                    "activity_type": "questions",
                    "total_questions": 1,
                }
            ],
            "count": 1,
        }


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


def test_quiz_list_requests_question_activity_and_exposes_activity_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = QuizzesModule(ModuleConfig(name="quizzes", description="Quizzes module"))
    context = _Context()
    db = _MixedQuizListDB()
    monkeypatch.setattr(module, "_open_db", lambda _context: db)

    result = module._list_quizzes_sync(context, None, None, None, 10, 0)

    assert db.activity_type == "questions"
    assert result["total"] == 1
    assert result["quizzes"][0]["activity_type"] == "questions"


def test_quiz_list_filters_real_database_by_workspace_activity_and_page(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    db_path = str(tmp_path / "mcp-quiz-list.db")
    seed = CharactersRAGDB(db_path, client_id="mcp-list-owner")
    first_id = seed.create_quiz(
        name="First question quiz",
        workspace_tag="workspace:target",
    )
    second_id = seed.create_quiz(
        name="Second question quiz",
        workspace_tag="workspace:target",
    )
    seed.create_quiz(
        name="OSCE quiz",
        workspace_tag="workspace:target",
        activity_type="osce",
    )
    deleted_id = seed.create_quiz(
        name="Deleted question quiz",
        workspace_tag="workspace:target",
    )
    seed.delete_quiz(deleted_id)
    seed.create_quiz(
        name="Other workspace quiz",
        workspace_tag="workspace:other",
    )
    seed.close_all_connections()

    module = QuizzesModule(ModuleConfig(name="quizzes", description="Quizzes module"))
    context = _Context()
    monkeypatch.setattr(
        module,
        "_open_db",
        lambda _context: CharactersRAGDB(db_path, client_id="mcp-list-owner"),
    )

    first_page = module._list_quizzes_sync(
        context,
        None,
        None,
        "workspace:target",
        1,
        0,
    )
    second_page = module._list_quizzes_sync(
        context,
        None,
        None,
        "workspace:target",
        1,
        1,
    )

    assert first_page["total"] == 2
    assert first_page["has_more"] is True
    assert first_page["next_offset"] == 1
    assert second_page["total"] == 2
    assert second_page["has_more"] is False
    assert second_page["next_offset"] is None
    assert {
        first_page["quizzes"][0]["id"],
        second_page["quizzes"][0]["id"],
    } == {first_id, second_id}
    assert all(
        quiz["workspace_tag"] == "workspace:target"
        and quiz["activity_type"] == "questions"
        for quiz in first_page["quizzes"] + second_page["quizzes"]
    )


@pytest.mark.parametrize(
    "invoke",
    [
        pytest.param(lambda module, context: module._get_quiz_sync(context, 101), id="get-quiz"),
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
            lambda module, context: module._delete_question_sync(
                context,
                {"question_id": 202},
            ),
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
    ],
)
def test_question_oriented_mcp_operations_reject_osce_without_leaking_guides(
    monkeypatch: pytest.MonkeyPatch,
    invoke: _QuizInvocation,
) -> None:
    module = QuizzesModule(ModuleConfig(name="quizzes", description="Quizzes module"))
    context = _Context()
    db = _OsceTargetDB()
    monkeypatch.setattr(module, "_open_db", lambda _context: db)

    with pytest.raises(ValueError) as exc_info:
        invoke(module, context)

    assert str(exc_info.value) == _UNSUPPORTED_ACTIVITY_ERROR
    assert "private candidate notes" not in str(exc_info.value)
    assert "private marking guide" not in str(exc_info.value)
    assert db.mutations == []


@pytest.mark.asyncio
async def test_generate_rejects_osce_request_with_stable_message() -> None:
    module = QuizzesModule(ModuleConfig(name="quizzes", description="Quizzes module"))

    with pytest.raises(ValueError) as exc_info:
        await module.execute_tool(
            "quizzes.generate",
            {"media_id": 1, "activity_type": "osce"},
            _Context(),
        )

    assert str(exc_info.value) == _UNSUPPORTED_ACTIVITY_ERROR


@pytest.mark.asyncio
async def test_quizzes_tool_catalog_does_not_register_osce_tools() -> None:
    module = QuizzesModule(ModuleConfig(name="quizzes", description="Quizzes module"))

    names = {tool["name"] for tool in await module.get_tools()}

    assert not any("osce" in name for name in names)
