import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from tldw_Server_API.app.core.MCP_unified.modules.implementations.quizzes_module import QuizzesModule
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class FakeQuizzesDB:
    def __init__(self) -> None:
        self._quiz_id = 0
        self._question_id = 0
        self._attempt_id = 0
        self.quizzes: Dict[int, Dict[str, Any]] = {}
        self.questions: Dict[int, Dict[str, Any]] = {}
        self.attempts: Dict[int, Dict[str, Any]] = {}

    def create_quiz(self, name: str, description=None, workspace_tag=None, media_id=None, time_limit_seconds=None, passing_score=None, client_id=None):
        self._quiz_id += 1
        qid = self._quiz_id
        self.quizzes[qid] = {
            "id": qid,
            "name": name,
            "description": description,
            "workspace_tag": workspace_tag,
            "media_id": media_id,
            "time_limit_seconds": time_limit_seconds,
            "passing_score": passing_score,
            "version": 1,
            "deleted": 0,
        }
        return qid

    def get_quiz(self, quiz_id: int, include_deleted: bool = False):
        quiz = self.quizzes.get(quiz_id)
        if not quiz:
            return None
        if not include_deleted and quiz.get("deleted"):
            return None
        return dict(quiz)

    def list_quizzes(self, q=None, media_id=None, workspace_tag=None, include_deleted=False, limit=50, offset=0):
        items = [v for v in self.quizzes.values() if include_deleted or not v.get("deleted")]
        return {"items": items[offset: offset + limit], "count": len(items)}

    def update_quiz(self, quiz_id: int, updates: Dict[str, Any], client_id=None):
        expected_version = updates.pop("expected_version", None)
        quiz = self.quizzes.get(quiz_id)
        if not quiz or quiz.get("deleted"):
            return False
        if expected_version is not None and quiz["version"] != expected_version:
            raise ConflictError("Version mismatch", entity="quizzes", identifier=quiz_id)
        quiz.update(updates)
        quiz["version"] += 1
        return True

    def delete_quiz(self, quiz_id: int, expected_version=None, hard_delete=False):
        quiz = self.quizzes.get(quiz_id)
        if not quiz:
            return False
        if hard_delete:
            self.quizzes.pop(quiz_id, None)
            return True
        if expected_version is not None and quiz["version"] != expected_version:
            raise ConflictError("Version mismatch", entity="quizzes", identifier=quiz_id)
        quiz["deleted"] = 1
        quiz["version"] += 1
        return True

    def create_question(self, quiz_id: int, question_type: str, question_text: str, correct_answer, options=None, explanation=None, points=1, order_index=0, tags=None, client_id=None):
        self._question_id += 1
        qid = self._question_id
        self.questions[qid] = {
            "id": qid,
            "quiz_id": quiz_id,
            "question_type": question_type,
            "question_text": question_text,
            "correct_answer": correct_answer,
            "options": options,
            "explanation": explanation,
            "points": points,
            "order_index": order_index,
            "tags": tags or [],
            "version": 1,
            "deleted": 0,
        }
        return qid

    def get_question(self, question_id: int, include_deleted: bool = False):
        row = self.questions.get(question_id)
        if not row:
            return None
        if not include_deleted and row.get("deleted"):
            return None
        return dict(row)

    def list_questions(self, quiz_id: int, q=None, include_answers=False, limit=50, offset=0):
        items = [v for v in self.questions.values() if v["quiz_id"] == quiz_id and not v.get("deleted")]
        if not include_answers:
            items = [dict({k: v for k, v in item.items() if k not in {"correct_answer", "explanation"}}) for item in items]
        return {"items": items[offset: offset + (limit or len(items))], "count": len(items)}

    def update_question(self, question_id: int, updates: Dict[str, Any], client_id=None):
        expected_version = updates.pop("expected_version", None)
        row = self.questions.get(question_id)
        if not row or row.get("deleted"):
            return False
        if expected_version is not None and row["version"] != expected_version:
            raise ConflictError("Version mismatch", entity="quiz_questions", identifier=question_id)
        row.update(updates)
        row["version"] += 1
        return True

    def delete_question(self, question_id: int, expected_version=None, hard_delete=False):
        row = self.questions.get(question_id)
        if not row:
            return False
        if expected_version is not None and row["version"] != expected_version:
            raise ConflictError("Version mismatch", entity="quiz_questions", identifier=question_id)
        if hard_delete:
            self.questions.pop(question_id, None)
        else:
            row["deleted"] = 1
            row["version"] += 1
        return True

    def start_attempt(self, quiz_id: int, client_id: str = "unknown") -> Dict[str, Any]:
        self._attempt_id += 1
        aid = self._attempt_id
        self.attempts[aid] = {"id": aid, "quiz_id": quiz_id}
        return {"id": aid, "quiz_id": quiz_id, "questions": []}

    def submit_attempt(self, attempt_id: int, answers: List[Dict]) -> Dict[str, Any]:
        if attempt_id not in self.attempts:
            raise ConflictError("Attempt not found", entity="quiz_attempts", identifier=attempt_id)
        return {"attempt_id": attempt_id, "score": 0, "answers": answers}

    def list_attempts(self, quiz_id=None, limit=50, offset=0):
        items = list(self.attempts.values())
        return {"items": items[offset: offset + limit], "count": len(items)}

    def get_attempt(self, attempt_id: int, include_questions=False, include_answers=False):
        return self.attempts.get(attempt_id)

    def close_all_connections(self) -> None:
        return None


def test_quizzes_module_get_media_content_uses_managed_media_database(monkeypatch, tmp_path: Path):
    mod = QuizzesModule(ModuleConfig(name="quizzes"))
    events = []

    class _FakeDb:
        def get_media_by_id(self, media_id: int):
            events.append(("get_media_by_id", media_id))
            return {"content": "quiz source"}

    @contextlib.contextmanager
    def _fake_managed_media_database(client_id, **kwargs):
        events.append(("open", client_id, kwargs))
        yield _FakeDb()

    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.modules.implementations.quizzes_module.MediaDatabase",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("quizzes_module should not construct MediaDatabase directly")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.MCP_unified.modules.implementations.quizzes_module.managed_media_database",
        _fake_managed_media_database,
        raising=False,
    )

    result = mod._get_media_content(str(tmp_path / "media.db"), 17)

    _ensure(result == "quiz source", f"Unexpected media content: {result!r}")
    _ensure(
        events == [
            ("open", "mcp_quizzes_gen", {"db_path": str(tmp_path / "media.db"), "initialize": False}),
            ("get_media_by_id", 17),
        ],
        f"Unexpected media database events: {events!r}",
    )


@pytest.mark.asyncio
async def test_quizzes_crud_and_generation(tmp_path: Path):
    mod = QuizzesModule(ModuleConfig(name="quizzes"))
    fake_db = FakeQuizzesDB()
    mod._open_db = lambda ctx: fake_db  # type: ignore[attr-defined]

    ctx = SimpleNamespace(
        db_paths={
            "media": str(tmp_path / "media.db"),
            "chacha": str(tmp_path / "chacha.db"),
        },
        client_id="test",
    )

    created = await mod.execute_tool(
        "quizzes.create",
        {"name": "Quiz 1", "description": "Desc"},
        context=ctx,
    )
    _ensure(created["success"] is True, f"Unexpected create payload: {created!r}")
    quiz_id = created["quiz_id"]

    listed = await mod.execute_tool("quizzes.list", {"limit": 10, "offset": 0}, context=ctx)
    _ensure(listed["total"] == 1, f"Unexpected quiz list payload: {listed!r}")

    updated = await mod.execute_tool(
        "quizzes.update",
        {"quiz_id": quiz_id, "updates": {"name": "Quiz 1b"}},
        context=ctx,
    )
    _ensure("name" in updated["updated_fields"], f"Unexpected quiz update payload: {updated!r}")

    q_created = await mod.execute_tool(
        "quizzes.questions.create",
        {
            "quiz_id": quiz_id,
            "question_type": "true_false",
            "question_text": "Is sky blue?",
            "correct_answer": True,
        },
        context=ctx,
    )
    _ensure(q_created["success"] is True, f"Unexpected question create payload: {q_created!r}")

    questions = await mod.execute_tool("quizzes.questions.list", {"quiz_id": quiz_id}, context=ctx)
    _ensure(questions["total"] == 1, f"Unexpected question list payload: {questions!r}")

    attempt = await mod.execute_tool("quizzes.attempts.start", {"quiz_id": quiz_id}, context=ctx)
    attempt_id = attempt["attempt"]["id"]

    submit = await mod.execute_tool(
        "quizzes.attempts.submit",
        {"attempt_id": attempt_id, "answers": []},
        context=ctx,
    )
    _ensure(submit["success"] is True, f"Unexpected attempt submit payload: {submit!r}")

    # Stub quiz generation
    async def _fake_llm(_prompt: str, *, provider: str, model: str | None):
        return json.dumps([
            {"question_type": "multiple_choice", "question_text": "Q?", "options": ["A", "B"], "correct_answer": 0}
        ])

    mod._get_media_content = lambda *_args, **_kwargs: "Content"  # type: ignore[attr-defined]
    mod._call_llm = _fake_llm  # type: ignore[attr-defined]

    generated = await mod.execute_tool(
        "quizzes.generate",
        {"media_id": 1, "num_questions": 1},
        context=ctx,
    )
    _ensure(generated["success"] is True, f"Unexpected generation payload: {generated!r}")
    _ensure(generated["questions_created"] == 1, f"Unexpected generation question count: {generated!r}")


@pytest.mark.asyncio
async def test_quizzes_update_rejects_invalid_metadata(tmp_path: Path):
    mod = QuizzesModule(ModuleConfig(name="quizzes"))
    fake_db = FakeQuizzesDB()
    mod._open_db = lambda ctx: fake_db  # type: ignore[attr-defined]

    ctx = SimpleNamespace(db_paths={"media": str(tmp_path / "media.db"), "chacha": str(tmp_path / "chacha.db")}, client_id="test")

    created = await mod.execute_tool("quizzes.create", {"name": "Quiz 1"}, context=ctx)
    quiz_id = created["quiz_id"]

    with pytest.raises(ValueError, match="name must be 1..256 chars"):
        await mod.execute_tool(
            "quizzes.update",
            {"quiz_id": quiz_id, "updates": {"name": "   "}},
            context=ctx,
        )

    with pytest.raises(ValueError, match="name must be 1..256 chars"):
        await mod.execute_tool(
            "quizzes.update",
            {"quiz_id": quiz_id, "updates": {"name": None}},
            context=ctx,
        )


@pytest.mark.asyncio
async def test_quizzes_create_rejects_boolean_time_limit_seconds(tmp_path: Path):
    mod = QuizzesModule(ModuleConfig(name="quizzes"))
    fake_db = FakeQuizzesDB()
    mod._open_db = lambda ctx: fake_db  # type: ignore[attr-defined]

    ctx = SimpleNamespace(db_paths={"media": str(tmp_path / "media.db"), "chacha": str(tmp_path / "chacha.db")}, client_id="test")

    with pytest.raises(ValueError, match="time_limit_seconds must be a non-negative integer"):
        await mod.execute_tool(
            "quizzes.create",
            {"name": "Quiz 1", "time_limit_seconds": True},
            context=ctx,
        )


@pytest.mark.asyncio
async def test_quizzes_update_partial_payload_uses_existing_name(tmp_path: Path):
    mod = QuizzesModule(ModuleConfig(name="quizzes"))
    fake_db = FakeQuizzesDB()
    mod._open_db = lambda ctx: fake_db  # type: ignore[attr-defined]

    ctx = SimpleNamespace(db_paths={"media": str(tmp_path / "media.db"), "chacha": str(tmp_path / "chacha.db")}, client_id="test")

    created = await mod.execute_tool(
        "quizzes.create",
        {"name": "Quiz 1", "passing_score": 50},
        context=ctx,
    )
    quiz_id = created["quiz_id"]

    updated = await mod.execute_tool(
        "quizzes.update",
        {"quiz_id": quiz_id, "updates": {"passing_score": 75}},
        context=ctx,
    )

    _ensure(updated["success"] is True, f"Unexpected quiz update payload: {updated!r}")
    row = fake_db.quizzes[quiz_id]
    _ensure(row["name"] == "Quiz 1", f"Quiz name should be preserved on partial update: {row!r}")
    _ensure(row["passing_score"] == 75, f"Quiz passing score was not updated: {row!r}")


@pytest.mark.asyncio
async def test_quizzes_question_update_rejects_invalid_shape(tmp_path: Path):
    mod = QuizzesModule(ModuleConfig(name="quizzes"))
    fake_db = FakeQuizzesDB()
    mod._open_db = lambda ctx: fake_db  # type: ignore[attr-defined]

    ctx = SimpleNamespace(db_paths={"media": str(tmp_path / "media.db"), "chacha": str(tmp_path / "chacha.db")}, client_id="test")

    created = await mod.execute_tool("quizzes.create", {"name": "Quiz 1"}, context=ctx)
    quiz_id = created["quiz_id"]
    question = await mod.execute_tool(
        "quizzes.questions.create",
        {
            "quiz_id": quiz_id,
            "question_type": "true_false",
            "question_text": "Is this valid?",
            "correct_answer": True,
        },
        context=ctx,
    )
    question_id = question["question_id"]

    with pytest.raises(ValueError, match="multiple_choice requires at least 2 options"):
        await mod.execute_tool(
            "quizzes.questions.update",
            {
                "question_id": question_id,
                "updates": {
                    "question_type": "multiple_choice",
                    "options": ["A"],
                    "correct_answer": 0,
                },
            },
            context=ctx,
        )

    with pytest.raises(ValueError, match="question_text must be 1..5000 chars"):
        await mod.execute_tool(
            "quizzes.questions.update",
            {
                "question_id": question_id,
                "updates": {"question_text": None},
            },
            context=ctx,
        )


@pytest.mark.asyncio
async def test_quizzes_question_partial_update_uses_existing_question_shape(tmp_path: Path):
    mod = QuizzesModule(ModuleConfig(name="quizzes"))
    fake_db = FakeQuizzesDB()
    mod._open_db = lambda ctx: fake_db  # type: ignore[attr-defined]

    ctx = SimpleNamespace(db_paths={"media": str(tmp_path / "media.db"), "chacha": str(tmp_path / "chacha.db")}, client_id="test")

    created = await mod.execute_tool("quizzes.create", {"name": "Quiz 1"}, context=ctx)
    quiz_id = created["quiz_id"]
    question = await mod.execute_tool(
        "quizzes.questions.create",
        {
            "quiz_id": quiz_id,
            "question_type": "multiple_choice",
            "question_text": "Pick one",
            "options": ["A", "B"],
            "correct_answer": 0,
        },
        context=ctx,
    )
    question_id = question["question_id"]

    updated = await mod.execute_tool(
        "quizzes.questions.update",
        {
            "question_id": question_id,
            "updates": {
                "options": ["C", "D"],
                "correct_answer": 1,
            },
        },
        context=ctx,
    )

    _ensure(updated["success"] is True, f"Unexpected question update payload: {updated!r}")
    row = fake_db.questions[question_id]
    _ensure(row["options"] == ["C", "D"], f"Unexpected question options after update: {row!r}")
    _ensure(row["correct_answer"] == 1, f"Unexpected question answer after update: {row!r}")


@pytest.mark.asyncio
async def test_quizzes_generate_fails_when_no_generated_questions_persist(tmp_path: Path):
    mod = QuizzesModule(ModuleConfig(name="quizzes"))

    class RejectingQuestionDB(FakeQuizzesDB):
        def create_question(self, *args: Any, **kwargs: Any):  # type: ignore[override]
            raise ValueError("invalid generated question")

    fake_db = RejectingQuestionDB()
    mod._open_db = lambda ctx: fake_db  # type: ignore[attr-defined]

    ctx = SimpleNamespace(db_paths={"media": str(tmp_path / "media.db"), "chacha": str(tmp_path / "chacha.db")}, client_id="test")

    mod._get_media_content = lambda *_args, **_kwargs: "Content"  # type: ignore[attr-defined]

    async def _fake_llm(_prompt: str, *, provider: str, model: str | None):
        return json.dumps([
            {"question_type": "multiple_choice", "question_text": "Q?", "options": ["A", "B"], "correct_answer": 0}
        ])

    mod._call_llm = _fake_llm  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="no valid questions were created"):
        await mod.execute_tool(
            "quizzes.generate",
            {"media_id": 1, "num_questions": 1},
            context=ctx,
        )

    _ensure(fake_db.quizzes == {}, f"Generated quiz cleanup should remove empty quiz records: {fake_db.quizzes!r}")


@pytest.mark.asyncio
async def test_quizzes_generate_preserves_empty_generation_error_when_cleanup_returns_false(tmp_path: Path):
    mod = QuizzesModule(ModuleConfig(name="quizzes"))

    class CleanupFailureDB(FakeQuizzesDB):
        def create_question(self, *args: Any, **kwargs: Any):  # type: ignore[override]
            raise ValueError("invalid generated question")

        def delete_quiz(self, quiz_id: int, hard_delete: bool = False, expected_version: int | None = None):  # type: ignore[override]
            return False

    fake_db = CleanupFailureDB()
    mod._open_db = lambda ctx: fake_db  # type: ignore[attr-defined]

    ctx = SimpleNamespace(db_paths={"media": str(tmp_path / "media.db"), "chacha": str(tmp_path / "chacha.db")}, client_id="test")

    mod._get_media_content = lambda *_args, **_kwargs: "Content"  # type: ignore[attr-defined]

    async def _fake_llm(_prompt: str, *, provider: str, model: str | None):
        return json.dumps([
            {"question_type": "multiple_choice", "question_text": "Q?", "options": ["A", "B"], "correct_answer": 0}
        ])

    mod._call_llm = _fake_llm  # type: ignore[attr-defined]

    with pytest.raises(ValueError, match="no valid questions were created"):
        await mod.execute_tool(
            "quizzes.generate",
            {"media_id": 1, "num_questions": 1},
            context=ctx,
        )


@pytest.mark.asyncio
async def test_quizzes_generate_cleans_up_quiz_after_unexpected_question_persistence_error(tmp_path: Path):
    mod = QuizzesModule(ModuleConfig(name="quizzes"))

    class UnexpectedQuestionPersistenceError(Exception):
        pass

    class UnexpectedFailureDB(FakeQuizzesDB):
        def create_question(self, *args: Any, **kwargs: Any):  # type: ignore[override]
            raise UnexpectedQuestionPersistenceError("unexpected persistence failure")

    fake_db = UnexpectedFailureDB()
    mod._open_db = lambda ctx: fake_db  # type: ignore[attr-defined]

    ctx = SimpleNamespace(db_paths={"media": str(tmp_path / "media.db"), "chacha": str(tmp_path / "chacha.db")}, client_id="test")

    mod._get_media_content = lambda *_args, **_kwargs: "Content"  # type: ignore[attr-defined]

    async def _fake_llm(_prompt: str, *, provider: str, model: str | None):
        return json.dumps([
            {"question_type": "multiple_choice", "question_text": "Q?", "options": ["A", "B"], "correct_answer": 0}
        ])

    mod._call_llm = _fake_llm  # type: ignore[attr-defined]

    with pytest.raises(UnexpectedQuestionPersistenceError, match="unexpected persistence failure"):
        await mod.execute_tool(
            "quizzes.generate",
            {"media_id": 1, "num_questions": 1},
            context=ctx,
        )

    _ensure(
        fake_db.quizzes == {},
        f"Unexpected question persistence failures should not leave orphan quizzes behind: {fake_db.quizzes!r}",
    )
