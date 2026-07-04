from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.Chat_Macros.repository import ChatMacroRepository
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


pytestmark = pytest.mark.unit


def _jobs_module():
    try:
        return import_module("tldw_Server_API.app.core.Chat_Macros.jobs")
    except ModuleNotFoundError as exc:
        pytest.fail(f"Chat macro jobs module is missing: {exc}")
    except ImportError as exc:
        pytest.fail(f"Chat macro jobs module imports are not usable: {exc}")


class _FakeJobManager:
    def __init__(self) -> None:
        self.created: dict[str, Any] | None = None

    def create_job(self, **kwargs: Any) -> dict[str, Any]:
        self.created = kwargs
        return {"id": 17, **kwargs}


def test_enqueue_chat_macro_run_job_uses_chat_macro_domain_type_and_minimal_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = _jobs_module()
    fake_manager = _FakeJobManager()
    monkeypatch.delenv("CHAT_MACROS_JOBS_QUEUE", raising=False)

    created = jobs.enqueue_chat_macro_run_job(
        macro_run_id="run-1",
        user_id="42",
        macro_digest="digest-1",
        normalized_args={"preset": "dev_handoff", "question": ["What changed?"]},
        job_manager=fake_manager,
    )

    assert created["id"] == 17
    assert fake_manager.created == {
        "domain": jobs.CHAT_MACROS_DOMAIN,
        "queue": "default",
        "job_type": jobs.CHAT_MACROS_JOB_TYPE,
        "payload": {
            "macro_run_id": "run-1",
            "user_id": "42",
            "macro_digest": "digest-1",
            "normalized_args": {"preset": "dev_handoff", "question": ["What changed?"]},
        },
        "owner_user_id": "42",
        "priority": 5,
        "max_retries": 1,
        "idempotency_key": "chat_macro_run:run-1",
    }


def test_chat_macro_jobs_queue_uses_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    jobs = _jobs_module()
    monkeypatch.setenv("CHAT_MACROS_JOBS_QUEUE", "high")

    assert jobs.chat_macro_jobs_queue() == "high"


@pytest.mark.asyncio
async def test_handle_chat_macro_job_rejects_wrong_domain_or_type() -> None:
    jobs = _jobs_module()
    valid_payload = {
        "macro_run_id": "run-1",
        "user_id": "42",
        "macro_digest": "digest-1",
        "normalized_args": {},
    }

    with pytest.raises(ValueError, match="unsupported_chat_macro_job"):
        await jobs.handle_chat_macro_job(
            {
                "id": 1,
                "domain": "other",
                "job_type": jobs.CHAT_MACROS_JOB_TYPE,
                "payload": valid_payload,
            }
        )

    with pytest.raises(ValueError, match="unsupported_chat_macro_job"):
        await jobs.handle_chat_macro_job(
            {
                "id": 2,
                "domain": jobs.CHAT_MACROS_DOMAIN,
                "job_type": "other",
                "payload": valid_payload,
            }
        )


@pytest.mark.asyncio
async def test_handle_chat_macro_job_loads_user_db_and_executes_macro_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs = _jobs_module()
    fake_db = object()
    captured: dict[str, Any] = {}

    async def fake_get_db_for_user(user_id: int, client_id: str | None = None) -> object:
        captured["user_id"] = user_id
        captured["client_id"] = client_id
        return fake_db

    class _FakeExecutor:
        async def execute_run(self, run_id: str) -> SimpleNamespace:
            captured["run_id"] = run_id
            return SimpleNamespace(run_id=run_id, status="completed")

    def fake_build_executor(**kwargs: Any) -> _FakeExecutor:
        captured["executor_kwargs"] = kwargs
        return _FakeExecutor()

    monkeypatch.setattr(jobs, "get_chacha_db_for_user_id", fake_get_db_for_user)
    monkeypatch.setattr(jobs, "build_chat_macro_executor", fake_build_executor)
    monkeypatch.setattr(jobs, "_close_worker_database", lambda db: captured.setdefault("closed_db", db))

    result = await jobs.handle_chat_macro_job(
        {
            "id": 3,
            "domain": jobs.CHAT_MACROS_DOMAIN,
            "job_type": jobs.CHAT_MACROS_JOB_TYPE,
            "owner_user_id": "42",
            "payload": {
                "macro_run_id": "run-1",
                "user_id": "42",
                "macro_digest": "digest-1",
                "normalized_args": {"preset": "dev_handoff"},
            },
        }
    )

    assert result == {"macro_run_id": "run-1", "status": "completed"}
    assert captured["user_id"] == 42
    assert captured["client_id"] == "chat-macro-worker-42"
    assert captured["executor_kwargs"]["chat_db"] is fake_db
    assert captured["executor_kwargs"]["user_id"] == "42"
    assert captured["run_id"] == "run-1"
    assert captured["closed_db"] is fake_db


@pytest.mark.asyncio
async def test_should_cancel_chat_macro_job_finalizes_job_and_marks_run_cancelled() -> None:
    jobs = _jobs_module()

    class _FakeCancelManager:
        def __init__(self) -> None:
            self.finalized: tuple[int, str] | None = None

        def get_job(self, job_id: int) -> dict[str, Any]:
            return {
                "id": job_id,
                "status": "processing",
                "cancel_requested_at": "2026-07-03T00:00:00Z",
                "cancellation_reason": "user requested",
            }

        def finalize_cancelled(self, job_id: int, *, reason: str | None = None) -> bool:
            self.finalized = (job_id, reason or "")
            return True

    class _FakeRepository:
        def __init__(self) -> None:
            self.cancel_requested_for: str | None = None
            self.status_updates: list[tuple[str, dict[str, Any]]] = []

        def request_cancel(self, run_id: str) -> SimpleNamespace:
            self.cancel_requested_for = run_id
            return SimpleNamespace(run_id=run_id, status="cancel_requested")

        def update_run_status(self, run_id: str, **kwargs: Any) -> SimpleNamespace:
            self.status_updates.append((run_id, kwargs))
            return SimpleNamespace(run_id=run_id, status=kwargs["status"])

    manager = _FakeCancelManager()
    repository = _FakeRepository()

    should_cancel = await jobs.should_cancel_chat_macro_job(
        {"id": 9, "payload": {"macro_run_id": "run-1"}},
        job_manager=manager,
        repository=repository,
    )

    assert should_cancel is True
    assert manager.finalized == (9, "user requested")
    assert repository.cancel_requested_for == "run-1"
    assert repository.status_updates == [
        (
            "run-1",
            {
                "status": "cancelled",
                "error_code": "cancelled",
                "error_message": "Macro job was cancelled before execution completed.",
            },
        )
    ]


@pytest.mark.asyncio
async def test_should_cancel_chat_macro_job_tolerates_missing_user_payload() -> None:
    jobs = _jobs_module()

    class _FakeCancelManager:
        def get_job(self, job_id: int) -> dict[str, Any]:
            return {
                "id": job_id,
                "status": "processing",
                "cancel_requested_at": "2026-07-03T00:00:00Z",
                "cancellation_reason": "user requested",
            }

        def finalize_cancelled(self, job_id: int, *, reason: str | None = None) -> bool:
            return True

    assert await jobs.should_cancel_chat_macro_job(
        {"id": 10, "payload": {"macro_run_id": "run-1"}},
        job_manager=_FakeCancelManager(),
    )


@pytest.fixture
def chat_db(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "chat-macro-jobs.db"), client_id="chat-macro-jobs-test")
    try:
        yield db
    finally:
        db.close_connection()


def _conversation_id(db: CharactersRAGDB) -> str:
    character_id = db.add_character_card(
        {
            "name": "Macro Jobs Assistant",
            "description": "desc",
            "personality": "helpful",
            "system_prompt": "You are helpful.",
            "client_id": "42",
        }
    )
    assert character_id
    conversation_id = db.add_conversation(
        {
            "character_id": character_id,
            "title": "Macro Jobs Conversation",
            "client_id": "42",
        }
    )
    assert conversation_id
    return str(conversation_id)


def test_post_chat_macro_final_output_persists_visible_assistant_message_with_metadata(
    chat_db: CharactersRAGDB,
) -> None:
    jobs = _jobs_module()
    repository = ChatMacroRepository(chat_db)
    repository.ensure_ready()
    conversation_id = _conversation_id(chat_db)
    run = repository.create_run(
        run_id="run-1",
        user_id="42",
        macro_name="wrapup",
        macro_command="wrapup",
        macro_source="builtin",
        macro_version=1,
        macro_digest="digest-1",
        normalized_args={"preset": "dev_handoff"},
        status="completed",
        surface="chat",
        conversation_id=conversation_id,
        output_profile="default",
    )

    message_id = jobs.post_chat_macro_final_output(
        chat_db=chat_db,
        repository=repository,
        run_id=run.run_id,
        final_output="Final wrapup.",
        post_idempotency_key="chat_macro:run-1:final",
    )

    messages = chat_db.get_messages_for_conversation(conversation_id)
    assert [message["id"] for message in messages] == [message_id]
    assert messages[0]["sender"] == "assistant"
    assert messages[0]["content"] == "Final wrapup."
    metadata = chat_db.get_message_metadata(str(message_id))
    assert metadata is not None
    assert metadata["extra"]["chat_macro"] == {
        "run_id": "run-1",
        "name": "wrapup",
        "command": "wrapup",
        "status": "completed",
        "detail_url": "/api/v1/chat/macros/runs/run-1",
        "output_profile": "default",
        "post_idempotency_key": "chat_macro:run-1:final",
    }


def test_duplicate_chat_macro_postback_reuses_existing_message_for_idempotency_key(
    chat_db: CharactersRAGDB,
) -> None:
    jobs = _jobs_module()
    repository = ChatMacroRepository(chat_db)
    repository.ensure_ready()
    conversation_id = _conversation_id(chat_db)
    run = repository.create_run(
        run_id="run-2",
        user_id="42",
        macro_name="wrapup",
        macro_command="wrapup",
        macro_source="builtin",
        macro_version=1,
        macro_digest="digest-1",
        normalized_args={},
        status="completed",
        surface="chat",
        conversation_id=conversation_id,
        output_profile="default",
    )

    first_id = jobs.post_chat_macro_final_output(
        chat_db=chat_db,
        repository=repository,
        run_id=run.run_id,
        final_output="Final wrapup.",
        post_idempotency_key="chat_macro:run-2:final",
    )
    second_id = jobs.post_chat_macro_final_output(
        chat_db=chat_db,
        repository=repository,
        run_id=run.run_id,
        final_output="Final wrapup again.",
        post_idempotency_key="chat_macro:run-2:final",
    )

    messages = chat_db.get_messages_for_conversation(conversation_id)
    assert second_id == first_id
    assert len(messages) == 1
    assert messages[0]["content"] == "Final wrapup."
