from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import ConflictError
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations.notes_module import NotesModule


class _FakeTaskStore:
    def __init__(self, db: "_FakeTaskDB") -> None:
        self._db = db

    def _fetch_projection(self, task_id: str) -> dict[str, Any] | None:
        projection = self._db.projections.get(task_id)
        return dict(projection) if projection else None


class _FakeTaskDB:
    def __init__(self) -> None:
        self.notes: dict[str, dict[str, Any]] = {
            "note-1": {"id": "note-1", "title": "Inbox", "version": 1, "content": "- [ ] Seed\n"},
            "note-2": {"id": "note-2", "title": "Later", "version": 3, "content": ""},
        }
        self.tasks: dict[str, dict[str, Any]] = {
            "task-1": self._task("task-1", note_id="note-1", text="Seed", version=1),
            "task-same": self._task("task-same", note_id="note-1", text="Already done", status="done", version=4),
            "task-conflict": self._task("task-conflict", note_id="note-1", text="Conflict", version=2),
        }
        self.projections: dict[str, dict[str, Any]] = {
            task_id: {
                "note_id": task["note_id"],
                "note_version": 1,
                "line_number": 1,
                "start_offset": 0,
                "end_offset": 10,
                "raw_line": "- [ ] Seed",
                "has_child_content": False,
                "projection_status": task["projection_status"],
            }
            for task_id, task in self.tasks.items()
        }
        self.task_store = _FakeTaskStore(self)
        self.closed = False

    def _task(
        self,
        task_id: str,
        *,
        note_id: str,
        text: str,
        status: str = "open",
        projection_status: str = "live",
        version: int = 1,
    ) -> dict[str, Any]:
        return {
            "id": task_id,
            "note_id": note_id,
            "text": text,
            "status": status,
            "metadata_json": {},
            "projection_status": projection_status,
            "version": version,
            "created_at": "2026-01-01T00:00:00Z",
            "updated_at": "2026-01-01T00:00:00Z",
            "completed_at": None,
        }

    def get_note_by_id(self, note_id: str) -> dict[str, Any] | None:
        note = self.notes.get(note_id)
        return dict(note) if note else None

    def get_task(self, task_id: str, include_deleted: bool = False) -> dict[str, Any] | None:  # noqa: ARG002
        task = self.tasks.get(task_id)
        return dict(task) if task else None

    def list_tasks(
        self,
        *,
        note_id: str | None = None,
        status: str | None = None,
        projection_status: str | None = None,
        include_deleted: bool = False,  # noqa: ARG002
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        rows = list(self.tasks.values())
        if note_id is not None:
            rows = [task for task in rows if task["note_id"] == note_id]
        if status is not None:
            rows = [task for task in rows if task["status"] == status]
        if projection_status is not None:
            rows = [task for task in rows if task["projection_status"] == projection_status]
        return [dict(task) for task in rows[:limit]]

    def close_all_connections(self) -> None:
        self.closed = True


class _FakeTaskService:
    def __init__(self, db: _FakeTaskDB) -> None:
        self.db = db
        self.reconcile_stale_calls = 0
        self.ensure_note_calls: list[str] = []
        self.create_calls = 0
        self.update_calls: list[dict[str, Any]] = []
        self.reconcile_note_calls: list[str] = []

    def reconcile_stale_notes(self, *, db: _FakeTaskDB, limit: int, actor: Any) -> Any:  # noqa: ARG002
        self.reconcile_stale_calls += 1
        return SimpleNamespace(status="clean", processed_notes=1, remaining_stale_notes=0)

    def ensure_note_reconciled(self, *, db: _FakeTaskDB, note_id: str, actor: Any) -> Any:  # noqa: ARG002
        self.ensure_note_calls.append(note_id)
        return SimpleNamespace(
            note_id=note_id,
            note_version=1,
            parsed_count=1,
            created_count=0,
            updated_count=0,
            unlinked_count=0,
            ambiguous_count=0,
            warning_count=0,
        )

    def reconcile_note_current(self, *, db: _FakeTaskDB, note_id: str, actor: Any) -> Any:  # noqa: ARG002
        self.reconcile_note_calls.append(note_id)
        return SimpleNamespace(
            note_id=note_id,
            note_version=1,
            parsed_count=1,
            created_count=0,
            updated_count=0,
            unlinked_count=0,
            ambiguous_count=0,
            warning_count=0,
        )

    def create_task_for_note(
        self,
        *,
        db: _FakeTaskDB,
        note_id: str,
        text: str,
        status: str,
        metadata: dict[str, Any],
        expected_note_version: int,
        actor: Any,  # noqa: ARG002
    ) -> dict[str, Any]:
        self.create_calls += 1
        task_id = f"task-created-{self.create_calls}"
        db.tasks[task_id] = db._task(task_id, note_id=note_id, text=text, status=status, version=1)
        db.tasks[task_id]["metadata_json"] = dict(metadata)
        db.projections[task_id] = {
            "note_id": note_id,
            "note_version": expected_note_version,
            "line_number": self.create_calls,
            "start_offset": 0,
            "end_offset": 12,
            "raw_line": f"- [ ] {text}",
            "has_child_content": False,
            "projection_status": "live",
        }
        return dict(db.tasks[task_id])

    def update_task(
        self,
        *,
        db: _FakeTaskDB,
        task_id: str,
        expected_task_version: int,
        expected_note_version: int | None,
        actor: Any,  # noqa: ARG002
        text: str | None = None,
        status: str | None = None,
        metadata: dict[str, Any] | None = None,
        record_only: bool = False,
    ) -> dict[str, Any]:
        self.update_calls.append(
            {
                "task_id": task_id,
                "expected_task_version": expected_task_version,
                "expected_note_version": expected_note_version,
                "text": text,
                "status": status,
                "metadata": metadata,
                "record_only": record_only,
            }
        )
        if task_id == "task-conflict":
            raise ConflictError("Task projection is ambiguous.", entity="tasks", entity_id=task_id)
        task = dict(db.tasks[task_id])
        if text is not None:
            task["text"] = text
        if status is not None:
            task["status"] = status
        if metadata is not None:
            task["metadata_json"] = dict(metadata)
        task["version"] = int(task["version"]) + 1
        db.tasks[task_id] = task
        return dict(task)

    def delete_task(self, **kwargs: Any) -> dict[str, Any]:
        task_id = kwargs["task_id"]
        task = dict(self.db.tasks[task_id])
        task["projection_status"] = "deleted"
        task["version"] = int(task["version"]) + 1
        self.db.tasks[task_id] = task
        return dict(task)


def _module_with_fakes() -> tuple[NotesModule, _FakeTaskDB, _FakeTaskService]:
    module = NotesModule(ModuleConfig(name="notes"))
    db = _FakeTaskDB()
    service = _FakeTaskService(db)
    module._open_db = lambda _ctx: db  # type: ignore[attr-defined]
    module._task_service = service  # type: ignore[attr-defined]
    return module, db, service


def _ctx(**metadata: Any) -> SimpleNamespace:
    return SimpleNamespace(
        request_id="req-1",
        user_id="7",
        client_id="unit",
        session_id="sess-1",
        metadata=metadata,
    )


@pytest.mark.asyncio
async def test_notes_task_tools_are_discoverable_with_policy_metadata_and_tight_schemas() -> None:
    module = NotesModule(ModuleConfig(name="notes"))

    tools = {tool["name"]: tool for tool in await module.get_tools()}

    expected = {
        "notes.tasks.list",
        "notes.tasks.get",
        "notes.tasks.create",
        "notes.tasks.update",
        "notes.tasks.set_status",
        "notes.tasks.delete",
        "notes.tasks.reconcile_note",
    }
    assert expected.issubset(tools)  # nosec B101
    for name in {"notes.tasks.list", "notes.tasks.get"}:
        assert tools[name]["metadata"]["readOnlyHint"] is True  # nosec B101
    for name in expected - {"notes.tasks.list", "notes.tasks.get"}:
        metadata = tools[name]["metadata"]
        assert metadata["category"] == "management"  # nosec B101
        assert metadata["auth_required"] is True  # nosec B101
        assert metadata["requires_confirmation"] is True  # nosec B101
        assert metadata["agent_write_policy"] == "approval_required"  # nosec B101
        assert metadata["autonomous_writes"] == "denied"  # nosec B101
        assert tools[name]["inputSchema"]["additionalProperties"] is False  # nosec B101

    assert "idempotencyKey" in tools["notes.tasks.create"]["inputSchema"]["properties"]  # nosec B101
    assert "idempotencyKey" in tools["notes.tasks.set_status"]["inputSchema"]["properties"]  # nosec B101


@pytest.mark.parametrize(
    ("tool_name", "arguments", "error_match"),
    [
        ("notes.tasks.list", {"status": "blocked"}, "status"),
        ("notes.tasks.list", {"limit": 501}, "limit"),
        ("notes.tasks.get", {}, "task_id"),
        ("notes.tasks.create", {"note_id": "", "text": "Task", "expected_note_version": 1}, "note_id"),
        ("notes.tasks.create", {"note_id": "note-1", "text": "x" * 2001, "expected_note_version": 1}, "text"),
        (
            "notes.tasks.create",
            {"note_id": "note-1", "text": "Task", "expected_note_version": 1, "metadata": {"owner": "agent"}},
            "metadata",
        ),
        ("notes.tasks.create", {"note_id": "note-1", "text": "Task", "expected_note_version": 0}, "expected"),
        ("notes.tasks.update", {"task_id": "task-1", "expected_task_version": 1}, "At least one"),
        (
            "notes.tasks.set_status",
            {"updates": [{"task_id": "task-1", "status": "blocked", "expected_task_version": 1}]},
            "status",
        ),
        (
            "notes.tasks.set_status",
            {"updates": [{"task_id": "task-1", "status": "done", "expected_task_version": 0}]},
            "expected",
        ),
        (
            "notes.tasks.set_status",
            {
                "updates": [
                    {"task_id": f"task-{idx}", "status": "done", "expected_task_version": 1}
                    for idx in range(51)
                ]
            },
            "updates",
        ),
        ("notes.tasks.delete", {"task_id": "task-1", "expected_task_version": -1}, "expected"),
        ("notes.tasks.reconcile_note", {"note_id": ""}, "note_id"),
    ],
)
def test_notes_task_tool_validators_reject_invalid_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    error_match: str,
) -> None:
    module = NotesModule(ModuleConfig(name="notes"))

    with pytest.raises(ValueError, match=error_match):
        module.validate_tool_arguments(tool_name, arguments)


@pytest.mark.asyncio
async def test_notes_tasks_list_and_get_execute_read_only_with_reconciliation_summary() -> None:
    module, _db, service = _module_with_fakes()

    listed = await module.execute_tool("notes.tasks.list", {"status": "open", "reconcile_limit": 10}, context=_ctx())
    fetched = await module.execute_tool("notes.tasks.get", {"task_id": "task-1"}, context=_ctx())

    assert service.reconcile_stale_calls == 1  # nosec B101
    assert listed["reconciliation"]["status"] == "clean"  # nosec B101
    assert [task["id"] for task in listed["tasks"]] == ["task-1", "task-conflict"]  # nosec B101
    assert fetched["id"] == "task-1"  # nosec B101
    assert fetched["note"]["id"] == "note-1"  # nosec B101
    assert fetched["projection"]["projection_status"] == "live"  # nosec B101


@pytest.mark.asyncio
async def test_agent_write_requires_confirmation_and_autonomous_write_is_denied_without_mutation() -> None:
    module, db, service = _module_with_fakes()

    needs_approval = await module.execute_tool(
        "notes.tasks.create",
        {"note_id": "note-1", "text": "Agent task", "expected_note_version": 1},
        context=_ctx(agent_context={"agent_id": "agent-1"}),
    )
    denied = await module.execute_tool(
        "notes.tasks.create",
        {"note_id": "note-1", "text": "Autonomous task", "expected_note_version": 1},
        context=_ctx(agent_context={"agent_id": "agent-1", "autonomous": True}),
    )

    assert needs_approval["status"] == "approval_required"  # nosec B101
    assert needs_approval["policy_decision"]["action"] == "require_approval"  # nosec B101
    assert denied["status"] == "denied"  # nosec B101
    assert denied["policy_decision"]["action"] == "deny"  # nosec B101
    assert "activity_notice_required" in denied["policy_decision"]["reason_code"]  # nosec B101
    assert service.create_calls == 0  # nosec B101
    assert all(not task_id.startswith("task-created") for task_id in db.tasks)  # nosec B101

    allowed = await module.execute_tool(
        "notes.tasks.create",
        {"note_id": "note-1", "text": "Confirmed task", "expected_note_version": 1, "__confirm_write": True},
        context=_ctx(agent_context={"agent_id": "agent-1"}),
    )

    assert allowed["id"] == "task-created-1"  # nosec B101
    assert service.create_calls == 1  # nosec B101


@pytest.mark.asyncio
async def test_idempotency_key_reuses_create_and_status_retry_results_without_duplicate_mutation() -> None:
    module, _db, service = _module_with_fakes()
    context = _ctx()
    create_args = {
        "note_id": "note-1",
        "text": "Idempotent",
        "expected_note_version": 1,
        "idempotencyKey": "create-1",
    }

    first_create = await module.execute_tool("notes.tasks.create", create_args, context=context)
    second_create = await module.execute_tool("notes.tasks.create", dict(create_args), context=context)

    assert first_create == second_create  # nosec B101
    assert service.create_calls == 1  # nosec B101

    status_args = {
        "updates": [{"task_id": "task-1", "status": "done", "expected_task_version": 1, "expected_note_version": 1}],
        "idempotencyKey": "status-1",
    }
    first_status = await module.execute_tool("notes.tasks.set_status", status_args, context=context)
    second_status = await module.execute_tool("notes.tasks.set_status", dict(status_args), context=context)

    assert first_status == second_status  # nosec B101
    assert len([call for call in service.update_calls if call["task_id"] == "task-1"]) == 1  # nosec B101


@pytest.mark.asyncio
async def test_set_status_returns_succeeded_failed_and_skipped_for_partial_batch_conflicts() -> None:
    module, _db, service = _module_with_fakes()

    result = await module.execute_tool(
        "notes.tasks.set_status",
        {
            "updates": [
                {"task_id": "task-1", "status": "done", "expected_task_version": 1, "expected_note_version": 1},
                {"task_id": "task-same", "status": "done", "expected_task_version": 4, "expected_note_version": 1},
                {"task_id": "task-conflict", "status": "done", "expected_task_version": 2, "expected_note_version": 1},
            ]
        },
        context=_ctx(),
    )

    assert [item["task"]["id"] for item in result["succeeded"]] == ["task-1"]  # nosec B101
    assert [item["task_id"] for item in result["skipped"]] == ["task-same"]  # nosec B101
    assert result["skipped"][0]["reason"] == "already_done"  # nosec B101
    assert [item["task_id"] for item in result["failed"]] == ["task-conflict"]  # nosec B101
    assert result["failed"][0]["error_type"] == "ConflictError"  # nosec B101
    assert [call["task_id"] for call in service.update_calls] == ["task-1", "task-conflict"]  # nosec B101
