"""Encrypted message store + authoring wiring for agent_task (ADR-184 1A).

Decision 1A: the raw agent message lives only in the per-owner encrypted
store keyed by ``message_ref``; the scheduled-tasks DB keeps its
``metadata_only`` posture. These tests pin the store's contract (roundtrip,
owner scoping, TTL, corruption, key refusal) and the preview-create wiring
(store before row; refusal when the store is down; no store writes for
recurring_question).
"""

from __future__ import annotations

import base64
import sqlite3
from typing import Any

import pytest

from tldw_Server_API.app.api.v1.schemas.scheduled_tasks_automation_schemas import (
    ScheduledTaskPreviewCreateRequest,
)
from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import (
    ScheduledTasksDatabase,
)
from tldw_Server_API.app.core.Scheduled_Tasks import automation_message_store as store_module
from tldw_Server_API.app.core.Scheduled_Tasks.automation_message_store import (
    AutomationMessageStore,
)
from tldw_Server_API.app.services.scheduled_task_automation_service import (
    ScheduledTaskAutomationError,
    ScheduledTaskAutomationService,
)

OWNER_ID = 4310
OTHER_OWNER_ID = 4311
ACTOR = "message-store-test"
RAW = "RAW_AGENT_PROMPT_DO_NOT_LEAK_1F"

_TEST_KEY = base64.urlsafe_b64encode(b"0" * 32).decode()


@pytest.fixture(autouse=True)
def _test_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(store_module, "_message_store_keys", lambda: (_TEST_KEY, None))


def _store(tmp_path: Any) -> AutomationMessageStore:
    return AutomationMessageStore(tmp_path / "message_store.db")


def _agent_preview_payload(message: str = RAW) -> ScheduledTaskPreviewCreateRequest:
    return ScheduledTaskPreviewCreateRequest(
        mode="create",
        family="agent_task",
        name="Triage agent",
        config={},
        input={"agent_ref": "agent:triage", "message": message},
        schedule={"kind": "daily", "time": "09:00", "timezone": "UTC"},
        visibility_policy={"mode": "metadata_only"},
    )


def test_roundtrip_resolves_and_blob_holds_no_plaintext(tmp_path):
    store = _store(tmp_path)
    store.store_message(OWNER_ID, "ref-1", RAW)

    assert store.resolve_message(OWNER_ID, "ref-1") == RAW  # nosec B101
    with sqlite3.connect(str(store.db_path)) as conn:
        blob = conn.execute(
            "SELECT encrypted_blob FROM automation_messages WHERE message_ref = ?",
            ("ref-1",),
        ).fetchone()[0]
    assert RAW not in str(blob)  # nosec B101


def test_resolve_is_owner_scoped(tmp_path):
    store = _store(tmp_path)
    store.store_message(OWNER_ID, "ref-1", RAW)

    assert store.resolve_message(OTHER_OWNER_ID, "ref-1") is None  # nosec B101


def test_expired_ref_resolves_none_and_purges(tmp_path):
    store = _store(tmp_path)
    store.store_message(OWNER_ID, "ref-1", RAW)
    # Force the expiry into the past.
    with sqlite3.connect(str(store.db_path)) as conn:
        conn.execute(
            "UPDATE automation_messages SET expires_at = '2000-01-01T00:00:00+00:00'"
            " WHERE message_ref = ?",
            ("ref-1",),
        )

    assert store.resolve_message(OWNER_ID, "ref-1") is None  # nosec B101
    assert store.purge_expired() == 1  # nosec B101
    assert store.resolve_message(OWNER_ID, "ref-1") is None  # nosec B101


def test_corrupt_blob_resolves_none(tmp_path):
    store = _store(tmp_path)
    store.store_message(OWNER_ID, "ref-1", RAW)
    with sqlite3.connect(str(store.db_path)) as conn:
        conn.execute(
            "UPDATE automation_messages SET encrypted_blob = 'not-json'"
            " WHERE message_ref = ?",
            ("ref-1",),
        )

    assert store.resolve_message(OWNER_ID, "ref-1") is None  # nosec B101


def test_unconfigured_key_refuses_store(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "_message_store_keys", lambda: (None, None))
    store = _store(tmp_path)

    with pytest.raises(RuntimeError):
        store.store_message(OWNER_ID, "ref-1", RAW)


def _service_with_store(
    tmp_path: Any, store: AutomationMessageStore
) -> tuple[ScheduledTaskAutomationService, ScheduledTasksDatabase]:
    repo = ScheduledTasksDatabase(tmp_path / "scheduled_tasks_ms.db")
    repo.ensure_schema()
    return (
        ScheduledTaskAutomationService(repository=repo, message_store=store),
        repo,
    )


def test_agent_preview_stores_message_and_row_keeps_only_metadata(tmp_path):
    store = _store(tmp_path)
    service, repo = _service_with_store(tmp_path, store)

    preview = service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=_agent_preview_payload(),
    )

    stored = repo.get_preview(owner_id=OWNER_ID, preview_id=preview.id)
    assert stored is not None  # nosec B101
    row_json = str(stored.normalized_config)
    assert RAW not in row_json  # nosec B101
    assert "message_redacted" in row_json  # nosec B101
    message_ref = stored.normalized_config["input"]["message_ref"]
    assert message_ref  # nosec B101
    assert store.resolve_message(OWNER_ID, message_ref) == RAW  # nosec B101


def test_recurring_question_preview_never_touches_the_store(tmp_path):
    class SpyStore(AutomationMessageStore):
        def __init__(self) -> None:
            super().__init__("/nonexistent/spy.db")
            self.calls: list[tuple] = []

        def store_message(self, *args: Any, **kwargs: Any) -> None:
            self.calls.append(args)

    spy = SpyStore()
    repo = ScheduledTasksDatabase(tmp_path / "scheduled_tasks_spy.db")
    repo.ensure_schema()
    service = ScheduledTaskAutomationService(repository=repo, message_store=spy)

    service.create_preview(
        owner_id=OWNER_ID,
        actor=ACTOR,
        payload=ScheduledTaskPreviewCreateRequest(
            mode="create",
            family="recurring_question",
            name="Daily check",
            config={},
            input={"question": "What changed?"},
            schedule={"kind": "daily", "time": "09:00", "timezone": "UTC"},
            visibility_policy={"mode": "findings_only"},
        ),
    )

    assert spy.calls == []  # nosec B101


def test_store_failure_refuses_authoring_without_persisting(tmp_path):
    class DownStore(AutomationMessageStore):
        def store_message(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("store down")

    repo = ScheduledTasksDatabase(tmp_path / "scheduled_tasks_down.db")
    repo.ensure_schema()
    service = ScheduledTaskAutomationService(
        repository=repo, message_store=DownStore("/nonexistent/down.db")
    )

    with pytest.raises(ScheduledTaskAutomationError) as excinfo:
        service.create_preview(
            owner_id=OWNER_ID,
            actor=ACTOR,
            payload=_agent_preview_payload(),
        )
    assert excinfo.value.code == "message_store_unavailable"  # nosec B101
    rows, total = repo.list_previews(owner_id=OWNER_ID, limit=10, offset=0)
    assert rows == [] and total == 0  # nosec B101
