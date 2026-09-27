"""Automation LLM executor wiring tests (TASK-13110).

The LLM call is mocked at the ``perform_chat_api_call_async`` boundary —
the executor's own unit under test is prompt construction, target
resolution, registration, and the seam transition inside the real
consumer registry.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import (
    DefinitionRow,
    ScheduledTasksDatabase,
)
from tldw_Server_API.app.core.Scheduled_Tasks import agent_task_jobs as atj
from tldw_Server_API.app.core.Scheduled_Tasks.agent_task_jobs import (
    handle_agent_task_job,
)
from tldw_Server_API.app.core.Scheduled_Tasks.automation_executors import (
    _execute_generation_only,
    register_automation_executors,
    resolve_execution_target,
)
from tldw_Server_API.app.core.config import settings

pytestmark = pytest.mark.unit

SLOT = "2026-08-22T09:00:00+00:00"


def _definition(
    input_config: dict[str, Any], family: str = "recurring_question"
) -> DefinitionRow:
    """Build a DefinitionRow for executor tests."""
    return DefinitionRow(
        id="def-1",
        owner_id=7,
        version=1,
        family=family,
        name="Exec Test",
        description=None,
        lifecycle="configured",
        health="ready",
        disabled_lock_kind="none",
        disabled_reason=None,
        schedule={"kind": "daily", "at": "09:00"},
        input=input_config,
        visibility_policy="owner",
        notification_policy={},
        approval_policy={},
        preview_id="pv-1",
        created_by="t",
        updated_by="t",
        created_at="2026-08-22T00:00:00+00:00",
        updated_at="2026-08-22T00:00:00+00:00",
        resolution_state="open",
        resolved_at=None,
        resolved_by=None,
        resolved_result_id=None,
        finding_policy={},
        retention_policy={},
    )


# ---------------------------------------------------------------------------
# Target resolution (AC#3)
# ---------------------------------------------------------------------------


def test_resolution_definition_overrides_win() -> None:
    """Definition-level provider/model/max_tokens beat config defaults."""
    definition = _definition(
        {"question": "q", "provider": "openai", "model": "gpt-x", "max_tokens": 512}
    )
    target = resolve_execution_target(
        definition, config_section={"executor_provider": "anthropic", "executor_model": "m2"}
    )
    assert target == {"provider": "openai", "model": "gpt-x", "max_tokens": 512}


def test_resolution_config_defaults_when_definition_silent() -> None:
    """Config defaults apply when the definition carries no overrides."""
    definition = _definition({"question": "q"})
    target = resolve_execution_target(
        definition,
        config_section={"executor_provider": "anthropic", "executor_model": "m2"},
    )
    assert target == {"provider": "anthropic", "model": "m2", "max_tokens": 1000}


def test_resolution_server_default_when_both_silent() -> None:
    """Both silent -> server-default resolution (provider/model omitted)."""
    definition = _definition({"question": "q"})
    target = resolve_execution_target(definition, config_section={})
    assert target == {"provider": None, "model": None, "max_tokens": 1000}


def test_resolution_caps_and_floors_max_tokens() -> None:
    """max_tokens caps at 4000 and floors junk/negatives to the default."""
    assert resolve_execution_target(
        _definition({"question": "q", "max_tokens": 99999}), config_section={}
    )["max_tokens"] == 4000
    assert resolve_execution_target(
        _definition({"question": "q", "max_tokens": -5}), config_section={}
    )["max_tokens"] == 1000
    assert resolve_execution_target(
        _definition({"question": "q", "max_tokens": "junk"}), config_section={}
    )["max_tokens"] == 1000


# ---------------------------------------------------------------------------
# Prompt construction + the mocked LLM boundary (AC#1/#2/#5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recurring_question_builds_one_generation_only_call(monkeypatch) -> None:
    """One completion call: question as user message, no tools, bounded tokens."""
    captured: list[dict[str, Any]] = []

    async def _fake_call(**kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return {"choices": [{"message": {"content": "the answer"}}]}

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
        _fake_call,
    )

    definition = _definition({"question": "What changed today?"})
    text = await _execute_generation_only(definition, {})

    assert text == "the answer"
    assert len(captured) == 1
    call = captured[0]
    assert call["messages"] == [{"role": "user", "content": "What changed today?"}]
    assert "scheduled automation" in call["system_message"].lower()
    assert "tools" not in call and "tool_choice" not in call
    assert call["max_tokens"] == 1000
    # Both provider/model omitted -> server-default resolution inside the entrypoint.
    assert "api_provider" not in call and "model" not in call


def test_resolution_blank_overrides_fall_through() -> None:
    """Whitespace definition overrides cannot suppress config defaults."""
    definition = _definition(
        {"question": "q", "provider": "   ", "model": "", "max_tokens": "junk"}
    )
    target = resolve_execution_target(
        definition,
        config_section={"executor_provider": "anthropic", "executor_model": "m2"},
    )
    assert target == {"provider": "anthropic", "model": "m2", "max_tokens": 1000}


@pytest.mark.asyncio
async def test_missing_prompt_raises_lookup_error() -> None:
    """Missing usable prompt raises LookupError for an honest failed run."""
    with pytest.raises(LookupError):
        await _execute_generation_only(_definition({}), {})
    with pytest.raises(LookupError):
        await _execute_generation_only(_definition({}, family="agent_task"), {})


@pytest.mark.asyncio
async def test_empty_completion_raises() -> None:
    """An empty completion raises instead of recording a blank success."""
    async def _empty(**kwargs: Any) -> dict[str, Any]:
        return {"choices": [{"message": {"content": ""}}]}

    import tldw_Server_API.app.core.Chat.chat_service as cs

    original = cs.perform_chat_api_call_async
    cs.perform_chat_api_call_async = _empty
    try:
        with pytest.raises(RuntimeError):
            await _execute_generation_only(_definition({"question": "q"}), {})
    finally:
        cs.perform_chat_api_call_async = original


# ---------------------------------------------------------------------------
# Registration + the consumer-seam transition (AC#4/#6)
# ---------------------------------------------------------------------------


def test_registration_is_idempotent_and_fills_both_families(monkeypatch) -> None:
    """Registration is idempotent and wires only recurring_question (phase 1)."""
    monkeypatch.setattr(atj, "_EXECUTORS", {})
    register_automation_executors()
    first = dict(atj._EXECUTORS)
    register_automation_executors()
    assert set(atj._EXECUTORS) == {"recurring_question"}
    assert atj._EXECUTORS == first  # same callables, not re-created


@pytest.mark.asyncio
async def test_registered_executor_flows_through_the_consumer(monkeypatch, tmp_path):
    base_dir = tmp_path / "executor_consumer"
    base_dir.mkdir(parents=True, exist_ok=True)
    prev = settings.get("USER_DB_BASE_DIR")
    settings.USER_DB_BASE_DIR = str(base_dir)
    import os

    os.environ["USER_DB_BASE_DIR"] = str(base_dir)
    os.environ["JOBS_DB_PATH"] = str(base_dir / "jobs.db")
    try:
        user_id = 2020
        db = ScheduledTasksDatabase.for_user(user_id=user_id)
        db.ensure_schema()
        preview = db.create_preview(
            owner_id=user_id,
            mode="create",
            family="recurring_question",
            definition_id=None,
            definition_version=None,
            status="valid",
            payload_hash="h1",
            normalized_config={},
            validation_errors=[],
            warnings=[],
            risk_class=None,
            visibility_policy="owner",
            schedule_preview={"kind": "daily", "at": "09:00"},
            redaction_policy={"fields": [], "mode": "none"},
            expires_at=(
                datetime.now(timezone.utc) + timedelta(hours=24)
            ).isoformat(),
            created_by="t",
        )
        definition = db.create_definition(
            owner_id=user_id,
            family="recurring_question",
            name="Wired",
            description=None,
            lifecycle="configured",
            health="ready",
            schedule={"kind": "daily", "at": "09:00"},
            input={"question": "What changed?"},
            visibility_policy="owner",
            notification_policy={},
            approval_policy={},
            preview_id=preview.id,
            created_by="t",
            updated_by="t",
        )

        monkeypatch.setattr(atj, "_EXECUTORS", {})
        # Before wiring: honest failure.
        before = await handle_agent_task_job(
            {
                "id": 1,
                "owner_user_id": user_id,
                "payload": {
                    "definition_id": definition.id,
                    "user_id": user_id,
                    "family": "recurring_question",
                    "scheduled_for": SLOT,
                },
            },
            scheduled_db=db,
        )
        assert before["status"] == "skipped"
        # A DIFFERENT slot for the wired attempt: the prior slot's terminal
        # run row would dedupe and replay the failed outcome.
        wired_slot = "2026-08-22T10:00:00+00:00"

        # After wiring: the production executor runs (LLM mocked).
        register_automation_executors()

        async def _fake_call(**kwargs: Any) -> dict[str, Any]:
            return {"choices": [{"message": {"content": "generated answer"}}]}

        monkeypatch.setattr(
            "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
            _fake_call,
        )
        after = await handle_agent_task_job(
            {
                "id": 2,
                "owner_user_id": user_id,
                "payload": {
                    "definition_id": definition.id,
                    "user_id": user_id,
                    "family": "recurring_question",
                    "scheduled_for": wired_slot,
                },
            },
            scheduled_db=db,
        )
        assert after["status"] == "succeeded"
        run = db.get_scheduled_task_run_by_slot(
            definition_id=definition.id, run_slot_key=wired_slot
        )
        assert run is not None
        assert run["result_summary"] == "generated answer"
    finally:
        if prev is not None:
            settings.USER_DB_BASE_DIR = prev
        else:
            try:
                del settings.USER_DB_BASE_DIR
            except AttributeError:
                pass
