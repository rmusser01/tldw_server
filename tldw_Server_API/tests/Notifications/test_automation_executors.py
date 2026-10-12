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
    assert set(atj._EXECUTORS) == {"recurring_question", "agent_task"}
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


# ---------------------------------------------------------------------------
# agent_task executor (ADR-184 phase 2)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_task_resolves_message_from_store(monkeypatch) -> None:
    """The raw prompt comes from the encrypted store, in memory, at dispatch."""
    captured: list[dict[str, Any]] = []

    async def _fake_call(**kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return {"choices": [{"message": {"content": "agent answer"}}]}

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
        _fake_call,
    )

    resolved: list[tuple[int, str]] = []

    class _FakeStore:
        def resolve_message(self, owner_id: int, message_ref: str) -> str | None:
            resolved.append((owner_id, message_ref))
            return "RAW_AGENT_PROMPT"

    import tldw_Server_API.app.core.Scheduled_Tasks.automation_executors as executors_module

    monkeypatch.setattr(
        executors_module.AutomationMessageStore,
        "for_user",
        classmethod(lambda cls, user_id: _FakeStore()),
    )
    # agent_ref now resolves through the registry seam at prompt build.
    _fake_agent_registry(monkeypatch)

    definition = _definition(
        {"agent_ref": "agent:triage", "message_redacted": True,
         "message_ref": "ref-9"},
        family="agent_task",
    )
    text = await executors_module._execute_agent_task(definition, {})

    assert text == "agent answer"
    assert resolved == [(7, "ref-9")]
    assert captured[0]["messages"] == [
        {"role": "user", "content": "RAW_AGENT_PROMPT"}
    ]


@pytest.mark.asyncio
async def test_agent_task_unresolvable_ref_fails_honestly(monkeypatch) -> None:
    """A purged/missing ref raises with a precise reason (failed run upstream)."""
    import tldw_Server_API.app.core.Scheduled_Tasks.automation_executors as executors_module

    class _EmptyStore:
        def resolve_message(self, owner_id: int, message_ref: str) -> str | None:
            return None

    monkeypatch.setattr(
        executors_module.AutomationMessageStore,
        "for_user",
        classmethod(lambda cls, user_id: _EmptyStore()),
    )

    definition = _definition(
        {"agent_ref": "agent:triage", "message_ref": "ref-gone"},
        family="agent_task",
    )
    with pytest.raises(LookupError, match="unresolvable"):
        await executors_module._execute_agent_task(definition, {})


@pytest.mark.asyncio
async def test_agent_task_without_ref_fails_before_store(monkeypatch) -> None:
    """Pre-store definitions (no ref) fail with the authoring-era reason."""
    import tldw_Server_API.app.core.Scheduled_Tasks.automation_executors as executors_module

    called = {"resolve": False}

    class _SpyStore:
        def resolve_message(self, owner_id: int, message_ref: str) -> str | None:
            called["resolve"] = True
            return None

    monkeypatch.setattr(
        executors_module.AutomationMessageStore,
        "for_user",
        classmethod(lambda cls, user_id: _SpyStore()),
    )

    definition = _definition(
        {"agent_ref": "agent:triage", "message_redacted": True},
        family="agent_task",
    )
    with pytest.raises(LookupError, match="no message_ref"):
        await executors_module._execute_agent_task(definition, {})
    assert called["resolve"] is False


# ---------------------------------------------------------------------------
# Read-only eligible-set derivation (ADR-184 read-only tool envelope)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_eligible_set_filters_by_declared_tier_and_permission(monkeypatch) -> None:
    catalogue = {"tools": [
        {"name": "media_search", "module": "media", "canExecute": True},
        {"name": "notes_write", "module": "notes", "canExecute": True},
        {"name": "chats_get", "module": "chats", "canExecute": False},
        {"name": "mystery_tool", "module": "not_a_module", "canExecute": True},
    ]}

    class _FakeToolsExecutor:
        async def list_tools(self, *, user_id, client_id):
            return catalogue

    import tldw_Server_API.app.core.Scheduled_Tasks.automation_executors as em
    monkeypatch.setattr(em, "_automation_tools_executor", lambda: _FakeToolsExecutor())
    eligible = await em.derive_read_only_eligible_tools(7)
    assert [t["name"] for t in eligible] == ["media_search"]

@pytest.mark.asyncio
async def test_eligible_set_fails_closed_on_catalogue_error(monkeypatch) -> None:
    import tldw_Server_API.app.core.Scheduled_Tasks.automation_executors as em

    class _Broken:
        async def list_tools(self, *, user_id, client_id):
            raise RuntimeError("mcp down")

    monkeypatch.setattr(em, "_automation_tools_executor", lambda: _Broken())
    assert await em.derive_read_only_eligible_tools(7) == []


# ---------------------------------------------------------------------------
# Bounded read-only tool loop (ADR-184 read-only tool envelope, task 3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_envelope_runs_tool_round_then_answers(monkeypatch) -> None:
    import tldw_Server_API.app.core.Scheduled_Tasks.automation_executors as em
    calls = []

    async def _fake_llm(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {"choices": [{"message": {"content": "", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "media_search", "arguments": "{\"q\": \"x\"}"}}]}}]}
        return {"choices": [{"message": {"content": "final answer"}}]}

    monkeypatch.setattr(em, "_envelope_llm_call", _fake_llm)

    async def _fake_exec(**kwargs):
        from tldw_Server_API.app.core.Chat.tool_auto_exec import ToolExecutionBatchResult
        # use the real batch builder via a minimal stub: return one success record
        rec = em._stub_tool_success("c1", "media_search", "result text")
        return em._batch_from_records([rec])

    monkeypatch.setattr(em, "_envelope_execute_calls", _fake_exec)
    result = await em.run_read_only_envelope(_definition({}, family="agent_task"), "do it",
        [{"name": "media_search", "module": "media"}])
    assert result.status == "succeeded" and result.text == "final answer"
    assert calls[1]["messages"][-1]["role"] == "tool"


@pytest.mark.asyncio
async def test_envelope_bounds_exhausted_succeeds_with_note(monkeypatch) -> None:
    """Tool rounds that never converge fall back to generation, note says truncat*."""
    import tldw_Server_API.app.core.Scheduled_Tasks.automation_executors as em

    async def _always_tool_calls(**kwargs):
        return {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "media_search", "arguments": "{}"}}]}}]}

    monkeypatch.setattr(em, "_envelope_llm_call", _always_tool_calls)

    async def _fake_exec(**kwargs):
        rec = em._stub_tool_success("c1", "media_search", "loop result")
        return em._batch_from_records([rec])

    monkeypatch.setattr(em, "_envelope_execute_calls", _fake_exec)

    async def _fake_generation(**kwargs):
        return {"choices": [{"message": {"content": "fallback answer"}}]}

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
        _fake_generation,
    )

    result = await em.run_read_only_envelope(
        _definition({}, family="agent_task"), "do it",
        [{"name": "media_search", "module": "media"}],
        max_rounds=3,
    )
    assert result.status == "succeeded"
    assert result.text == "fallback answer"
    assert "truncat" in (result.note or "")


@pytest.mark.asyncio
async def test_envelope_out_of_set_call_is_approval_required(monkeypatch) -> None:
    """A tool call outside the eligible set is classified, never executed."""
    import tldw_Server_API.app.core.Scheduled_Tasks.automation_executors as em

    async def _rogue_tool_call(**kwargs):
        return {"choices": [{"message": {"content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "notes_write", "arguments": "{}"}}]}}]}

    monkeypatch.setattr(em, "_envelope_llm_call", _rogue_tool_call)

    executed: list[dict] = []

    async def _must_not_execute(**kwargs):
        executed.append(kwargs)
        raise AssertionError("out-of-set tool call must never reach execution")

    monkeypatch.setattr(em, "_envelope_execute_calls", _must_not_execute)

    result = await em.run_read_only_envelope(
        _definition({}, family="agent_task"), "do it",
        [{"name": "media_search", "module": "media"}])
    assert result.status == "approval_required"
    assert "notes_write" in (result.note or "")
    assert executed == []


@pytest.mark.asyncio
async def test_envelope_presented_empty_note(monkeypatch) -> None:
    """Eligible tools the presentation layer drops: plain generation + note."""
    from tldw_Server_API.app.core.Chat.run_first_presentation import (
        ChatRunFirstPresentation,
    )

    def _presents_nothing(**kwargs):
        return ChatRunFirstPresentation(
            llm_tools=[],
            effective_tool_names=[],
            prompt_fragment=None,
            presentation_variant="test",
            eligible=False,
            ineligible_reason="no_effective_tools",
        )

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Chat.chat_service.present_chat_tools",
        _presents_nothing,
    )

    async def _fake_generation(**kwargs):
        return {"choices": [{"message": {"content": "plain answer"}}]}

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
        _fake_generation,
    )

    import tldw_Server_API.app.core.Scheduled_Tasks.automation_executors as em

    result = await em.run_read_only_envelope(
        _definition({}, family="agent_task"), "do it",
        [{"name": "media_search", "module": "media"}])
    assert result.status == "succeeded"
    assert result.text == "plain answer"
    assert "not presented" in (result.note or "")


# ---------------------------------------------------------------------------
# Envelope wiring inside _execute_agent_task (ADR-184 task 4)
# ---------------------------------------------------------------------------


def _fake_message_store(monkeypatch) -> None:
    """Resolve any message_ref to a fixed raw prompt (no store on disk)."""
    import tldw_Server_API.app.core.Scheduled_Tasks.automation_executors as em

    class _FakeStore:
        def resolve_message(self, owner_id: int, message_ref: str) -> str | None:
            return "RAW_AGENT_PROMPT"

    monkeypatch.setattr(
        em.AutomationMessageStore,
        "for_user",
        classmethod(lambda cls, user_id: _FakeStore()),
    )


class _FakeAgentRegistryEntry:
    """Entry stand-in: carries a persona prompt when one is given."""

    def __init__(self, system_prompt: str | None = None) -> None:
        self.system_prompt = system_prompt


class _FakeAgentRegistry:
    """Registry stand-in matching the ``.get(ref)`` seam contract."""

    def __init__(self, entries: dict[str, Any]) -> None:
        self._entries = entries

    def get(self, ref: str) -> Any:
        return self._entries.get(ref)


def _fake_agent_registry(
    monkeypatch, entries: dict[str, Any] | None = None
) -> None:
    """Patch the registry seam: 'agent:triage' resolves (no persona by default)."""
    import tldw_Server_API.app.core.Scheduled_Tasks.automation_executors as em

    resolved = entries if entries is not None else {"agent:triage": _FakeAgentRegistryEntry()}
    monkeypatch.setattr(em, "_resolve_agent_registry", lambda: _FakeAgentRegistry(resolved))


@pytest.mark.asyncio
async def test_agent_task_with_tools_runs_envelope_with_rederived_set(monkeypatch) -> None:
    """Tools requested -> the eligible set is re-derived HERE, hint never read."""
    import tldw_Server_API.app.core.Scheduled_Tasks.automation_executors as em

    _fake_message_store(monkeypatch)
    derived: list[int] = []

    async def _fake_derive(owner_id: int):
        derived.append(owner_id)
        return [{"name": "media_search", "module": "media"}]

    monkeypatch.setattr(em, "derive_read_only_eligible_tools", _fake_derive)

    envelope_calls: list[tuple[str, list[dict[str, Any]]]] = []

    async def _fake_envelope(definition, raw_message, eligible, **kwargs):
        envelope_calls.append((raw_message, eligible))
        return em.EnvelopeResult("succeeded", "envelope answer", None)

    monkeypatch.setattr(em, "run_read_only_envelope", _fake_envelope)

    definition = _definition(
        {"agent_ref": "agent:triage", "message_ref": "ref-9",
         "tools": ["media_search"]},
        family="agent_task",
    )
    # A deliberately stale audit hint: the executor re-derives the
    # authoritative set and must never trust the payload key.
    text = await em._execute_agent_task(definition, {"_envelope_eligible": ["stale"]})

    assert text == "envelope answer"
    assert derived == [7]
    assert envelope_calls == [
        ("RAW_AGENT_PROMPT", [{"name": "media_search", "module": "media"}])
    ]


@pytest.mark.asyncio
async def test_agent_task_envelope_approval_required_raises_prefixed(monkeypatch) -> None:
    """Envelope approval_required surfaces as the documented error prefix."""
    import tldw_Server_API.app.core.Scheduled_Tasks.automation_executors as em

    _fake_message_store(monkeypatch)

    async def _fake_derive(owner_id: int):
        return [{"name": "media_search", "module": "media"}]

    monkeypatch.setattr(em, "derive_read_only_eligible_tools", _fake_derive)

    async def _fake_envelope(definition, raw_message, eligible, **kwargs):
        return em.EnvelopeResult(
            "approval_required", "", "attempted non-envelope tool: notes_write"
        )

    monkeypatch.setattr(em, "run_read_only_envelope", _fake_envelope)

    definition = _definition(
        {"agent_ref": "agent:triage", "message_ref": "ref-9",
         "tools": ["media_search"]},
        family="agent_task",
    )
    with pytest.raises(
        RuntimeError,
        match=r"^approval_required: attempted non-envelope tool: notes_write",
    ):
        await em._execute_agent_task(definition, {})


@pytest.mark.asyncio
async def test_agent_task_tools_with_empty_eligible_fails_closed_to_generation(
    monkeypatch,
) -> None:
    """An empty re-derived set presents nothing: plain generation, no envelope."""
    import tldw_Server_API.app.core.Scheduled_Tasks.automation_executors as em

    _fake_message_store(monkeypatch)
    _fake_agent_registry(monkeypatch)

    async def _empty_derive(owner_id: int):
        return []

    monkeypatch.setattr(em, "derive_read_only_eligible_tools", _empty_derive)

    async def _must_not_run(definition, raw_message, eligible, **kwargs):
        raise AssertionError("envelope must not run with an empty eligible set")

    monkeypatch.setattr(em, "run_read_only_envelope", _must_not_run)

    async def _fake_generation(**kwargs: Any) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "plain answer"}}]}

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
        _fake_generation,
    )

    definition = _definition(
        {"agent_ref": "agent:triage", "message_ref": "ref-9",
         "tools": ["media_search"]},
        family="agent_task",
    )
    assert await em._execute_agent_task(definition, {}) == "plain answer"


@pytest.mark.asyncio
async def test_agent_task_without_tools_skips_derivation_entirely(monkeypatch) -> None:
    """No tools requested: no catalogue read, no envelope, generation only."""
    import tldw_Server_API.app.core.Scheduled_Tasks.automation_executors as em

    _fake_message_store(monkeypatch)
    _fake_agent_registry(monkeypatch)

    async def _must_not_derive(owner_id: int):
        raise AssertionError("eligible-set derivation must not run without tools")

    monkeypatch.setattr(em, "derive_read_only_eligible_tools", _must_not_derive)

    async def _fake_generation(**kwargs: Any) -> dict[str, Any]:
        assert "tools" not in kwargs
        return {"choices": [{"message": {"content": "gen only"}}]}

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
        _fake_generation,
    )

    definition = _definition(
        {"agent_ref": "agent:triage", "message_ref": "ref-9"}, family="agent_task"
    )
    assert await em._execute_agent_task(definition, {}) == "gen only"


# ---------------------------------------------------------------------------
# agent_ref persona resolution (ADR-184 envelope, task 5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_ref_resolves_system_prompt(monkeypatch) -> None:
    """A resolvable agent_ref runs under the entry's persona as system prompt."""
    import tldw_Server_API.app.core.Scheduled_Tasks.automation_executors as em

    _fake_message_store(monkeypatch)
    _fake_agent_registry(
        monkeypatch,
        {"agent:triage": _FakeAgentRegistryEntry("You are triage.")},
    )

    captured: list[dict[str, Any]] = []

    async def _fake_call(**kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return {"choices": [{"message": {"content": "triaged"}}]}

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
        _fake_call,
    )

    definition = _definition(
        {"agent_ref": "agent:triage", "message_ref": "ref-9"}, family="agent_task"
    )
    assert await em._execute_agent_task(definition, {}) == "triaged"
    assert captured[0]["system_message"] == "You are triage."


@pytest.mark.asyncio
async def test_unknown_agent_ref_fails_honestly(monkeypatch) -> None:
    """Unknown/blocked agent_ref: LookupError naming the reason, LLM never called."""
    import tldw_Server_API.app.core.Scheduled_Tasks.automation_executors as em

    _fake_message_store(monkeypatch)
    _fake_agent_registry(monkeypatch, entries={})  # resolves nothing

    called: list[dict[str, Any]] = []

    async def _must_not_call(**kwargs: Any) -> dict[str, Any]:
        called.append(kwargs)
        raise AssertionError("the LLM must not be called for an unknown agent_ref")

    monkeypatch.setattr(
        "tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call_async",
        _must_not_call,
    )

    definition = _definition(
        {"agent_ref": "agent:ghost", "message_ref": "ref-9"}, family="agent_task"
    )
    with pytest.raises(LookupError, match="unknown or blocked"):
        await em._execute_agent_task(definition, {})
    assert called == []


@pytest.mark.asyncio
async def test_agent_ref_entry_without_persona_names_the_agent(monkeypatch) -> None:
    """Real AgentRegistryEntry carries no persona: the prompt names the agent.

    The fallback keeps the generation-only base (bounded, unattended
    semantics) prefixed with the resolved agent's identity.
    """
    import tldw_Server_API.app.core.Scheduled_Tasks.automation_executors as em

    _fake_agent_registry(monkeypatch)  # 'agent:triage' -> entry, no persona

    prompt = em._definition_system_prompt(
        _definition({"agent_ref": "agent:triage"}, family="agent_task")
    )
    assert prompt.startswith("You are the scheduled agent 'agent:triage'. ")
    assert "scheduled automation" in prompt
