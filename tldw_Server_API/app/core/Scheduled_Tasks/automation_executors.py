"""Production LLM executors for scheduled automations (TASK-13110).

Wires the TASK-13021 consumer's executor seam to the server's canonical
chat entrypoint (``perform_chat_api_call_async`` — the same surface the
Flashcards, Research, and MCP modules use). Phase-1 scope per tldw_chatbook
ADR-077 decision 4 (owner-accepted): generation-only completions. The one
routed exception (ADR-184 read-only envelope) is ``_execute_agent_task``
with a tool-requesting input: it derives the owner's read-only eligible
tools and runs them inside the bounded envelope; every other path stays
generation-only and never passes ``tools``/``tool_choice``.

Model precedence (definition first, server last):

1. the definition's ``input``/``config`` ``model`` (+ optional ``provider``)
2. automation executor defaults from the server config
   (``[Scheduled_Tasks_Automation] executor_provider`` / ``executor_model``)
3. omit both and let ``perform_chat_api_call_async`` resolve the server's
   configured default provider/model

System-prompt precedence (``_definition_system_prompt``): an authored
``input.system_prompt`` override, then the resolved ``input.agent_ref``
entry's persona (unknown/blocked refs fail honestly via ``LookupError``;
persona-less entries get an agent-naming fallback), then the
generation-only base.

Credentials resolve through the existing provider-config layer inside the
chat entrypoint — this module adds no secret handling of its own.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.Automation_Message_Store_DB import (
    AutomationMessageStore,
)
from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import DefinitionRow
from tldw_Server_API.app.core.Scheduled_Tasks.agent_task_jobs import (
    APPROVAL_REQUIRED_ERROR_PREFIX,
    _phase1_tools_requested,
    register_executor,
)
from tldw_Server_API.app.core.Workflows.adapters._common import extract_openai_content

if TYPE_CHECKING:
    from tldw_Server_API.app.core.Chat.tool_auto_exec import (
        ToolExecutionBatchResult,
        ToolExecutionRecord,
    )

#: Fixed, generation-only system prompt (phase 1: no tool use, bounded output).
_GENERATION_ONLY_SYSTEM_PROMPT = (
    "You are a scheduled automation assistant. Answer the user's request "
    "directly and concisely in plain text. This is an unattended scheduled "
    "run: do not ask questions, do not request tools or side effects, and "
    "keep the answer self-contained."
)

_DEFAULT_MAX_TOKENS = 1000
_MAX_TOKENS_CAP = 4000

_EXECUTOR_SYSTEM_PROMPT_KEY = "system_prompt"
_REGISTERED = False


def _config_section() -> dict[str, Any]:
    """Return the ``[Scheduled_Tasks_Automation]`` config section (may be empty)."""
    try:
        from tldw_Server_API.app.core.config import settings

        section = settings.get("Scheduled_Tasks_Automation")
        return section if isinstance(section, dict) else {}
    except Exception:  # noqa: BLE001 - degrade to defaults, never break a run
        logger.warning(
            "Automation executor config read failed; using built-in defaults"
        )
        return {}


def _as_positive_int(value: Any, fallback: int) -> int:
    """Coerce a config value to a positive int, tolerating junk."""
    if isinstance(value, bool) or value is None:
        return fallback
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def resolve_execution_target(
    definition: DefinitionRow, *, config_section: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Resolve the provider/model/max_tokens one run will use.

    Precedence: definition ``input``/``config`` overrides, then the
    automation config defaults (``executor_provider``/``executor_model``/
    ``executor_max_tokens``), then server-default resolution (both keys
    omitted so the chat entrypoint applies its own configured default).
    """
    section = config_section if config_section is not None else _config_section()
    source: dict[str, Any] = (
        definition.input if isinstance(definition.input, dict) else {}
    )

    # Sanitize each layer BEFORE applying precedence, so a blank/junk
    # definition override falls through to the config default instead of
    # suppressing it (review #5 on PR #2804).
    def _coerce_nonempty_str(value: Any) -> str | None:
        if value is None or isinstance(value, bool):
            return None
        text = str(value).strip()
        return text or None

    provider = _coerce_nonempty_str(source.get("provider")) or _coerce_nonempty_str(
        section.get("executor_provider")
    )
    model = _coerce_nonempty_str(source.get("model")) or _coerce_nonempty_str(
        section.get("executor_model")
    )
    definition_max = _as_positive_int(source.get("max_tokens"), 0)
    max_tokens = definition_max or _as_positive_int(
        section.get("executor_max_tokens"), _DEFAULT_MAX_TOKENS
    )
    return {
        "provider": provider,
        "model": model,
        "max_tokens": min(max_tokens, _MAX_TOKENS_CAP),
    }


def _definition_user_prompt(definition: DefinitionRow) -> str:
    """Extract the generation-only user prompt from a definition's input.

    Only ``recurring_question`` takes this path directly — its ``question``
    persists in full. ``agent_task`` input is redacted at rest by the
    automation service (``message_redacted`` metadata replaces the raw
    message); that family resolves its prompt from the encrypted store in
    ``_execute_agent_task`` instead (ADR-184 1A), so a usable prompt
    surviving THIS path is not a state production reaches for it — if one
    ever appears it is used, otherwise the LookupError records an honest
    failed run.
    """
    source: dict[str, Any] = (
        definition.input if isinstance(definition.input, dict) else {}
    )
    question = str(source.get("question") or "").strip()
    if question:
        return question
    raise LookupError(
        f"{definition.family} definition has no usable persisted prompt "
        "(input.question missing; agent_task messages are redacted at rest)"
    )


#: Persona/prompt attributes an agent-registry entry may carry, in
#: preference order. The real ``AgentRegistryEntry`` (agent_registry.py)
#: carries none of these today — its closest text field, ``description``,
#: is human marketing copy, not a run persona — so real entries take the
#: agent-naming fallback below.
_AGENT_PERSONA_ATTRS = ("system_prompt", "persona", "prompt")


class _AgentRegistryLookup:
    """Expose ``.get(ref)`` over the real ``AgentRegistry.get_entry``."""

    def __init__(self, registry: Any) -> None:
        self._registry = registry

    def get(self, ref: str) -> Any:
        try:
            return self._registry.get_entry(ref)
        except Exception:  # noqa: BLE001 — degrade to unknown, fail closed
            return None


def _resolve_agent_registry() -> Any:
    """Return an object with ``.get(ref)`` -> entry-or-None, or ``None``.

    Patch seam for tests. Default: lazily import and construct the real
    ``AgentRegistry``; any failure returns ``None`` so agent_ref
    resolution fails closed without tests ever needing the real registry.
    """
    try:
        from tldw_Server_API.app.core.Agent_Client_Protocol.agent_registry import (
            AgentRegistry,
        )

        return _AgentRegistryLookup(AgentRegistry())
    except Exception:  # noqa: BLE001 — registry unavailable -> fail closed
        logger.warning(
            "Agent registry unavailable; agent_ref resolution will fail closed"
        )
        return None


def _agent_ref_system_prompt(agent_ref: str) -> str:
    """Resolve one agent_ref to its system prompt (ADR-184 envelope, task 5).

    Raises:
        LookupError: the ref is unknown or blocked (registry unavailable
            or agent not found) — an honest failed run, never a silent
            persona-less fallback for an agent that does not exist.
    """
    registry = _resolve_agent_registry()
    entry = registry.get(agent_ref) if registry is not None else None
    if entry is None:
        raise LookupError(
            f"agent_ref {agent_ref!r} unknown or blocked "
            "(registry unavailable or agent not found)"
        )
    for attr in _AGENT_PERSONA_ATTRS:
        value = getattr(entry, attr, None)
        if value is None or isinstance(value, bool):
            continue
        text = str(value).strip()
        if text:
            return text
    # AgentRegistryEntry carries no persona field: name the resolved agent
    # over the generation-only base so the run still says whom it acts as.
    return (
        f"You are the scheduled agent '{agent_ref}'. "
        + _GENERATION_ONLY_SYSTEM_PROMPT
    )


def _definition_system_prompt(definition: DefinitionRow) -> str:
    """Return the system prompt, allowing a definition-level override.

    Precedence: an authored ``input.system_prompt`` override wins; a
    non-empty ``input.agent_ref`` must still RESOLVE (unknown/blocked
    refs raise ``LookupError`` even when an override exists), and the
    entry's persona is the prompt when no override is authored; otherwise
    the generation-only base.
    """
    source: dict[str, Any] = (
        definition.input if isinstance(definition.input, dict) else {}
    )
    override = str(source.get(_EXECUTOR_SYSTEM_PROMPT_KEY) or "").strip()
    agent_ref = str(source.get("agent_ref") or "").strip()
    if agent_ref:
        entry_prompt = _agent_ref_system_prompt(agent_ref)
        return override or entry_prompt
    return override or _GENERATION_ONLY_SYSTEM_PROMPT


async def _run_generation(definition: DefinitionRow, user_prompt: str) -> str:
    """One generation-only completion (shared by both executor families)."""
    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

    target = resolve_execution_target(definition)
    call_kwargs: dict[str, Any] = {
        "messages": [{"role": "user", "content": user_prompt}],
        "system_message": _definition_system_prompt(definition),
        "max_tokens": target["max_tokens"],
    }
    if target["provider"]:
        call_kwargs["api_provider"] = target["provider"]
    if target["model"]:
        call_kwargs["model"] = target["model"]

    response = await perform_chat_api_call_async(**call_kwargs)
    text = (extract_openai_content(response) or "").strip()
    if not text:
        raise RuntimeError("automation executor received an empty completion")
    return text


def _automation_tools_executor():
    from tldw_Server_API.app.core.Tools.tool_executor import ToolExecutor
    return ToolExecutor()


async def derive_read_only_eligible_tools(owner_id: int) -> list[dict[str, Any]]:
    """Catalogue entries the owner can execute from read_only-declared modules.

    Fail closed: any catalogue failure yields an empty set — nothing is
    presented, nothing can execute (spec §3).
    """
    try:
        # NOTE: package is `MCP_unified` (lowercase u) — the task brief's
        # `MCP_Unified` spelling raises ModuleNotFoundError, which the
        # fail-closed except below would silently swallow (empty set even
        # for a healthy catalogue).
        from tldw_Server_API.app.core.MCP_unified.module_surface import MODULE_RISK_TIERS
        listing = await _automation_tools_executor().list_tools(
            user_id=str(owner_id), client_id="scheduled_automation"
        )
    except Exception:  # noqa: BLE001 — fail closed by construction
        return []
    eligible = []
    for entry in listing.get("tools", []):
        if not isinstance(entry, dict) or not entry.get("canExecute"):
            continue
        tier = MODULE_RISK_TIERS.get(str(entry.get("module")), ("", ""))
        if tier[0] == "read_only":
            eligible.append(entry)
    return eligible


# ---------------------------------------------------------------------------
# Read-only tool envelope: the bounded tool-calling loop (ADR-184)
# ---------------------------------------------------------------------------


@dataclass
class EnvelopeResult:
    """Outcome of one bounded read-only envelope run.

    ``status`` is one of ``{"succeeded", "approval_required", "failed"}``.
    ``note`` carries truncation / not-presented context, and names the
    attempted tool for membership violations.
    """

    status: str
    text: str
    note: str | None = None


async def _envelope_llm_call(**kwargs: Any) -> Any:
    """LLM-call seam for the envelope loop (tests monkeypatch this)."""
    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

    return await perform_chat_api_call_async(**kwargs)


async def _envelope_execute_calls(**kwargs: Any) -> Any:
    """Tool-execution seam for the envelope loop (tests monkeypatch this)."""
    from tldw_Server_API.app.core.Chat.tool_auto_exec import execute_assistant_tool_calls

    return await execute_assistant_tool_calls(**kwargs)


def _stub_tool_success(
    tool_call_id: str, tool_name: str, output: str
) -> ToolExecutionRecord:
    """Test helper: one successful execution record (real dataclass shape)."""
    from tldw_Server_API.app.core.Chat.tool_auto_exec import ToolExecutionRecord

    return ToolExecutionRecord(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        ok=True,
        content=output,
        result=output,
    )


def _batch_from_records(
    records: list[ToolExecutionRecord],
) -> ToolExecutionBatchResult:
    """Test helper: wrap records in a real ToolExecutionBatchResult."""
    from tldw_Server_API.app.core.Chat.tool_auto_exec import ToolExecutionBatchResult

    checked = list(records)
    return ToolExecutionBatchResult(
        requested_calls=len(checked),
        processed_calls=len(checked),
        execution_attempts=sum(1 for record in checked if not record.skipped),
        executed_calls=sum(1 for record in checked if record.ok),
        truncated=False,
        results=checked,
    )


async def run_read_only_envelope(
    definition: DefinitionRow,
    raw_message: str,
    eligible: list[dict[str, Any]],
    *,
    max_rounds: int = 5,
) -> EnvelopeResult:
    """Run the bounded read-only tool-calling loop for one raw message.

    Tools are presented from ``eligible`` only; every assistant tool call
    is checked against that membership set (name comparison, never
    deny-string parsing). An out-of-set call short-circuits as
    ``approval_required`` with the attempted tool named in ``note`` — it
    never reaches execution. Exhausted rounds fall back to a plain
    generation-only completion with a truncation note.
    """
    from tldw_Server_API.app.core.Chat.chat_service import present_chat_tools

    eligible_names = {t["name"] for t in eligible}
    presented = present_chat_tools(
        tools=[{"type": "function", "function": {"name": t["name"],
                "description": t.get("description", ""), "parameters": t.get("inputSchema", {})}}
               for t in eligible],
        allow_catalog=sorted(eligible_names), rollout_mode=None,
        provider_key="", provider_allowlist=None, streaming=False,
        presentation_variant=None,
    )
    if not presented.llm_tools:
        text = await _run_generation(definition, raw_message)
        return EnvelopeResult("succeeded", text, "tools eligible but not presented")
    messages = [{"role": "user", "content": raw_message}]
    for round_index in range(max_rounds):
        response = await _envelope_llm_call(
            messages=messages, tools=presented.llm_tools,
            system_message=_definition_system_prompt(definition),
            max_tokens=resolve_execution_target(definition)["max_tokens"])
        message = (response.get("choices") or [{}])[0].get("message", {})
        tool_calls = message.get("tool_calls") or []
        content = (message.get("content") or "").strip()
        if not tool_calls:
            return EnvelopeResult("succeeded", content or "(empty)", None)
        for call in tool_calls:
            name = ((call.get("function") or {}).get("name") or "")
            if name not in eligible_names:  # membership, never deny-string parsing
                return EnvelopeResult("approval_required", content or "",
                                      f"attempted non-envelope tool: {name}")
        batch = await _envelope_execute_calls(
            tool_calls=tool_calls, user_id=str(definition.owner_id),
            client_id="scheduled_automation", max_tool_calls=8, timeout_ms=30000,
            allow_catalog=sorted(eligible_names), attach_idempotency=True,
            idempotency_seed=f"automation:{definition.id}:{round_index}")
        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        # NOTE: the real batch/record fields are `.results` / `.content` —
        # the task brief's `.records` / `.output` spellings do not exist on
        # ToolExecutionBatchResult / ToolExecutionRecord.
        for record in batch.results:
            messages.append({"role": "tool", "tool_call_id": record.tool_call_id,
                             "content": record.content or ""})
    text = await _run_generation(definition, raw_message)
    return EnvelopeResult("succeeded", text,
                          f"tool rounds truncated after {max_rounds}")


async def _execute_generation_only(
    definition: DefinitionRow, payload: dict[str, Any]
) -> str:
    """Run one generation-only completion for a scheduled definition."""
    user_prompt = _definition_user_prompt(definition)
    return await _run_generation(definition, user_prompt)


async def _execute_agent_task(definition: DefinitionRow, payload: dict[str, Any]) -> str:
    """Run one agent_task definition (ADR-184 phase 2 + read-only envelope).

    Ruling 1A read path: the raw message lives only in the owner's
    encrypted store; resolve it here, in memory, at dispatch. An
    unresolvable ref (purged TTL, store miss, undecryptable blob) is an
    honest failed run via the raised LookupError -- never a silent skip.

    Read-only envelope (task 4): when the input requests tools, the
    eligible set is RE-derived here at execution time via
    ``derive_read_only_eligible_tools`` — the consumer gate's derivation
    is routing-only, and ``payload["_envelope_eligible"]`` is an audit
    hint this executor deliberately never reads. A non-empty eligible set
    runs ``run_read_only_envelope`` instead of ``_run_generation``; an
    empty set fails closed to generation-only (nothing presented, nothing
    executes).

    Outcome contract: this executor returns text. An envelope
    ``approval_required`` classification (out-of-set tool call) is raised
    as ``RuntimeError(APPROVAL_REQUIRED_ERROR_PREFIX + note)`` — the
    consumer's executor-exception handler recognizes that exact prefix
    and finishes the run ``approval_required`` (terminal, health-neutral)
    instead of ``failed``.
    """
    source = definition.input if isinstance(definition.input, dict) else {}
    message_ref = str(source.get("message_ref") or "").strip()
    if not message_ref:
        raise LookupError(
            "agent_task definition carries no message_ref (authored before "
            "the ADR-184 message store, or metadata incomplete)"
        )
    # The store read is sqlite I/O (mkdir, connect, query) -- keep it off
    # the event loop so one slow read cannot stall other jobs' dispatch.
    store = AutomationMessageStore.for_user(definition.owner_id)
    raw_message = await asyncio.to_thread(
        store.resolve_message, definition.owner_id, message_ref
    )
    if raw_message is None:
        raise LookupError(
            f"agent_task message_ref {message_ref!r} is unresolvable "
            "(purged TTL, store unavailable, or wrong owner)"
        )
    if _phase1_tools_requested(definition):
        eligible = await derive_read_only_eligible_tools(definition.owner_id)
        if eligible:
            result = await run_read_only_envelope(definition, raw_message, eligible)
            if result.status == "approval_required":
                raise RuntimeError(
                    f"{APPROVAL_REQUIRED_ERROR_PREFIX} {result.note or ''}".strip()
                )
            return result.text
    return await _run_generation(definition, raw_message)


def register_automation_executors() -> None:
    """Register the production executors for the wired families.

    ``recurring_question`` (phase-1 generation-only) and ``agent_task``
    (ADR-184 phase 2: the message resolved from the encrypted store at
    dispatch; a tool-requesting input whose names are all read-only
    eligible runs inside the bounded envelope). agent_task dispatch
    additionally sits behind the deployment certification gate, and tool
    requests the gate cannot route terminate ``approval_required`` before
    this executor runs. Idempotent: safe at every worker startup. The seam
    stays test-overridable — tests replace entries in the consumer's
    registry directly.
    """
    global _REGISTERED
    register_executor("recurring_question", _execute_generation_only)
    # ADR-184 phase 2: agent_task resolves its message from the encrypted
    # store. Dispatch still sits behind the deployment certification gate
    # (Phase 4D); the consumer routes fully read-only-eligible tool
    # requests here (the envelope re-derives the eligible set at execution
    # time) and every other tool request terminates approval_required
    # (ruling 2A) before this executor runs.
    register_executor("agent_task", _execute_agent_task)
    _REGISTERED = True
    logger.info(
        "Automation LLM executors registered (recurring_question + "
        "agent_task via the ADR-184 message store; read-only-eligible "
        "tool requests run inside the bounded envelope)"
    )


__all__ = [
    "register_automation_executors",
    "resolve_execution_target",
]
