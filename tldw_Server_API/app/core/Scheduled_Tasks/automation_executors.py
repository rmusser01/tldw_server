"""Production LLM executors for scheduled automations (TASK-13110).

Wires the TASK-13021 consumer's executor seam to the server's canonical
chat entrypoint (``perform_chat_api_call_async`` — the same surface the
Flashcards, Research, and MCP modules use). Phase-1 scope per tldw_chatbook
ADR-077 decision 4 (owner-accepted): generation-only completions. Tools are
already refused upstream by the consumer's phase-1 boundary; these
executors never pass ``tools``/``tool_choice``.

Model precedence (definition first, server last):

1. the definition's ``input``/``config`` ``model`` (+ optional ``provider``)
2. automation executor defaults from the server config
   (``[Scheduled_Tasks_Automation] executor_provider`` / ``executor_model``)
3. omit both and let ``perform_chat_api_call_async`` resolve the server's
   configured default provider/model

Credentials resolve through the existing provider-config layer inside the
chat entrypoint — this module adds no secret handling of its own.
"""

from __future__ import annotations

import asyncio
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.Automation_Message_Store_DB import (
    AutomationMessageStore,
)
from tldw_Server_API.app.core.DB_Management.Scheduled_Tasks_DB import DefinitionRow
from tldw_Server_API.app.core.Scheduled_Tasks.agent_task_jobs import register_executor
from tldw_Server_API.app.core.Workflows.adapters._common import extract_openai_content

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

    Only ``recurring_question`` is wired in phase 1 — its ``question``
    persists in full. ``agent_task`` input is redacted at rest by the
    automation service (``message_redacted`` metadata replaces the raw
    message), so a usable prompt surviving persistence is not a state
    production can reach; if one ever appears it is used, otherwise the
    LookupError records an honest failed run.
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


def _definition_system_prompt(definition: DefinitionRow) -> str:
    """Return the system prompt, allowing a definition-level override."""
    source: dict[str, Any] = (
        definition.input if isinstance(definition.input, dict) else {}
    )
    override = str(source.get(_EXECUTOR_SYSTEM_PROMPT_KEY) or "").strip()
    return override or _GENERATION_ONLY_SYSTEM_PROMPT


async def _execute_generation_only(
    definition: DefinitionRow, payload: dict[str, Any]
) -> str:
    """Run one generation-only completion for a scheduled definition."""
    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

    user_prompt = _definition_user_prompt(definition)
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


async def _execute_agent_task(definition: DefinitionRow, payload: dict[str, Any]) -> str:
    """Run one agent_task definition generation-only (ADR-184 phase 2).

    Ruling 1A read path: the raw message lives only in the owner's
    encrypted store; resolve it here, in memory, at dispatch. An
    unresolvable ref (purged TTL, store miss, undecryptable blob) is an
    honest failed run via the raised LookupError -- never a silent skip.

    Ruling 2A step 1: the consumer's tool gate terminates tool-requesting
    definitions as ``approval_required`` before this executor runs; what
    reaches here executes generation-only (no tool loop exists yet -- the
    read-only envelope is the next slice and will route through this same
    seam).
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
    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

    target = resolve_execution_target(definition)
    call_kwargs: dict[str, Any] = {
        "messages": [{"role": "user", "content": raw_message}],
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


def register_automation_executors() -> None:
    """Register the production executors for the wired families.

    ``recurring_question`` (phase-1 generation-only) and ``agent_task``
    (ADR-184 phase 2: generation-only with the message resolved from the
    encrypted store at dispatch). agent_task dispatch additionally sits
    behind the deployment certification gate, and tool-requesting runs
    terminate ``approval_required`` before the executor runs. Idempotent:
    safe at every worker startup. The seam stays test-overridable — tests
    replace entries in the consumer's registry directly.
    """
    global _REGISTERED
    register_executor("recurring_question", _execute_generation_only)
    # ADR-184 phase 2: agent_task executes generation-only with its message
    # resolved from the encrypted store. Dispatch still sits behind the
    # deployment certification gate (Phase 4D) and tool-requesting
    # definitions terminate approval_required (ruling 2A) before this
    # executor runs.
    register_executor("agent_task", _execute_agent_task)
    _REGISTERED = True
    logger.info(
        "Automation LLM executors registered (recurring_question + "
        "agent_task generation-only via the ADR-184 message store; "
        "tool-requesting runs terminate approval_required)"
    )


__all__ = [
    "register_automation_executors",
    "resolve_execution_target",
]
