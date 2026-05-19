"""MCP Runners — AgentDrivenRunner and LLMDrivenRunner for the ACP-MCP adapter.

AgentDrivenRunner: calls the agent's entry tool and translates the response into
AgentEvent instances.

LLMDrivenRunner: implements a ReAct loop where an LLM decides which tools to call,
ToolGate approves each call, and transport executes them against the MCP server.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

from tldw_Server_API.app.core.Agent_Client_Protocol import metrics as acp_metrics
from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_llm_caller import (
    LLMCaller,
    LLMToolCall,
    mcp_tools_to_openai_format,
)
from loguru import logger

from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_transport import MCPTransport
from tldw_Server_API.app.core.Agent_Client_Protocol.events import AgentEvent, AgentEventKind
from tldw_Server_API.app.core.Agent_Client_Protocol.tool_gate import ToolGate

_RUN_FIRST_METRIC_EXCEPTIONS: tuple[type[Exception], ...] = (
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
)


def _extract_text_content(result: dict[str, Any]) -> str:
    """Pull text from an MCP content array.

    MCP convention: ``result["content"][0]["text"]``.
    Falls back to ``str(result)`` if the structure is unexpected.
    """
    try:
        return result.get("content", [{}])[0].get("text", str(result))
    except (IndexError, AttributeError):
        return str(result)


async def _emit_malformed_step_error(
    emit: Callable[[AgentEventKind, dict[str, Any]], Awaitable[None]],
    *,
    step_type: str,
    missing: str,
) -> None:
    """Emit a consistent error for malformed structured response steps."""
    await emit(
        AgentEventKind.ERROR,
        {
            "message": "Malformed structured response step",
            "step_type": step_type,
            "missing": missing,
        },
    )


# ---------------------------------------------------------------------------
# AgentDrivenRunner
# ---------------------------------------------------------------------------


class AgentDrivenRunner:
    """Call an agent's entry tool and translate the response into AgentEvents.

    In *simple mode* (``structured_response=False``), the raw text from the
    MCP response is emitted as a single ``COMPLETION`` event.

    In *structured mode* (``structured_response=True``), the text is parsed as
    JSON with a ``{"steps": [...]}`` envelope.  Each step is emitted as an
    appropriate event type.
    """

    def __init__(
        self,
        transport: MCPTransport,
        event_callback: Callable[[AgentEvent], Awaitable[None]],
        session_id: str,
        cancel_event: asyncio.Event,
        entry_tool: str = "execute",
        structured_response: bool = False,
    ) -> None:
        self._transport = transport
        self._emit = event_callback
        self._session_id = session_id
        self._cancel = cancel_event
        self._entry_tool = entry_tool
        self._structured = structured_response
        self._seq = 0

    # -- helpers --

    async def _emit_event(self, kind: AgentEventKind, payload: dict[str, Any]) -> None:
        event = AgentEvent(
            session_id=self._session_id,
            kind=kind,
            payload=payload,
            sequence=self._seq,
        )
        self._seq += 1
        await self._emit(event)

    # -- public API --

    async def run(self, messages: list[dict]) -> None:
        """Call entry tool with *messages* and emit events from the response."""
        if self._cancel.is_set():
            return

        try:
            result = await self._transport.call_tool(
                self._entry_tool, {"messages": messages}
            )
        except Exception as exc:
            await self._emit_event(AgentEventKind.ERROR, {"error": str(exc)})
            return

        raw_text = _extract_text_content(result)

        if not self._structured:
            await self._emit_event(AgentEventKind.COMPLETION, {"text": raw_text})
            return

        # Structured mode: parse JSON steps
        try:
            parsed = json.loads(raw_text)
            steps = parsed.get("steps", [])
        except (json.JSONDecodeError, AttributeError):
            # Fallback: emit raw text as completion
            await self._emit_event(AgentEventKind.COMPLETION, {"text": raw_text})
            return

        has_completion = False
        for step in steps:
            if not isinstance(step, dict):
                await _emit_malformed_step_error(
                    self._emit_event,
                    step_type="",
                    missing="step mapping",
                )
                continue

            step_type = step.get("type", "")
            if step_type == "thinking":
                if "text" not in step:
                    await _emit_malformed_step_error(
                        self._emit_event,
                        step_type=step_type,
                        missing="text",
                    )
                    continue
                await self._emit_event(AgentEventKind.THINKING, {"text": step["text"]})
            elif step_type == "tool_call":
                if "tool_name" not in step:
                    await _emit_malformed_step_error(
                        self._emit_event,
                        step_type=step_type,
                        missing="tool_name",
                    )
                    continue
                await self._emit_event(
                    AgentEventKind.TOOL_CALL,
                    {
                        "tool_name": step["tool_name"],
                        "arguments": step.get("arguments", {}),
                    },
                )
            elif step_type == "tool_result":
                if "tool_name" not in step:
                    await _emit_malformed_step_error(
                        self._emit_event,
                        step_type=step_type,
                        missing="tool_name",
                    )
                    continue
                await self._emit_event(
                    AgentEventKind.TOOL_RESULT,
                    {
                        "tool_name": step["tool_name"],
                        "output": step.get("output", ""),
                    },
                )
            elif step_type == "completion":
                if "text" not in step:
                    await _emit_malformed_step_error(
                        self._emit_event,
                        step_type=step_type,
                        missing="text",
                    )
                    continue
                has_completion = True
                await self._emit_event(AgentEventKind.COMPLETION, {"text": step["text"]})
            else:
                logger.warning("Unknown step type {!r} in structured response", step_type)

        if not has_completion:
            await self._emit_event(AgentEventKind.COMPLETION, {"text": raw_text})


# ---------------------------------------------------------------------------
# LLMDrivenRunner
# ---------------------------------------------------------------------------


class LLMDrivenRunner:
    """ReAct loop: LLM decides tools, ToolGate approves, transport executes.

    Each iteration:
    1. Call the LLM with the conversation history and available tools.
    2. If the LLM requests tool calls, gate each one and execute approved calls.
    3. If the LLM produces text without tool calls, emit COMPLETION and stop.
    4. Stop after ``max_iterations`` with a ``max_iterations`` completion.
    """

    def __init__(
        self,
        transport: MCPTransport,
        event_callback: Callable[[AgentEvent], Awaitable[None]],
        session_id: str,
        cancel_event: asyncio.Event,
        llm_caller: LLMCaller,
        tool_gate: ToolGate,
        tools: list[dict],
        max_iterations: int = 20,
        llm_tools: list[dict[str, Any]] | None = None,
        prompt_fragment: str | None = None,
        run_first_metrics_context: dict[str, Any] | None = None,
    ) -> None:
        self._transport = transport
        self._emit = event_callback
        self._session_id = session_id
        self._cancel = cancel_event
        self._llm = llm_caller
        self._gate = tool_gate
        self._tools = tools
        self._max_iterations = max_iterations
        self._llm_tools = list(llm_tools) if llm_tools is not None else None
        self._prompt_fragment = str(prompt_fragment or "").strip() or None
        self._run_first_metrics_context = self._normalize_run_first_metrics_context(
            run_first_metrics_context
        )
        self._seq = 0

    # -- helpers --

    async def _emit_event(
        self,
        kind: AgentEventKind,
        payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event = AgentEvent(
            session_id=self._session_id,
            kind=kind,
            payload=payload,
            sequence=self._seq,
            metadata=metadata or {},
        )
        self._seq += 1
        await self._emit(event)

    @staticmethod
    def _normalize_run_first_metrics_context(
        context: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not isinstance(context, dict):
            return None

        presentation_variant = str(context.get("presentation_variant") or "").strip()
        if not presentation_variant:
            return None

        return {
            "agent_type": str(context.get("agent_type") or "").strip() or "mcp",
            "presentation_variant": presentation_variant,
            "cohort": str(context.get("cohort") or "").strip() or "unknown",
            "provider": str(context.get("provider") or "").strip() or "unknown",
            "model": str(context.get("model") or "").strip() or "unknown",
            "eligible": bool(context.get("eligible")),
            "ineligible_reason": (
                str(context.get("ineligible_reason") or "").strip() or None
            ),
        }

    def _record_run_first_rollout(self) -> None:
        if self._run_first_metrics_context is None:
            return
        try:
            acp_metrics.record_run_first_rollout(**self._run_first_metrics_context)
        except _RUN_FIRST_METRIC_EXCEPTIONS as exc:
            logger.warning(
                "ACP run-first metric emission failed for rollout on session {}: {}",
                self._session_id,
                exc,
            )

    def _record_run_first_first_tool(self, first_tool: str) -> None:
        if self._run_first_metrics_context is None:
            return
        try:
            acp_metrics.record_run_first_first_tool(
                **self._run_first_metrics_context,
                first_tool=first_tool,
            )
        except _RUN_FIRST_METRIC_EXCEPTIONS as exc:
            logger.warning(
                "ACP run-first metric emission failed for first_tool on session {}: {}",
                self._session_id,
                exc,
            )

    def _record_run_first_fallback_after_run(self, fallback_tool: str) -> None:
        if self._run_first_metrics_context is None:
            return
        try:
            acp_metrics.record_run_first_fallback_after_run(
                **self._run_first_metrics_context,
                fallback_tool=fallback_tool,
            )
        except _RUN_FIRST_METRIC_EXCEPTIONS as exc:
            logger.warning(
                "ACP run-first metric emission failed for fallback_after_run on session {}: {}",
                self._session_id,
                exc,
            )

    def _record_run_first_completion_proxy(self, outcome: str) -> None:
        if self._run_first_metrics_context is None:
            return
        try:
            acp_metrics.record_run_first_completion_proxy(
                **self._run_first_metrics_context,
                outcome=outcome,
            )
        except _RUN_FIRST_METRIC_EXCEPTIONS as exc:
            logger.warning(
                "ACP run-first metric emission failed for completion_proxy on session {}: {}",
                self._session_id,
                exc,
            )

    # -- public API --

    async def run(self, messages: list[dict]) -> None:
        """Run the ReAct loop until completion or max iterations."""
        self._record_run_first_rollout()
        history = list(messages)
        first_tool_name: str | None = None
        fallback_recorded = False
        if self._prompt_fragment:
            if (
                history
                and isinstance(history[0], dict)
                and history[0].get("role") == "system"
                and isinstance(history[0].get("content"), str)
            ):
                merged_system = dict(history[0])
                merged_system["content"] = (
                    f"{history[0]['content'].rstrip()}\n\n{self._prompt_fragment}"
                )
                history[0] = merged_system
            else:
                history.insert(0, {"role": "system", "content": self._prompt_fragment})

        openai_tools = (
            list(self._llm_tools)
            if self._llm_tools is not None
            else mcp_tools_to_openai_format(self._tools)
        )

        for _i in range(self._max_iterations):
            if self._cancel.is_set():
                break

            try:
                response = await self._llm.call(history, openai_tools)
            except Exception as exc:
                self._record_run_first_completion_proxy("error")
                await self._emit_event(AgentEventKind.ERROR, {"error": str(exc)})
                return

            if response.tool_calls:
                # Per OpenAI convention: assistant message with tool_calls
                # must come BEFORE the corresponding tool result messages.
                assistant_tool_calls = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in response.tool_calls
                ]
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "tool_calls": assistant_tool_calls,
                }
                if response.text:
                    assistant_msg["content"] = response.text
                history.append(assistant_msg)

                # Now process each tool call
                for tc in response.tool_calls:
                    if self._cancel.is_set():
                        break

                    if first_tool_name is None:
                        first_tool_name = tc.name
                        self._record_run_first_first_tool(first_tool_name)
                    elif (
                        first_tool_name == "run"
                        and not fallback_recorded
                        and tc.name != "run"
                    ):
                        fallback_recorded = True
                        self._record_run_first_fallback_after_run(tc.name)

                    try:
                        gate_result = await self._gate.request_approval(
                            self._session_id,
                            tc.name,
                            tc.arguments,
                            cancel_event=self._cancel,
                        )
                    except asyncio.CancelledError:
                        break
                    except Exception as exc:
                        self._record_run_first_completion_proxy("error")
                        await self._emit_event(AgentEventKind.ERROR, {"error": str(exc)})
                        return

                    if self._cancel.is_set():
                        break

                    if not gate_result.approved:
                        reason = gate_result.reason or "denied"
                        denied_msg = f"Permission denied: {reason}"
                        await self._emit_event(
                            AgentEventKind.TOOL_RESULT,
                            {
                                "tool_name": tc.name,
                                "output": denied_msg,
                                "is_error": True,
                            },
                        )
                        history.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": f"Error: {denied_msg}",
                        })
                        continue

                    await self._emit_event(
                        AgentEventKind.TOOL_CALL,
                        {"tool_name": tc.name, "arguments": tc.arguments},
                        metadata={"_already_approved": True},
                    )

                    try:
                        result = await self._transport.call_tool(tc.name, tc.arguments)
                        output = _extract_text_content(result)
                        is_error = result.get("isError", False)
                    except Exception as exc:
                        output = str(exc)
                        is_error = True

                    await self._emit_event(
                        AgentEventKind.TOOL_RESULT,
                        {"tool_name": tc.name, "output": output, "is_error": is_error},
                    )
                    history.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output,
                    })

            if response.text and not response.tool_calls:
                self._record_run_first_completion_proxy("end_turn")
                await self._emit_event(
                    AgentEventKind.COMPLETION,
                    {"text": response.text, "stop_reason": "end_turn"},
                )
                return
        else:
            # Exhausted max_iterations without returning
            self._record_run_first_completion_proxy("max_iterations")
            await self._emit_event(
                AgentEventKind.COMPLETION,
                {"text": "Reached maximum iterations", "stop_reason": "max_iterations"},
            )
            return

        if self._cancel.is_set():
            self._record_run_first_completion_proxy("cancelled")
