"""MCPAdapter — main adapter wiring transport + runners + heartbeat + lifecycle events."""
from __future__ import annotations

import asyncio
from contextlib import suppress
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.base import (
    AdapterConfig,
    PromptOptions,
    ProtocolAdapter,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_runners import (
    AgentDrivenRunner,
    LLMDrivenRunner,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_tool_presentation import (
    present_acp_tools,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.adapters.mcp_transport import (
    MCPTransport,
    create_transport,
)
from tldw_Server_API.app.core.Agent_Client_Protocol.events import AgentEvent, AgentEventKind
from tldw_Server_API.app.core.exceptions import ValidationError
from tldw_Server_API.app.core.config import (
    resolve_acp_run_first_presentation_variant,
    resolve_acp_run_first_provider_allowlist,
    resolve_acp_run_first_rollout_mode,
    resolve_run_first_cohort_label,
)


class MCPAdapter(ProtocolAdapter):
    """MCP protocol adapter supporting multiple transports and orchestration modes."""

    protocol_name = "mcp"

    def __init__(self) -> None:
        self._transport: MCPTransport | None = None
        self._config: AdapterConfig | None = None
        self._tools: list[dict[str, Any]] = []
        self._connected = False
        self._cancel_event = asyncio.Event()
        self._heartbeat_task: asyncio.Task | None = None

    # -- ProtocolAdapter interface --

    async def connect(self, config: AdapterConfig) -> None:
        """Connect to an MCP server via the configured transport."""
        pc = config.protocol_config
        self._transport = create_transport(pc)
        self._config = config
        try:
            await self._transport.connect()
            logger.info("MCPAdapter: transport connected (session={})", config.session_id)

            await self._emit(AgentEventKind.LIFECYCLE, {"event": "agent_started", "exit_code": None})

            self._tools = await self._transport.list_tools()
            logger.debug("MCPAdapter: discovered {} tools", len(self._tools))

            await self._emit(AgentEventKind.LIFECYCLE, {"event": "agent_ready", "exit_code": None})
            self._connected = True
        except Exception:
            if self._transport is not None:
                with suppress(Exception):
                    await self._transport.close()
            self._transport = None
            self._config = None
            self._tools = []
            self._connected = False
            raise

    async def disconnect(self) -> None:
        """Disconnect from the MCP server and clean up resources."""
        self._cancel_event.set()
        self._stop_heartbeat()
        if self._transport is not None:
            await self._transport.close()
        await self._emit(AgentEventKind.LIFECYCLE, {"event": "agent_exited", "exit_code": None})
        self._connected = False
        self._transport = None

    async def send_prompt(
        self,
        messages: list[dict],
        options: PromptOptions | None = None,
    ) -> None:
        """Send a prompt through the configured orchestration runner."""
        if not self._connected or self._transport is None or self._config is None:
            raise RuntimeError("Not connected")

        pc = self._config.protocol_config
        self._cancel_event.clear()

        # Refresh tools if configured
        if pc.get("mcp_refresh_tools", False):
            self._tools = await self._transport.list_tools()

        # Start heartbeat
        self._start_heartbeat()

        # Emit status change
        await self._emit(AgentEventKind.STATUS_CHANGE, {"from_status": "idle", "to_status": "working"})

        try:
            orchestration = pc.get("mcp_orchestration", "agent_driven")
            if orchestration == "llm_driven":
                llm_caller = pc.get("llm_caller")
                tool_gate = pc.get("tool_gate")
                if llm_caller is None or tool_gate is None:
                    raise ValueError(
                        "MCP llm_driven orchestration requires protocol_config keys: "
                        "llm_caller and tool_gate"
                    )
                provider = str(pc.get("mcp_llm_provider") or "").strip()
                model = str(pc.get("mcp_llm_model") or "").strip()
                rollout_mode = resolve_acp_run_first_rollout_mode()
                presented_tools = present_acp_tools(
                    session_id=self._config.session_id,
                    tools=self._tools,
                    rollout_mode=rollout_mode,
                    provider_key=f"{provider}:{model}" if provider or model else "",
                    provider_allowlist=resolve_acp_run_first_provider_allowlist(),
                    presentation_variant=resolve_acp_run_first_presentation_variant(),
                )
                run_first_metrics_context = {
                    "agent_type": "mcp",
                    "presentation_variant": presented_tools.presentation_variant,
                    "cohort": resolve_run_first_cohort_label(
                        rollout_mode,
                        eligible=presented_tools.eligible,
                        ineligible_reason=presented_tools.ineligible_reason,
                    ),
                    "provider": provider or "unknown",
                    "model": model or "unknown",
                    "eligible": presented_tools.eligible,
                    "ineligible_reason": presented_tools.ineligible_reason,
                }
                runner = LLMDrivenRunner(
                    transport=self._transport,
                    event_callback=self._config.event_callback,
                    session_id=self._config.session_id,
                    cancel_event=self._cancel_event,
                    llm_caller=llm_caller,
                    tool_gate=tool_gate,
                    tools=self._tools,
                    max_iterations=pc.get("mcp_max_iterations", 20),
                    llm_tools=presented_tools.openai_tools,
                    prompt_fragment=presented_tools.prompt_fragment,
                    run_first_metrics_context=run_first_metrics_context,
                )
            elif orchestration == "agent_driven":
                runner = AgentDrivenRunner(
                    transport=self._transport,
                    event_callback=self._config.event_callback,
                    session_id=self._config.session_id,
                    cancel_event=self._cancel_event,
                    entry_tool=pc.get("mcp_entry_tool", "execute"),
                    structured_response=pc.get("mcp_structured_response", False),
                )
            else:
                raise ValidationError(
                    f"Invalid mcp_orchestration value: {orchestration!r}. "
                    "Supported values are 'agent_driven' and 'llm_driven'."
                )
            await runner.run(messages)
        finally:
            self._stop_heartbeat()
            await self._emit(
                AgentEventKind.STATUS_CHANGE, {"from_status": "working", "to_status": "idle"}
            )

    async def send_tool_result(self, tool_id: str, output: str, is_error: bool = False) -> None:
        """No-op for now; future sampling support."""

    async def cancel(self) -> None:
        """Request cancellation of the current operation."""
        self._cancel_event.set()

    @property
    def is_connected(self) -> bool:
        """Whether the adapter currently has an active connection."""
        return self._connected and self._transport is not None and self._transport.is_connected

    @property
    def supports_streaming(self) -> bool:
        """Whether the adapter streams events or returns them in bulk."""
        return True

    # -- Internal helpers --

    async def _emit(self, kind: AgentEventKind, payload: dict[str, Any]) -> None:
        """Emit an AgentEvent via the configured callback."""
        if self._config is None:
            return
        event = AgentEvent(
            session_id=self._config.session_id,
            kind=kind,
            payload=payload,
        )
        await self._config.event_callback(event)

    def _start_heartbeat(self) -> None:
        """Start the background heartbeat task."""
        self._stop_heartbeat()
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    def _stop_heartbeat(self) -> None:
        """Cancel the background heartbeat task if running."""
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            self._heartbeat_task = None

    async def _heartbeat_loop(self) -> None:
        """Emit periodic heartbeat events every 15 seconds."""
        import time
        start = time.monotonic()
        try:
            while True:
                await asyncio.sleep(15)
                elapsed = int(time.monotonic() - start)
                await self._emit(
                    AgentEventKind.HEARTBEAT,
                    {"elapsed_sec": elapsed, "state": "executing"},
                )
        except asyncio.CancelledError:
            pass
