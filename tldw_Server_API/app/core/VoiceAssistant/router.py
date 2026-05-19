# VoiceAssistant/router.py
# Voice Command Router - Orchestrates the voice command pipeline
#
# Pipeline: STT (transcription) -> Intent Parse -> Action Execute -> TTS (response)
#
#######################################################################################################################
import time
from typing import Any, Callable, Optional
from urllib.parse import urljoin

from loguru import logger

from tldw_Server_API.app.core.http_client import RetryPolicy, afetch
from tldw_Server_API.app.core.Persona.connections import (
    PersonaConnectionConfigError,
    PersonaConnectionSecretError,
    PersonaConnectionTargetError,
    build_connection_headers,
    connection_content_from_row,
    render_nested_templates,
    render_template_value,
    safe_template_context,
    validate_connection_request_target,
)

from .intent_parser import IntentParser, get_intent_parser
from .registry import VoiceCommandRegistry, get_voice_command_registry
from .schemas import (
    ActionResult,
    ActionType,
    ParsedIntent,
    VoiceIntent,
    VoiceSessionContext,
    VoiceSessionState,
)
from .session import VoiceSessionManager, get_voice_session_manager
from .workflow_handler import VoiceWorkflowHandler, get_voice_workflow_handler

_PERSONA_CONNECTION_MEMORY_TYPE = "persona_connection"
_EXTERNAL_ACTION_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}


def _build_external_payload(
    *,
    action_config: dict[str, Any],
    entities: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    raw_slot_map = action_config.get("slot_to_param_map") or action_config.get("param_map") or {}
    if isinstance(raw_slot_map, dict):
        for param_name, source in raw_slot_map.items():
            if not isinstance(source, str):
                continue
            slot_name = source.strip().strip("{}")
            if slot_name in entities:
                payload[str(param_name)] = entities[slot_name]

    if not payload:
        payload.update(entities)

    defaults = action_config.get("default_payload")
    if isinstance(defaults, dict):
        for key, value in defaults.items():
            payload.setdefault(str(key), value)

    return payload


def _parse_external_response_body(response: Any) -> Any:
    headers = getattr(response, "headers", {}) or {}
    content_type = str(headers.get("content-type") or headers.get("Content-Type") or "").lower()
    if "json" in content_type and callable(getattr(response, "json", None)):
        try:
            return response.json()
        except Exception as exc:
            logger.debug(f"Failed to parse external action response as JSON: {exc}")
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    return None


async def _close_response(response: Any) -> None:
    close = getattr(response, "aclose", None)
    if callable(close):
        await close()
        return
    close = getattr(response, "close", None)
    if callable(close):
        close()


class VoiceCommandRouter:
    """
    Routes voice commands through the processing pipeline.

    Responsibilities:
    - Coordinate between intent parser, session manager, and action handlers
    - Execute matched commands via MCP tools, workflows, or custom handlers
    - Generate appropriate responses for TTS
    - Handle confirmation flows
    """

    def __init__(
        self,
        registry: Optional[VoiceCommandRegistry] = None,
        parser: Optional[IntentParser] = None,
        session_manager: Optional[VoiceSessionManager] = None,
        workflow_handler: Optional[VoiceWorkflowHandler] = None,
    ):
        """
        Initialize the voice command router.

        Args:
            registry: Voice command registry
            parser: Intent parser
            session_manager: Session manager
            workflow_handler: Workflow execution handler
        """
        self.registry = registry or get_voice_command_registry()
        self.parser = parser or get_intent_parser()
        self.session_manager = session_manager or get_voice_session_manager()
        self.workflow_handler = workflow_handler or get_voice_workflow_handler()

        # Custom action handlers
        self._custom_handlers: dict[str, Callable] = {}
        self._register_builtin_handlers()

    def _register_builtin_handlers(self) -> None:
        """Register handlers for built-in commands."""
        self._custom_handlers["stop"] = self._handle_stop
        self._custom_handlers["cancel"] = self._handle_cancel
        self._custom_handlers["help"] = self._handle_help
        self._custom_handlers["repeat"] = self._handle_repeat
        self._custom_handlers["confirmation"] = self._handle_confirmation
        self._custom_handlers["empty_input"] = self._handle_empty
        self._custom_handlers["workflow_status"] = self._handle_workflow_status
        self._custom_handlers["workflow_cancel"] = self._handle_workflow_cancel

    def register_custom_handler(
        self,
        action_name: str,
        handler: Callable,
    ) -> None:
        """
        Register a custom action handler.

        Args:
            action_name: Name of the action
            handler: Async callable that takes (intent, session) and returns ActionResult
        """
        self._custom_handlers[action_name] = handler
        logger.debug(f"Registered custom voice handler: {action_name}")

    async def process_command(
        self,
        text: str,
        user_id: int,
        session_id: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        db: Optional[Any] = None,
        persona_id: Optional[str] = None,
    ) -> tuple[ActionResult, str]:
        """
        Process a voice command through the full pipeline.

        Args:
            text: Transcribed text from STT
            user_id: User ID
            session_id: Optional existing session ID
            metadata: Optional command metadata
            db: Optional CharactersRAGDB instance for persistence/analytics

        Returns:
            Tuple of (ActionResult, session_id)
        """
        start_time = time.time()

        # Ensure session cleanup loop is running
        await self.session_manager.start()

        # Get or create session
        session, created = await self.session_manager.get_or_create_session(
            session_id, user_id, metadata
        )

        # Refresh registry from DB when available
        if db:
            try:
                self.registry.load_defaults()
                self.registry.refresh_user_commands(
                    db,
                    user_id,
                    include_disabled=True,
                    persona_id=persona_id,
                )
                if created:
                    from .db_helpers import save_voice_session

                    save_voice_session(db, session)
            except Exception as e:
                logger.warning(f"Failed to refresh voice commands from DB: {e}")

        parsed: Optional[ParsedIntent] = None

        try:
            previous_state = session.state
            # Update session state
            await self.session_manager.update_state(
                session.session_id, VoiceSessionState.PROCESSING
            )

            # Add user turn to history
            await self.session_manager.add_turn(
                session.session_id, "user", text
            )

            # Build context for parser
            context = {
                "awaiting_confirmation": previous_state == VoiceSessionState.AWAITING_CONFIRMATION,
                "conversation_history": session.conversation_history,
                "last_action_result": session.last_action_result,
            }

            # Parse intent
            parsed = await self.parser.parse(
                text,
                user_id,
                context,
                persona_id=persona_id,
            )
            logger.debug(
                f"Parsed intent: {parsed.intent.action_type} "
                f"(method={parsed.match_method}, confidence={parsed.intent.confidence:.2f})"
            )

            # Execute action
            result = await self._execute_intent(
                parsed.intent,
                session,
                db=db,
                persona_id=persona_id,
            )

            # Add assistant response to history
            await self.session_manager.add_turn(
                session.session_id,
                "assistant",
                result.response_text,
                {"action_type": result.action_type.value, "success": result.success},
            )

            # Store action result
            await self.session_manager.set_last_action_result(
                session.session_id,
                {
                    "success": result.success,
                    "action_type": result.action_type.value,
                    "data": result.result_data,
                },
            )

            # Update session state
            if session.pending_intent:
                await self.session_manager.update_state(
                    session.session_id, VoiceSessionState.AWAITING_CONFIRMATION
                )
            elif result.success:
                await self.session_manager.update_state(
                    session.session_id, VoiceSessionState.IDLE
                )
            else:
                await self.session_manager.update_state(
                    session.session_id, VoiceSessionState.ERROR
                )

            result.execution_time_ms = (time.time() - start_time) * 1000

            # Persist session + analytics when DB is available
            if db and parsed:
                try:
                    from .db_helpers import record_voice_command_event, save_voice_session

                    command_name = None
                    if parsed.intent.command_id:
                        cmd = self.registry.get_command(
                            parsed.intent.command_id,
                            user_id,
                            persona_id=persona_id,
                        )
                        command_name = cmd.name if cmd else None
                    if not command_name:
                        command_name = parsed.intent.action_config.get("action") or parsed.intent.action_type.value

                    record_voice_command_event(
                        db,
                        command_id=parsed.intent.command_id,
                        command_name=command_name,
                        user_id=user_id,
                        persona_id=persona_id,
                        action_type=result.action_type,
                        success=result.success,
                        response_time_ms=result.execution_time_ms,
                        session_id=session.session_id,
                        resolution_type=(
                            "direct_command"
                            if parsed.intent.command_id
                            else "planner_fallback"
                        ),
                    )
                    save_voice_session(db, session)
                except Exception as e:
                    logger.warning(f"Failed to record voice analytics: {e}")

            return result, session.session_id

        except Exception as e:
            logger.error(f"Error processing voice command: {e}")
            await self.session_manager.update_state(
                session.session_id, VoiceSessionState.ERROR
            )
            return ActionResult(
                success=False,
                action_type=ActionType.CUSTOM,
                response_text="I'm sorry, I encountered an error processing your request.",
                error_message=str(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            ), session.session_id

    async def match_registered_command(
        self,
        text: str,
        *,
        user_id: int,
        persona_id: str | None = None,
        db: Optional[Any] = None,
        include_disabled: bool = False,
        context: Optional[dict[str, Any]] = None,
    ) -> Optional[ParsedIntent]:
        """Run the deterministic registered-command fast path without executing the action."""
        self.registry.load_defaults()
        if db:
            self.registry.refresh_user_commands(
                db,
                user_id,
                include_disabled=include_disabled,
                persona_id=persona_id,
            )
        return await self.parser.parse_registered_command(
            text,
            user_id=user_id,
            persona_id=persona_id,
            context=context,
            include_disabled=include_disabled,
        )

    # ------------------------------------------------------------------
    # Dry-run validation (REVIEW.md 4.12)
    # ------------------------------------------------------------------

    async def validate_command_config(
        self,
        command: "VoiceCommand",
        *,
        db: Optional[Any] = None,
        persona_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Validate a voice command's action config without executing anything.

        Returns a list of step dicts, each with keys: name, passed, message, details.
        """
        from .schemas import ActionType  # already imported at module level, but explicit

        steps: list[dict[str, Any]] = []

        # ---- Step 1: config_schema -- basic structural checks ----
        steps.append(self._validate_config_schema(command))

        # ---- Step 2: action_target -- does the referenced target exist? ----
        target_step = await self._validate_action_target(command, db=db, persona_id=persona_id)
        steps.append(target_step)

        # ---- Step 3: phrases -- at least one non-empty phrase ----
        steps.append(self._validate_phrases(command))

        return steps

    @staticmethod
    def _validate_phrases(command: "VoiceCommand") -> dict[str, Any]:
        """Validate that at least one non-blank phrase is configured."""
        non_empty = [p for p in (command.phrases or []) if p.strip()]
        if non_empty:
            return {
                "name": "phrases",
                "passed": True,
                "message": f"{len(non_empty)} trigger phrase(s) configured",
                "details": None,
            }
        return {
            "name": "phrases",
            "passed": False,
            "message": "No trigger phrases configured",
            "details": None,
        }

    @staticmethod
    def _validate_config_schema(command: "VoiceCommand") -> dict[str, Any]:
        """Check that the action_config contains the required keys for its action_type."""
        action_type = command.action_type
        config = command.action_config or {}

        if action_type == ActionType.MCP_TOOL:
            if not config.get("tool_name"):
                return {
                    "name": "config_schema",
                    "passed": False,
                    "message": "Missing required field 'tool_name' in action_config for mcp_tool action",
                    "details": {"required_fields": ["tool_name"]},
                }
        elif action_type == ActionType.WORKFLOW:
            has_id = bool(config.get("workflow_id"))
            has_template = bool(config.get("workflow_template"))
            has_definition = bool(config.get("workflow_definition"))
            if not (has_id or has_template or has_definition):
                return {
                    "name": "config_schema",
                    "passed": False,
                    "message": "Workflow action requires 'workflow_id', 'workflow_template', or 'workflow_definition' in action_config",
                    "details": {"required_one_of": ["workflow_id", "workflow_template", "workflow_definition"]},
                }
        elif action_type == ActionType.CUSTOM:
            if not config.get("action"):
                return {
                    "name": "config_schema",
                    "passed": False,
                    "message": "Missing required field 'action' in action_config for custom action",
                    "details": {"required_fields": ["action"]},
                }
        # LLM_CHAT has no required config keys

        return {
            "name": "config_schema",
            "passed": True,
            "message": f"Action config is well-formed for '{action_type.value}'",
            "details": None,
        }

    async def _validate_action_target(
        self,
        command: "VoiceCommand",
        *,
        db: Optional[Any] = None,
        persona_id: str | None = None,
    ) -> dict[str, Any]:
        """Check whether the referenced action target (tool, workflow, handler) exists."""
        action_type = command.action_type
        config = command.action_config or {}

        if action_type == ActionType.MCP_TOOL:
            return await self._validate_mcp_tool_exists(config)

        if action_type == ActionType.WORKFLOW:
            return self._validate_workflow_target_exists(config)

        if action_type == ActionType.CUSTOM:
            return self._validate_custom_handler_exists(config, command=command, db=db, persona_id=persona_id)

        if action_type == ActionType.LLM_CHAT:
            return {
                "name": "action_target",
                "passed": True,
                "message": "LLM chat actions use the default LLM provider (no specific target to validate)",
                "details": None,
            }

        return {
            "name": "action_target",
            "passed": False,
            "message": f"Unknown action type: {action_type}",
            "details": None,
        }

    async def _validate_mcp_tool_exists(self, config: dict[str, Any]) -> dict[str, Any]:
        """Check whether an MCP tool name resolves to an available tool."""
        tool_name = config.get("tool_name", "")
        if not tool_name:
            return {
                "name": "action_target",
                "passed": False,
                "message": "No tool_name specified",
                "details": None,
            }
        try:
            from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext

            protocol = MCPProtocol()
            context = RequestContext(
                request_id="validate-dry-run",
                user_id="0",
                client_id="voice_assistant_validator",
                session_id="dry-run",
            )
            result = await protocol._handle_tools_list({}, context)
            available = [t.get("name") for t in result.get("tools", [])]

            if tool_name in available:
                return {
                    "name": "action_target",
                    "passed": True,
                    "message": f"MCP tool '{tool_name}' is available",
                    "details": None,
                }
            return {
                "name": "action_target",
                "passed": False,
                "message": f"MCP tool '{tool_name}' not found in available tools",
                "details": {"available_tools_sample": available[:20]},
            }
        except (ImportError, ModuleNotFoundError):
            return {
                "name": "action_target",
                "passed": False,
                "message": "MCP protocol module is not available",
                "details": None,
            }
        except Exception as e:
            return {
                "name": "action_target",
                "passed": False,
                "message": f"Error checking MCP tools: {e}",
                "details": None,
            }

    def _validate_workflow_target_exists(self, config: dict[str, Any]) -> dict[str, Any]:
        """Check whether a workflow template or ID resolves."""
        template_name = config.get("workflow_template")
        workflow_id = config.get("workflow_id")
        workflow_def = config.get("workflow_definition")

        if template_name:
            templates = self.workflow_handler.get_voice_workflow_templates()
            if template_name in templates:
                return {
                    "name": "action_target",
                    "passed": True,
                    "message": f"Workflow template '{template_name}' found",
                    "details": None,
                }
            return {
                "name": "action_target",
                "passed": False,
                "message": f"Workflow template '{template_name}' not found",
                "details": {"available_templates": list(templates.keys())},
            }

        if workflow_id:
            return {
                "name": "action_target",
                "passed": True,
                "message": f"Workflow ID '{workflow_id}' configured (runtime resolution)",
                "details": None,
            }

        if workflow_def:
            if isinstance(workflow_def, dict) and workflow_def.get("steps"):
                return {
                    "name": "action_target",
                    "passed": True,
                    "message": "Inline workflow definition with steps found",
                    "details": {"step_count": len(workflow_def["steps"])},
                }
            return {
                "name": "action_target",
                "passed": False,
                "message": "Inline workflow definition is missing 'steps'",
                "details": None,
            }

        return {
            "name": "action_target",
            "passed": False,
            "message": "No workflow target specified",
            "details": None,
        }

    def _validate_custom_handler_exists(
        self,
        config: dict[str, Any],
        *,
        command: Optional["VoiceCommand"] = None,
        db: Optional[Any] = None,
        persona_id: str | None = None,
    ) -> dict[str, Any]:
        """Check whether a custom action handler is registered."""
        action_name = config.get("action", "")

        # External connection-backed commands are valid if they have a connection_id
        if command and getattr(command, "connection_id", None):
            return {
                "name": "action_target",
                "passed": True,
                "message": f"Custom action backed by external connection '{command.connection_id}'",
                "details": None,
            }

        if not action_name:
            return {
                "name": "action_target",
                "passed": False,
                "message": "No 'action' name specified in action_config",
                "details": None,
            }

        if action_name in self._custom_handlers:
            return {
                "name": "action_target",
                "passed": True,
                "message": f"Custom handler '{action_name}' is registered",
                "details": None,
            }

        return {
            "name": "action_target",
            "passed": False,
            "message": f"Custom handler '{action_name}' is not registered",
            "details": {"available_handlers": sorted(self._custom_handlers.keys())},
        }

    async def _execute_intent(
        self,
        intent: VoiceIntent,
        session: VoiceSessionContext,
        *,
        db: Optional[Any] = None,
        persona_id: str | None = None,
    ) -> ActionResult:
        """Execute an intent and return the result."""

        # Check if confirmation is required
        if intent.requires_confirmation and session.state != VoiceSessionState.AWAITING_CONFIRMATION:
            await self.session_manager.set_pending_intent(session.session_id, intent)
            return ActionResult(
                success=True,
                action_type=intent.action_type,
                response_text=self._get_confirmation_prompt(intent),
            )

        # Route to appropriate handler based on action type
        if intent.action_type == ActionType.CUSTOM:
            return await self._execute_custom(intent, session, db=db, persona_id=persona_id)
        elif intent.action_type == ActionType.MCP_TOOL:
            return await self._execute_mcp_tool(intent, session)
        elif intent.action_type == ActionType.WORKFLOW:
            return await self._execute_workflow(intent, session)
        elif intent.action_type == ActionType.LLM_CHAT:
            return await self._execute_llm_chat(intent, session)
        else:
            return ActionResult(
                success=False,
                action_type=intent.action_type,
                response_text="I don't know how to handle that type of action.",
                error_message=f"Unknown action type: {intent.action_type}",
            )

    def _get_confirmation_prompt(self, intent: VoiceIntent) -> str:
        """Generate a confirmation prompt for an intent."""
        action_desc = intent.action_config.get("description", "this action")

        if intent.action_type == ActionType.MCP_TOOL:
            tool_name = intent.action_config.get("tool_name", "unknown tool")
            return f"Do you want me to execute {tool_name}? Say yes or no."

        if intent.action_type == ActionType.WORKFLOW:
            workflow_name = intent.action_config.get("workflow_name", "the workflow")
            return f"Should I start {workflow_name}? Say yes or no."

        return f"Should I proceed with {action_desc}? Say yes or no."

    async def _execute_custom(
        self,
        intent: VoiceIntent,
        session: VoiceSessionContext,
        *,
        db: Optional[Any] = None,
        persona_id: str | None = None,
    ) -> ActionResult:
        """Execute a custom action."""
        external_command = self._get_external_connection_command(
            intent=intent,
            user_id=session.user_id,
            persona_id=persona_id,
        )
        if external_command is not None:
            return await self._execute_external_connection_action(
                intent=intent,
                session=session,
                db=db,
                persona_id=persona_id,
                command=external_command,
            )

        action = intent.action_config.get("action", "")

        handler = self._custom_handlers.get(action)
        if handler:
            return await handler(intent, session)

        return ActionResult(
            success=False,
            action_type=ActionType.CUSTOM,
            response_text="I don't recognize that command.",
            error_message=f"No handler for custom action: {action}",
        )

    def _get_external_connection_command(
        self,
        *,
        intent: VoiceIntent,
        user_id: int,
        persona_id: str | None,
    ) -> Any | None:
        if not intent.command_id:
            return None
        command = self.registry.get_command(
            intent.command_id,
            user_id,
            persona_id=persona_id,
        )
        if command is None or not getattr(command, "connection_id", None):
            return None
        return command

    def _load_persona_connection(
        self,
        *,
        db: Any,
        user_id: int,
        persona_id: str,
        connection_id: str,
    ) -> dict[str, Any] | None:
        rows = db.list_persona_memory_entries(
            user_id=str(user_id),
            persona_id=persona_id,
            memory_type=_PERSONA_CONNECTION_MEMORY_TYPE,
            include_archived=False,
            include_deleted=False,
            limit=200,
            offset=0,
        )
        for row in rows or []:
            if str(row.get("id") or "").strip() != connection_id:
                continue
            content = connection_content_from_row(row)
            content["id"] = connection_id
            return content
        return None

    def _format_external_action_result(
        self,
        *,
        action_config: dict[str, Any],
        status_code: int,
        body: Any,
    ) -> str:
        explicit_message = str(action_config.get("success_message") or "").strip()
        if explicit_message:
            return explicit_message
        if isinstance(body, dict):
            for key in ("response_text", "message", "detail", "status", "result"):
                value = body.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            results = body.get("results")
            if isinstance(results, list):
                if not results:
                    return "The action completed but returned no results."
                if len(results) == 1:
                    return "The action completed and returned one result."
                return f"The action completed and returned {len(results)} results."
        if isinstance(body, list):
            if not body:
                return "The action completed but returned no results."
            return f"The action completed and returned {len(body)} results."
        if isinstance(body, str) and body.strip():
            return body.strip()
        if 200 <= status_code < 300:
            return "Done."
        return "The external action completed."

    async def _execute_external_connection_action(
        self,
        *,
        intent: VoiceIntent,
        session: VoiceSessionContext,
        db: Any | None,
        persona_id: str | None,
        command: Any,
    ) -> ActionResult:
        resolved_persona_id = str(
            persona_id
            or getattr(command, "persona_id", None)
            or ""
        ).strip()
        connection_id = str(getattr(command, "connection_id", "") or "").strip()
        if db is None:
            return ActionResult(
                success=False,
                action_type=intent.action_type,
                response_text="I couldn't reach that external action right now.",
                error_message="External actions require a database connection.",
            )
        if not resolved_persona_id or not connection_id:
            return ActionResult(
                success=False,
                action_type=intent.action_type,
                response_text="I couldn't resolve that external action.",
                error_message="Missing persona_id or connection_id for external action.",
            )

        connection = self._load_persona_connection(
            db=db,
            user_id=session.user_id,
            persona_id=resolved_persona_id,
            connection_id=connection_id,
        )
        if connection is None:
            return ActionResult(
                success=False,
                action_type=intent.action_type,
                response_text="I couldn't find the configured connection for that command.",
                error_message=f"Connection '{connection_id}' was not found for persona '{resolved_persona_id}'.",
            )

        action_config = dict(intent.action_config or {})
        payload = _build_external_payload(
            action_config=action_config,
            entities=dict(intent.entities or {}),
        )
        template_context = safe_template_context(intent.entities or {}, payload)

        method = str(action_config.get("method") or action_config.get("http_method") or "POST").strip().upper()
        if method not in _EXTERNAL_ACTION_METHODS:
            return ActionResult(
                success=False,
                action_type=intent.action_type,
                response_text="I couldn't determine how to call that external action.",
                error_message=f"Unsupported external action method: {method}",
        )

        base_url = str(connection.get("base_url") or "").strip()
        path = str(action_config.get("path") or action_config.get("request_path") or "").strip()
        rendered_path = render_template_value(path, template_context) if path else ""
        url = urljoin(base_url.rstrip("/") + "/", rendered_path.lstrip("/")) if rendered_path else base_url

        try:
            validate_connection_request_target(url, connection)
        except (PersonaConnectionConfigError, PersonaConnectionTargetError) as exc:
            return ActionResult(
                success=False,
                action_type=intent.action_type,
                response_text="I couldn't complete that external action.",
                error_message=str(exc),
            )

        rendered_payload = render_nested_templates(payload, template_context)
        try:
            headers, _secret = build_connection_headers(
                connection,
                payload=rendered_payload if isinstance(rendered_payload, dict) else {},
                auth_header_name=str(action_config.get("auth_header_name") or "").strip() or None,
            )
        except PersonaConnectionSecretError as exc:
            return ActionResult(
                success=False,
                action_type=intent.action_type,
                response_text="I couldn't complete that external action.",
                error_message=str(exc),
            )
        timeout_ms = int(connection.get("timeout_ms") or 15000)
        timeout_seconds = max(0.1, timeout_ms / 1000.0)
        retry_policy = RetryPolicy()
        retry_policy.attempts = 1
        retry_policy.retry_on_unsafe = False
        request_kwargs: dict[str, Any] = {
            "method": method,
            "url": url,
            "headers": headers or None,
            "timeout": timeout_seconds,
            "retry": retry_policy,
        }
        if method in {"GET", "DELETE", "HEAD"}:
            request_kwargs["params"] = rendered_payload if isinstance(rendered_payload, dict) else None
        else:
            request_kwargs["json"] = rendered_payload

        try:
            response = await afetch(**request_kwargs)
            try:
                response_body = _parse_external_response_body(response)
            finally:
                await _close_response(response)
        except Exception as exc:
            logger.error(f"External connection action failed: {exc}")
            return ActionResult(
                success=False,
                action_type=intent.action_type,
                response_text="I couldn't reach that external action right now.",
                error_message=str(exc),
            )

        if int(getattr(response, "status_code", 500)) >= 400:
            error_detail = response_body if isinstance(response_body, str) else str(response_body)
            return ActionResult(
                success=False,
                action_type=intent.action_type,
                response_text="I couldn't complete that external action.",
                error_message=error_detail,
                result_data={
                    "status_code": int(getattr(response, "status_code", 500)),
                    "body": response_body,
                    "connection_id": connection_id,
                    "url": url,
                },
            )

        response_text = self._format_external_action_result(
            action_config=action_config,
            status_code=int(getattr(response, "status_code", 200)),
            body=response_body,
        )
        return ActionResult(
            success=True,
            action_type=intent.action_type,
            response_text=response_text,
            result_data={
                "status_code": int(getattr(response, "status_code", 200)),
                "body": response_body,
                "connection_id": connection_id,
                "url": url,
                "request_payload": rendered_payload,
            },
        )

    async def _execute_mcp_tool(
        self,
        intent: VoiceIntent,
        session: VoiceSessionContext,
    ) -> ActionResult:
        """Execute an MCP tool."""
        tool_name = intent.action_config.get("tool_name")
        if not tool_name:
            return ActionResult(
                success=False,
                action_type=ActionType.MCP_TOOL,
                response_text="I couldn't determine which tool to use.",
                error_message="No tool_name in action config",
            )

        try:
            # Import MCP protocol
            from tldw_Server_API.app.core.MCP_unified.protocol import MCPProtocol, RequestContext

            # Build tool arguments from intent entities
            arguments = {
                k: v for k, v in intent.entities.items()
                if k not in ("tool_name",)
            }

            # Add any explicit arguments from config
            for key, value in intent.action_config.items():
                if key not in ("tool_name", "extract_query", "extract_content"):
                    arguments[key] = value

            # Create request context
            context = RequestContext(
                request_id=f"voice-{session.session_id}",
                user_id=str(session.user_id),
                client_id="voice_assistant",
                session_id=session.session_id,
            )

            # Get MCP protocol instance
            protocol = MCPProtocol()

            # Execute tool
            result = await protocol._handle_tools_call(
                params={"name": tool_name, "arguments": arguments},
                context=context,
            )

            # Format response based on tool result
            response_text = self._format_tool_result(tool_name, result)

            return ActionResult(
                success=True,
                action_type=ActionType.MCP_TOOL,
                result_data=result,
                response_text=response_text,
            )

        except ImportError:
            logger.warning("MCP protocol not available")
            return ActionResult(
                success=False,
                action_type=ActionType.MCP_TOOL,
                response_text="The tool system is not available right now.",
                error_message="MCP protocol not available",
            )
        except Exception as e:
            logger.error(f"MCP tool execution failed: {e}")
            return ActionResult(
                success=False,
                action_type=ActionType.MCP_TOOL,
                response_text=f"I couldn't complete that action. {str(e)}",
                error_message=str(e),
            )

    def _format_tool_result(self, tool_name: str, result: dict[str, Any]) -> str:
        """Format tool result for TTS response."""
        if not result:
            return "The action completed but returned no results."

        # Handle search results
        if "results" in result and isinstance(result["results"], list):
            count = len(result["results"])
            if count == 0:
                return "I didn't find any results."
            elif count == 1:
                item = result["results"][0]
                title = item.get("title", item.get("name", "one item"))
                return f"I found one result: {title}"
            else:
                # Summarize first few results
                titles = [
                    r.get("title", r.get("name", "untitled"))
                    for r in result["results"][:3]
                ]
                return f"I found {count} results. The top results are: {', '.join(titles)}"

        # Handle note creation
        if "note_id" in result or "id" in result:
            return "I've created the note for you."

        # Generic success message
        if result.get("success") or result.get("status") == "ok":
            return "Done."

        # Return a generic message for other results
        return "The action completed successfully."

    async def _execute_workflow(
        self,
        intent: VoiceIntent,
        session: VoiceSessionContext,
    ) -> ActionResult:
        """Execute a workflow using the voice workflow handler."""
        workflow_id = intent.action_config.get("workflow_id")
        workflow_template = intent.action_config.get("workflow_template")
        workflow_definition = None

        # Check for workflow template (built-in voice workflow)
        if workflow_template:
            templates = self.workflow_handler.get_voice_workflow_templates()
            workflow_definition = templates.get(workflow_template)
            if not workflow_definition:
                return ActionResult(
                    success=False,
                    action_type=ActionType.WORKFLOW,
                    response_text=f"I don't recognize the workflow template: {workflow_template}.",
                    error_message=f"Unknown workflow template: {workflow_template}",
                )

        # If no template or ID, check for inline definition
        if not workflow_id and not workflow_definition:
            workflow_definition = intent.action_config.get("workflow_definition")

        if not workflow_id and not workflow_definition:
            return ActionResult(
                success=False,
                action_type=ActionType.WORKFLOW,
                response_text="I couldn't determine which workflow to run.",
                error_message="No workflow_id or workflow_definition in action config",
            )

        # Build workflow inputs from entities
        workflow_inputs = dict(intent.entities)

        # Add any explicit inputs from action config
        if "inputs" in intent.action_config:
            workflow_inputs.update(intent.action_config["inputs"])

        # Determine sync/async mode
        sync_mode = intent.action_config.get("sync", True)

        # Execute via workflow handler
        return await self.workflow_handler.execute_workflow(
            workflow_id=workflow_id,
            workflow_definition=workflow_definition,
            inputs=workflow_inputs,
            user_id=session.user_id,
            session=session,
            sync=sync_mode,
        )

    async def _execute_llm_chat(
        self,
        intent: VoiceIntent,
        session: VoiceSessionContext,
    ) -> ActionResult:
        """Execute a general LLM chat response."""
        message = intent.action_config.get("message", intent.raw_text)

        try:
            from tldw_Server_API.app.core.LLM_Calls.LLM_API_Calls_Unified import chat_api_call_async

            # Build conversation context
            session.get_context_messages(max_turns=5)

            # System prompt for voice assistant
            system_prompt = """You are a helpful voice assistant. Keep your responses concise and natural-sounding for speech.
Avoid using markdown, code blocks, or special formatting.
Respond conversationally and get straight to the point."""

            response = await chat_api_call_async(
                input_data=message,
                custom_prompt=system_prompt,
                api_endpoint="openai",
                api_key=None,
                temp=0.7,
                max_tokens=150,
            )

            if not response:
                response = "I'm not sure how to respond to that."

            return ActionResult(
                success=True,
                action_type=ActionType.LLM_CHAT,
                result_data={"response": response},
                response_text=response,
            )

        except ImportError:
            return ActionResult(
                success=False,
                action_type=ActionType.LLM_CHAT,
                response_text="I'm having trouble connecting to my language model.",
                error_message="LLM API not available",
            )
        except Exception as e:
            logger.error(f"LLM chat failed: {e}")
            return ActionResult(
                success=False,
                action_type=ActionType.LLM_CHAT,
                response_text="I encountered an error generating a response.",
                error_message=str(e),
            )

    # Built-in command handlers

    async def _handle_stop(
        self,
        intent: VoiceIntent,
        session: VoiceSessionContext,
    ) -> ActionResult:
        """Handle stop command."""
        # Clear any pending operations
        await self.session_manager.set_pending_intent(session.session_id, None)
        return ActionResult(
            success=True,
            action_type=ActionType.CUSTOM,
            response_text="Stopped.",
        )

    async def _handle_cancel(
        self,
        intent: VoiceIntent,
        session: VoiceSessionContext,
    ) -> ActionResult:
        """Handle cancel command."""
        # Clear pending intent if any
        pending = session.pending_intent
        await self.session_manager.set_pending_intent(session.session_id, None)

        if pending:
            return ActionResult(
                success=True,
                action_type=ActionType.CUSTOM,
                response_text="Cancelled.",
            )
        return ActionResult(
            success=True,
            action_type=ActionType.CUSTOM,
            response_text="Nothing to cancel.",
        )

    async def _handle_help(
        self,
        intent: VoiceIntent,
        session: VoiceSessionContext,
    ) -> ActionResult:
        """Handle help command."""
        commands = self.registry.get_all_commands(session.user_id)
        command_names = [cmd.name for cmd in commands[:5]]

        response = f"You can say things like: {', '.join(command_names)}. You can also ask me general questions."

        return ActionResult(
            success=True,
            action_type=ActionType.CUSTOM,
            result_data={"available_commands": [c.name for c in commands]},
            response_text=response,
        )

    async def _handle_repeat(
        self,
        intent: VoiceIntent,
        session: VoiceSessionContext,
    ) -> ActionResult:
        """Handle repeat command."""
        last_result = session.last_action_result
        if last_result and "response_text" in last_result:
            return ActionResult(
                success=True,
                action_type=ActionType.CUSTOM,
                response_text=last_result["response_text"],
            )

        # Find last assistant message
        for turn in reversed(session.conversation_history):
            if turn["role"] == "assistant":
                return ActionResult(
                    success=True,
                    action_type=ActionType.CUSTOM,
                    response_text=turn["content"],
                )

        return ActionResult(
            success=True,
            action_type=ActionType.CUSTOM,
            response_text="I don't have anything to repeat.",
        )

    async def _handle_confirmation(
        self,
        intent: VoiceIntent,
        session: VoiceSessionContext,
    ) -> ActionResult:
        """Handle confirmation response."""
        confirmed = intent.action_config.get("confirmed", False)
        pending = session.pending_intent

        # Clear pending intent
        await self.session_manager.set_pending_intent(session.session_id, None)

        if not pending:
            return ActionResult(
                success=True,
                action_type=ActionType.CUSTOM,
                response_text="There's nothing pending to confirm.",
            )

        if confirmed:
            # Execute the pending intent
            pending.requires_confirmation = False  # Prevent loop
            return await self._execute_intent(pending, session)
        else:
            return ActionResult(
                success=True,
                action_type=ActionType.CUSTOM,
                response_text="Okay, I won't do that.",
            )

    async def _handle_empty(
        self,
        intent: VoiceIntent,
        session: VoiceSessionContext,
    ) -> ActionResult:
        """Handle empty input."""
        return ActionResult(
            success=True,
            action_type=ActionType.CUSTOM,
            response_text="",  # No response for empty input
        )

    async def _handle_workflow_status(
        self,
        intent: VoiceIntent,
        session: VoiceSessionContext,
    ) -> ActionResult:
        """Handle workflow status query."""
        # Check for active workflow in session metadata
        last_result = session.last_action_result
        if not last_result:
            return ActionResult(
                success=True,
                action_type=ActionType.CUSTOM,
                response_text="I don't have any active workflows to check.",
            )

        run_id = None
        if isinstance(last_result, dict):
            run_id = last_result.get("data", {}).get("run_id")

        if not run_id:
            return ActionResult(
                success=True,
                action_type=ActionType.CUSTOM,
                response_text="I couldn't find a workflow to check.",
            )

        status = await self.get_workflow_status(run_id, session.user_id)
        if not status:
            return ActionResult(
                success=True,
                action_type=ActionType.CUSTOM,
                response_text="I couldn't find that workflow.",
            )

        workflow_status = status.get("status", "unknown")
        if workflow_status == "succeeded":
            return ActionResult(
                success=True,
                action_type=ActionType.CUSTOM,
                result_data=status,
                response_text="The workflow completed successfully.",
            )
        elif workflow_status == "failed":
            error = status.get("error", "unknown error")
            return ActionResult(
                success=True,
                action_type=ActionType.CUSTOM,
                result_data=status,
                response_text=f"The workflow failed: {error}",
            )
        elif workflow_status == "cancelled":
            return ActionResult(
                success=True,
                action_type=ActionType.CUSTOM,
                result_data=status,
                response_text="The workflow was cancelled.",
            )
        elif workflow_status in ("running", "pending"):
            return ActionResult(
                success=True,
                action_type=ActionType.CUSTOM,
                result_data=status,
                response_text=f"The workflow is still {workflow_status}.",
            )
        else:
            return ActionResult(
                success=True,
                action_type=ActionType.CUSTOM,
                result_data=status,
                response_text=f"The workflow status is {workflow_status}.",
            )

    async def _handle_workflow_cancel(
        self,
        intent: VoiceIntent,
        session: VoiceSessionContext,
    ) -> ActionResult:
        """Handle workflow cancel request."""
        # Check for active workflow in session metadata
        last_result = session.last_action_result
        run_id = None
        if isinstance(last_result, dict):
            run_id = last_result.get("data", {}).get("run_id")

        if not run_id:
            return ActionResult(
                success=True,
                action_type=ActionType.CUSTOM,
                response_text="I don't have any active workflows to cancel.",
            )

        cancelled = await self.cancel_workflow(run_id, session.user_id)
        if cancelled:
            return ActionResult(
                success=True,
                action_type=ActionType.CUSTOM,
                response_text="I've cancelled the workflow.",
            )
        else:
            return ActionResult(
                success=False,
                action_type=ActionType.CUSTOM,
                response_text="I couldn't cancel that workflow. It may have already completed.",
            )

    # Workflow helper methods

    async def get_workflow_status(
        self,
        run_id: str,
        user_id: int,
    ) -> Optional[dict[str, Any]]:
        """
        Get the status of a workflow run.

        Args:
            run_id: Workflow run ID
            user_id: User ID

        Returns:
            Status dict or None if not found/authorized
        """
        return await self.workflow_handler.get_workflow_status(run_id, user_id)

    async def cancel_workflow(
        self,
        run_id: str,
        user_id: int,
    ) -> bool:
        """
        Cancel a running workflow.

        Args:
            run_id: Workflow run ID
            user_id: User ID

        Returns:
            True if cancelled, False otherwise
        """
        return await self.workflow_handler.cancel_workflow(run_id, user_id)

    def stream_workflow_progress(
        self,
        run_id: str,
        user_id: int,
        poll_interval: float = 0.5,
        timeout_seconds: float = 300.0,
    ):
        """
        Stream workflow progress events.

        Args:
            run_id: Workflow run ID
            user_id: User ID
            poll_interval: How often to poll for events
            timeout_seconds: Maximum streaming duration

        Returns:
            AsyncGenerator yielding WorkflowProgressEvent objects
        """
        return self.workflow_handler.stream_workflow_progress(
            run_id=run_id,
            user_id=user_id,
            poll_interval=poll_interval,
            timeout_seconds=timeout_seconds,
        )


# Singleton instance
_router_instance: Optional[VoiceCommandRouter] = None


def get_voice_command_router() -> VoiceCommandRouter:
    """Get the singleton voice command router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = VoiceCommandRouter()
    return _router_instance


#
# End of VoiceAssistant/router.py
#######################################################################################################################
