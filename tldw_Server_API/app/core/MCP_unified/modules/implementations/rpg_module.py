"""MCP tools for the generic RPG campaign/session runtime."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.RPG_DB import RPGRepository
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import MultiDatabaseRetriever
from tldw_Server_API.app.core.RPG.rules.adapters import build_default_adapter_registry
from tldw_Server_API.app.core.RPG.rules.lookup import RulesLookupService
from tldw_Server_API.app.core.RPG.rules.retrieval import RulesRetrievalAdapter
from tldw_Server_API.app.core.RPG.rules.source_validation import RPGRulesSourceValidator
from tldw_Server_API.app.core.RPG.service import RPGService

from ..base import BaseModule, ModuleConfig, create_tool_definition

_TOOL_ADAPTERS_LIST = "rpg.adapters.list"
_TOOL_SESSIONS_GET = "rpg.sessions.get"
_TOOL_RULES_LOOKUP = "rpg.rules.lookup"
_TOOL_CONTEXT_BUILD = "rpg.context.build"
_TOOL_EVENTS_RECORD = "rpg.events.record"
_TOOL_PROPOSALS_APPLY = "rpg.proposals.apply"
_TOOL_PROPOSALS_REJECT = "rpg.proposals.reject"

_READ_TOOL_NAMES = {
    _TOOL_ADAPTERS_LIST,
    _TOOL_SESSIONS_GET,
    _TOOL_RULES_LOOKUP,
    _TOOL_CONTEXT_BUILD,
}
_WRITE_TOOL_NAMES = {
    _TOOL_EVENTS_RECORD,
    _TOOL_PROPOSALS_APPLY,
    _TOOL_PROPOSALS_REJECT,
}
_ALL_TOOL_NAMES = _READ_TOOL_NAMES | _WRITE_TOOL_NAMES
_MAX_CONTEXT_CHARS = 24000
_MIN_CONTEXT_CHARS = 1000
_MAX_IDEMPOTENCY_KEY_CHARS = 256
_MAX_QUERY_CHARS = 500
_MAX_REVIEW_NOTES_CHARS = 2000


class RPGModule(BaseModule):
    """Expose RPG runtime orchestration tools through MCP Unified."""

    def __init__(self, config: ModuleConfig) -> None:
        super().__init__(config)

    async def on_initialize(self) -> None:
        logger.info("Initializing RPG MCP module: {}", self.name)

    async def on_shutdown(self) -> None:
        logger.info("Shutting down RPG MCP module: {}", self.name)

    async def check_health(self) -> dict[str, bool]:
        return {"initialized": True, "adapter_registry": True}

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            self._strict_tool(
                name=_TOOL_ADAPTERS_LIST,
                description="List bundled RPG rules adapters.",
                parameters={"properties": {}, "required": []},
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
            self._strict_tool(
                name=_TOOL_SESSIONS_GET,
                description="Get RPG session metadata and the current deterministic snapshot.",
                parameters={
                    "properties": {
                        "session_id": {"type": "integer", "minimum": 1},
                    },
                    "required": ["session_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True, "auth_required": True},
            ),
            self._strict_tool(
                name=_TOOL_RULES_LOOKUP,
                description="Look up cited RPG rules references for a session.",
                parameters={
                    "properties": {
                        "session_id": {"type": "integer", "minimum": 1},
                        "query": {"type": "string", "minLength": 1, "maxLength": _MAX_QUERY_CHARS},
                    },
                    "required": ["session_id", "query"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True, "auth_required": True},
            ),
            self._strict_tool(
                name=_TOOL_CONTEXT_BUILD,
                description="Build bounded RPG session context with citation diagnostics.",
                parameters={
                    "properties": {
                        "session_id": {"type": "integer", "minimum": 1},
                        "query": {"type": "string", "maxLength": _MAX_QUERY_CHARS},
                        "max_chars": {
                            "type": "integer",
                            "minimum": _MIN_CONTEXT_CHARS,
                            "maximum": _MAX_CONTEXT_CHARS,
                        },
                    },
                    "required": ["session_id"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True, "auth_required": True},
            ),
            self._strict_tool(
                name=_TOOL_EVENTS_RECORD,
                description="Record trusted MCP-origin RPG session events or create a proposal by authority policy.",
                parameters={
                    "properties": {
                        "session_id": {"type": "integer", "minimum": 1},
                        "expected_last_event_sequence": {"type": "integer", "minimum": 0},
                        "events": {"type": "array", "items": {"type": "object"}, "minItems": 1},
                        "idempotencyKey": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": _MAX_IDEMPOTENCY_KEY_CHARS,
                        },
                        "idempotency_key": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": _MAX_IDEMPOTENCY_KEY_CHARS,
                        },
                    },
                    "required": ["session_id", "expected_last_event_sequence", "events"],
                },
                metadata=self._write_metadata(),
            ),
            self._strict_tool(
                name=_TOOL_PROPOSALS_APPLY,
                description="Apply a pending RPG proposal atomically.",
                parameters={
                    "properties": {
                        "session_id": {"type": "integer", "minimum": 1},
                        "proposal_id": {"type": "integer", "minimum": 1},
                        "expected_last_event_sequence": {"type": "integer", "minimum": 0},
                        "review_notes": {"type": "string", "maxLength": _MAX_REVIEW_NOTES_CHARS},
                        "idempotencyKey": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": _MAX_IDEMPOTENCY_KEY_CHARS,
                        },
                        "idempotency_key": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": _MAX_IDEMPOTENCY_KEY_CHARS,
                        },
                    },
                    "required": ["session_id", "proposal_id", "expected_last_event_sequence"],
                },
                metadata=self._write_metadata(),
            ),
            self._strict_tool(
                name=_TOOL_PROPOSALS_REJECT,
                description="Reject a pending RPG proposal.",
                parameters={
                    "properties": {
                        "session_id": {"type": "integer", "minimum": 1},
                        "proposal_id": {"type": "integer", "minimum": 1},
                        "review_notes": {"type": "string", "maxLength": _MAX_REVIEW_NOTES_CHARS},
                        "idempotencyKey": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": _MAX_IDEMPOTENCY_KEY_CHARS,
                        },
                        "idempotency_key": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": _MAX_IDEMPOTENCY_KEY_CHARS,
                        },
                    },
                    "required": ["session_id", "proposal_id"],
                },
                metadata=self._write_metadata(),
            ),
        ]

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> Any:
        if tool_name == _TOOL_ADAPTERS_LIST:
            return self._list_adapters()
        if tool_name not in _ALL_TOOL_NAMES:
            raise ValueError(f"Unknown RPG tool: {tool_name}")

        args = arguments or {}
        if tool_name in _WRITE_TOOL_NAMES:
            self.validate_tool_arguments(tool_name, args)
            self._required_idempotency_key(args)
        else:
            self._validate_read_arguments(tool_name, args)
        service, closeables = self._service_for_context(
            context,
            include_rules_retrieval=tool_name in {_TOOL_RULES_LOOKUP, _TOOL_CONTEXT_BUILD},
        )
        try:
            if tool_name == _TOOL_SESSIONS_GET:
                return self._get_session(service, self._positive_int_arg(args, "session_id"))
            if tool_name == _TOOL_RULES_LOOKUP:
                return _to_jsonable(
                    await service.lookup_rules(
                        session_id=self._positive_int_arg(args, "session_id"),
                        query=self._str_arg(args, "query", max_chars=_MAX_QUERY_CHARS),
                    )
                )
            if tool_name == _TOOL_CONTEXT_BUILD:
                return _to_jsonable(
                    await service.build_context(
                        session_id=self._positive_int_arg(args, "session_id"),
                        query=self._optional_str_arg(args, "query", max_chars=_MAX_QUERY_CHARS),
                        max_chars=self._optional_bounded_int_arg(
                            args,
                            "max_chars",
                            _MAX_CONTEXT_CHARS,
                            min_value=_MIN_CONTEXT_CHARS,
                            max_value=_MAX_CONTEXT_CHARS,
                        ),
                    )
                )
            if tool_name == _TOOL_EVENTS_RECORD:
                result = service.record_events(
                    session_id=self._positive_int_arg(args, "session_id"),
                    events=self._events_arg(args),
                    source_type="mcp",
                    expected_last_event_sequence=self._non_negative_int_arg(args, "expected_last_event_sequence"),
                    idempotency_key=self._required_idempotency_key(args),
                )
                return {
                    "committed_events": [_to_jsonable(event) for event in result.committed_events],
                    "proposal": _to_jsonable(result.proposal) if result.proposal is not None else None,
                }
            if tool_name == _TOOL_PROPOSALS_APPLY:
                result = service.apply_proposal(
                    session_id=self._positive_int_arg(args, "session_id"),
                    proposal_id=self._positive_int_arg(args, "proposal_id"),
                    expected_last_event_sequence=self._non_negative_int_arg(args, "expected_last_event_sequence"),
                    idempotency_key=self._required_idempotency_key(args),
                    review_notes=self._optional_str_arg(args, "review_notes", max_chars=_MAX_REVIEW_NOTES_CHARS),
                )
                return {
                    "committed_events": [_to_jsonable(event) for event in result.committed_events],
                    "proposal": _to_jsonable(result.proposal) if result.proposal is not None else None,
                }
            if tool_name == _TOOL_PROPOSALS_REJECT:
                return _to_jsonable(
                    service.reject_proposal(
                        session_id=self._positive_int_arg(args, "session_id"),
                        proposal_id=self._positive_int_arg(args, "proposal_id"),
                        idempotency_key=self._required_idempotency_key(args),
                        review_notes=self._optional_str_arg(args, "review_notes", max_chars=_MAX_REVIEW_NOTES_CHARS),
                    )
                )
        finally:
            for closeable in reversed(closeables):
                self._close_db(closeable)
        raise ValueError(f"Unknown RPG tool: {tool_name}")

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        if tool_name not in _WRITE_TOOL_NAMES:
            return
        args = arguments or {}
        if tool_name == _TOOL_EVENTS_RECORD and not self._events_arg(args):
            raise ValueError("events must contain at least one event")
        if tool_name in {_TOOL_EVENTS_RECORD, _TOOL_PROPOSALS_APPLY}:
            self._non_negative_int_arg(args, "expected_last_event_sequence")
        if tool_name in {_TOOL_EVENTS_RECORD, _TOOL_PROPOSALS_APPLY, _TOOL_PROPOSALS_REJECT}:
            self._positive_int_arg(args, "session_id")
        if tool_name in {_TOOL_PROPOSALS_APPLY, _TOOL_PROPOSALS_REJECT}:
            self._positive_int_arg(args, "proposal_id")
        if tool_name in {_TOOL_PROPOSALS_APPLY, _TOOL_PROPOSALS_REJECT}:
            self._optional_str_arg(args, "review_notes", max_chars=_MAX_REVIEW_NOTES_CHARS)

    def _validate_read_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        if tool_name == _TOOL_SESSIONS_GET:
            self._positive_int_arg(arguments, "session_id")
        elif tool_name == _TOOL_RULES_LOOKUP:
            self._positive_int_arg(arguments, "session_id")
            self._str_arg(arguments, "query", max_chars=_MAX_QUERY_CHARS)
        elif tool_name == _TOOL_CONTEXT_BUILD:
            self._positive_int_arg(arguments, "session_id")
            self._optional_str_arg(arguments, "query", max_chars=_MAX_QUERY_CHARS)
            if arguments.get("max_chars") is not None:
                self._optional_bounded_int_arg(
                    arguments,
                    "max_chars",
                    _MAX_CONTEXT_CHARS,
                    min_value=_MIN_CONTEXT_CHARS,
                    max_value=_MAX_CONTEXT_CHARS,
                )

    def _service_for_context(
        self,
        context: Any | None,
        *,
        include_rules_retrieval: bool = False,
    ) -> tuple[RPGService, list[Any]]:
        if context is None or not str(getattr(context, "user_id", "") or "").strip():
            raise ValueError("RPG MCP tools require an authenticated user context")
        db_paths = getattr(context, "db_paths", None)
        if not isinstance(db_paths, dict) or not db_paths.get("chacha"):
            raise ValueError("ChaChaNotes DB path not available in context")
        try:
            owner_user_id = int(str(context.user_id))
        except (TypeError, ValueError) as exc:
            raise ValueError("RPG MCP tools require an integer user context") from exc
        db = CharactersRAGDB(
            db_path=str(db_paths["chacha"]),
            client_id=f"mcp_rpg_{self.config.name}",
        )
        closeables: list[Any] = [db]
        rules_source_validator = None
        rules_lookup_service = None
        media_path = str(db_paths.get("media") or "").strip() if include_rules_retrieval else ""
        if media_path:
            media_db = MediaDatabase(
                db_path=media_path,
                client_id=str(owner_user_id),
            )
            closeables.append(media_db)
            collections_db = CollectionsDatabase.from_backend(owner_user_id, media_db.backend)
            rules_source_validator = RPGRulesSourceValidator(
                media_db=media_db,
                collections_db=collections_db,
            )
            rag_retriever = MultiDatabaseRetriever(
                {"media_db": media_path},
                user_id=str(owner_user_id),
                media_db=media_db,
            )
            rules_lookup_service = RulesLookupService(
                retriever=RulesRetrievalAdapter(
                    source_validator=rules_source_validator,
                    rag_retriever=rag_retriever,
                )
            )
        return (
            RPGService(
                repo=RPGRepository.initialized(db),
                owner_user_id=owner_user_id,
                rules_source_validator=rules_source_validator,
                rules_lookup_service=rules_lookup_service,
            ),
            closeables,
        )

    def _get_session(self, service: RPGService, session_id: int) -> dict[str, Any]:
        session = service.repo.get_session(owner_user_id=service.owner_user_id, session_id=session_id)
        snapshot = service.get_snapshot(session_id)
        return {
            "session": _to_jsonable(session),
            "snapshot": {
                "snapshot_version": snapshot.snapshot_version,
                "last_event_sequence": snapshot.last_event_sequence,
                "state": _to_jsonable(snapshot.snapshot),
                "diagnostics": _to_jsonable(snapshot.diagnostics),
            },
        }

    def _list_adapters(self) -> dict[str, Any]:
        registry = build_default_adapter_registry()
        return {"adapters": [_to_jsonable(info) for info in registry.list_infos()]}

    @staticmethod
    def _strict_tool(
        *,
        name: str,
        description: str,
        parameters: dict[str, Any],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        tool = create_tool_definition(
            name=name,
            description=description,
            parameters=parameters,
            metadata=metadata,
        )
        tool["inputSchema"]["additionalProperties"] = False
        return tool

    @staticmethod
    def _write_metadata() -> dict[str, Any]:
        return {
            "category": "management",
            "readOnlyHint": False,
            "auth_required": True,
            "is_write": True,
            "mutates_state": True,
            "requires_confirmation": True,
            "agent_write_policy": "approval_required",
            "governance_preflight_required": True,
            "sensitive": True,
        }

    @staticmethod
    def _integer_arg(args: dict[str, Any], name: str) -> int:
        value = args.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        return value

    @classmethod
    def _positive_int_arg(cls, args: dict[str, Any], name: str) -> int:
        value = cls._integer_arg(args, name)
        if value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @classmethod
    def _non_negative_int_arg(cls, args: dict[str, Any], name: str) -> int:
        value = cls._integer_arg(args, name)
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return value

    @staticmethod
    def _optional_int_arg(args: dict[str, Any], name: str, default: int) -> int:
        value = args.get(name)
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        return value

    @classmethod
    def _optional_bounded_int_arg(
        cls,
        args: dict[str, Any],
        name: str,
        default: int,
        *,
        min_value: int,
        max_value: int,
    ) -> int:
        value = cls._optional_int_arg(args, name, default)
        if value < min_value or value > max_value:
            raise ValueError(f"{name} must be between {min_value} and {max_value}")
        return value

    @staticmethod
    def _str_arg(args: dict[str, Any], name: str, *, max_chars: int | None = None) -> str:
        value = args.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        trimmed = value.strip()
        if max_chars is not None and len(trimmed) > max_chars:
            raise ValueError(f"{name} must be <= {max_chars} characters")
        return trimmed

    @staticmethod
    def _optional_str_arg(args: dict[str, Any], name: str, *, max_chars: int | None = None) -> str | None:
        value = args.get(name)
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        trimmed = value.strip()
        if not trimmed:
            return None
        if max_chars is not None and len(trimmed) > max_chars:
            raise ValueError(f"{name} must be <= {max_chars} characters")
        return trimmed

    @staticmethod
    def _events_arg(args: dict[str, Any]) -> list[dict[str, Any]]:
        events = args.get("events")
        if not isinstance(events, list) or not events:
            raise ValueError("events must contain at least one event")
        if not all(isinstance(event, dict) for event in events):
            raise ValueError("events must be objects")
        return [dict(event) for event in events]

    @staticmethod
    def _idempotency_key(args: dict[str, Any]) -> str:
        value = args.get("idempotencyKey")
        if value is None:
            value = args.get("idempotency_key")
        if not isinstance(value, str) or not value.strip():
            return ""
        return value.strip()

    @classmethod
    def _required_idempotency_key(cls, args: dict[str, Any]) -> str:
        key = cls._idempotency_key(args)
        if not key:
            raise ValueError("idempotencyKey is required")
        if len(key) > _MAX_IDEMPOTENCY_KEY_CHARS:
            raise ValueError(f"idempotencyKey must be <= {_MAX_IDEMPOTENCY_KEY_CHARS} characters")
        return key

    @staticmethod
    def _close_db(db: Any) -> None:
        close_all = getattr(db, "close_all_connections", None)
        if callable(close_all):
            close_all()
            return
        close_connection = getattr(db, "close_connection", None)
        if callable(close_connection):
            close_connection()
            return
        close = getattr(db, "close", None)
        if callable(close):
            close()


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _to_jsonable(asdict(value))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value
