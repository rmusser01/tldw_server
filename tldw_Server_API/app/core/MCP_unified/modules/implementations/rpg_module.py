"""MCP tools for the generic RPG campaign/session runtime."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.DB_Management.Collections_DB import CollectionsDatabase
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.DB_Management.RPG_DB import RPGRepository
from tldw_Server_API.app.core.RAG.rag_service.database_retrievers import MultiDatabaseRetriever
from tldw_Server_API.app.core.RPG.errors import RPGError
from tldw_Server_API.app.core.RPG.rules.adapters import build_default_adapter_registry
from tldw_Server_API.app.core.RPG.rules.answering import ChatRulesAnswerGenerator, RulesAnswerOptions
from tldw_Server_API.app.core.RPG.rules.lookup import RulesLookupService
from tldw_Server_API.app.core.RPG.rules.refs import RulesPackRef
from tldw_Server_API.app.core.RPG.rules.retrieval import RulesRetrievalAdapter
from tldw_Server_API.app.core.RPG.rules.source_validation import RPGRulesSourceValidator
from tldw_Server_API.app.core.RPG.service import RPGService

from ..base import BaseModule, ModuleConfig, create_tool_definition

_TOOL_ADAPTERS_LIST = "rpg.adapters.list"
_TOOL_SESSIONS_GET = "rpg.sessions.get"
_TOOL_CAMPAIGN_RULES_PACKS_GET = "rpg.campaigns.rules_packs.get"
_TOOL_CAMPAIGN_RULES_PACKS_REPLACE = "rpg.campaigns.rules_packs.replace"
_TOOL_SESSION_RULES_PACKS_GET = "rpg.sessions.rules_packs.get"
_TOOL_SESSION_RULES_PACKS_REPLACE = "rpg.sessions.rules_packs.replace"
_TOOL_RULES_LOOKUP = "rpg.rules.lookup"
_TOOL_CONTEXT_BUILD = "rpg.context.build"
_TOOL_EVENTS_RECORD = "rpg.events.record"
_TOOL_PROPOSALS_APPLY = "rpg.proposals.apply"
_TOOL_PROPOSALS_REJECT = "rpg.proposals.reject"

_PERM_RPG_CAMPAIGNS_READ = "rpg.campaigns.read"
_PERM_RPG_CAMPAIGNS_MANAGE = "rpg.campaigns.manage"
_PERM_RPG_SESSIONS_READ = "rpg.sessions.read"
_PERM_RPG_SESSIONS_MANAGE = "rpg.sessions.manage"
_PERM_RPG_PROPOSALS_REVIEW = "rpg.proposals.review"
_PERM_RPG_RULES_READ = "rpg.rules.read"
_PERM_MEDIA_READ = "media.read"
_PERM_CHAT_COMPLETIONS = "chat.completions"

_READ_TOOL_NAMES = {
    _TOOL_ADAPTERS_LIST,
    _TOOL_SESSIONS_GET,
    _TOOL_CAMPAIGN_RULES_PACKS_GET,
    _TOOL_SESSION_RULES_PACKS_GET,
    _TOOL_RULES_LOOKUP,
    _TOOL_CONTEXT_BUILD,
}
_WRITE_TOOL_NAMES = {
    _TOOL_CAMPAIGN_RULES_PACKS_REPLACE,
    _TOOL_SESSION_RULES_PACKS_REPLACE,
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
_MAX_RULES_PACK_REFS = 50
_MAX_PROVIDER_CHARS = 100
_MAX_MODEL_CHARS = 200
_MIN_ANSWER_TOKENS = 64
_MAX_ANSWER_TOKENS = 2000
_DEFAULT_ANSWER_TOKENS = 600
_DEFAULT_ANSWER_TEMPERATURE = 0.2
_MAX_ANSWER_TEMPERATURE = 2.0
_ANSWER_GENERATION_CONTROLS_METADATA = "mcp_rpg_answer_generation_controls"

_TOOL_REQUIRED_PERMISSIONS = {
    _TOOL_SESSIONS_GET: [_PERM_RPG_SESSIONS_READ],
    _TOOL_CAMPAIGN_RULES_PACKS_GET: [_PERM_RPG_CAMPAIGNS_READ, _PERM_MEDIA_READ],
    _TOOL_CAMPAIGN_RULES_PACKS_REPLACE: [_PERM_RPG_CAMPAIGNS_MANAGE, _PERM_MEDIA_READ],
    _TOOL_SESSION_RULES_PACKS_GET: [_PERM_RPG_SESSIONS_READ, _PERM_MEDIA_READ],
    _TOOL_SESSION_RULES_PACKS_REPLACE: [_PERM_RPG_SESSIONS_MANAGE, _PERM_MEDIA_READ],
    _TOOL_RULES_LOOKUP: [_PERM_RPG_RULES_READ, _PERM_MEDIA_READ],
    _TOOL_CONTEXT_BUILD: [_PERM_RPG_SESSIONS_READ, _PERM_RPG_RULES_READ, _PERM_MEDIA_READ],
    _TOOL_EVENTS_RECORD: [_PERM_RPG_SESSIONS_MANAGE],
    _TOOL_PROPOSALS_APPLY: [_PERM_RPG_PROPOSALS_REVIEW],
    _TOOL_PROPOSALS_REJECT: [_PERM_RPG_PROPOSALS_REVIEW],
}


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
                metadata=self._read_metadata(required_permissions=[_PERM_RPG_SESSIONS_READ]),
            ),
            self._strict_tool(
                name=_TOOL_CAMPAIGN_RULES_PACKS_GET,
                description="Get rules-pack references attached to an RPG campaign.",
                parameters={
                    "properties": {
                        "campaign_id": {"type": "integer", "minimum": 1},
                    },
                    "required": ["campaign_id"],
                },
                metadata=self._read_metadata(required_permissions=[_PERM_RPG_CAMPAIGNS_READ, _PERM_MEDIA_READ]),
            ),
            self._strict_tool(
                name=_TOOL_CAMPAIGN_RULES_PACKS_REPLACE,
                description="Replace the full rules-pack reference list attached to an RPG campaign.",
                parameters={
                    "properties": {
                        "campaign_id": {"type": "integer", "minimum": 1},
                        "expected_version": {"type": "integer", "minimum": 1},
                        "refs": {
                            "type": "array",
                            "items": self._rules_pack_ref_schema(),
                            "maxItems": _MAX_RULES_PACK_REFS,
                        },
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
                    "required": ["campaign_id", "expected_version", "refs"],
                },
                metadata=self._write_metadata(
                    required_permissions=[_PERM_RPG_CAMPAIGNS_MANAGE, _PERM_MEDIA_READ],
                ),
            ),
            self._strict_tool(
                name=_TOOL_SESSION_RULES_PACKS_GET,
                description="Get rules-pack references attached to an RPG session.",
                parameters={
                    "properties": {
                        "session_id": {"type": "integer", "minimum": 1},
                    },
                    "required": ["session_id"],
                },
                metadata=self._read_metadata(required_permissions=[_PERM_RPG_SESSIONS_READ, _PERM_MEDIA_READ]),
            ),
            self._strict_tool(
                name=_TOOL_SESSION_RULES_PACKS_REPLACE,
                description="Replace the full rules-pack reference list attached to an RPG session.",
                parameters={
                    "properties": {
                        "session_id": {"type": "integer", "minimum": 1},
                        "expected_version": {"type": "integer", "minimum": 1},
                        "refs": {
                            "type": "array",
                            "items": self._rules_pack_ref_schema(),
                            "maxItems": _MAX_RULES_PACK_REFS,
                        },
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
                    "required": ["session_id", "expected_version", "refs"],
                },
                metadata=self._write_metadata(
                    required_permissions=[_PERM_RPG_SESSIONS_MANAGE, _PERM_MEDIA_READ],
                ),
            ),
            self._strict_tool(
                name=_TOOL_RULES_LOOKUP,
                description="Look up cited RPG rules references or generate a grounded answer for a session.",
                parameters={
                    "properties": {
                        "session_id": {"type": "integer", "minimum": 1},
                        "query": {"type": "string", "minLength": 1, "maxLength": _MAX_QUERY_CHARS},
                        "mode": {"type": "string", "enum": ["lookup", "answer"], "default": "lookup"},
                        "provider": {"type": "string", "maxLength": _MAX_PROVIDER_CHARS},
                        "model": {"type": "string", "maxLength": _MAX_MODEL_CHARS},
                        "temperature": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": _MAX_ANSWER_TEMPERATURE,
                            "default": _DEFAULT_ANSWER_TEMPERATURE,
                        },
                        "max_tokens": {
                            "type": "integer",
                            "minimum": _MIN_ANSWER_TOKENS,
                            "maximum": _MAX_ANSWER_TOKENS,
                            "default": _DEFAULT_ANSWER_TOKENS,
                        },
                    },
                    "required": ["session_id", "query"],
                },
                metadata=self._read_metadata(required_permissions=[_PERM_RPG_RULES_READ, _PERM_MEDIA_READ]),
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
                metadata=self._read_metadata(
                    required_permissions=[_PERM_RPG_SESSIONS_READ, _PERM_RPG_RULES_READ, _PERM_MEDIA_READ],
                ),
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
                metadata=self._write_metadata(required_permissions=[_PERM_RPG_SESSIONS_MANAGE]),
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
                metadata=self._write_metadata(required_permissions=[_PERM_RPG_PROPOSALS_REVIEW]),
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
                metadata=self._write_metadata(required_permissions=[_PERM_RPG_PROPOSALS_REVIEW]),
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
        rules_pack_refs = (
            self._rules_pack_refs_arg(args)
            if tool_name in {_TOOL_CAMPAIGN_RULES_PACKS_REPLACE, _TOOL_SESSION_RULES_PACKS_REPLACE}
            else None
        )
        self._require_context_permissions(tool_name, args, context)
        service, closeables = self._service_for_context(
            context,
            include_rules_retrieval=tool_name in {_TOOL_RULES_LOOKUP, _TOOL_CONTEXT_BUILD},
            include_rules_source_validation=tool_name in {
                _TOOL_CAMPAIGN_RULES_PACKS_REPLACE,
                _TOOL_SESSION_RULES_PACKS_REPLACE,
            },
            require_media_db=self._has_enabled_rules_pack_refs(rules_pack_refs),
        )
        try:
            if tool_name == _TOOL_SESSIONS_GET:
                return self._get_session(service, self._positive_int_arg(args, "session_id"))
            if tool_name == _TOOL_CAMPAIGN_RULES_PACKS_GET:
                return _to_jsonable(
                    service.list_campaign_rules_pack_refs(
                        self._positive_int_arg(args, "campaign_id"),
                    )
                )
            if tool_name == _TOOL_CAMPAIGN_RULES_PACKS_REPLACE:
                return _to_jsonable(
                    await service.replace_campaign_rules_pack_refs(
                        campaign_id=self._positive_int_arg(args, "campaign_id"),
                        refs=rules_pack_refs or [],
                        expected_version=self._positive_int_arg(args, "expected_version"),
                        idempotency_key=self._required_idempotency_key(args),
                        source_type="mcp",
                    )
                )
            if tool_name == _TOOL_SESSION_RULES_PACKS_GET:
                return _to_jsonable(
                    service.list_session_rules_pack_refs(
                        self._positive_int_arg(args, "session_id"),
                    )
                )
            if tool_name == _TOOL_SESSION_RULES_PACKS_REPLACE:
                return _to_jsonable(
                    await service.replace_session_rules_pack_refs(
                        session_id=self._positive_int_arg(args, "session_id"),
                        refs=rules_pack_refs or [],
                        expected_version=self._positive_int_arg(args, "expected_version"),
                        idempotency_key=self._required_idempotency_key(args),
                        source_type="mcp",
                    )
                )
            if tool_name == _TOOL_RULES_LOOKUP:
                return _to_jsonable(
                    await service.lookup_rules(
                        session_id=self._positive_int_arg(args, "session_id"),
                        query=self._str_arg(args, "query", max_chars=_MAX_QUERY_CHARS),
                        mode=self._lookup_mode_arg(args),
                        answer_options=self._answer_options_arg(args),
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
        if tool_name == _TOOL_CAMPAIGN_RULES_PACKS_REPLACE:
            self._positive_int_arg(args, "campaign_id")
            self._positive_int_arg(args, "expected_version")
            self._rules_pack_refs_arg(args)
        if tool_name == _TOOL_SESSION_RULES_PACKS_REPLACE:
            self._positive_int_arg(args, "session_id")
            self._positive_int_arg(args, "expected_version")
            self._rules_pack_refs_arg(args)
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
        elif tool_name == _TOOL_CAMPAIGN_RULES_PACKS_GET:
            self._positive_int_arg(arguments, "campaign_id")
        elif tool_name == _TOOL_SESSION_RULES_PACKS_GET:
            self._positive_int_arg(arguments, "session_id")
        elif tool_name == _TOOL_RULES_LOOKUP:
            self._positive_int_arg(arguments, "session_id")
            self._str_arg(arguments, "query", max_chars=_MAX_QUERY_CHARS)
            self._lookup_mode_arg(arguments)
            self._answer_options_arg(arguments)
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

    def _require_context_permissions(
        self,
        tool_name: str,
        args: dict[str, Any],
        context: Any | None,
    ) -> None:
        required = list(_TOOL_REQUIRED_PERMISSIONS.get(tool_name, ()))
        if tool_name == _TOOL_RULES_LOOKUP and self._lookup_mode_arg(args) == "answer":
            required.append(_PERM_CHAT_COMPLETIONS)
        if not required:
            return
        if context is None:
            raise ValueError("RPG MCP tools require an authenticated user context")

        metadata = getattr(context, "metadata", None)
        roles, permissions = self._context_role_and_permission_sets(metadata)
        if "admin" in roles or "*" in permissions:
            if tool_name == _TOOL_RULES_LOOKUP and self._lookup_mode_arg(args) == "answer":
                self._require_answer_generation_controls(metadata)
            return

        missing = [permission for permission in required if permission not in permissions]
        if missing:
            raise PermissionError(f"RPG MCP tool requires permissions: {', '.join(missing)}")
        if tool_name == _TOOL_RULES_LOOKUP and self._lookup_mode_arg(args) == "answer":
            self._require_answer_generation_controls(metadata)

    @staticmethod
    def _context_role_and_permission_sets(metadata: Any) -> tuple[set[str], set[str]]:
        if not isinstance(metadata, dict):
            return set(), set()
        return (
            _normalized_metadata_values(metadata.get("roles")),
            _normalized_metadata_values(metadata.get("permissions")),
        )

    @staticmethod
    def _require_answer_generation_controls(metadata: Any) -> None:
        if not isinstance(metadata, dict):
            raise PermissionError("RPG MCP answer mode requires host-enforced answer generation controls")
        value = metadata.get(_ANSWER_GENERATION_CONTROLS_METADATA)
        if value is True:
            return
        if isinstance(value, str) and value.strip().lower() in {"enforced", "true", "1", "yes"}:
            return
        raise PermissionError("RPG MCP answer mode requires host-enforced answer generation controls")

    def _service_for_context(
        self,
        context: Any | None,
        *,
        include_rules_retrieval: bool = False,
        include_rules_source_validation: bool = False,
        require_media_db: bool = False,
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
        include_media = include_rules_retrieval or include_rules_source_validation
        media_path = str(db_paths.get("media") or "").strip() if include_media else ""
        if require_media_db and not media_path:
            raise ValueError("Media DB path not available in context")
        closeables: list[Any] = []
        rules_source_validator = None
        rules_lookup_service = None
        try:
            db = CharactersRAGDB(
                db_path=str(db_paths["chacha"]),
                client_id=f"mcp_rpg_{self.config.name}",
            )
            closeables.append(db)
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
                if include_rules_retrieval:
                    rag_retriever = MultiDatabaseRetriever(
                        {"media_db": media_path},
                        user_id=str(owner_user_id),
                        media_db=media_db,
                    )
                    closeables.append(rag_retriever)
                    rules_lookup_service = RulesLookupService(
                        retriever=RulesRetrievalAdapter(
                            source_validator=rules_source_validator,
                            rag_retriever=rag_retriever,
                        ),
                        answer_generator=ChatRulesAnswerGenerator(),
                    )
            elif include_rules_retrieval:
                rules_lookup_service = RulesLookupService(
                    retriever=_MissingMediaRulesRetriever(),
                    answer_generator=ChatRulesAnswerGenerator(),
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
        except Exception:
            for closeable in reversed(closeables):
                self._close_db(closeable)
            raise

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
    def _read_metadata(*, required_permissions: list[str] | None = None) -> dict[str, Any]:
        metadata = {
            "category": "retrieval",
            "readOnlyHint": True,
            "auth_required": True,
        }
        if required_permissions is not None:
            metadata["required_permissions"] = list(required_permissions)
        return metadata

    @staticmethod
    def _write_metadata(*, required_permissions: list[str] | None = None) -> dict[str, Any]:
        metadata = {
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
        if required_permissions is not None:
            metadata["required_permissions"] = list(required_permissions)
        return metadata

    @staticmethod
    def _rules_pack_ref_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "source_type": {"type": "string", "enum": ["media_item", "media_collection"]},
                "source_id": {"type": "integer", "minimum": 1},
                "display_name": {"type": "string"},
                "enabled": {"type": "boolean", "default": True},
                "metadata": {"type": "object", "default": {}},
            },
            "required": ["source_type", "source_id"],
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
    def _optional_float_arg(args: dict[str, Any], name: str, default: float) -> float:
        value = args.get(name)
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        return float(value)

    @classmethod
    def _optional_bounded_float_arg(
        cls,
        args: dict[str, Any],
        name: str,
        default: float,
        *,
        min_value: float,
        max_value: float,
    ) -> float:
        value = cls._optional_float_arg(args, name, default)
        if value < min_value or value > max_value:
            raise ValueError(f"{name} must be between {min_value:g} and {max_value:g}")
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

    @classmethod
    def _lookup_mode_arg(cls, args: dict[str, Any]) -> str:
        value = args.get("mode")
        if value is None:
            return "lookup"
        if value not in {"lookup", "answer"}:
            raise ValueError("mode must be lookup or answer")
        return str(value)

    @classmethod
    def _answer_options_arg(cls, args: dict[str, Any]) -> RulesAnswerOptions:
        return RulesAnswerOptions(
            provider=cls._optional_str_arg(args, "provider", max_chars=_MAX_PROVIDER_CHARS),
            model=cls._optional_str_arg(args, "model", max_chars=_MAX_MODEL_CHARS),
            temperature=cls._optional_bounded_float_arg(
                args,
                "temperature",
                _DEFAULT_ANSWER_TEMPERATURE,
                min_value=0.0,
                max_value=_MAX_ANSWER_TEMPERATURE,
            ),
            max_tokens=cls._optional_bounded_int_arg(
                args,
                "max_tokens",
                _DEFAULT_ANSWER_TOKENS,
                min_value=_MIN_ANSWER_TOKENS,
                max_value=_MAX_ANSWER_TOKENS,
            ),
        )

    @classmethod
    def _rules_pack_refs_arg(cls, args: dict[str, Any]) -> list[dict[str, Any]]:
        refs = args.get("refs")
        if not isinstance(refs, list):
            raise ValueError("refs must be a list")
        if len(refs) > _MAX_RULES_PACK_REFS:
            raise ValueError(f"refs must contain at most {_MAX_RULES_PACK_REFS} items")

        normalized: list[dict[str, Any]] = []
        for index, ref in enumerate(refs):
            if not isinstance(ref, dict):
                raise ValueError(f"refs[{index}] must be an object")
            source_type = ref.get("source_type")
            if source_type not in {"media_item", "media_collection"}:
                raise ValueError(f"refs[{index}].source_type must be media_item or media_collection")
            source_id = cls._positive_int_arg(ref, "source_id")
            enabled = ref.get("enabled", True)
            if not isinstance(enabled, bool):
                raise ValueError(f"refs[{index}].enabled must be a boolean")
            metadata = ref.get("metadata", {})
            if metadata is None:
                metadata = {}
            if not isinstance(metadata, dict):
                raise ValueError(f"refs[{index}].metadata must be an object")
            item: dict[str, Any] = {
                "source_type": source_type,
                "source_id": source_id,
                "enabled": enabled,
                "metadata": dict(metadata),
            }
            display_name = ref.get("display_name")
            if display_name is not None:
                if not isinstance(display_name, str):
                    raise ValueError(f"refs[{index}].display_name must be a string")
                item["display_name"] = display_name.strip()
            normalized.append(item)
        return normalized

    @staticmethod
    def _has_enabled_rules_pack_refs(refs: list[dict[str, Any]] | None) -> bool:
        if not refs:
            return False
        return any(ref.get("enabled", True) is True for ref in refs)

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


class _MissingMediaRulesRetriever:
    async def retrieve(
        self,
        *,
        owner_user_id: int,
        query: str,
        refs: list[RulesPackRef],
        max_results: int,
    ) -> Any:
        del owner_user_id, query, refs, max_results
        raise _MissingMediaRulesAccessError("Media DB path not available in context")


class _MissingMediaRulesAccessError(ValueError, RPGError):
    pass


def _normalized_metadata_values(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value.strip().lower()} if value.strip() else set()
    if isinstance(value, (list, tuple, set)):
        return {str(item).strip().lower() for item in value if str(item).strip()}
    return set()


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
