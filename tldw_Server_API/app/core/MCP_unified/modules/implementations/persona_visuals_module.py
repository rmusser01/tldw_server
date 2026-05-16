"""MCP tools for Persona Visual packs and Buddy runtime state triggers."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Persona.visual_jobs import create_generate_candidate_job
from tldw_Server_API.app.core.Persona.visual_library_service import (
    PersonaVisualLibraryService,
    PersonaVisualLibraryServiceError,
)
from tldw_Server_API.app.core.Persona.visuals import (
    MAX_TRIGGER_DURATION_MS,
    MIN_TRIGGER_DURATION_MS,
    VISUAL_STATE_IDS,
    custom_visual_state_id_error,
)

from ..base import BaseModule, create_tool_definition


_PERSONA_VISUALS_NONCRITICAL_EXCEPTIONS = (
    asyncio.CancelledError,
    AssertionError,
    AttributeError,
    ConnectionError,
    FileNotFoundError,
    ImportError,
    IndexError,
    KeyError,
    LookupError,
    OSError,
    PermissionError,
    RuntimeError,
    TimeoutError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
)

_MAX_LIBRARY_ITEMS_OFFSET = 1_000_000
_MAX_TRIGGER_REASON_LENGTH = 200


class PersonaVisualsModule(BaseModule):
    """Internal persona visual-pack tools for draft edits and runtime state triggers."""

    async def on_initialize(self) -> None:
        logger.info("Initializing Persona Visuals module: {}", self.name)

    async def on_shutdown(self) -> None:
        logger.info("Shutting down Persona Visuals module: {}", self.name)

    async def check_health(self) -> dict[str, bool]:
        return {
            "initialized": True,
            "driver_available": CharactersRAGDB is not None,
            "jobs_available": JobManager is not None,
        }

    async def get_tools(self) -> list[dict[str, Any]]:
        return [
            create_tool_definition(
                name="persona_visuals.capabilities",
                description="Summarize active and draft visual packs for the scoped persona.",
                parameters={
                    "properties": {
                        "persona_id": {"type": "string"},
                    },
                },
                metadata={"category": "retrieval", "readOnlyHint": True, "auth_required": True},
            ),
            create_tool_definition(
                name="persona_visuals.library_items",
                description="List the current user's reference-backed personal Persona Visual library items.",
                parameters={
                    "properties": {
                        "limit": {"type": "integer", "minimum": 1, "maximum": 500, "default": 100},
                        "offset": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": _MAX_LIBRARY_ITEMS_OFFSET,
                            "default": 0,
                        },
                    },
                },
                metadata={"category": "retrieval", "readOnlyHint": True, "auth_required": True},
            ),
            create_tool_definition(
                name="persona_visuals.trigger_state",
                description="Emit a transient visual state override for the current persona session.",
                parameters={
                    "properties": {
                        "persona_id": {"type": "string"},
                        "state": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 96,
                            "description": (
                                "Built-in visual state or custom state declared by the "
                                "active Persona Visual pack."
                            ),
                        },
                        "duration_ms": {
                            "type": "integer",
                            "minimum": MIN_TRIGGER_DURATION_MS,
                            "maximum": MAX_TRIGGER_DURATION_MS,
                            "default": 1500,
                        },
                        "reason": {"type": "string", "maxLength": 200},
                    },
                    "required": ["state"],
                },
                metadata={"category": "runtime", "auth_required": True},
            ),
            create_tool_definition(
                name="persona_visuals.create_draft_pack",
                description="Create a draft visual pack for the scoped persona without activating it.",
                parameters={
                    "properties": {
                        "persona_id": {"type": "string"},
                        "title": {"type": "string", "minLength": 1, "maxLength": 200},
                        "manifest": {"type": "object"},
                        "parent_pack_id": {"type": "string"},
                    },
                    "required": ["title"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="persona_visuals.update_manifest",
                description="Replace a draft visual-pack manifest without activating the pack.",
                parameters={
                    "properties": {
                        "persona_id": {"type": "string"},
                        "pack_id": {"type": "string", "minLength": 1},
                        "manifest": {"type": "object"},
                        "expected_version": {"type": "integer", "minimum": 1},
                    },
                    "required": ["pack_id", "manifest"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="persona_visuals.use_library_item",
                description=(
                    "Create an inactive target-persona draft from a personal visual-library item."
                ),
                parameters={
                    "properties": {
                        "item_id": {"type": "string", "minLength": 1},
                        "target_persona_id": {"type": "string", "minLength": 1},
                        "title": {"type": "string", "minLength": 1, "maxLength": 200},
                    },
                    "required": ["item_id"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
            create_tool_definition(
                name="persona_visuals.enqueue_generation",
                description="Queue a persona visual generation job for human review.",
                parameters={
                    "properties": {
                        "persona_id": {"type": "string"},
                        "pack_id": {"type": "string", "minLength": 1},
                        "prompt": {"type": "string", "minLength": 1, "maxLength": 5000},
                        "target_state": {"type": "string", "enum": sorted(VISUAL_STATE_IDS)},
                        "backend": {"type": "string", "maxLength": 100},
                    },
                    "required": ["pack_id", "prompt"],
                },
                metadata={"category": "management", "auth_required": True},
            ),
        ]

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> Any:
        args = self.sanitize_input(arguments or {})
        try:
            self.validate_tool_arguments(tool_name, args)
        except _PERSONA_VISUALS_NONCRITICAL_EXCEPTIONS as exc:
            raise ValueError(f"Invalid arguments for {tool_name}: {exc}") from exc

        if tool_name == "persona_visuals.capabilities":
            return await asyncio.to_thread(self._capabilities_sync, args, context)
        if tool_name == "persona_visuals.library_items":
            return await asyncio.to_thread(self._library_items_sync, args, context)
        if tool_name == "persona_visuals.trigger_state":
            return await asyncio.to_thread(self._trigger_state_sync, args, context)
        if tool_name == "persona_visuals.create_draft_pack":
            return await asyncio.to_thread(self._create_draft_pack_sync, args, context)
        if tool_name == "persona_visuals.update_manifest":
            return await asyncio.to_thread(self._update_manifest_sync, args, context)
        if tool_name == "persona_visuals.use_library_item":
            return await asyncio.to_thread(self._use_library_item_sync, args, context)
        if tool_name == "persona_visuals.enqueue_generation":
            return await asyncio.to_thread(self._enqueue_generation_sync, args, context)
        raise ValueError(f"Unknown tool: {tool_name}")

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        if tool_name == "persona_visuals.capabilities":
            self._validate_optional_string(arguments, "persona_id")
            return
        if tool_name == "persona_visuals.library_items":
            if arguments.get("limit") is not None:
                limit = int(arguments["limit"])
                if limit < 1 or limit > 500:
                    raise ValueError("limit must be between 1 and 500")
            if arguments.get("offset") is not None:
                self._library_items_offset(arguments["offset"])
            return
        if tool_name == "persona_visuals.trigger_state":
            self._validate_optional_string(arguments, "persona_id")
            self._normalize_visual_state_id(arguments.get("state"))
            if arguments.get("duration_ms") is not None:
                int(arguments["duration_ms"])
            self._validate_optional_string(arguments, "reason")
            if len(str(arguments.get("reason") or "")) > _MAX_TRIGGER_REASON_LENGTH:
                raise ValueError(f"reason must be <= {_MAX_TRIGGER_REASON_LENGTH} chars")
            return
        if tool_name == "persona_visuals.create_draft_pack":
            self._validate_optional_string(arguments, "persona_id")
            title = str(arguments.get("title") or "").strip()
            if not title:
                raise ValueError("title is required")
            if len(title) > 200:
                raise ValueError("title must be <= 200 chars")
            self._validate_optional_string(arguments, "parent_pack_id")
            manifest = arguments.get("manifest", {})
            if manifest is not None and not isinstance(manifest, dict):
                raise ValueError("manifest must be an object")
            return
        if tool_name == "persona_visuals.update_manifest":
            self._validate_optional_string(arguments, "persona_id")
            pack_id = str(arguments.get("pack_id") or "").strip()
            if not pack_id:
                raise ValueError("pack_id is required")
            if not isinstance(arguments.get("manifest"), dict):
                raise ValueError("manifest must be an object")
            if arguments.get("expected_version") is not None and int(arguments["expected_version"]) < 1:
                raise ValueError("expected_version must be positive")
            return
        if tool_name == "persona_visuals.use_library_item":
            item_id = str(arguments.get("item_id") or "").strip()
            if not item_id:
                raise ValueError("item_id is required")
            if arguments.get("target_persona_id") is not None:
                self._validate_optional_string(arguments, "target_persona_id")
                if not str(arguments.get("target_persona_id") or "").strip():
                    raise ValueError("target_persona_id cannot be empty")
            title = arguments.get("title")
            if title is not None:
                if not isinstance(title, str):
                    raise ValueError("title must be a string")
                normalized_title = title.strip()
                if not normalized_title:
                    raise ValueError("title cannot be empty")
                if len(normalized_title) > 200:
                    raise ValueError("title must be <= 200 chars")
            return
        if tool_name == "persona_visuals.enqueue_generation":
            self._validate_optional_string(arguments, "persona_id")
            pack_id = str(arguments.get("pack_id") or "").strip()
            prompt = str(arguments.get("prompt") or "").strip()
            if not pack_id:
                raise ValueError("pack_id is required")
            if not prompt:
                raise ValueError("prompt is required")
            if len(prompt) > 5000:
                raise ValueError("prompt must be <= 5000 chars")
            target_state = str(arguments.get("target_state") or "").strip()
            if target_state and target_state not in VISUAL_STATE_IDS:
                raise ValueError(f"Unknown visual state: {target_state}")
            self._validate_optional_string(arguments, "backend")
            return
        raise ValueError(f"Unknown tool: {tool_name}")

    def _capabilities_sync(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        user_id = self._resolve_user_id(context)
        persona_id = self._resolve_persona_id(args, context)
        db = self._open_db(context)
        try:
            self._require_persona(db, persona_id=persona_id, user_id=user_id)
            active_pack = db.get_active_persona_visual_pack(persona_id=persona_id, user_id=user_id)
            packs = db.list_persona_visual_packs(persona_id=persona_id, user_id=user_id)
            draft_packs = [pack for pack in packs if str(pack.get("status") or "") == "draft"]
            return {
                "persona_id": persona_id,
                "states": sorted(self._runtime_visual_state_ids(active_pack)),
                "active_pack": self._pack_summary(db, active_pack, persona_id=persona_id, user_id=user_id)
                if active_pack
                else None,
                "draft_packs": [
                    self._pack_summary(db, pack, persona_id=persona_id, user_id=user_id)
                    for pack in draft_packs
                ],
            }
        finally:
            self._close_db(db)

    def _library_items_sync(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        user_id = self._resolve_user_id(context)
        limit = self._bounded_int(args.get("limit"), default=100, minimum=1, maximum=500)
        offset = self._library_items_offset(args.get("offset"))
        db = self._open_db(context)
        try:
            items = db.list_persona_visual_library_items(
                user_id=user_id,
                include_deleted=False,
                limit=limit,
                offset=offset,
            )
            summaries = [self._library_item_summary(item) for item in items]
            return {
                "items": summaries,
                "count": len(summaries),
                "limit": limit,
                "offset": offset,
                "reference_backed": True,
            }
        finally:
            self._close_db(db)

    def _trigger_state_sync(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        user_id = self._resolve_user_id(context)
        persona_id = self._resolve_persona_id(args, context)
        state = self._normalize_visual_state_id(args.get("state"))
        duration_ms = self._clamp_duration(args.get("duration_ms", 1500))
        session_id = str(getattr(context, "session_id", "") or "").strip() or None
        db = self._open_db(context)
        try:
            self._require_persona(db, persona_id=persona_id, user_id=user_id)
            active_pack = db.get_active_persona_visual_pack(persona_id=persona_id, user_id=user_id)
            if not self._visual_state_available_for_runtime(active_pack, state):
                raise ValueError(
                    f"Custom visual state {state} is not available in the active Persona Visual pack"
                )
        finally:
            self._close_db(db)
        return {
            "type": "visual_state_override",
            "persona_id": persona_id,
            "session_id": session_id,
            "state": state,
            "duration_ms": duration_ms,
            "reason": self._safe_trigger_reason(args.get("reason")),
        }

    def _create_draft_pack_sync(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        user_id = self._resolve_user_id(context)
        persona_id = self._resolve_persona_id(args, context)
        db = self._open_db(context)
        try:
            self._require_persona(db, persona_id=persona_id, user_id=user_id)
            parent_pack_id = str(args.get("parent_pack_id") or "").strip() or None
            pack = db.create_persona_visual_pack(
                persona_id=persona_id,
                user_id=user_id,
                title=str(args.get("title") or "").strip(),
                manifest=args.get("manifest") if isinstance(args.get("manifest"), dict) else {},
                status="draft",
                parent_pack_id=parent_pack_id,
            )
            return {
                "persona_id": persona_id,
                "pack": self._pack_summary(db, pack, persona_id=persona_id, user_id=user_id),
            }
        finally:
            self._close_db(db)

    def _update_manifest_sync(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        user_id = self._resolve_user_id(context)
        persona_id = self._resolve_persona_id(args, context)
        pack_id = str(args.get("pack_id") or "").strip()
        db = self._open_db(context)
        try:
            self._require_persona(db, persona_id=persona_id, user_id=user_id)
            pack = db.get_persona_visual_pack(
                pack_id=pack_id,
                persona_id=persona_id,
                user_id=user_id,
            )
            if pack is None:
                raise ValueError(f"Persona visual pack not found: {pack_id}")
            if str(pack.get("status") or "") != "draft":
                raise PermissionError(
                    "Only draft visual packs can be updated by persona_visuals.update_manifest"
                )
            updated = db.update_persona_visual_pack_manifest(
                pack_id=pack_id,
                persona_id=persona_id,
                user_id=user_id,
                manifest=args["manifest"],
                expected_version=args.get("expected_version"),
            )
            if updated is None:
                raise ValueError(f"Persona visual pack not found: {pack_id}")
            return {
                "persona_id": persona_id,
                "pack": self._pack_summary(db, updated, persona_id=persona_id, user_id=user_id),
            }
        finally:
            self._close_db(db)

    def _use_library_item_sync(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        user_id = self._resolve_user_id(context)
        target_persona_id = self._resolve_target_persona_id(args, context)
        item_id = str(args.get("item_id") or "").strip()
        title = str(args.get("title") or "").strip() or None
        db = self._open_db(context)
        try:
            service = PersonaVisualLibraryService(db)
            duplicated = service.use_item_for_persona(
                user_id=user_id,
                item_id=item_id,
                target_persona_id=target_persona_id,
                title=title,
            )
            return {
                "library_item_id": item_id,
                "persona_id": target_persona_id,
                "pack": self._pack_summary(
                    db,
                    duplicated,
                    persona_id=target_persona_id,
                    user_id=user_id,
                ),
                "review_required": True,
                "activated": False,
            }
        except PersonaVisualLibraryServiceError as exc:
            raise ValueError(str(exc)) from exc
        finally:
            self._close_db(db)

    def _enqueue_generation_sync(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        user_id = self._resolve_user_id(context)
        persona_id = self._resolve_persona_id(args, context)
        pack_id = str(args.get("pack_id") or "").strip()
        target_state = str(args.get("target_state") or "").strip() or None
        if target_state and target_state not in VISUAL_STATE_IDS:
            raise ValueError(f"Unknown visual state: {target_state}")
        db = self._open_db(context)
        try:
            self._require_persona(db, persona_id=persona_id, user_id=user_id)
            pack = db.get_persona_visual_pack(
                pack_id=pack_id,
                persona_id=persona_id,
                user_id=user_id,
            )
            if pack is None:
                raise ValueError(f"Persona visual pack not found: {pack_id}")
            if str(pack.get("status") or "") != "draft":
                raise PermissionError("Only draft visual packs can enqueue persona visual generation")
        finally:
            self._close_db(db)

        jobs_manager = self._get_jobs_manager()
        job = create_generate_candidate_job(
            jobs_manager,
            user_id=user_id,
            persona_id=persona_id,
            pack_id=pack_id,
            prompt=str(args.get("prompt") or "").strip(),
            target_state=target_state,
            backend=str(args.get("backend") or "").strip() or None,
        )
        return {
            "persona_id": persona_id,
            "pack_id": pack_id,
            "job_id": str(job.get("id") or ""),
            "status": None if job.get("status") is None else str(job.get("status")),
            "review_required": True,
        }

    def _open_db(self, context: Any | None) -> CharactersRAGDB:
        if context is None or not getattr(context, "db_paths", None):
            raise ValueError("Missing user context for persona visuals")
        chacha_path = context.db_paths.get("chacha")
        if not chacha_path:
            raise ValueError("ChaChaNotes DB path not available in context")
        return CharactersRAGDB(
            db_path=chacha_path,
            client_id=f"mcp_persona_visuals_{self.config.name}",
        )

    @staticmethod
    def _close_db(db: CharactersRAGDB) -> None:
        try:
            db.close_all_connections()
        except _PERSONA_VISUALS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Failed to close ChaChaNotes DB connections after persona visuals call: {}", exc)

    @staticmethod
    def _resolve_user_id(context: Any | None) -> str:
        if context is None:
            raise ValueError("Missing user context for persona visuals")
        user_id = str(getattr(context, "user_id", "") or "").strip()
        metadata = getattr(context, "metadata", None)
        if not user_id and isinstance(metadata, dict):
            user_id = str(metadata.get("user_id") or "").strip()
        if not user_id:
            raise ValueError("Missing user context for persona visuals")
        return user_id

    def _resolve_persona_id(self, args: dict[str, Any], context: Any | None) -> str:
        explicit = str(args.get("persona_id") or "").strip()
        if explicit:
            return explicit

        scopes: list[Any] = []
        if context is not None:
            context_scope = getattr(context, "persona_scope", None)
            if context_scope:
                scopes.append(context_scope)
            metadata = getattr(context, "metadata", None)
            if isinstance(metadata, dict) and metadata.get("persona_scope"):
                scopes.append(metadata.get("persona_scope"))

        for scope in scopes:
            persona_id = self._persona_id_from_scope(scope)
            if persona_id:
                return persona_id
        raise ValueError("Missing persona context for persona visuals")

    @staticmethod
    def _persona_id_from_scope(scope: Any) -> str | None:
        if not isinstance(scope, dict):
            return None
        direct = str(scope.get("persona_id") or "").strip()
        if direct:
            return direct
        for key in ("persona_ids", "personas"):
            value = scope.get(key)
            if isinstance(value, list) and len(value) == 1:
                candidate = str(value[0] or "").strip()
                if candidate:
                    return candidate
        explicit_ids = scope.get("explicit_ids")
        materialized = scope.get("materialized_scope")
        if not isinstance(explicit_ids, dict) and isinstance(materialized, dict):
            explicit_ids = materialized.get("explicit_ids")
        if isinstance(explicit_ids, dict):
            value = explicit_ids.get("persona_id") or explicit_ids.get("persona_ids")
            if isinstance(value, list) and len(value) == 1:
                candidate = str(value[0] or "").strip()
                if candidate:
                    return candidate
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _resolve_target_persona_id(self, args: dict[str, Any], context: Any | None) -> str:
        if args.get("target_persona_id") is not None:
            explicit = str(args.get("target_persona_id") or "").strip()
            if not explicit:
                raise ValueError("target_persona_id cannot be empty")
            return explicit
        try:
            return self._resolve_persona_id({}, context)
        except ValueError as exc:
            raise ValueError("Missing target persona context for persona visuals library reuse") from exc

    @staticmethod
    def _require_persona(db: CharactersRAGDB, *, persona_id: str, user_id: str) -> dict[str, Any]:
        profile = db.get_persona_profile(persona_id, user_id=user_id, include_deleted=False)
        if profile is None:
            raise ValueError(f"Persona not found for user: {persona_id}")
        return profile

    def _pack_summary(
        self,
        db: CharactersRAGDB,
        pack: dict[str, Any] | None,
        *,
        persona_id: str,
        user_id: str,
    ) -> dict[str, Any] | None:
        if pack is None:
            return None
        assets = pack.get("assets")
        if not isinstance(assets, list):
            assets = db.list_persona_visual_assets(
                pack_id=str(pack.get("id") or ""),
                persona_id=persona_id,
                user_id=user_id,
            )
        return {
            "id": str(pack.get("id") or ""),
            "persona_id": str(pack.get("persona_id") or persona_id),
            "title": str(pack.get("title") or ""),
            "renderer_type": str(pack.get("renderer_type") or "sprite_frames"),
            "status": str(pack.get("status") or ""),
            "manifest_version": int(pack.get("manifest_version") or 1),
            "revision_number": int(pack.get("revision_number") or 1),
            "version": int(pack.get("version") or 1),
            "provenance": str(pack.get("provenance") or ""),
            "parent_pack_id": pack.get("parent_pack_id"),
            "assets_count": len(assets),
        }

    @staticmethod
    def _library_item_summary(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(item.get("id") or ""),
            "title": str(item.get("title") or ""),
            "notes": item.get("notes"),
            "tags": [str(tag) for tag in list(item.get("tags") or [])],
            "source_persona_id": item.get("source_persona_id"),
            "source_pack_id": item.get("source_pack_id"),
            "source_persona_name": item.get("source_persona_name"),
            "source_pack_title": item.get("source_pack_title"),
            "source_pack_version": item.get("source_pack_version"),
            "source_current_version": item.get("source_current_version"),
            "source_available": bool(item.get("source_available")),
            "source_changed": bool(item.get("source_changed")),
            "version": int(item.get("version") or 1),
        }

    def _get_jobs_manager(self) -> Any:
        settings = self.config.settings or {}
        manager = settings.get("jobs_manager")
        if manager is not None:
            return manager
        factory = settings.get("jobs_manager_factory")
        if callable(factory):
            return factory()
        db_path = settings.get("jobs_db_path")
        return JobManager(Path(db_path)) if db_path else JobManager()

    @staticmethod
    def _clamp_duration(value: Any) -> int:
        try:
            duration_ms = int(value)
        except (TypeError, ValueError):
            duration_ms = 1500
        return max(MIN_TRIGGER_DURATION_MS, min(MAX_TRIGGER_DURATION_MS, duration_ms))

    @staticmethod
    def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(maximum, parsed))

    @staticmethod
    def _library_items_offset(value: Any) -> int:
        if value is None:
            return 0
        try:
            offset = int(value)
        except (TypeError, ValueError):
            raise ValueError("offset must be an integer between 0 and 1000000") from None
        if offset < 0 or offset > _MAX_LIBRARY_ITEMS_OFFSET:
            raise ValueError("offset must be between 0 and 1000000")
        return offset

    @staticmethod
    def _validate_optional_string(arguments: dict[str, Any], key: str) -> None:
        if arguments.get(key) is not None and not isinstance(arguments.get(key), str):
            raise ValueError(f"{key} must be a string")

    @staticmethod
    def _normalize_visual_state_id(value: Any) -> str:
        state = str(value or "").strip()
        if not state:
            raise ValueError("state is required")
        if state in VISUAL_STATE_IDS:
            return state
        custom_error = custom_visual_state_id_error(state)
        if custom_error:
            raise ValueError(custom_error)
        return state

    @staticmethod
    def _visual_state_available_for_runtime(
        active_pack: dict[str, Any] | None,
        state: str,
    ) -> bool:
        return state in PersonaVisualsModule._runtime_visual_state_ids(active_pack)

    @staticmethod
    def _runtime_visual_state_ids(active_pack: dict[str, Any] | None) -> set[str]:
        state_ids = set(VISUAL_STATE_IDS)
        manifest = active_pack.get("manifest") if isinstance(active_pack, dict) else None
        if not isinstance(manifest, dict):
            return state_ids
        states = manifest.get("states")
        state_catalog = manifest.get("state_catalog")
        if not isinstance(states, dict) or not isinstance(state_catalog, dict):
            return state_ids
        state_ids.update(
            state
            for state in states
            if state in state_catalog and custom_visual_state_id_error(state) is None
        )
        return state_ids

    @staticmethod
    def _safe_trigger_reason(value: Any) -> str:
        if value is not None and not isinstance(value, str):
            return "mcp_runtime"
        reason = " ".join(str(value or "mcp_runtime").split())
        if not reason:
            return "mcp_runtime"
        return reason[:_MAX_TRIGGER_REASON_LENGTH]


__all__ = ["PersonaVisualsModule"]
