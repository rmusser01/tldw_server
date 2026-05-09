from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Persona.visual_jobs import create_generate_candidate_job
from tldw_Server_API.app.core.Persona.visuals import (
    MAX_TRIGGER_DURATION_MS,
    MIN_TRIGGER_DURATION_MS,
    VISUAL_STATE_IDS,
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
                name="persona_visuals.trigger_state",
                description="Emit a transient visual state override for the current persona session.",
                parameters={
                    "properties": {
                        "persona_id": {"type": "string"},
                        "state": {"type": "string", "enum": sorted(VISUAL_STATE_IDS)},
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
        if tool_name == "persona_visuals.trigger_state":
            return await asyncio.to_thread(self._trigger_state_sync, args, context)
        if tool_name == "persona_visuals.create_draft_pack":
            return await asyncio.to_thread(self._create_draft_pack_sync, args, context)
        if tool_name == "persona_visuals.update_manifest":
            return await asyncio.to_thread(self._update_manifest_sync, args, context)
        if tool_name == "persona_visuals.enqueue_generation":
            return await asyncio.to_thread(self._enqueue_generation_sync, args, context)
        raise ValueError(f"Unknown tool: {tool_name}")

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        if tool_name == "persona_visuals.capabilities":
            self._validate_optional_string(arguments, "persona_id")
            return
        if tool_name == "persona_visuals.trigger_state":
            self._validate_optional_string(arguments, "persona_id")
            state = str(arguments.get("state") or "").strip()
            if not state:
                raise ValueError("state is required")
            if state not in VISUAL_STATE_IDS:
                raise ValueError(f"Unknown visual state: {state}")
            if arguments.get("duration_ms") is not None:
                int(arguments["duration_ms"])
            self._validate_optional_string(arguments, "reason")
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
                "states": sorted(VISUAL_STATE_IDS),
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

    def _trigger_state_sync(self, args: dict[str, Any], context: Any | None) -> dict[str, Any]:
        user_id = self._resolve_user_id(context)
        persona_id = self._resolve_persona_id(args, context)
        state = str(args.get("state") or "").strip()
        if state not in VISUAL_STATE_IDS:
            raise ValueError(f"Unknown visual state: {state}")
        duration_ms = self._clamp_duration(args.get("duration_ms", 1500))
        session_id = str(getattr(context, "session_id", "") or "").strip() or None
        db = self._open_db(context)
        try:
            self._require_persona(db, persona_id=persona_id, user_id=user_id)
        finally:
            self._close_db(db)
        return {
            "type": "visual_state_override",
            "persona_id": persona_id,
            "session_id": session_id,
            "state": state,
            "duration_ms": duration_ms,
            "reason": str(args.get("reason") or "mcp_runtime").strip() or "mcp_runtime",
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
    def _validate_optional_string(arguments: dict[str, Any], key: str) -> None:
        if arguments.get(key) is not None and not isinstance(arguments.get(key), str):
            raise ValueError(f"{key} must be a string")


__all__ = ["PersonaVisualsModule"]
