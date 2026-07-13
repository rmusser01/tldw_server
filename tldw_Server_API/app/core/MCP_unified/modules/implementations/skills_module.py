"""Read-only MCP catalog and dry-render tools for user-scoped Skills."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, TypeVar

from loguru import logger

from ....Context_Integrity.resolver import ContextIntegrityBlocked
from ....DB_Management.ChaChaNotes_DB import CharactersRAGDB
from ....Skills.exceptions import SkillNotFoundError
from ....Skills.runtime_metadata import build_skill_runtime_metadata
from ....Skills.skill_executor import SkillExecutor
from ....Skills.skills_service import SKILL_NAME_PATTERN, SkillMetadata, SkillsService
from ..base import BaseModule, create_tool_definition

DEFAULT_LIST_PAGE_SIZE = 50
MAX_LIST_PAGE_SIZE = 100
MAX_QUERY_CHARS = 200
MAX_ARGUMENT_CHARS = 10_000
HARD_MAX_RENDERED_SKILL_CHARS = 100_000

T = TypeVar("T")
ServiceOperation = Callable[[SkillsService], Awaitable[T]]


class _SkillNotFound(ValueError):
    """Bounded public not-found response."""


class _RenderedSkillTooLarge(ValueError):
    """Bounded public rendered-output rejection."""


class _ContextIntegrityBlocked(PermissionError):
    """Bounded public render-time integrity rejection."""


def _clamped_integer(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    """Return a clamped integer setting, excluding booleans and coercion."""
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(minimum, min(value, maximum))


class SkillsModule(BaseModule):
    """Expose model-visible Skill metadata and side-effect-free rendering."""

    async def on_initialize(self) -> None:
        """Initialize bounded settings and one stateless dry-run executor."""
        settings = self.config.settings or {}
        self._list_page_size = _clamped_integer(
            settings.get("list_page_size"),
            default=DEFAULT_LIST_PAGE_SIZE,
            minimum=1,
            maximum=MAX_LIST_PAGE_SIZE,
        )
        self._max_rendered_skill_chars = _clamped_integer(
            settings.get("max_rendered_skill_chars"),
            default=HARD_MAX_RENDERED_SKILL_CHARS,
            minimum=1,
            maximum=HARD_MAX_RENDERED_SKILL_CHARS,
        )
        self._executor = SkillExecutor()

    async def on_shutdown(self) -> None:
        """Release no resources; all databases are request scoped."""

    async def check_health(self) -> dict[str, bool]:
        """Report availability of the request-scoped Skills dependencies."""
        return {
            "initialized": hasattr(self, "_executor"),
            "database_driver_available": CharactersRAGDB is not None,
            "skills_service_available": SkillsService is not None,
        }

    async def get_tools(self) -> list[dict[str, Any]]:
        """Return the exact read-only Skills tool catalog."""
        return [
            create_tool_definition(
                name="skills.list",
                description="List model-visible Skills metadata.",
                parameters={
                    "properties": {
                        "q": {
                            "type": "string",
                            "maxLength": MAX_QUERY_CHARS,
                        },
                        "limit": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_LIST_PAGE_SIZE,
                            "default": self._list_page_size,
                        },
                        "offset": {
                            "type": "integer",
                            "minimum": 0,
                            "default": 0,
                        },
                    },
                    "required": [],
                },
                metadata={"category": "search", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="skills.get",
                description="Get metadata for one model-visible Skill.",
                parameters={
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
            create_tool_definition(
                name="skills.render",
                description="Render one model-visible Skill without model or tool execution.",
                parameters={
                    "properties": {
                        "skill_name": {"type": "string"},
                        "arguments": {
                            "type": "string",
                            "maxLength": MAX_ARGUMENT_CHARS,
                            "default": "",
                        },
                    },
                    "required": ["skill_name"],
                },
                metadata={"category": "retrieval", "readOnlyHint": True},
            ),
        ]

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        context: Any | None = None,
    ) -> Any:
        """Validate and dispatch one Skills operation without rewriting prompt text."""
        if not isinstance(arguments, dict):
            raise TypeError("arguments must be an object")
        args = dict(arguments)
        self.validate_tool_arguments(tool_name, args)

        if tool_name == "skills.list":
            return await self._run_with_service(
                context,
                tool_name,
                lambda service: self._list_skills(service, args),
            )
        if tool_name == "skills.get":
            return await self._run_with_service(
                context,
                tool_name,
                lambda service: self._get_skill(service, args),
            )
        if tool_name == "skills.render":
            return await self._run_with_service(
                context,
                tool_name,
                lambda service: self._render_skill(service, args),
            )
        raise ValueError(f"Unknown tool: {tool_name}")

    def validate_tool_arguments(self, tool_name: str, arguments: dict[str, Any]) -> None:
        """Apply exact per-tool key, type, and size allowlists."""
        if not isinstance(arguments, dict):
            raise TypeError("arguments must be an object")

        allowed_keys = {
            "skills.list": frozenset({"q", "limit", "offset"}),
            "skills.get": frozenset({"name"}),
            "skills.render": frozenset({"skill_name", "arguments"}),
        }
        allowed = allowed_keys.get(tool_name)
        if allowed is None:
            raise ValueError(f"Unknown tool: {tool_name}")
        if set(arguments) - allowed:
            raise ValueError(f"unexpected arguments for {tool_name}")

        if tool_name == "skills.list":
            query = arguments.get("q")
            if query is not None:
                if not isinstance(query, str):
                    raise ValueError("q must be a string")
                if len(query) > MAX_QUERY_CHARS:
                    raise ValueError(f"q must be at most {MAX_QUERY_CHARS} characters")
            self._validate_integer(
                "limit",
                arguments.get("limit", self._list_page_size),
                minimum=1,
                maximum=MAX_LIST_PAGE_SIZE,
            )
            self._validate_integer("offset", arguments.get("offset", 0), minimum=0)
            return

        if tool_name == "skills.get":
            self._validate_skill_name("name", arguments.get("name"))
            return

        self._validate_skill_name("skill_name", arguments.get("skill_name"))
        prompt_arguments = arguments.get("arguments", "")
        if not isinstance(prompt_arguments, str):
            raise ValueError("arguments must be a string")
        if len(prompt_arguments) > MAX_ARGUMENT_CHARS:
            raise ValueError(f"arguments must be at most {MAX_ARGUMENT_CHARS} characters")

    @staticmethod
    def _validate_integer(
        field: str,
        value: Any,
        *,
        minimum: int,
        maximum: int | None = None,
    ) -> None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} must be an integer")
        if maximum is None and value < minimum:
            raise ValueError(f"{field} must be >= {minimum}")
        if maximum is not None and not minimum <= value <= maximum:
            raise ValueError(f"{field} must be {minimum}..{maximum}")

    @staticmethod
    def _validate_skill_name(field: str, value: Any) -> None:
        if not isinstance(value, str) or not SKILL_NAME_PATTERN.fullmatch(value.strip().lower()):
            raise ValueError(f"{field} must be a valid skill name")

    async def _list_skills(
        self,
        service: SkillsService,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        query = args.get("q")
        effective_query = query if query is None or query.strip() else None
        limit = args.get("limit", self._list_page_size)
        offset = args.get("offset", 0)
        items, total = await service.list_model_visible_skills_page(
            q=effective_query,
            limit=limit,
            offset=offset,
        )
        count = len(items)
        next_value = offset + count
        return {
            "skills": [self._format_metadata(item) for item in items],
            "count": count,
            "total": total,
            "limit": limit,
            "offset": offset,
            "next_offset": next_value if next_value < total else None,
        }

    async def _get_skill(
        self,
        service: SkillsService,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            metadata = await service.get_model_visible_skill_metadata(args["name"])
        except ContextIntegrityBlocked:
            raise _SkillNotFound("skill_not_found") from None
        return self._format_metadata(metadata)

    async def _render_skill(
        self,
        service: SkillsService,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        skill_name = args["skill_name"]
        try:
            await service.get_model_visible_skill_metadata(skill_name)
        except ContextIntegrityBlocked:
            raise _SkillNotFound("skill_not_found") from None
        skill_data = await service.get_skill(skill_name)
        if not bool(skill_data.get("user_invocable", True)) or bool(
            skill_data.get("disable_model_invocation", False)
        ):
            raise _SkillNotFound("skill_not_found")

        result = await self._executor.execute(
            skill_data,
            args.get("arguments", ""),
            context=None,
            dry_run=True,
        )
        if len(result.rendered_prompt) > self._max_rendered_skill_chars:
            raise _RenderedSkillTooLarge(
                f"rendered_skill_too_large: limit={self._max_rendered_skill_chars}"
            )

        return {
            "skill_name": result.skill_name,
            "rendered_prompt": result.rendered_prompt,
            "declared_tools": list(result.allowed_tools),
            "model_override": result.model_override,
            "execution_mode": result.execution_mode,
            "supporting_files_omitted": bool(skill_data.get("supporting_files")),
            "dry_run": True,
            "version": skill_data.get("version"),
        }

    @staticmethod
    def _format_metadata(metadata: SkillMetadata) -> dict[str, Any]:
        """Return only the approved model-facing metadata fields."""
        return {
            "name": metadata.name,
            "description": metadata.description,
            "argument_hint": metadata.argument_hint,
            "user_invocable": metadata.user_invocable,
            "disable_model_invocation": metadata.disable_model_invocation,
            "declared_tools": list(metadata.allowed_tools),
            "model": metadata.model,
            "context": metadata.context,
            "runtime": build_skill_runtime_metadata(
                context=metadata.context,
                allowed_tools=metadata.allowed_tools,
                model=metadata.model,
                disable_model_invocation=metadata.disable_model_invocation,
            ),
            "version": metadata.version,
        }

    async def _run_with_service(
        self,
        context: Any,
        operation_name: str,
        operation: ServiceOperation[T],
    ) -> T:
        user_id, chacha_path = self._trusted_user_context(context)
        db: CharactersRAGDB | None = None
        try:
            db, service = await self._construct_service(user_id, chacha_path)
            return await self._await_retained(operation(service))
        except asyncio.CancelledError:
            raise
        except _SkillNotFound:
            raise
        except _RenderedSkillTooLarge:
            raise
        except _ContextIntegrityBlocked:
            raise
        except SkillNotFoundError:
            raise _SkillNotFound("skill_not_found") from None
        except ContextIntegrityBlocked:
            raise _ContextIntegrityBlocked("context_integrity_blocked") from None
        except Exception as exc:  # noqa: BLE001 - public boundary sanitizes every unexpected failure
            self._log_failure(operation_name, user_id, exc)
            raise RuntimeError("skills_unavailable") from None
        finally:
            if db is not None:
                try:
                    await self._close_database(db)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - cleanup failures must remain bounded
                    self._log_failure(operation_name, user_id, exc)

    @staticmethod
    def _trusted_user_context(context: Any) -> tuple[int, Path]:
        if context is None:
            raise PermissionError("skills_user_context_required")
        raw_user_id = getattr(context, "user_id", None)
        if isinstance(raw_user_id, bool):
            raise PermissionError("skills_user_context_required")
        try:
            user_id = int(str(raw_user_id))
        except (TypeError, ValueError):
            raise PermissionError("skills_user_context_required") from None
        if user_id <= 0:
            raise PermissionError("skills_user_context_required")

        db_paths = getattr(context, "db_paths", None)
        if not isinstance(db_paths, dict):
            raise PermissionError("skills_user_context_required")
        raw_path = db_paths.get("chacha")
        if isinstance(raw_path, bool) or not isinstance(raw_path, (str, Path)):
            raise PermissionError("skills_user_context_required")
        if isinstance(raw_path, str) and not raw_path.strip():
            raise PermissionError("skills_user_context_required")
        return user_id, Path(raw_path)

    async def _construct_service(
        self,
        user_id: int,
        chacha_path: Path,
    ) -> tuple[CharactersRAGDB, SkillsService]:
        task = asyncio.create_task(
            asyncio.to_thread(self._construct_service_sync, user_id, chacha_path)
        )
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            constructed: tuple[CharactersRAGDB, SkillsService] | None = None
            with contextlib.suppress(asyncio.CancelledError, Exception):
                constructed = await task
            if constructed is not None:
                with contextlib.suppress(Exception):
                    await self._close_database(constructed[0])
            raise

    def _construct_service_sync(
        self,
        user_id: int,
        chacha_path: Path,
    ) -> tuple[CharactersRAGDB, SkillsService]:
        db: CharactersRAGDB | None = None
        try:
            db = CharactersRAGDB(
                db_path=chacha_path,
                client_id=f"mcp_skills_{user_id}",
            )
            service = SkillsService(user_id, Path(chacha_path).parent, db)
            return db, service
        except Exception:
            if db is not None:
                db.close_all_connections()
            raise

    @staticmethod
    async def _await_retained(awaitable: Awaitable[T]) -> T:
        """Await an operation without letting request cancellation orphan its worker."""
        task = asyncio.create_task(awaitable)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
            raise

    @staticmethod
    async def _close_database(db: CharactersRAGDB) -> None:
        task = asyncio.create_task(asyncio.to_thread(db.close_all_connections))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                await task
            raise

    @staticmethod
    def _log_failure(operation: str, user_id: int, exc: Exception) -> None:
        logger.error(
            "skills operation={} user_id={} exception={}",
            operation,
            user_id,
            type(exc).__name__,
        )
