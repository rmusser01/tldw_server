"""MCP prompt catalog adapter for library and allowlisted config prompts."""

from __future__ import annotations

import base64
import json
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from ....DB_Management.Prompts_DB import DatabaseError, PromptsDatabase
from ....exception_types import PromptCatalogError
from ....Prompt_Management.structured_prompts import (
    PromptBlock,
    PromptDefinition,
    PromptVariableDefinition,
    StructuredPromptAssemblyError,
    assemble_prompt_definition,
    convert_legacy_prompt_to_definition,
    extract_legacy_prompt_variables,
    normalize_legacy_prompt_template,
)
from ....Utils.prompt_loader import load_prompt
from ...persona_scope import assert_identifier_in_scope, get_explicit_scope_ids

LIBRARY_PROMPT_PREFIX = "library:"
CONFIG_PROMPT_PREFIX = "config:"
DEFAULT_PROMPT_PAGE_SIZE = 50
MAX_PROMPT_PAGE_SIZE = 100
MAX_RENDERED_PROMPT_CHARS = 100_000

_CURSOR_VERSION = 1
_CONFIG_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,100}$")
_ALLOWED_CONFIG_ROLES = {"system", "developer", "user", "assistant"}
_CATALOG_DB_EXCEPTIONS = (DatabaseError, OSError, RuntimeError, TypeError, ValueError)


@dataclass(frozen=True)
class PromptCatalogCursor:
    """Opaque pagination cursor state for prompt catalog sources."""

    library_after_name: str | None = None
    library_after_uuid: str | None = None
    library_done: bool = False
    config_index: int = 0


@dataclass(frozen=True)
class PromptCatalogListResult:
    """Prompt catalog list page with optional cursor and sanitized warnings."""

    prompts: list[dict[str, Any]]
    next_cursor: PromptCatalogCursor | None = None
    warnings: list[dict[str, Any]] = field(default_factory=list)


def encode_prompt_cursor(cursor: PromptCatalogCursor | None) -> str | None:
    """Encode prompt catalog pagination state for MCP clients.

    Args:
        cursor: Cursor state returned by a catalog source, or ``None`` when
            there is no next page.

    Returns:
        An unpadded URL-safe base64 JSON cursor string, or ``None`` when no
        cursor was provided.
    """

    if cursor is None:
        return None

    payload: dict[str, Any] = {
        "v": _CURSOR_VERSION,
        "library_done": cursor.library_done,
        "config_index": cursor.config_index,
    }
    if cursor.library_after_name is not None:
        payload["library_after_name"] = cursor.library_after_name
    if cursor.library_after_uuid is not None:
        payload["library_after_uuid"] = cursor.library_after_uuid
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_prompt_cursor(raw_cursor: str | None) -> PromptCatalogCursor:
    """Decode and validate an MCP prompt catalog cursor.

    Args:
        raw_cursor: Opaque cursor string supplied by the MCP client. ``None``
            and the empty string both represent the first page.

    Returns:
        Parsed prompt catalog cursor state.

    Raises:
        PromptCatalogError: If the cursor is malformed, unsupported, or
            contains inconsistent pagination state.
    """

    if raw_cursor is None or raw_cursor == "":
        return PromptCatalogCursor()
    if not isinstance(raw_cursor, str):
        raise PromptCatalogError("invalid_cursor", "Invalid prompt cursor.")

    try:
        padding = "=" * (-len(raw_cursor) % 4)
        raw = base64.urlsafe_b64decode((raw_cursor + padding).encode("ascii"))
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeEncodeError, UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise PromptCatalogError("invalid_cursor", "Invalid prompt cursor.") from exc

    version = payload.get("v") if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or type(version) is not int or version != _CURSOR_VERSION:
        raise PromptCatalogError("invalid_cursor", "Invalid prompt cursor.")

    library_after_name = payload.get("library_after_name")
    library_after_uuid = payload.get("library_after_uuid")
    has_library_name = "library_after_name" in payload
    has_library_uuid = "library_after_uuid" in payload
    if has_library_name != has_library_uuid:
        raise PromptCatalogError("invalid_cursor", "Invalid prompt cursor.")
    if has_library_name:
        if (
            not isinstance(library_after_name, str)
            or not library_after_name
            or not isinstance(library_after_uuid, str)
            or not library_after_uuid
        ):
            raise PromptCatalogError("invalid_cursor", "Invalid prompt cursor.")
        try:
            uuid.UUID(library_after_uuid)
        except ValueError as exc:
            raise PromptCatalogError("invalid_cursor", "Invalid prompt cursor.") from exc

    library_done = payload.get("library_done", False)
    if not isinstance(library_done, bool):
        raise PromptCatalogError("invalid_cursor", "Invalid prompt cursor.")
    if library_done and has_library_name:
        raise PromptCatalogError("invalid_cursor", "Invalid prompt cursor.")

    config_index = payload.get("config_index", 0)
    if isinstance(config_index, bool) or not isinstance(config_index, int) or config_index < 0:
        raise PromptCatalogError("invalid_cursor", "Invalid prompt cursor.")
    if not library_done and config_index != 0:
        raise PromptCatalogError("invalid_cursor", "Invalid prompt cursor.")

    return PromptCatalogCursor(
        library_after_name=library_after_name,
        library_after_uuid=library_after_uuid,
        library_done=library_done,
        config_index=config_index,
    )


def clamp_prompt_page_size(raw_value: Any) -> int:
    """Clamp a requested prompt page size to catalog defaults and limits."""

    if type(raw_value) is not int:
        return DEFAULT_PROMPT_PAGE_SIZE
    if raw_value < 1:
        return 1
    return min(raw_value, MAX_PROMPT_PAGE_SIZE)


class MCPPromptFormatter:
    """Convert tldw prompt records and config entries into MCP prompt shapes."""

    def __init__(self, max_rendered_chars: int = MAX_RENDERED_PROMPT_CHARS) -> None:
        self.max_rendered_chars = max(1, int(max_rendered_chars))

    def validate_arguments(self, arguments: Any | None) -> dict[str, str]:
        """Validate MCP prompt arguments as a mapping of string names to strings."""

        if arguments is None:
            return {}
        if not isinstance(arguments, Mapping):
            raise PromptCatalogError("invalid_arguments", "Prompt arguments must be an object.")
        validated: dict[str, str] = {}
        for key, value in arguments.items():
            if not isinstance(key, str) or not key:
                raise PromptCatalogError("invalid_argument_name", "Prompt argument names must be valid strings.")
            if not isinstance(value, str):
                raise PromptCatalogError("invalid_argument_type", "Prompt argument values must be strings.")
            validated[key] = value
        return validated

    def library_prompt_definition(self, row: Mapping[str, Any]) -> dict[str, Any]:
        """Build an MCP list definition for a Prompt Library row."""

        definition = self._definition_from_library_row(row)
        prompt_uuid = str(row.get("uuid") or "")
        return {
            "name": f"{LIBRARY_PROMPT_PREFIX}{prompt_uuid}",
            "title": str(row.get("name") or prompt_uuid),
            "description": row.get("details") or None,
            "arguments": self._mcp_arguments(definition.variables),
            "_meta": {
                "tldw": {
                    "source": "library",
                    "prompt_id": row.get("id"),
                    "prompt_uuid": prompt_uuid,
                    "version": row.get("version"),
                    "tags": row.get("keywords") or [],
                }
            },
        }

    def render_library_prompt(self, row: Mapping[str, Any], arguments: Any | None) -> dict[str, Any]:
        """Render a Prompt Library row as MCP prompt messages."""

        validated_arguments = self.validate_arguments(arguments)
        definition = self._definition_from_library_row(row)
        messages = self._assemble_to_mcp_messages(definition, validated_arguments)
        return {
            "description": str(row.get("name") or row.get("uuid") or ""),
            "messages": messages,
            "_meta": self.library_prompt_definition(row)["_meta"],
        }

    def config_prompt_definition(
        self,
        entry: Mapping[str, Any],
        parts: list[Mapping[str, str]],
    ) -> dict[str, Any]:
        """Build an MCP list definition for an allowlisted config prompt."""

        definition = self._definition_from_config_parts(entry, parts)
        prompt_id = str(entry.get("id") or "")
        return {
            "name": f"{CONFIG_PROMPT_PREFIX}{prompt_id}",
            "title": str(entry.get("title") or prompt_id),
            "description": entry.get("description") or None,
            "arguments": self._mcp_arguments(definition.variables),
            "_meta": {
                "tldw": {
                    "source": "config",
                    "prompt_id": prompt_id,
                    "part_count": len(parts),
                }
            },
        }

    def render_config_prompt(
        self,
        entry: Mapping[str, Any],
        parts: list[Mapping[str, str]],
        arguments: Any | None,
    ) -> dict[str, Any]:
        """Render an allowlisted config prompt as MCP prompt messages."""

        validated_arguments = self.validate_arguments(arguments)
        definition = self._definition_from_config_parts(entry, parts)
        messages = self._assemble_to_mcp_messages(definition, validated_arguments)
        return {
            "description": str(entry.get("title") or entry.get("id") or ""),
            "messages": messages,
            "_meta": self.config_prompt_definition(entry, parts)["_meta"],
        }

    def _definition_from_library_row(self, row: Mapping[str, Any]) -> PromptDefinition:
        raw_definition = row.get("prompt_definition")
        if raw_definition:
            try:
                return PromptDefinition.model_validate(raw_definition)
            except ValueError as exc:
                raise PromptCatalogError(
                    "invalid_prompt_definition",
                    "Prompt definition is invalid.",
                    internal=True,
                ) from exc
        return convert_legacy_prompt_to_definition(
            system_prompt=str(row.get("system_prompt") or ""),
            user_prompt=str(row.get("user_prompt") or ""),
        )

    def _definition_from_config_parts(
        self,
        entry: Mapping[str, Any],
        parts: list[Mapping[str, str]],
    ) -> PromptDefinition:
        variables = extract_legacy_prompt_variables(*(part.get("content") for part in parts))
        blocks = [
            PromptBlock(
                id=f"config_part_{index}",
                name=str(part.get("name") or f"Part {index + 1}"),
                role=part["role"],
                content=normalize_legacy_prompt_template(part.get("content")),
                enabled=True,
                order=(index + 1) * 10,
                is_template=bool(extract_legacy_prompt_variables(part.get("content"))),
            )
            for index, part in enumerate(parts)
        ]
        variable_definitions = [
            PromptVariableDefinition(
                name=variable_name,
                label=variable_name.replace("_", " ").title(),
                required=True,
                input_type="textarea",
            )
            for variable_name in variables
        ]
        return PromptDefinition(variables=variable_definitions, blocks=blocks)

    def _assemble_to_mcp_messages(
        self,
        definition: PromptDefinition,
        arguments: Mapping[str, str],
    ) -> list[dict[str, Any]]:
        try:
            assembled = assemble_prompt_definition(definition, arguments)
        except StructuredPromptAssemblyError as exc:
            raise PromptCatalogError("invalid_arguments", "Prompt arguments are invalid.") from exc
        mcp_messages = self._to_mcp_messages(assembled.messages)
        rendered_size = sum(len(message["content"]["text"]) for message in mcp_messages)
        if rendered_size > self.max_rendered_chars:
            raise PromptCatalogError("rendered_prompt_too_large", "Rendered prompt is too large.")
        return mcp_messages

    def _to_mcp_messages(self, messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        mcp_messages: list[dict[str, Any]] = []
        pending_instructions: list[str] = []

        for message in messages:
            role = str(message.get("role") or "")
            content = str(message.get("content") or "")
            if role in {"system", "developer"}:
                if content:
                    pending_instructions.append(content)
                continue
            if role == "user":
                if pending_instructions:
                    content = (
                        "System instructions:\n"
                        + "\n\n".join(pending_instructions)
                        + "\n\nUser prompt:\n"
                        + content
                    )
                    pending_instructions = []
                mcp_messages.append(self._mcp_text_message("user", content))
                continue
            if role == "assistant":
                mcp_messages.append(self._mcp_text_message("assistant", content))

        if pending_instructions:
            mcp_messages.append(
                self._mcp_text_message(
                    "user",
                    "System instructions:\n" + "\n\n".join(pending_instructions),
                )
            )
        return mcp_messages

    @staticmethod
    def _mcp_text_message(role: str, text: str) -> dict[str, Any]:
        return {"role": role, "content": {"type": "text", "text": text}}

    @staticmethod
    def _mcp_arguments(variables: list[PromptVariableDefinition]) -> list[dict[str, Any]]:
        return [
            {
                "name": variable.name,
                "title": variable.label or variable.name.replace("_", " ").title(),
                "description": variable.description,
                "required": bool(variable.required),
            }
            for variable in variables
        ]


class UserPromptCatalogSource:
    """Catalog source backed by the per-user Prompt Library database."""

    def __init__(self, formatter: MCPPromptFormatter) -> None:
        self.formatter = formatter

    def list_prompts(
        self,
        context: Any,
        cursor: PromptCatalogCursor,
        limit: int,
    ) -> PromptCatalogListResult:
        """List visible non-deleted Prompt Library prompts."""

        if cursor.library_done:
            return PromptCatalogListResult(prompts=[])

        db: PromptsDatabase | None = None
        try:
            db = self._open_db(context)
            query, params = self._list_query(context, cursor, limit + 1)
            rows = [db._deserialize_prompt_record(dict(row)) for row in db.execute_query(query, params).fetchall()]
        except PromptCatalogError as exc:
            if not exc.internal:
                raise
            return self._library_unavailable_result(exc)
        except _CATALOG_DB_EXCEPTIONS as exc:
            return self._library_unavailable_result(exc)
        finally:
            if db is not None:
                self._close_db(db)

        page_rows = rows[:limit]
        prompts: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        for row in page_rows:
            try:
                prompts.append(self.formatter.library_prompt_definition(row))
            except PromptCatalogError as exc:
                if not exc.internal:
                    raise
                warnings.append(self._prompt_unavailable_warning(row))
            except (TypeError, ValueError) as exc:
                logger.debug("Prompt Library row unavailable during list: {}", exc.__class__.__name__)
                warnings.append(self._prompt_unavailable_warning(row))

        next_cursor = None
        if len(rows) > limit and page_rows:
            last_row = page_rows[-1]
            next_cursor = PromptCatalogCursor(
                library_after_name=str(last_row.get("name") or ""),
                library_after_uuid=str(last_row.get("uuid") or ""),
            )
        return PromptCatalogListResult(prompts=prompts, next_cursor=next_cursor, warnings=warnings)

    @staticmethod
    def _library_unavailable_result(exc: Exception) -> PromptCatalogListResult:
        logger.debug("Prompt Library catalog unavailable during list: {}", exc.__class__.__name__)
        return PromptCatalogListResult(
            prompts=[],
            warnings=[{"source": "library", "code": "prompt_db_unavailable"}],
        )

    @staticmethod
    def _prompt_unavailable_warning(row: Mapping[str, Any]) -> dict[str, Any]:
        warning: dict[str, Any] = {"source": "library", "code": "prompt_unavailable"}
        prompt_uuid = row.get("uuid")
        if isinstance(prompt_uuid, str) and prompt_uuid:
            warning["prompt_uuid"] = prompt_uuid
        return warning

    def get_prompt(self, context: Any, name: str, arguments: Any | None) -> dict[str, Any]:
        """Fetch and render a single Prompt Library prompt by MCP name."""

        prompt_uuid = self._parse_library_name(name)
        db: PromptsDatabase | None = None
        try:
            db = self._open_db(context)
            row = db.get_prompt_by_uuid(prompt_uuid, include_deleted=False)
            if not row:
                raise PromptCatalogError("prompt_not_found", "Prompt not found.")
            try:
                assert_identifier_in_scope(context, "prompt_id", row.get("id"), label="Prompt")
            except PermissionError as exc:
                raise PromptCatalogError("permission_denied", "Prompt access denied.") from exc
            return self.formatter.render_library_prompt(row, arguments)
        except PromptCatalogError as exc:
            if exc.internal:
                raise self._prompt_db_unavailable_error() from exc
            raise
        except _CATALOG_DB_EXCEPTIONS as exc:
            raise self._prompt_db_unavailable_error() from exc
        finally:
            if db is not None:
                self._close_db(db)

    @staticmethod
    def _prompt_db_unavailable_error() -> PromptCatalogError:
        return PromptCatalogError(
            "prompt_db_unavailable",
            "Prompt library is unavailable.",
            internal=True,
        )

    def _open_db(self, context: Any) -> PromptsDatabase:
        db_paths = getattr(context, "db_paths", None)
        if not isinstance(db_paths, Mapping) or not db_paths.get("prompts"):
            raise PromptCatalogError("prompt_db_unavailable", "Prompt database is unavailable.", internal=True)
        return PromptsDatabase(db_path=str(db_paths["prompts"]), client_id="mcp_prompt_catalog")

    def _list_query(
        self,
        context: Any,
        cursor: PromptCatalogCursor,
        limit: int,
    ) -> tuple[str, tuple[Any, ...]]:
        where_clauses = ["deleted = 0"]
        params: list[Any] = []

        scoped_ids = get_explicit_scope_ids(context, "prompt_id")
        if scoped_ids is not None:
            numeric_ids = sorted({int(value) for value in scoped_ids if str(value).isdigit()})
            if not numeric_ids:
                return "SELECT * FROM Prompts WHERE 1 = 0 LIMIT ?", (limit,)
            placeholders = ",".join("?" for _ in numeric_ids)
            where_clauses.append(f"id IN ({placeholders})")
            params.extend(numeric_ids)

        if cursor.library_after_name is not None and cursor.library_after_uuid is not None:
            where_clauses.append("(name COLLATE NOCASE > ? OR (name COLLATE NOCASE = ? AND uuid > ?))")
            params.extend(
                [
                    cursor.library_after_name,
                    cursor.library_after_name,
                    cursor.library_after_uuid,
                ]
            )

        params.append(limit)
        # The only dynamic SQL fragment above is an id IN placeholder list built
        # from "?" tokens; all user-derived values remain bound parameters.
        query = (
            "SELECT * FROM Prompts WHERE "  # nosec B608
            + " AND ".join(where_clauses)
            + " ORDER BY name COLLATE NOCASE ASC, uuid ASC LIMIT ?"
        )
        return query, tuple(params)

    @staticmethod
    def _parse_library_name(name: str) -> str:
        if not isinstance(name, str) or not name.startswith(LIBRARY_PROMPT_PREFIX):
            raise PromptCatalogError("invalid_prompt_name", "Invalid library prompt name.")
        prompt_uuid = name[len(LIBRARY_PROMPT_PREFIX) :]
        try:
            return str(uuid.UUID(prompt_uuid))
        except ValueError as exc:
            raise PromptCatalogError("invalid_prompt_name", "Invalid library prompt name.") from exc

    @staticmethod
    def _close_db(db: PromptsDatabase) -> None:
        try:
            db.close_connection()
        except _CATALOG_DB_EXCEPTIONS as exc:
            logger.debug("Failed to close Prompt Library catalog DB: {}", exc.__class__.__name__)


class ConfigPromptCatalogSource:
    """Catalog source backed by explicit allowlisted config prompt entries."""

    def __init__(self, formatter: MCPPromptFormatter, config: Mapping[str, Any] | None = None) -> None:
        self.formatter = formatter
        self.config = dict(config or {})
        self.enabled = self.config.get("enabled", True) is not False
        raw_entries = self.config.get("entries") or []
        self.entries = [entry for entry in raw_entries if isinstance(entry, Mapping)]

    def list_prompts(self, cursor: PromptCatalogCursor, limit: int) -> PromptCatalogListResult:
        """List available allowlisted config prompts, omitting missing entries."""

        if not self.enabled:
            return PromptCatalogListResult(prompts=[])

        prompts: list[dict[str, Any]] = []
        warnings: list[dict[str, Any]] = []
        index = max(0, cursor.config_index)
        while index < len(self.entries) and len(prompts) < limit:
            entry = self.entries[index]
            entry_id = self._entry_id(entry)
            try:
                parts = self._load_entry_parts(entry)
                prompts.append(self.formatter.config_prompt_definition(entry, parts))
            except PromptCatalogError:
                if entry_id:
                    warnings.append(
                        {"source": "config", "code": "config_prompt_unavailable", "id": entry_id}
                    )
            index += 1

        next_cursor = None
        if index < len(self.entries):
            next_cursor = PromptCatalogCursor(library_done=True, config_index=index)
        return PromptCatalogListResult(prompts=prompts, next_cursor=next_cursor, warnings=warnings)

    def has_entries_after(self, config_index: int) -> bool:
        """Return whether the config source has entries at or after an index."""

        return self.enabled and 0 <= config_index < len(self.entries)

    def get_prompt(self, name: str, arguments: Any | None) -> dict[str, Any]:
        """Fetch and render an allowlisted config prompt by MCP name."""

        if not self.enabled:
            raise PromptCatalogError("prompt_not_found", "Prompt not found.")
        prompt_id = self._parse_config_name(name)
        for entry in self.entries:
            if self._entry_id(entry) != prompt_id:
                continue
            parts = self._load_entry_parts(entry)
            return self.formatter.render_config_prompt(entry, parts, arguments)
        raise PromptCatalogError("prompt_not_found", "Prompt not found.")

    def _load_entry_parts(self, entry: Mapping[str, Any]) -> list[Mapping[str, str]]:
        entry_id = self._entry_id(entry)
        if not entry_id:
            raise PromptCatalogError("invalid_config_prompt", "Config prompt entry is invalid.", internal=True)

        raw_messages = entry.get("messages")
        if isinstance(raw_messages, list):
            parts = [self._load_part(part) for part in raw_messages if isinstance(part, Mapping)]
        else:
            parts = [self._load_part({**entry, "role": "user"})]
        if not parts:
            raise PromptCatalogError("config_prompt_unavailable", "Config prompt is unavailable.", internal=True)
        return parts

    def _load_part(self, part: Mapping[str, Any]) -> Mapping[str, str]:
        role = str(part.get("role") or "user")
        module = part.get("module")
        key = part.get("key")
        if role not in _ALLOWED_CONFIG_ROLES or not isinstance(module, str) or not isinstance(key, str):
            raise PromptCatalogError("invalid_config_prompt", "Config prompt entry is invalid.", internal=True)
        content = load_prompt(module, key)
        if not isinstance(content, str) or not content:
            raise PromptCatalogError("config_prompt_unavailable", "Config prompt is unavailable.", internal=True)
        return {
            "role": role,
            "module": module,
            "key": key,
            "name": str(part.get("title") or key),
            "content": content,
        }

    @staticmethod
    def _parse_config_name(name: str) -> str:
        if not isinstance(name, str) or not name.startswith(CONFIG_PROMPT_PREFIX):
            raise PromptCatalogError("invalid_prompt_name", "Invalid config prompt name.")
        prompt_id = name[len(CONFIG_PROMPT_PREFIX) :]
        if not _CONFIG_ID_RE.fullmatch(prompt_id):
            raise PromptCatalogError("invalid_prompt_name", "Invalid config prompt name.")
        return prompt_id

    @staticmethod
    def _entry_id(entry: Mapping[str, Any]) -> str | None:
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not _CONFIG_ID_RE.fullmatch(entry_id):
            return None
        return entry_id
