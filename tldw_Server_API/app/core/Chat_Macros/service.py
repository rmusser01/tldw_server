"""Service facade for chat macro storage, registry sync, and settings."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from .exceptions import MacroNotFoundError, MacroStorageError, MacroValidationError
from .models import MacroDefinition
from .output_profiles import MacroOutputProfile, merge_output_profile, normalize_output_profile
from .parser import load_macro_definition
from .repository import ChatMacroRepository
from .settings import normalize_settings
from .storage import MACRO_NAME_RE, ChatMacroStorage, StoredMacro


@dataclass(slots=True)
class ChatMacroCatalogItem:
    """Resolved built-in or user macro metadata exposed by the service."""

    name: str
    command: str
    description: str | None
    enabled: bool
    source: str
    immutable: bool
    digest: str
    definition: MacroDefinition
    builtin_version: int | None = None


class ChatMacrosService:
    """Coordinate macro definitions, settings overlays, and registry state."""

    def __init__(
        self,
        *,
        user_id: str,
        storage: ChatMacroStorage,
        repository: ChatMacroRepository,
        core_commands: Iterable[str] | None = None,
    ) -> None:
        """Create a user-scoped service with reserved core commands."""
        self.user_id = str(user_id)
        self.storage = storage
        self.repository = repository
        self.core_commands = {command.lower() for command in (core_commands or _default_core_commands())}

    def list_macros(self) -> list[ChatMacroCatalogItem]:
        """List resolved macros and synchronize changed registry rows."""
        items = self._catalog_items()
        for item in items:
            self._reject_core_collision(item.definition)
        self._sync_registry_catalog(items)
        return sorted(items, key=lambda item: (item.command, item.source))

    def get_macro(self, name: str) -> ChatMacroCatalogItem:
        """Load one macro and synchronize its registry row when changed."""
        builtin = self._builtin_item(name)
        if builtin is not None:
            self._sync_registry_if_changed(builtin)
            return builtin
        stored = self.storage.read(name)
        enabled_overrides = dict(self.get_settings().get("user_macro_enabled", {}))
        item = self._user_item(stored, enabled_overrides=enabled_overrides)
        self._reject_core_collision(item.definition)
        self._sync_registry_if_changed(item)
        return item

    def validate_macro(self, raw: str) -> MacroDefinition:
        """Validate a user definition without persisting it."""
        definition = load_macro_definition(raw)
        self._validate_macro_name(definition.name)
        self._validate_future_permissions(definition)
        self._reject_core_collision(definition)
        return definition

    def create_macro(
        self,
        name: str,
        raw: str,
        supporting_files: dict[str, str | bytes] | None = None,
    ) -> ChatMacroCatalogItem:
        """Create a validated user macro and synchronize the catalog."""
        definition = self.validate_macro(raw)
        self._reject_macro_collision(definition, exclude_name=name)
        stored = self.storage.create(name, raw, supporting_files)
        item = self._user_item(stored)
        self._sync_registry_catalog(self._catalog_items())
        return item

    def update_macro(
        self,
        name: str,
        raw: str,
        supporting_files: dict[str, str | bytes] | None = None,
    ) -> ChatMacroCatalogItem:
        """Replace a user macro; built-in definitions remain immutable."""
        if self._builtin_item(name) is not None:
            raise MacroStorageError("built-in macros are immutable")
        definition = self.validate_macro(raw)
        self._reject_macro_collision(definition, exclude_name=name)
        stored = self.storage.update(name, raw, supporting_files)
        item = self._user_item(stored)
        self._sync_registry_catalog(self._catalog_items())
        return item

    def set_macro_enabled(self, name: str, enabled: bool) -> ChatMacroCatalogItem:
        """Persist a built-in or user enabled-state override without rewriting YAML."""
        if self._load_builtin(name) is not None:
            return self.set_builtin_enabled(name, enabled)

        stored = self.storage.read(name)
        settings = self.get_settings()
        overrides = dict(settings.get("user_macro_enabled", {}))
        overrides[name] = enabled
        settings["user_macro_enabled"] = overrides
        self.save_settings(settings)
        item = self._user_item(stored, enabled_overrides=overrides)
        self._sync_registry_if_changed(item)
        return item

    def delete_macro(self, name: str) -> None:
        """Delete a user macro and remove its settings override and registry row."""
        if self._builtin_item(name) is not None:
            raise MacroStorageError("built-in macros are immutable")
        self.storage.delete(name)
        settings = self.get_settings()
        overrides = dict(settings.get("user_macro_enabled", {}))
        if overrides.pop(name, None) is not None:
            settings["user_macro_enabled"] = overrides
            self.save_settings(settings)
        self._sync_registry_catalog(self._catalog_items())

    def set_builtin_enabled(self, name: str, enabled: bool) -> ChatMacroCatalogItem:
        """Enable or disable an immutable built-in through user settings."""
        if self._load_builtin(name) is None:
            raise MacroNotFoundError(f"built-in macro not found: {name}")
        settings = self.get_settings()
        disabled = set(settings.get("disabled_builtins", []))
        if enabled:
            disabled.discard(name)
        else:
            disabled.add(name)
        settings["disabled_builtins"] = sorted(disabled)
        self.save_settings(settings)
        item = self._builtin_item(name)
        if item is None:
            raise MacroNotFoundError(f"built-in macro not found: {name}")
        self._sync_registry(item)
        return item

    def clone_builtin(self, name: str, *, new_name: str, command: str | None = None) -> ChatMacroCatalogItem:
        """Clone a built-in definition into mutable user storage."""
        builtin = self._load_builtin(name)
        if builtin is None:
            raise MacroNotFoundError(f"built-in macro not found: {name}")
        command = command or new_name
        payload = builtin.definition.model_dump(mode="json")
        payload.update({"name": new_name, "command": command, "builtin_version": None})
        raw = yaml.safe_dump(payload, sort_keys=False)
        return self.create_macro(new_name, raw)

    def get_settings(self) -> dict[str, Any]:
        """Return normalized user-scoped chat macro settings."""
        return normalize_settings(self.repository.get_settings(self.user_id))

    def save_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """Validate and persist user-scoped chat macro settings."""
        normalized = normalize_settings(settings)
        return self.repository.save_settings(self.user_id, normalized)

    def resolve_output_profile(
        self,
        name: str | None = None,
        *,
        local_overrides: dict[str, Any] | None = None,
    ) -> MacroOutputProfile:
        """Resolve a named output profile with optional macro-local overrides."""
        settings = self.get_settings()
        profile_name = name or "default"
        raw_profile = settings.get("output_profiles", {}).get(profile_name)
        if raw_profile is None:
            raw_profile = settings.get("output_profiles", {}).get("default", {})
            profile_name = "default"
        return merge_output_profile(normalize_output_profile(profile_name, raw_profile), local_overrides)

    def _builtin_items(self) -> list[ChatMacroCatalogItem]:
        settings = self.get_settings()
        disabled = set(settings.get("disabled_builtins", []))
        return [
            item
            for item in (
                self._builtin_item(path.name, disabled=disabled)
                for path in _builtin_root().iterdir()
                if path.is_dir()
            )
            if item is not None
        ]

    def _catalog_items(self) -> list[ChatMacroCatalogItem]:
        settings = self.get_settings()
        disabled = set(settings.get("disabled_builtins", []))
        enabled_overrides = dict(settings.get("user_macro_enabled", {}))
        builtins = [
            item
            for item in (
                self._builtin_item(path.name, disabled=disabled)
                for path in _builtin_root().iterdir()
                if path.is_dir()
            )
            if item is not None
        ]
        return builtins + [
            self._user_item(stored, enabled_overrides=enabled_overrides)
            for stored in self.storage.list()
        ]

    def _builtin_item(
        self,
        name: str,
        *,
        disabled: set[str] | None = None,
    ) -> ChatMacroCatalogItem | None:
        loaded = self._load_builtin(name)
        if loaded is None:
            return None
        disabled = disabled if disabled is not None else set(
            self.get_settings().get("disabled_builtins", [])
        )
        return ChatMacroCatalogItem(
            name=loaded.name,
            command=loaded.definition.command,
            description=loaded.definition.description,
            enabled=loaded.definition.enabled and loaded.name not in disabled,
            source="builtin",
            immutable=True,
            digest=loaded.digest,
            definition=loaded.definition,
            builtin_version=loaded.definition.builtin_version,
        )

    def _load_builtin(self, name: str) -> StoredMacro | None:
        path = _builtin_root() / name / "MACRO.yaml"
        if not path.is_file():
            return None
        raw = path.read_text(encoding="utf-8")
        definition = load_macro_definition(raw)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        return StoredMacro(name=definition.name, definition=definition, raw=raw, digest=digest, supporting_files={})

    @staticmethod
    def _user_item(
        stored: StoredMacro,
        *,
        enabled_overrides: dict[str, bool] | None = None,
    ) -> ChatMacroCatalogItem:
        override = (enabled_overrides or {}).get(stored.name)
        return ChatMacroCatalogItem(
            name=stored.name,
            command=stored.definition.command,
            description=stored.definition.description,
            enabled=stored.definition.enabled if override is None else override,
            source="user",
            immutable=False,
            digest=stored.digest,
            definition=stored.definition,
            builtin_version=stored.definition.builtin_version,
        )

    def _reject_core_collision(self, definition: MacroDefinition) -> None:
        if definition.command.lower() in self.core_commands:
            raise MacroValidationError("macro command conflicts with core command")

    def _reject_macro_collision(self, definition: MacroDefinition, *, exclude_name: str) -> None:
        for item in self._catalog_items():
            if item.name == exclude_name and item.source == "user":
                continue
            if item.name == definition.name:
                raise MacroValidationError("macro name conflicts with another macro")
            if item.command == definition.command:
                raise MacroValidationError("macro command conflicts with another macro")

    @staticmethod
    def _validate_future_permissions(definition: MacroDefinition) -> None:
        if definition.permissions.tool_calls:
            raise MacroValidationError("tool_calls are not allowed in chat macro definitions")
        if definition.permissions.skills:
            raise MacroValidationError("skills are not allowed in chat macro definitions")
        if definition.execution.mode_default != "background":
            raise MacroValidationError("only background macro execution is supported in v1")

    def _sync_registry(self, item: ChatMacroCatalogItem) -> None:
        self.repository.upsert_registry_entry(
            user_id=self.user_id,
            name=item.name,
            command=item.command,
            description=item.description,
            enabled=item.enabled,
            source=item.source,
            builtin_version=item.builtin_version,
            schema_version=item.definition.schema_version,
            digest=item.digest,
            validation_status="valid",
            validation_error=None,
        )

    def _sync_registry_if_changed(self, item: ChatMacroCatalogItem) -> None:
        existing = next(
            (
                row
                for row in self.repository.list_registry_entries(self.user_id)
                if row.get("command") == item.command
            ),
            None,
        )
        if not self._registry_entry_matches(existing, item):
            self._sync_registry(item)

    def _sync_registry_catalog(self, items: list[ChatMacroCatalogItem]) -> None:
        existing = {
            str(row["command"]): row
            for row in self.repository.list_registry_entries(self.user_id)
        }
        active_commands = {item.command for item in items}
        for item in items:
            if not self._registry_entry_matches(existing.get(item.command), item):
                self._sync_registry(item)
        if set(existing) != active_commands:
            self.repository.mark_registry_entries_deleted_except(self.user_id, active_commands)

    @staticmethod
    def _registry_entry_matches(row: dict[str, Any] | None, item: ChatMacroCatalogItem) -> bool:
        if row is None:
            return False
        return (
            row.get("name") == item.name
            and row.get("command") == item.command
            and row.get("description") == item.description
            and bool(row.get("enabled")) is item.enabled
            and row.get("source") == item.source
            and row.get("builtin_version") == item.builtin_version
            and row.get("schema_version") == item.definition.schema_version
            and row.get("digest") == item.digest
            and row.get("validation_status") == "valid"
            and row.get("validation_error") is None
        )

    @staticmethod
    def _validate_macro_name(name: str) -> None:
        if not MACRO_NAME_RE.fullmatch(name or ""):
            raise MacroValidationError("invalid macro name")


def _builtin_root() -> Path:
    return Path(__file__).resolve().parent / "builtin"


def _default_core_commands() -> set[str]:
    """Return reserved built-in slash command names, failing closed on import errors."""
    try:
        from tldw_Server_API.app.core.Chat.command_router import list_commands
    except ImportError as exc:
        logger.exception("Failed to load reserved chat command names")
        raise MacroStorageError("chat command router is unavailable") from exc
    return {str(command["name"]).lower() for command in list_commands()}
