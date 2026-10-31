# VoiceAssistant/registry.py
# Voice Command Registry - Manages storage and retrieval of voice commands
#
#######################################################################################################################
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml
from loguru import logger

from .schemas import ActionType, VoiceCommand


class VoiceCommandRegistry:
    """
    Manages voice command registration and lookup.

    Commands can be loaded from:
    - Database (per-user custom commands)
    - YAML config file (system defaults)
    - In-memory registration (runtime additions)
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the voice command registry.

        Args:
            config_path: Path to YAML config file with default commands.
                        If None, loads from Config_Files/voice_commands.yaml
        """
        self._commands: dict[str, VoiceCommand] = {}
        self._user_commands: dict[tuple[int, str | None], dict[str, VoiceCommand]] = {}
        self._config_path = config_path
        self._loaded = False

    @staticmethod
    def _persona_bucket_key(user_id: int, persona_id: str | None = None) -> tuple[int, str | None]:
        normalized_persona_id = str(persona_id).strip() if persona_id is not None else None
        if normalized_persona_id == "":
            normalized_persona_id = None
        return user_id, normalized_persona_id

    @staticmethod
    def _phrase_has_slots(phrase: str) -> bool:
        return bool(re.search(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}", str(phrase or "")))

    @classmethod
    def _matches_phrase(cls, text: str, phrase: str) -> float:
        phrase_lower = str(phrase or "").lower().strip()
        if not phrase_lower:
            return 0.0
        if text == phrase_lower:
            return 1.0
        if cls._phrase_has_slots(phrase_lower):
            pattern = "^" + re.sub(
                r"\\\{([a-zA-Z_][a-zA-Z0-9_]*)\\\}",
                r"(?P<\1>.+?)",
                re.escape(phrase_lower),
            ) + "$"
            pattern = re.sub(r"\\ ", r"\\s+", pattern)
            return 0.97 if re.match(pattern, text, re.IGNORECASE) else 0.0
        if text.startswith(phrase_lower):
            score = len(phrase_lower) / len(text)
            return min(score + 0.1, 0.99)
        return 0.0

    def _commands_for_user_scope(
        self,
        user_id: int,
        *,
        persona_id: str | None = None,
    ) -> list[VoiceCommand]:
        if persona_id is not None:
            return list(self._user_commands.get(self._persona_bucket_key(user_id, persona_id), {}).values())

        out: list[VoiceCommand] = []
        seen: set[str] = set()
        for (bucket_user_id, _bucket_persona_id), commands in self._user_commands.items():
            if bucket_user_id != user_id:
                continue
            for command in commands.values():
                if command.id in seen:
                    continue
                seen.add(command.id)
                out.append(command)
        return out

    def _get_default_config_path(self) -> Path:
        """Get the default config file path."""
        # Navigate up from this file to find Config_Files
        current_dir = Path(__file__).parent
        # tldw_Server_API/app/core/VoiceAssistant -> tldw_Server_API/Config_Files
        config_dir = current_dir.parent.parent.parent / "Config_Files"
        return config_dir / "voice_commands.yaml"

    def load_defaults(self) -> None:
        """Load default commands from YAML config file."""
        if self._loaded:
            return

        config_path = Path(self._config_path) if self._config_path else self._get_default_config_path()

        if not config_path.exists():
            logger.warning(f"Voice commands config not found at {config_path}, using built-in defaults")
            self._load_builtin_defaults()
            self._loaded = True
            return

        try:
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)

            commands = config.get("commands", [])
            for cmd_data in commands:
                try:
                    command = VoiceCommand(
                        id=cmd_data.get("id", str(uuid.uuid4())),
                        user_id=0,  # System commands have user_id 0
                        name=cmd_data["name"],
                        phrases=cmd_data["phrases"],
                        action_type=ActionType(cmd_data["action_type"]),
                        action_config=cmd_data.get("action_config", {}),
                        priority=cmd_data.get("priority", 0),
                        enabled=cmd_data.get("enabled", True),
                        requires_confirmation=cmd_data.get("requires_confirmation", False),
                        description=cmd_data.get("description"),
                        created_at=datetime.utcnow(),
                    )
                    self._commands[command.id] = command
                except Exception as e:
                    logger.warning(f"Failed to load voice command '{cmd_data.get('name', 'unknown')}': {e}")

            logger.info(f"Loaded {len(self._commands)} voice commands from {config_path}")
            self._loaded = True

        except Exception as e:
            logger.error(f"Failed to load voice commands config: {e}")
            self._load_builtin_defaults()
            self._loaded = True

    def _load_builtin_defaults(self) -> None:
        """Load built-in default commands when config file is unavailable."""
        defaults = [
            VoiceCommand(
                id="builtin-search-media",
                user_id=0,
                name="Search Media",
                phrases=[
                    "search for",
                    "find",
                    "look up",
                    "search",
                ],
                action_type=ActionType.MCP_TOOL,
                action_config={
                    "tool_name": "media.search",
                    "extract_query": True,
                },
                priority=10,
                description="Search through ingested media content",
            ),
            VoiceCommand(
                id="builtin-create-note",
                user_id=0,
                name="Create Note",
                phrases=[
                    "create a note",
                    "make a note",
                    "take a note",
                    "note that",
                    "remember that",
                ],
                action_type=ActionType.MCP_TOOL,
                action_config={
                    "tool_name": "notes.create",
                    "extract_content": True,
                },
                priority=10,
                description="Create a new note",
            ),
            VoiceCommand(
                id="builtin-stop",
                user_id=0,
                name="Stop",
                phrases=["stop", "cancel", "never mind", "abort"],
                action_type=ActionType.CUSTOM,
                action_config={"action": "stop"},
                priority=100,  # High priority for control commands
                description="Stop current operation",
            ),
            VoiceCommand(
                id="builtin-help",
                user_id=0,
                name="Help",
                phrases=["help", "what can you do", "commands", "show commands"],
                action_type=ActionType.CUSTOM,
                action_config={"action": "help"},
                priority=50,
                description="List available commands",
            ),
        ]

        for cmd in defaults:
            self._commands[cmd.id] = cmd

        logger.info(f"Loaded {len(defaults)} built-in default voice commands")

    def register_command(self, command: VoiceCommand) -> None:
        """
        Register a voice command.

        Args:
            command: The command to register
        """
        if command.user_id == 0:
            self._commands[command.id] = command
        else:
            bucket_key = self._persona_bucket_key(command.user_id, command.persona_id)
            if bucket_key not in self._user_commands:
                self._user_commands[bucket_key] = {}
            self._user_commands[bucket_key][command.id] = command

        logger.debug(f"Registered voice command: {command.name} (id={command.id})")

    def unregister_command(
        self,
        command_id: str,
        user_id: int = 0,
        persona_id: str | None = None,
    ) -> bool:
        """
        Unregister a voice command.

        Args:
            command_id: ID of the command to remove
            user_id: User ID (0 for system commands)

        Returns:
            True if command was removed, False if not found
        """
        if user_id == 0:
            if command_id in self._commands:
                del self._commands[command_id]
                return True
        else:
            removed = False
            if persona_id is not None:
                bucket_key = self._persona_bucket_key(user_id, persona_id)
                if bucket_key in self._user_commands and command_id in self._user_commands[bucket_key]:
                    del self._user_commands[bucket_key][command_id]
                    removed = True
            else:
                for bucket_key, commands in self._user_commands.items():
                    if bucket_key[0] != user_id or command_id not in commands:
                        continue
                    del commands[command_id]
                    removed = True
            if removed:
                return True
        return False

    def get_command(
        self,
        command_id: str,
        user_id: int = 0,
        persona_id: str | None = None,
    ) -> Optional[VoiceCommand]:
        """
        Get a command by ID.

        Args:
            command_id: ID of the command
            user_id: User ID to check user commands first

        Returns:
            The command if found, None otherwise
        """
        # Check user commands first
        if user_id:
            if persona_id is not None:
                scoped_commands = self._user_commands.get(self._persona_bucket_key(user_id, persona_id), {})
                if command_id in scoped_commands:
                    return scoped_commands[command_id]
            else:
                for (bucket_user_id, _bucket_persona_id), commands in self._user_commands.items():
                    if bucket_user_id != user_id:
                        continue
                    if command_id in commands:
                        return commands[command_id]
        # Then check system commands
        return self._commands.get(command_id)

    def get_all_commands(
        self,
        user_id: int = 0,
        include_system: bool = True,
        include_disabled: bool = False,
        persona_id: str | None = None,
    ) -> list[VoiceCommand]:
        """
        Get all available commands for a user.

        Args:
            user_id: User ID
            include_system: Whether to include system commands

        Returns:
            List of available commands, sorted by priority (descending)
        """
        commands = []

        if include_system:
            commands.extend(self._commands.values())

        commands.extend(self._commands_for_user_scope(user_id, persona_id=persona_id))

        if include_disabled:
            return sorted(commands, key=lambda c: c.priority, reverse=True)

        # Filter enabled commands and sort by priority
        enabled = [c for c in commands if c.enabled]
        return sorted(enabled, key=lambda c: c.priority, reverse=True)

    def refresh_user_commands(
        self,
        db,
        user_id: int,
        include_disabled: bool = True,
        persona_id: str | None = None,
    ) -> list[VoiceCommand]:
        """
        Replace cached user commands from the database.

        Args:
            db: CharactersRAGDB instance
            user_id: User ID
            include_disabled: Include disabled commands in the cache

        Returns:
            List of loaded commands
        """
        commands = self.load_user_commands(db, user_id, include_disabled=include_disabled, persona_id=persona_id)
        self.apply_user_commands(user_id, commands, persona_id=persona_id)
        return commands

    def load_user_commands(
        self,
        db,
        user_id: int,
        include_disabled: bool = True,
        persona_id: str | None = None,
    ) -> list[VoiceCommand]:
        """Read a user's commands from the database without touching the cache (safe off the event loop)."""
        from .db_helpers import get_user_voice_commands

        return get_user_voice_commands(
            db,
            user_id=user_id,
            include_system=False,
            enabled_only=not include_disabled,
            persona_id=persona_id,
        )

    def apply_user_commands(
        self,
        user_id: int,
        commands: list[VoiceCommand],
        persona_id: str | None = None,
    ) -> None:
        """Replace the cached user commands; call it on the event loop, which owns the cache."""
        if persona_id is not None:
            self._user_commands[self._persona_bucket_key(user_id, persona_id)] = {cmd.id: cmd for cmd in commands}
        else:
            keys_to_remove = [key for key in self._user_commands if key[0] == user_id]
            for key in keys_to_remove:
                del self._user_commands[key]
            for command in commands:
                bucket_key = self._persona_bucket_key(user_id, command.persona_id)
                if bucket_key not in self._user_commands:
                    self._user_commands[bucket_key] = {}
                self._user_commands[bucket_key][command.id] = command

    def find_matching_commands(
        self,
        text: str,
        user_id: int = 0,
        persona_id: str | None = None,
        include_disabled: bool = False,
    ) -> list[tuple[VoiceCommand, str, float]]:
        """
        Find commands that match the given text.

        Uses prefix matching on registered phrases.

        Args:
            text: The text to match against
            user_id: User ID for user-specific commands

        Returns:
            List of (command, matched_phrase, score) tuples, sorted by score descending
        """
        if not self._loaded:
            self.load_defaults()

        text_lower = text.lower().strip()
        matches = []

        all_commands = self.get_all_commands(
            user_id,
            include_system=True,
            include_disabled=include_disabled,
            persona_id=persona_id,
        )

        for command in all_commands:
            for phrase in command.phrases:
                score = self._matches_phrase(text_lower, phrase)
                if score > 0:
                    matches.append((command, phrase, score))

        # Sort by score descending, then by priority descending
        matches.sort(key=lambda x: (x[2], x[0].priority), reverse=True)
        return matches

    def load_user_commands_from_db(
        self,
        user_id: int,
        db_rows: list[dict[str, Any]],
        persona_id: str | None = None,
    ) -> None:
        """
        Load user commands from database rows.

        Args:
            user_id: User ID
            db_rows: List of database row dicts with command data
        """
        if persona_id is not None:
            self._user_commands[self._persona_bucket_key(user_id, persona_id)] = {}

        for row in db_rows:
            try:
                phrases = row.get("phrases", "[]")
                if isinstance(phrases, str):
                    phrases = json.loads(phrases)

                action_config = row.get("action_config", "{}")
                if isinstance(action_config, str):
                    action_config = json.loads(action_config)

                command = VoiceCommand(
                    id=row["id"],
                    user_id=user_id,
                    persona_id=row.get("persona_id"),
                    connection_id=row.get("connection_id"),
                    name=row["name"],
                    phrases=phrases,
                    action_type=ActionType(row["action_type"]),
                    action_config=action_config,
                    priority=row.get("priority", 0),
                    enabled=bool(row.get("enabled", 1)),
                    requires_confirmation=bool(row.get("requires_confirmation", 0)),
                    description=row.get("description"),
                )
                bucket_key = self._persona_bucket_key(user_id, command.persona_id)
                if bucket_key not in self._user_commands:
                    self._user_commands[bucket_key] = {}
                self._user_commands[bucket_key][command.id] = command
            except Exception as e:
                logger.warning(f"Failed to load user voice command from DB: {e}")

        loaded_count = sum(
            len(commands)
            for (bucket_user_id, bucket_persona_id), commands in self._user_commands.items()
            if bucket_user_id == user_id and (persona_id is None or bucket_persona_id == persona_id)
        )
        logger.debug(f"Loaded {loaded_count} commands for user {user_id}")


# Singleton instance
_registry_instance: Optional[VoiceCommandRegistry] = None


def get_voice_command_registry() -> VoiceCommandRegistry:
    """Get the singleton voice command registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = VoiceCommandRegistry()
    return _registry_instance


#
# End of VoiceAssistant/registry.py
#######################################################################################################################
