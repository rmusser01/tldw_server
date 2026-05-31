"""Utilities for managing the first-time setup flow.

This module centralises reading and writing of the main ``config.txt`` file so the
first-time setup experience can query and update configuration without duplicating
file handling logic across routers or UI layers.
"""

from __future__ import annotations

import os
import re
import shutil
from configparser import ConfigParser
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from tldw_Server_API.app.core.config_paths import resolve_config_file, resolve_config_root
from tldw_Server_API.app.core.testing import env_flag_enabled
from tldw_Server_API.app.core.Utils.Utils import get_project_root

SETUP_SECTION = "Setup"
CONFIG_FILENAME = "config.txt"
# Legacy compatibility: install_manager expects a module-level config path constant.
CONFIG_RELATIVE_PATH = resolve_config_root() / CONFIG_FILENAME
REMOTE_ACCESS_FIELD = "allow_remote_setup_access"

SENSITIVE_KEY_MARKERS = ("key", "token", "secret", "password", "api_key")
PLACEHOLDER_VALUES = {
    "",
    "your_api_key_here",
    "YOUR_API_KEY_HERE",
    "default-secret-key-for-single-user",
    "test-api-key-12345",
    "CHANGE_ME_TO_SECURE_API_KEY",
    "TestPassword123!",
    "change-me-in-production",
}

_USER_DB_BASE_DIR_SECTION = "TTS-Settings"
_USER_DB_BASE_DIR_KEY = "USER_DB_BASE_DIR"
_USER_DB_BASE_DIR_ALLOWED_ROOT_ENVS = (
    "USER_DB_BASE_DIR_ALLOWED_ROOTS",
    "TLDW_USER_DB_BASE_DIR_ALLOWED_ROOTS",
)
_INGESTION_SOURCE_ALLOWED_ROOTS_SECTION = "Files"
_INGESTION_SOURCE_ALLOWED_ROOTS_KEY = "ingestion_source_allowed_roots"
_OPTIONAL_EMPTY_VALUE_FIELDS = {
    ("LlamaCpp", "executable_path"),
    ("LlamaCpp", "models_dir"),
    ("LlamaCpp", "default_host"),
    ("LlamaCpp", "default_port"),
    ("LlamaCpp", "default_threads"),
    ("LlamaCpp", "default_n_gpu_layers"),
    ("LlamaCpp", "default_ctx_size"),
    ("LlamaCpp", "port_probe_max"),
    ("LlamaCpp", "allowed_paths"),
    ("LlamaCpp", "log_output_file"),
}
_provider_catalog_update_fields: set[tuple[str, str]] | None = None

_remote_access_hook: Callable[[bool], None] | None = None


def register_remote_access_hook(callback: Callable[[bool], None]) -> None:
    """Register a callback fired whenever remote setup access toggles."""
    global _remote_access_hook
    _remote_access_hook = callback


SECTION_LABELS: dict[str, str] = {
    "API": "API Providers",
    "Processing": "Processing",
    "Media-Processing": "Media Processing",
    "Server": "Server Settings",
    "Chat-Module": "Chat Module",
    "Character-Chat": "Character Chat",
    "Chat-Dictionaries": "Chat Dictionaries",
    "Settings": "General Settings",
    "Auto-Save": "Auto-Save",
    "Database": "Database",
    "AuthNZ": "Authentication",
    "Embeddings": "Embeddings",
    "RAG": "Retrieval Augmented Generation",
    "Chunking": "Chunking",
    "STT-Settings": "Speech-to-Text",
    "TTS-Settings": "Text-to-Speech",
    "Logging": "Logging",
    "Setup": "Setup",
    "Prompts": "Prompts",
    "Search-Engines": "Search Engines",
    "Local-API": "Local API Providers",
    "Claims": "Claims Extraction",
    "MCP": "Model Context Protocol",
    "MCP-Unified": "MCP Unified",
}

SECTION_DESCRIPTIONS: dict[str, str] = {
    "Setup": "Controls the guided setup flow shown on first launch.",
    "AuthNZ": "Configure authentication mode and credentials.",
    "API": "Add API keys for the providers you plan to use.",
    "Server": "Server-level behaviour toggles.",
    "Database": "Manage database storage locations and options.",
    "Embeddings": "Configure embedding providers and defaults.",
    "RAG": "Tune retrieval and augmentation behaviour.",
    "Logging": "Adjust logging paths and verbosity.",
    "Processing": "Base ingestion defaults such as preferred hardware and general processing strategy.",
    "Media-Processing": "Control file-size limits, conversion timeouts, and media ingestion safeguards.",
    "Settings": "Miscellaneous server toggles, retention policies, and behavioural guards.",
    "Auto-Save": "Persistence defaults for chats, notes, and other artefacts.",
    "Chat-Module": "Primary chat flow defaults including rate limits, streaming, and fallbacks.",
    "Character-Chat": "Persona chat controls, limits, and import behaviour.",
    "Chunking": "Adjust chunking defaults, overlap, and adaptive strategies for ingested documents.",
    "Prompts": "Manage reusable prompt templates for summarisation, chat, and automation.",
    "Search-Engines": "Configure search providers, query budgets, and language preferences.",
    "Local-API": "Point to locally hosted model backends such as Ollama or Kobold.",
    "Claims": "Control ingestion-time claim extraction and rebuild workers.",
    "MCP": "Configure Model Context Protocol hosts, tokens, and available tool sets.",
    "MCP-Unified": "Manage the unified MCP service, registry, and role-based access.",
    "Chat-Dictionaries": "Enable and configure dictionary-based replacements in chat flows.",
    "STT-Settings": "Speech-to-text providers, streaming, and buffering controls.",
    "TTS-Settings": "Voice synthesis providers, defaults, and output configuration.",
}

FIELD_HINTS: dict[tuple[str, str], str] = {
    ("AuthNZ", "single_user_api_key"): "Strong secret used for X-API-KEY requests in single-user mode.",
    ("AuthNZ", "auth_mode"): "Use 'single_user' for local setups or 'multi_user' for JWT-based auth.",
    ("Setup", "allow_remote_setup_access"): "Permit the setup API outside localhost. Only enable on trusted networks.",
    ("Files", "ingestion_source_allowed_roots"): "Allowed base directories for local ingestion sources. Separate multiple roots with commas or your platform path separator.",
    ("API", "openai_api_key"): "Personal or organisational OpenAI key.",
    ("API", "anthropic_api_key"): "Anthropic Claude API key.",
    ("API", "google_api_key"): "Google Generative AI key.",
    ("API", "groq_api_key"): "Groq LPU inference key.",
    ("Database", "sqlite_path"): "Path to the main content SQLite database.",
    ("Server", "disable_cors"): "If true, skips CORS middleware entirely (same-origin clients only).",
    ("Server", "trusted_proxies"): "Proxy IPs/CIDRs trusted for X-Forwarded-For/X-Real-IP resolution.",
    ("Setup", "setup_ip_allowlist"): "Comma-separated IPs/CIDRs allowed for remote Setup UI.",
    ("Setup", "setup_ip_denylist"): "Comma-separated IPs/CIDRs explicitly blocked from Setup UI.",
}


def _read_config_lines() -> list[str]:
    """Read the raw config file preserving comments for contextual hints."""
    config_path = get_config_file_path()
    with config_path.open("r", encoding="utf-8") as stream:
        return stream.readlines()


def _build_comment_index() -> tuple[dict[tuple[str, str], str], dict[str, str]]:
    """Return mappings of field and section comments gathered from config.txt."""
    lines = _read_config_lines()
    field_comments: dict[tuple[str, str], str] = {}
    section_comments: dict[str, str] = {}

    current_section: str | None = None
    pending_field_comments: list[str] = []
    pending_section_comments: list[str] = []
    seen_field_in_section = False

    for raw in lines:
        stripped = raw.strip()

        if not stripped:
            pending_field_comments.clear()
            if not seen_field_in_section:
                pending_section_comments.clear()
            continue

        # Treat both '#' and ';' as comment markers (INI-style)
        if stripped.startswith('#') or stripped.startswith(';'):
            # Support both '#' and ';' as full-line comment markers
            comment_text = stripped.lstrip('#;').strip()
            if comment_text:
                pending_field_comments.append(comment_text)
                if not seen_field_in_section:
                    pending_section_comments.append(comment_text)
            continue

        if stripped.startswith('[') and stripped.endswith(']'):
            # Close previous section comment if we never encountered a field.
            if current_section and pending_section_comments and not seen_field_in_section:
                section_comments[current_section] = ' '.join(pending_section_comments).strip()

            current_section = stripped[1:-1].strip()
            pending_field_comments.clear()
            pending_section_comments = []
            seen_field_in_section = False
            continue

        if '=' in raw and current_section:
            line_without_inline = raw
            inline_comment = ''
            # Support inline comments introduced by '#' or ';'
            if '#' in raw or ';' in raw:
                # Prefer the earliest occurring comment marker
                hash_idx = raw.find('#') if '#' in raw else len(raw) + 1
                semi_idx = raw.find(';') if ';' in raw else len(raw) + 1
                cut_idx = min(hash_idx, semi_idx)
                if cut_idx < len(raw):
                    before = raw[:cut_idx]
                    after = raw[cut_idx + 1 :]
                    line_without_inline = before
                    inline_comment = after.strip()

            key = line_without_inline.split('=', 1)[0].strip()
            comment_parts = list(pending_field_comments)
            if inline_comment:
                comment_parts.append(inline_comment)

            if comment_parts:
                field_comments[(current_section, key)] = ' '.join(comment_parts).strip()

            if pending_section_comments and not seen_field_in_section:
                section_comments[current_section] = ' '.join(pending_section_comments).strip()

            pending_field_comments.clear()
            seen_field_in_section = True
            continue

    if current_section and pending_section_comments and not seen_field_in_section:
        section_comments[current_section] = ' '.join(pending_section_comments).strip()

    return field_comments, section_comments


def _humanise(value: str) -> str:
    words = re.split(r'[_\-]+', value)
    return ' '.join(word.capitalize() for word in words if word)


def _humanise_section_name(section: str) -> str:
    return SECTION_LABELS.get(section, section.replace('_', ' ').replace('-', ' ').title())


def _generate_default_hint(section: str, key: str) -> str:
    friendly_key = _humanise(key)
    section_label = _humanise_section_name(section)
    return f"Adjust {friendly_key} in {section_label}."


def _normalise_text(value: str) -> str:
    return re.sub(r'\s+', ' ', value.strip()).lower()


def _tokenise(text: str) -> list[str]:
    return [token for token in re.findall(r'[a-z0-9]+', text.lower()) if len(token) > 2]


def _score_entry(query_lower: str, query_tokens: list[str], *candidates: str) -> float:
    best = 0.0
    for candidate in candidates:
        if not candidate:
            continue
        candidate_lower = _normalise_text(candidate)
        ratio = SequenceMatcher(None, query_lower, candidate_lower).ratio()
        token_hits = 0
        for token in query_tokens:
            if token in candidate_lower:
                token_hits += 1
        token_score = (token_hits / len(query_tokens)) if query_tokens else 0.0
        score = (ratio * 0.7) + (token_score * 0.3)
        if score > best:
            best = score
    return min(best, 1.0)


def get_config_file_path() -> Path:
    """Return the full filesystem path to ``config.txt``."""
    return resolve_config_file(CONFIG_FILENAME)


def _load_config_parser() -> ConfigParser:
    """Load the configuration file into a ``ConfigParser`` instance."""
    config_path = get_config_file_path()
    parser = ConfigParser()
    parser.optionxform = str  # preserve key case

    if not config_path.exists():
        raise FileNotFoundError(
            f"Expected configuration file at {config_path}. Run the installer or create config.txt."
        )

    parser.read(config_path, encoding="utf-8")
    return parser


def _coerce_to_string(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def validate_config_value_single_line(section: str, key: str, value: Any) -> str:
    """Return a config value string after rejecting multiline injection chars."""
    text = _coerce_to_string(value)
    if any(char in text for char in ("\r", "\n", "\x00")):
        raise ValueError(f"Invalid config value for {section}.{key}: line breaks and NUL bytes are not allowed.")
    return text


def _infer_type(raw_value: str) -> str:
    lowered = raw_value.strip().lower()
    if lowered in {"true", "false", "yes", "no", "on", "off", "1", "0"}:
        return "boolean"
    try:
        int(raw_value)
        return "integer"
    except ValueError:
        pass
    try:
        float(raw_value)
        return "number"
    except ValueError:
        pass
    return "string"


def _split_allowed_roots(raw: str) -> list[str]:
    if not raw:
        return []
    parts: list[str] = []
    for chunk in raw.split(os.pathsep):
        for entry in chunk.split(","):
            entry = entry.strip()
            if entry:
                parts.append(entry)
    return parts


def _normalize_root_path(raw: str, *, project_root: Path) -> Path | None:
    value = str(raw).strip()
    if not value:
        return None
    expanded = os.path.expanduser(value)
    if os.path.isabs(expanded):
        return Path(os.path.normpath(expanded))
    candidate = Path(os.path.normpath(os.path.join(str(project_root), expanded)))
    try:
        candidate.relative_to(project_root)
    except ValueError:
        return None
    return candidate


def _normalize_user_db_base_dir_candidate(raw: Any, *, project_root: Path) -> Path:
    value = str(raw).strip()
    if not value:
        raise ValueError("USER_DB_BASE_DIR must not be empty")
    expanded = os.path.expanduser(value)
    if not os.path.isabs(expanded):
        candidate = Path(os.path.normpath(os.path.join(str(project_root), expanded)))
        try:
            candidate.relative_to(project_root)
        except ValueError as exc:
            raise ValueError("Relative USER_DB_BASE_DIR must stay within the project root") from exc
    else:
        candidate = Path(os.path.normpath(expanded))
    return candidate


def _collect_allowed_user_db_base_roots(parser: ConfigParser, *, project_root: Path) -> list[Path]:
    roots: list[Path] = []

    default_root = (project_root / "Databases").resolve()
    roots.append(default_root)

    current_raw = parser.get(_USER_DB_BASE_DIR_SECTION, _USER_DB_BASE_DIR_KEY, fallback="").strip()
    if current_raw:
        try:
            current_base = _normalize_user_db_base_dir_candidate(current_raw, project_root=project_root)
        except ValueError:
            current_base = None
        if current_base is not None:
            parent = current_base.parent
            if parent not in roots:
                roots.append(parent)

    for env_name in _USER_DB_BASE_DIR_ALLOWED_ROOT_ENVS:
        raw = os.getenv(env_name)
        for entry in _split_allowed_roots(raw or ""):
            allowed_root = _normalize_root_path(entry, project_root=project_root)
            if allowed_root is None:
                continue
            if allowed_root not in roots:
                roots.append(allowed_root)

    return roots


def _validate_user_db_base_dir_update(parser: ConfigParser, new_value: Any) -> None:
    project_root = Path(get_project_root()).resolve()
    candidate = _normalize_user_db_base_dir_candidate(new_value, project_root=project_root)
    allowed_roots = _collect_allowed_user_db_base_roots(parser, project_root=project_root)
    if allowed_roots and not any(candidate.is_relative_to(root) for root in allowed_roots):
        allowed_str = ", ".join(str(root) for root in allowed_roots)
        raise ValueError(
            "USER_DB_BASE_DIR must be under one of the allowed roots. "
            f"Allowed roots: {allowed_str}. "
            "Set USER_DB_BASE_DIR_ALLOWED_ROOTS and/or "
            "TLDW_USER_DB_BASE_DIR_ALLOWED_ROOTS to allow additional locations."
        )


def _validate_ingestion_source_allowed_roots_update(new_value: Any) -> None:
    project_root = Path(get_project_root()).resolve()
    for entry in _split_allowed_roots(str(new_value or "")):
        normalized = _normalize_root_path(entry, project_root=project_root)
        if normalized is None:
            raise ValueError("Ingestion source allowed roots must not contain empty entries")


def validate_ingestion_source_allowed_roots(value: Any) -> None:
    """Validate a proposed local-ingestion roots value without writing config."""
    _validate_ingestion_source_allowed_roots_update(value)


def _setup_provider_catalog_update_fields() -> set[tuple[str, str]]:
    """Return setup provider catalog fields that may be added to existing sections."""
    global _provider_catalog_update_fields
    if _provider_catalog_update_fields is not None:
        return _provider_catalog_update_fields

    from tldw_Server_API.app.core.Setup.provider_catalog import get_setup_provider_catalog

    fields: set[tuple[str, str]] = set()
    for entry in get_setup_provider_catalog().providers:
        for key in (entry.api_key_field, entry.base_url_field, entry.model_field):
            if key:
                fields.add((entry.config_section, key))
    _provider_catalog_update_fields = fields
    return fields


def _is_setup_provider_catalog_update(section: str, key: str) -> bool:
    """Return True when the field belongs to the first-run provider catalog."""
    return (section, key) in _setup_provider_catalog_update_fields()


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in SENSITIVE_KEY_MARKERS)


def _is_placeholder(value: str) -> bool:
    return value.strip() in PLACEHOLDER_VALUES


def get_setup_flags(config: ConfigParser | None = None) -> dict[str, bool]:
    """Return the setup enablement and completion flags."""
    parser = config or _load_config_parser()

    enabled = parser.getboolean(SETUP_SECTION, "enable_first_time_setup", fallback=False)
    completed = parser.getboolean(SETUP_SECTION, "setup_completed", fallback=False)

    return {
        "enabled": enabled,
        "completed": completed,
        "needs_setup": enabled and not completed,
    }


def needs_setup() -> bool:
    """Return ``True`` when the setup screen should be displayed."""
    return get_setup_flags()["needs_setup"]


def get_status_snapshot() -> dict[str, Any]:
    """Return high-level status information for the setup API."""
    parser = _load_config_parser()
    flags = get_setup_flags(parser)

    remote_flag = parser.getboolean(SETUP_SECTION, REMOTE_ACCESS_FIELD, fallback=False)
    env_override = env_flag_enabled("TLDW_SETUP_ALLOW_REMOTE")
    remote_active = remote_flag or env_override

    placeholder_fields: list[dict[str, str]] = []
    for section in parser.sections():
        for key, value in parser.items(section):
            if _is_placeholder(value):
                placeholder_fields.append({
                    "section": section,
                    "key": key,
                    "value": value,
                })

    return {
        "enabled": flags["enabled"],
        "setup_completed": flags["completed"],
        "needs_setup": flags["needs_setup"],
        "config_path": str(get_config_file_path()),
        "allow_remote_setup_access": remote_flag,
        "remote_access_env_override": env_override,
        "remote_access_active": remote_active,
        "placeholder_fields": placeholder_fields,
    }


def get_config_snapshot() -> dict[str, Any]:
    """Return the raw configuration structured for the setup UI."""
    parser = _load_config_parser()
    field_comments, section_comments = _build_comment_index()
    sections: list[dict[str, Any]] = []

    for section in parser.sections():
        entries = []
        for key, value in parser.items(section):
            field_hint = FIELD_HINTS.get((section, key))
            if not field_hint:
                field_hint = field_comments.get((section, key))
            if not field_hint:
                field_hint = section_comments.get(section)
            if not field_hint:
                field_hint = _generate_default_hint(section, key)

            is_secret = _is_sensitive_key(key)
            # Determine placeholder against the real value before masking
            placeholder_flag = _is_placeholder(value)

            # Never return secret values to the client. Indicate if it is set without exposing it.
            presented_value = "" if is_secret else value
            presented_type = "string" if is_secret else _infer_type(value)

            entry: dict[str, Any] = {
                "key": key,
                "value": presented_value,
                "type": presented_type,
                "is_secret": is_secret,
                "placeholder": placeholder_flag,
                "hint": field_hint,
            }
            # Extra hint for clients if a secret exists but is intentionally masked
            if is_secret:
                entry["is_set"] = bool(str(value).strip()) and not placeholder_flag

            entries.append(entry)

        sections.append({
            "name": section,
            "label": SECTION_LABELS.get(section, section.replace("_", " ").replace("-", " ").title()),
            "description": SECTION_DESCRIPTIONS.get(section) or section_comments.get(section),
            "fields": entries,
        })

    return {
        "config_path": str(get_config_file_path()),
        "sections": sections,
    }


def update_config(updates: dict[str, dict[str, Any]], *, create_backup: bool = True) -> Path | None:
    """Apply updates to the configuration file.

    Args:
        updates: Mapping of section -> key/value pairs to persist.
        create_backup: When True, write a timestamped ``.bak`` file alongside the config.

    Returns:
        The backup path if created, otherwise ``None``.
    """
    if not updates:
        raise ValueError("No updates provided for configuration")

    parser = _load_config_parser()
    config_path = get_config_file_path()

    # Validate sections/keys and types against existing config
    _validate_updates(parser, updates)

    # Stage updates into the in-memory parser so that downstream hooks can read booleans
    for section, items in updates.items():
        if not parser.has_section(section):
            parser.add_section(section)
        for key, value in items.items():
            parser.set(section, key, _coerce_to_string(value))

    backup_path = None
    if create_backup:
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        backup_path = config_path.with_suffix(config_path.suffix + f".pre-setup-{timestamp}.bak")
        shutil.copy2(config_path, backup_path)
        logger.info(f"Created backup of config.txt at {backup_path}")

    # Write changes back while preserving comments and unrelated formatting
    _write_config_preserving_comments(config_path, updates)

    logger.info("Configuration file updated via setup manager (comments preserved)")

    if (
        _remote_access_hook
        and SETUP_SECTION in updates
        and REMOTE_ACCESS_FIELD in updates[SETUP_SECTION]
    ):
        try:
            new_value = parser.getboolean(SETUP_SECTION, REMOTE_ACCESS_FIELD, fallback=False)
            _remote_access_hook(new_value)
        except Exception:  # noqa: BLE001
            logger.exception("Failed to propagate remote setup access change")

    return backup_path


def _write_config_preserving_comments(config_path: Path, updates: dict[str, dict[str, Any]]) -> None:
    """Write updated key-values while preserving comments and unrelated formatting.

    Strategy:
      - Read original lines as text.
      - Track current section.
      - For each line containing a key in the active section, replace the value portion
        before any inline comment marker ('#' or ';').
      - Leave comments and spacing outside the key/value token intact where reasonable.
    """
    try:
        with config_path.open("r", encoding="utf-8", newline="") as handle:
            original_lines = handle.read().splitlines(keepends=True)
    except FileNotFoundError:
        raise
    default_line_ending = "\r\n" if any(line.endswith("\r\n") for line in original_lines) else "\n"

    # Prepare a mutable copy of updates: section -> key -> str(value)
    pending: dict[str, dict[str, str]] = {
        section: {k: validate_config_value_single_line(section, k, v) for k, v in items.items()}
        for section, items in updates.items()
    }

    current_section: str | None = None
    out_lines: list[str] = []

    def _append_pending_items_for_section(section: str | None) -> None:
        if not section:
            return
        items = pending.get(section, {})
        if not items:
            return
        for key, value in list(items.items()):
            out_lines.append(f"{key} = {value}\n")
            items.pop(key)

    for raw in original_lines:
        line = raw
        stripped = line.strip()

        # Section header detection: [Section]
        if stripped.startswith("[") and stripped.endswith("]") and len(stripped) >= 2:
            _append_pending_items_for_section(current_section)
            current_section = stripped[1:-1].strip()
            out_lines.append(line)
            continue

        # If in a section that has updates, try to match a key line
        items = pending.get(current_section or "", {})
        if items and "=" in line:
            # Separate inline comment if present (# or ;) taking the earliest marker
            hash_idx = line.find("#") if "#" in line else len(line) + 1
            semi_idx = line.find(";") if ";" in line else len(line) + 1
            cut_idx = min(hash_idx, semi_idx)

            code_part = line[:cut_idx]
            comment_part = line[cut_idx:] if cut_idx < len(line) else ""

            if "=" in code_part:
                left, right = code_part.split("=", 1)
                key = left.strip()
                if key in items:
                    # Preserve leading indentation from the original 'left'
                    leading_ws_len = len(left) - len(left.lstrip(" \t"))
                    leading_ws = left[:leading_ws_len]
                    new_value = items.pop(key)
                    line_ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
                    # Build normalized key/value token; keep one space around '='
                    new_code = f"{leading_ws}{key} = {new_value}"
                    # Ensure a space before inline comment if it exists and doesn't already start with whitespace
                    if comment_part and not comment_part.startswith((" ", "\t")):
                        comment_part = " " + comment_part
                    out_lines.append(new_code + comment_part + ("" if comment_part else line_ending))
                    continue

        out_lines.append(line)

    _append_pending_items_for_section(current_section)

    # Sanity: Ensure all keys were applied; if not, create missing sections at EOF.
    leftovers = sum(len(items) for items in pending.values())
    if leftovers:
        logger.warning("Some config updates targeted missing sections; appending {} keys at EOF.", leftovers)
        for section, items in pending.items():
            if not items:
                continue
            if out_lines and out_lines[-1].strip():
                out_lines.append(default_line_ending)
            out_lines.append(f"[{section}]{default_line_ending}")
            for key, value in items.items():
                out_lines.append(f"{key} = {value}{default_line_ending}")

    with config_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("".join(out_lines))


def _validate_updates(parser: ConfigParser, updates: dict[str, dict[str, Any]]) -> None:
    """Validate sections, keys, and basic types for setup updates.

    Rules:
      - Section must already exist in config.
      - Key must already exist in the section.
      - Type of new value must match the inferred type of the current value when
        the current value is boolean/integer/number. String values accept any.
    """
    for section, items in updates.items():
        section_allows_provider_catalog_fields = bool(items) and all(
            _is_setup_provider_catalog_update(section, key) for key in items
        )
        if not parser.has_section(section) and not section_allows_provider_catalog_fields:
            raise ValueError(f"Unknown section '{section}' in updates")
        for key, new_value in items.items():
            serialized_value = validate_config_value_single_line(section, key, new_value)
            allows_new_ingestion_roots = (
                section == _INGESTION_SOURCE_ALLOWED_ROOTS_SECTION
                and key == _INGESTION_SOURCE_ALLOWED_ROOTS_KEY
            )
            allows_provider_catalog_field = _is_setup_provider_catalog_update(section, key)
            if (
                not parser.has_option(section, key)
                and not allows_new_ingestion_roots
                and not allows_provider_catalog_field
            ):
                raise ValueError(f"Unknown key '{key}' in section '{section}'")

            if section == _USER_DB_BASE_DIR_SECTION and key == _USER_DB_BASE_DIR_KEY:
                _validate_user_db_base_dir_update(parser, new_value)
            if section == _INGESTION_SOURCE_ALLOWED_ROOTS_SECTION and key == _INGESTION_SOURCE_ALLOWED_ROOTS_KEY:
                _validate_ingestion_source_allowed_roots_update(new_value)

            if (allows_new_ingestion_roots or allows_provider_catalog_field) and not parser.has_option(section, key):
                continue

            current_value = parser.get(section, key, fallback="")
            expected_type = _infer_type(current_value)
            if serialized_value.strip() == "" and (section, key) in _OPTIONAL_EMPTY_VALUE_FIELDS:
                continue

            # Accept any string when expected type is string
            if expected_type == "string":
                continue

            raw = serialized_value
            if expected_type == "boolean":
                lowered = raw.strip().lower()
                if lowered not in {"true", "false", "yes", "no", "on", "off", "1", "0"}:
                    raise ValueError(
                        f"Invalid boolean for {section}.{key}: '{new_value}'. Use true/false or on/off/1/0."
                    )
            elif expected_type == "integer":
                try:
                    int(raw)
                except Exception:  # noqa: BLE001
                    raise ValueError(f"Invalid integer for {section}.{key}: '{new_value}'") from None
            elif expected_type == "number":
                try:
                    float(raw)
                except Exception:  # noqa: BLE001
                    raise ValueError(f"Invalid number for {section}.{key}: '{new_value}'") from None


def mark_setup_completed(completed: bool = True) -> None:
    """Set the ``setup_completed`` flag in ``config.txt``."""
    update_config({SETUP_SECTION: {"setup_completed": completed}}, create_backup=False)


def reset_setup_flags() -> None:
    """Reset setup flags to their default values."""
    update_config({SETUP_SECTION: {
        "enable_first_time_setup": True,
        "setup_completed": False,
    }})
    logger.info("Setup flags reset to defaults")


def answer_setup_question(question: str, *, limit: int = 4) -> dict[str, Any]:
    """Return contextual guidance for setup questions without requiring an external LLM."""
    query = (question or '').strip()
    if not query:
        raise ValueError("Question must not be empty")

    snapshot = get_config_snapshot()
    sections = snapshot.get("sections", [])
    query_lower = _normalise_text(query)
    query_tokens = _tokenise(query)

    catalogue: list[dict[str, Any]] = []
    for section in sections:
        section_name = section.get("name", "")
        section_label = section.get("label", section_name)
        section_description = section.get("description") or ''

        catalogue.append({
            "type": "section",
            "section": section_name,
            "section_label": section_label,
            "label": section_label,
            "description": section_description,
            "hint": section_description,
        })

        for field in section.get("fields", []):
            catalogue.append({
                "type": "field",
                "section": section_name,
                "section_label": section_label,
                "key": field.get("key"),
                "label": _humanise(field.get("key", "")) or field.get("key", ""),
                "description": section_description,
                "hint": field.get("hint", ""),
            })

    scored_entries: list[dict[str, Any]] = []
    for entry in catalogue:
        label = entry.get("label", "")
        hint = entry.get("hint", "")
        description = entry.get("description", "")
        extras = ' '.join(filter(None, [entry.get("key"), entry.get("section"), hint]))
        score = _score_entry(query_lower, query_tokens, label, hint, description, extras)

        if entry.get("key") and entry["key"].lower() in query_lower:
            score = min(score + 0.2, 1.0)
        if entry.get("section") and entry["section"].lower() in query_lower:
            score = min(score + 0.1, 1.0)

        if score >= 0.15:
            entry_with_score = dict(entry)
            entry_with_score["score"] = round(score, 3)
            scored_entries.append(entry_with_score)

    scored_entries.sort(key=lambda item: item["score"], reverse=True)
    top_matches = scored_entries[:limit] if scored_entries else []

    if top_matches:
        primary = top_matches[0]
        section_label = primary.get("section_label", "the configuration")
        if primary.get("type") == "field":
            field_label = primary.get("label", "This setting")
            hint = primary.get("hint", "")
            answer_lines = [
                f"{field_label} lives in the {section_label} section.",
                hint or "Update this value from the setup page to tailor the behaviour.",
            ]
        else:
            hint = primary.get("hint", "") or primary.get("description", "")
            answer_lines = [
                f"The {section_label} section groups related configuration options.",
            ]
            if hint:
                answer_lines.append(hint)

        if len(top_matches) > 1:
            answer_lines.append("\nOther related entries:")
            for sibling in top_matches[1:]:
                sibling_label = sibling.get("label") or sibling.get("key") or sibling.get("section_label")
                sibling_section = sibling.get("section_label")
                answer_lines.append(f"- {sibling_label} ({sibling_section})")

        answer_text = '\n'.join(answer_lines).strip()
    else:
        answer_text = (
            "I couldn't link that question to a specific setting yet. Try mentioning the section or exact "
            "setting name (for example, 'AuthNZ single_user_api_key')."
        )

    return {
        "answer": answer_text,
        "matches": top_matches,
    }
