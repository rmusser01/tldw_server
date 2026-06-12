"""Shared permission-rule subject extraction for runtime enforcement and simulation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .permission_rules import PermissionRuleSubject

MAX_PERMISSION_SUBJECTS = 128
MAX_SUBJECT_VALUE_LENGTH = 4096
MAX_COMMAND_ARGV_TOKENS = 256

PATH_ARGUMENT_KEYS = frozenset(
    {
        "path",
        "file",
        "file_path",
        "filepath",
        "filename",
        "base_path",
        "target_path",
        "source_path",
        "destination_path",
        "new_path",
        "old_path",
    }
)
PATH_ARGUMENT_LIST_KEYS = frozenset({"paths", "files", "file_paths"})
DOMAIN_ARGUMENT_KEYS = frozenset({"url", "uri", "domain", "host", "base_url"})
DOMAIN_ARGUMENT_LIST_KEYS = frozenset({"urls", "uris", "domains", "hosts"})
COMMAND_ARGUMENT_KEYS = frozenset({"command", "cmd", "shell_command"})
COMMAND_ARGV_KEYS = frozenset({"argv"})


class PermissionSubjectLimitError(Exception):
    """Raised when one tool call exceeds permission-subject extraction limits."""

    def __init__(self, limit: str) -> None:
        super().__init__(limit)
        self.limit = limit


def extract_permission_rule_subjects(
    tool_name: str,
    arguments: dict[str, Any],
) -> list[tuple[PermissionRuleSubject, str, Sequence[str] | None]]:
    """Extract bounded permission-rule subjects from one gateway tool call."""

    subjects: list[tuple[PermissionRuleSubject, str, Sequence[str] | None]] = []
    _append_permission_subject(subjects, "tool", tool_name, None)
    if tool_name.lower().startswith("mcp__"):
        _append_permission_subject(subjects, "mcp", tool_name, None)
    if not isinstance(arguments, Mapping):
        return subjects

    for key, value in arguments.items():
        if not isinstance(key, str):
            continue
        normalized_key = key.strip().lower()
        if normalized_key in PATH_ARGUMENT_KEYS:
            for item in _string_values(value):
                _append_permission_subject(subjects, "path", item, None)
        elif normalized_key in PATH_ARGUMENT_LIST_KEYS:
            for item in _nested_string_values(value):
                _append_permission_subject(subjects, "path", item, None)
        elif normalized_key in DOMAIN_ARGUMENT_KEYS:
            for item in _string_values(value):
                _append_permission_subject(subjects, "domain", item, None)
        elif normalized_key in DOMAIN_ARGUMENT_LIST_KEYS:
            for item in _nested_string_values(value):
                _append_permission_subject(subjects, "domain", item, None)
        elif normalized_key in COMMAND_ARGUMENT_KEYS:
            for item in _string_values(value):
                _append_permission_subject(subjects, "command", item, None)
        elif normalized_key in COMMAND_ARGV_KEYS:
            argv = _argv_argument(value)
            if argv is not None:
                if len(argv) > MAX_COMMAND_ARGV_TOKENS:
                    raise PermissionSubjectLimitError("max_command_argv_tokens")
                _append_permission_subject(subjects, "command", " ".join(argv), argv)
    return subjects


def _append_permission_subject(
    subjects: list[tuple[PermissionRuleSubject, str, Sequence[str] | None]],
    subject_type: PermissionRuleSubject,
    value: str,
    argv: Sequence[str] | None,
) -> None:
    """Append one extracted subject while enforcing extraction limits."""

    if len(subjects) >= MAX_PERMISSION_SUBJECTS:
        raise PermissionSubjectLimitError("max_permission_subjects")
    if len(value) > MAX_SUBJECT_VALUE_LENGTH:
        raise PermissionSubjectLimitError("max_subject_value_length")
    subjects.append((subject_type, value, argv))


def _string_values(value: Any) -> tuple[str, ...]:
    """Return one non-empty string value or an empty tuple."""

    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    return ()


def _nested_string_values(value: Any) -> tuple[str, ...]:
    """Return non-empty string values from a scalar or shallow sequence."""

    if isinstance(value, (str, bytes, Mapping)):
        return _string_values(value)
    if isinstance(value, Sequence):
        return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    return ()


def _argv_argument(value: Any) -> tuple[str, ...] | None:
    """Return argv tokens when a tool call provides an argv-shaped argument."""

    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        return None
    argv = tuple(value)
    if not argv or not all(isinstance(item, str) for item in argv):
        return None
    return argv


__all__ = [
    "COMMAND_ARGUMENT_KEYS",
    "COMMAND_ARGV_KEYS",
    "DOMAIN_ARGUMENT_KEYS",
    "DOMAIN_ARGUMENT_LIST_KEYS",
    "MAX_COMMAND_ARGV_TOKENS",
    "MAX_PERMISSION_SUBJECTS",
    "MAX_SUBJECT_VALUE_LENGTH",
    "PATH_ARGUMENT_KEYS",
    "PATH_ARGUMENT_LIST_KEYS",
    "PermissionSubjectLimitError",
    "extract_permission_rule_subjects",
]
