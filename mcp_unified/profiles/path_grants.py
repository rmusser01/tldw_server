"""Authoring helpers for MCP workspace-relative path grants."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

PATH_GRANT_ACTIONS = frozenset({"read", "edit", "write"})
PATH_GRANT_EFFECTS = frozenset({"allow", "deny"})
PATH_GRANT_AUTHORING_KEYS = (
    "path_grant_authoring",
    "path_grant_hierarchy",
    "hierarchical_path_grants",
)

_WINDOWS_DRIVE_PREFIX_RE = re.compile(r"^[A-Za-z]:")
_AUTHORING_LEVELS = (
    ("org", "org"),
    ("workspace", "workspace"),
    ("folders", "folder"),
    ("folder", "folder"),
    ("files", "file"),
    ("file", "file"),
)


@dataclass(frozen=True, slots=True)
class PathGrantDiagnostic:
    """Validation diagnostic emitted while compiling authored path grants."""

    code: str
    message: str
    source: str
    severity: str = "error"

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-serializable diagnostic payload."""

        return {
            "code": self.code,
            "message": self.message,
            "source": self.source,
            "severity": self.severity,
        }


@dataclass(frozen=True, slots=True)
class PathGrantCompilationResult:
    """Flat grant compilation result plus operator-facing preview metadata."""

    path_grants: list[dict[str, Any]]
    preview: list[dict[str, Any]]
    diagnostics: list[dict[str, str]]

    @property
    def has_errors(self) -> bool:
        """Return whether any error-severity diagnostics were emitted."""

        return any(item.get("severity") == "error" for item in self.diagnostics)


def _diagnostic(code: str, message: str, source: str) -> PathGrantDiagnostic:
    """Build a standard error diagnostic."""

    return PathGrantDiagnostic(code=code, message=message, source=source)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Return a mapping view for object-like values."""

    return value if isinstance(value, Mapping) else {}


def _as_rule_list(value: Any) -> list[Mapping[str, Any]]:
    """Normalize a raw authored-rule collection into mapping entries."""

    if isinstance(value, Mapping):
        return [value]
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _as_action_list(value: Any) -> list[str]:
    """Normalize an action value into sorted unique valid action names."""

    if isinstance(value, str):
        raw_items: Iterable[Any] = [value]
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray, Mapping)):
        raw_items = value
    else:
        raw_items = []
    return sorted({str(item or "").strip().lower() for item in raw_items if str(item or "").strip()})


def _normalize_path_prefix(raw_value: Any) -> str | None:
    """Normalize a workspace-relative path-grant prefix."""

    value = str(raw_value or "").strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    value = re.sub(r"/+", "/", value)
    if not value:
        return None
    if value == ".":
        return "."
    if value.startswith("/") or _WINDOWS_DRIVE_PREFIX_RE.match(value):
        return None

    parts: list[str] = []
    for part in value.split("/"):
        cleaned = str(part or "").strip()
        if not cleaned or cleaned == ".":
            continue
        if cleaned == "..":
            return None
        parts.append(cleaned)
    if not parts:
        return "."
    return "/".join(parts)


def _iter_authored_rules(authoring_document: Any) -> Iterable[tuple[Mapping[str, Any], str, str]]:
    """Yield authored grant rules with source and hierarchy level metadata."""

    document = _as_mapping(authoring_document)
    if not document and isinstance(authoring_document, list):
        document = {"rules": authoring_document}

    for key, level in _AUTHORING_LEVELS:
        for index, rule in enumerate(_as_rule_list(document.get(key))):
            yield rule, f"{key}[{index}]", level

    for index, rule in enumerate(_as_rule_list(document.get("rules"))):
        level = str(rule.get("level") or rule.get("scope") or "rule").strip().lower()
        if level not in {"org", "workspace", "folder", "file"}:
            level = "rule"
        yield rule, f"rules[{index}]", level


def _compile_rules(rules: Iterable[tuple[Mapping[str, Any], str, str]]) -> PathGrantCompilationResult:
    """Compile normalized rule tuples into flat path grants."""

    diagnostics: list[PathGrantDiagnostic] = []
    merged: dict[tuple[str, str], dict[str, Any]] = {}

    for rule, source, level in rules:
        prefix = _normalize_path_prefix(rule.get("prefix", rule.get("path")))
        if prefix is None:
            diagnostics.append(_diagnostic("invalid_prefix", "path grant prefix must be workspace-relative", source))
            continue

        actions = _as_action_list(rule.get("actions"))
        invalid_actions = [action for action in actions if action not in PATH_GRANT_ACTIONS]
        actions = [action for action in actions if action in PATH_GRANT_ACTIONS]
        if invalid_actions or not actions:
            diagnostics.append(_diagnostic("invalid_actions", "path grant actions must be read, edit, or write", source))
            continue

        effect = str(rule.get("effect") or "allow").strip().lower()
        if effect not in PATH_GRANT_EFFECTS:
            diagnostics.append(_diagnostic("invalid_effect", "path grant effect must be allow or deny", source))
            continue

        key = (prefix, effect)
        existing = merged.get(key)
        if existing is None:
            merged[key] = {
                "prefix": prefix,
                "actions": set(actions),
                "effect": effect,
                "source": source,
                "level": level,
            }
        else:
            existing["actions"].update(actions)

    items = sorted(merged.values(), key=lambda item: (str(item["prefix"]), str(item["effect"])))
    path_grants = [
        {
            "prefix": str(item["prefix"]),
            "actions": sorted(item["actions"]),
            "effect": str(item["effect"]),
        }
        for item in items
    ]
    preview = [
        {
            "prefix": str(item["prefix"]),
            "actions": sorted(item["actions"]),
            "effect": str(item["effect"]),
            "source": str(item["source"]),
            "level": str(item["level"]),
        }
        for item in items
    ]
    return PathGrantCompilationResult(
        path_grants=path_grants,
        preview=preview,
        diagnostics=[item.as_dict() for item in diagnostics],
    )


def compile_hierarchical_path_grants(authoring_document: Any) -> PathGrantCompilationResult:
    """Compile hierarchical authored path grants into flat runtime grants."""

    return _compile_rules(_iter_authored_rules(authoring_document))


def compile_policy_path_grants(policy_document: Mapping[str, Any]) -> PathGrantCompilationResult:
    """Compile flat or authored path grants from an MCP profile policy document."""

    document = _as_mapping(policy_document)
    if "path_grants" in document:
        rules = (
            (rule, f"path_grants[{index}]", "runtime")
            for index, rule in enumerate(_as_rule_list(document.get("path_grants")))
        )
        return _compile_rules(rules)

    for key in PATH_GRANT_AUTHORING_KEYS:
        if key in document:
            return compile_hierarchical_path_grants(document.get(key))

    return PathGrantCompilationResult(path_grants=[], preview=[], diagnostics=[])


def has_path_grant_policy(policy_document: Mapping[str, Any]) -> bool:
    """Return whether a policy document declares flat or authored path grants."""

    document = _as_mapping(policy_document)
    return "path_grants" in document or any(key in document for key in PATH_GRANT_AUTHORING_KEYS)
