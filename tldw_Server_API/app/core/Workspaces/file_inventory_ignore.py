from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any, Literal

from tldw_Server_API.app.core.Workspaces.file_inventory_models import bounded_inventory_diagnostics

MAX_GITIGNORE_BYTES = 64 * 1024
IGNORE_POLICY_VERSION = "workspace-file-inventory-ignore-v1"

BUILTIN_GENERATED_DIRS: tuple[str, ...] = (
    ".git",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".next",
    ".turbo",
    "dist",
    "build",
    "coverage",
    "target",
)
BUILTIN_SECRET_FILE_PATTERNS: tuple[str, ...] = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa",
    "id_ed25519",
    ".netrc",
)

IgnoreRuleSource = Literal["builtin", "workspace", "gitignore"]


@dataclass(frozen=True)
class InventoryIgnoreDecision:
    ignored: bool
    reason: str | None = None


@dataclass(frozen=True)
class InventoryIgnoreRule:
    source: IgnoreRuleSource
    raw_pattern: str
    match_pattern: str
    reason: str
    anchored: bool = False
    directory_only: bool = False
    unsupported: bool = False


@dataclass(frozen=True)
class InventoryIgnorePolicy:
    rules: tuple[InventoryIgnoreRule, ...]
    diagnostics: tuple[dict[str, str], ...]
    fingerprint: str


def build_inventory_ignore_policy(
    *,
    workspace_patterns: Iterable[str] | None = None,
    gitignore_texts: Iterable[tuple[str, str]] | None = None,
) -> InventoryIgnorePolicy:
    rules: list[InventoryIgnoreRule] = []
    diagnostics: list[dict[str, str]] = []

    for pattern in workspace_patterns or ():
        parsed = _parse_ignore_pattern(str(pattern), source="workspace", diagnostics=diagnostics, path_hint="workspace")
        if parsed is not None:
            rules.append(parsed)

    for path_hint, text in gitignore_texts or ():
        source_hint = str(path_hint or ".gitignore")
        encoded = str(text or "").encode("utf-8", errors="replace")
        if len(encoded) > MAX_GITIGNORE_BYTES:
            diagnostics.append(
                _diagnostic(
                    "ignore_file_too_large",
                    source_hint,
                    f"Ignore file exceeded {MAX_GITIGNORE_BYTES} bytes and was skipped.",
                )
            )
            continue
        for raw_line in str(text or "").splitlines():
            parsed = _parse_ignore_pattern(
                raw_line,
                source="gitignore",
                diagnostics=diagnostics,
                path_hint=source_hint,
            )
            if parsed is not None:
                rules.append(parsed)

    rules.extend(_builtin_rules())
    bounded = tuple(bounded_inventory_diagnostics(diagnostics))
    return InventoryIgnorePolicy(
        rules=tuple(rules),
        diagnostics=bounded,
        fingerprint=_policy_fingerprint(rules),
    )


def should_ignore_inventory_path(
    relative_path: str,
    *,
    is_dir: bool,
    policy: InventoryIgnorePolicy,
) -> InventoryIgnoreDecision:
    normalized = _normalize_relative_path(relative_path)
    if normalized is None:
        return InventoryIgnoreDecision(True, "unsafe_relative_path")

    segments = normalized.split("/")
    for rule in policy.rules:
        if _rule_matches(rule, normalized, segments, is_dir=is_dir):
            return InventoryIgnoreDecision(True, rule.reason)
    return InventoryIgnoreDecision(False, None)


def _builtin_rules() -> tuple[InventoryIgnoreRule, ...]:
    rules: list[InventoryIgnoreRule] = []
    for dirname in BUILTIN_GENERATED_DIRS:
        rules.append(
            InventoryIgnoreRule(
                source="builtin",
                raw_pattern=dirname,
                match_pattern=dirname,
                reason=f"builtin:generated_dir:{dirname}",
                directory_only=True,
            )
        )
    for pattern in BUILTIN_SECRET_FILE_PATTERNS:
        rules.append(
            InventoryIgnoreRule(
                source="builtin",
                raw_pattern=pattern,
                match_pattern=pattern,
                reason=f"builtin:secret_file:{pattern}",
            )
        )
    return tuple(rules)


def _parse_ignore_pattern(
    raw_line: str,
    *,
    source: IgnoreRuleSource,
    diagnostics: list[dict[str, str]],
    path_hint: str,
) -> InventoryIgnoreRule | None:
    stripped = raw_line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if "\x00" in stripped:
        diagnostics.append(
            _diagnostic(
                "malformed_gitignore_pattern",
                path_hint,
                "Ignore pattern contained a null byte and was skipped.",
            )
        )
        return None

    unsupported = _is_unsupported_pattern(stripped)
    if unsupported:
        diagnostics.append(
            _diagnostic(
                "unsupported_gitignore_pattern",
                path_hint,
                f"Ignore pattern '{stripped}' uses unsupported syntax and will be treated conservatively.",
            )
        )

    raw_pattern = stripped
    match_source = stripped[1:] if stripped.startswith("!") else stripped
    directory_only = match_source.endswith("/")
    if directory_only:
        match_source = match_source.rstrip("/")
    anchored = match_source.startswith("/")
    match_pattern = match_source.lstrip("/")
    if not match_pattern:
        return None

    return InventoryIgnoreRule(
        source=source,
        raw_pattern=raw_pattern,
        match_pattern=match_pattern,
        reason=f"{source}:{raw_pattern}",
        anchored=anchored,
        directory_only=directory_only,
        unsupported=unsupported,
    )


def _rule_matches(
    rule: InventoryIgnoreRule,
    relative_path: str,
    segments: list[str],
    *,
    is_dir: bool,
) -> bool:
    pattern = rule.match_pattern
    if rule.directory_only:
        if rule.anchored:
            if relative_path == pattern:
                return is_dir
            return relative_path.startswith(f"{pattern}/")
        if "/" in pattern:
            if relative_path == pattern or relative_path.endswith(f"/{pattern}"):
                return is_dir
            return relative_path.startswith(f"{pattern}/") or f"/{pattern}/" in relative_path
        if is_dir and fnmatchcase(segments[-1], pattern):
            return True
        return any(fnmatchcase(segment, pattern) for segment in segments[:-1])

    if rule.anchored:
        return _path_or_parent_matches_pattern(relative_path, pattern)

    if "/" in pattern:
        return (
            _path_or_parent_matches_pattern(relative_path, pattern)
            or relative_path.endswith(f"/{pattern}")
        )

    basename = segments[-1]
    if fnmatchcase(basename, pattern):
        return True
    if is_dir and fnmatchcase(relative_path, pattern):
        return True
    return any(segment == pattern for segment in segments[:-1])


def _path_or_parent_matches_pattern(relative_path: str, pattern: str) -> bool:
    if fnmatchcase(relative_path, pattern) or relative_path == pattern:
        return True
    if relative_path.startswith(f"{pattern}/"):
        return True

    parent = relative_path
    while "/" in parent:
        parent = parent.rsplit("/", 1)[0]
        if fnmatchcase(parent, pattern) or parent == pattern:
            return True
    return False


def _is_unsupported_pattern(pattern: str) -> bool:
    return pattern.startswith("!") or "**" in pattern or "[" in pattern or "]" in pattern


def _normalize_relative_path(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    raw = value.strip().replace("\\", "/")
    if not raw or raw.startswith("/") or raw.startswith("~") or "\x00" in raw:
        return None
    parts: list[str] = []
    for part in raw.split("/"):
        if not part or part == ".":
            continue
        if part == "..":
            return None
        parts.append(part)
    if not parts:
        return None
    return "/".join(parts)


def _policy_fingerprint(rules: list[InventoryIgnoreRule]) -> str:
    payload = {
        "version": IGNORE_POLICY_VERSION,
        "rules": sorted(
            {
                (
                    rule.source,
                    rule.raw_pattern,
                    rule.match_pattern,
                    rule.anchored,
                    rule.directory_only,
                    rule.unsupported,
                )
                for rule in rules
            }
        ),
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _diagnostic(code: str, path_hint: str, message: str) -> dict[str, str]:
    return {"code": code, "path_hint": path_hint, "message": message}


__all__ = [
    "BUILTIN_GENERATED_DIRS",
    "BUILTIN_SECRET_FILE_PATTERNS",
    "IGNORE_POLICY_VERSION",
    "InventoryIgnoreDecision",
    "InventoryIgnorePolicy",
    "InventoryIgnoreRule",
    "MAX_GITIGNORE_BYTES",
    "build_inventory_ignore_policy",
    "should_ignore_inventory_path",
]
