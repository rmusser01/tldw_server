"""JavaScript and TypeScript import resolution helpers for CodeGraph."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SOURCE_EXTENSIONS = (
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".mts",
    ".cts",
    ".mjs",
    ".cjs",
    ".json",
)


@dataclass(frozen=True)
class JsTsProjectConfig:
    """Nearest JS/TS project config relevant to a source file."""

    config_path: str
    base_url: str
    paths: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class ImportResolution:
    """Resolved or classified JS/TS import specifier."""

    specifier: str
    resolution_kind: str
    resolved_path: str | None = None
    reason: str | None = None
    candidates: tuple[str, ...] = ()


def load_js_ts_project_config(workspace_root: Path, source_path: str | Path) -> JsTsProjectConfig | None:
    """Load the nearest tsconfig.json or jsconfig.json for a workspace-relative file."""
    workspace = workspace_root.resolve()
    source_abs = _resolve_under_workspace(workspace, source_path)
    if source_abs is None:
        return None

    current = source_abs.parent
    while current == workspace or workspace in current.parents:
        for config_name in ("tsconfig.json", "jsconfig.json"):
            config_path = current / config_name
            if config_path.is_file():
                return _read_project_config(workspace, config_path)
        if current == workspace:
            break
        current = current.parent
    return None


def resolve_js_ts_import(workspace_root: Path, source_path: str | Path, specifier: str) -> ImportResolution:
    """Resolve a JS/TS import specifier without following dependencies outside the workspace."""
    workspace = workspace_root.resolve()
    source_abs = _resolve_under_workspace(workspace, source_path)
    if source_abs is None:
        return ImportResolution(specifier=specifier, resolution_kind="unresolved", reason="source_outside_workspace")

    config = load_js_ts_project_config(workspace, source_abs)
    return resolve_js_ts_import_with_config(workspace, source_abs, specifier, config)


def resolve_js_ts_import_with_config(
    workspace_root: Path,
    source_path: str | Path,
    specifier: str,
    config: JsTsProjectConfig | None,
) -> ImportResolution:
    """Resolve an import using a caller-supplied project config cache entry."""
    workspace = workspace_root.resolve()
    source_abs = _resolve_under_workspace(workspace, source_path)
    if source_abs is None:
        return ImportResolution(specifier=specifier, resolution_kind="unresolved", reason="source_outside_workspace")

    if specifier.startswith(("./", "../")):
        return resolve_relative_import(workspace, source_abs, specifier)

    if config is not None:
        alias_result = resolve_path_alias_import(workspace, config, specifier)
        if alias_result is not None:
            return alias_result

    return ImportResolution(specifier=specifier, resolution_kind="external", reason="external_package")


def resolve_relative_import(workspace_root: Path, source_abs: Path, specifier: str) -> ImportResolution:
    """Resolve a relative import from one source file."""
    candidate_base = (source_abs.parent / specifier).resolve()
    if not _is_under_workspace(workspace_root, candidate_base):
        return ImportResolution(specifier=specifier, resolution_kind="unresolved", reason="relative_target_escapes_workspace")

    resolved = _resolve_source_candidate(workspace_root, candidate_base)
    if resolved is not None:
        return ImportResolution(
            specifier=specifier,
            resolution_kind="relative",
            resolved_path=_workspace_relative(workspace_root, resolved),
        )

    return ImportResolution(
        specifier=specifier,
        resolution_kind="unresolved",
        reason="not_found",
        candidates=_candidate_strings(workspace_root, candidate_base),
    )


def resolve_path_alias_import(
    workspace_root: Path,
    config: JsTsProjectConfig,
    specifier: str,
) -> ImportResolution | None:
    """Resolve a specifier through tsconfig/jsconfig paths, if it matches one."""
    matches = list(_matching_path_aliases(config.paths, specifier))
    if not matches:
        return None

    base_abs = (workspace_root / config.base_url).resolve()
    attempted: list[str] = []
    saw_escape = False
    for target_pattern, wildcard_value in matches:
        target = target_pattern.replace("*", wildcard_value) if wildcard_value is not None else target_pattern
        candidate_base = (base_abs / target).resolve()
        if not _is_under_workspace(workspace_root, candidate_base):
            saw_escape = True
            continue
        resolved = _resolve_source_candidate(workspace_root, candidate_base)
        if resolved is not None:
            return ImportResolution(
                specifier=specifier,
                resolution_kind="alias",
                resolved_path=_workspace_relative(workspace_root, resolved),
            )
        attempted.extend(_candidate_strings(workspace_root, candidate_base))

    if saw_escape and not attempted:
        return ImportResolution(
            specifier=specifier,
            resolution_kind="unresolved",
            reason="alias_target_escapes_workspace",
        )

    return ImportResolution(
        specifier=specifier,
        resolution_kind="unresolved",
        reason="not_found",
        candidates=tuple(dict.fromkeys(attempted)),
    )


def _read_project_config(workspace_root: Path, config_path: Path) -> JsTsProjectConfig:
    """Read a tsconfig/jsconfig file and normalize the subset needed for import aliases."""
    raw = config_path.read_text(encoding="utf-8")
    try:
        parsed: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        try:
            parsed = json.loads(_strip_jsonc_comments(raw))
        except json.JSONDecodeError:
            parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}

    compiler_options = parsed.get("compilerOptions", {})
    if not isinstance(compiler_options, dict):
        compiler_options = {}

    config_dir = config_path.parent
    base_url_value = compiler_options.get("baseUrl", ".")
    base_url = str(base_url_value) if isinstance(base_url_value, str) else "."
    base_abs = (config_dir / base_url).resolve()
    if not _is_under_workspace(workspace_root, base_abs):
        base_abs = config_dir.resolve()

    paths_value = compiler_options.get("paths", {})
    paths: dict[str, tuple[str, ...]] = {}
    if isinstance(paths_value, dict):
        for pattern, targets in paths_value.items():
            if not isinstance(pattern, str):
                continue
            if isinstance(targets, str):
                paths[pattern] = (targets,)
            elif isinstance(targets, list):
                paths[pattern] = tuple(target for target in targets if isinstance(target, str))

    return JsTsProjectConfig(
        config_path=_workspace_relative(workspace_root, config_path.resolve()),
        base_url=_workspace_relative(workspace_root, base_abs),
        paths=paths,
    )


def _matching_path_aliases(
    paths: dict[str, tuple[str, ...]],
    specifier: str,
) -> tuple[tuple[str, str | None], ...]:
    """Return alias targets whose pattern matches a specifier, longest prefix first."""
    matches: list[tuple[int, tuple[str, str | None]]] = []
    for pattern, targets in paths.items():
        wildcard_value = _match_alias_pattern(pattern, specifier)
        if wildcard_value is None and pattern != specifier:
            continue
        static_prefix = pattern.split("*", 1)[0]
        for target in targets:
            matches.append((len(static_prefix), (target, wildcard_value)))
    return tuple(item for _, item in sorted(matches, key=lambda entry: entry[0], reverse=True))


def _match_alias_pattern(pattern: str, specifier: str) -> str | None:
    """Return the wildcard text for a matching paths pattern, or None when unmatched."""
    if "*" not in pattern:
        return "" if pattern == specifier else None
    prefix, suffix = pattern.split("*", 1)
    if not specifier.startswith(prefix) or (suffix and not specifier.endswith(suffix)):
        return None
    end = len(specifier) - len(suffix) if suffix else len(specifier)
    return specifier[len(prefix) : end]


def _resolve_source_candidate(workspace_root: Path, candidate_base: Path) -> Path | None:
    """Resolve a file or index-file candidate under the workspace, if it exists."""
    candidates: list[Path]
    if candidate_base.suffix:
        candidates = [candidate_base]
    else:
        candidates = [candidate_base.with_suffix(extension) for extension in _SOURCE_EXTENSIONS]
        candidates.extend(candidate_base / f"index{extension}" for extension in _SOURCE_EXTENSIONS)

    for candidate in candidates:
        resolved = candidate.resolve()
        if _is_under_workspace(workspace_root, resolved) and resolved.is_file():
            return resolved
    return None


def _candidate_strings(workspace_root: Path, candidate_base: Path) -> tuple[str, ...]:
    """Return workspace-relative candidate paths considered for a missing import."""
    if not _is_under_workspace(workspace_root, candidate_base):
        return ()
    if candidate_base.suffix:
        return (_workspace_relative(workspace_root, candidate_base),)
    candidates = [candidate_base.with_suffix(extension) for extension in _SOURCE_EXTENSIONS]
    candidates.extend(candidate_base / f"index{extension}" for extension in _SOURCE_EXTENSIONS)
    return tuple(_workspace_relative(workspace_root, candidate.resolve()) for candidate in candidates)


def _resolve_under_workspace(workspace_root: Path, path: str | Path) -> Path | None:
    """Resolve a path only when the result remains inside the workspace root."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = workspace_root / candidate
    resolved = candidate.resolve()
    if not _is_under_workspace(workspace_root, resolved):
        return None
    return resolved


def _is_under_workspace(workspace_root: Path, path: Path) -> bool:
    """Return whether path is the workspace root or a descendant of it."""
    resolved_workspace = workspace_root.resolve()
    resolved_path = path.resolve()
    return resolved_path == resolved_workspace or resolved_workspace in resolved_path.parents


def _workspace_relative(workspace_root: Path, path: Path) -> str:
    """Render a resolved path as a POSIX workspace-relative path."""
    return path.resolve().relative_to(workspace_root.resolve()).as_posix()


def _strip_jsonc_comments(text: str) -> str:
    """Remove JSONC line and block comments while preserving string literal contents."""
    result: list[str] = []
    in_string = False
    escape = False
    index = 0
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue
        if char == "/" and next_char == "/":
            index = text.find("\n", index)
            if index == -1:
                break
            result.append("\n")
            index += 1
            continue
        if char == "/" and next_char == "*":
            end = text.find("*/", index + 2)
            index = len(text) if end == -1 else end + 2
            continue
        result.append(char)
        index += 1
    return "".join(result)


__all__ = [
    "ImportResolution",
    "JsTsProjectConfig",
    "load_js_ts_project_config",
    "resolve_js_ts_import",
    "resolve_js_ts_import_with_config",
    "resolve_path_alias_import",
    "resolve_relative_import",
]
