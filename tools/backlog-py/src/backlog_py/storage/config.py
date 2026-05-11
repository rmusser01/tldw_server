from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

from backlog_py.core.models import BacklogConfig, BacklogProject
from backlog_py.security.paths import PathContainmentError, assert_path_within_base


_KEY_ALIASES = {
    "project_name": ("project_name", "projectName"),
    "statuses": ("statuses",),
    "default_status": ("default_status", "defaultStatus"),
    "remote_operations": ("remote_operations", "remoteOperations"),
    "auto_commit": ("auto_commit", "autoCommit"),
    "bypass_git_hooks": ("bypass_git_hooks", "bypassGitHooks"),
    "check_active_branches": ("check_active_branches", "checkActiveBranches"),
    "active_branch_days": ("active_branch_days", "activeBranchDays"),
    "definition_of_done": ("definition_of_done", "definitionOfDone"),
}


def load_config(path: Path) -> BacklogConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Backlog config must contain a mapping: {path}")

    return BacklogConfig(
        project_name=_string_value(raw, "project_name", _default_project_name(path)),
        statuses=_optional_string_list(_get(raw, "statuses", None)),
        default_status=_string_value(raw, "default_status", "To Do"),
        remote_operations=_bool_value(raw, "remote_operations", True),
        auto_commit=_bool_value(raw, "auto_commit", False),
        bypass_git_hooks=_bool_value(raw, "bypass_git_hooks", False),
        check_active_branches=_bool_value(raw, "check_active_branches", True),
        active_branch_days=_int_value(raw, "active_branch_days", 30),
        definition_of_done=_optional_string_list(_get(raw, "definition_of_done", None)),
    )


def get_definition_of_done_defaults(project: BacklogProject) -> list[str]:
    return list(load_config(project.config_path).definition_of_done or [])


def replace_definition_of_done_defaults(project: BacklogProject, items: list[str]) -> list[str]:
    normalized = [str(item) for item in items]
    raw = _load_raw_config(project.config_path)
    key = "definition_of_done" if "definition_of_done" in raw else "definitionOfDone"
    raw[key] = normalized
    yaml_text = yaml.safe_dump(raw, sort_keys=False, allow_unicode=False).strip()
    _atomic_write_text(project.config_path, f"{yaml_text}\n")
    return normalized


def _get(raw: dict[Any, Any], normalized_key: str, default: Any) -> Any:
    for key in _KEY_ALIASES[normalized_key]:
        if key in raw:
            return raw[key]
    return default


def _load_raw_config(path: Path) -> dict[Any, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Backlog config must contain a mapping: {path}")
    return raw


def _atomic_write_text(path: Path, content: str) -> None:
    try:
        safe_path = assert_path_within_base(path.parent, path)
    except PathContainmentError as exc:
        raise ValueError(str(exc)) from exc
    temp_name: str | None = None
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=safe_path.parent,
        prefix=f".{safe_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temp_file:
        temp_name = temp_file.name
        temp_file.write(content)
        temp_file.flush()
        os.fsync(temp_file.fileno())
    try:
        os.replace(temp_name, safe_path)
    except Exception:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
        raise


def _optional_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("Backlog config list values must be lists")
    return [str(item) for item in value]


def _string_value(raw: dict[Any, Any], normalized_key: str, default: str) -> str:
    value = _get(raw, normalized_key, default)
    if not isinstance(value, str):
        raise ValueError(f"Backlog config value {normalized_key} must be a string")
    return value


def _bool_value(raw: dict[Any, Any], normalized_key: str, default: bool) -> bool:
    value = _get(raw, normalized_key, default)
    if not isinstance(value, bool):
        raise ValueError(f"Backlog config value {normalized_key} must be a boolean")
    return value


def _int_value(raw: dict[Any, Any], normalized_key: str, default: int) -> int:
    value = _get(raw, normalized_key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Backlog config value {normalized_key} must be an integer")
    return value


def _default_project_name(path: Path) -> str:
    if path.name == "backlog.config.yml":
        return path.parent.name
    if path.parent.name in {"backlog", ".backlog"}:
        return path.parent.parent.name
    return path.parent.name
