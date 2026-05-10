from __future__ import annotations

import os
from pathlib import Path

from backlog_py.core.models import BacklogProject
from backlog_py.storage.config import load_config


def discover_project(cwd: Path, explicit_cwd: Path | None = None) -> BacklogProject:
    start = _effective_cwd(cwd, explicit_cwd).resolve()
    root, backlog_dir, config_path = _find_project_paths(start)

    return BacklogProject(
        root=root,
        backlog_dir=backlog_dir,
        config_path=config_path,
        config=load_config(config_path),
    )


def _effective_cwd(cwd: Path, explicit_cwd: Path | None) -> Path:
    if explicit_cwd is not None:
        return explicit_cwd

    env_cwd = os.environ.get("BACKLOG_CWD")
    if env_cwd:
        return Path(env_cwd)

    return cwd


def _find_project_paths(start: Path) -> tuple[Path, Path, Path]:
    for candidate_root in (start, *start.parents):
        discovered = _config_for_root(candidate_root)
        if discovered is not None:
            return discovered

    raise FileNotFoundError(f"No Backlog.md config found from {start}")


def _config_for_root(root: Path) -> tuple[Path, Path, Path] | None:
    root_config = root / "backlog.config.yml"
    if root_config.is_file():
        return root, root / "backlog", root_config

    backlog_config = root / "backlog" / "config.yml"
    if backlog_config.is_file():
        return root, root / "backlog", backlog_config

    dot_backlog_config = root / ".backlog" / "config.yml"
    if dot_backlog_config.is_file():
        return root, root / ".backlog", dot_backlog_config

    return None
