from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BacklogConfig:
    project_name: str
    statuses: list[str] | None = None
    default_status: str = "To Do"
    remote_operations: bool = True
    auto_commit: bool = False
    bypass_git_hooks: bool = False
    check_active_branches: bool = True
    active_branch_days: int = 30
    definition_of_done: list[str] | None = None


@dataclass(frozen=True)
class BacklogProject:
    root: Path
    backlog_dir: Path
    config_path: Path
    config: BacklogConfig
