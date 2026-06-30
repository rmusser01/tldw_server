"""Data models and state machine for Agent Orchestration.

State machine:
    todo → inprogress → review → complete
                  ↘                  ↗
                    → triage --------→

Tasks have optional dependencies: a task with dependency_id can't
start until the dependency is 'complete'.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "inprogress"
    REVIEW = "review"
    COMPLETE = "complete"
    TRIAGE = "triage"


# Valid transitions in the state machine
_VALID_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.TODO: {TaskStatus.IN_PROGRESS},
    TaskStatus.IN_PROGRESS: {TaskStatus.REVIEW, TaskStatus.TRIAGE},
    TaskStatus.REVIEW: {TaskStatus.COMPLETE, TaskStatus.TRIAGE, TaskStatus.IN_PROGRESS},
    TaskStatus.TRIAGE: {TaskStatus.TODO, TaskStatus.IN_PROGRESS},
    TaskStatus.COMPLETE: set(),  # Terminal state
}


def is_valid_transition(from_status: TaskStatus, to_status: TaskStatus) -> bool:
    """Check if a status transition is allowed."""
    return to_status in _VALID_TRANSITIONS.get(from_status, set())


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_VALID_RUN_TRANSITIONS: dict[RunStatus, set[RunStatus]] = {
    RunStatus.PENDING: {RunStatus.RUNNING, RunStatus.FAILED, RunStatus.CANCELLED},
    RunStatus.RUNNING: {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED},
    RunStatus.COMPLETED: set(),
    RunStatus.FAILED: set(),
    RunStatus.CANCELLED: set(),
}


def is_valid_run_transition(from_status: RunStatus, to_status: RunStatus) -> bool:
    """Check if a run status transition is allowed."""
    return to_status in _VALID_RUN_TRANSITIONS.get(from_status, set())


@dataclass
class ACPWorkspace:
    """A persistent workspace binding a project to a filesystem directory.

    Note: ``env_vars`` are stored as plaintext JSON in the per-user SQLite DB.
    Do not store high-sensitivity secrets here without external encryption.
    """
    id: int
    name: str
    root_path: str
    description: str = ""
    workspace_type: str = "manual"  # manual | discovered | monorepo_child
    parent_workspace_id: int | None = None
    env_vars: dict[str, str] = field(default_factory=dict)
    git_remote_url: str | None = None
    git_default_branch: str | None = None
    git_current_branch: str | None = None
    git_is_dirty: bool | None = None
    last_health_check: str | None = None
    health_status: str = "unknown"  # healthy | degraded | missing
    user_id: int = 0
    created_at: str = ""
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "root_path": self.root_path,
            "description": self.description,
            "workspace_type": self.workspace_type,
            "parent_workspace_id": self.parent_workspace_id,
            "env_vars": dict(self.env_vars),
            "git_remote_url": self.git_remote_url,
            "git_default_branch": self.git_default_branch,
            "git_current_branch": self.git_current_branch,
            "git_is_dirty": self.git_is_dirty,
            "last_health_check": self.last_health_check,
            "health_status": self.health_status,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }


@dataclass
class AgentProject:
    """A project grouping related tasks."""
    id: int
    name: str
    description: str = ""
    workspace_id: int | None = None
    user_id: int = 0
    created_at: str = ""
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "workspace_id": self.workspace_id,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }


@dataclass
class AgentTask:
    """A single task within a project."""
    id: int
    project_id: int
    title: str
    description: str = ""
    status: TaskStatus = TaskStatus.TODO
    agent_type: str | None = None
    dependency_id: int | None = None
    reviewer_agent_type: str | None = None
    max_review_attempts: int = 3
    review_count: int = 0
    success_criteria: str = ""
    user_id: int = 0
    created_at: str = ""
    updated_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "agent_type": self.agent_type,
            "dependency_id": self.dependency_id,
            "reviewer_agent_type": self.reviewer_agent_type,
            "max_review_attempts": self.max_review_attempts,
            "review_count": self.review_count,
            "success_criteria": self.success_criteria,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }


@dataclass
class AgentRun:
    """A single execution run of a task."""
    id: int
    task_id: int
    session_id: str | None = None
    status: RunStatus = RunStatus.PENDING
    agent_type: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    result_summary: str = ""
    error: str | None = None
    token_usage: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "session_id": self.session_id,
            "status": self.status.value,
            "agent_type": self.agent_type,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result_summary": self.result_summary,
            "error": self.error,
            "token_usage": dict(self.token_usage),
        }
