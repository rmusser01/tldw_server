"""Legacy in-memory orchestration service for agent projects, tasks, and runs.

Provides CRUD operations, dependency gating, cycle detection,
and reviewer gate logic for older service-level tests. Production code uses
``db_factory.get_orchestration_db`` for durable per-user SQLite state.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from loguru import logger

from .models import (
    AgentProject,
    AgentRun,
    AgentTask,
    RunStatus,
    TaskStatus,
    is_valid_transition,
)


class CycleDependencyError(ValueError):
    """Raised when a task dependency would create a cycle."""


class OrchestrationService:
    """In-memory orchestration service."""

    def __init__(self) -> None:
        self._projects: dict[int, AgentProject] = {}
        self._tasks: dict[int, AgentTask] = {}
        self._runs: dict[int, AgentRun] = {}
        self._project_seq = 0
        self._task_seq = 0
        self._run_seq = 0
        self._lock = asyncio.Lock()

    # -- Projects --

    async def create_project(
        self, name: str, description: str = "", user_id: int = 0, metadata: dict[str, Any] | None = None,
    ) -> AgentProject:
        async with self._lock:
            self._project_seq += 1
            now = datetime.now(timezone.utc).isoformat()
            project = AgentProject(
                id=self._project_seq,
                name=name,
                description=description,
                user_id=user_id,
                created_at=now,
                metadata=metadata or {},
            )
            self._projects[project.id] = project
        return project

    async def get_project(self, project_id: int) -> AgentProject | None:
        return self._projects.get(project_id)

    async def list_projects(self, user_id: int | None = None) -> list[AgentProject]:
        results = list(self._projects.values())
        if user_id is not None:
            results = [p for p in results if p.user_id == user_id]
        results.sort(key=lambda p: p.created_at, reverse=True)
        return results

    async def delete_project(self, project_id: int) -> bool:
        async with self._lock:
            if project_id not in self._projects:
                return False
            # Delete associated tasks and runs
            task_ids = [t.id for t in self._tasks.values() if t.project_id == project_id]
            for tid in task_ids:
                run_ids = [r.id for r in self._runs.values() if r.task_id == tid]
                for rid in run_ids:
                    del self._runs[rid]
                del self._tasks[tid]
            del self._projects[project_id]
        return True

    # -- Tasks --

    def _detect_cycle(self, task_id: int, dependency_id: int) -> bool:
        """Check if adding dependency_id as a dependency of task_id would create a cycle."""
        visited: set[int] = set()
        current = dependency_id
        while current is not None:
            if current == task_id:
                return True
            if current in visited:
                return False  # Hit a non-cyclic loop
            visited.add(current)
            dep_task = self._tasks.get(current)
            if dep_task is None:
                return False
            current = dep_task.dependency_id
        return False

    async def create_task(
        self,
        project_id: int,
        title: str,
        description: str = "",
        agent_type: str | None = None,
        dependency_id: int | None = None,
        reviewer_agent_type: str | None = None,
        max_review_attempts: int = 3,
        success_criteria: str = "",
        user_id: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> AgentTask:
        async with self._lock:
            if project_id not in self._projects:
                raise ValueError(f"Project {project_id} not found")
            if dependency_id is not None:
                if dependency_id not in self._tasks:
                    raise ValueError(f"Dependency task {dependency_id} not found")
                # Cycle detection (use temp task_id = next seq)
                temp_id = self._task_seq + 1
                if self._detect_cycle(temp_id, dependency_id):
                    raise CycleDependencyError(
                        f"Adding dependency {dependency_id} would create a cycle"
                    )
            self._task_seq += 1
            now = datetime.now(timezone.utc).isoformat()
            task = AgentTask(
                id=self._task_seq,
                project_id=project_id,
                title=title,
                description=description,
                agent_type=agent_type,
                dependency_id=dependency_id,
                reviewer_agent_type=reviewer_agent_type,
                max_review_attempts=max_review_attempts,
                success_criteria=success_criteria,
                user_id=user_id,
                created_at=now,
                metadata=metadata or {},
            )
            self._tasks[task.id] = task
        return task

    async def get_task(self, task_id: int) -> AgentTask | None:
        return self._tasks.get(task_id)

    async def list_tasks(
        self, project_id: int, status: TaskStatus | None = None,
    ) -> list[AgentTask]:
        results = [t for t in self._tasks.values() if t.project_id == project_id]
        if status is not None:
            results = [t for t in results if t.status == status]
        results.sort(key=lambda t: t.created_at)
        return results

    async def transition_task(self, task_id: int, new_status: TaskStatus) -> AgentTask:
        """Transition a task to a new status, enforcing the state machine."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise ValueError(f"Task {task_id} not found")
            if not is_valid_transition(task.status, new_status):
                raise ValueError(
                    f"Invalid transition: {task.status.value} → {new_status.value}"
                )
            task.status = new_status
            task.updated_at = datetime.now(timezone.utc).isoformat()
        return task

    async def check_dependency_ready(self, task_id: int) -> bool:
        """Check if a task's dependency is complete (or has no dependency)."""
        task = self._tasks.get(task_id)
        if task is None:
            return False
        if task.dependency_id is None:
            return True
        dep = self._tasks.get(task.dependency_id)
        if dep is None:
            return False
        return dep.status == TaskStatus.COMPLETE

    # -- Runs --

    async def create_run(
        self, task_id: int, agent_type: str | None = None, session_id: str | None = None,
    ) -> AgentRun:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise ValueError(f"Task {task_id} not found")
            self._run_seq += 1
            now = datetime.now(timezone.utc).isoformat()
            run = AgentRun(
                id=self._run_seq,
                task_id=task_id,
                session_id=session_id,
                agent_type=agent_type or task.agent_type,
                started_at=now,
                status=RunStatus.RUNNING,
            )
            self._runs[run.id] = run
        return run

    async def complete_run(
        self, run_id: int, result_summary: str = "", token_usage: dict[str, int] | None = None,
    ) -> AgentRun:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise ValueError(f"Run {run_id} not found")
            run.status = RunStatus.COMPLETED
            run.completed_at = datetime.now(timezone.utc).isoformat()
            run.result_summary = result_summary
            if token_usage:
                run.token_usage = token_usage
        return run

    async def fail_run(self, run_id: int, error: str = "") -> AgentRun:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise ValueError(f"Run {run_id} not found")
            run.status = RunStatus.FAILED
            run.completed_at = datetime.now(timezone.utc).isoformat()
            run.error = error
        return run

    async def list_runs(self, task_id: int) -> list[AgentRun]:
        results = [r for r in self._runs.values() if r.task_id == task_id]
        results.sort(key=lambda r: r.started_at or "", reverse=True)
        return results

    # -- Reviewer Gate --

    async def submit_review(
        self, task_id: int, approved: bool, feedback: str = "",
    ) -> AgentTask:
        """Submit a review result. Approved → complete, rejected → back to inprogress or triage."""
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                raise ValueError(f"Task {task_id} not found")
            if task.status != TaskStatus.REVIEW:
                raise ValueError(f"Task {task_id} is not in review status")

            task.review_count += 1
            now = datetime.now(timezone.utc).isoformat()
            task.updated_at = now

            if approved:
                task.status = TaskStatus.COMPLETE
                logger.info("Task {} approved after {} review(s)", task_id, task.review_count)
            else:
                if task.review_count >= task.max_review_attempts:
                    task.status = TaskStatus.TRIAGE
                    logger.warning(
                        "Task {} sent to triage after {} failed review(s)",
                        task_id, task.review_count,
                    )
                else:
                    task.status = TaskStatus.IN_PROGRESS
                    logger.info(
                        "Task {} rejected (review {}/{}), returning to in_progress",
                        task_id, task.review_count, task.max_review_attempts,
                    )
        return task

    # -- Summary --

    async def get_project_summary(self, project_id: int) -> dict[str, Any]:
        """Get task counts by status for a project."""
        tasks = await self.list_tasks(project_id)
        counts: dict[str, int] = {}
        for task in tasks:
            counts[task.status.value] = counts.get(task.status.value, 0) + 1
        return {
            "project_id": project_id,
            "total_tasks": len(tasks),
            "status_counts": counts,
        }


# Module-level singleton
_service: OrchestrationService | None = None
_service_lock = asyncio.Lock()


async def get_orchestration_service() -> OrchestrationService:
    global _service
    if _service is None:
        async with _service_lock:
            if _service is None:
                _service = OrchestrationService()
    return _service


# Backward-compatible import for older callers. New production code should
# import this from Agent_Orchestration.db_factory directly.
from .db_factory import get_orchestration_db  # noqa: E402
