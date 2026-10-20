"""Transaction/store protocols and explicit backend selection; no domain policy."""

from __future__ import annotations

from typing import Any, Protocol

from tldw_Server_API.app.core.Jobs.operations.workflow_contracts import WorkflowOutcome


class WorkflowTransaction(Protocol):
    """A future enlisted repository sees the same native connection/backend."""

    @property
    def connection(self) -> Any: ...

    @property
    def backend(self) -> str: ...


class WorkflowStore(Protocol):
    """Schema-only backend surface; lifecycle operations are not activated."""

    def begin_install(self, connection: Any) -> None: ...

    def install_support(self, connection: Any) -> None: ...

    def inspect_support(self, connection: Any) -> WorkflowOutcome: ...


def workflow_store(backend: str) -> WorkflowStore:
    """Select an inert leaf; imports never connect or migrate."""
    if backend == "sqlite":
        from . import jobs_workflow_sqlite

        return jobs_workflow_sqlite
    if backend == "postgres":
        from . import jobs_workflow_postgres

        return jobs_workflow_postgres
    raise ValueError("unsupported Jobs workflow backend")
