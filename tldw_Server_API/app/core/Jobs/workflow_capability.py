"""Inert prepared-workflow capability and explicit support-schema entrypoints."""

from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tldw_Server_API.app.core.DB_Management.jobs_workflow_migrations import (
    inspect_workflow_support_in_transaction,
    install_workflow_support_in_transaction,
)
from tldw_Server_API.app.core.DB_Management.jobs_workflow_store import workflow_store

from .operations.workflow_contracts import WorkflowOutcome, WorkflowPolicy

if TYPE_CHECKING:
    from .manager import JobManager


@dataclass(frozen=True)
class PreparedWorkflowCapability:
    """Bind trusted policy only; construction performs no IO or admission."""

    manager: JobManager
    policy: WorkflowPolicy

    def __post_init__(self) -> None:
        if type(self.policy) is not WorkflowPolicy:
            raise ValueError("workflow capability requires a trusted policy")


def install_workflow_support(manager: JobManager) -> None:
    """Explicitly own one atomic install transaction, including rollback on error."""
    store = workflow_store(manager.backend)
    with closing(manager._connect()) as connection:
        try:
            store.begin_install(connection)
            install_workflow_support_in_transaction(connection, backend=manager.backend)
            connection.commit()
        except BaseException:
            connection.rollback()
            raise


def inspect_workflow_support(manager: JobManager) -> WorkflowOutcome:
    """Read readiness only; do not migrate, admit, or activate application wiring."""
    with closing(manager._connect()) as connection:
        try:
            return inspect_workflow_support_in_transaction(connection, backend=manager.backend)
        finally:
            connection.rollback()
