"""Inert prepared-workflow capability and explicit support-schema entrypoints."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tldw_Server_API.app.core.DB_Management.jobs_workflow_migrations import (
    inspect_workflow_support_in_transaction,
    install_workflow_support_in_transaction,
)
from tldw_Server_API.app.core.DB_Management.jobs_workflow_store import (
    LiveWorkflowTransaction,
    WorkflowTransaction,
    workflow_store,
)

from .operations.workflow_contracts import (
    WorkflowAdmissionCommand,
    WorkflowAdmissionResult,
    WorkflowEffect,
    WorkflowEffectResult,
    WorkflowFenceError,
    WorkflowLookupResult,
    WorkflowOutcome,
    WorkflowPolicy,
    WorkflowRoot,
    _text,
)

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

    @contextmanager
    def transaction(self, owner_user_id: str) -> Iterator[WorkflowTransaction]:
        """Begin once, establish fail-closed owner scope, commit once, close."""
        _text(owner_user_id, 512)
        manager = self.manager
        variables = (manager._RLS_OWNER_USER_ID, manager._RLS_IS_ADMIN, manager._RLS_DOMAIN_ALLOWLIST)
        tokens = [variable.set(value) for variable, value in zip(variables, (owner_user_id, False, None))]
        tx = None
        transaction_token = None
        try:
            with closing(manager._connect()) as connection:
                try:
                    store = workflow_store(manager.backend)
                    transaction_id = store.begin_runtime(connection, owner_user_id)
                    if store.inspect_support(connection) is not WorkflowOutcome.APPLIED:
                        raise WorkflowFenceError(WorkflowOutcome.UNAVAILABLE)
                    tx = LiveWorkflowTransaction(manager, self.policy, connection, owner_user_id, transaction_id)
                    transaction_token = manager._WORKFLOW_TRANSACTION.set(tx)
                    yield tx
                    tx.validate_fences()
                    connection.commit()
                except BaseException:
                    connection.rollback()
                    raise
                finally:
                    if tx is not None:
                        tx.active = False
        finally:
            if transaction_token is not None:
                manager._WORKFLOW_TRANSACTION.reset(transaction_token)
            for variable, token in zip(variables, tokens):
                variable.reset(token)
        # Only a successful outer commit reaches observers. Facts were copied
        # from the UUID-locked row, never re-resolved through an integer ID.
        for event in tx.effect_events:
            manager._emit_workflow_effect_side_effects(*event)

    def admit(self, command: WorkflowAdmissionCommand) -> WorkflowAdmissionResult:
        """Replay exact retained authority before evaluating mutable policy."""
        if type(command) is not WorkflowAdmissionCommand:
            raise ValueError("workflow admission requires exact command facts")
        with self.transaction(command.owner_user_id) as tx:
            result = tx.admit(command)
        if tx.admission_event is not None:
            self.manager._emit_create_side_effects(
                tx.admission_event, backend=self.manager.backend, idempotency_key=command.idempotency_key
            )
        return result

    def lookup(self, root: WorkflowRoot) -> WorkflowLookupResult:
        """Read UUID authority on one owner-scoped transaction, ignoring ID hints."""
        if type(root) is not WorkflowRoot:
            raise ValueError("workflow lookup requires exact root facts")
        with self.transaction(root.owner_user_id) as tx:
            return tx.lookup(root)

    def lookup_effect(self, root: WorkflowRoot, effect: WorkflowEffect) -> WorkflowEffectResult:
        """Read exact retained proof without requiring a lease or Jobs history."""
        if type(root) is not WorkflowRoot:
            raise ValueError("workflow effect lookup requires exact root facts")
        with self.transaction(root.owner_user_id) as tx:
            return tx.lookup_effect(root, effect)

    def release_effect(self, root: WorkflowRoot, effect: WorkflowEffect) -> WorkflowEffectResult:
        """Release an existing exact proof, retaining its immutable tombstone."""
        if type(root) is not WorkflowRoot:
            raise ValueError("workflow effect release requires exact root facts")
        with self.transaction(root.owner_user_id) as tx:
            return tx.release_effect(root, effect)


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
