"""Explicit domain hooks preserve workflow transactions without upward imports."""

from dataclasses import FrozenInstanceError

import pytest

from tldw_Server_API.app.core.DB_Management.jobs_workflow_store import recover_workflow_expiry_in_transaction
from tldw_Server_API.app.core.Jobs.operations.workflow_contracts import WorkflowFenceError, WorkflowOutcome


def test_missing_domain_hooks_cannot_close_an_enrolled_episode(workflow_case):
    case = workflow_case
    case.admit()
    case.acquire()
    case.expire_current_lease()
    row = case.job_row()
    before = case.observe()
    with case.connection(runtime=True) as connection:
        if case.manager.backend == "sqlite":
            connection.execute("BEGIN IMMEDIATE")
        with pytest.raises(WorkflowFenceError) as error:
            recover_workflow_expiry_in_transaction(connection, backend=case.manager.backend, locked_root=row)
        assert error.value.outcome is WorkflowOutcome.UNAVAILABLE
    assert case.observe() == before


def test_legacy_recovery_needs_neither_hooks_nor_storage():
    result = recover_workflow_expiry_in_transaction(
        object(), backend="sqlite", locked_root={"workflow_capability_version": None}
    )
    assert result.outcome is WorkflowOutcome.MISSING


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_domain_factory_binds_exact_native_operations(backend):
    from tldw_Server_API.app.core.Jobs.operations.postgres import admission as pg_admission
    from tldw_Server_API.app.core.Jobs.operations.postgres import lifecycle as pg_lifecycle
    from tldw_Server_API.app.core.Jobs.operations.sqlite import admission as sqlite_admission
    from tldw_Server_API.app.core.Jobs.operations.sqlite import lifecycle as sqlite_lifecycle
    from tldw_Server_API.app.core.Jobs.operations.workflow_bindings import workflow_backend_hooks
    from tldw_Server_API.app.core.Utils.jobs_workflow_hooks import WorkflowBackendHooks

    admission, lifecycle = (sqlite_admission, sqlite_lifecycle) if backend == "sqlite" else (pg_admission, pg_lifecycle)
    hooks = workflow_backend_hooks(backend)
    assert type(hooks) is WorkflowBackendHooks
    assert hooks.backend == backend
    assert hooks.quota_rejection is admission._quota_rejection
    assert hooks.insert_job is admission._insert_job
    assert hooks.bump_counters is admission._bump_counters
    assert hooks.insert_created_event is admission._insert_created_event
    assert hooks.prepared_counter_transition is lifecycle._prepared_counter_transition
    assert hooks.insert_prepared_event is lifecycle._insert_prepared_event
    assert hooks.quota_lock_key is (None if backend == "sqlite" else admission._quota_lock_key)
    with pytest.raises(FrozenInstanceError):
        hooks.backend = "invalid"
