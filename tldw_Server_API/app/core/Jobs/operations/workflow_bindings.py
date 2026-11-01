"""Compose existing Jobs leaves without persistence importing domain logic."""

from tldw_Server_API.app.core.Utils.jobs_workflow_hooks import WorkflowBackendHooks


def workflow_backend_hooks(backend: str) -> WorkflowBackendHooks:
    """Bind one native backend; importing this module performs no backend setup."""
    if backend == "sqlite":
        from .sqlite import admission, lifecycle
    elif backend == "postgres":
        from .postgres import admission, lifecycle
    else:
        raise ValueError("workflow hooks require a supported backend")
    return WorkflowBackendHooks(
        backend=backend,
        quota_rejection=admission._quota_rejection,
        insert_job=admission._insert_job,
        bump_counters=admission._bump_counters,
        insert_created_event=admission._insert_created_event,
        prepared_counter_transition=lifecycle._prepared_counter_transition,
        insert_prepared_event=lifecycle._insert_prepared_event,
        quota_lock_key=admission._quota_lock_key if backend == "postgres" else None,
    )
