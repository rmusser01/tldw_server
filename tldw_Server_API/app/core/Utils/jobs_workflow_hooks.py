"""Inert dependency ports for native Jobs admission and lifecycle operations."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class WorkflowBackendHooks:
    """Trusted native leaves supplied by the Jobs composition layer."""

    backend: str
    quota_rejection: Callable[..., Any]
    insert_job: Callable[..., Any]
    bump_counters: Callable[..., Any]
    insert_created_event: Callable[..., Any]
    prepared_counter_transition: Callable[..., Any]
    insert_prepared_event: Callable[..., Any]
    quota_lock_key: Callable[..., int] | None = None

    def __post_init__(self) -> None:
        if self.backend not in {"sqlite", "postgres"}:
            raise ValueError("workflow hooks require a supported backend")
        operations = (
            self.quota_rejection,
            self.insert_job,
            self.bump_counters,
            self.insert_created_event,
            self.prepared_counter_transition,
            self.insert_prepared_event,
        )
        if not all(callable(operation) for operation in operations):
            raise ValueError("workflow hooks require callable native operations")
        if self.backend == "postgres" and not callable(self.quota_lock_key):
            raise ValueError("PostgreSQL workflow hooks require quota locking")
