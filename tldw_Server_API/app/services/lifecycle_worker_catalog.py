"""Catalog helpers for declarative lifecycle worker specs."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from tldw_Server_API.app.services.lifecycle_worker_specs import (
    WorkerLifecycleContext,
    WorkerSpec,
    validate_worker_spec_graph,
)

SpecProvider = Callable[[WorkerLifecycleContext], Sequence[WorkerSpec]]


def collect_worker_specs(
    context: WorkerLifecycleContext,
    providers: Sequence[SpecProvider],
) -> list[WorkerSpec]:
    """Collect worker specs from providers and validate the resulting graph."""

    specs: list[WorkerSpec] = []
    for provider in providers:
        specs.extend(provider(context))
    validate_worker_spec_graph(specs)
    return specs


def assert_legacy_worker_spec_parity(
    legacy_names: set[str],
    specs: Sequence[WorkerSpec],
) -> None:
    """Assert that every legacy managed worker name has a declarative spec."""

    spec_names = {spec.name for spec in specs}
    missing_names = sorted(legacy_names - spec_names)
    if missing_names:
        raise AssertionError(
            "Missing declarative worker specs for legacy managed worker(s): "
            f"{', '.join(missing_names)}"
        )
