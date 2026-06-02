from __future__ import annotations

from dataclasses import fields

import pytest

pytestmark = pytest.mark.unit


def test_runtime_state_stores_only_lifecycle_session() -> None:
    from tldw_Server_API.app.services.lifespan_worker_runtime_state import (
        LifespanWorkerRuntimeState,
    )

    assert {field.name for field in fields(LifespanWorkerRuntimeState)} == {
        "worker_lifecycle_session",
    }


def test_apply_startup_worker_bootstrap_handles_stores_lifecycle_session() -> None:
    from tldw_Server_API.app.services.lifespan_worker_runtime_state import (
        LifespanWorkerRuntimeState,
    )
    from tldw_Server_API.app.services.startup_worker_bootstrap import (
        StartupWorkerBootstrapHandles,
    )

    session = object()
    runtime = LifespanWorkerRuntimeState()
    startup_handles = StartupWorkerBootstrapHandles(
        worker_lifecycle_session=session,
    )

    runtime.apply_startup_worker_bootstrap_handles(startup_handles)

    assert runtime.worker_lifecycle_session is session


def test_runtime_state_no_longer_exposes_worker_inventory_sync_bridge() -> None:
    from tldw_Server_API.app.services.lifespan_worker_runtime_state import (
        LifespanWorkerRuntimeState,
    )

    assert not hasattr(LifespanWorkerRuntimeState, "sync_from_worker_inventory")
    assert not hasattr(LifespanWorkerRuntimeState(), "owned_job_pollers")
    assert not hasattr(LifespanWorkerRuntimeState(), "worker_inventory")


def test_runtime_state_no_longer_exposes_shutdown_apply_bridge_methods() -> None:
    from tldw_Server_API.app.services.lifespan_worker_runtime_state import (
        LifespanWorkerRuntimeState,
    )

    apply_methods = {
        name
        for name in dir(LifespanWorkerRuntimeState)
        if name.startswith("apply_")
    }

    assert apply_methods == {"apply_startup_worker_bootstrap_handles"}
