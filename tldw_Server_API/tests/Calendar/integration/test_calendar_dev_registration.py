"""Calendar registration contracts on the current dev router and worker graph."""

from tldw_Server_API.app.api.v1.router_groups.content import iter_content_router_specs
from tldw_Server_API.app.api.v1.router_groups.minimal import MINIMAL_REQUIRED_ROUTER_NAMES
from tldw_Server_API.app.services.startup_content_jobs_pollers import (
    provide_content_jobs_worker_specs,
)


def test_calendar_router_is_registered_as_opt_in() -> None:
    specs = {spec.name: spec for spec in iter_content_router_specs()}

    assert specs["calendar"].route_key == "calendar"
    assert specs["calendar"].default_stable is False


def test_calendar_router_is_available_to_minimal_test_app() -> None:
    assert "calendar" in MINIMAL_REQUIRED_ROUTER_NAMES


def test_calendar_sync_workers_are_registered() -> None:
    specs = {spec.name: spec for spec in provide_content_jobs_worker_specs()}

    assert "calendar_sync_jobs_task" in specs
    assert "calendar_sync_scheduler_task" in specs
