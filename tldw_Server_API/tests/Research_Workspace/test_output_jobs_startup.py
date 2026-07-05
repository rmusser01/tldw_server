from __future__ import annotations

import pytest

from tldw_Server_API.app.services.startup_content_jobs_pollers import (
    provide_content_jobs_worker_specs,
)

pytestmark = pytest.mark.unit


def test_content_job_specs_include_research_workspace_output_worker() -> None:
    specs = provide_content_jobs_worker_specs()
    names = {spec.name for spec in specs}

    assert "research_workspace_output_jobs_task" in names
