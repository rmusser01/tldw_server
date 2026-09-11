"""Contract for the minimal configuration used by the critical frontend E2E job."""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit


_WORKFLOW_PATH = Path(".github/workflows/frontend-e2e-tiers.yml")
_FIXTURE_PATH = Path("tldw_Server_API/Config_Files/e2e-critical-config.txt")


def test_critical_e2e_workflow_selects_the_tracked_minimal_config_fixture() -> None:
    """Keep critical E2E provider discovery isolated from developer local endpoints."""
    workflow = yaml.safe_load(_WORKFLOW_PATH.read_text(encoding="utf-8"))
    env = workflow["jobs"]["critical"]["env"]

    assert env["TLDW_CONFIG_FILE"] == str(_FIXTURE_PATH)
    assert _FIXTURE_PATH.is_file()
