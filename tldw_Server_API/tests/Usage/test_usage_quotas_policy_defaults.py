"""No stock RG policy carries a usage quota for evaluations (spec 2)."""

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

_POLICIES = Path(__file__).resolve().parents[2] / "Config_Files" / "resource_governor_policies.yaml"


def test_stock_evaluation_policies_have_no_daily_caps() -> None:
    policies = yaml.safe_load(_POLICIES.read_text())["policies"]
    offenders = [
        f"{pid}.{category}"
        for pid, policy in policies.items()
        if pid.startswith("evals.")
        for category, spec in policy.items()
        if isinstance(spec, dict) and "daily_cap" in spec
    ]
    assert offenders == []
