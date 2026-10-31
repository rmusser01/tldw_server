"""The adapter catalog fixture supplies inventory without disabling admission."""

import os
import subprocess  # nosec B404 - isolated pytest regression with fixed interpreter and arguments.
import sys
from pathlib import Path

import pytest

from tldw_Server_API.app.core.exceptions import ChatModelNotAvailableError
from tldw_Server_API.app.core.LLM_Calls import gateway

pytestmark = [pytest.mark.unit, pytest.mark.provider_inventory]


@pytest.mark.parametrize(
    "provider,model",
    [("openai", "gpt-4o-mini"), ("groq", "llama3-8b"), ("google", "gemini-2.5-pro")],
)
def test_offline_inventory_keeps_model_membership_checks(
    offline_adapter_inventory: list[str],
    provider: str,
    model: str,
) -> None:
    request = {"model": model, "api_key": "synthetic-inventory-key", "app_config": {}}
    gateway._validate_provider_model_selection(provider, request)
    assert offline_adapter_inventory == [provider]

    with pytest.raises(ChatModelNotAvailableError):
        gateway._validate_provider_model_selection(provider, {**request, "model": "not-in-test-catalog"})
    assert offline_adapter_inventory == [provider, provider]


def test_unit_inventory_stays_offline_when_network_tests_are_enabled() -> None:
    """A fresh pytest process must keep unit inventory finite even with live tests opted in."""
    test_file = Path(__file__).resolve()
    env = {**os.environ, "ENABLE_NETWORK_TESTS": "true", "PYTEST_ADDOPTS": ""}
    result = subprocess.run(  # nosec B603 - repo-owned test path, fixed argv, no shell.
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-o",
            "usefixtures=no_live_network",
            f"{test_file}::test_offline_inventory_keeps_model_membership_checks",
            "--tb=short",
        ],
        cwd=test_file.parents[4],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
