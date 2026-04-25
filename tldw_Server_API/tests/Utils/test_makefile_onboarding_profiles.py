"""Contract tests for public onboarding Makefile targets."""

import re
from pathlib import Path

import pytest


def _require(condition: bool, message: str) -> None:
    """Fail with a descriptive assertion message when a contract is broken."""
    if not condition:
        pytest.fail(message)


def _read_makefile() -> str:
    """Return the root Makefile body."""
    return Path("Makefile").read_text(encoding="utf-8")


def _target_block(makefile_text: str, target: str) -> str:
    """Return a Make target block from the Makefile or fail clearly."""
    pattern = rf"^{re.escape(target)}:.*?(?=^[A-Za-z0-9_.-]+:|\Z)"
    match = re.search(pattern, makefile_text, flags=re.MULTILINE | re.DOTALL)
    _require(match is not None, f"Make target {target} should exist")
    return match.group(0)


def test_public_onboarding_targets_exist() -> None:
    """All public onboarding profile targets should be present."""
    text = _read_makefile()

    for target in (
        "setup-wizard-tools",
        "setup-docker-single",
        "start-docker-single",
        "verify-docker-single",
        "setup-docker-multi",
        "start-docker-multi",
        "verify-docker-multi",
        "install-local",
        "setup-local-single",
        "start-local-single",
        "verify-local-single",
    ):
        _target_block(text, target)


def test_setup_targets_delegate_to_tldw_setup_profiles_and_setup_venv() -> None:
    """Setup targets should use the lightweight wizard venv and explicit profiles."""
    text = _read_makefile()

    _require(
        "DOCKER_SINGLE_COMPOSE ?= Dockerfiles/docker-compose.single-user.yml" in text,
        "Expected single-user Docker compose helper variable",
    )
    _require(
        "DOCKER_MULTI_COMPOSE ?= Dockerfiles/docker-compose.multi-user-postgres.yml" in text,
        "Expected multi-user Docker compose helper variable",
    )
    _require("SETUP_VENV_DIR ?= .setup-venv" in text, "Expected .setup-venv helper variable")
    _require(
        "SETUP_VENV_PYTHON ?= $(SETUP_VENV_DIR)/bin/python" in text,
        "Expected setup venv Python helper variable",
    )
    _require(
        "TLDW_SETUP ?= $(SETUP_VENV_PYTHON) -m tldw_Server_API.cli.wizard.cli" in text,
        "Expected tldw-setup module helper to use setup venv Python",
    )
    _require("TLDW_BASE_URL ?= http://127.0.0.1:8000" in text, "Expected API base URL helper variable")
    _require("TLDW_WEBUI_URL ?= http://127.0.0.1:8080" in text, "Expected WebUI URL helper variable")

    setup_wizard_tools = _target_block(text, "setup-wizard-tools")
    _require("$(PYTHON) -m venv $(SETUP_VENV_DIR)" in setup_wizard_tools, "Expected setup venv creation")
    _require("typer>=0.12.0" in setup_wizard_tools, "Expected wizard runtime dependency install")
    _require("httpx>=0.24.0" in setup_wizard_tools, "Expected wizard verify dependency install")

    expected_profiles = {
        "setup-docker-single": "docker-single-webui",
        "setup-docker-multi": "docker-multi-postgres",
        "setup-local-single": "local-single",
    }
    for target, profile in expected_profiles.items():
        block = _target_block(text, target)
        _require("setup-wizard-tools" in block, f"{target} should depend on setup-wizard-tools")
        _require("$(TLDW_SETUP) init" in block, f"{target} should run tldw-setup init")
        _require(f"--profile {profile}" in block, f"{target} should use profile {profile}")
        _require('--env-file "$(TLDW_ENV_FILE)"' in block, f"{target} should pass the env file")


def test_verify_targets_use_first_value() -> None:
    """Profile verify targets should use first-value env parsing."""
    text = _read_makefile()

    expected_profiles = {
        "verify-docker-single": "docker-single-webui",
        "verify-docker-multi": "docker-multi-postgres",
        "verify-local-single": "local-single",
    }
    for target, profile in expected_profiles.items():
        block = _target_block(text, target)
        _require("setup-wizard-tools" in block, f"{target} should depend on setup-wizard-tools")
        _require("$(TLDW_SETUP) verify" in block, f"{target} should run tldw-setup verify")
        _require(f"--profile {profile}" in block, f"{target} should use profile {profile}")
        _require("--first-value" in block, f"{target} should pass --first-value")


def test_quickstart_install_is_install_only_and_does_not_start_local_server() -> None:
    """The local install alias should install dependencies only."""
    text = _read_makefile()
    quickstart_install = _target_block(text, "quickstart-install")

    _require("install-local" in quickstart_install, "quickstart-install should delegate to install-local")
    _require(
        "quickstart-local" not in quickstart_install,
        "quickstart-install should not chain into quickstart-local",
    )
    _require("uvicorn" not in quickstart_install, "quickstart-install should not start uvicorn")


def test_default_output_targets_do_not_print_full_api_keys() -> None:
    """Default quickstart/start output should point users to explicit secret printing."""
    text = _read_makefile()

    _require("make show-api-key" in text, "Makefile should mention make show-api-key")

    for target in (
        "quickstart",
        "quickstart-docker",
        "quickstart-docker-webui",
        "start-docker-single",
        "start-docker-multi",
        "start-local-single",
    ):
        block = _target_block(text, target)
        _require("grep '^SINGLE_USER_API_KEY='" not in block, f"{target} should not read the API key")
        _require("cut -d= -f2-" not in block, f"{target} should not print the API key value")
        _require("Your API Key:" not in block, f"{target} should not label full API key output")
