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
    _require(
        "cryptography>=41.0.0" in setup_wizard_tools,
        "Expected wizard profile encryption dependency install",
    )

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


def test_start_local_single_uses_plain_uvicorn_without_reload() -> None:
    """Local start should run uvicorn directly without development reload."""
    text = _read_makefile()
    block = _target_block(text, "start-local-single")

    _require(
        "uvicorn tldw_Server_API.app.main:app" in block
        or "-m uvicorn tldw_Server_API.app.main:app" in block,
        "start-local-single should run plain uvicorn for the FastAPI app",
    )
    _require("--reload" not in block, "start-local-single should not use --reload")


def test_start_local_single_exports_selected_env_file_to_server() -> None:
    """Local start should pass the same env file selected during setup/verify."""
    text = _read_makefile()
    block = _target_block(text, "start-local-single")

    _require(
        'TLDW_ENV_FILE="$(TLDW_ENV_FILE)"' in block,
        "start-local-single should export TLDW_ENV_FILE for app config loading",
    )


def test_install_local_does_not_start_local_server() -> None:
    """Local install should install dependencies only, without chaining startup."""
    text = _read_makefile()
    install_local = _target_block(text, "install-local")

    _require("uvicorn" not in install_local, "install-local should not start uvicorn")
    _require(
        "start-local-single" not in install_local,
        "install-local should not chain into start-local-single",
    )
    _require(
        "quickstart-local" not in install_local,
        "install-local should not chain into quickstart-local",
    )


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


def test_quickstart_local_installs_before_setup_and_start() -> None:
    """Local quickstart should be runnable from a clean checkout."""
    text = _read_makefile()
    quickstart_local = _target_block(text, "quickstart-local")

    _require(
        "install-local setup-local-single start-local-single" in quickstart_local,
        "quickstart-local should install before setup/start",
    )


def test_setup_docker_multi_uses_shell_env_for_admin_bootstrap_secrets() -> None:
    """Multi-user setup should keep admin bootstrap secrets out of process argv."""
    text = _read_makefile()
    setup_docker_multi = _target_block(text, "setup-docker-multi")

    for name in ("ADMIN_USERNAME", "ADMIN_PASSWORD", "ADMIN_EMAIL"):
        _require(f"$({name})" not in setup_docker_multi, f"setup-docker-multi should not expand $({name})")

    _require('test -n "$$ADMIN_USERNAME"' in setup_docker_multi, "Expected shell env username check")
    _require('test -n "$$ADMIN_PASSWORD"' in setup_docker_multi, "Expected shell env password check")
    _require(
        'ADMIN_USERNAME="$$ADMIN_USERNAME"' in setup_docker_multi,
        "Expected shell env username assignment",
    )
    _require(
        'ADMIN_PASSWORD="$$ADMIN_PASSWORD"' in setup_docker_multi,
        "Expected shell env password assignment",
    )
    _require(
        'ADMIN_EMAIL="$$ADMIN_EMAIL"' in setup_docker_multi,
        "Expected shell env email assignment",
    )
    for flag in ("--admin-username", "--admin-password", "--admin-email"):
        _require(flag not in setup_docker_multi, f"setup-docker-multi should not pass {flag} in argv")


def test_public_docker_start_paths_quote_paths_and_wait_for_readiness() -> None:
    """Public Docker start commands should handle paths with spaces and wait before verify."""
    text = _read_makefile()

    _require("DOCKER_WAIT_FLAG ?= --wait" in text, "Expected Docker readiness wait helper variable")

    expected_fragments = {
        "start-docker-single": (
            '--env-file "$(TLDW_ENV_FILE)"',
            '-f "$(DOCKER_SINGLE_COMPOSE)"',
            '-f "$(DOCKER_WEBUI_COMPOSE)"',
            "$(DOCKER_WAIT_FLAG)",
        ),
        "start-docker-multi": (
            'TLDW_ENV_FILE="$$TLDW_ENV_FILE_ABS"',
            '-f "$(DOCKER_MULTI_COMPOSE)"',
            "$(DOCKER_WAIT_FLAG)",
            "config >/dev/null",
        ),
        "quickstart-docker": (
            '--env-file "$(TLDW_ENV_FILE)"',
            '-f "$(DOCKER_SINGLE_COMPOSE)"',
            "$(DOCKER_WAIT_FLAG)",
        ),
    }
    for target, fragments in expected_fragments.items():
        block = _target_block(text, target)
        for fragment in fragments:
            _require(fragment in block, f"{target} should include {fragment}")

    start_docker_multi = _target_block(text, "start-docker-multi")
    _require(
        '--env-file "$(TLDW_ENV_FILE)"' not in start_docker_multi,
        "start-docker-multi should use raw service env_file, not compose --env-file",
    )
    _require(
        "config >/dev/null" in start_docker_multi
        and start_docker_multi.index("config >/dev/null") < start_docker_multi.index(" up -d "),
        "start-docker-multi should validate compose config before starting services",
    )
    _require(
        "config >/dev/null &&" in start_docker_multi
        or "config >/dev/null || exit 1" in start_docker_multi,
        "start-docker-multi should fail fast when compose config validation fails",
    )


def test_quickstart_docker_is_api_only_and_skips_webui_verify() -> None:
    """API-only Docker quickstart should not verify a WebUI it did not start."""
    text = _read_makefile()
    quickstart_docker = _target_block(text, "quickstart-docker")

    _require("$(DOCKER_SINGLE_COMPOSE)" in quickstart_docker, "quickstart-docker should use API compose")
    _require(
        "$(DOCKER_WEBUI_COMPOSE)" not in quickstart_docker,
        "quickstart-docker should not include the WebUI compose overlay",
    )
    _require("$(TLDW_SETUP) verify" in quickstart_docker, "quickstart-docker should verify the API")
    _require(
        '--webui-url ""' in quickstart_docker,
        "quickstart-docker should explicitly skip WebUI verification",
    )


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
