"""Managed container startup must retain editable configuration and credentials."""

from __future__ import annotations

import base64
import configparser
import os
import subprocess  # nosec B404
import sys
from pathlib import Path

import pytest
import yaml
from hypothesis import given
from hypothesis import strategies as st

REPO = Path(__file__).resolve().parents[3]
PACKAGED_CONFIG = REPO / "tldw_Server_API" / "Config_Files"
ENTRYPOINT = REPO / "Dockerfiles" / "entrypoints" / "tldw-app-first-run.sh"


@pytest.fixture
def persistent_config(tmp_path: Path) -> Path:
    """Supply the private .env layout of an already initialized paired instance."""
    directory = tmp_path / "managed-config"
    directory.mkdir()
    (directory / ".env").write_text(
        "AUTH_MODE=single_user\n"
        "SINGLE_USER_API_KEY=disposable-managed-config-test-key\n"
        "MCP_JWT_SECRET=disposable-managed-config-test-jwt-secret\n"
        "MCP_API_KEY_SALT=disposable-managed-config-test-api-salt\n"
        "DATABASE_URL=sqlite:///./Databases/users.db\n"
        "JOBS_DB_URL=sqlite:///./Databases/users.db\n"
        f"BYOK_ENCRYPTION_KEY={base64.urlsafe_b64encode(b'x' * 32).decode()}\n",
        encoding="utf-8",
    )
    return directory


def run_entrypoint(
    directory: Path, *command: str, managed: bool = True, set_config_dir: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run the actual startup script with an isolated, disposable environment."""
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONPATH": str(REPO),
        "PYTHON_BIN": sys.executable,
        "TLDW_ENV_FILE": str(directory / ".env"),
        "TLDW_AUTH_MARKER_DIR": str(directory.parent / "markers"),
        "TLDW_RUN_AUTH_INIT_ON_START": "0",
    }
    if managed:
        env["TLDW_MANAGED_GATEWAY"] = "1"
    if set_config_dir:
        env["TLDW_CONFIG_DIR"] = str(directory)
    return subprocess.run(  # nosec B603 -- fixed repository script and local test commands
        ["/bin/sh", str(ENTRYPOINT), *(command or ("/usr/bin/true",))],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def test_paired_compose_resolves_configuration_inside_its_existing_volume() -> None:
    app = yaml.safe_load((REPO / "Dockerfiles/app-bundle/compose.yaml").read_text())["services"]["app"]

    assert app["environment"].get("TLDW_CONFIG_DIR") == "/app/managed-config"
    assert "backend_config:/app/managed-config" in app["volumes"]


def test_managed_entrypoint_initializes_configuration_assets(persistent_config: Path) -> None:
    result = run_entrypoint(persistent_config)

    assert result.returncode == 0, result.stderr
    assert (persistent_config / "config.txt").is_file()
    assert (persistent_config / "config.txt").read_bytes() == (PACKAGED_CONFIG / "config.txt").read_bytes()
    assert (persistent_config / "tts_providers_config.yaml").read_bytes() == (
        PACKAGED_CONFIG / "tts_providers_config.yaml"
    ).read_bytes()
    assert (persistent_config / "Prompts/chat.prompts.yaml").is_file()


def test_managed_entrypoint_preserves_existing_settings_and_credentials(persistent_config: Path) -> None:
    original_env = (persistent_config / ".env").read_bytes()
    (persistent_config / "config.txt").write_text("[API]\ncustom_openai_api_model = user-model\n")
    (persistent_config / "tts_providers_config.yaml").write_text("# User's edited voice settings\n")

    result = run_entrypoint(persistent_config)

    assert result.returncode == 0, result.stderr
    assert (persistent_config / "config.txt").read_text() == "[API]\ncustom_openai_api_model = user-model\n"
    assert (persistent_config / "tts_providers_config.yaml").read_text() == "# User's edited voice settings\n"
    assert (persistent_config / ".env").read_bytes() == original_env


def test_setup_provider_write_survives_repeated_managed_startup(persistent_config: Path) -> None:
    first = run_entrypoint(
        persistent_config,
        sys.executable,
        "-c",
        "from tldw_Server_API.app.core.Setup.setup_manager import update_config; "
        "update_config({'API': {'custom_openai_api_model': 'retained-model', "
        "'custom_openai_api_ip': 'http://provider.invalid/v1', "
        "'custom_openai_api_key': 'disposable-provider-key', 'default_api': 'custom-openai-api'}})",
    )
    second = run_entrypoint(persistent_config)
    config = configparser.ConfigParser(interpolation=None)
    config.read(persistent_config / "config.txt")

    assert first.returncode == second.returncode == 0, first.stderr + second.stderr
    assert config.get("API", "custom_openai_api_model", fallback="") == "retained-model"
    assert config.get("API", "custom_openai_api_ip", fallback="") == "http://provider.invalid/v1"
    assert config.get("API", "custom_openai_api_key", fallback="") == "disposable-provider-key"
    assert config.get("API", "default_api", fallback="") == "custom-openai-api"


def test_managed_startup_requires_an_explicit_persistent_config_root(persistent_config: Path) -> None:
    result = run_entrypoint(persistent_config, set_config_dir=False)

    assert result.returncode != 0
    assert "TLDW_CONFIG_DIR" in result.stderr


def test_non_managed_startup_does_not_seed_configuration(persistent_config: Path) -> None:
    result = run_entrypoint(persistent_config, managed=False)

    assert result.returncode == 0, result.stderr
    assert not (persistent_config / "config.txt").exists()


def test_config_seeding_preserves_existing_files_and_excludes_runtime_secrets(tmp_path: Path) -> None:
    from tldw_Server_API.scripts.initialize_managed_config import initialize_managed_config

    source = tmp_path / "defaults"
    source.mkdir()
    (source / "config.txt").write_text("[API]\nmodel = default-model\n")
    (source / ".env").write_text("packaged-env-must-not-be-used")
    (source / "session_encryption.key").write_text("packaged-key-must-not-be-used")
    (source / "config.txt.pre-setup.bak").write_text("obsolete configuration")
    (source / "Prompts").mkdir()
    (source / "Prompts/chat.prompts.yaml").write_text("default prompt")
    destination = tmp_path / "persistent"
    destination.mkdir()
    (destination / "config.txt").write_text("user's provider configuration")
    (destination / ".env").write_text("retained credentials")

    initialize_managed_config(destination, source=source)

    assert (destination / "config.txt").read_text() == "user's provider configuration"
    assert (destination / ".env").read_text() == "retained credentials"
    assert (destination / "Prompts/chat.prompts.yaml").read_text() == "default prompt"
    assert not (destination / "session_encryption.key").exists()
    assert not (destination / "config.txt.pre-setup.bak").exists()


@given(contents=st.binary(max_size=512))
def test_repeated_initialization_never_overwrites_user_configuration(contents: bytes) -> None:
    from tempfile import TemporaryDirectory

    from tldw_Server_API.scripts.initialize_managed_config import initialize_managed_config

    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "defaults"
        source.mkdir()
        (source / "config.txt").write_text("packaged default")
        destination = root / "persistent"
        initialize_managed_config(destination, source=source)
        (destination / "config.txt").write_bytes(contents)

        initialize_managed_config(destination, source=source)

        assert (destination / "config.txt").read_bytes() == contents


def test_seed_failure_never_leaves_partial_configuration(tmp_path: Path, monkeypatch) -> None:
    from tldw_Server_API.scripts import initialize_managed_config as module

    source = tmp_path / "defaults"
    source.mkdir()
    (source / "config.txt").write_text("packaged default")
    destination = tmp_path / "persistent"

    def interrupted_copy(_source, target) -> None:
        target.write(b"partial")
        raise OSError("interrupted test copy")

    monkeypatch.setattr(module.shutil, "copyfileobj", interrupted_copy)
    with pytest.raises(OSError, match="interrupted test copy"):
        module.initialize_managed_config(destination, source=source)

    assert not (destination / "config.txt").exists()
    assert list(destination.iterdir()) == []


def test_missing_default_config_fails_without_touching_existing_credentials(tmp_path: Path) -> None:
    from tldw_Server_API.scripts.initialize_managed_config import initialize_managed_config

    source = tmp_path / "defaults"
    source.mkdir()
    destination = tmp_path / "persistent"
    destination.mkdir()
    (destination / ".env").write_text("retained credentials")

    with pytest.raises(ValueError, match="config.txt"):
        initialize_managed_config(destination, source=source)

    assert (destination / ".env").read_text() == "retained credentials"


def test_config_directory_conflict_refuses_startup_without_overwriting(tmp_path: Path) -> None:
    from tldw_Server_API.scripts.initialize_managed_config import initialize_managed_config

    source = tmp_path / "defaults"
    source.mkdir()
    (source / "config.txt").write_text("packaged default")
    destination = tmp_path / "persistent"
    destination.mkdir()
    (destination / "config.txt").mkdir()

    with pytest.raises(ValueError, match="config.txt"):
        initialize_managed_config(destination, source=source)

    assert (destination / "config.txt").is_dir()


@pytest.mark.parametrize("asset", ["config.txt", "Prompts"])
def test_existing_symlinks_are_refused_without_writing_outside_volume(tmp_path: Path, asset: str) -> None:
    from tldw_Server_API.scripts.initialize_managed_config import initialize_managed_config

    source = tmp_path / "defaults"
    source.mkdir()
    (source / "config.txt").write_text("packaged default")
    (source / "Prompts").mkdir()
    (source / "Prompts/chat.yaml").write_text("packaged prompt")
    destination = tmp_path / "persistent"
    destination.mkdir()
    outside = tmp_path / "outside"
    if asset == "config.txt":
        outside.write_text("retained outside file")
    else:
        outside.mkdir()
    (destination / asset).symlink_to(outside)

    with pytest.raises(ValueError, match="symlink"):
        initialize_managed_config(destination, source=source)

    if asset == "config.txt":
        assert outside.read_text() == "retained outside file"
    else:
        assert list(outside.iterdir()) == []


def test_initializer_cli_reports_failure_without_starting_server(persistent_config: Path) -> None:
    (persistent_config / "config.txt").mkdir()

    result = run_entrypoint(persistent_config)

    assert result.returncode != 0
    assert "Persistent configuration initialization failed" in result.stderr
    assert "Persistent configuration asset must be a file: config.txt" in result.stderr
