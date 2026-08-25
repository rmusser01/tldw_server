"""
Unit tests for MCP secret generation during AuthNZ initialization.
"""

from pathlib import Path

import pytest

from tldw_Server_API.app.core.AuthNZ import initialize as initialize_module
from tldw_Server_API.app.core.AuthNZ.initialize import (
    check_environment,
    _detect_env_issues,
    _ensure_env_file,
    _resolve_env_locations,
    generate_secure_keys,
)

pytestmark = pytest.mark.unit


def test_detect_env_issues_requires_mcp_secrets(monkeypatch):
    monkeypatch.delenv("MCP_JWT_SECRET", raising=False)
    monkeypatch.delenv("MCP_API_KEY_SALT", raising=False)

    env_values = {"SINGLE_USER_API_KEY": "a" * 32}
    missing_keys, _issues = _detect_env_issues("single_user", env_values)

    assert missing_keys == {"MCP_JWT_SECRET", "MCP_API_KEY_SALT"}


def test_generate_secure_keys_includes_mcp_secrets():
    keys = generate_secure_keys(requested_keys={"MCP_JWT_SECRET", "MCP_API_KEY_SALT"})

    assert all(
        key in keys and len(keys[key]) >= 32
        for key in ("MCP_JWT_SECRET", "MCP_API_KEY_SALT")
    )


def test_detect_env_issues_allows_quickstart_default_single_user_key(monkeypatch):
    monkeypatch.setenv("SINGLE_USER_API_KEY", "THIS-IS-A-SECURE-KEY-123-FAKE-KEY")
    env_values = {
        "SINGLE_USER_API_KEY": "THIS-IS-A-SECURE-KEY-123-FAKE-KEY",
        "MCP_JWT_SECRET": "x" * 32,
        "MCP_API_KEY_SALT": "y" * 32,
    }
    missing_keys, issues = _detect_env_issues("single_user", env_values)

    assert "SINGLE_USER_API_KEY" not in missing_keys
    assert not any("default placeholder" in issue for issue in issues)


def test_resolve_env_locations_prefers_config_files_env_path():
    env_candidates, _template_candidates, cfg_dir = _resolve_env_locations()

    assert env_candidates == [cfg_dir / ".env", cfg_dir / ".ENV"]
    assert cfg_dir.name == "Config_Files"
    assert isinstance(cfg_dir, Path)


def test_authnz_initializer_excludes_canonical_env_in_exclusive_mode(
    tmp_path: Path,
    monkeypatch,
):
    sentinel = "TLDW_AUTH_INIT_EXCLUSIVE_SENTINEL"
    project_root = tmp_path / "tldw_Server_API"
    canonical_env = project_root / "Config_Files" / ".env"
    explicit_env = tmp_path / "runtime.env"
    canonical_env.parent.mkdir(parents=True)
    canonical_env.write_text(f"{sentinel}=host-secret\n", encoding="utf-8")
    explicit_env.write_text(
        "AUTH_MODE=single_user\n"
        f"SINGLE_USER_API_KEY={'a' * 32}\n"
        f"MCP_JWT_SECRET={'b' * 32}\n"
        f"MCP_API_KEY_SALT={'c' * 32}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        initialize_module,
        "__file__",
        str(project_root / "app/core/AuthNZ/initialize.py"),
    )
    monkeypatch.setenv("TLDW_ENV_FILE", str(explicit_env))
    monkeypatch.setenv("TLDW_ENV_FILE_EXCLUSIVE", "true")
    monkeypatch.delenv(sentinel, raising=False)

    result = check_environment()

    assert result["env_path"] == explicit_env.resolve()  # nosec B101
    assert sentinel not in initialize_module.os.environ  # nosec B101


def test_authnz_initializer_does_not_create_missing_exclusive_env_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    missing_env = tmp_path / "missing.env"
    monkeypatch.setenv("TLDW_ENV_FILE", str(missing_env))
    monkeypatch.setenv("TLDW_ENV_FILE_EXCLUSIVE", "true")

    with pytest.raises(FileNotFoundError, match="TLDW_ENV_FILE_EXCLUSIVE"):
        _ensure_env_file()

    assert not missing_env.exists()  # nosec B101


def test_prompt_yes_no_uses_yes_default_when_stdin_is_closed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _closed_stdin(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _closed_stdin)

    result = initialize_module._prompt_yes_no(
        "Generate missing keys?",
        default_yes=True,
        non_interactive=False,
    )

    assert result is True
    assert "No interactive input detected; using default: yes" in capsys.readouterr().out


def test_prompt_yes_no_uses_no_default_when_stdin_is_closed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def _closed_stdin(_prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _closed_stdin)

    result = initialize_module._prompt_yes_no(
        "Generate replacement keys?",
        default_yes=False,
        non_interactive=False,
    )

    assert result is False
    assert "No interactive input detected; using default: no" in capsys.readouterr().out
