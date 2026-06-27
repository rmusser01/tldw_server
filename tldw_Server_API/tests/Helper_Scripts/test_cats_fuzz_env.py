from __future__ import annotations

from pathlib import Path

import pytest

from Helper_Scripts.cats_fuzz import DEFAULT_TEST_API_KEY
from Helper_Scripts.cats_fuzz.env import build_child_env, find_sensitive_values
from Helper_Scripts.cats_fuzz.openapi_export import build_openapi_export_command


@pytest.mark.unit
def test_find_sensitive_values_detects_provider_keys() -> None:
    found = find_sensitive_values({"OPENAI_API_KEY": "sk-real", "SAFE": "value"})

    assert found == {"OPENAI_API_KEY": "set"}


@pytest.mark.unit
def test_find_sensitive_values_ignores_empty_sensitive_values() -> None:
    found = find_sensitive_values({"ANTHROPIC_API_KEY": "", "SAFE": "value"})

    assert found == {}


@pytest.mark.unit
def test_build_child_env_rejects_real_credentials_by_default(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="real credentials"):
        build_child_env(tmp_path, parent_env={"OPENAI_API_KEY": "sk-real"})


@pytest.mark.unit
def test_build_child_env_sets_test_paths_and_sentinels(tmp_path: Path) -> None:
    env = build_child_env(tmp_path, parent_env={}, allow_external=False)

    assert env["AUTH_MODE"] == "single_user"
    assert env["SINGLE_USER_API_KEY"] == DEFAULT_TEST_API_KEY
    assert env["SINGLE_USER_TEST_API_KEY"] == DEFAULT_TEST_API_KEY
    assert env["DATABASE_URL"].startswith("sqlite:///")
    assert Path(env["TLDW_ENV_FILE"]).exists()
    assert env["OPENAI_API_KEY"] == ""


@pytest.mark.unit
def test_build_child_env_blanks_sensitive_parent_values_when_external_allowed(tmp_path: Path) -> None:
    env = build_child_env(
        tmp_path,
        parent_env={"OPENAI_API_KEY": "sk-real", "SAFE": "value"},
        allow_external=True,
    )

    assert env["SAFE"] == "value"
    assert env["OPENAI_API_KEY"] == ""


@pytest.mark.unit
def test_openapi_export_command_uses_module_and_output_path(tmp_path: Path) -> None:
    output = tmp_path / "openapi.json"
    command = build_openapi_export_command(output)

    assert command[:3] == ["python", "-m", "Helper_Scripts.cats_fuzz.openapi_export"]
    assert str(output) in command
