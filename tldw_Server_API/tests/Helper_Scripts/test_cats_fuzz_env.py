from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path

import pytest
from fastapi import FastAPI

from Helper_Scripts.cats_fuzz import DEFAULT_TEST_API_KEY
from Helper_Scripts.cats_fuzz import env as cats_env
from Helper_Scripts.cats_fuzz.env import build_child_env, find_sensitive_values
from Helper_Scripts.cats_fuzz.openapi_export import (
    build_openapi_export_command,
    export_openapi,
    main,
)


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
def test_build_child_env_overwrites_single_user_key_without_rejecting(tmp_path: Path) -> None:
    env = build_child_env(tmp_path, parent_env={"SINGLE_USER_API_KEY": "dev-key"})

    assert env["SINGLE_USER_API_KEY"] == DEFAULT_TEST_API_KEY


@pytest.mark.unit
def test_build_child_env_blanks_tool_tokens_without_rejecting(tmp_path: Path) -> None:
    env = build_child_env(
        tmp_path,
        parent_env={"GITHUB_TOKEN": "gh-real", "SAFE": "value"},
    )

    assert env["SAFE"] == "value"
    assert env["GITHUB_TOKEN"] == ""


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
def test_build_server_env_removes_guarded_test_flags_and_writes_safe_env_file(tmp_path: Path) -> None:
    child_env = build_child_env(
        tmp_path,
        parent_env={
            "SAFE": "value",
            "TESTING": "true",
            "TLDW_TEST_MODE": "1",
        },
    )
    assert child_env["TEST_MODE"] == "true"

    server_env = cats_env.build_server_env(tmp_path, child_env)

    for flag in ("TEST_MODE", "TESTING", "TLDW_TEST_MODE"):
        assert server_env.get(flag, "") == ""

    assert server_env["SAFE"] == "value"
    assert server_env["AUTH_MODE"] == "single_user"
    assert server_env["SINGLE_USER_API_KEY"] == DEFAULT_TEST_API_KEY
    assert server_env["SINGLE_USER_TEST_API_KEY"] == DEFAULT_TEST_API_KEY
    assert server_env["DATABASE_URL"].startswith("sqlite:///")
    assert server_env["USER_DB_BASE_DIR"] == child_env["USER_DB_BASE_DIR"]
    assert server_env["MINIMAL_TEST_APP"] == "1"
    assert server_env["MINIMAL_TEST_INCLUDE_AUDIO"] == "1"
    assert server_env["TLDW_ENV_FILE"] != child_env["TLDW_ENV_FILE"]

    server_env_file = Path(server_env["TLDW_ENV_FILE"])
    assert server_env_file.exists()
    server_env_text = server_env_file.read_text(encoding="utf-8")
    assert "TEST_MODE" not in server_env_text
    assert "TESTING" not in server_env_text
    assert "TLDW_TEST_MODE" not in server_env_text


@pytest.mark.unit
def test_openapi_export_command_uses_module_and_output_path(tmp_path: Path) -> None:
    output = tmp_path / "openapi.json"
    command = build_openapi_export_command(output)

    assert command[:3] == ["python", "-m", "Helper_Scripts.cats_fuzz.openapi_export"]
    assert str(output) in command


@pytest.mark.unit
def test_export_openapi_clears_cache_creates_parent_and_writes_deterministic_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_app = FastAPI(title="CATS Test")
    fake_app.openapi_schema = {"stale": True}
    fake_module = types.ModuleType("tldw_Server_API.app.main")
    fake_module.app = fake_app
    monkeypatch.setitem(sys.modules, "tldw_Server_API.app.main", fake_module)
    output = tmp_path / "nested" / "openapi.json"

    digest = export_openapi(output)
    first_bytes = output.read_bytes()
    second_digest = export_openapi(output)
    second_bytes = output.read_bytes()

    assert output.parent.exists()
    assert digest == hashlib.sha256(first_bytes).hexdigest()
    assert second_digest == digest
    assert second_bytes == first_bytes
    schema = json.loads(first_bytes)
    assert schema["info"]["title"] == "CATS Test"
    assert "stale" not in schema


@pytest.mark.unit
def test_openapi_export_main_prints_digest_and_returns_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "openapi.json"
    calls: list[Path] = []

    def fake_export(path: Path) -> str:
        calls.append(path)
        return "abc123"

    monkeypatch.setattr("Helper_Scripts.cats_fuzz.openapi_export.export_openapi", fake_export)

    result = main(["--output", str(output)])

    assert result == 0
    assert calls == [output]
    assert capsys.readouterr().out == "abc123\n"
