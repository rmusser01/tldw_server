from __future__ import annotations

import os
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Sandbox.runners.seatbelt_policy import (
    build_seatbelt_env,
    render_seatbelt_profile,
    resolve_command_argv,
)


def test_render_seatbelt_profile_limits_writes_to_workspace_and_temp() -> None:
    profile = render_seatbelt_profile(
        command_path="/bin/echo",
        workspace_path="/tmp/workspace",
        home_path="/tmp/home",
        temp_path="/tmp/temp",
        network_policy="deny_all",
    )

    assert "(version 1)" in profile
    assert "(deny default)" in profile
    assert "(allow process-exec" in profile
    assert '(literal "/bin/echo")' in profile
    assert '(subpath "/tmp/workspace")' in profile
    assert '(subpath "/tmp/home")' in profile
    assert '(subpath "/tmp/temp")' in profile
    assert '(deny network*)' in profile
    assert "/tmp/control" not in profile


def test_render_seatbelt_profile_rejects_allowlist_policy() -> None:
    with pytest.raises(ValueError, match="allowlist"):
        render_seatbelt_profile(
            command_path="/bin/echo",
            workspace_path="/tmp/workspace",
            home_path="/tmp/home",
            temp_path="/tmp/temp",
            network_policy="allowlist",
        )


def test_build_seatbelt_env_does_not_inherit_unexpected_host_env(monkeypatch) -> None:
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "should-not-leak")

    env = build_seatbelt_env(
        workspace_path="/tmp/workspace",
        home_path="/tmp/home",
        temp_path="/tmp/temp",
        spec_env={"LANG": "C", "CUSTOM": "1"},
    )

    assert env["HOME"] == "/tmp/home"
    assert env["TMPDIR"] == "/tmp/temp"
    assert env["TMP"] == "/tmp/temp"
    assert env["TEMP"] == "/tmp/temp"
    assert env["PWD"] == "/tmp/workspace"
    assert env["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"
    assert env["LANG"] == "C"
    assert env["CUSTOM"] == "1"
    assert "AWS_SECRET_ACCESS_KEY" not in env


def test_build_seatbelt_env_does_not_allow_path_override() -> None:
    env = build_seatbelt_env(
        workspace_path="/tmp/workspace",
        home_path="/tmp/home",
        temp_path="/tmp/temp",
        spec_env={"PATH": "/tmp/evil", "CUSTOM": "1"},
    )

    assert env["PATH"] == "/usr/bin:/bin:/usr/sbin:/sbin"
    assert env["CUSTOM"] == "1"


def test_render_seatbelt_profile_escapes_embedded_quotes_and_newlines() -> None:
    profile = render_seatbelt_profile(
        command_path='/tmp/evil"cmd\nnext',
        workspace_path="/tmp/workspace",
        home_path="/tmp/home",
        temp_path="/tmp/temp",
        network_policy="deny_all",
    )

    assert 'literal "/tmp/evil\\"cmd\\nnext"' in profile


def test_resolve_command_argv_uses_controlled_path(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    tool_name = "runner-tool.cmd" if os.name == "nt" else "runner-tool"
    tool = bin_dir / tool_name
    tool.write_text("@echo off\r\nexit /b 0\r\n" if os.name == "nt" else "#!/bin/sh\nexit 0\n")
    tool.chmod(0o755)

    argv = resolve_command_argv([tool_name, "--flag"], str(bin_dir))

    assert argv == [str(tool), "--flag"]
