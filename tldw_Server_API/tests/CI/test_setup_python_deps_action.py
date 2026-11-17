"""Exercise the checked-in bootstrap shell without downloading an installer."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

pytest_plugins = ["pytester"]
pytestmark = pytest.mark.unit

ACTION_PATH = Path(__file__).resolve().parents[3] / ".github/actions/setup-python-deps/action.yml"


@pytest.mark.parametrize("installer_exit", [0, 37], ids=["success", "installer-failure"])
def test_uv_bootstrap_uses_configured_python_and_propagates_failure(
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
    installer_exit: int,
) -> None:
    steps = yaml.safe_load(ACTION_PATH.read_text(encoding="utf-8"))["runs"]["steps"]
    bootstrap = next(step for step in steps if step["name"] == "Setup uv")
    assert bootstrap["if"] == "inputs.use-uv == 'true'"
    assert steps.index(bootstrap) > max(
        index for index, step in enumerate(steps) if str(step.get("uses", "")).startswith("actions/setup-python@")
    )
    assert "run" in bootstrap, "uv bootstrap must use the configured Python's installer"

    bash = shutil.which("bash")
    assert bash is not None
    installer = pytester.path / "python"
    installer.write_text(
        f'#!/bin/sh\nprintf "%s\\n" "$@" > installer-argv.txt\nexit {installer_exit}\n',
        encoding="utf-8",
    )
    installer.chmod(0o755)

    # Expose only the installer double, so a different command cannot install anything.
    with monkeypatch.context() as command_env:
        command_env.setenv("PATH", str(pytester.path))
        result = pytester.run(
            bash,
            "--noprofile",
            "--norc",
            "-e",
            "-o",
            "pipefail",
            "-c",
            bootstrap["run"],
            timeout=10,
        )

    assert result.ret == installer_exit
    assert (pytester.path / "installer-argv.txt").read_text().splitlines() == [
        "-m",
        "pip",
        "install",
        "--only-binary=:all:",
        "--index-url",
        "https://pypi.org/simple",
        "uv==0.13.7",
    ]
