from __future__ import annotations

from tldw_Server_API.cli.wizard.utils import format as format_utils


def test_maybe_format_limits_tools_to_python_files(monkeypatch) -> None:
    commands: list[list[str]] = []

    monkeypatch.setattr(format_utils, "_tool_path", lambda name: name)
    monkeypatch.setattr(
        format_utils.subprocess,
        "run",
        lambda command, *, check: commands.append(command),
    )

    format_utils.maybe_format(["Docs/plan.md", "tldw_Server_API/app.py", "types.pyi"])

    assert commands == [
        ["black", "tldw_Server_API/app.py", "types.pyi"],
        ["ruff", "check", "--fix", "tldw_Server_API/app.py", "types.pyi"],
    ]


def test_maybe_format_skips_when_no_python_files(monkeypatch) -> None:
    commands: list[list[str]] = []

    monkeypatch.setattr(format_utils, "_tool_path", lambda name: name)
    monkeypatch.setattr(
        format_utils.subprocess,
        "run",
        lambda command, *, check: commands.append(command),
    )

    format_utils.maybe_format(["Docs/plan.md", "README.md"])

    assert commands == []
