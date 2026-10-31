from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner

from tldw_Server_API.app.core.Admin_Webhooks import legacy_import
from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    WebhookError,
    WebhookErrorCode,
)
from tldw_Server_API.cli.commands import admin_webhooks


@pytest.mark.unit
def test_command_tree_exposes_import_rollback_and_rotation_operations() -> None:
    runner = CliRunner()

    root = runner.invoke(admin_webhooks.admin_webhooks_group, ["--help"])
    rotation = runner.invoke(
        admin_webhooks.admin_webhooks_group,
        ["rotate-key", "--help"],
    )

    assert root.exit_code == 0
    assert {
        "destroy-rollback-key",
        "extract-rollback-backup",
        "import-legacy",
        "reject-source",
        "rotate-key",
        "rotation-status",
    }.issubset(root.output.split())
    assert rotation.exit_code == 0
    assert {"finalize", "resume", "start", "verify"}.issubset(rotation.output.split())


@pytest.mark.unit
def test_import_apply_requires_quiescence_before_runtime_initialization(
    monkeypatch,
) -> None:
    runtime_called = False

    def fail_if_called(_operation):
        nonlocal runtime_called
        runtime_called = True
        raise AssertionError("runtime must not initialize")

    monkeypatch.setattr(admin_webhooks, "_run", fail_if_called)

    result = CliRunner().invoke(
        admin_webhooks.admin_webhooks_group,
        [
            "import-legacy",
            "--apply",
            "--approved-report-digest",
            "sha256:" + ("a" * 64),
            "--report",
            "report.json",
            "--operator-id",
            "9",
        ],
    )

    assert result.exit_code == 2
    assert "--apply requires --all-writers-quiesced" in result.output
    assert runtime_called is False


@pytest.mark.unit
def test_import_apply_requires_literal_digest_before_runtime_initialization(
    monkeypatch,
) -> None:
    runtime_called = False

    def fail_if_called(_operation):
        nonlocal runtime_called
        runtime_called = True
        raise AssertionError("runtime must not initialize")

    monkeypatch.setattr(admin_webhooks, "_run", fail_if_called)

    result = CliRunner().invoke(
        admin_webhooks.admin_webhooks_group,
        [
            "import-legacy",
            "--apply",
            "--all-writers-quiesced",
            "--report",
            "report.json",
            "--operator-id",
            "9",
        ],
    )

    assert result.exit_code == 2
    assert "--apply requires --approved-report-digest" in result.output
    assert runtime_called is False


@pytest.mark.unit
def test_runtime_preserves_closed_key_rotation_error_code(monkeypatch) -> None:
    async def failing_runtime(_operation):
        raise WebhookError(WebhookErrorCode.KEY_UNAVAILABLE)

    monkeypatch.setattr(admin_webhooks, "_with_runtime", failing_runtime)

    with pytest.raises(click.ClickException) as caught:
        admin_webhooks._run(lambda _importer, _rotation, _repository: None)

    assert str(caught.value) == "admin_webhook_key_unavailable"


@pytest.mark.unit
@pytest.mark.parametrize("operation", ["dry_run", "apply", "extract", "destroy"])
def test_unsupported_artifact_commands_reject_before_runtime_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    """These controls run on Windows too, using the real CLI error boundary."""
    monkeypatch.setattr(legacy_import, "os", SimpleNamespace(**{**vars(os), "name": "nt"}))
    runtime_calls: list[str] = []

    async def forbidden_runtime(_operation):
        runtime_calls.append("database initialization")
        raise AssertionError("runtime details must remain private")

    monkeypatch.setattr(admin_webhooks, "_with_runtime", forbidden_runtime)
    paths = {
        "report": str(tmp_path / "report.json"),
        "backup": str(tmp_path / "backup.enc"),
        "key": str(tmp_path / "rollback.key"),
        "output": str(tmp_path / "plaintext.json"),
    }
    if operation in {"dry_run", "apply"}:
        arguments = ["import-legacy", "--report", paths["report"]]
        if operation == "dry_run":
            arguments.append("--dry-run")
        else:
            arguments.extend([
                "--apply", "--all-writers-quiesced", "--approved-report-digest",
                "sha256:" + "a" * 64,
            ])
    else:
        command = "extract-rollback-backup" if operation == "extract" else "destroy-rollback-key"
        arguments = [command, "--backup", paths["backup"], "--rollback-key-file", paths["key"], "--confirm"]
        if operation == "extract":
            arguments.extend(["--output", paths["output"]])
    arguments.extend(["--operator-id", "9"])

    result = CliRunner().invoke(admin_webhooks.admin_webhooks_group, arguments)

    assert result.exit_code == 1
    assert runtime_calls == []
    assert "admin_webhook_legacy_host_unsupported" in result.output
    assert "runtime details" not in result.output
    assert not tuple(tmp_path.iterdir())


@pytest.mark.unit
@pytest.mark.parametrize(
    "arguments",
    [
        ["rotation-status"],
        ["rotate-key", "start", "--operation-id", "rotation", "--source-key-id", "old", "--target-key-id", "new", "--operator-id", "9"],
        ["rotate-key", "resume", "--operation-id", "rotation", "--operator-id", "9"],
        ["rotate-key", "verify", "--operation-id", "rotation", "--operator-id", "9"],
        ["rotate-key", "finalize", "--operation-id", "rotation", "--operator-id", "9"],
        ["reject-source", "--source-kind", "system_ops", "--source-identity", "legacy", "--source-record-fingerprint", "hmac-sha256:" + "a" * 64, "--reason-code", "operator_excluded", "--operator-id", "9"],
    ],
)
def test_platform_neutral_commands_remain_available_without_artifact_capability(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str],
) -> None:
    """The artifact guard must not become a global admin-webhook platform lock."""
    monkeypatch.setattr(legacy_import, "os", SimpleNamespace(**{**vars(os), "name": "nt"}))

    async def runtime(_operation):
        return {"available": True}

    monkeypatch.setattr(admin_webhooks, "_with_runtime", runtime)

    result = CliRunner().invoke(admin_webhooks.admin_webhooks_group, arguments)

    assert result.exit_code == 0
    assert result.output.strip() == '{"available":true}'
