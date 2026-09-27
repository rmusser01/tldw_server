from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

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
