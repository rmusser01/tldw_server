"""The ChatOps counter-emit failure log must not leak the metric or its labels.

This replaces test_discord_endpoint_sanitizers.py and test_slack_endpoint_sanitizers.py,
which were 100% identical after normalising the provider vocabulary -- the same
assertions written twice, so a third ChatOps provider would have meant a third copy,
and a fix to the assertion would have needed two edits.

See ADR-050.
"""

from __future__ import annotations

import pytest


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []

    def debug(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.debugs.append(message)


# module name, counter attribute, expected log line, scope label, scope value
CHATOPS_PROVIDERS = [
    pytest.param(
        "discord", "_emit_discord_counter", "Failed to emit Discord metric",
        "guild_id", "guild-secret", id="discord",
    ),
    pytest.param(
        "slack", "_emit_slack_counter", "Failed to emit Slack metric",
        "team_id", "team-secret", id="slack",
    ),
]


@pytest.mark.parametrize(
    ("module_name", "counter_attr", "expected_log", "scope_label", "scope_value"),
    CHATOPS_PROVIDERS,
)
def test_emit_counter_failure_log_is_sanitized(
    monkeypatch,
    module_name: str,
    counter_attr: str,
    expected_log: str,
    scope_label: str,
    scope_value: str,
):
    import importlib

    module = importlib.import_module(
        f"tldw_Server_API.app.api.v1.endpoints.{module_name}"
    )
    secret_metric = f"{module_name}.secret.metric"
    secret_path = f"/private/{module_name}-metrics.db"
    raised = f"{module_name} metrics exploded at {secret_path}"

    def _raise_log_counter(*_args, **_kwargs):
        raise RuntimeError(raised)

    logger_stub = _LoggerStub()
    monkeypatch.setattr(module, "log_counter", _raise_log_counter)
    monkeypatch.setattr(module, "logger", logger_stub)

    getattr(module, counter_attr)(secret_metric, **{scope_label: scope_value})

    logged = str(logger_stub.debugs)
    assert logger_stub.debugs == [expected_log]
    assert secret_metric not in logged
    assert scope_value not in logged
    assert raised not in logged
    assert secret_path not in logged
