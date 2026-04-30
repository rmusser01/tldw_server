class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []

    def debug(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.debugs.append(message)


def test_emit_slack_counter_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import slack

    def _raise_log_counter(*_args, **_kwargs):
        raise RuntimeError("slack metrics exploded at /private/slack-metrics.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(slack, "log_counter", _raise_log_counter)
    monkeypatch.setattr(slack, "logger", logger_stub)

    slack._emit_slack_counter(
        "slack.secret.metric",
        team_id="team-secret",
    )

    assert logger_stub.debugs == ["Failed to emit Slack metric"]
    assert "slack.secret.metric" not in str(logger_stub.debugs)
    assert "team-secret" not in str(logger_stub.debugs)
    assert "slack metrics exploded" not in str(logger_stub.debugs)
    assert "/private/slack-metrics.db" not in str(logger_stub.debugs)
