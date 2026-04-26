class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []

    def debug(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.debugs.append(message)


def test_safe_increment_metric_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import chatbooks

    def _raise_increment_counter(*_args, **_kwargs):
        raise RuntimeError("chatbook metrics exploded at /private/chatbooks.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(chatbooks, "increment_counter", _raise_increment_counter)
    monkeypatch.setattr(chatbooks, "logger", logger_stub)

    chatbooks._safe_increment_metric(
        "chatbooks.secret.metric",
        labels={"user_id": "secret-user"},
        error_context="private-export-context",
    )

    assert logger_stub.debugs == ["metrics increment failed"]
    assert "chatbooks.secret.metric" not in str(logger_stub.debugs)
    assert "secret-user" not in str(logger_stub.debugs)
    assert "private-export-context" not in str(logger_stub.debugs)
    assert "chatbook metrics exploded" not in str(logger_stub.debugs)
    assert "/private/chatbooks.db" not in str(logger_stub.debugs)
