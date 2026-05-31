class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []

    def debug(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.debugs.append(message)


def test_emit_discord_counter_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import discord

    def _raise_log_counter(*_args, **_kwargs):
        raise RuntimeError("discord metrics exploded at /private/discord-metrics.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(discord, "log_counter", _raise_log_counter)
    monkeypatch.setattr(discord, "logger", logger_stub)

    discord._emit_discord_counter(
        "discord.secret.metric",
        guild_id="guild-secret",
    )

    assert logger_stub.debugs == ["Failed to emit Discord metric"]
    assert "discord.secret.metric" not in str(logger_stub.debugs)
    assert "guild-secret" not in str(logger_stub.debugs)
    assert "discord metrics exploded" not in str(logger_stub.debugs)
    assert "/private/discord-metrics.db" not in str(logger_stub.debugs)
