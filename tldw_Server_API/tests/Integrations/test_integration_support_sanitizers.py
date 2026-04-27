import pytest


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def debug(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.debugs.append(message)

    def error(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.errors.append(message)

    def warning(self, message: str, *args, **kwargs) -> None:
        if args or kwargs:
            message = message.format(*args, **kwargs)
        self.warnings.append(message)


def test_emit_discord_support_counter_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import discord_support

    def _raise_log_counter(*_args, **_kwargs):
        raise RuntimeError("discord support metrics exploded at /private/discord-support.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(discord_support, "log_counter", _raise_log_counter)
    monkeypatch.setattr(discord_support, "logger", logger_stub)

    discord_support._emit_discord_counter(
        "discord.support.secret.metric",
        guild_id="guild-secret",
    )

    assert logger_stub.debugs == ["Failed to emit Discord metric"]
    assert "discord.support.secret.metric" not in str(logger_stub.debugs)
    assert "guild-secret" not in str(logger_stub.debugs)
    assert "discord support metrics exploded" not in str(logger_stub.debugs)
    assert "/private/discord-support.db" not in str(logger_stub.debugs)


def test_emit_slack_support_counter_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import slack_support

    def _raise_log_counter(*_args, **_kwargs):
        raise RuntimeError("slack support metrics exploded at /private/slack-support.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(slack_support, "log_counter", _raise_log_counter)
    monkeypatch.setattr(slack_support, "logger", logger_stub)

    slack_support._emit_slack_counter(
        "slack.support.secret.metric",
        team_id="team-secret",
    )

    assert logger_stub.debugs == ["Failed to emit Slack metric"]
    assert "slack.support.secret.metric" not in str(logger_stub.debugs)
    assert "team-secret" not in str(logger_stub.debugs)
    assert "slack support metrics exploded" not in str(logger_stub.debugs)
    assert "/private/slack-support.db" not in str(logger_stub.debugs)


def test_decrypt_discord_payload_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import discord_support

    def _raise_loads_envelope(_encrypted_blob: str):
        raise RuntimeError("discord decrypt failed at /private/discord-secrets.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(discord_support, "loads_envelope", _raise_loads_envelope)
    monkeypatch.setattr(discord_support, "logger", logger_stub)

    result = discord_support._decrypt_discord_payload("discord-encrypted-secret")

    assert result is None
    assert logger_stub.warnings == ["Failed to decrypt Discord installation payload"]
    assert "discord-encrypted-secret" not in str(logger_stub.warnings)
    assert "discord decrypt failed" not in str(logger_stub.warnings)
    assert "/private/discord-secrets.db" not in str(logger_stub.warnings)


def test_decrypt_slack_payload_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import slack_support

    def _raise_loads_envelope(_encrypted_blob: str):
        raise RuntimeError("slack decrypt failed at /private/slack-secrets.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(slack_support, "loads_envelope", _raise_loads_envelope)
    monkeypatch.setattr(slack_support, "logger", logger_stub)

    result = slack_support._decrypt_slack_payload("slack-encrypted-secret")

    assert result is None
    assert logger_stub.warnings == ["Failed to decrypt Slack installation payload"]
    assert "slack-encrypted-secret" not in str(logger_stub.warnings)
    assert "slack decrypt failed" not in str(logger_stub.warnings)
    assert "/private/slack-secrets.db" not in str(logger_stub.warnings)


def test_decrypt_telegram_payload_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import telegram_support

    def _raise_loads_envelope(_encrypted_blob: str):
        raise RuntimeError("telegram decrypt failed at /private/telegram-secrets.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(telegram_support, "loads_envelope", _raise_loads_envelope)
    monkeypatch.setattr(telegram_support, "logger", logger_stub)

    result = telegram_support._decrypt_telegram_payload("telegram-encrypted-secret")

    assert result is None
    assert logger_stub.warnings == ["Failed to decrypt Telegram bot config payload"]
    assert "telegram-encrypted-secret" not in str(logger_stub.warnings)
    assert "telegram decrypt failed" not in str(logger_stub.warnings)
    assert "/private/telegram-secrets.db" not in str(logger_stub.warnings)


@pytest.mark.asyncio
async def test_telegram_webhook_scope_list_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import telegram_support

    class _Repo:
        async def list_secrets(self, *, provider: str):
            raise RuntimeError("telegram config list exploded at /private/telegram-secrets.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(telegram_support, "logger", logger_stub)

    with pytest.raises(telegram_support.HTTPException) as exc_info:
        await telegram_support._resolve_webhook_scope_from_secret(
            repo=_Repo(),
            webhook_secret="webhook-secret",
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Telegram bot configuration is unavailable"
    assert logger_stub.errors == ["Failed to list Telegram bot configs for webhook resolution"]
    assert "telegram config list exploded" not in str(logger_stub.errors)
    assert "/private/telegram-secrets.db" not in str(logger_stub.errors)


@pytest.mark.asyncio
async def test_telegram_webhook_scope_fetch_failure_log_is_sanitized(monkeypatch):
    from tldw_Server_API.app.api.v1.endpoints import telegram_support

    class _Repo:
        async def list_secrets(self, *, provider: str):
            return [{"scope_type": "team", "scope_id": 42}]

        async def fetch_secret(self, scope_type: str, scope_id: int, provider: str):
            raise RuntimeError("telegram config fetch exploded at /private/telegram-secrets.db")

    logger_stub = _LoggerStub()
    monkeypatch.setattr(telegram_support, "logger", logger_stub)

    with pytest.raises(telegram_support.HTTPException) as exc_info:
        await telegram_support._resolve_webhook_scope_from_secret(
            repo=_Repo(),
            webhook_secret="webhook-secret",
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Telegram bot configuration is unavailable"
    assert logger_stub.errors == ["Failed to load Telegram bot config for webhook resolution"]
    assert "team" not in str(logger_stub.errors)
    assert "42" not in str(logger_stub.errors)
    assert "telegram config fetch exploded" not in str(logger_stub.errors)
    assert "/private/telegram-secrets.db" not in str(logger_stub.errors)
