import os
import json
import builtins
from pathlib import Path

from tenacity import Future, RetryError

import tldw_Server_API.app.core.Monitoring.notification_service as notification_service
from tldw_Server_API.app.core.Monitoring.notification_service import NotificationService
from tldw_Server_API.app.core.DB_Management.TopicMonitoring_DB import TopicAlert


def _retry_error(message: str) -> RetryError:
    attempt = Future(3)
    attempt.set_exception(RuntimeError(message))
    return RetryError(attempt)


def _capture_notification_logs(level: str = "DEBUG") -> tuple[list[str], int]:
    messages: list[str] = []
    sink_id = notification_service.logger.add(
        lambda message: messages.append(str(message.record.get("message") or "")),
        level=level,
    )
    return messages, sink_id


def test_notification_threshold_and_file_sink(tmp_path, monkeypatch):


    out = tmp_path / "notifs.log"
    monkeypatch.setenv("MONITORING_NOTIFY_ENABLED", "true")
    monkeypatch.setenv("MONITORING_NOTIFY_MIN_SEVERITY", "critical")
    monkeypatch.setenv("MONITORING_NOTIFY_FILE", str(out))

    svc = NotificationService()
    assert svc.get_notification_file_path() == str(out)

    # Below threshold (warning) should not write
    a1 = TopicAlert(
        user_id="u",
        scope_type="user",
        scope_id="u",
        source="chat.input",
        watchlist_id="w",
        rule_category="adult",
        rule_severity="warning",
        pattern="nsfw",
        text_snippet="...nsfw...",
    )
    result = svc.notify(a1)
    assert result == "skipped"
    assert not out.exists() or out.read_text() == ""

    # Meets threshold (critical) should write
    a2 = TopicAlert(
        user_id="u",
        scope_type="user",
        scope_id="u",
        source="chat.input",
        watchlist_id="w",
        rule_category="self_harm",
        rule_severity="critical",
        pattern="suicide",
        text_snippet="...",
    )
    result = svc.notify(a2)
    assert result == "logged"
    text = out.read_text()
    assert "self_harm" in text and "critical" in text


def test_notification_handles_invalid_smtp_port(monkeypatch):


    monkeypatch.setenv("MONITORING_NOTIFY_SMTP_PORT", "not-a-number")

    svc = NotificationService()
    assert svc.smtp_port == 587


def test_notification_splits_email_recipients(monkeypatch):


    monkeypatch.setenv("MONITORING_NOTIFY_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("MONITORING_NOTIFY_SMTP_PORT", "2525")
    monkeypatch.setenv("MONITORING_NOTIFY_EMAIL_TO", "a@example.com, b@example.com")
    monkeypatch.setenv("MONITORING_NOTIFY_EMAIL_FROM", "sender@example.com")

    sent: dict[str, object] = {}

    class _FakeSMTP:
        def __init__(self, host, port, timeout=None):
            sent["host"] = host
            sent["port"] = port
            sent["timeout"] = timeout

        def __enter__(self):

            return self

        def __exit__(self, exc_type, exc, tb):

            return False

        def starttls(self):

            sent["starttls"] = True

        def login(self, user, password):

            sent["login"] = (user, password)

        def sendmail(self, from_addr, to_addrs, msg):

            sent["from"] = from_addr
            sent["to"] = list(to_addrs)
            sent["msg"] = msg

    monkeypatch.setattr(notification_service.smtplib, "SMTP", _FakeSMTP)

    svc = NotificationService()
    alert = TopicAlert(
        user_id="u",
        scope_type="user",
        scope_id="u",
        source="chat.input",
        watchlist_id="w",
        rule_category="adult",
        rule_severity="critical",
        pattern="nsfw",
        text_snippet="...nsfw...",
    )
    svc._send_email(alert)

    assert sent["to"] == ["a@example.com", "b@example.com"]
    assert sent["from"] == "sender@example.com"


def test_notification_update_settings_normalizes_relative_file(tmp_path, monkeypatch):


    from tldw_Server_API.app.core.Utils import Utils as utils_module

    svc = NotificationService()
    monkeypatch.setattr(utils_module, "get_project_root", lambda: str(tmp_path))

    relative = "logs/monitoring.jsonl"
    updated = svc.update_settings(file=relative)

    expected = tmp_path / relative
    assert svc.file_path == str(expected)
    assert updated["file"] == str(expected)
    assert expected.parent.exists()


def test_notification_update_settings_file_failure_log_is_sanitized(tmp_path, monkeypatch):
    svc = NotificationService()
    original_file = svc.file_path
    secret_path = tmp_path / "secret-user-dir" / "monitoring.jsonl"
    messages, sink_id = _capture_notification_logs("WARNING")
    original_mkdir = notification_service.Path.mkdir

    def _raise_mkdir(self, parents=False, exist_ok=False):  # noqa: ANN001
        _ = (parents, exist_ok)
        raise OSError(f"cannot create {secret_path}")

    monkeypatch.setattr(notification_service.Path, "mkdir", _raise_mkdir)

    try:
        updated = svc.update_settings(file=str(secret_path))
    finally:
        monkeypatch.setattr(notification_service.Path, "mkdir", original_mkdir)
        notification_service.logger.remove(sink_id)

    assert updated["file"] == original_file
    assert any("Failed to update MONITORING_NOTIFY_FILE" in message for message in messages)
    assert "secret-user-dir" not in "\n".join(messages)
    assert str(secret_path) not in "\n".join(messages)


def test_notify_file_sink_failure_log_is_sanitized(tmp_path, monkeypatch):
    svc = NotificationService()
    svc.enabled = True
    svc.min_severity = "info"
    svc.file_path = str(tmp_path / "secret-notification-sink.jsonl")
    messages, sink_id = _capture_notification_logs("WARNING")

    def _raise_open(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = (args, kwargs)
        raise OSError(f"permission denied for {svc.file_path}")

    monkeypatch.setattr(builtins, "open", _raise_open)

    alert = TopicAlert(
        user_id="u",
        scope_type="user",
        scope_id="u",
        source="chat.input",
        watchlist_id="w",
        rule_category="adult",
        rule_severity="critical",
        pattern="nsfw",
        text_snippet="...nsfw...",
    )

    try:
        result = svc.notify(alert)
    finally:
        notification_service.logger.remove(sink_id)

    assert result == "failed"
    assert any("Notification file sink failed" in message for message in messages)
    assert "secret-notification-sink" not in "\n".join(messages)
    assert svc.file_path not in "\n".join(messages)


def test_notification_send_webhook_invokes_fetch(monkeypatch):


    import tldw_Server_API.app.core.http_client as http_client

    svc = NotificationService()
    svc.webhook_url = "https://example.com/hook"

    calls: dict[str, object] = {}

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):

            return False

    def _fake_create_client(timeout=None):

        calls["client_timeout"] = timeout
        return _FakeClient()

    def _fake_fetch(method, url, client, headers, json, timeout=None):

        calls["method"] = method
        calls["url"] = url
        calls["headers"] = headers
        calls["json"] = json
        calls["timeout"] = timeout

    monkeypatch.setattr(http_client, "create_client", _fake_create_client)
    monkeypatch.setattr(http_client, "fetch", _fake_fetch)

    svc._send_webhook({"event": "test"})

    assert calls["url"] == "https://example.com/hook"
    assert calls["method"] == "POST"


def test_send_webhook_safe_swallows_retry_exhaustion(monkeypatch):
    svc = NotificationService()
    monkeypatch.setattr(
        svc,
        "_send_webhook",
        lambda payload: (_ for _ in ()).throw(_retry_error("webhook boom")),
    )

    svc._send_webhook_safe({"event": "test"})


def test_send_email_safe_swallows_retry_exhaustion(monkeypatch):
    svc = NotificationService()
    alert = TopicAlert(
        user_id="u1",
        scope_type="user",
        scope_id="u1",
        source="chat.input",
        watchlist_id="watch-1",
        rule_category="system",
        rule_severity="critical",
        pattern="cpu high",
        text_snippet="CPU at 95%",
    )
    monkeypatch.setattr(
        svc,
        "_send_email",
        lambda payload: (_ for _ in ()).throw(_retry_error("email boom")),
    )

    svc._send_email_safe(alert)


def test_notify_generic_only_schedules_webhook_path(monkeypatch, tmp_path):
    svc = NotificationService()
    svc.enabled = True
    svc.min_severity = "info"
    svc.file_path = str(tmp_path / "notifications.jsonl")
    svc.webhook_url = "https://example.com/hook"
    svc.email_to = "alerts@example.com"
    svc.smtp_host = "smtp.example.com"
    svc.email_from = "sender@example.com"

    targets: list[object] = []

    class _FakeThread:
        def __init__(self, *, target=None, args=(), daemon=None):  # noqa: ANN001, ANN002
            _ = (args, daemon)
            targets.append(target)

        def start(self) -> None:
            return None

    monkeypatch.setattr(notification_service.threading, "Thread", _FakeThread)

    result = svc.notify_generic({"type": "guardian_alert", "severity": "warning", "user_id": "u1"})

    assert result == "logged"
    assert svc._send_webhook_safe in targets
    assert svc._send_email_safe not in targets


def test_notify_generic_webhook_thread_failure_log_is_sanitized(monkeypatch, tmp_path):
    svc = NotificationService()
    svc.enabled = True
    svc.min_severity = "info"
    svc.file_path = str(tmp_path / "notifications.jsonl")
    svc.webhook_url = "https://example.com/hook"
    secret_thread_detail = "thread-token=/tmp/private-webhook-dispatch"
    messages, sink_id = _capture_notification_logs("DEBUG")

    class _FailingThread:
        def __init__(self, *, target=None, args=(), daemon=None):  # noqa: ANN001, ANN002
            _ = (target, args, daemon)
            raise RuntimeError(secret_thread_detail)

    monkeypatch.setattr(notification_service.threading, "Thread", _FailingThread)

    try:
        result = svc.notify_generic({"type": "guardian_alert", "severity": "warning", "user_id": "u1"})
    finally:
        notification_service.logger.remove(sink_id)

    assert result == "logged"
    assert any("Webhook thread start failed" in message for message in messages)
    assert secret_thread_detail not in "\n".join(messages)
    assert "private-webhook-dispatch" not in "\n".join(messages)


def test_flush_digest_returns_count_without_dispatch(monkeypatch):
    svc = NotificationService()
    svc.enabled = True
    svc.min_severity = "info"
    svc.digest_mode = "hourly"
    svc.notify_or_batch({"type": "guardian_alert", "severity": "info", "user_id": "u1"})
    svc.notify_or_batch({"type": "guardian_alert", "severity": "info", "user_id": "u1"})
    monkeypatch.setattr(
        svc,
        "notify_generic",
        lambda payload: (_ for _ in ()).throw(AssertionError("flush_digest must not dispatch")),
    )

    assert svc.flush_digest("u1") == 2
    assert svc.get_pending_digest_count("u1") == 0
