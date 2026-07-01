import os
import json
import builtins
from pathlib import Path

import pytest
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


def test_notification_send_email_fails_closed_when_starttls_fails(monkeypatch):
    monkeypatch.setenv("MONITORING_NOTIFY_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("MONITORING_NOTIFY_EMAIL_TO", "a@example.com")
    monkeypatch.setenv("MONITORING_NOTIFY_EMAIL_FROM", "sender@example.com")
    monkeypatch.setenv("MONITORING_NOTIFY_SMTP_STARTTLS", "true")

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
            raise notification_service.smtplib.SMTPException("tls unavailable")

        def login(self, user, password):
            sent["login"] = (user, password)

        def sendmail(self, from_addr, to_addrs, msg):
            sent["sendmail"] = True

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

    with pytest.raises(RuntimeError, match="SMTP STARTTLS failed"):
        svc._send_email(alert)

    assert sent["starttls"] is True
    assert "sendmail" not in sent


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


def test_notify_generic_uses_single_delivery_worker(monkeypatch, tmp_path):
    svc = NotificationService()
    svc.enabled = True
    svc.min_severity = "info"
    svc.file_path = str(tmp_path / "notifications.jsonl")
    svc.webhook_url = "https://example.com/hook"

    created_threads: list[object] = []

    class _FakeThread:
        def __init__(self, *, target=None, args=(), daemon=None):  # noqa: ANN001, ANN002
            _ = (args, daemon)
            created_threads.append(target)

        def start(self) -> None:
            return None

    monkeypatch.setattr(notification_service.threading, "Thread", _FakeThread)

    for _ in range(5):
        assert svc.notify_generic(
            {"type": "guardian_alert", "severity": "warning", "user_id": "u1"}
        ) == "logged"

    assert len(created_threads) == 1


def test_digest_buffer_is_capped_per_recipient(monkeypatch):
    monkeypatch.setenv("MONITORING_NOTIFY_MAX_DIGEST_ITEMS_PER_RECIPIENT", "2")
    svc = NotificationService()
    svc.enabled = True
    svc.min_severity = "info"
    svc.digest_mode = "hourly"
    delivered: list[dict[str, object]] = []

    monkeypatch.setattr(
        svc,
        "notify_generic",
        lambda payload: delivered.append(payload) or "logged",
    )

    for idx in range(3):
        assert svc.notify_or_batch(
            {
                "type": "guardian_alert",
                "severity": "info",
                "user_id": "u1",
                "idx": idx,
            }
        ) == "batched"

    assert svc.get_pending_digest_count("u1") == 2
    assert svc.flush_digest("u1") == 2
    assert [item["idx"] for item in delivered[0]["items"]] == [1, 2]


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
    secret_detail = "webhook retry failed at /tmp/private-webhook-retry"  # nosec B105
    messages, sink_id = _capture_notification_logs("INFO")
    monkeypatch.setattr(
        svc,
        "_send_webhook",
        lambda payload: (_ for _ in ()).throw(_retry_error(secret_detail)),
    )

    try:
        svc._send_webhook_safe({"event": "test"})
    finally:
        notification_service.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Webhook notify failed" in joined
    assert secret_detail not in joined
    assert "private-webhook-retry" not in joined


def test_send_webhook_safe_sanitizes_runtime_failure_log(monkeypatch):
    svc = NotificationService()
    secret_detail = "webhook runtime failed at /tmp/private-webhook-runtime"  # nosec B105
    messages, sink_id = _capture_notification_logs("INFO")
    monkeypatch.setattr(
        svc,
        "_send_webhook",
        lambda payload: (_ for _ in ()).throw(RuntimeError(secret_detail)),
    )

    try:
        svc._send_webhook_safe({"event": "test"})
    finally:
        notification_service.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Webhook notify failed" in joined
    assert secret_detail not in joined
    assert "private-webhook-runtime" not in joined


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
    secret_detail = "email retry failed at /tmp/private-email-retry"  # nosec B105
    messages, sink_id = _capture_notification_logs("INFO")
    monkeypatch.setattr(
        svc,
        "_send_email",
        lambda payload: (_ for _ in ()).throw(_retry_error(secret_detail)),
    )

    try:
        svc._send_email_safe(alert)
    finally:
        notification_service.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Email notify failed" in joined
    assert secret_detail not in joined
    assert "private-email-retry" not in joined


def test_send_email_safe_sanitizes_runtime_failure_log(monkeypatch):
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
    secret_detail = "email runtime failed at /tmp/private-email-runtime"  # nosec B105
    messages, sink_id = _capture_notification_logs("INFO")
    monkeypatch.setattr(
        svc,
        "_send_email",
        lambda payload: (_ for _ in ()).throw(RuntimeError(secret_detail)),
    )

    try:
        svc._send_email_safe(alert)
    finally:
        notification_service.logger.remove(sink_id)

    joined = "\n".join(messages)
    assert "Email notify failed" in joined
    assert secret_detail not in joined
    assert "private-email-runtime" not in joined


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
    assert targets == [svc._delivery_worker]
    queued = list(svc._delivery_queue.queue)
    assert len(queued) == 1
    assert queued[0][0] == "webhook"


def test_notify_generic_redacts_sensitive_payload_before_storage_and_webhook(monkeypatch, tmp_path):
    svc = NotificationService()
    svc.enabled = True
    svc.min_severity = "info"
    svc.file_path = str(tmp_path / "notifications.jsonl")
    svc.webhook_url = "https://example.com/hook"

    class _FakeThread:
        def __init__(self, *, target=None, args=(), daemon=None):  # noqa: ANN001, ANN002
            _ = (target, args, daemon)

        def start(self) -> None:
            return None

    monkeypatch.setattr(notification_service.threading, "Thread", _FakeThread)

    payload = {
        "type": "guardian_alert",
        "severity": "warning",
        "api_key": "api-key-secret",
        "nested": {"access-token": "access-secret"},
        "items": [{"refresh_token": "refresh-secret"}],  # nosec B105
        "message": "authorization=Bearer-secret token=query-token",
    }

    assert svc.notify_generic(payload) == "logged"

    written = Path(svc.file_path).read_text()
    assert "api-key-secret" not in written
    assert "access-secret" not in written
    assert "refresh-secret" not in written
    assert "Bearer-secret" not in written
    assert "query-token" not in written
    assert written.count("[REDACTED]") >= 5
    assert "ts" not in payload

    queued = list(svc._delivery_queue.queue)
    assert queued[0][0] == "webhook"
    queued_payload = queued[0][1]
    assert queued_payload["api_key"] == "[REDACTED]"
    assert queued_payload["nested"]["access-token"] == "[REDACTED]"
    assert queued_payload["items"][0]["refresh_token"] == "[REDACTED]"
    assert queued_payload["message"] == "authorization=[REDACTED] token=[REDACTED]"


def test_notify_redacts_topic_alert_metadata_before_storage_and_webhook(monkeypatch, tmp_path):
    svc = NotificationService()
    svc.enabled = True
    svc.min_severity = "info"
    svc.file_path = str(tmp_path / "notifications.jsonl")
    svc.webhook_url = "https://example.com/hook"

    class _FakeThread:
        def __init__(self, *, target=None, args=(), daemon=None):  # noqa: ANN001, ANN002
            _ = (target, args, daemon)

        def start(self) -> None:
            return None

    monkeypatch.setattr(notification_service.threading, "Thread", _FakeThread)

    alert = TopicAlert(
        user_id="u",
        scope_type="user",
        scope_id="u",
        source="chat.input",
        watchlist_id="w",
        rule_category="credentials",
        rule_severity="critical",
        pattern="api_key=pattern-secret",
        text_snippet="token=snippet-secret",
        metadata={"password": "metadata-secret"},  # nosec B105
    )

    assert svc.notify(alert) == "logged"

    written = Path(svc.file_path).read_text()
    assert "pattern-secret" not in written
    assert "snippet-secret" not in written
    assert "metadata-secret" not in written

    queued_payload = list(svc._delivery_queue.queue)[0][1]
    assert queued_payload["pattern"] == "api_key=[REDACTED]"
    assert queued_payload["snippet"] == "token=[REDACTED]"
    assert queued_payload["metadata"]["password"] == "[REDACTED]"


def test_notify_generic_webhook_enqueue_failure_log_is_sanitized(monkeypatch, tmp_path):
    svc = NotificationService()
    svc.enabled = True
    svc.min_severity = "info"
    svc.file_path = str(tmp_path / "notifications.jsonl")
    svc.webhook_url = "https://example.com/hook"
    secret_thread_detail = "thread-token=/tmp/private-webhook-dispatch"  # nosec B105
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
    assert any("Webhook delivery enqueue failed" in message for message in messages)
    assert secret_thread_detail not in "\n".join(messages)
    assert "private-webhook-dispatch" not in "\n".join(messages)


def test_notify_webhook_enqueue_failure_log_is_sanitized(monkeypatch, tmp_path):
    svc = NotificationService()
    svc.enabled = True
    svc.min_severity = "info"
    svc.file_path = str(tmp_path / "notifications.jsonl")
    svc.webhook_url = "https://example.com/hook"
    secret_thread_detail = "thread-token=/tmp/private-topic-webhook-dispatch"  # nosec B105
    messages, sink_id = _capture_notification_logs("DEBUG")

    class _FailingThread:
        def __init__(self, *, target=None, args=(), daemon=None):  # noqa: ANN001, ANN002
            _ = (target, args, daemon)
            raise RuntimeError(secret_thread_detail)

    monkeypatch.setattr(notification_service.threading, "Thread", _FailingThread)

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

    assert result == "logged"
    assert any("Webhook delivery enqueue failed" in message for message in messages)
    joined = "\n".join(messages)
    assert secret_thread_detail not in joined
    assert "private-topic-webhook-dispatch" not in joined


def test_notify_email_enqueue_failure_log_is_sanitized(monkeypatch, tmp_path):
    svc = NotificationService()
    svc.enabled = True
    svc.min_severity = "info"
    svc.file_path = str(tmp_path / "notifications.jsonl")
    svc.smtp_host = "smtp.example.com"
    svc.email_from = "alerts@example.com"
    svc.email_to = "recipient@example.com"
    secret_thread_detail = "thread-token=/tmp/private-topic-email-dispatch"  # nosec B105
    messages, sink_id = _capture_notification_logs("DEBUG")

    class _FailingThread:
        def __init__(self, *, target=None, args=(), daemon=None):  # noqa: ANN001, ANN002
            _ = (target, args, daemon)
            raise RuntimeError(secret_thread_detail)

    monkeypatch.setattr(notification_service.threading, "Thread", _FailingThread)

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

    assert result == "logged"
    assert any("Email delivery enqueue failed" in message for message in messages)
    joined = "\n".join(messages)
    assert secret_thread_detail not in joined
    assert "private-topic-email-dispatch" not in joined


def test_notify_generic_file_sink_failure_log_is_sanitized(monkeypatch, tmp_path):
    svc = NotificationService()
    svc.enabled = True
    svc.min_severity = "info"
    svc.file_path = str(tmp_path / "secret-generic-notification-sink.jsonl")
    messages, sink_id = _capture_notification_logs("WARNING")

    def _raise_open(*args, **kwargs):  # noqa: ANN002, ANN003
        _ = (args, kwargs)
        raise OSError(f"permission denied for {svc.file_path}")

    monkeypatch.setattr(builtins, "open", _raise_open)

    try:
        result = svc.notify_generic({"type": "guardian_alert", "severity": "warning", "user_id": "u1"})
    finally:
        notification_service.logger.remove(sink_id)

    assert result == "failed"
    assert any("Notification file sink failed" in message for message in messages)
    joined = "\n".join(messages)
    assert "secret-generic-notification-sink" not in joined
    assert svc.file_path not in joined


def test_flush_digest_dispatches_compiled_payload(monkeypatch):
    svc = NotificationService()
    svc.enabled = True
    svc.min_severity = "info"
    svc.digest_mode = "hourly"
    delivered: list[dict[str, object]] = []

    def _record_digest(payload: dict[str, object]) -> str:
        delivered.append(payload)
        return "logged"

    monkeypatch.setattr(svc, "notify_generic", _record_digest)

    assert svc.notify_or_batch(
        {"type": "guardian_alert", "severity": "info", "user_id": "u1"}
    ) == "batched"
    assert svc.notify_or_batch(
        {"type": "guardian_alert", "severity": "critical", "user_id": "u1"}
    ) == "batched"
    assert svc.flush_digest("u1") == 2
    assert svc.get_pending_digest_count("u1") == 0

    assert len(delivered) == 1
    digest = delivered[0]
    assert digest["type"] == "monitoring_digest"
    assert digest["recipient"] == "u1"
    assert digest["digest_mode"] == "hourly"
    assert digest["item_count"] == 2
    assert digest["severity"] == "critical"
    assert [item["severity"] for item in digest["items"]] == ["info", "critical"]


def test_flush_digest_uses_rule_severity_for_topic_alert_payloads(monkeypatch):
    svc = NotificationService()
    svc.enabled = True
    svc.min_severity = "info"
    svc.digest_mode = "hourly"
    delivered: list[dict[str, object]] = []

    monkeypatch.setattr(
        svc,
        "notify_generic",
        lambda payload: delivered.append(payload) or "logged",
    )

    assert svc.notify_or_batch(
        {"type": "topic_alert", "rule_severity": "critical", "user_id": "u1"}
    ) == "batched"
    assert svc.notify_or_batch(
        {"type": "guardian_alert", "severity": "warning", "user_id": "u1"}
    ) == "batched"

    assert svc.flush_digest("u1") == 2
    assert delivered[0]["severity"] == "critical"


def test_digest_recipient_keys_preserve_falsy_user_ids(monkeypatch):
    svc = NotificationService()
    svc.enabled = True
    svc.min_severity = "info"
    svc.digest_mode = "hourly"

    monkeypatch.setattr(svc, "notify_generic", lambda payload: "logged")

    assert svc.notify_or_batch(
        {"type": "guardian_alert", "severity": "info", "user_id": 0}
    ) == "batched"
    assert svc.get_pending_digest_count(0) == 1
    assert svc.get_pending_digest_count("0") == 1
    assert svc.get_pending_digest_count("_default") == 0
    assert svc.flush_digest(0) == 1
    assert svc.get_pending_digest_count(0) == 0


def test_flush_digest_requeues_when_delivery_fails(monkeypatch):
    svc = NotificationService()
    svc.enabled = True
    svc.min_severity = "info"
    svc.digest_mode = "daily"
    svc.notify_or_batch({"type": "guardian_alert", "severity": "warning", "user_id": "u1"})
    svc.notify_or_batch({"type": "guardian_alert", "severity": "critical", "user_id": "u1"})
    attempts: list[dict[str, object]] = []

    def _fail_digest(payload: dict[str, object]) -> str:
        attempts.append(payload)
        return "failed"

    monkeypatch.setattr(svc, "notify_generic", _fail_digest)

    assert svc.flush_digest("u1") == 0
    assert svc.get_pending_digest_count("u1") == 2
    assert len(attempts) == 1


def test_flush_digest_requeues_when_delivery_raises(monkeypatch):
    svc = NotificationService()
    svc.enabled = True
    svc.min_severity = "info"
    svc.digest_mode = "hourly"
    svc.notify_or_batch({"type": "guardian_alert", "severity": "info", "user_id": "u1"})

    def _raise_digest(payload: dict[str, object]) -> str:
        raise RuntimeError("digest sink unavailable")

    monkeypatch.setattr(svc, "notify_generic", _raise_digest)

    assert svc.flush_digest("u1") == 0
    assert svc.get_pending_digest_count("u1") == 1


def test_flush_digest_requeues_when_delivery_raises_unexpected_exception(monkeypatch):
    class UnexpectedDigestError(Exception):
        pass

    svc = NotificationService()
    svc.enabled = True
    svc.min_severity = "info"
    svc.digest_mode = "hourly"
    svc.notify_or_batch({"type": "guardian_alert", "severity": "warning", "user_id": "u1"})

    def _raise_digest(payload: dict[str, object]) -> str:
        raise UnexpectedDigestError("digest sink unavailable")

    monkeypatch.setattr(svc, "notify_generic", _raise_digest)

    assert svc.flush_digest("u1") == 0
    assert svc.get_pending_digest_count("u1") == 1


def test_flush_digest_does_not_requeue_threshold_skipped_delivery(monkeypatch):
    svc = NotificationService()
    svc.enabled = True
    svc.min_severity = "info"
    svc.digest_mode = "hourly"
    svc.notify_or_batch({"type": "guardian_alert", "severity": "warning", "user_id": "u1"})

    svc.min_severity = "critical"
    monkeypatch.setattr(svc, "notify_generic", lambda payload: "skipped")

    assert svc.flush_digest("u1") == 1
    assert svc.get_pending_digest_count("u1") == 0


def test_flush_digest_without_recipient_dispatches_one_digest_per_recipient(monkeypatch):
    svc = NotificationService()
    svc.enabled = True
    svc.min_severity = "info"
    svc.digest_mode = "hourly"
    delivered: list[dict[str, object]] = []
    monkeypatch.setattr(
        svc,
        "notify_generic",
        lambda payload: delivered.append(payload) or "logged",
    )

    svc.notify_or_batch({"type": "guardian_alert", "severity": "info", "user_id": "u1"})
    svc.notify_or_batch({"type": "guardian_alert", "severity": "warning", "user_id": "u2"})

    assert svc.flush_digest() == 2
    assert svc.get_pending_digest_count() == 0
    assert {payload["recipient"] for payload in delivered} == {"u1", "u2"}
    assert all(payload["item_count"] == 1 for payload in delivered)
