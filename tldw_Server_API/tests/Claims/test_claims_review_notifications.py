import json
import os
import tempfile
import time
from contextlib import contextmanager

from tldw_Server_API.app.core.Claims_Extraction import claims_alert_delivery, claims_notifications, claims_service
from tldw_Server_API.app.core.Claims_Extraction.claims_notifications import dispatch_claim_review_notifications
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase


class _FakeEmailService:
    def __init__(self) -> None:
        self.sent = []

    async def send_email(self, *, to_email: str, subject: str, html_body: str, text_body: str):
        self.sent.append(
            {
                "to_email": to_email,
                "subject": subject,
                "html_body": html_body,
                "text_body": text_body,
            }
        )
        return True


def test_build_review_email_bodies_escapes_html() -> None:
    html_body, text_body = claims_notifications._build_review_email_bodies(
        [
            {
                "kind": "review_update",
                "created_at": "2026-06-23T00:00:00Z",
                "payload": {
                    "new_status": "approved",
                    "claim_text": "<script>alert(1)</script>",
                },
            }
        ]
    )

    assert "<script>" not in html_body
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_body
    assert "<script>alert(1)</script>" in text_body


def _seed_review_notification_db() -> MediaDatabase:

    tmpdir = tempfile.mkdtemp(prefix="claims_review_notify_")
    db_path = os.path.join(tmpdir, "media.db")
    db = MediaDatabase(db_path=db_path, client_id="1")
    db.initialize_db()
    media_id, _, _ = db.add_media_with_keywords(
        title="Doc",
        media_type="text",
        content="A. B.",
        keywords=None,
    )
    db.upsert_claims(
        [
            {
                "media_id": media_id,
                "chunk_index": 0,
                "span_start": None,
                "span_end": None,
                "claim_text": "A.",
                "confidence": 0.9,
                "extractor": "heuristic",
                "extractor_version": "v1",
                "chunk_hash": "abc",
            }
        ]
    )
    row = db.execute_query("SELECT id, uuid FROM Claims WHERE media_id = ?", (media_id,)).fetchone()
    claim_id = int(row["id"]) if isinstance(row, dict) else int(row[0])
    claim_uuid = row["uuid"] if isinstance(row, dict) else row[1]
    return db, claim_id, claim_uuid


def test_claims_review_notifications_deliver_email(monkeypatch):

    from tldw_Server_API.app.core.AuthNZ import email_service as email_module

    fake_service = _FakeEmailService()
    monkeypatch.setattr(email_module, "get_email_service", lambda: fake_service)

    db, claim_id, claim_uuid = _seed_review_notification_db()
    try:
        db.upsert_claims_monitoring_settings(
            user_id="1",
            threshold_ratio=0.2,
            baseline_ratio=0.1,
            slack_webhook_url=None,
            webhook_url=None,
            email_recipients=json.dumps(["review@example.com"]),
            enabled=True,
        )
        notification = db.insert_claim_notification(
            user_id="1",
            kind="review_update",
            target_user_id="1",
            target_review_group=None,
            resource_type="claim",
            resource_id=str(claim_id),
            payload_json=json.dumps(
                {
                    "claim_id": claim_id,
                    "claim_uuid": claim_uuid,
                    "claim_text": "A.",
                    "old_status": "pending",
                    "new_status": "approved",
                }
            ),
        )
        notif_id = int(notification.get("id"))
        dispatch_claim_review_notifications(
            db_path=str(db.db_path_str),
            owner_user_id="1",
            notification_ids=[notif_id],
        )
        delivered = None
        for _ in range(20):
            row = db.get_claim_notification(notif_id)
            delivered = row.get("delivered_at")
            if delivered:
                break
            time.sleep(0.05)
        assert delivered is not None
        assert fake_service.sent
    finally:
        db.close_connection()


def test_deliver_claim_review_notifications_now_returns_skipped_when_disabled(monkeypatch, tmp_path):
    class _FakeDb:
        def get_claims_monitoring_settings(self, user_id):
            assert user_id == "1"
            return {"enabled": False}

        def close_connection(self) -> None:
            pass

    @contextmanager
    def _fake_managed_media_database(*_args, **_kwargs):
        yield _FakeDb()

    monkeypatch.setattr(claims_notifications, "managed_media_database", _fake_managed_media_database)
    db_path = str(tmp_path / "claims-review.db")

    result = claims_notifications.deliver_claim_review_notifications_now(
        db_path=db_path,
        owner_user_id="1",
        notification_ids=[7],
    )

    assert result == {"outcome": "skipped", "reason": "settings_disabled", "notification_ids": [7]}


def test_deliver_claim_review_notifications_now_returns_failed_when_db_unavailable(monkeypatch, tmp_path):
    @contextmanager
    def _fake_managed_media_database(*_args, **_kwargs):
        yield None

    monkeypatch.setattr(claims_notifications, "managed_media_database", _fake_managed_media_database)

    result = claims_notifications.deliver_claim_review_notifications_now(
        db_path=str(tmp_path / "claims-review.db"),
        owner_user_id="1",
        notification_ids=[7],
    )

    assert result == {"outcome": "failed", "reason": "database_initialization_failed", "notification_ids": [7]}


def test_deliver_claim_review_notifications_now_returns_success_contract(monkeypatch, tmp_path):
    class _FakeDb:
        def __init__(self) -> None:
            self.marked_ids: list[int] = []

        def get_claims_monitoring_settings(self, user_id):
            assert user_id == "1"
            return {
                "enabled": True,
                "slack_webhook_url": None,
                "webhook_url": None,
                "email_recipients": json.dumps(["review@example.com"]),
            }

        def get_claim_notifications_by_ids(self, notification_ids):
            assert notification_ids == [2, 7]
            return [
                {
                    "id": 7,
                    "user_id": "1",
                    "kind": "review_update",
                    "payload_json": json.dumps({"claim_text": "A.", "new_status": "approved"}),
                    "created_at": "2026-03-16T00:00:00Z",
                    "delivered_at": None,
                },
                {
                    "id": 2,
                    "user_id": "1",
                    "kind": "review_assignment",
                    "payload_json": json.dumps({"claim_text": "B.", "new_status": "pending"}),
                    "created_at": "2026-03-17T00:00:00Z",
                    "delivered_at": None,
                },
            ]

        def mark_claim_notifications_delivered(self, notification_ids):
            self.marked_ids.extend(notification_ids)
            return len(notification_ids)

    fake_db = _FakeDb()
    managed_calls: list[dict[str, object]] = []
    email_calls: list[dict[str, object]] = []

    @contextmanager
    def _fake_managed_media_database(client_id, *, initialize=True, **kwargs):
        managed_calls.append(
            {
                "client_id": client_id,
                "initialize": initialize,
                "kwargs": kwargs,
            }
        )
        yield fake_db

    def _fake_deliver_review_email_sync(**kwargs):
        email_calls.append(kwargs)
        return True

    monkeypatch.setattr(claims_notifications, "managed_media_database", _fake_managed_media_database)
    monkeypatch.setattr(claims_notifications, "_deliver_review_email_sync", _fake_deliver_review_email_sync)
    db_path = str(tmp_path / "claims-review.db")

    result = claims_notifications.deliver_claim_review_notifications_now(
        db_path=db_path,
        owner_user_id="1",
        notification_ids=[7, 2, 7, 0, -5],
    )

    assert result == {"outcome": "ok", "notification_ids": [2, 7], "delivered": 2}
    assert fake_db.marked_ids == [2, 7]
    assert email_calls and email_calls[0]["recipients"] == ["review@example.com"]
    assert managed_calls == [
        {
            "client_id": claims_notifications.settings.get("SERVER_CLIENT_ID", "SERVER_API_V1"),
            "initialize": False,
            "kwargs": {
                "db_path": db_path,
                "suppress_init_exceptions": claims_notifications._CLAIMS_NOTIFICATION_NONCRITICAL_EXCEPTIONS,
                "suppress_close_exceptions": claims_notifications._CLAIMS_NOTIFICATION_NONCRITICAL_EXCEPTIONS,
            },
        }
    ]


def test_deliver_claim_review_notifications_now_requires_all_configured_channels(
    monkeypatch,
    tmp_path,
) -> None:
    class _FakeDb:
        def __init__(self) -> None:
            self.marked_ids: list[int] = []

        def get_claims_monitoring_settings(self, user_id):
            assert user_id == "1"
            return {
                "enabled": True,
                "slack_webhook_url": "https://example.test/slack",
                "webhook_url": None,
                "email_recipients": json.dumps(["review@example.com"]),
            }

        def get_claim_notifications_by_ids(self, notification_ids):
            assert notification_ids == [7]
            return [
                {
                    "id": 7,
                    "user_id": "1",
                    "kind": "review_update",
                    "payload_json": json.dumps({"claim_text": "A.", "new_status": "approved"}),
                    "created_at": "2026-03-16T00:00:00Z",
                    "delivered_at": None,
                }
            ]

        def mark_claim_notifications_delivered(self, notification_ids):
            self.marked_ids.extend(notification_ids)
            return len(notification_ids)

    fake_db = _FakeDb()

    @contextmanager
    def _fake_managed_media_database(*_args, **_kwargs):
        yield fake_db

    monkeypatch.setattr(claims_notifications, "managed_media_database", _fake_managed_media_database)
    monkeypatch.setattr(claims_notifications, "_deliver_review_webhook", lambda **_kwargs: False)
    monkeypatch.setattr(claims_notifications, "_deliver_review_email_sync", lambda **_kwargs: True)

    result = claims_notifications.deliver_claim_review_notifications_now(
        db_path=str(tmp_path / "claims-review.db"),
        owner_user_id="1",
        notification_ids=[7],
    )

    assert result == {"outcome": "failed", "reason": "delivery_failed", "notification_ids": [7]}
    assert fake_db.marked_ids == []


def test_deliver_claim_review_notifications_now_marks_only_pending_notifications(monkeypatch, tmp_path):
    class _FakeDb:
        def __init__(self) -> None:
            self.marked_ids: list[int] = []

        def get_claims_monitoring_settings(self, user_id):
            assert user_id == "1"
            return {
                "enabled": True,
                "slack_webhook_url": None,
                "webhook_url": None,
                "email_recipients": "review@example.com",
            }

        def get_claim_notifications_by_ids(self, notification_ids):
            assert notification_ids == [2, 7]
            return [
                {
                    "id": 2,
                    "user_id": "1",
                    "kind": "review_update",
                    "payload_json": json.dumps({"claim_text": "Already delivered", "new_status": "approved"}),
                    "created_at": "2026-03-16T00:00:00Z",
                    "delivered_at": "2026-03-16T00:01:00Z",
                },
                {
                    "id": 7,
                    "user_id": "1",
                    "kind": "review_update",
                    "payload_json": json.dumps({"claim_text": "Needs delivery", "new_status": "approved"}),
                    "created_at": "2026-03-17T00:00:00Z",
                    "delivered_at": None,
                },
            ]

        def mark_claim_notifications_delivered(self, notification_ids):
            self.marked_ids.extend(notification_ids)
            return len(notification_ids)

    fake_db = _FakeDb()
    email_calls: list[dict[str, object]] = []

    @contextmanager
    def _fake_managed_media_database(*_args, **_kwargs):
        yield fake_db

    def _fake_deliver_review_email_sync(**kwargs):
        email_calls.append(kwargs)
        return True

    monkeypatch.setattr(claims_notifications, "managed_media_database", _fake_managed_media_database)
    monkeypatch.setattr(claims_notifications, "_deliver_review_email_sync", _fake_deliver_review_email_sync)

    result = claims_notifications.deliver_claim_review_notifications_now(
        db_path=str(tmp_path / "claims-review.db"),
        owner_user_id="1",
        notification_ids=[7, 2],
    )

    assert result == {"outcome": "ok", "notification_ids": [2, 7], "delivered": 1}
    assert fake_db.marked_ids == [7]
    assert email_calls and email_calls[0]["subject"] == "Claims review notifications (1)"
    assert "Needs delivery" in email_calls[0]["text_body"]
    assert "Already delivered" not in email_calls[0]["text_body"]


def test_deliver_claim_review_notifications_now_skips_already_delivered(monkeypatch, tmp_path):
    class _FakeDb:
        def get_claims_monitoring_settings(self, user_id):
            assert user_id == "1"
            return {
                "enabled": True,
                "slack_webhook_url": None,
                "webhook_url": None,
                "email_recipients": "review@example.com",
            }

        def get_claim_notifications_by_ids(self, notification_ids):
            assert notification_ids == [7]
            return [
                {
                    "id": 7,
                    "user_id": "1",
                    "kind": "review_update",
                    "payload_json": json.dumps({"claim_text": "A.", "new_status": "approved"}),
                    "created_at": "2026-03-16T00:00:00Z",
                    "delivered_at": "2026-03-16T00:01:00Z",
                }
            ]

        def mark_claim_notifications_delivered(self, notification_ids):
            raise AssertionError(f"Already delivered notifications should not be marked again: {notification_ids}")

    @contextmanager
    def _fake_managed_media_database(*_args, **_kwargs):
        yield _FakeDb()

    monkeypatch.setattr(claims_notifications, "managed_media_database", _fake_managed_media_database)
    monkeypatch.setattr(
        claims_notifications,
        "_deliver_review_email_sync",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(f"Email should not be delivered: {kwargs}")),
    )
    db_path = str(tmp_path / "claims-review.db")

    result = claims_notifications.deliver_claim_review_notifications_now(
        db_path=db_path,
        owner_user_id="1",
        notification_ids=[7],
    )

    assert result == {"outcome": "skipped", "reason": "already_delivered", "notification_ids": [7]}


def test_deliver_claim_review_notifications_now_filters_mixed_owner_rows(monkeypatch, tmp_path):
    class _FakeDb:
        def __init__(self) -> None:
            self.marked_ids: list[int] = []

        def get_claims_monitoring_settings(self, user_id):
            assert user_id == "1"
            return {
                "enabled": True,
                "slack_webhook_url": None,
                "webhook_url": None,
                "email_recipients": "review@example.com",
            }

        def get_claim_notifications_by_ids(self, notification_ids):
            assert notification_ids == [2, 7]
            return [
                {
                    "id": 2,
                    "user_id": "2",
                    "kind": "review_update",
                    "payload_json": json.dumps({"claim_text": "Wrong owner", "new_status": "approved"}),
                    "created_at": "2026-03-16T00:00:00Z",
                    "delivered_at": None,
                },
                {
                    "id": 7,
                    "user_id": "1",
                    "kind": "review_update",
                    "payload_json": json.dumps({"claim_text": "Right owner", "new_status": "approved"}),
                    "created_at": "2026-03-17T00:00:00Z",
                    "delivered_at": None,
                },
            ]

        def mark_claim_notifications_delivered(self, notification_ids):
            self.marked_ids.extend(notification_ids)
            return len(notification_ids)

    fake_db = _FakeDb()
    email_calls: list[dict[str, object]] = []

    @contextmanager
    def _fake_managed_media_database(*_args, **_kwargs):
        yield fake_db

    def _fake_deliver_review_email_sync(**kwargs):
        email_calls.append(kwargs)
        return True

    monkeypatch.setattr(claims_notifications, "managed_media_database", _fake_managed_media_database)
    monkeypatch.setattr(claims_notifications, "_deliver_review_email_sync", _fake_deliver_review_email_sync)

    result = claims_notifications.deliver_claim_review_notifications_now(
        db_path=str(tmp_path / "claims-review.db"),
        owner_user_id="1",
        notification_ids=[7, 2],
    )

    assert result == {"outcome": "ok", "notification_ids": [2, 7], "delivered": 1}
    assert fake_db.marked_ids == [7]
    assert email_calls and "Right owner" in email_calls[0]["text_body"]
    assert "Wrong owner" not in email_calls[0]["text_body"]


def test_deliver_claim_review_notifications_now_skips_mismatched_owner_rows(monkeypatch, tmp_path):
    class _FakeDb:
        def get_claims_monitoring_settings(self, user_id):
            assert user_id == "1"
            return {
                "enabled": True,
                "slack_webhook_url": None,
                "webhook_url": None,
                "email_recipients": "review@example.com",
            }

        def get_claim_notifications_by_ids(self, notification_ids):
            assert notification_ids == [7]
            return [
                {
                    "id": 7,
                    "user_id": "2",
                    "kind": "review_update",
                    "payload_json": json.dumps({"claim_text": "Wrong owner", "new_status": "approved"}),
                    "created_at": "2026-03-16T00:00:00Z",
                    "delivered_at": None,
                }
            ]

        def mark_claim_notifications_delivered(self, notification_ids):
            raise AssertionError(f"Mismatched rows should not be marked: {notification_ids}")

    @contextmanager
    def _fake_managed_media_database(*_args, **_kwargs):
        yield _FakeDb()

    monkeypatch.setattr(claims_notifications, "managed_media_database", _fake_managed_media_database)
    monkeypatch.setattr(
        claims_notifications,
        "_deliver_review_email_sync",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError(f"Email should not be delivered: {kwargs}")),
    )

    result = claims_notifications.deliver_claim_review_notifications_now(
        db_path=str(tmp_path / "claims-review.db"),
        owner_user_id="1",
        notification_ids=[7],
    )

    assert result == {"outcome": "skipped", "reason": "notifications_owner_mismatch", "notification_ids": [7]}


def test_deliver_claim_review_notifications_now_ignores_invalid_notification_ids(monkeypatch, tmp_path):
    managed_calls: list[dict[str, object]] = []

    @contextmanager
    def _fake_managed_media_database(*_args, **_kwargs):
        managed_calls.append({"called": True})

        class _FakeDb:
            def get_claims_monitoring_settings(self, user_id):
                assert user_id == "1"
                return {}

        yield _FakeDb()

    monkeypatch.setattr(claims_notifications, "managed_media_database", _fake_managed_media_database)

    result = claims_notifications.deliver_claim_review_notifications_now(
        db_path=str(tmp_path / "claims-review.db"),
        owner_user_id="1",
        notification_ids=[True, False, "invalid", float("inf"), -2, 0, None],
    )

    assert result == {"outcome": "skipped", "reason": "no_notification_ids", "notification_ids": []}
    assert managed_calls == []

    result = claims_notifications.deliver_claim_review_notifications_now(
        db_path=str(tmp_path / "claims-review.db"),
        owner_user_id="1",
        notification_ids=[True, "5", 5, 0, -1, "bad"],
    )

    assert result == {"outcome": "skipped", "reason": "no_channels", "notification_ids": [5]}
    assert managed_calls == [{"called": True}]


def test_dispatch_claim_review_notifications_uses_managed_media_database(monkeypatch, tmp_path):
    class _FakeDb:
        def __init__(self) -> None:
            self.closed = False
            self.marked_ids: list[int] = []

        def get_claims_monitoring_settings(self, user_id):
            assert user_id == "1"
            return {
                "enabled": True,
                "slack_webhook_url": None,
                "webhook_url": None,
                "email_recipients": json.dumps(["review@example.com"]),
            }

        def get_claim_notifications_by_ids(self, notification_ids):
            assert notification_ids == [7]
            return [
                {
                    "id": 7,
                    "user_id": "1",
                    "kind": "review_update",
                    "payload_json": json.dumps({"claim_text": "A.", "new_status": "approved"}),
                    "created_at": "2026-03-16T00:00:00Z",
                }
            ]

        def mark_claim_notifications_delivered(self, notification_ids):
            self.marked_ids.extend(notification_ids)
            return len(notification_ids)

        def close_connection(self) -> None:
            self.closed = True

    class _ImmediateThread:
        def __init__(self, *, target, daemon):
            self._target = target
            self.daemon = daemon

        def start(self) -> None:
            self._target()

    fake_db = _FakeDb()
    managed_calls: list[dict[str, object]] = []

    @contextmanager
    def _fake_managed_media_database(client_id, *, initialize=True, **kwargs):
        managed_calls.append(
            {
                "client_id": client_id,
                "initialize": initialize,
                "kwargs": kwargs,
            }
        )
        try:
            yield fake_db
        finally:
            fake_db.close_connection()

    monkeypatch.setattr(claims_notifications, "managed_media_database", _fake_managed_media_database, raising=False)
    monkeypatch.setattr(
        claims_notifications,
        "create_media_database",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("legacy raw factory should not be used")),
        raising=False,
    )
    monkeypatch.setattr(claims_notifications.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(claims_notifications, "_deliver_review_email_sync", lambda **kwargs: True, raising=False)
    db_path = str(tmp_path / "claims-review.db")

    dispatch_claim_review_notifications(
        db_path=db_path,
        owner_user_id="1",
        notification_ids=[7],
    )

    assert fake_db.closed is True
    assert fake_db.marked_ids == [7]
    assert managed_calls == [
        {
            "client_id": claims_notifications.settings.get("SERVER_CLIENT_ID", "SERVER_API_V1"),
            "initialize": True,
            "kwargs": {
                "db_path": db_path,
                "suppress_init_exceptions": claims_notifications._CLAIMS_NOTIFICATION_NONCRITICAL_EXCEPTIONS,
                "suppress_close_exceptions": claims_notifications._CLAIMS_NOTIFICATION_NONCRITICAL_EXCEPTIONS,
            },
        }
    ]


def test_submit_claims_notification_delivery_drops_when_queue_is_full(monkeypatch):
    class _FullSemaphore:
        def acquire(self, *, blocking: bool = True) -> bool:
            assert blocking is False
            return False

        def release(self) -> None:
            raise AssertionError("Saturated notification slots should not be released.")

    started_threads: list[object] = []

    class _Thread:
        def __init__(self, **kwargs) -> None:
            started_threads.append(kwargs)

        def start(self) -> None:
            raise AssertionError("Saturated notification dispatch should not start a thread.")

    monkeypatch.setattr(claims_notifications, "_notification_slots", _FullSemaphore())
    monkeypatch.setattr(claims_notifications.threading, "Thread", _Thread)

    accepted = claims_notifications.submit_claims_notification_delivery(lambda: None)

    assert accepted is False
    assert started_threads == []


def test_dispatch_claims_alert_notifications_uses_bounded_submission(monkeypatch, tmp_path):
    submissions: list[dict[str, object]] = []

    def _submit(fn, *args, **kwargs):
        submissions.append({"fn": fn, "args": args, "kwargs": kwargs})
        return True

    monkeypatch.setattr(claims_service, "submit_claims_notification_delivery", _submit)
    db_path = str(tmp_path / "claims-alert.db")

    claims_service._dispatch_claims_alert_notifications(
        config_row={
            "id": 9,
            "channels_json": json.dumps({"slack": True, "webhook": True}),
            "slack_webhook_url": "https://example.test/slack",
            "webhook_url": "https://example.test/webhook",
        },
        payload={
            "window_ratio": 0.5,
            "threshold": 0.2,
            "baseline_ratio": 0.1,
        },
        db_path=db_path,
        user_id="1",
    )

    assert len(submissions) == 2
    assert submissions[0]["fn"] is claims_alert_delivery.deliver_claims_alert_webhook
    assert submissions[0]["kwargs"]["channel"] == "slack"
    assert submissions[1]["fn"] is claims_alert_delivery.deliver_claims_alert_webhook
    assert submissions[1]["kwargs"]["channel"] == "webhook"
