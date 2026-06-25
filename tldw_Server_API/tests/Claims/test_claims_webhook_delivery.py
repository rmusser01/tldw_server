import json
import types
from contextlib import contextmanager

from tldw_Server_API.app.core.Claims_Extraction import claims_alert_delivery
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase


def test_claims_webhook_delivery_retries_and_records(monkeypatch, tmp_path):

    db_path = tmp_path / "media.db"
    db = MediaDatabase(db_path=str(db_path), client_id="test")
    db.initialize_db()
    db.close_connection()

    class DummyClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):

            return False

    class DummyResponse:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    responses = [500, 200]
    delivery_calls = []

    def fake_fetch(*_args, **_kwargs):

        status = responses.pop(0)
        return DummyResponse(status)

    def fake_record_delivery(**kwargs):

        delivery_calls.append(kwargs)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.http_client.create_client",
        lambda **_kwargs: DummyClient(),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.http_client.fetch",
        fake_fetch,
    )
    monkeypatch.setattr(claims_alert_delivery, "record_claims_webhook_delivery", fake_record_delivery)
    monkeypatch.setattr(claims_alert_delivery.random, "uniform", lambda *_args, **_kwargs: 1.0)
    proxy_time = types.SimpleNamespace(
        time=claims_alert_delivery.time.time,
        sleep=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(claims_alert_delivery, "time", proxy_time)

    delivered = claims_alert_delivery.deliver_claims_alert_webhook(
        url="https://example.com/webhook",
        payload={"ok": True},
        channel="webhook",
        db_path=str(db_path),
        user_id="1",
        alert_id=42,
        event_id=7,
    )

    assert delivered is True
    assert [call["status"] for call in delivery_calls] == ["failure", "success"]

    db = MediaDatabase(db_path=str(db_path), client_id="test")
    rows = db.execute_query("SELECT payload_json FROM claims_monitoring_events ORDER BY id ASC").fetchall()
    db.close_connection()
    assert len(rows) == 2
    payloads = [json.loads(row["payload_json"]) for row in rows]
    assert payloads[0]["status"] == "failure"
    assert payloads[0]["attempt"] == 1
    assert payloads[0]["event_id"] == 7
    assert payloads[1]["status"] == "success"
    assert payloads[1]["attempt"] == 2
    assert payloads[1]["event_id"] == 7


def test_claims_webhook_backoff_schedule(monkeypatch, tmp_path):
    db_path = tmp_path / "media.db"
    db = MediaDatabase(db_path=str(db_path), client_id="test")
    db.initialize_db()
    db.close_connection()

    class DummyClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):

            return False

    class DummyResponse:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

    def fake_fetch(*_args, **_kwargs):

        return DummyResponse(500)

    sleeps = []

    monkeypatch.setattr(
        "tldw_Server_API.app.core.http_client.create_client",
        lambda **_kwargs: DummyClient(),
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.http_client.fetch",
        fake_fetch,
    )
    monkeypatch.setattr(claims_alert_delivery, "record_claims_webhook_delivery", lambda **_kwargs: None)
    monkeypatch.setattr(claims_alert_delivery.random, "uniform", lambda *_args, **_kwargs: 1.0)
    proxy_time = types.SimpleNamespace(
        time=claims_alert_delivery.time.time,
        sleep=lambda delay: sleeps.append(delay),
    )
    monkeypatch.setattr(claims_alert_delivery, "time", proxy_time)

    delivered = claims_alert_delivery.deliver_claims_alert_webhook(
        url="https://example.com/webhook",
        payload={"ok": False},
        channel="webhook",
        db_path=str(db_path),
        user_id="1",
        alert_id=99,
    )

    assert delivered is False
    assert sleeps == [5, 15, 45, 120]

    db = MediaDatabase(db_path=str(db_path), client_id="test")
    row = db.execute_query("SELECT COUNT(*) AS total FROM claims_monitoring_events").fetchone()
    db.close_connection()
    total = int(row["total"]) if isinstance(row, dict) else int(row[0])
    assert total == 5


def test_record_webhook_event_uses_managed_media_database(monkeypatch):
    class _FakeDb:
        def __init__(self) -> None:
            self.insert_calls: list[dict[str, object]] = []

        def insert_claims_monitoring_event(self, **kwargs) -> None:
            self.insert_calls.append(kwargs)

        def close_connection(self) -> None:
            pass

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
        yield fake_db

    monkeypatch.setattr(claims_alert_delivery, "managed_media_database", _fake_managed_media_database, raising=False)

    claims_alert_delivery._record_webhook_event(
        db_path="/tmp/claims-webhook.db",
        user_id="1",
        channel="webhook",
        status="failure",
        attempt=2,
        reason="timeout",
        status_code=504,
        alert_id=42,
        event_id=7,
    )

    assert len(fake_db.insert_calls) == 1
    assert fake_db.insert_calls[0]["user_id"] == "1"
    assert fake_db.insert_calls[0]["event_type"] == "webhook_delivery"
    assert json.loads(fake_db.insert_calls[0]["payload_json"])["event_id"] == 7
    assert managed_calls == [
        {
            "client_id": claims_alert_delivery.settings.get("SERVER_CLIENT_ID", "SERVER_API_V1"),
            "initialize": True,
            "kwargs": {
                "db_path": "/tmp/claims-webhook.db",
                "suppress_init_exceptions": claims_alert_delivery._CLAIMS_ALERT_DELIVERY_NONCRITICAL_EXCEPTIONS,
                "suppress_close_exceptions": claims_alert_delivery._CLAIMS_ALERT_DELIVERY_NONCRITICAL_EXCEPTIONS,
            },
        }
    ]
