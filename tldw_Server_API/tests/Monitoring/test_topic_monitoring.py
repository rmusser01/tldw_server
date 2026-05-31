import os
import json
import asyncio
from pathlib import Path
import pytest
from loguru import logger

from tldw_Server_API.app.core.Monitoring.topic_monitoring_service import (
    TopicMonitoringService,
    get_topic_monitoring_service,
    _reset_topic_monitoring_service,
)
from tldw_Server_API.app.api.v1.schemas.monitoring_schemas import Watchlist, WatchlistRule
from tldw_Server_API.app.core.DB_Management.TopicMonitoring_DB import (
    TopicMonitoringDB,
    TopicAlert,
    WatchlistRecord,
    WatchlistRuleRecord,
)


pytestmark = pytest.mark.unit


def test_topic_monitoring_alert_creation(tmp_path, monkeypatch):


    # Point alerts DB to a temp file
    db_file = tmp_path / "alerts.db"
    monkeypatch.setenv("MONITORING_ALERTS_DB", str(db_file))
    monkeypatch.setenv("MONITORING_ENABLED", "true")
    # Use an in-memory watchlists file to avoid writing to repo
    wl_file = tmp_path / "watchlists.json"
    wl_file.write_text(json.dumps({"watchlists": []}), encoding="utf-8")
    monkeypatch.setenv("MONITORING_WATCHLISTS_FILE", str(wl_file))

    # Ensure the singleton picks up the temp paths for this test run
    _reset_topic_monitoring_service()
    svc = get_topic_monitoring_service()
    svc.reload()

    # Create watchlist for user 'u1' with a literal pattern
    wl = Watchlist(
        name="Test WL",
        description="Detect 'badword'",
        enabled=True,
        scope_type="user",
        scope_id="u1",
        rules=[WatchlistRule(pattern="badword", category="custom", severity="warning")],
    )
    wl = svc.upsert_watchlist(wl)

    # Evaluate input text and generate alert
    count = svc.evaluate_and_alert(user_id="u1", text="This has a badword here.", source="chat.input")
    assert count >= 1

    # Check the alert persisted
    db = TopicMonitoringDB(db_path=str(db_file))
    items = db.list_alerts(user_id="u1")
    assert len(items) >= 1
    assert any("badword" in (it.get("pattern") or "") for it in items)


def test_topic_monitoring_regex_pattern_with_flags(tmp_path, monkeypatch):


    db_file = tmp_path / "alerts.db"
    monkeypatch.setenv("MONITORING_ALERTS_DB", str(db_file))
    monkeypatch.setenv("MONITORING_ENABLED", "true")
    wl_file = tmp_path / "watchlists.json"
    wl_file.write_text(json.dumps({"watchlists": []}), encoding="utf-8")
    monkeypatch.setenv("MONITORING_WATCHLISTS_FILE", str(wl_file))

    _reset_topic_monitoring_service()
    svc = get_topic_monitoring_service()
    svc.reload()

    wl = Watchlist(
        name="Regex WL",
        description="Detect regex with flags",
        enabled=True,
        scope_type="user",
        scope_id="u1",
        rules=[WatchlistRule(pattern="/badword/i", category="custom", severity="warning")],
    )
    svc.upsert_watchlist(wl)

    count = svc.evaluate_and_alert(user_id="u1", text="BADWORD here", source="chat.input")
    assert count >= 1

    db = TopicMonitoringDB(db_path=str(db_file))
    items = db.list_alerts(user_id="u1")
    assert any((it.get("pattern") or "") == "badword" for it in items)


def test_topic_monitoring_skips_empty_pattern(tmp_path, monkeypatch):


    db_file = tmp_path / "alerts.db"
    monkeypatch.setenv("MONITORING_ALERTS_DB", str(db_file))
    monkeypatch.setenv("MONITORING_ENABLED", "true")
    wl_file = tmp_path / "watchlists.json"
    wl_file.write_text(json.dumps({"watchlists": []}), encoding="utf-8")
    monkeypatch.setenv("MONITORING_WATCHLISTS_FILE", str(wl_file))

    _reset_topic_monitoring_service()
    svc = get_topic_monitoring_service()
    svc.reload()

    wl = Watchlist(
        name="Empty Pattern WL",
        description="Should be ignored",
        enabled=True,
        scope_type="user",
        scope_id="u1",
        rules=[WatchlistRule(pattern="", category="custom", severity="warning")],
    )
    svc.upsert_watchlist(wl)

    count = svc.evaluate_and_alert(user_id="u1", text="anything", source="chat.input")
    assert count == 0

    db = TopicMonitoringDB(db_path=str(db_file))
    items = db.list_alerts(user_id="u1")
    assert items == []


def test_topic_monitoring_streaming_dedupe(tmp_path, monkeypatch):
    db_file = tmp_path / "alerts.db"
    monkeypatch.setenv("MONITORING_ALERTS_DB", str(db_file))
    monkeypatch.setenv("MONITORING_ENABLED", "true")
    wl_file = tmp_path / "watchlists.json"
    wl_file.write_text(json.dumps({"watchlists": []}), encoding="utf-8")
    monkeypatch.setenv("MONITORING_WATCHLISTS_FILE", str(wl_file))

    _reset_topic_monitoring_service()
    svc = get_topic_monitoring_service()
    svc.reload()

    wl = Watchlist(
        name="Streaming WL",
        description="Detect 'alert'",
        enabled=True,
        scope_type="user",
        scope_id="u1",
        rules=[WatchlistRule(pattern="alert", category="custom", severity="warning")],
    )
    svc.upsert_watchlist(wl)

    created1 = svc.evaluate_and_alert(
        user_id="u1",
        text="alert me please",
        source="chat.output",
        source_id="stream-1",
        chunk_id="stream-1:1",
        chunk_seq=1,
    )
    created2 = svc.evaluate_and_alert(
        user_id="u1",
        text="alert me please",
        source="chat.output",
        source_id="stream-1",
        chunk_id="stream-1:2",
        chunk_seq=2,
    )

    assert created1 == 1
    assert created2 == 0

    db = TopicMonitoringDB(db_path=str(db_file))
    items = db.list_alerts(user_id="u1")
    assert len(items) == 1


def test_topic_monitoring_streaming_dedupe_is_per_watchlist(tmp_path, monkeypatch):
    db_file = tmp_path / "alerts.db"
    monkeypatch.setenv("MONITORING_ALERTS_DB", str(db_file))
    monkeypatch.setenv("MONITORING_ENABLED", "true")
    wl_file = tmp_path / "watchlists.json"
    wl_file.write_text(json.dumps({"watchlists": []}), encoding="utf-8")
    monkeypatch.setenv("MONITORING_WATCHLISTS_FILE", str(wl_file))

    _reset_topic_monitoring_service()
    svc = get_topic_monitoring_service()
    svc.reload()

    rule = WatchlistRule(rule_id="shared-rule", pattern="alert", category="custom", severity="warning")
    wl1 = Watchlist(
        name="Streaming WL One",
        description="First watchlist",
        enabled=True,
        scope_type="user",
        scope_id="u1",
        rules=[rule],
    )
    wl2 = Watchlist(
        name="Streaming WL Two",
        description="Second watchlist",
        enabled=True,
        scope_type="user",
        scope_id="u1",
        rules=[rule],
    )
    svc.upsert_watchlist(wl1)
    svc.upsert_watchlist(wl2)

    created = svc.evaluate_and_alert(
        user_id="u1",
        text="alert me please",
        source="chat.output",
        source_id="stream-1",
        chunk_id="stream-1:1",
        chunk_seq=1,
    )

    assert created == 2

    db = TopicMonitoringDB(db_path=str(db_file))
    items = db.list_alerts(user_id="u1")
    assert len(items) == 2


def test_topic_monitoring_allows_duplicate_rule_ids_across_watchlists(tmp_path, monkeypatch):
    db_file = tmp_path / "alerts.db"
    monkeypatch.setenv("MONITORING_ALERTS_DB", str(db_file))
    monkeypatch.setenv("MONITORING_ENABLED", "true")
    wl_file = tmp_path / "watchlists.json"
    wl_file.write_text(json.dumps({"watchlists": []}), encoding="utf-8")
    monkeypatch.setenv("MONITORING_WATCHLISTS_FILE", str(wl_file))

    _reset_topic_monitoring_service()
    svc = get_topic_monitoring_service()
    svc.reload()

    rule = WatchlistRule(pattern="shared", category="custom", severity="warning")
    wl1 = Watchlist(
        name="WL One",
        description="First watchlist",
        enabled=True,
        scope_type="user",
        scope_id="u1",
        rules=[rule],
    )
    wl2 = Watchlist(
        name="WL Two",
        description="Second watchlist",
        enabled=True,
        scope_type="user",
        scope_id="u2",
        rules=[rule],
    )

    svc.upsert_watchlist(wl1)
    svc.upsert_watchlist(wl2)

    db = TopicMonitoringDB(db_path=str(db_file))
    watchlists = db.list_watchlists(include_rules=True)
    by_name = {wl.get("name"): wl for wl in watchlists}
    rule_id_1 = by_name["WL One"]["rules"][0]["rule_id"]
    rule_id_2 = by_name["WL Two"]["rules"][0]["rule_id"]
    assert rule_id_1 == rule_id_2


def test_list_watchlists_include_rules_materializes_rows_without_index_error(tmp_path: Path) -> None:
    db_file = tmp_path / "alerts.db"
    db = TopicMonitoringDB(db_path=str(db_file))

    watchlist_id = "wl-materialize"
    db.upsert_watchlist(
        WatchlistRecord(
            id=watchlist_id,
            name="Materialize Test",
            scope_type="user",
            scope_id="u1",
        )
    )
    db.replace_watchlist_rules(
        watchlist_id,
        [
            WatchlistRuleRecord(
                rule_id="rule-1",
                watchlist_id=watchlist_id,
                pattern="badword",
                category="custom",
                severity="warning",
                tags=["a", "b"],
            )
        ],
    )

    rows = db.list_watchlists(include_rules=True)

    assert len(rows) == 1
    assert rows[0]["id"] == watchlist_id
    assert rows[0]["rules"][0]["rule_id"] == "rule-1"
    assert rows[0]["rules"][0]["watchlist_id"] == watchlist_id


def test_topic_monitoring_global_watchlist_without_user_id(tmp_path, monkeypatch):
    db_file = tmp_path / "alerts.db"
    monkeypatch.setenv("MONITORING_ALERTS_DB", str(db_file))
    monkeypatch.setenv("MONITORING_ENABLED", "true")
    wl_file = tmp_path / "watchlists.json"
    wl_file.write_text(json.dumps({"watchlists": []}), encoding="utf-8")
    monkeypatch.setenv("MONITORING_WATCHLISTS_FILE", str(wl_file))

    _reset_topic_monitoring_service()
    svc = get_topic_monitoring_service()
    svc.reload()

    wl = Watchlist(
        name="Global WL",
        description="Global watchlist",
        enabled=True,
        scope_type="global",
        scope_id=None,
        rules=[WatchlistRule(pattern="needle", category="custom", severity="warning")],
    )
    svc.upsert_watchlist(wl)

    created = svc.evaluate_and_alert(user_id=None, text="find the needle here", source="ingestion")
    assert created >= 1

    db = TopicMonitoringDB(db_path=str(db_file))
    items = db.list_alerts(scope_type="global")
    assert len(items) >= 1
    assert items[0].get("user_id") is None


def test_topic_monitoring_global_dedupe(tmp_path, monkeypatch):
    db_file = tmp_path / "alerts.db"
    monkeypatch.setenv("MONITORING_ALERTS_DB", str(db_file))
    monkeypatch.setenv("MONITORING_ENABLED", "true")
    wl_file = tmp_path / "watchlists.json"
    wl_file.write_text(json.dumps({"watchlists": []}), encoding="utf-8")
    monkeypatch.setenv("MONITORING_WATCHLISTS_FILE", str(wl_file))

    _reset_topic_monitoring_service()
    svc = get_topic_monitoring_service()
    svc.reload()

    wl = Watchlist(
        name="Global WL Dedupe",
        description="Global watchlist",
        enabled=True,
        scope_type="global",
        scope_id=None,
        rules=[WatchlistRule(pattern="needle", category="custom", severity="warning")],
    )
    svc.upsert_watchlist(wl)

    created1 = svc.evaluate_and_alert(user_id=None, text="needle here", source="ingestion")
    created2 = svc.evaluate_and_alert(user_id=None, text="needle here", source="ingestion")

    assert created1 == 1
    assert created2 == 0


def test_list_alerts_without_user_id_returns_all(tmp_path: Path) -> None:
    db_file = tmp_path / "alerts.db"
    db = TopicMonitoringDB(db_path=str(db_file))
    db.insert_alert(
        TopicAlert(
            user_id=None,
            scope_type="global",
            scope_id=None,
            source="ingestion",
            watchlist_id="w1",
            rule_category="test",
            rule_severity="warning",
            pattern="needle",
            text_snippet="needle",
        )
    )
    db.insert_alert(
        TopicAlert(
            user_id="u1",
            scope_type="user",
            scope_id="u1",
            source="chat.input",
            watchlist_id="w2",
            rule_category="test",
            rule_severity="warning",
            pattern="badword",
            text_snippet="badword",
        )
    )

    all_items = db.list_alerts()
    assert len(all_items) == 2

    user_items = db.list_alerts(user_id="u1")
    assert len(user_items) == 1
    assert user_items[0].get("user_id") == "u1"


def test_topic_monitoring_reload_updates_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db1 = tmp_path / "db1" / "alerts.db"
    wl1 = tmp_path / "wl1" / "watchlists.json"
    wl1.parent.mkdir(parents=True, exist_ok=True)
    wl1.write_text(json.dumps({"watchlists": []}), encoding="utf-8")
    monkeypatch.setenv("MONITORING_ALERTS_DB", str(db1))
    monkeypatch.setenv("MONITORING_WATCHLISTS_FILE", str(wl1))
    monkeypatch.setenv("MONITORING_ENABLED", "true")

    _reset_topic_monitoring_service()
    svc = get_topic_monitoring_service()

    db2 = tmp_path / "db2" / "alerts.db"
    wl2 = tmp_path / "wl2" / "watchlists.json"
    wl2.parent.mkdir(parents=True, exist_ok=True)
    wl2.write_text(json.dumps({"watchlists": []}), encoding="utf-8")
    monkeypatch.setenv("MONITORING_ALERTS_DB", str(db2))
    monkeypatch.setenv("MONITORING_WATCHLISTS_FILE", str(wl2))

    svc.reload()

    assert Path(svc._db_path) == db2
    assert Path(svc._watchlists_path) == wl2

    wl = Watchlist(
        name="Reload WL",
        description="Reload target",
        enabled=True,
        scope_type="user",
        scope_id="u1",
        rules=[WatchlistRule(pattern="reload", category="custom", severity="warning")],
    )
    svc.upsert_watchlist(wl)
    created = svc.evaluate_and_alert(user_id="u1", text="reload", source="chat.input")
    assert created == 1

    db = TopicMonitoringDB(db_path=str(db2))
    items = db.list_alerts(user_id="u1")
    assert len(items) == 1


def test_topic_monitoring_reload_refreshes_dedupe_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_file = tmp_path / "alerts.db"
    wl_file = tmp_path / "watchlists.json"
    wl_file.write_text(json.dumps({"watchlists": []}), encoding="utf-8")

    monkeypatch.setenv("MONITORING_ALERTS_DB", str(db_file))
    monkeypatch.setenv("MONITORING_WATCHLISTS_FILE", str(wl_file))
    monkeypatch.setenv("MONITORING_ENABLED", "true")
    monkeypatch.setenv("TOPIC_MONITOR_DEDUP_SECONDS", "300")
    monkeypatch.setenv("TOPIC_MONITOR_SIMHASH_DISTANCE", "3")

    _reset_topic_monitoring_service()
    svc = get_topic_monitoring_service()

    assert svc._dedup_window_seconds == 300
    assert svc._simhash_distance == 3

    monkeypatch.setenv("TOPIC_MONITOR_DEDUP_SECONDS", "30")
    monkeypatch.setenv("TOPIC_MONITOR_SIMHASH_DISTANCE", "1")

    svc.reload()

    assert svc._dedup_window_seconds == 30
    assert svc._simhash_distance == 1


def test_topic_monitoring_dedupe_prunes_stale_streams(tmp_path, monkeypatch):
    import tldw_Server_API.app.core.Monitoring.topic_monitoring_service as tms

    db_file = tmp_path / "alerts.db"
    monkeypatch.setenv("MONITORING_ALERTS_DB", str(db_file))
    monkeypatch.setenv("MONITORING_ENABLED", "true")
    monkeypatch.setenv("TOPIC_MONITOR_DEDUP_SECONDS", "1")
    wl_file = tmp_path / "watchlists.json"
    wl_file.write_text(json.dumps({"watchlists": []}), encoding="utf-8")
    monkeypatch.setenv("MONITORING_WATCHLISTS_FILE", str(wl_file))

    _reset_topic_monitoring_service()
    svc = get_topic_monitoring_service()
    svc.reload()

    monkeypatch.setattr(tms.time, "monotonic", lambda: 0.0)
    svc._dedupe_should_skip(stream_id="s1", rule_id="r1", text="alpha")
    assert "s1" in svc._dedupe_state

    monkeypatch.setattr(tms.time, "monotonic", lambda: 2.0)
    svc._dedupe_should_skip(stream_id="s2", rule_id="r1", text="beta")
    assert "s1" not in svc._dedupe_state


def test_topic_monitoring_notify_fallback_log_omits_raw_exception(tmp_path, monkeypatch):
    import tldw_Server_API.app.core.Monitoring.topic_monitoring_service as tms

    db_file = tmp_path / "alerts.db"
    monkeypatch.setenv("MONITORING_ALERTS_DB", str(db_file))
    monkeypatch.setenv("MONITORING_ENABLED", "true")
    wl_file = tmp_path / "watchlists.json"
    wl_file.write_text(json.dumps({"watchlists": []}), encoding="utf-8")
    monkeypatch.setenv("MONITORING_WATCHLISTS_FILE", str(wl_file))

    _reset_topic_monitoring_service()
    svc = get_topic_monitoring_service()
    svc.reload()

    wl = Watchlist(
        name="Notify Fallback WL",
        description="Force notification fallback",
        enabled=True,
        scope_type="user",
        scope_id="u1",
        rules=[WatchlistRule(pattern="notify", category="custom", severity="warning")],
    )
    svc.upsert_watchlist(wl)

    class FailingNotifier:
        def notify(self, alert):
            raise RuntimeError("raw secret path /tmp/topic-monitoring-secret api_key=abc123")

    logs: list[str] = []
    handler_id = logger.add(logs.append, level="DEBUG", format="{message}")
    monkeypatch.setattr(tms, "get_notification_service", lambda: FailingNotifier())
    try:
        created = svc.evaluate_and_alert(user_id="u1", text="notify user", source="chat.input")
    finally:
        logger.remove(handler_id)

    assert created == 1
    joined = "\n".join(logs)
    assert "Topic monitoring notify skipped" in joined
    assert "raw secret path" not in joined
    assert "/tmp/topic-monitoring-secret" not in joined
    assert "api_key=abc123" not in joined


@pytest.mark.asyncio
async def test_topic_monitoring_background_fallback_log_omits_raw_exception(monkeypatch):
    svc = TopicMonitoringService.__new__(TopicMonitoringService)
    monkeypatch.setattr(svc, "_monitoring_active", lambda: True)
    monkeypatch.setattr(svc, "_applicable_watchlists", lambda *args, **kwargs: [("wl", object())])

    def fail_evaluate(*args, **kwargs):
        raise RuntimeError("raw background path /tmp/topic-background-secret token=abc123")

    async def run_inline(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(svc, "evaluate_and_alert", fail_evaluate)
    monkeypatch.setattr(asyncio, "to_thread", run_inline)

    logs: list[str] = []
    handler_id = logger.add(logs.append, level="DEBUG", format="{message}")
    try:
        svc.schedule_evaluate_and_alert(user_id="u1", text="body", source="chat.input")
        await asyncio.sleep(0)
    finally:
        logger.remove(handler_id)

    joined = "\n".join(logs)
    assert "Topic monitoring background evaluation failed" in joined
    assert "raw background path" not in joined
    assert "/tmp/topic-background-secret" not in joined
    assert "token=abc123" not in joined
