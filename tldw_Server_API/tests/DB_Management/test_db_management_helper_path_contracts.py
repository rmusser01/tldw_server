from pathlib import Path

import pytest


def test_topic_monitoring_db_uses_shared_trusted_path(monkeypatch, tmp_path) -> None:
    from tldw_Server_API.app.core.DB_Management import TopicMonitoring_DB as topic_mod

    calls = []
    safe_path = tmp_path / "safe-topic.db"
    input_path = tmp_path / "monitoring" / "input-topic.db"

    monkeypatch.setattr(
        topic_mod,
        "resolve_trusted_database_path",
        lambda db_path, **kwargs: calls.append((db_path, kwargs["label"])) or safe_path,
    )

    db = topic_mod.TopicMonitoringDB(str(input_path))

    if db.db_path != str(safe_path):
        pytest.fail("expected TopicMonitoringDB to store the shared trusted path result")
    expected_calls = [(str(input_path), "topic monitoring db")]
    if calls != expected_calls:
        pytest.fail(f"expected TopicMonitoringDB trusted-path call {expected_calls!r}, got {calls!r}")


def test_watchlist_alert_rules_helpers_use_shared_trusted_path(monkeypatch, tmp_path) -> None:
    from tldw_Server_API.app.core.DB_Management import watchlist_alert_rules_db as alert_rules_db

    calls = []
    safe_path = tmp_path / "safe-alerts.db"
    input_path = tmp_path / "input-alerts.db"

    monkeypatch.setattr(
        alert_rules_db,
        "resolve_trusted_database_path",
        lambda db_path, **kwargs: calls.append((db_path, kwargs["label"])) or safe_path,
    )

    alert_rules_db.ensure_watchlist_alert_rules_table(str(input_path))

    expected_calls = [(str(input_path), "watchlist alert rules db")]
    if calls != expected_calls:
        pytest.fail(
            f"expected watchlist alert rules trusted-path call {expected_calls!r}, got {calls!r}"
        )


def test_voice_registry_db_uses_shared_trusted_path(monkeypatch, tmp_path) -> None:
    from tldw_Server_API.app.core.DB_Management import Voice_Registry_DB as voice_mod

    calls = []
    user_db_base = tmp_path / "user-dbs"
    safe_path = user_db_base / "safe-voice.db"
    input_path = tmp_path / "input-voice.db"

    monkeypatch.setattr(
        voice_mod,
        "resolve_trusted_database_path",
        lambda db_path, **kwargs: calls.append((db_path, kwargs["label"])) or safe_path,
    )
    monkeypatch.setattr(
        voice_mod.DatabasePaths,
        "get_user_db_base_dir",
        lambda *args, **kwargs: user_db_base,
        raising=True,
    )

    db = voice_mod.VoiceRegistryDB(input_path)

    if db.db_path != safe_path:
        pytest.fail("expected VoiceRegistryDB to store the shared trusted path result")
    expected_calls = [(input_path, "voice registry db")]
    if calls != expected_calls:
        pytest.fail(f"expected VoiceRegistryDB trusted-path call {expected_calls!r}, got {calls!r}")


def test_guardian_db_uses_trusted_parent_dir_helper(monkeypatch, tmp_path) -> None:
    from tldw_Server_API.app.core.DB_Management import Guardian_DB as guardian_mod

    calls = []
    safe_path = tmp_path / "safe-guardian.db"
    input_path = tmp_path / "guardian" / "input-guardian.db"

    monkeypatch.setattr(
        guardian_mod,
        "ensure_trusted_database_parent_dir",
        lambda db_path, **kwargs: calls.append((db_path, kwargs["label"])) or safe_path,
    )

    db = guardian_mod.GuardianDB(str(input_path))

    if db.db_path != str(safe_path):
        pytest.fail("expected GuardianDB to store the trusted parent helper result")
    expected_calls = [(Path(input_path), "guardian database")]
    if calls != expected_calls:
        pytest.fail(f"expected GuardianDB helper call {expected_calls!r}, got {calls!r}")


def test_personalization_db_uses_trusted_parent_exists_helper(monkeypatch, tmp_path) -> None:
    from tldw_Server_API.app.core.DB_Management import Personalization_DB as personalization_mod

    calls = []
    safe_path = tmp_path / "safe-personalization.db"
    input_path = tmp_path / "personalization" / "input-personalization.db"

    monkeypatch.setattr(
        personalization_mod,
        "require_trusted_database_parent_exists",
        lambda db_path, **kwargs: calls.append((db_path, kwargs["label"])) or safe_path,
    )

    db = personalization_mod.PersonalizationDB(str(input_path))

    if db.db_path != str(safe_path):
        pytest.fail("expected PersonalizationDB to store the trusted parent helper result")
    expected_calls = [(str(input_path), "personalization database")]
    if calls != expected_calls:
        pytest.fail(f"expected PersonalizationDB helper call {expected_calls!r}, got {calls!r}")
