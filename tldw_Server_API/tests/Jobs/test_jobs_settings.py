from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Jobs.settings import JobsSettingMode, JobsSettings


pytestmark = pytest.mark.unit


def test_jobs_settings_snapshots_construction_time_values() -> None:
    env = {
        "JOBS_DB_URL": "postgresql://example/jobs",
        "JOBS_DB_PATH": "data/jobs-a.db",
        "JOBS_MAX_JSON_BYTES": "123",
        "JOBS_LEASE_MAX_SECONDS": "45",
        "JOBS_EVENTS_OUTBOX": "true",
        "JOBS_COUNTERS_ENABLED": "false",
    }

    settings = JobsSettings.from_env(env)
    env["JOBS_MAX_JSON_BYTES"] = "999"

    assert settings.db_url == "postgresql://example/jobs"
    assert settings.db_path == "data/jobs-a.db"
    assert settings.max_json_bytes == 123
    assert settings.lease_max_seconds == 45
    assert settings.events_outbox_enabled is True
    assert settings.counters_enabled is False


def test_jobs_settings_refresh_reads_new_environment_values() -> None:
    env = {"JOBS_MAX_JSON_BYTES": "123", "JOBS_LEASE_MAX_SECONDS": "45"}
    settings = JobsSettings.from_env(env)
    env["JOBS_MAX_JSON_BYTES"] = "456"

    refreshed = settings.refresh(env)

    assert settings.max_json_bytes == 123
    assert refreshed.max_json_bytes == 456
    assert refreshed.lease_max_seconds == 45


def test_jobs_settings_refresh_preserves_construction_time_values() -> None:
    settings = JobsSettings.from_env(
        {
            "JOBS_DB_URL": "postgresql://example/original",
            "JOBS_DB_PATH": "data/jobs-original.db",
            "JOBS_MAX_JSON_BYTES": "123",
            "JOBS_LEASE_MAX_SECONDS": "45",
        }
    )

    refreshed = settings.refresh(
        {
            "JOBS_DB_URL": "postgresql://example/replacement",
            "JOBS_DB_PATH": "data/jobs-replacement.db",
            "JOBS_MAX_JSON_BYTES": "456",
            "JOBS_LEASE_MAX_SECONDS": "67",
        }
    )

    assert refreshed.db_url == "postgresql://example/original"
    assert refreshed.db_path == "data/jobs-original.db"
    assert refreshed.max_json_bytes == 456
    assert refreshed.lease_max_seconds == 67


def test_jobs_settings_refresh_reads_new_booleans_and_allowed_queues() -> None:
    settings = JobsSettings.from_env(
        {
            "JOBS_EVENTS_OUTBOX": "false",
            "JOBS_COUNTERS_ENABLED": "true",
            "JOBS_ALLOWED_QUEUES": "default,low",
            "JOBS_ALLOWED_QUEUES_CHATBOOKS": "export",
        }
    )

    refreshed = settings.refresh(
        {
            "JOBS_EVENTS_OUTBOX": "true",
            "JOBS_COUNTERS_ENABLED": "false",
            "JOBS_ALLOWED_QUEUES": "default,high",
            "JOBS_ALLOWED_QUEUES_CHATBOOKS": "import,export",
        }
    )

    assert refreshed.events_outbox_enabled is True
    assert refreshed.counters_enabled is False
    assert refreshed.allowed_queues_for_domain(None) == ["default", "high"]
    assert refreshed.allowed_queues_for_domain("chatbooks") == ["default", "high", "import", "export"]


def test_jobs_settings_allowed_queues_are_domain_aware() -> None:
    settings = JobsSettings.from_env(
        {
            "JOBS_ALLOWED_QUEUES": "default,low",
            "JOBS_ALLOWED_QUEUES_CHATBOOKS": "export,import",
        }
    )

    assert settings.allowed_queues_for_domain(None) == ["default", "low"]
    assert settings.allowed_queues_for_domain("chatbooks") == ["default", "low", "export", "import"]


def test_jobs_settings_allowed_queues_remove_duplicates() -> None:
    settings = JobsSettings.from_env(
        {
            "JOBS_ALLOWED_QUEUES": "default,low,default",
            "JOBS_ALLOWED_QUEUES_CHATBOOKS": "low,export,export",
        }
    )

    assert settings.allowed_queues_for_domain("chatbooks") == ["default", "low", "export"]


def test_jobs_settings_classifies_known_keys() -> None:
    assert JobsSettings.setting_mode("JOBS_DB_URL") is JobsSettingMode.CONSTRUCTION_TIME
    assert JobsSettings.setting_mode("JOBS_DB_PATH") is JobsSettingMode.CONSTRUCTION_TIME
    assert JobsSettings.setting_mode("JOBS_MAX_JSON_BYTES") is JobsSettingMode.SNAPSHOT_REFRESHABLE
    assert JobsSettings.setting_mode("JOBS_LEASE_MAX_SECONDS") is JobsSettingMode.SNAPSHOT_REFRESHABLE
    assert JobsSettings.setting_mode("JOBS_EVENTS_OUTBOX") is JobsSettingMode.SNAPSHOT_REFRESHABLE
    assert JobsSettings.setting_mode("JOBS_COUNTERS_ENABLED") is JobsSettingMode.SNAPSHOT_REFRESHABLE
    assert JobsSettings.setting_mode("JOBS_ALLOWED_QUEUES") is JobsSettingMode.OPERATION_TIME
    assert JobsSettings.setting_mode("JOBS_ALLOWED_QUEUES_CHATBOOKS") is JobsSettingMode.OPERATION_TIME
    assert JobsSettings.setting_mode("JOBS_UNKNOWN") is JobsSettingMode.OPERATION_TIME
