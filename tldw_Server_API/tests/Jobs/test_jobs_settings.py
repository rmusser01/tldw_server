"""Contract tests for Jobs settings snapshot and classification behavior."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest

from tldw_Server_API.app.core.Jobs.settings import JobsSettingMode, JobsSettings

pytestmark = pytest.mark.unit


def _repo_root() -> Path:
    """Return the repository root from this test module location."""

    return Path(__file__).resolve().parents[3]


def _literal_jobs_env_keys(path: Path) -> set[str]:
    """Collect literal JOBS_* keys read by direct environment helpers."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        is_env_reader = (
            isinstance(node.func, ast.Attribute) and node.func.attr == "getenv"
        ) or (
            isinstance(node.func, ast.Name) and node.func.id == "env_flag_enabled"
        )
        if not is_env_reader:
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str) and first_arg.value.startswith("JOBS_"):
            keys.add(first_arg.value)
    return keys


def _literal_jobs_quota_bases(path: Path) -> set[str]:
    """Collect literal JOBS_* quota base keys read by quota helpers."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    keys: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "_quota_get":
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str) and first_arg.value.startswith("JOBS_"):
            keys.add(first_arg.value)
    return keys


def test_jobs_settings_snapshots_construction_time_values() -> None:
    """Verify settings snapshot construction-time values immediately."""

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
    """Verify refresh reads updated snapshot-refreshable integer values."""

    env = {"JOBS_MAX_JSON_BYTES": "123", "JOBS_LEASE_MAX_SECONDS": "45"}
    settings = JobsSettings.from_env(env)
    env["JOBS_MAX_JSON_BYTES"] = "456"

    refreshed = settings.refresh(env)

    assert settings.max_json_bytes == 123
    assert refreshed.max_json_bytes == 456
    assert refreshed.lease_max_seconds == 45


def test_jobs_settings_refresh_preserves_construction_time_values() -> None:
    """Verify refresh preserves DB settings that are construction-time only."""

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
    """Verify refresh reads updated booleans and queue allow-list extras."""

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
    assert refreshed.allowed_queue_extras_for_domain(None) == ["default", "high"]
    assert refreshed.allowed_queue_extras_for_domain("chatbooks") == ["default", "high", "import", "export"]


def test_jobs_settings_allowed_queue_extras_are_domain_aware() -> None:
    """Verify domain-specific queue extras extend global extras."""

    settings = JobsSettings.from_env(
        {
            "JOBS_ALLOWED_QUEUES": "default,low",
            "JOBS_ALLOWED_QUEUES_CHATBOOKS": "export,import",
        }
    )

    assert settings.allowed_queue_extras_for_domain(None) == ["default", "low"]
    assert settings.allowed_queue_extras_for_domain("chatbooks") == ["default", "low", "export", "import"]


def test_jobs_settings_allowed_queue_extras_remove_duplicates() -> None:
    """Verify queue extras preserve order while removing duplicates."""

    settings = JobsSettings.from_env(
        {
            "JOBS_ALLOWED_QUEUES": "default,low,default",
            "JOBS_ALLOWED_QUEUES_CHATBOOKS": "low,export,export",
        }
    )

    assert settings.allowed_queue_extras_for_domain("chatbooks") == ["default", "low", "export"]


def test_jobs_settings_ignores_none_domain_queue_values() -> None:
    """Verify null domain queue values do not become literal queue names."""

    env = cast(Mapping[str, str], {"JOBS_ALLOWED_QUEUES_CHATBOOKS": None})

    settings = JobsSettings.from_env(env)

    assert settings.allowed_queue_extras_for_domain("chatbooks") == []


def test_jobs_settings_classifies_known_keys() -> None:
    """Verify known Jobs keys map to their expected consumption phase."""

    assert JobsSettings.setting_mode("JOBS_DB_URL") is JobsSettingMode.CONSTRUCTION_TIME
    assert JobsSettings.setting_mode("JOBS_DB_PATH") is JobsSettingMode.CONSTRUCTION_TIME
    assert JobsSettings.setting_mode("JOBS_TEST_NOW_EPOCH") is JobsSettingMode.CONSTRUCTION_TIME
    assert JobsSettings.setting_mode("JOBS_MAX_JSON_BYTES") is JobsSettingMode.SNAPSHOT_REFRESHABLE
    assert JobsSettings.setting_mode("JOBS_LEASE_MAX_SECONDS") is JobsSettingMode.SNAPSHOT_REFRESHABLE
    assert JobsSettings.setting_mode("JOBS_EVENTS_OUTBOX") is JobsSettingMode.SNAPSHOT_REFRESHABLE
    assert JobsSettings.setting_mode("JOBS_COUNTERS_ENABLED") is JobsSettingMode.SNAPSHOT_REFRESHABLE
    assert JobsSettings.setting_mode("JOBS_ALLOWED_QUEUES") is JobsSettingMode.SNAPSHOT_REFRESHABLE
    assert JobsSettings.setting_mode("JOBS_ALLOWED_QUEUES_CHATBOOKS") is JobsSettingMode.SNAPSHOT_REFRESHABLE
    assert JobsSettings.setting_mode("JOBS_ALLOWED_JOB_TYPES_CHATBOOKS") is JobsSettingMode.OPERATION_TIME
    assert JobsSettings.setting_mode("JOBS_EXPIRED_RECOVERY_BATCH_SIZE") is JobsSettingMode.OPERATION_TIME
    assert JobsSettings.setting_mode("JOBS_QUOTA_MAX_INFLIGHT") is JobsSettingMode.OPERATION_TIME
    assert JobsSettings.setting_mode("JOBS_QUOTA_MAX_INFLIGHT_CHATBOOKS_USER_1") is JobsSettingMode.OPERATION_TIME
    assert JobsSettings.setting_mode("JOBS_PG_ACQUIRE_PRIORITY_DESC_DOMAINS") is JobsSettingMode.OPERATION_TIME
    assert JobsSettings.setting_mode("JOBS_SQLITE_ACQUIRE_TIE_BREAK") is JobsSettingMode.OPERATION_TIME
    assert JobsSettings.setting_mode("JOBS_ACQUIRE_TIE_BREAK_CHATBOOKS") is JobsSettingMode.OPERATION_TIME
    assert JobsSettings.setting_mode("JOBS_UNKNOWN") is JobsSettingMode.UNCLASSIFIED


def test_jobs_settings_classifies_current_manager_and_admin_env_keys() -> None:
    """Verify current Jobs runtime env reads are covered by classification."""

    root = _repo_root()
    paths = [
        root / "tldw_Server_API/app/core/Jobs/manager.py",
        root / "tldw_Server_API/app/api/v1/endpoints/jobs_admin.py",
    ]
    for path in paths:
        assert path.exists()

    keys = {key for path in paths for key in _literal_jobs_env_keys(path)}
    keys.update(key for path in paths for key in _literal_jobs_quota_bases(path))
    keys.update(
        {
            "JOBS_PG_ACQUIRE_PRIORITY_DESC_DOMAINS",
            "JOBS_POSTGRES_ACQUIRE_PRIORITY_DESC_DOMAINS",
            "JOBS_SQLITE_ACQUIRE_PRIORITY_DESC_DOMAINS",
            "JOBS_PG_ACQUIRE_TIE_BREAK",
            "JOBS_SQLITE_ACQUIRE_TIE_BREAK",
            "JOBS_PG_ACQUIRE_TIE_BREAK_CHATBOOKS",
            "JOBS_ACQUIRE_TIE_BREAK_CHATBOOKS",
        }
    )

    assert keys
    assert {
        key for key in keys if JobsSettings.setting_mode(key) is JobsSettingMode.UNCLASSIFIED
    } == set()
