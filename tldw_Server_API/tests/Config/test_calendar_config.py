"""Calendar's typed settings stay live, env-only and outside public config."""

from dataclasses import asdict

import pytest

pytestmark = pytest.mark.unit


def test_calendar_worker_loader_resolves_only_existing_keys_in_order():
    from tldw_Server_API.app.core.config_sections.calendar import load_calendar_sync_worker_config

    class Env(dict):
        def get(self, key):
            asked.append(key)
            return super().get(key)

    asked = []
    settings = load_calendar_sync_worker_config(default_worker_id="default-worker", env=Env())
    assert asdict(settings) == {
        "worker_id": "default-worker", "lease_seconds": 120,
        "renew_threshold_seconds": 10, "renew_jitter_seconds": 0,
    }
    assert asked == ["CALENDAR_SYNC_WORKER_ID", "JOBS_LEASE_SECONDS", "CALENDAR_SYNC_LEASE_SECONDS",
                     "CALENDAR_SYNC_RENEW_THRESHOLD_SECONDS", "CALENDAR_SYNC_RENEW_JITTER_SECONDS"]


def test_calendar_startup_loader_does_not_parse_inactive_worker_settings():
    from tldw_Server_API.app.core.config_sections.calendar import load_calendar_sync_startup_config

    assert asdict(load_calendar_sync_startup_config(env={
        "CALENDAR_SYNC_LEASE_SECONDS": "invalid", "CALENDAR_SECRET_ENCRYPTION_KEY": "test-value",  # nosec B105 - synthetic inactive config; never used as a credential.
    })) == {"jobs_worker_enabled": False, "scheduler_enabled": False}


def test_calendar_startup_flags_fail_closed_on_environment_read_error():
    from tldw_Server_API.app.core.config_sections.calendar import load_calendar_sync_startup_config

    class BrokenEnv(dict):
        def get(self, key):
            raise OSError("unavailable")

    assert asdict(load_calendar_sync_startup_config(env=BrokenEnv())) == {
        "jobs_worker_enabled": False, "scheduler_enabled": False,
    }


@pytest.mark.parametrize("error", [OSError, TypeError, ValueError, RuntimeError])
def test_calendar_scheduler_preserves_guarded_read_error_fallback(monkeypatch, error):
    from tldw_Server_API.app.core.config_sections import calendar
    from tldw_Server_API.app.services.calendar_sync_scheduler import _scheduler_interval_seconds

    def unavailable(*args, **kwargs):
        raise error("unavailable")

    monkeypatch.setattr(calendar, "resolve", unavailable)
    assert _scheduler_interval_seconds() == 60.0


def test_calendar_secret_is_env_only_live_and_absent_from_aggregate(monkeypatch):
    import configparser

    from tldw_Server_API.app.core.config_sections import load_config_sections
    from tldw_Server_API.app.core.config_sections.calendar import calendar_secret_encryption_key

    monkeypatch.setenv("CALENDAR_SECRET_ENCRYPTION_KEY", " test-only-calendar-key ")
    assert calendar_secret_encryption_key() == "test-only-calendar-key"
    assert calendar_secret_encryption_key(env={}) == ""
    parser = configparser.ConfigParser()
    parser.read_dict({"Calendar": {"secret_encryption_key": "file-key"}})  # nosec B105 - synthetic file-config value must not supply the env-only secret.
    aggregate = load_config_sections(parser)
    assert "test-only-calendar-key" not in repr(aggregate)
    assert "calendar" not in asdict(aggregate)
    monkeypatch.delenv("CALENDAR_SECRET_ENCRYPTION_KEY")
    assert calendar_secret_encryption_key() == ""
