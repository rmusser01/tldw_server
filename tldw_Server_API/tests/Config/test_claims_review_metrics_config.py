import configparser

import pytest

from tldw_Server_API.app.core import config
from tldw_Server_API.app.services.claims_review_metrics_scheduler import resolve_scheduler_config

pytestmark = pytest.mark.unit

VALUES = {
    "CLAIMS_REVIEW_METRICS_SCHEDULER_ENABLED": "true",
    "CLAIMS_REVIEW_METRICS_INTERVAL_SEC": "7200",
    "CLAIMS_REVIEW_METRICS_LOOKBACK_DAYS": "7",
    "CLAIMS_REVIEW_METRICS_JOBS_ENABLED": "true",
    "CLAIMS_JOBS_ENABLED": "true",
    "CLAIMS_JOBS_QUEUE": "metrics",
    "CLAIMS_JOBS_MAX_RETRIES_REVIEW_METRICS": "4",
}


@pytest.fixture
def parser(monkeypatch):
    parser = configparser.ConfigParser()
    parser.read_dict({"ClaimsMonitoring": VALUES})
    for key in VALUES:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(config, "_load_env_files_early", lambda: None)
    monkeypatch.setattr(config, "load_and_log_configs", lambda: {})
    monkeypatch.setattr(config, "load_comprehensive_config", lambda: parser)
    return parser


def test_file_settings_reach_scheduler_route_window_queue_and_retries(parser):
    resolved = resolve_scheduler_config(config.load_settings())
    assert resolved.enabled and resolved.mode == "jobs"
    assert (resolved.interval_seconds, resolved.lookback_days) == (7200, 7)
    assert resolved.job_settings == {"CLAIMS_JOBS_QUEUE": "metrics", "CLAIMS_JOBS_MAX_RETRIES_REVIEW_METRICS": "4"}


def test_environment_overrides_config_without_numeric_load_failure(parser, monkeypatch):
    monkeypatch.setenv("CLAIMS_REVIEW_METRICS_SCHEDULER_ENABLED", "false")
    monkeypatch.setenv("CLAIMS_REVIEW_METRICS_INTERVAL_SEC", "bad")
    monkeypatch.setenv("CLAIMS_REVIEW_METRICS_LOOKBACK_DAYS", "1000")
    monkeypatch.setenv("CLAIMS_JOBS_QUEUE", "replacement")
    resolved = resolve_scheduler_config(config.load_settings())
    assert not resolved.enabled
    assert (resolved.interval_seconds, resolved.lookback_days) == (86400, 366)
    assert resolved.job_settings["CLAIMS_JOBS_QUEUE"] == "replacement"


def test_invalid_config_numbers_remain_for_consumer_normalization(parser):
    parser.set("ClaimsMonitoring", "CLAIMS_REVIEW_METRICS_INTERVAL_SEC", "bad")
    parser.set("ClaimsMonitoring", "CLAIMS_REVIEW_METRICS_LOOKBACK_DAYS", "-1")
    resolved = resolve_scheduler_config(config.load_settings())
    assert (resolved.interval_seconds, resolved.lookback_days) == (86400, 2)


def test_malformed_interpolation_in_numeric_file_value_reaches_normalization(parser):
    parser.read_string("[ClaimsMonitoring]\nCLAIMS_REVIEW_METRICS_INTERVAL_SEC = 50%\n")
    resolved = resolve_scheduler_config(config.load_settings())
    assert resolved.enabled and resolved.interval_seconds == 86400
