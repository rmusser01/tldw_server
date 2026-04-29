from __future__ import annotations

from loguru import logger

import pytest

pytestmark = pytest.mark.unit

_LEAK = "admin backend exploded at /tmp/admin-secret-token"


def _assert_safe_log(rendered: str) -> None:
    assert "admin backend exploded" not in rendered
    assert "/tmp/admin-secret-token" not in rendered
    assert "exc_info" not in rendered


def _capture_logs():
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)), format="{message} {extra}")
    return records, sink_id


def test_bundle_size_estimate_log_omits_raw_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.services import admin_bundle_service as bundle_service

    def _fail_resolve_dataset_path(*_args, **_kwargs):
        raise RuntimeError(_LEAK)

    monkeypatch.setattr(bundle_service, "_resolve_dataset_db_path", _fail_resolve_dataset_path)

    records, sink_id = _capture_logs()
    try:
        assert bundle_service._estimate_total_db_size(["authnz"], user_id=None) == 1024
    finally:
        logger.remove(sink_id)

    _assert_safe_log("\n".join(records))


def test_retention_floor_log_omits_raw_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.services import admin_data_ops_service as data_ops_service

    class _FailingSettings:
        def __getattr__(self, _name):
            raise RuntimeError(_LEAK)

    monkeypatch.setattr(data_ops_service, "get_settings", lambda: _FailingSettings())

    records, sink_id = _capture_logs()
    try:
        assert data_ops_service._effective_retention_days("privilege_snapshots_weekly", 7) == 7
    finally:
        logger.remove(sink_id)

    _assert_safe_log("\n".join(records))


def test_bulk_confirm_threshold_log_omits_raw_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.services import admin_profiles_service as profiles_service

    monkeypatch.delenv("BULK_UPDATE_CONFIRM_THRESHOLD", raising=False)

    def _fail_load_config():
        raise RuntimeError(_LEAK)

    monkeypatch.setattr(profiles_service, "load_comprehensive_config", _fail_load_config)

    records, sink_id = _capture_logs()
    try:
        assert profiles_service._get_bulk_confirm_threshold() == 1000
    finally:
        logger.remove(sink_id)

    _assert_safe_log("\n".join(records))


def test_bulk_candidate_user_id_log_omits_raw_exception() -> None:
    from tldw_Server_API.app.services import admin_profiles_service as profiles_service

    class _ExplodingUser:
        def get(self, _key):
            raise RuntimeError(_LEAK)

    records, sink_id = _capture_logs()
    try:
        assert profiles_service._coerce_bulk_candidate_user_id(_ExplodingUser()) is None
    finally:
        logger.remove(sink_id)

    _assert_safe_log("\n".join(records))
