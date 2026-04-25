import pytest


pytestmark = pytest.mark.unit


def test_should_start_inprocess_worker_uses_route_policy_when_flag_unset(monkeypatch):
    from tldw_Server_API.app.services.worker_startup_policy import should_start_inprocess_worker

    monkeypatch.delenv("MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED", raising=False)
    monkeypatch.setenv("ROUTES_ENABLE", "media-ingest-heavy-jobs")

    assert should_start_inprocess_worker(
        "MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED",
        "media-ingest-heavy-jobs",
        sidecar_mode=False,
        default_stable=False,
        test_mode=False,
    )


def test_should_start_inprocess_worker_respects_explicit_enable_in_test_mode(monkeypatch):
    from tldw_Server_API.app.services.worker_startup_policy import should_start_inprocess_worker

    monkeypatch.setenv("MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED", "true")
    monkeypatch.delenv("ROUTES_ENABLE", raising=False)

    assert should_start_inprocess_worker(
        "MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED",
        "media-ingest-heavy-jobs",
        sidecar_mode=False,
        default_stable=False,
        test_mode=True,
    )


def test_should_start_inprocess_worker_disables_startup_in_sidecar_mode(monkeypatch):
    from tldw_Server_API.app.services.worker_startup_policy import should_start_inprocess_worker

    monkeypatch.setenv("MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED", "true")

    assert not should_start_inprocess_worker(
        "MEDIA_INGEST_HEAVY_JOBS_WORKER_ENABLED",
        "media-ingest-heavy-jobs",
        sidecar_mode=True,
        default_stable=False,
        test_mode=False,
    )
