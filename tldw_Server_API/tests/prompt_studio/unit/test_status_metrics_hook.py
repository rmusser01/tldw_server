import pytest
from tldw_Server_API.app.core.Metrics.metrics_manager import get_metrics_registry
from tldw_Server_API.app.api.v1.endpoints.prompt_studio import prompt_studio_status


pytestmark = pytest.mark.unit


_STATUS_SENSITIVE_MARKERS = (
    "prompt studio jobs exploded",
    "metrics registry leaked",
    "per type gauge leaked",
    "/private/jobs.db",
    "/private/metrics.sock",
    "/private/gauges.sock",
)


class _LoggerStub:
    def __init__(self):
        self.errors: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self.debugs: list[tuple[str, tuple[object, ...], dict[str, object]]] = []

    def error(self, message: str, *args: object, **kwargs: object) -> None:
        self.errors.append((message, args, kwargs))

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        self.debugs.append((message, args, kwargs))


def _patch_status_success_dependencies(monkeypatch):
    class _JobsManagerStub:
        backend = "sqlite"

    monkeypatch.setattr(prompt_studio_status, "_get_jobs_manager", lambda: _JobsManagerStub())
    monkeypatch.setattr(prompt_studio_status, "_get_prompt_studio_queue", lambda: "default")
    monkeypatch.setattr(prompt_studio_status, "_get_by_status", lambda *args, **kwargs: {"queued": 1, "processing": 0})
    monkeypatch.setattr(
        prompt_studio_status,
        "_get_by_type_and_status",
        lambda *args, **kwargs: ({"optimization": 1}, {"optimization": 1}, {"optimization": 0}),
    )
    monkeypatch.setattr(prompt_studio_status, "_get_avg_processing_time_seconds", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(prompt_studio_status, "_get_success_rate", lambda *args, **kwargs: 100.0)
    monkeypatch.setattr(
        prompt_studio_status,
        "_get_lease_stats",
        lambda *args, **kwargs: {"active": 0, "expiring_soon": 0, "stale_processing": 0},
    )
    monkeypatch.setattr(prompt_studio_status.JobManager, "set_rls_context", lambda **kwargs: None)
    monkeypatch.setattr(prompt_studio_status.JobManager, "clear_rls_context", lambda: None)


def test_status_endpoint_sets_prometheus_gauges(prompt_studio_dual_backend_client):


    backend_label, client, db = prompt_studio_dual_backend_client

    r = client.get("/api/v1/prompt-studio/status")
    assert r.status_code == 200

    reg = get_metrics_registry()
    # Metrics should be registered and have at least one value after hitting status
    for name in (
        "prompt_studio_queue_depth",
        "prompt_studio_processing",
        "prompt_studio_leases_active",
        "prompt_studio_leases_expiring_soon",
        "prompt_studio_leases_stale_processing",
    ):
        stats = reg.get_metric_stats(name)
        assert isinstance(stats, dict)
        # stats can be empty if nothing recorded, but after endpoint call we expect 'latest'
        assert "latest" in stats


@pytest.mark.asyncio
async def test_status_sanitizes_backend_failure(monkeypatch):
    logger_stub = _LoggerStub()

    def _raise_backend_failure():
        raise RuntimeError("prompt studio jobs exploded at /private/jobs.db")

    monkeypatch.setattr(prompt_studio_status, "logger", logger_stub)
    monkeypatch.setattr(prompt_studio_status, "_get_jobs_manager", _raise_backend_failure)

    response = await prompt_studio_status.get_prompt_studio_status(
        warn_seconds=30,
        user_context={"user_id": "user-1", "is_admin": False},
    )

    assert response.success is False
    assert response.error == "Failed to compute Prompt Studio status"
    assert logger_stub.errors == [("Failed to compute Prompt Studio status", (), {})]
    rendered = " ".join([logger_stub.errors[0][0], *(str(arg) for arg in logger_stub.errors[0][1])])
    for marker in _STATUS_SENSITIVE_MARKERS:
        assert marker not in rendered


@pytest.mark.asyncio
async def test_status_sanitizes_metrics_registry_failure(monkeypatch):
    logger_stub = _LoggerStub()
    _patch_status_success_dependencies(monkeypatch)

    def _raise_metrics_failure():
        raise RuntimeError("metrics registry leaked /private/metrics.sock")

    monkeypatch.setattr(prompt_studio_status, "logger", logger_stub)
    monkeypatch.setattr(prompt_studio_status, "get_metrics_registry", _raise_metrics_failure)

    response = await prompt_studio_status.get_prompt_studio_status(
        warn_seconds=30,
        user_context={"user_id": "user-1", "is_admin": False},
    )

    assert response.success is True
    assert logger_stub.debugs == [("Failed to set Prompt Studio gauges", (), {})]
    rendered = repr(logger_stub.debugs)
    for marker in _STATUS_SENSITIVE_MARKERS:
        assert marker not in rendered


@pytest.mark.asyncio
async def test_status_sanitizes_per_type_gauge_refresh_failure(monkeypatch):
    logger_stub = _LoggerStub()
    _patch_status_success_dependencies(monkeypatch)

    class _RegistryStub:
        def set_gauge(self, *args, **kwargs):
            return None

    def _raise_per_type_failure(*args, **kwargs):
        raise RuntimeError("per type gauge leaked /private/gauges.sock")

    monkeypatch.setattr(prompt_studio_status, "logger", logger_stub)
    monkeypatch.setattr(prompt_studio_status, "get_metrics_registry", lambda: _RegistryStub())
    monkeypatch.setattr(
        prompt_studio_status.prompt_studio_metrics,
        "update_job_queue_size",
        _raise_per_type_failure,
    )

    response = await prompt_studio_status.get_prompt_studio_status(
        warn_seconds=30,
        user_context={"user_id": "user-1", "is_admin": False},
    )

    assert response.success is True
    assert logger_stub.debugs == [("Failed to refresh per-type gauges", (), {})]
    rendered = repr(logger_stub.debugs)
    for marker in _STATUS_SENSITIVE_MARKERS:
        assert marker not in rendered
