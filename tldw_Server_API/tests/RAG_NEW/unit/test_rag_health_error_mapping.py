import pytest

from tldw_Server_API.app.api.v1.endpoints import rag_health


pytestmark = pytest.mark.unit


class _HealthyCoordinator:
    circuit_breakers = {}


class _HealthyCache:
    def get_stats(self):
        return {"hit_rate": 1.0, "size": 1}


class _HealthyMetrics:
    def get_current_metrics(self):
        return {"recent_queries": 1}


class _HealthyBatchProcessor:
    active_jobs = []

    def get_statistics(self):
        return {"job_success_rate": 1.0}


class _LoggerStub:
    def __init__(self):
        self.errors = []

    def error(self, message, *args, **kwargs):
        self.errors.append(str(message))


def _assert_sanitized_error_log(logger_stub, expected_message):
    assert logger_stub.errors == [expected_message]
    assert "/private/" not in logger_stub.errors[0]
    assert "exploded" not in logger_stub.errors[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("component", "safe_error", "safe_log"),
    [
        ("cache", "RAG cache health check failed", "Cache health check failed"),
        ("metrics", "RAG metrics health check failed", "Metrics health check failed"),
        ("batch_processor", "RAG batch processor health check failed", "Batch processor health check failed"),
    ],
)
async def test_health_check_sanitizes_component_failures(monkeypatch, component, safe_error, safe_log):
    logger_stub = _LoggerStub()

    monkeypatch.setattr(rag_health, "get_coordinator", lambda: _HealthyCoordinator())
    monkeypatch.setattr(rag_health, "get_rag_cache", lambda: _HealthyCache())
    monkeypatch.setattr(rag_health, "get_metrics_collector", lambda: _HealthyMetrics())
    monkeypatch.setattr(rag_health, "get_batch_processor", lambda: _HealthyBatchProcessor())
    monkeypatch.setattr(rag_health, "logger", logger_stub)

    def _raise_component_failure():
        raise RuntimeError(f"{component} exploded at /private/rag.db")

    if component == "cache":
        monkeypatch.setattr(rag_health, "get_rag_cache", _raise_component_failure)
    elif component == "metrics":
        monkeypatch.setattr(rag_health, "get_metrics_collector", _raise_component_failure)
    else:
        monkeypatch.setattr(rag_health, "get_batch_processor", _raise_component_failure)

    response = await rag_health.health_check()

    assert response["status"] == "unhealthy"
    assert response["components"][component]["status"] == "unhealthy"
    assert response["components"][component]["error"] == safe_error
    _assert_sanitized_error_log(logger_stub, safe_log)


@pytest.mark.asyncio
async def test_health_check_sanitizes_outer_failure_log(monkeypatch):
    logger_stub = _LoggerStub()

    def _raise_coordinator_failure():
        raise RuntimeError("coordinator exploded at /private/rag.db")

    monkeypatch.setattr(rag_health, "logger", logger_stub)
    monkeypatch.setattr(rag_health, "get_coordinator", _raise_coordinator_failure)

    response = await rag_health.health_check()

    assert response["status"] == "unhealthy"
    assert response["error"] == "Error occured during RAG health check"
    _assert_sanitized_error_log(logger_stub, "Health check error")
