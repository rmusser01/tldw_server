import sys
from types import ModuleType

import pytest

from tldw_Server_API.app.core.Embeddings import health_checks
from tldw_Server_API.app.core.Embeddings.health_checks import HealthChecker, HealthStatus


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_run_check_sanitizes_component_checker_failures():
    checker = HealthChecker.__new__(HealthChecker)

    async def failing_checker():
        raise RuntimeError("redis socket failed at /private/redis.sock")

    health = await checker._run_check("cache", failing_checker)

    assert health.status is HealthStatus.UNHEALTHY
    assert health.message == "Check failed"
    assert "redis socket failed" not in health.message
    assert "/private/redis.sock" not in health.message


@pytest.mark.asyncio
async def test_check_health_sanitizes_unexpected_gather_failures():
    checker = HealthChecker.__new__(HealthChecker)
    checker.checkers = {"cache": lambda: (_ for _ in ()).throw(NameError("token from /private/cache"))}
    checker.health_history = []
    checker.max_history = 100

    report = await checker.check_health()

    component = report["components"]["cache"]
    assert component["status"] == HealthStatus.UNHEALTHY.value
    assert component["message"] == "Check failed"
    assert "token" not in component["message"]
    assert "/private/cache" not in component["message"]


@pytest.mark.asyncio
async def test_system_health_sanitizes_resource_check_failures(monkeypatch):
    checker = HealthChecker.__new__(HealthChecker)

    def fail_cpu_percent(interval):
        raise RuntimeError("psutil failed for /private/proc")

    monkeypatch.setattr(health_checks.psutil, "cpu_percent", fail_cpu_percent)

    health = await checker._check_system_health()

    assert health.status is HealthStatus.UNHEALTHY
    assert health.message == "System check failed"
    assert "/private/proc" not in health.message


@pytest.mark.asyncio
async def test_provider_health_sanitizes_provider_status_failures(monkeypatch):
    checker = HealthChecker.__new__(HealthChecker)
    fake_module = ModuleType("async_embeddings")

    def fail_get_service():
        raise RuntimeError("provider token leaked at /private/provider")

    fake_module.get_async_embedding_service = fail_get_service
    monkeypatch.setitem(
        sys.modules,
        "tldw_Server_API.app.core.Embeddings.async_embeddings",
        fake_module,
    )

    health = await checker._check_provider_health()

    assert health.status is HealthStatus.UNHEALTHY
    assert health.message == "Provider check failed"
    assert "provider token" not in health.message
    assert "/private/provider" not in health.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("method_name", "dependency_name", "expected_status", "expected_message"),
    [
        ("_check_cache_health", "get_multi_tier_cache", HealthStatus.DEGRADED, "Cache check failed"),
        ("_check_rate_limiter_health", "get_rate_limiter", HealthStatus.DEGRADED, "Rate limiter check failed"),
        ("_check_dlq_health", "get_recovery_manager", HealthStatus.DEGRADED, "DLQ check failed"),
        ("_check_connection_pool_health", "get_pool_manager", HealthStatus.DEGRADED, "Connection pool check failed"),
        ("_check_config_health", "get_config", HealthStatus.UNHEALTHY, "Config check failed"),
    ],
)
async def test_component_health_sanitizes_dependency_failures(
    monkeypatch,
    method_name,
    dependency_name,
    expected_status,
    expected_message,
):
    checker = HealthChecker.__new__(HealthChecker)

    def fail_dependency():
        raise RuntimeError("dependency failed at /private/dependency")

    monkeypatch.setattr(health_checks, dependency_name, fail_dependency)

    health = await getattr(checker, method_name)()

    assert health.status is expected_status
    assert health.message == expected_message
    assert "dependency failed" not in health.message
    assert "/private/dependency" not in health.message


@pytest.mark.asyncio
async def test_database_health_sanitizes_chromadb_failures(monkeypatch):
    checker = HealthChecker.__new__(HealthChecker)
    checker.thresholds = {"db_latency_ms": 100}
    chromadb_module = ModuleType("chromadb")
    chromadb_config_module = ModuleType("chromadb.config")

    class Settings:
        def __init__(self, anonymized_telemetry):
            self.anonymized_telemetry = anonymized_telemetry

    def fail_persistent_client(path, settings):
        raise RuntimeError("chromadb path failed at /private/chroma")

    chromadb_module.PersistentClient = fail_persistent_client
    chromadb_config_module.Settings = Settings
    monkeypatch.setitem(sys.modules, "chromadb", chromadb_module)
    monkeypatch.setitem(sys.modules, "chromadb.config", chromadb_config_module)

    health = await checker._check_database_health()

    assert health.status is HealthStatus.UNHEALTHY
    assert health.message == "Database check failed"
    assert "chromadb path" not in health.message
    assert "/private/chroma" not in health.message
