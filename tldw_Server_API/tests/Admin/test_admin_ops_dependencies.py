"""Tests for dependency health, uptime stats, health history recording,
and history pruning in admin_system_ops_service.

Tests exercise service functions directly (not HTTP) following the monkeypatch +
JSON store isolation pattern established in test_admin_ops_new_endpoints.py.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Shared fixture: isolate the JSON store to a temp directory
# ---------------------------------------------------------------------------


def _configure_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Any, Path]:
    """Redirect the system-ops JSON store to *tmp_path*."""
    from tldw_Server_API.app.services import admin_system_ops_service

    store_path = tmp_path / "system_ops.json"
    monkeypatch.setattr(admin_system_ops_service, "_STORE_PATH", store_path)
    return admin_system_ops_service, store_path


# ═══════════════════════════════════════════════════════════════════════════
# 1. GET /admin/dependencies — verify 5 components, status, latency
# ═══════════════════════════════════════════════════════════════════════════


class TestGetAllDependencies:
    """Tests for the GET /admin/dependencies endpoint logic."""

    @pytest.mark.asyncio
    async def test_returns_5_components_with_status_and_latency(self, monkeypatch, tmp_path):
        """Mocking all health checks should yield exactly 5 items with expected fields."""
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

        service, _ = _configure_store(monkeypatch, tmp_path)

        monkeypatch.setattr(
            "tldw_Server_API.app.api.v1.endpoints.admin.admin_ops._require_platform_admin",
            lambda _: None,
        )

        # Mock each individual dependency check function
        async def _healthy_authnz():
            return {"status": "healthy", "type": "sqlite", "pool_size": 5}

        async def _healthy_chacha():
            return {"status": "healthy", "cached_instances": 3, "init_failures": 0}

        async def _healthy_workflows():
            return {"status": "healthy", "queue_depth": 0}

        async def _healthy_embeddings():
            return {"status": "healthy", "providers_healthy": 2, "providers_total": 2}

        async def _healthy_metrics():
            return {"status": "healthy"}

        monkeypatch.setattr(admin_ops, "_check_authnz_database", _healthy_authnz)
        monkeypatch.setattr(admin_ops, "_check_chacha_notes", _healthy_chacha)
        monkeypatch.setattr(admin_ops, "_check_workflows_engine", _healthy_workflows)
        monkeypatch.setattr(admin_ops, "_check_embeddings_service", _healthy_embeddings)
        monkeypatch.setattr(admin_ops, "_check_metrics_registry", _healthy_metrics)

        mock_principal = type("P", (), {"user_id": 1, "email": "a@b.c", "username": "a"})()
        result = await admin_ops.get_all_dependencies(principal=mock_principal)

        assert "items" in result
        items = result["items"]
        assert len(items) == 5

        expected_names = {
            "AuthNZ Database",
            "ChaChaNotes",
            "Workflows Engine",
            "Embeddings Service",
            "Metrics Registry",
        }
        actual_names = {item["name"] for item in items}
        assert actual_names == expected_names

        for item in items:
            assert "status" in item
            assert "latency_ms" in item
            assert isinstance(item["latency_ms"], (int, float))

        assert "checked_at" in result

    @pytest.mark.asyncio
    async def test_degraded_dependency_reported_correctly(self, monkeypatch, tmp_path):
        """A degraded dependency check should surface its status and error."""
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

        _configure_store(monkeypatch, tmp_path)

        monkeypatch.setattr(
            "tldw_Server_API.app.api.v1.endpoints.admin.admin_ops._require_platform_admin",
            lambda _: None,
        )

        async def _healthy():
            return {"status": "healthy"}

        async def _degraded_embeddings():
            return {"status": "degraded", "error": "Provider timeout", "providers_healthy": 1, "providers_total": 3}

        monkeypatch.setattr(admin_ops, "_check_authnz_database", _healthy)
        monkeypatch.setattr(admin_ops, "_check_chacha_notes", _healthy)
        monkeypatch.setattr(admin_ops, "_check_workflows_engine", _healthy)
        monkeypatch.setattr(admin_ops, "_check_embeddings_service", _degraded_embeddings)
        monkeypatch.setattr(admin_ops, "_check_metrics_registry", _healthy)

        mock_principal = type("P", (), {"user_id": 1, "email": "a@b.c", "username": "a"})()
        result = await admin_ops.get_all_dependencies(principal=mock_principal)

        embeddings_item = next(i for i in result["items"] if i["name"] == "Embeddings Service")
        assert embeddings_item["status"] == "degraded"
        assert embeddings_item["error"] == "Provider timeout"

    @pytest.mark.asyncio
    async def test_check_dep_sanitizes_backend_failure(self):
        """Unexpected dependency probe failures should not expose backend details."""
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

        async def _raise_probe_error():
            raise RuntimeError("dependency backend exploded at /private/admin-deps.db")

        result = await admin_ops._check_dep("AuthNZ Database", _raise_probe_error)

        assert result["name"] == "AuthNZ Database"
        assert result["status"] == "degraded"
        assert result["error"] == "AuthNZ Database health check failed"
        assert result["metadata"] == {}

    @pytest.mark.asyncio
    async def test_chacha_health_sanitizes_snapshot_failure(self, monkeypatch):
        """ChaChaNotes probe failures should not expose DB paths or backend details."""
        from tldw_Server_API.app.api.v1.API_Deps import ChaCha_Notes_DB_Deps
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

        def _raise_snapshot_error():
            raise RuntimeError("chacha db exploded at /private/chacha.db")

        monkeypatch.setattr(ChaCha_Notes_DB_Deps, "get_chacha_health_snapshot", _raise_snapshot_error)

        result = await admin_ops._check_chacha_notes()

        assert result == {"status": "unhealthy", "error": "ChaChaNotes health check failed"}

    @pytest.mark.asyncio
    async def test_workflows_health_sanitizes_engine_failure(self, monkeypatch):
        """Workflow probe failures should not expose scheduler backend details."""
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops
        from tldw_Server_API.app.core.Workflows import engine

        def _raise_instance_error():
            raise RuntimeError("workflow engine exploded at /private/workflows.db")

        monkeypatch.setattr(engine.WorkflowScheduler, "instance", staticmethod(_raise_instance_error))

        result = await admin_ops._check_workflows_engine()

        assert result == {"status": "degraded", "error": "Workflows engine health check failed"}

    @pytest.mark.asyncio
    async def test_embeddings_health_sanitizes_service_failure(self, monkeypatch):
        """Embedding probe failures should not expose provider/backend details."""
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops
        from tldw_Server_API.app.core.Embeddings import async_embeddings

        async def _raise_embedding_service_error():
            raise RuntimeError("embedding service exploded at /private/embeddings.db")

        monkeypatch.setattr(async_embeddings, "get_embedding_service", _raise_embedding_service_error, raising=False)

        result = await admin_ops._check_embeddings_service()

        assert result == {"status": "degraded", "error": "Embeddings service health check failed"}

    @pytest.mark.asyncio
    async def test_metrics_health_sanitizes_registry_failure(self, monkeypatch):
        """Metrics probe failures should not expose registry backend details."""
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops
        from tldw_Server_API.app.core.Metrics import metrics_manager

        def _raise_metrics_error():
            raise RuntimeError("metrics registry exploded at /private/metrics.db")

        monkeypatch.setattr(metrics_manager, "get_metrics_registry", _raise_metrics_error)

        result = await admin_ops._check_metrics_registry()

        assert result == {"status": "degraded", "error": "Metrics registry health check failed"}


# ═══════════════════════════════════════════════════════════════════════════
# 2. GET /admin/dependencies/{name}/uptime — uptime % math + sparkline
# ═══════════════════════════════════════════════════════════════════════════


class TestDependencyUptime:
    """Tests for get_uptime_stats service function."""

    def test_uptime_percentage_all_healthy(self, monkeypatch, tmp_path):
        """All healthy checks yield 100% uptime."""
        service, store_path = _configure_store(monkeypatch, tmp_path)

        now = datetime.now(timezone.utc)
        store_data = service._default_store()

        # Seed 10 healthy checks over 10 hours
        entries = []
        for i in range(10):
            entries.append({
                "dependency_name": "AuthNZ Database",
                "status": "healthy",
                "latency_ms": 5.0 + i,
                "checked_at": (now - timedelta(hours=i)).isoformat(),
            })
        store_data["dependency_health_history"] = entries
        store_path.write_text(json.dumps(store_data), encoding="utf-8")

        stats = service.get_uptime_stats("AuthNZ Database", days=30)

        assert stats["dependency_name"] == "AuthNZ Database"
        assert stats["total_checks"] == 10
        assert stats["healthy_checks"] == 10
        assert stats["uptime_pct"] == 100.0
        assert stats["downtime_minutes"] == 0

    def test_uptime_percentage_mixed_status(self, monkeypatch, tmp_path):
        """Mix of healthy and unhealthy checks computes correct uptime %."""
        service, store_path = _configure_store(monkeypatch, tmp_path)

        now = datetime.now(timezone.utc)
        store_data = service._default_store()

        # 7 healthy + 3 unhealthy = 70% uptime
        entries = []
        for i in range(10):
            entries.append({
                "dependency_name": "ChaChaNotes",
                "status": "healthy" if i < 7 else "unhealthy",
                "latency_ms": 10.0,
                "checked_at": (now - timedelta(hours=i)).isoformat(),
            })
        store_data["dependency_health_history"] = entries
        store_path.write_text(json.dumps(store_data), encoding="utf-8")

        stats = service.get_uptime_stats("ChaChaNotes", days=30)

        assert stats["total_checks"] == 10
        assert stats["healthy_checks"] == 7
        assert stats["uptime_pct"] == 70.0
        assert stats["downtime_minutes"] == 3 * 60  # 3 unhealthy * 60 min each

    def test_uptime_no_checks_defaults_to_100(self, monkeypatch, tmp_path):
        """No health checks at all defaults to 100% uptime."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        stats = service.get_uptime_stats("NonExistentDep", days=30)

        assert stats["total_checks"] == 0
        assert stats["healthy_checks"] == 0
        assert stats["uptime_pct"] == 100.0

    def test_uptime_sparkline_has_correct_length(self, monkeypatch, tmp_path):
        """Sparkline should have 7*24 = 168 entries for a 30-day window (capped at 7 days)."""
        service, store_path = _configure_store(monkeypatch, tmp_path)

        now = datetime.now(timezone.utc)
        store_data = service._default_store()

        entries = []
        for i in range(48):
            entries.append({
                "dependency_name": "TestDep",
                "status": "healthy",
                "latency_ms": 5.0,
                "checked_at": (now - timedelta(hours=i)).isoformat(),
            })
        store_data["dependency_health_history"] = entries
        store_path.write_text(json.dumps(store_data), encoding="utf-8")

        stats = service.get_uptime_stats("TestDep", days=30)

        # sparkline_hours = min(30, 7) * 24 = 168
        assert len(stats["sparkline"]) == 168

    def test_uptime_sparkline_short_window(self, monkeypatch, tmp_path):
        """For a 3-day window, sparkline has 3*24 = 72 entries."""
        service, store_path = _configure_store(monkeypatch, tmp_path)

        now = datetime.now(timezone.utc)
        store_data = service._default_store()

        entries = []
        for i in range(24):
            entries.append({
                "dependency_name": "ShortDep",
                "status": "healthy",
                "latency_ms": 3.0,
                "checked_at": (now - timedelta(hours=i)).isoformat(),
            })
        store_data["dependency_health_history"] = entries
        store_path.write_text(json.dumps(store_data), encoding="utf-8")

        stats = service.get_uptime_stats("ShortDep", days=3)

        assert len(stats["sparkline"]) == 72

    def test_uptime_sparkline_unhealthy_slot(self, monkeypatch, tmp_path):
        """An unhealthy check in a sparkline slot sets that slot to 0."""
        service, store_path = _configure_store(monkeypatch, tmp_path)

        now = datetime.now(timezone.utc)
        store_data = service._default_store()

        entries = [
            {
                "dependency_name": "SparkDep",
                "status": "unhealthy",
                "latency_ms": 100.0,
                # Place it 1 hour ago so it falls in the sparkline range
                "checked_at": (now - timedelta(hours=1)).isoformat(),
            },
            {
                "dependency_name": "SparkDep",
                "status": "healthy",
                "latency_ms": 5.0,
                "checked_at": (now - timedelta(hours=2)).isoformat(),
            },
        ]
        store_data["dependency_health_history"] = entries
        store_path.write_text(json.dumps(store_data), encoding="utf-8")

        stats = service.get_uptime_stats("SparkDep", days=1)

        sparkline = stats["sparkline"]
        # There should be a 0 somewhere in the sparkline (the unhealthy hour)
        assert 0 in sparkline

    def test_uptime_avg_latency(self, monkeypatch, tmp_path):
        """Average latency is correctly computed from latency_ms values."""
        service, store_path = _configure_store(monkeypatch, tmp_path)

        now = datetime.now(timezone.utc)
        store_data = service._default_store()

        entries = [
            {
                "dependency_name": "LatencyDep",
                "status": "healthy",
                "latency_ms": 10.0,
                "checked_at": (now - timedelta(hours=1)).isoformat(),
            },
            {
                "dependency_name": "LatencyDep",
                "status": "healthy",
                "latency_ms": 20.0,
                "checked_at": (now - timedelta(hours=2)).isoformat(),
            },
            {
                "dependency_name": "LatencyDep",
                "status": "healthy",
                "latency_ms": 30.0,
                "checked_at": (now - timedelta(hours=3)).isoformat(),
            },
        ]
        store_data["dependency_health_history"] = entries
        store_path.write_text(json.dumps(store_data), encoding="utf-8")

        stats = service.get_uptime_stats("LatencyDep", days=30)

        assert stats["avg_latency_ms"] == 20.0

    def test_uptime_days_clamped(self, monkeypatch, tmp_path):
        """Days parameter is clamped: <1 becomes 1, >90 becomes 90."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        stats_low = service.get_uptime_stats("AnyDep", days=0)
        assert stats_low["days"] == 1

        stats_high = service.get_uptime_stats("AnyDep", days=200)
        assert stats_high["days"] == 90


# ═══════════════════════════════════════════════════════════════════════════
# 3. Health History Recording — hourly deduplication
# ═══════════════════════════════════════════════════════════════════════════


class TestHealthHistoryRecording:
    """Tests for record_health_snapshot including deduplication."""

    def test_record_health_snapshot_basic(self, monkeypatch, tmp_path):
        """Recording a snapshot with multiple deps stores all entries."""
        service, store_path = _configure_store(monkeypatch, tmp_path)

        results = [
            {"name": "AuthNZ Database", "status": "healthy", "latency_ms": 5.0},
            {"name": "ChaChaNotes", "status": "degraded", "latency_ms": 50.0},
            {"name": "Workflows Engine", "status": "healthy", "latency_ms": 3.0},
        ]

        recorded = service.record_health_snapshot(results)

        assert recorded == 3

        # Verify they are in the store
        store_data = json.loads(store_path.read_text(encoding="utf-8"))
        history = store_data["dependency_health_history"]
        assert len(history) == 3

        names = {e["dependency_name"] for e in history}
        assert names == {"AuthNZ Database", "ChaChaNotes", "Workflows Engine"}

    def test_dedup_within_hourly_window(self, monkeypatch, tmp_path):
        """Second snapshot within 1 hour is deduplicated (not recorded)."""
        service, store_path = _configure_store(monkeypatch, tmp_path)

        results = [
            {"name": "AuthNZ Database", "status": "healthy", "latency_ms": 5.0},
        ]

        first_count = service.record_health_snapshot(results)
        assert first_count == 1

        # Call again immediately (within the same hour)
        second_count = service.record_health_snapshot(results)
        assert second_count == 0

        # Store should still have only 1 entry
        store_data = json.loads(store_path.read_text(encoding="utf-8"))
        history = store_data["dependency_health_history"]
        assert len(history) == 1

    def test_dedup_allows_after_hourly_window(self, monkeypatch, tmp_path):
        """A snapshot recorded >1 hour after the last is NOT deduplicated."""
        service, store_path = _configure_store(monkeypatch, tmp_path)

        now = datetime.now(timezone.utc)

        # Seed an entry from 2 hours ago
        store_data = service._default_store()
        store_data["dependency_health_history"] = [
            {
                "dependency_name": "AuthNZ Database",
                "status": "healthy",
                "latency_ms": 5.0,
                "checked_at": (now - timedelta(hours=2)).isoformat(),
            },
        ]
        store_path.write_text(json.dumps(store_data), encoding="utf-8")

        # Now record a new snapshot (should not be deduped since >1h passed)
        results = [
            {"name": "AuthNZ Database", "status": "healthy", "latency_ms": 6.0},
        ]
        recorded = service.record_health_snapshot(results)

        assert recorded == 1

        store_data = json.loads(store_path.read_text(encoding="utf-8"))
        history = store_data["dependency_health_history"]
        assert len(history) == 2

    def test_dedup_per_dependency(self, monkeypatch, tmp_path):
        """Deduplication is per-dependency, not global."""
        service, store_path = _configure_store(monkeypatch, tmp_path)

        # Record for dep A
        service.record_health_snapshot([
            {"name": "DepA", "status": "healthy", "latency_ms": 5.0},
        ])

        # Recording for dep B should still work even immediately after
        recorded = service.record_health_snapshot([
            {"name": "DepB", "status": "healthy", "latency_ms": 3.0},
        ])
        assert recorded == 1

        # But dep A again should be deduped
        recorded_a_again = service.record_health_snapshot([
            {"name": "DepA", "status": "healthy", "latency_ms": 5.0},
        ])
        assert recorded_a_again == 0

    def test_empty_name_skipped(self, monkeypatch, tmp_path):
        """Items with blank or empty name are skipped."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        recorded = service.record_health_snapshot([
            {"name": "", "status": "healthy", "latency_ms": 5.0},
            {"name": "  ", "status": "healthy", "latency_ms": 5.0},
        ])

        assert recorded == 0


# ═══════════════════════════════════════════════════════════════════════════
# 4. History Pruning — entries > 90 days are pruned
# ═══════════════════════════════════════════════════════════════════════════


class TestHealthHistoryPruning:
    """Tests for _prune_health_history logic."""

    def test_prune_removes_old_entries(self, monkeypatch, tmp_path):
        """Entries older than 90 days should be removed after recording."""
        service, store_path = _configure_store(monkeypatch, tmp_path)

        now = datetime.now(timezone.utc)

        store_data = service._default_store()
        store_data["dependency_health_history"] = [
            {
                "dependency_name": "OldDep",
                "status": "healthy",
                "latency_ms": 5.0,
                "checked_at": (now - timedelta(days=100)).isoformat(),
            },
            {
                "dependency_name": "OldDep",
                "status": "unhealthy",
                "latency_ms": 50.0,
                "checked_at": (now - timedelta(days=95)).isoformat(),
            },
            {
                "dependency_name": "RecentDep",
                "status": "healthy",
                "latency_ms": 3.0,
                "checked_at": (now - timedelta(days=10)).isoformat(),
            },
        ]
        store_path.write_text(json.dumps(store_data), encoding="utf-8")

        # Trigger pruning by recording a new snapshot
        service.record_health_snapshot([
            {"name": "NewDep", "status": "healthy", "latency_ms": 2.0},
        ])

        store_data = json.loads(store_path.read_text(encoding="utf-8"))
        history = store_data["dependency_health_history"]

        dep_names = [e["dependency_name"] for e in history]
        assert "OldDep" not in dep_names
        assert "RecentDep" in dep_names
        assert "NewDep" in dep_names

    def test_prune_keeps_entries_within_90_days(self, monkeypatch, tmp_path):
        """Entries at exactly 89 days old should be kept."""
        service, store_path = _configure_store(monkeypatch, tmp_path)

        now = datetime.now(timezone.utc)

        store_data = service._default_store()
        store_data["dependency_health_history"] = [
            {
                "dependency_name": "BorderDep",
                "status": "healthy",
                "latency_ms": 5.0,
                "checked_at": (now - timedelta(days=89)).isoformat(),
            },
        ]
        store_path.write_text(json.dumps(store_data), encoding="utf-8")

        # Trigger pruning
        service.record_health_snapshot([
            {"name": "TriggerDep", "status": "healthy", "latency_ms": 1.0},
        ])

        store_data = json.loads(store_path.read_text(encoding="utf-8"))
        history = store_data["dependency_health_history"]
        dep_names = [e["dependency_name"] for e in history]
        assert "BorderDep" in dep_names

    def test_prune_function_directly(self, monkeypatch, tmp_path):
        """Test _prune_health_history as a standalone function."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        now = datetime.now(timezone.utc)

        entries = [
            {
                "dependency_name": "Expired1",
                "status": "healthy",
                "latency_ms": 5.0,
                "checked_at": (now - timedelta(days=91)).isoformat(),
            },
            {
                "dependency_name": "Expired2",
                "status": "healthy",
                "latency_ms": 5.0,
                "checked_at": (now - timedelta(days=120)).isoformat(),
            },
            {
                "dependency_name": "Fresh",
                "status": "healthy",
                "latency_ms": 5.0,
                "checked_at": (now - timedelta(days=1)).isoformat(),
            },
        ]

        pruned = service._prune_health_history(entries)

        dep_names = [e["dependency_name"] for e in pruned]
        assert "Expired1" not in dep_names
        assert "Expired2" not in dep_names
        assert "Fresh" in dep_names

    def test_prune_caps_per_dependency(self, monkeypatch, tmp_path):
        """When a single dependency exceeds _HEALTH_HISTORY_MAX_PER_DEPENDENCY, oldest are pruned."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        now = datetime.now(timezone.utc)
        max_per_dep = service._HEALTH_HISTORY_MAX_PER_DEPENDENCY

        # Create max + 5 entries for one dependency
        entries = []
        for i in range(max_per_dep + 5):
            entries.append({
                "dependency_name": "BulkDep",
                "status": "healthy",
                "latency_ms": 5.0,
                "checked_at": (now - timedelta(hours=i)).isoformat(),
            })

        pruned = service._prune_health_history(entries)

        bulk_entries = [e for e in pruned if e["dependency_name"] == "BulkDep"]
        assert len(bulk_entries) <= max_per_dep
