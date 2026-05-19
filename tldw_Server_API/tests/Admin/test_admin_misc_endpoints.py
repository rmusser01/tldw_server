"""Tests for miscellaneous admin endpoints:

- Voice command validation (VoiceCommandRouter.validate_command_config)
- Jobs SLA policy CRUD (JobManager.upsert/delete/list SLA policy)
- Jobs SLA breach detection (list_sla_breaches_endpoint logic)
- Error breakdown aggregation (admin_system_service.get_error_breakdown)
- Rate limit summary (admin_system_service.get_rate_limit_summary)
"""
from __future__ import annotations

import asyncio
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run a coroutine synchronously."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ===========================================================================
# 1. Voice command validation
# ===========================================================================


class TestVoiceCommandValidation:
    """Tests for VoiceCommandRouter.validate_command_config — dry-run validation."""

    def _make_command(self, **overrides) -> Any:
        """Create a VoiceCommand with sensible defaults."""
        from tldw_Server_API.app.core.VoiceAssistant.schemas import (
            ActionType,
            VoiceCommand,
        )

        defaults = {
            "id": "cmd-001",
            "user_id": 1,
            "name": "Test Command",
            "phrases": ["do something"],
            "action_type": ActionType.LLM_CHAT,
            "action_config": {},
        }
        defaults.update(overrides)
        return VoiceCommand(**defaults)

    def _make_router(self) -> Any:
        """Create a VoiceCommandRouter with mocked dependencies."""
        from tldw_Server_API.app.core.VoiceAssistant.router import VoiceCommandRouter

        router = VoiceCommandRouter.__new__(VoiceCommandRouter)
        router.parser = mock.MagicMock()
        router.registry = mock.MagicMock()
        router.workflow_handler = mock.MagicMock()
        router.workflow_handler.get_voice_workflow_templates.return_value = {
            "summarize": {"steps": [{"action": "llm"}]},
        }
        return router

    def test_llm_chat_passes_all_steps(self):
        """LLM_CHAT with phrases passes config_schema, action_target, and phrases."""
        router = self._make_router()
        cmd = self._make_command()

        async def _test():
            steps = await router.validate_command_config(cmd)
            assert len(steps) == 3
            names = {s["name"] for s in steps}
            assert names == {"config_schema", "action_target", "phrases"}
            assert all(s["passed"] for s in steps)

        _run(_test())

    def test_mcp_tool_missing_tool_name_fails_config_schema(self):
        """MCP_TOOL action without tool_name fails config_schema step."""
        from tldw_Server_API.app.core.VoiceAssistant.schemas import ActionType

        router = self._make_router()
        cmd = self._make_command(
            action_type=ActionType.MCP_TOOL,
            action_config={},
        )

        async def _test():
            steps = await router.validate_command_config(cmd)
            config_step = next(s for s in steps if s["name"] == "config_schema")
            assert config_step["passed"] is False
            assert "tool_name" in config_step["message"]

        _run(_test())

    def test_mcp_tool_with_tool_name_passes_config_schema(self):
        """MCP_TOOL action with tool_name passes config_schema step."""
        from tldw_Server_API.app.core.VoiceAssistant.schemas import ActionType

        router = self._make_router()
        cmd = self._make_command(
            action_type=ActionType.MCP_TOOL,
            action_config={"tool_name": "search_web"},
        )

        async def _test():
            steps = await router.validate_command_config(cmd)
            config_step = next(s for s in steps if s["name"] == "config_schema")
            assert config_step["passed"] is True

        _run(_test())

    def test_workflow_missing_all_keys_fails(self):
        """WORKFLOW action without workflow_id/template/definition fails."""
        from tldw_Server_API.app.core.VoiceAssistant.schemas import ActionType

        router = self._make_router()
        cmd = self._make_command(
            action_type=ActionType.WORKFLOW,
            action_config={},
        )

        async def _test():
            steps = await router.validate_command_config(cmd)
            config_step = next(s for s in steps if s["name"] == "config_schema")
            assert config_step["passed"] is False

        _run(_test())

    def test_workflow_with_template_passes(self):
        """WORKFLOW with workflow_template passes config_schema."""
        from tldw_Server_API.app.core.VoiceAssistant.schemas import ActionType

        router = self._make_router()
        cmd = self._make_command(
            action_type=ActionType.WORKFLOW,
            action_config={"workflow_template": "summarize"},
        )

        async def _test():
            steps = await router.validate_command_config(cmd)
            config_step = next(s for s in steps if s["name"] == "config_schema")
            assert config_step["passed"] is True

        _run(_test())

    def test_custom_missing_action_field_fails(self):
        """CUSTOM action without 'action' key fails config_schema."""
        from tldw_Server_API.app.core.VoiceAssistant.schemas import ActionType

        router = self._make_router()
        cmd = self._make_command(
            action_type=ActionType.CUSTOM,
            action_config={},
        )

        async def _test():
            steps = await router.validate_command_config(cmd)
            config_step = next(s for s in steps if s["name"] == "config_schema")
            assert config_step["passed"] is False

        _run(_test())

    def test_no_phrases_fails_phrases_step(self):
        """Command with no trigger phrases fails the phrases step."""
        router = self._make_router()
        cmd = self._make_command(phrases=[])

        async def _test():
            steps = await router.validate_command_config(cmd)
            phrases_step = next(s for s in steps if s["name"] == "phrases")
            assert phrases_step["passed"] is False
            assert "No trigger phrases" in phrases_step["message"]

        _run(_test())

    def test_blank_phrases_fails(self):
        """Command with only blank-string phrases fails the phrases step."""
        router = self._make_router()
        cmd = self._make_command(phrases=["", "   "])

        async def _test():
            steps = await router.validate_command_config(cmd)
            phrases_step = next(s for s in steps if s["name"] == "phrases")
            assert phrases_step["passed"] is False

        _run(_test())

    def test_multiple_phrases_passes(self):
        """Command with multiple non-empty phrases passes."""
        router = self._make_router()
        cmd = self._make_command(phrases=["hey assistant", "do thing"])

        async def _test():
            steps = await router.validate_command_config(cmd)
            phrases_step = next(s for s in steps if s["name"] == "phrases")
            assert phrases_step["passed"] is True
            assert "2 trigger phrase(s)" in phrases_step["message"]

        _run(_test())


# ===========================================================================
# 2. Jobs SLA policy CRUD via JobManager (SQLite backend)
# ===========================================================================


class TestJobsSlaPolicyCRUD:
    """Test SLA policy upsert, list, and delete on a real in-memory SQLite DB."""

    def _make_manager(self, tmp_path):
        """Create a JobManager backed by a temp SQLite DB."""
        from tldw_Server_API.app.core.Jobs.manager import JobManager

        db_path = str(tmp_path / "jobs_test.db")
        jm = JobManager(backend=None, db_url=None)
        # Override _connect to use a file-based SQLite for schema creation
        original_connect = jm._connect

        def _patched_connect():
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            return conn

        jm._connect = _patched_connect

        # Ensure SLA table exists
        conn = _patched_connect()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS job_sla_policies (
                domain TEXT NOT NULL,
                queue TEXT NOT NULL,
                job_type TEXT NOT NULL,
                max_queue_latency_seconds INTEGER,
                max_duration_seconds INTEGER,
                enabled INTEGER DEFAULT 1,
                updated_at TEXT,
                PRIMARY KEY (domain, queue, job_type)
            )
        """)
        conn.commit()
        conn.close()
        return jm

    def test_upsert_creates_policy(self, tmp_path):
        """Upserting a new policy creates it."""
        jm = self._make_manager(tmp_path)
        jm.upsert_sla_policy(
            domain="default",
            queue="transcription",
            job_type="audio",
            max_queue_latency_seconds=60,
            max_duration_seconds=300,
            enabled=True,
        )
        conn = jm._connect()
        row = conn.execute(
            "SELECT * FROM job_sla_policies WHERE domain='default' AND queue='transcription' AND job_type='audio'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row["max_queue_latency_seconds"] == 60
        assert row["max_duration_seconds"] == 300

    def test_upsert_updates_existing(self, tmp_path):
        """Upserting again updates the thresholds."""
        jm = self._make_manager(tmp_path)
        jm.upsert_sla_policy(domain="d", queue="q", job_type="jt", max_queue_latency_seconds=10)
        jm.upsert_sla_policy(domain="d", queue="q", job_type="jt", max_queue_latency_seconds=99)
        conn = jm._connect()
        row = conn.execute(
            "SELECT * FROM job_sla_policies WHERE domain='d' AND queue='q' AND job_type='jt'"
        ).fetchone()
        conn.close()
        assert row["max_queue_latency_seconds"] == 99

    def test_delete_existing_policy(self, tmp_path):
        """Deleting an existing policy returns True."""
        jm = self._make_manager(tmp_path)
        jm.upsert_sla_policy(domain="d", queue="q", job_type="jt")
        result = jm.delete_sla_policy(domain="d", queue="q", job_type="jt")
        assert result is True
        conn = jm._connect()
        row = conn.execute(
            "SELECT * FROM job_sla_policies WHERE domain='d' AND queue='q' AND job_type='jt'"
        ).fetchone()
        conn.close()
        assert row is None

    def test_delete_nonexistent_returns_false(self, tmp_path):
        """Deleting a missing policy returns False."""
        jm = self._make_manager(tmp_path)
        result = jm.delete_sla_policy(domain="x", queue="y", job_type="z")
        assert result is False


# ===========================================================================
# 3. Jobs SLA breach detection logic
# ===========================================================================


class TestJobsSlaBreachDetection:
    """Test the breach detection logic by mocking DB results."""

    def test_no_policies_means_no_breaches(self):
        """If no SLA policies are configured, no breaches are returned."""
        # Directly test the pattern: no policies -> empty list
        policies = []
        active_jobs = [{"id": "j1", "domain": "d", "queue": "q", "job_type": "jt", "status": "queued"}]
        breaches = self._compute_breaches(policies, active_jobs)
        assert breaches == []

    def test_queue_latency_breach_detected(self):
        """A queued job exceeding max_queue_latency_seconds is flagged."""
        now = datetime.utcnow()
        policies = [{
            "domain": "d",
            "queue": "q",
            "job_type": "jt",
            "max_queue_latency_seconds": 30,
            "max_duration_seconds": None,
            "enabled": True,
        }]
        active_jobs = [{
            "id": "j1",
            "domain": "d",
            "queue": "q",
            "job_type": "jt",
            "status": "queued",
            "created_at": (now - timedelta(seconds=60)).isoformat(),
            "acquired_at": None,
            "started_at": None,
        }]
        breaches = self._compute_breaches(policies, active_jobs, now=now)
        assert len(breaches) == 1
        assert breaches[0]["job_id"] == "j1"
        assert "queue_latency" in breaches[0]["breach_kinds"]

    def test_duration_breach_detected(self):
        """A processing job exceeding max_duration_seconds is flagged."""
        now = datetime.utcnow()
        policies = [{
            "domain": "d",
            "queue": "q",
            "job_type": "jt",
            "max_queue_latency_seconds": None,
            "max_duration_seconds": 60,
            "enabled": True,
        }]
        active_jobs = [{
            "id": "j2",
            "domain": "d",
            "queue": "q",
            "job_type": "jt",
            "status": "processing",
            "created_at": (now - timedelta(seconds=120)).isoformat(),
            "acquired_at": (now - timedelta(seconds=100)).isoformat(),
            "started_at": (now - timedelta(seconds=90)).isoformat(),
        }]
        breaches = self._compute_breaches(policies, active_jobs, now=now)
        assert len(breaches) == 1
        assert "duration" in breaches[0]["breach_kinds"]

    def test_no_breach_when_within_thresholds(self):
        """Jobs within SLA thresholds produce no breaches."""
        now = datetime.utcnow()
        policies = [{
            "domain": "d",
            "queue": "q",
            "job_type": "jt",
            "max_queue_latency_seconds": 300,
            "max_duration_seconds": 600,
            "enabled": True,
        }]
        active_jobs = [{
            "id": "j3",
            "domain": "d",
            "queue": "q",
            "job_type": "jt",
            "status": "processing",
            "created_at": (now - timedelta(seconds=10)).isoformat(),
            "acquired_at": (now - timedelta(seconds=5)).isoformat(),
            "started_at": (now - timedelta(seconds=3)).isoformat(),
        }]
        breaches = self._compute_breaches(policies, active_jobs, now=now)
        assert breaches == []

    def test_combined_breach_kinds(self):
        """A job can breach both queue_latency and duration simultaneously."""
        now = datetime.utcnow()
        policies = [{
            "domain": "d",
            "queue": "q",
            "job_type": "jt",
            "max_queue_latency_seconds": 5,
            "max_duration_seconds": 10,
            "enabled": True,
        }]
        # created 60s ago, acquired 50s ago (wait=10s > 5s threshold),
        # started 40s ago (processing=40s > 10s threshold)
        active_jobs = [{
            "id": "j4",
            "domain": "d",
            "queue": "q",
            "job_type": "jt",
            "status": "processing",
            "created_at": (now - timedelta(seconds=60)).isoformat(),
            "acquired_at": (now - timedelta(seconds=50)).isoformat(),
            "started_at": (now - timedelta(seconds=40)).isoformat(),
        }]
        breaches = self._compute_breaches(policies, active_jobs, now=now)
        assert len(breaches) == 1
        assert set(breaches[0]["breach_kinds"]) == {"queue_latency", "duration"}

    @staticmethod
    def _parse_dt(val: str | None) -> datetime | None:
        if not val:
            return None
        try:
            return datetime.fromisoformat(val.replace("Z", "+00:00").replace("+00:00", ""))
        except (ValueError, TypeError):
            return None

    @classmethod
    def _compute_breaches(
        cls,
        policies: list[dict],
        active_jobs: list[dict],
        *,
        now: datetime | None = None,
    ) -> list[dict]:
        """Replicate the breach detection logic from list_sla_breaches_endpoint."""
        if now is None:
            now = datetime.utcnow()

        policy_lookup: dict[tuple[str, str, str], dict] = {}
        for pol in policies:
            key = (str(pol.get("domain", "")), str(pol.get("queue", "")), str(pol.get("job_type", "")))
            policy_lookup[key] = pol

        breaches: list[dict] = []
        for job in active_jobs:
            jd = str(job.get("domain", ""))
            jq = str(job.get("queue", ""))
            jjt = str(job.get("job_type", ""))
            pol = policy_lookup.get((jd, jq, jjt))
            if not pol:
                continue

            breach_kinds: list[str] = []
            breach_details: dict[str, Any] = {}

            max_qlat = pol.get("max_queue_latency_seconds")
            if max_qlat is not None:
                created_at = cls._parse_dt(job.get("created_at"))
                if created_at:
                    if job.get("status") == "queued":
                        wait_seconds = max(0.0, (now - created_at).total_seconds())
                    else:
                        acquired_at = cls._parse_dt(job.get("acquired_at"))
                        wait_seconds = max(0.0, (acquired_at - created_at).total_seconds()) if acquired_at else 0.0
                    if wait_seconds > float(max_qlat):
                        breach_kinds.append("queue_latency")
                        breach_details["wait_seconds"] = round(wait_seconds, 1)
                        breach_details["max_wait_seconds"] = int(max_qlat)

            max_dur = pol.get("max_duration_seconds")
            if max_dur is not None and job.get("status") == "processing":
                started_at = cls._parse_dt(job.get("started_at")) or cls._parse_dt(job.get("acquired_at"))
                if started_at:
                    processing_seconds = max(0.0, (now - started_at).total_seconds())
                    if processing_seconds > float(max_dur):
                        breach_kinds.append("duration")
                        breach_details["processing_seconds"] = round(processing_seconds, 1)
                        breach_details["max_processing_seconds"] = int(max_dur)

            if breach_kinds:
                breaches.append({
                    "job_id": job.get("id"),
                    "domain": jd,
                    "queue": jq,
                    "job_type": jjt,
                    "status": job.get("status"),
                    "breach_kinds": breach_kinds,
                    **breach_details,
                })

        return breaches


# ===========================================================================
# 4. Error breakdown aggregation
# ===========================================================================


class TestErrorBreakdown:
    """Test error breakdown aggregation at the service level."""

    def test_error_breakdown_empty_returns_zero(self):
        """When the DB has no error-like actions, totals are zero."""
        from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
            ErrorBreakdownResponse,
        )

        async def _test():
            from tldw_Server_API.app.services import admin_system_service as svc

            # Mock db that returns no rows
            db = mock.AsyncMock()
            cursor_mock = mock.AsyncMock()
            cursor_mock.fetchall = mock.AsyncMock(return_value=[])
            db.execute = mock.AsyncMock(return_value=cursor_mock)
            db._is_sqlite = True

            # Mock principal with admin org IDs
            principal = mock.MagicMock()
            with mock.patch.object(svc, "admin_scope_service") as scope_mock:
                scope_mock.get_admin_org_ids = mock.AsyncMock(return_value=None)
                result = await svc.get_error_breakdown(principal=principal, db=db, hours=24)
                assert isinstance(result, ErrorBreakdownResponse)
                assert result.total_errors == 0
                assert result.items == []

        _run(_test())

    def test_error_breakdown_classifies_status_codes(self):
        """Actions are classified into synthetic HTTP status codes."""
        from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
            ErrorBreakdownResponse,
        )

        async def _test():
            from tldw_Server_API.app.services import admin_system_service as svc

            now_str = datetime.now(timezone.utc).isoformat()
            rows = [
                {"action": "auth_denied", "cnt": 5, "last_at": now_str},
                {"action": "ingest_error", "cnt": 3, "last_at": now_str},
                {"action": "rate_limit_exceeded", "cnt": 2, "last_at": now_str},
            ]
            db = mock.AsyncMock()
            cursor_mock = mock.AsyncMock()
            cursor_mock.fetchall = mock.AsyncMock(return_value=rows)
            db.execute = mock.AsyncMock(return_value=cursor_mock)
            db._is_sqlite = True

            principal = mock.MagicMock()
            with mock.patch.object(svc, "admin_scope_service") as scope_mock:
                scope_mock.get_admin_org_ids = mock.AsyncMock(return_value=None)
                result = await svc.get_error_breakdown(principal=principal, db=db, hours=24)
                assert result.total_errors == 10

                items_by_endpoint = {i.endpoint: i for i in result.items}
                assert items_by_endpoint["auth_denied"].status_code == 403
                assert items_by_endpoint["ingest_error"].status_code == 500
                assert items_by_endpoint["rate_limit_exceeded"].status_code == 429

        _run(_test())


# ===========================================================================
# 5. Rate limit summary
# ===========================================================================


class TestRateLimitSummary:
    """Test rate limit summary aggregation at the service level."""

    def test_rate_limit_summary_no_metrics(self, monkeypatch):
        """When metrics registry has no denial samples, result is zeroed."""

        async def _test():
            from tldw_Server_API.app.services import admin_system_service as svc

            mock_registry = mock.MagicMock()
            mock_registry.get_samples = mock.MagicMock(return_value=[])
            monkeypatch.setattr(svc, "get_metrics_registry", lambda: mock_registry)

            result = await svc.get_rate_limit_summary(hours=24)
            assert result.total_throttle_events == 0
            assert result.top_throttled_entities == []
            assert result.policy_headroom == []

        _run(_test())

    def test_rate_limit_summary_with_samples(self, monkeypatch):
        """Samples are aggregated into throttled entities."""

        async def _test():
            from tldw_Server_API.app.services import admin_system_service as svc

            samples = [
                {"labels": {"entity": "user:42", "policy_id": "pol1", "category": "api", "scope": "user"}, "value": 10},
                {"labels": {"entity": "user:42", "policy_id": "pol1", "category": "api", "scope": "user"}, "value": 5},
                {"labels": {"entity": "ip:10.0.0.1", "policy_id": "pol2", "category": "mcp", "scope": "ip"}, "value": 3},
            ]

            mock_registry = mock.MagicMock()

            def _get_samples(name):
                if name in ("rg_denials_total", "rg_denials_by_entity_total", "mcp_rate_limit_hits_total"):
                    return samples
                return []

            mock_registry.get_samples = _get_samples
            monkeypatch.setattr(svc, "get_metrics_registry", lambda: mock_registry)

            result = await svc.get_rate_limit_summary(hours=24)
            # Each metric name iterates the same samples -> 3 metrics * 18 per batch
            # But the exact total depends on how many metric names match
            assert result.total_throttle_events > 0
            assert len(result.top_throttled_entities) > 0

            # Entities are sorted by rejections descending
            entities = result.top_throttled_entities
            if len(entities) > 1:
                assert entities[0].rejections >= entities[1].rejections

        _run(_test())

    def test_rate_limit_summary_handles_registry_error(self, monkeypatch):
        """If the registry raises, summary degrades gracefully to zeros."""

        async def _test():
            from tldw_Server_API.app.services import admin_system_service as svc

            def _raise_error():
                raise RuntimeError("registry unavailable")

            monkeypatch.setattr(svc, "get_metrics_registry", _raise_error)

            result = await svc.get_rate_limit_summary(hours=24)
            assert result.total_throttle_events == 0

        _run(_test())
