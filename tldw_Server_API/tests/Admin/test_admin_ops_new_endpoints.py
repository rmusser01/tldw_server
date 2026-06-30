"""Tests for new admin_ops endpoints: realtime stats, compliance posture,
incident SLA metrics, incident notifications, and email deliveries.

Tests exercise service functions directly (not HTTP) following the monkeypatch +
JSON store isolation pattern established in test_incidents_service.py and
test_admin_api_key_usage.py.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from fastapi import HTTPException


class _LoggerStub:
    def __init__(self) -> None:
        self.debug_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.warning_records: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.debug_records.append((message, args, kwargs))

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.warning_records.append((message, args, kwargs))


# ---------------------------------------------------------------------------
# Shared fixture: isolate the JSON store to a temp directory
# ---------------------------------------------------------------------------


def _configure_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Any, Path]:
    """Redirect the system-ops JSON store to *tmp_path*."""
    from tldw_Server_API.app.services import admin_system_ops_service

    store_path = tmp_path / "system_ops.json"
    monkeypatch.setattr(admin_system_ops_service, "_STORE_PATH", store_path)
    return admin_system_ops_service, store_path


def _create_incident(
    service: Any,
    *,
    title: str = "Queue backlog",
    status: str = "investigating",
    severity: str = "high",
    summary: str = "Jobs delayed",
    tags: list[str] | None = None,
    actor: str = "alice_admin",
) -> dict[str, Any]:
    """Helper to create an incident with sensible defaults."""
    return service.create_incident(
        title=title,
        status=status,
        severity=severity,
        summary=summary,
        tags=tags or ["queue"],
        actor=actor,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. Incident SLA Metrics
# ═══════════════════════════════════════════════════════════════════════════


class TestIncidentSlaMetrics:
    """Tests for the SLA metrics logic that computes MTTA and MTTR from incidents."""

    def test_sla_metrics_computes_mtta_and_mttr(self, monkeypatch, tmp_path):
        """Incidents with acknowledged_at and resolved_at yield correct averages."""
        service, store_path = _configure_store(monkeypatch, tmp_path)

        now = datetime.now(timezone.utc)

        # Seed store with two incidents having known timestamps
        store_data = service._default_store()
        store_data["incidents"] = [
            {
                "id": "inc_aaa",
                "title": "Outage A",
                "status": "resolved",
                "severity": "high",
                "summary": None,
                "tags": [],
                "created_at": (now - timedelta(minutes=60)).isoformat(),
                "updated_at": now.isoformat(),
                "acknowledged_at": (now - timedelta(minutes=50)).isoformat(),
                "resolved_at": (now - timedelta(minutes=30)).isoformat(),
                "created_by": "bot",
                "updated_by": "bot",
                "timeline": [],
                "assigned_to_user_id": None,
                "assigned_to_label": None,
                "root_cause": None,
                "impact": None,
                "action_items": [],
            },
            {
                "id": "inc_bbb",
                "title": "Outage B",
                "status": "resolved",
                "severity": "medium",
                "summary": None,
                "tags": [],
                "created_at": (now - timedelta(minutes=120)).isoformat(),
                "updated_at": now.isoformat(),
                "acknowledged_at": (now - timedelta(minutes=100)).isoformat(),
                "resolved_at": (now - timedelta(minutes=60)).isoformat(),
                "created_by": "bot",
                "updated_by": "bot",
                "timeline": [],
                "assigned_to_user_id": None,
                "assigned_to_label": None,
                "root_cause": None,
                "impact": None,
                "action_items": [],
            },
        ]
        store_path.write_text(json.dumps(store_data), encoding="utf-8")

        # list_incidents computes mtta_minutes and mttr_minutes per incident
        incidents, total = service.list_incidents(
            status=None,
            severity=None,
            tag=None,
            limit=10000,
            offset=0,
        )

        assert total == 2

        # Incident A: MTTA = 10 min (60-50), MTTR = 30 min (60-30)
        inc_a = next(i for i in incidents if i["id"] == "inc_aaa")
        assert inc_a["mtta_minutes"] == 10.0
        assert inc_a["mttr_minutes"] == 30.0

        # Incident B: MTTA = 20 min (120-100), MTTR = 60 min (120-60)
        inc_b = next(i for i in incidents if i["id"] == "inc_bbb")
        assert inc_b["mtta_minutes"] == 20.0
        assert inc_b["mttr_minutes"] == 60.0

    def test_sla_metrics_null_when_not_acknowledged(self, monkeypatch, tmp_path):
        """Open incidents without acknowledged_at have null MTTA."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        service.create_incident(
            title="New issue",
            status="open",
            severity="low",
            summary="Just opened",
            tags=[],
            actor="bot",
        )

        incidents, _ = service.list_incidents(
            status=None,
            severity=None,
            tag=None,
            limit=100,
            offset=0,
        )

        assert len(incidents) == 1
        # Open incident: no acknowledged_at, no resolved_at
        assert incidents[0]["mtta_minutes"] is None
        assert incidents[0]["mttr_minutes"] is None

    def test_sla_metrics_null_mttr_when_not_resolved(self, monkeypatch, tmp_path):
        """Acknowledged but unresolved incidents have MTTA but null MTTR."""
        service, store_path = _configure_store(monkeypatch, tmp_path)

        now = datetime.now(timezone.utc)
        store_data = service._default_store()
        store_data["incidents"] = [
            {
                "id": "inc_ccc",
                "title": "In progress",
                "status": "investigating",
                "severity": "high",
                "summary": None,
                "tags": [],
                "created_at": (now - timedelta(minutes=30)).isoformat(),
                "updated_at": now.isoformat(),
                "acknowledged_at": (now - timedelta(minutes=25)).isoformat(),
                "resolved_at": None,
                "created_by": "bot",
                "updated_by": "bot",
                "timeline": [],
                "assigned_to_user_id": None,
                "assigned_to_label": None,
                "root_cause": None,
                "impact": None,
                "action_items": [],
            },
        ]
        store_path.write_text(json.dumps(store_data), encoding="utf-8")

        incidents, total = service.list_incidents(
            status=None,
            severity=None,
            tag=None,
            limit=100,
            offset=0,
        )

        assert total == 1
        assert incidents[0]["mtta_minutes"] == 5.0  # 30 - 25
        assert incidents[0]["mttr_minutes"] is None

    def test_sla_metrics_empty_incidents(self, monkeypatch, tmp_path):
        """No incidents yields empty list and zero total."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        incidents, total = service.list_incidents(
            status=None,
            severity=None,
            tag=None,
            limit=100,
            offset=0,
        )

        assert total == 0
        assert incidents == []


# ═══════════════════════════════════════════════════════════════════════════
# 2. Incident Notifications
# ═══════════════════════════════════════════════════════════════════════════


class TestIncidentNotifications:
    """Tests for notify_incident_stakeholders service function."""

    def test_notify_sends_to_all_recipients(self, monkeypatch, tmp_path):
        """All recipients receive notification and timeline event is added."""
        service, _ = _configure_store(monkeypatch, tmp_path)
        incident = _create_incident(service)

        # Mock the email service so no real emails are sent
        mock_email_svc = mock.MagicMock()
        mock_email_svc.send_email = mock.AsyncMock(return_value=None)

        monkeypatch.setattr(
            "tldw_Server_API.app.core.AuthNZ.email_service.get_email_service",
            lambda: mock_email_svc,
        )

        result = service.notify_incident_stakeholders(
            incident_id=incident["id"],
            recipients=["alice@example.com", "bob@example.com", "carol@example.com"],
            message="Please review the incident.",
            actor="admin_user",
        )

        assert result["incident_id"] == incident["id"]
        assert len(result["notifications"]) == 3
        assert all(n["status"] == "sent" for n in result["notifications"])

        # Verify timeline event was appended to the incident
        updated = service.get_incident(incident_id=incident["id"])
        timeline_msgs = [e["message"] for e in updated["timeline"]]
        assert any("3/3" in msg for msg in timeline_msgs)

    def test_notify_handles_partial_failure(self, monkeypatch, tmp_path):
        """When one send fails, results include both sent and failed entries."""
        service, _ = _configure_store(monkeypatch, tmp_path)
        incident = _create_incident(service)

        call_count = 0

        async def _send_email_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("to_email") == "fail@example.com":
                raise RuntimeError("SMTP connection refused")

        mock_email_svc = mock.MagicMock()
        mock_email_svc.send_email = _send_email_side_effect

        monkeypatch.setattr(
            "tldw_Server_API.app.core.AuthNZ.email_service.get_email_service",
            lambda: mock_email_svc,
        )

        result = service.notify_incident_stakeholders(
            incident_id=incident["id"],
            recipients=["ok1@example.com", "fail@example.com", "ok2@example.com"],
            actor="admin_user",
        )

        statuses = {n["email"]: n["status"] for n in result["notifications"]}
        assert statuses["ok1@example.com"] == "sent"
        assert statuses["fail@example.com"] == "failed"
        assert statuses["ok2@example.com"] == "sent"

        # Timeline records the partial result
        updated = service.get_incident(incident_id=incident["id"])
        timeline_msgs = [e["message"] for e in updated["timeline"]]
        assert any("2/3" in msg for msg in timeline_msgs)

    def test_notify_nonexistent_incident_raises(self, monkeypatch, tmp_path):
        """Notifying a nonexistent incident raises ValueError."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        with pytest.raises(ValueError, match="not_found"):
            service.notify_incident_stakeholders(
                incident_id="inc_does_not_exist",
                recipients=["someone@example.com"],
                actor="admin_user",
            )

    def test_notify_empty_recipients_skipped(self, monkeypatch, tmp_path):
        """Blank recipient strings are skipped."""
        service, _ = _configure_store(monkeypatch, tmp_path)
        incident = _create_incident(service)

        mock_email_svc = mock.MagicMock()
        mock_email_svc.send_email = mock.AsyncMock(return_value=None)

        monkeypatch.setattr(
            "tldw_Server_API.app.core.AuthNZ.email_service.get_email_service",
            lambda: mock_email_svc,
        )

        result = service.notify_incident_stakeholders(
            incident_id=incident["id"],
            recipients=["", "  ", "valid@example.com"],
            actor="admin_user",
        )

        # Only the non-blank recipient should have been processed
        assert len(result["notifications"]) == 1
        assert result["notifications"][0]["email"] == "valid@example.com"


# ═══════════════════════════════════════════════════════════════════════════
# 3. Email Deliveries
# ═══════════════════════════════════════════════════════════════════════════


class TestEmailDeliveries:
    """Tests for record_email_delivery and list_email_deliveries."""

    def test_record_and_list_email_deliveries(self, monkeypatch, tmp_path):
        """Record several deliveries and retrieve them with pagination."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        for i in range(5):
            service.record_email_delivery(
                recipient=f"user{i}@example.com",
                subject=f"Subject {i}",
                template="welcome",
                status="sent",
            )

        items, total = service.list_email_deliveries(limit=3, offset=0)

        assert total == 5
        assert len(items) == 3
        # Newest first
        assert items[0]["recipient"] == "user4@example.com"

    def test_email_delivery_order_uses_append_order_for_timestamp_ties(self, monkeypatch, tmp_path):
        """Records with equal timestamps still list the most recent append first."""
        service, _ = _configure_store(monkeypatch, tmp_path)
        monkeypatch.setattr(service, "_now_iso", lambda: "2026-04-26T01:00:00Z")

        for i in range(5):
            service.record_email_delivery(
                recipient=f"user{i}@example.com",
                subject=f"Subject {i}",
                template="welcome",
                status="sent",
            )

        items, total = service.list_email_deliveries(limit=3, offset=0)

        assert total == 5
        assert [item["recipient"] for item in items] == [
            "user4@example.com",
            "user3@example.com",
            "user2@example.com",
        ]

    def test_email_delivery_pagination_offset(self, monkeypatch, tmp_path):
        """Offset parameter correctly skips entries."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        for i in range(5):
            service.record_email_delivery(
                recipient=f"user{i}@example.com",
                subject=f"Subject {i}",
                template=None,
                status="sent",
            )

        items, total = service.list_email_deliveries(limit=10, offset=2)

        assert total == 5
        assert len(items) == 3  # 5 - 2 offset

    def test_email_delivery_status_filter(self, monkeypatch, tmp_path):
        """Filtering by status returns only matching entries."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        service.record_email_delivery(
            recipient="ok@example.com",
            subject="Test",
            template=None,
            status="sent",
        )
        service.record_email_delivery(
            recipient="fail@example.com",
            subject="Test",
            template=None,
            status="failed",
            error="SMTP error",
        )
        service.record_email_delivery(
            recipient="ok2@example.com",
            subject="Test",
            template=None,
            status="sent",
        )

        items, total = service.list_email_deliveries(status="failed")

        assert total == 1
        assert items[0]["recipient"] == "fail@example.com"
        assert items[0]["error"] == "SMTP error"

    def test_email_delivery_empty_log(self, monkeypatch, tmp_path):
        """Empty delivery log returns zero total and empty list."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        items, total = service.list_email_deliveries()

        assert total == 0
        assert items == []

    def test_email_delivery_record_fields(self, monkeypatch, tmp_path):
        """Each recorded delivery has all expected fields."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        entry = service.record_email_delivery(
            recipient="test@example.com",
            subject="Welcome!",
            template="onboarding",
            status="sent",
            error=None,
        )

        assert entry["id"].startswith("edl_")
        assert entry["recipient"] == "test@example.com"
        assert entry["subject"] == "Welcome!"
        assert entry["template"] == "onboarding"
        assert entry["status"] == "sent"
        assert entry["error"] is None
        assert entry["sent_at"] is not None

    def test_email_delivery_log_cap(self, monkeypatch, tmp_path):
        """Log is capped at _EMAIL_DELIVERY_LOG_CAP entries."""
        service, _ = _configure_store(monkeypatch, tmp_path)

        cap = service._EMAIL_DELIVERY_LOG_CAP

        # Record cap + 10 entries
        for i in range(cap + 10):
            service.record_email_delivery(
                recipient=f"user{i}@example.com",
                subject="Cap test",
                template=None,
                status="sent",
            )

        items, total = service.list_email_deliveries(limit=cap + 100)

        # Total should not exceed the cap
        assert total <= cap


# ═══════════════════════════════════════════════════════════════════════════
# 4. Realtime Stats (endpoint logic uses ACP session store)
# ═══════════════════════════════════════════════════════════════════════════


class TestRealtimeStats:
    """Test the realtime stats logic.

    The ``get_realtime_stats`` endpoint is async and depends on the ACP
    session store.  We test the aggregation logic by mocking the store.
    """

    @pytest.mark.asyncio
    async def test_realtime_stats_returns_active_sessions_and_tokens(self, monkeypatch):
        """When ACP store is available, stats reflect session counts and tokens."""
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

        # Mock the ACP session store
        mock_store = mock.AsyncMock()
        mock_store.list_sessions.return_value = ([], 7)  # 7 active sessions
        mock_store.get_agent_metrics.return_value = [
            {"total_prompt_tokens": 100, "total_completion_tokens": 50, "total_tokens": 150},
            {"total_prompt_tokens": 200, "total_completion_tokens": 100, "total_tokens": 300},
        ]

        async def _fake_get_acp_session_store():
            return mock_store

        monkeypatch.setattr(
            "tldw_Server_API.app.api.v1.endpoints.admin.admin_ops._require_platform_admin",
            lambda _: None,
        )

        # Patch the import inside the endpoint
        monkeypatch.setattr(
            "tldw_Server_API.app.services.admin_acp_sessions_service.get_acp_session_store",
            _fake_get_acp_session_store,
        )

        mock_principal = mock.MagicMock()
        result = await admin_ops.get_realtime_stats(principal=mock_principal)

        assert result["active_sessions"] == 7
        assert result["tokens_today"]["prompt"] == 300
        assert result["tokens_today"]["completion"] == 150
        assert result["tokens_today"]["total"] == 450

    @pytest.mark.asyncio
    async def test_realtime_stats_graceful_degradation(self, monkeypatch):
        """When ACP store is unavailable, stats return zeroed values."""
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

        logger_stub = _LoggerStub()
        monkeypatch.setattr(admin_ops, "logger", logger_stub)
        monkeypatch.setattr(
            "tldw_Server_API.app.api.v1.endpoints.admin.admin_ops._require_platform_admin",
            lambda _: None,
        )

        # Make the ACP import raise so the endpoint hits the except branch
        def _raise_import(*args, **kwargs):
            raise RuntimeError("ACP not available")

        monkeypatch.setattr(
            "tldw_Server_API.app.services.admin_acp_sessions_service.get_acp_session_store",
            _raise_import,
        )

        mock_principal = mock.MagicMock()
        result = await admin_ops.get_realtime_stats(principal=mock_principal)

        assert result["active_sessions"] == 0
        assert result["tokens_today"]["prompt"] == 0
        assert result["tokens_today"]["completion"] == 0
        assert result["tokens_today"]["total"] == 0
        assert logger_stub.debug_records == [("Realtime stats: ACP session store unavailable", (), {})]


# ═══════════════════════════════════════════════════════════════════════════
# 5. Compliance Posture (endpoint logic queries the DB pool)
# ═══════════════════════════════════════════════════════════════════════════


class TestCompliancePosture:
    """Test the compliance posture aggregation logic.

    The endpoint queries AuthNZ tables directly, so we mock the DB pool.
    """

    @pytest.mark.asyncio
    async def test_compliance_posture_computes_mfa_adoption(self, monkeypatch):
        """MFA adoption rate is computed correctly from user counts."""
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

        monkeypatch.setattr(
            "tldw_Server_API.app.api.v1.endpoints.admin.admin_ops._require_platform_admin",
            lambda _: None,
        )

        # Mock the database pool (SQLite path: pool attribute is None)
        mock_pool = mock.AsyncMock()
        mock_pool.pool = None  # Indicates SQLite

        call_index = 0

        async def _fake_fetchone(query, *args):
            nonlocal call_index
            call_index += 1
            if "users" in query:
                # 10 total active users, 6 with MFA enabled
                return {"total": 10, "mfa_on": 6}
            elif "api_keys" in query:
                # 8 active keys, 5 within rotation threshold
                return {"total": 8, "compliant": 5}
            return None

        mock_pool.fetchone = _fake_fetchone

        monkeypatch.setattr(
            "tldw_Server_API.app.api.v1.endpoints.admin.admin_ops.get_db_pool",
            mock.AsyncMock(return_value=mock_pool),
        )

        mock_principal = mock.MagicMock()
        result = await admin_ops.get_compliance_posture(principal=mock_principal)

        assert result["total_users"] == 10
        assert result["mfa_enabled_count"] == 6
        assert result["mfa_adoption_pct"] == 60.0
        assert result["keys_total"] == 8
        assert result["key_rotation_compliance_pct"] == 62.5
        assert result["keys_needing_rotation"] == 3
        # Overall: 60*0.4 + 62.5*0.4 + 20 = 24 + 25 + 20 = 69.0
        assert result["overall_score"] == 69.0
        assert result["audit_logging_enabled"] is True

    @pytest.mark.asyncio
    async def test_compliance_posture_handles_no_users(self, monkeypatch):
        """Zero users yields zeroed metrics without error."""
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

        monkeypatch.setattr(
            "tldw_Server_API.app.api.v1.endpoints.admin.admin_ops._require_platform_admin",
            lambda _: None,
        )

        mock_pool = mock.AsyncMock()
        mock_pool.pool = None

        async def _fake_fetchone(query, *args):
            if "users" in query:
                return {"total": 0, "mfa_on": 0}
            elif "api_keys" in query:
                return {"total": 0, "compliant": 0}
            return None

        mock_pool.fetchone = _fake_fetchone

        monkeypatch.setattr(
            "tldw_Server_API.app.api.v1.endpoints.admin.admin_ops.get_db_pool",
            mock.AsyncMock(return_value=mock_pool),
        )

        mock_principal = mock.MagicMock()
        result = await admin_ops.get_compliance_posture(principal=mock_principal)

        assert result["total_users"] == 0
        assert result["mfa_adoption_pct"] == 0.0
        assert result["key_rotation_compliance_pct"] == 0.0
        # Overall: 0*0.4 + 0*0.4 + 20 = 20.0
        assert result["overall_score"] == 20.0

    @pytest.mark.asyncio
    async def test_compliance_posture_handles_db_error_gracefully(self, monkeypatch):
        """DB errors yield zeroed metrics rather than an exception."""
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

        logger_stub = _LoggerStub()
        monkeypatch.setattr(admin_ops, "logger", logger_stub)
        monkeypatch.setattr(
            "tldw_Server_API.app.api.v1.endpoints.admin.admin_ops._require_platform_admin",
            lambda _: None,
        )

        mock_pool = mock.AsyncMock()
        mock_pool.pool = None

        async def _raise_always(query, *args):
            raise RuntimeError("DB unavailable")

        mock_pool.fetchone = _raise_always

        monkeypatch.setattr(
            "tldw_Server_API.app.api.v1.endpoints.admin.admin_ops.get_db_pool",
            mock.AsyncMock(return_value=mock_pool),
        )

        mock_principal = mock.MagicMock()
        result = await admin_ops.get_compliance_posture(principal=mock_principal)

        # Both queries fail, so all counts are zero
        assert result["total_users"] == 0
        assert result["mfa_adoption_pct"] == 0.0
        assert result["key_rotation_compliance_pct"] == 0.0
        assert result["overall_score"] == 20.0
        assert logger_stub.warning_records == [
            ("compliance/posture: MFA query failed", (), {}),
            ("compliance/posture: API key rotation query failed", (), {}),
        ]

    @pytest.mark.asyncio
    async def test_compliance_posture_overall_score_capped_at_100(self, monkeypatch):
        """Overall score never exceeds 100 even with perfect compliance."""
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

        monkeypatch.setattr(
            "tldw_Server_API.app.api.v1.endpoints.admin.admin_ops._require_platform_admin",
            lambda _: None,
        )

        mock_pool = mock.AsyncMock()
        mock_pool.pool = None

        async def _fake_fetchone(query, *args):
            if "users" in query:
                return {"total": 10, "mfa_on": 10}  # 100% MFA
            elif "api_keys" in query:
                return {"total": 5, "compliant": 5}  # 100% rotation
            return None

        mock_pool.fetchone = _fake_fetchone

        monkeypatch.setattr(
            "tldw_Server_API.app.api.v1.endpoints.admin.admin_ops.get_db_pool",
            mock.AsyncMock(return_value=mock_pool),
        )

        mock_principal = mock.MagicMock()
        result = await admin_ops.get_compliance_posture(principal=mock_principal)

        # 100*0.4 + 100*0.4 + 20 = 100.0 (capped)
        assert result["overall_score"] == 100.0


class TestAdminOpsErrorSanitization:
    """Direct endpoint tests for narrow admin_ops fallback sanitization."""

    @pytest.mark.asyncio
    async def test_billing_analytics_sanitizes_backend_warning_log(self, monkeypatch):
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

        class _ExplodingBillingRepo:
            def __init__(self, db_pool: object) -> None:
                self.db_pool = db_pool

            async def list_all_subscriptions(self):
                raise RuntimeError("billing backend exploded at /private/billing.db")

        logger_stub = _LoggerStub()
        monkeypatch.setattr(admin_ops, "logger", logger_stub)
        monkeypatch.setattr(
            "tldw_Server_API.app.api.v1.endpoints.admin.admin_ops._require_platform_admin",
            lambda _: None,
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.Billing.runtime_flags.is_billing_enabled",
            lambda: True,
        )
        monkeypatch.setattr(
            "tldw_Server_API.app.core.AuthNZ.repos.billing_repo.AuthnzBillingRepo",
            _ExplodingBillingRepo,
        )
        monkeypatch.setattr(admin_ops, "get_db_pool", mock.AsyncMock(return_value=object()))

        result = await admin_ops.get_billing_analytics(principal=mock.MagicMock())

        assert result["mrr_cents"] == 0
        assert result["subscriber_count"] == 0
        assert result["plan_distribution"] == []
        assert logger_stub.warning_records == [("billing/analytics: failed to compute metrics", (), {})]

    @pytest.mark.asyncio
    async def test_get_dependency_uptime_sanitizes_backend_error(self, monkeypatch):
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

        monkeypatch.setattr(
            "tldw_Server_API.app.api.v1.endpoints.admin.admin_ops._require_platform_admin",
            lambda _: None,
        )

        def _raise_uptime(name: str, days: int) -> dict[str, Any]:
            _ = (name, days)
            raise OSError("uptime backend exploded")

        monkeypatch.setattr(admin_ops, "svc_get_uptime_stats", _raise_uptime)

        with pytest.raises(HTTPException) as exc_info:
            await admin_ops.get_dependency_uptime(
                name="authnz",
                days=30,
                principal=mock.MagicMock(),
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to get dependency uptime"

    @pytest.mark.asyncio
    async def test_list_email_deliveries_sanitizes_backend_error(self, monkeypatch):
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

        monkeypatch.setattr(
            "tldw_Server_API.app.api.v1.endpoints.admin.admin_ops._require_platform_admin",
            lambda _: None,
        )

        def _raise_deliveries(*, limit: int, offset: int, status: str | None):
            _ = (limit, offset, status)
            raise OSError("email delivery backend exploded")

        monkeypatch.setattr(admin_ops, "svc_list_email_deliveries", _raise_deliveries)

        with pytest.raises(HTTPException) as exc_info:
            await admin_ops.list_email_deliveries(
                limit=50,
                offset=0,
                status=None,
                principal=mock.MagicMock(),
            )

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to list email deliveries"

    @pytest.mark.asyncio
    async def test_list_email_deliveries_returns_canonical_pagination(self, monkeypatch):
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

        monkeypatch.setattr(
            "tldw_Server_API.app.api.v1.endpoints.admin.admin_ops._require_platform_admin",
            lambda _: None,
        )

        def _list_deliveries(*, limit: int, offset: int, status: str | None):
            assert limit == 2
            assert offset == 1
            assert status == "failed"
            return ([{"id": "delivery-2"}], 3)

        monkeypatch.setattr(admin_ops, "svc_list_email_deliveries", _list_deliveries)

        response = await admin_ops.list_email_deliveries(
            limit=2,
            offset=1,
            status="failed",
            principal=mock.MagicMock(),
        )
        payload = response.model_dump(mode="json")

        assert payload["items"] == [
            {
                "id": "delivery-2",
                "recipient": "",
                "subject": "",
                "template": None,
                "status": "",
                "error": None,
                "sent_at": None,
            }
        ]
        assert payload["total"] == 3
        assert payload["limit"] == 2
        assert payload["offset"] == 1
        assert payload["pagination"] == {
            "mode": "offset",
            "limit": 2,
            "offset": 1,
            "total": 3,
            "has_more": True,
            "next_offset": 3,
        }
        assert payload["has_more"] is True
        assert payload["next_offset"] == 3

    @pytest.mark.asyncio
    async def test_reload_llm_pricing_catalog_sanitizes_backend_error_log(self, monkeypatch):
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

        def _raise_reload() -> None:
            raise RuntimeError("pricing reload exploded at /private/pricing.json")

        messages: list[str] = []
        sink_id = admin_ops.logger.add(lambda message: messages.append(str(message)), level="ERROR")
        monkeypatch.setattr(admin_ops, "reset_pricing_catalog", _raise_reload)

        try:
            with pytest.raises(HTTPException) as exc_info:
                await admin_ops.reload_llm_pricing_catalog()
        finally:
            admin_ops.logger.remove(sink_id)

        joined = "\n".join(messages)
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to reload pricing catalog"
        assert "Failed to reload pricing catalog" in joined
        assert "pricing reload exploded" not in joined
        assert "/private/" not in joined

    @pytest.mark.asyncio
    async def test_reload_chat_model_alias_caches_sanitizes_backend_error_log(self, monkeypatch):
        from tldw_Server_API.app.api.v1.endpoints.admin import admin_ops

        def _raise_reload() -> None:
            raise RuntimeError("model alias reload exploded at /private/model_aliases.json")

        messages: list[str] = []
        sink_id = admin_ops.logger.add(lambda message: messages.append(str(message)), level="ERROR")
        monkeypatch.setattr(admin_ops, "invalidate_model_alias_caches", _raise_reload)

        try:
            with pytest.raises(HTTPException) as exc_info:
                await admin_ops.reload_chat_model_alias_caches()
        finally:
            admin_ops.logger.remove(sink_id)

        joined = "\n".join(messages)
        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Failed to reload chat model alias caches"
        assert "Failed to reload chat model alias caches" in joined
        assert "model alias reload exploded" not in joined
        assert "/private/" not in joined
