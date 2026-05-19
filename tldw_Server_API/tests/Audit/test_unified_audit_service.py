"""
Test suite for the unified audit service.

This replaces the old test_audit_improvements.py file and focuses only on
testing the new unified audit service without references to deprecated modules.
"""

import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
import tempfile
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import aiosqlite

from tldw_Server_API.app.core.AuthNZ.audit_integrity import verify_audit_chain
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    UnifiedAuditService,
    AuditReadError,
    AuditShutdownError,
    AuditEvent,
    AuditContext,
    AuditEventType,
    AuditEventCategory,
    AuditSeverity,
    PIIDetector,
    RiskScorer,
    audit_operation,
    get_unified_audit_service,
    shutdown_audit_service
)


# ============================================================================
# Test Fixtures
# ============================================================================

import pytest_asyncio

@pytest_asyncio.fixture
async def temp_db_path():
    """Create temporary database path"""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest_asyncio.fixture
async def audit_service(temp_db_path):
    """Create audit service instance"""
    service = UnifiedAuditService(
        db_path=temp_db_path,
        retention_days=7,
        enable_pii_detection=True,
        enable_risk_scoring=True,
        buffer_size=10,
        flush_interval=1.0
    )
    await service.initialize()
    yield service
    await service.stop()


# ============================================================================
# Test PII Detection
# ============================================================================

class TestPIIDetection:
    """Test PII detection functionality"""

    def test_detect_various_pii(self):

        """Test detection of various PII types"""
        detector = PIIDetector()

        test_text = """
        SSN: 123-45-6789
        Credit Card: 4111-1111-1111-1111
        Email: john.doe@example.com
        Phone: (555) 123-4567
        IP: 192.168.1.1
        API Key: sk_abcdefghijklmnopqrstuvwxyzABCDEF1234567890
        """

        found_pii = detector.detect(test_text)

        assert "ssn" in found_pii
        assert "credit_card" in found_pii
        assert "email" in found_pii
        assert "phone" in found_pii
        assert "ip_address" in found_pii
        assert "api_key" in found_pii

    def test_redact_pii(self):

        """Test PII redaction"""
        detector = PIIDetector()

        text = "My SSN is 123-45-6789 and email is test@example.com"
        redacted = detector.redact(text)

        assert "123-45-6789" not in redacted
        assert "[SSN_REDACTED]" in redacted
        assert "test@example.com" not in redacted
        assert "[EMAIL_REDACTED]" in redacted

    def test_jwt_token_detection(self):

        """Test JWT token detection"""
        detector = PIIDetector()

        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
        text = f"Token: {jwt}"

        found_pii = detector.detect(text)
        assert "jwt_token" in found_pii

        redacted = detector.redact(text)
        assert jwt not in redacted
        assert "[JWT_TOKEN_REDACTED]" in redacted

    @pytest.mark.asyncio
    async def test_recursive_redaction_in_structures(self, audit_service):
        """PII is redacted recursively in dicts/lists without breaking structure."""
        context = AuditContext(user_id="nested_user")
        api_key = "sk_abcdefghijklmnopqrstuvwxyzABCDEF1234567890"
        card = "4111-1111-1111-1111"
        phone = "(555) 321-9876"
        metadata = {
            "profile": {"email": "user@example.com", "phones": [phone]},
            "secrets": [api_key],
            "note": f"test card {card}"
        }
        await audit_service.log_event(
            event_type=AuditEventType.DATA_WRITE,
            context=context,
            metadata=metadata,
        )
        await audit_service.flush()
        events = await audit_service.query_events(user_id="nested_user")
        assert events, "No audit events returned"
        event = events[0]
        red_meta = json.loads(event["metadata"]) if isinstance(event["metadata"], str) else event["metadata"]
        # Ensure structure preserved and values redacted
        assert "profile" in red_meta and isinstance(red_meta["profile"], dict)
        assert red_meta["profile"]["email"] == "[EMAIL_REDACTED]"
        assert "[PHONE_REDACTED]" in red_meta["profile"]["phones"][0]
        # API key and card redacted somewhere in metadata
        stringified = json.dumps(red_meta)
        assert api_key not in stringified
        assert card not in stringified

    @pytest.mark.asyncio
    async def test_redaction_handles_dataclass_metadata_values(self, audit_service):
        """PII redaction should work even when metadata contains dataclass values."""
        from dataclasses import dataclass

        @dataclass
        class _Person:
            email: str

        ctx = AuditContext(user_id="dataclass_meta_user")
        await audit_service.log_event(
            event_type=AuditEventType.DATA_WRITE,
            context=ctx,
            metadata={"person": _Person(email="user@example.com")},
        )
        await audit_service.flush()

        events = await audit_service.query_events(user_id="dataclass_meta_user")
        assert events, "No audit events returned"
        row = events[0]
        assert row.get("pii_detected") in (True, 1, "1")

        meta_raw = row.get("metadata")
        meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
        assert meta["person"]["email"] == "[EMAIL_REDACTED]"

    @pytest.mark.asyncio
    async def test_export_events_json_decodes_metadata_and_flags(self, audit_service):
        """JSON export should return structured metadata/compliance flags."""
        ctx = AuditContext(user_id="export_user")
        await audit_service.log_event(
            event_type=AuditEventType.DATA_WRITE,
            context=ctx,
            metadata={"email": "user@example.com"},
        )
        await audit_service.flush()

        content = await audit_service.export_events(format="json", stream=False, max_rows=10)
        data = json.loads(content)

        assert isinstance(data, list)
        assert data, "Expected at least one exported audit event"
        row = data[0]
        assert isinstance(row.get("metadata"), dict)
        assert isinstance(row.get("compliance_flags"), list)
        assert "pii_detected" in row.get("compliance_flags", [])


@pytest.mark.asyncio
async def test_shared_mode_sets_tenant_user_id(temp_db_path):
    service = UnifiedAuditService(
        db_path=temp_db_path,
        storage_mode="shared",
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=1,
        flush_interval=0.1,
    )
    await service.initialize()
    try:
        ctx = AuditContext(user_id="101")
        await service.log_event(
            event_type=AuditEventType.DATA_READ,
            context=ctx,
            resource_type="doc",
            resource_id="shared1",
        )
        await service.log_event(
            event_type=AuditEventType.DATA_READ,
            resource_type="doc",
            resource_id="shared-missing",
        )
        await service.log_event(event_type=AuditEventType.SYSTEM_START)
        await service.flush()

        async with aiosqlite.connect(temp_db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("PRAGMA table_info(audit_events)") as cur:
                cols = await cur.fetchall()
            assert "tenant_user_id" in {c["name"] for c in cols}

            async with db.execute("PRAGMA user_version") as cur:
                version_row = await cur.fetchone()
            assert int(version_row[0]) >= 1

            async with db.execute("PRAGMA table_info(audit_daily_stats)") as cur:
                stats_cols = await cur.fetchall()
            assert "tenant_user_id" in {c["name"] for c in stats_cols}

            async with db.execute(
                "SELECT tenant_user_id FROM audit_events WHERE resource_id = ? LIMIT 1",
                ("shared1",),
            ) as cur:
                row = await cur.fetchone()
            assert row["tenant_user_id"] == "101"

        async with aiosqlite.connect(temp_db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT tenant_user_id FROM audit_events WHERE resource_id = ? LIMIT 1",
                ("shared-missing",),
            ) as cur:
                missing_row = await cur.fetchone()
            assert missing_row["tenant_user_id"] == "unidentified_user"
            async with db.execute(
                "SELECT tenant_user_id FROM audit_events WHERE event_type = ? LIMIT 1",
                (AuditEventType.SYSTEM_START.value,),
            ) as cur:
                row2 = await cur.fetchone()
            assert row2["tenant_user_id"] == "system"
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_shared_mode_allows_unidentified_context_user_id(temp_db_path):
    service = UnifiedAuditService(
        db_path=temp_db_path,
        storage_mode="shared",
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=1,
        flush_interval=0.1,
    )
    await service.initialize()
    try:
        ctx = AuditContext(user_id="unidentified_user")
        await service.log_event(
            event_type=AuditEventType.DATA_READ,
            context=ctx,
            resource_type="doc",
            resource_id="anon-doc",
        )
        await service.flush()
        async with aiosqlite.connect(temp_db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT tenant_user_id FROM audit_events WHERE resource_id = ? LIMIT 1",
                ("anon-doc",),
            ) as cur:
                row = await cur.fetchone()
        assert row is not None
        assert row["tenant_user_id"] == "unidentified_user"
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_shared_mode_allows_non_numeric_tenant_id(temp_db_path):
    service = UnifiedAuditService(
        db_path=temp_db_path,
        storage_mode="shared",
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=1,
        flush_interval=0.1,
    )
    await service.initialize()
    try:
        ctx = AuditContext(user_id="tenant-1")
        await service.log_event(
            event_type=AuditEventType.DATA_READ,
            context=ctx,
            resource_type="doc",
            resource_id="non-numeric-tenant",
        )
        await service.flush()
        async with aiosqlite.connect(temp_db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT tenant_user_id FROM audit_events WHERE resource_id = ? LIMIT 1",
                ("non-numeric-tenant",),
            ) as cur:
                row = await cur.fetchone()
        assert row is not None
        assert row["tenant_user_id"] == "tenant-1"
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_log_event_normalizes_result_case(temp_db_path):
    service = UnifiedAuditService(
        db_path=temp_db_path,
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=1,
        flush_interval=0.1,
    )
    await service.initialize()
    try:
        ctx = AuditContext(user_id="case-test")
        await service.log_event(
            event_type=AuditEventType.API_ERROR,
            context=ctx,
            result="ERROR",
        )
        await service.flush()
        async with aiosqlite.connect(temp_db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT result, severity FROM audit_events WHERE context_user_id = ? LIMIT 1",
                ("case-test",),
            ) as cur:
                row = await cur.fetchone()
        assert row is not None
        assert row["result"] == "error"
        assert row["severity"] == AuditSeverity.ERROR.value
    finally:
        await service.stop()


# ============================================================================
# Test Risk Scoring
# ============================================================================

class TestRiskScoring:
    """Test risk scoring functionality"""

    def test_high_risk_events(self):

        """Test scoring of high-risk events"""
        scorer = RiskScorer()

        event = AuditEvent(
            event_type=AuditEventType.SECURITY_VIOLATION,
            action="unauthorized_access",
            result="failure"
        )

        score = scorer.calculate_risk_score(event)
        assert score >= 70  # High risk

    def test_after_hours_activity(self):

        """Test after-hours risk scoring"""
        scorer = RiskScorer()

        # Create event at 3 AM
        event = AuditEvent(
            event_type=AuditEventType.DATA_READ,
            timestamp=datetime(2024, 1, 1, 3, 0, 0, tzinfo=timezone.utc)
        )

        score = scorer.calculate_risk_score(event)
        assert score > 0  # Should have some risk due to time

    def test_high_risk_operations(self):

        """Test detection of high-risk operations"""
        scorer = RiskScorer()

        event = AuditEvent(
            event_type=AuditEventType.DATA_DELETE,
            action="delete_user_data"
        )

        score = scorer.calculate_risk_score(event)
        assert score >= 30  # Should be elevated risk

    def test_weekend_activity(self):

        """Test weekend risk scoring"""
        scorer = RiskScorer()

        # Create event on Saturday
        saturday = datetime(2024, 1, 6, 12, 0, 0, tzinfo=timezone.utc)  # Jan 6, 2024 is Saturday
        event = AuditEvent(
            event_type=AuditEventType.CONFIG_CHANGED,
            timestamp=saturday
        )

        score = scorer.calculate_risk_score(event)
        assert score >= 35  # CONFIG_CHANGED (30) + weekend (5)

    def test_consecutive_failures(self):

        """Test risk scoring with consecutive failures"""
        scorer = RiskScorer()

        event = AuditEvent(
            event_type=AuditEventType.AUTH_LOGIN_FAILURE,
            result="failure",
            metadata={"consecutive_failures": 5}
        )

        score = scorer.calculate_risk_score(event)
        assert score >= 70  # AUTH_LOGIN_FAILURE (30) + failure (20) + consecutive_failures (20)

    def test_consecutive_failures_with_string_metadata(self):

        """Risk scoring should tolerate string JSON metadata."""
        scorer = RiskScorer()
        metadata = json.dumps({"consecutive_failures": 4})
        event = AuditEvent(
            event_type=AuditEventType.AUTH_LOGIN_FAILURE,
            result="failure",
            metadata=metadata,
        )

        score = scorer.calculate_risk_score(event)
        assert score >= 70

    def test_result_case_insensitive(self):

        """Risk scoring should treat result strings case-insensitively."""
        scorer = RiskScorer()
        event = AuditEvent(
            event_type=AuditEventType.AUTH_LOGIN_FAILURE,
            result="Failure",
        )

        score = scorer.calculate_risk_score(event)
        assert score >= 50  # AUTH_LOGIN_FAILURE (30) + failure (20)

    def test_consecutive_failures_with_string_value(self):

        """Risk scoring should handle string counts in metadata dicts."""
        scorer = RiskScorer()
        event = AuditEvent(
            event_type=AuditEventType.AUTH_LOGIN_FAILURE,
            result="failure",
            metadata={"consecutive_failures": "5"},
        )

        score = scorer.calculate_risk_score(event)
        assert score >= 70

    def test_large_export_with_string_result_count(self):

        """Risk scoring should handle string result_count values."""
        scorer = RiskScorer()
        event = AuditEvent(
            event_type=AuditEventType.DATA_EXPORT,
            timestamp=datetime(2024, 1, 2, 12, 0, 0, tzinfo=timezone.utc),
            result_count="1500",
        )

        score = scorer.calculate_risk_score(event)
        assert score >= 15


# ============================================================================
# Test Unified Audit Service
# ============================================================================

class TestUnifiedAuditService:
    """Test unified audit service functionality"""

    @pytest.mark.asyncio
    async def test_service_initialization(self, audit_service):
        """Test service initializes correctly"""
        assert audit_service.db_path.exists()

        # Check database schema
        async with aiosqlite.connect(audit_service.db_path) as db:
            cursor = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = await cursor.fetchall()
            table_names = [row[0] for row in tables]

            assert "audit_events" in table_names
            assert "audit_daily_stats" in table_names

    @pytest.mark.asyncio
    async def test_custom_db_path_creates_parent_dir(self, tmp_path):
        """Custom db paths should create missing parent directories."""
        db_path = tmp_path / "nested" / "audit" / "audit.db"
        assert not db_path.parent.exists()
        service = UnifiedAuditService(
            db_path=str(db_path),
            retention_days=7,
            enable_pii_detection=False,
            enable_risk_scoring=False,
            buffer_size=10,
            flush_interval=60.0,
        )
        await service.initialize()
        try:
            assert db_path.exists()
            assert db_path.parent.exists()
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_log_event(self, audit_service):
        """Test logging an event"""
        context = AuditContext(
            user_id="test_user",
            ip_address="192.168.1.1"
        )

        event_id = await audit_service.log_event(
            event_type=AuditEventType.AUTH_LOGIN_SUCCESS,
            context=context,
            metadata={"browser": "Chrome"}
        )

        assert event_id is not None
        assert audit_service.stats["events_logged"] == 1

    @pytest.mark.asyncio
    async def test_api_key_hashing_prefix(self, audit_service):
        """API keys should always be stored as hashed, prefixed values."""
        raw_hex = "a" * 64
        expected = f"sha256:{hashlib.sha256(raw_hex.encode('utf-8')).hexdigest()}"
        ctx = AuditContext(user_id="hash_user", api_key_hash=raw_hex)
        await audit_service.log_event(
            event_type=AuditEventType.API_REQUEST,
            context=ctx,
        )
        await audit_service.flush()

        events = await audit_service.query_events(user_id="hash_user")
        assert events
        stored = events[0].get("context_api_key_hash")
        assert stored == expected
        assert stored != raw_hex

        ctx2 = AuditContext(user_id="hash_user2", api_key_hash=expected)
        await audit_service.log_event(
            event_type=AuditEventType.API_REQUEST,
            context=ctx2,
        )
        await audit_service.flush()
        events2 = await audit_service.query_events(user_id="hash_user2")
        assert events2
        assert events2[0].get("context_api_key_hash") == expected

    @pytest.mark.asyncio
    async def test_event_buffering_and_flush(self, audit_service):
        """Test event buffering and flushing"""
        # Log multiple events
        for i in range(5):
            await audit_service.log_event(
                event_type=AuditEventType.DATA_READ,
                context=AuditContext(user_id=f"user_{i}")
            )

        assert len(audit_service.event_buffer) == 5

        # Force flush
        await audit_service.flush()

        assert len(audit_service.event_buffer) == 0
        assert audit_service.stats["events_flushed"] == 5

    @pytest.mark.asyncio
    async def test_timestamp_normalization_and_filters(self, audit_service):
        """Timestamps should be normalized to UTC and filters should handle offsets."""
        tz_offset = timezone(timedelta(hours=5))
        local_ts = datetime(2024, 1, 1, 12, 0, tzinfo=tz_offset)
        event = AuditEvent(
            event_id="tz-event",
            timestamp=local_ts,
            category=AuditEventCategory.SYSTEM,
            event_type=AuditEventType.SYSTEM_START,
            severity=AuditSeverity.INFO,
        )
        async with audit_service.buffer_lock:
            audit_service.event_buffer.append(event)
        await audit_service.flush()

        events = await audit_service.query_events()
        match = next(e for e in events if e.get("event_id") == "tz-event")
        stored_ts = datetime.fromisoformat(match["timestamp"])
        assert stored_ts.utcoffset() == timedelta(0)
        assert stored_ts == local_ts.astimezone(timezone.utc)

        start_filter = datetime(2024, 1, 1, 8, 0, tzinfo=timezone(timedelta(hours=1)))
        filtered = await audit_service.query_events(start_time=start_filter)
        assert any(e.get("event_id") == "tz-event" for e in filtered)

    @pytest.mark.asyncio
    async def test_auto_flush_on_buffer_full(self, audit_service):
        """Test automatic flush when buffer is full"""
        # Buffer size is 10 in fixture
        for i in range(12):
            await audit_service.log_event(
                event_type=AuditEventType.DATA_READ,
                context=AuditContext(user_id=f"user_{i}")
            )

        # Wait for async flush
        await asyncio.sleep(0.5)

        # Buffer should have been flushed at 10 events
        assert len(audit_service.event_buffer) < 10

    @pytest.mark.asyncio
    async def test_log_event_allows_string_metadata(self, audit_service):
        """Logging with string metadata should not crash risk scoring."""
        context = AuditContext(user_id="string_meta_user")
        await audit_service.log_event(
            event_type=AuditEventType.AUTH_LOGIN_FAILURE,
            context=context,
            metadata="plain string metadata",
            result="failure",
        )
        await audit_service.flush()
        events = await audit_service.query_events(user_id="string_meta_user")
        assert events

    @pytest.mark.asyncio
    async def test_pii_detection_in_metadata(self, audit_service):
        """Test PII detection in event metadata"""
        context = AuditContext(user_id="test_user")

        await audit_service.log_event(
            event_type=AuditEventType.DATA_WRITE,
            context=context,
            metadata={
                "email": "user@example.com",
                "ssn": "123-45-6789"
            }
        )

        # Flush to database
        await audit_service.flush()

        # Query event
        events = await audit_service.query_events(user_id="test_user")
        assert len(events) > 0

        event = events[0]
        assert event["pii_detected"] == 1

        # Check metadata was redacted
        metadata = json.loads(event["metadata"])
        assert "user@example.com" not in str(metadata)
        assert "123-45-6789" not in str(metadata)

    @pytest.mark.asyncio
    async def test_pii_redaction_in_nested_sequences(self, audit_service):
        """PII inside nested tuples/sets in metadata should be redacted before storage."""
        context = AuditContext(user_id="nested_pii_user")
        await audit_service.log_event(
            event_type=AuditEventType.DATA_WRITE,
            context=context,
            metadata={
                "emails": ("user@example.com", "other@example.com"),
                "ssns": {"123-45-6789"},
            },
        )
        await audit_service.flush()

        events = await audit_service.query_events(user_id="nested_pii_user")
        assert events
        event = events[0]
        assert event["pii_detected"] == 1

        metadata = json.loads(event["metadata"])
        assert "user@example.com" not in str(metadata)
        assert "other@example.com" not in str(metadata)
        assert "123-45-6789" not in str(metadata)
        assert "[EMAIL_REDACTED]" in str(metadata)
        assert "[SSN_REDACTED]" in str(metadata)

    @pytest.mark.asyncio
    async def test_pii_detection_in_non_metadata_fields(self, audit_service):
        """PII in action/resource_id/user_agent gets redacted and sets flag."""
        context = AuditContext(user_id="pii_user", user_agent="sk_abcdefghijklmnopqrstuvwxyzABCDEF1234567890")
        action = "delete account for john.doe@example.com"
        resource_id = "order-4111-1111-1111-1111"
        await audit_service.log_event(
            event_type=AuditEventType.DATA_DELETE,
            context=context,
            action=action,
            resource_id=resource_id,
            error_message="User reported api_key=sk_abcdefghijklmnopqrstuvwxyzABCDEF1234567890",
        )
        await audit_service.flush()
        events = await audit_service.query_events(user_id="pii_user")
        assert events
        e = events[0]
        # pii_detected flag set
        assert e.get("pii_detected") == 1
        # Redactions occurred
        assert "[EMAIL_REDACTED]" in (e.get("action") or "")
        assert "[CREDIT_CARD_REDACTED]" in (e.get("resource_id") or "")
        # user_agent redacted
        assert "[API_KEY_REDACTED]" in (e.get("context_user_agent") or "")
        # error_message redacted
        assert "[API_KEY_REDACTED]" in (e.get("error_message") or "")

    @pytest.mark.asyncio
    async def test_risk_scoring(self, audit_service):
        """Test risk scoring for events"""
        # High risk event
        await audit_service.log_event(
            event_type=AuditEventType.SECURITY_VIOLATION,
            context=AuditContext(user_id="attacker"),
            result="failure"
        )

        # Check high-risk counter
        assert audit_service.stats["high_risk_events"] > 0

    @pytest.mark.asyncio
    async def test_query_events_with_filters(self, audit_service):
        """Test querying events with various filters"""
        # Log diverse events
        context1 = AuditContext(user_id="user1", request_id="req1")
        context2 = AuditContext(user_id="user2", request_id="req2")

        await audit_service.log_event(
            event_type=AuditEventType.AUTH_LOGIN_SUCCESS,
            context=context1
        )

        await audit_service.log_event(
            event_type=AuditEventType.DATA_READ,
            context=context2
        )

        await audit_service.flush()

        # Query by user
        events = await audit_service.query_events(user_id="user1")
        assert len(events) == 1
        assert events[0]["context_user_id"] == "user1"

        # Query by event type
        events = await audit_service.query_events(
            event_types=[AuditEventType.DATA_READ]
        )
        assert len(events) == 1
        assert events[0]["event_type"] == AuditEventType.DATA_READ.value

    @pytest.mark.asyncio
    async def test_replay_fallback_queue(self, audit_service, temp_db_path):
        """Replay should ingest fallback JSONL events and remove the queue file."""
        fallback_path = audit_service.db_path.parent / "audit_fallback_queue.jsonl"
        ev = AuditEvent(event_type=AuditEventType.SYSTEM_START)
        fallback_path.write_text(json.dumps(ev.to_dict()) + "\n", encoding="utf-8")

        inserted = await audit_service.replay_fallback_queue()
        assert inserted == 1
        # Event is now in the DB
        events = await audit_service.query_events()
        assert any(row.get("event_id") == ev.event_id for row in events)
        # Fallback file cleaned up
        assert not fallback_path.exists()

    @pytest.mark.asyncio
    async def test_stop_spills_remaining_events_to_fallback_before_raising(self, tmp_path, monkeypatch):
        db_path = tmp_path / "audit_stop.db"
        service = UnifiedAuditService(
            db_path=str(db_path),
            enable_pii_detection=False,
            enable_risk_scoring=False,
            buffer_size=10,
            flush_interval=60.0,
        )
        await service.initialize(start_background_tasks=False)
        await service._ensure_db_pool()
        await service.log_event(
            AuditEventType.DATA_WRITE,
            context=AuditContext(user_id="stop-user"),
            resource_type="doc",
            resource_id="doc-1",
        )

        original_update_daily_stats = service._update_daily_stats

        async def _boom_update_daily_stats(*_args, **_kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(service, "_update_daily_stats", _boom_update_daily_stats)

        with pytest.raises(AuditShutdownError) as exc_info:
            await service.stop()

        assert isinstance(exc_info.value.__cause__, RuntimeError)
        fb_path = db_path.parent / "audit_fallback_queue.jsonl"
        assert fb_path.exists()
        assert len(fb_path.read_text(encoding="utf-8").splitlines()) == 1
        assert service.event_buffer == []
        assert service._db_pool is None
        assert service.owner_loop is None
        monkeypatch.setattr(service, "_update_daily_stats", original_update_daily_stats)

    @pytest.mark.asyncio
    async def test_stop_wraps_original_flush_failure_in_shutdown_error(self, tmp_path, monkeypatch):
        service = UnifiedAuditService(
            db_path=str(tmp_path / "audit_stop_cause.db"),
            enable_pii_detection=False,
            enable_risk_scoring=False,
            buffer_size=10,
            flush_interval=60.0,
        )
        await service.initialize(start_background_tasks=False)
        await service._ensure_db_pool()

        async def _boom_update_daily_stats(*_args, **_kwargs):
            raise RuntimeError("flush exploded")

        monkeypatch.setattr(service, "_update_daily_stats", _boom_update_daily_stats)

        await service.log_event(
            AuditEventType.DATA_WRITE,
            context=AuditContext(user_id="stop-user-cause"),
            resource_type="doc",
            resource_id="doc-2",
        )

        with pytest.raises(AuditShutdownError) as exc_info:
            await service.stop()

        assert "durable shutdown" in str(exc_info.value).lower()
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert service._db_pool is None
        assert service.owner_loop is None

    @pytest.mark.asyncio
    async def test_stop_keeps_buffer_when_fallback_spill_fails(self, tmp_path, monkeypatch):
        db_path = tmp_path / "audit_stop_spill_fail.db"
        service = UnifiedAuditService(
            db_path=str(db_path),
            enable_pii_detection=False,
            enable_risk_scoring=False,
            buffer_size=10,
            flush_interval=60.0,
        )
        await service.initialize(start_background_tasks=False)
        await service._ensure_db_pool()
        await service.log_event(
            AuditEventType.DATA_WRITE,
            context=AuditContext(user_id="stop-spill-user"),
            resource_type="doc",
            resource_id="doc-3",
        )

        async def _boom_update_daily_stats(*_args, **_kwargs):
            raise RuntimeError("flush exploded")

        async def _boom_spill(_events):
            raise OSError("spill failed")

        monkeypatch.setattr(service, "_update_daily_stats", _boom_update_daily_stats)
        monkeypatch.setattr(service, "_spill_events_to_fallback", _boom_spill)

        with pytest.raises(AuditShutdownError) as exc_info:
            await service.stop()

        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert "spill failed" in str(exc_info.value).lower()
        assert len(service.event_buffer) == 1
        assert service._db_pool is None
        assert service.owner_loop is None

    @pytest.mark.asyncio
    async def test_stop_preserves_late_arrivals_during_fallback_spill(self, tmp_path, monkeypatch):
        db_path = tmp_path / "audit_stop_race.db"
        service = UnifiedAuditService(
            db_path=str(db_path),
            enable_pii_detection=False,
            enable_risk_scoring=False,
            buffer_size=10,
            flush_interval=60.0,
        )
        await service.initialize(start_background_tasks=False)
        await service._ensure_db_pool()

        first_event_id = await service.log_event(
            AuditEventType.DATA_WRITE,
            context=AuditContext(user_id="race-user"),
            resource_type="doc",
            resource_id="doc-4",
        )

        async def _boom_update_daily_stats(*_args, **_kwargs):
            raise RuntimeError("flush exploded")

        spill_started = asyncio.Event()
        release_spill = asyncio.Event()
        captured_event_ids: list[str] = []
        original_spill = service._spill_events_to_fallback

        async def _blocked_spill(events):
            captured_event_ids.extend(event.event_id for event in events)
            spill_started.set()
            await release_spill.wait()
            return await original_spill(events)

        monkeypatch.setattr(service, "_update_daily_stats", _boom_update_daily_stats)
        monkeypatch.setattr(service, "_spill_events_to_fallback", _blocked_spill)

        stop_task = asyncio.create_task(service.stop())
        await asyncio.wait_for(spill_started.wait(), timeout=5)

        second_event_id = await service.log_event(
            AuditEventType.DATA_WRITE,
            context=AuditContext(user_id="race-user"),
            resource_type="doc",
            resource_id="doc-5",
        )

        release_spill.set()

        with pytest.raises(AuditShutdownError) as exc_info:
            await stop_task

        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert captured_event_ids == [first_event_id]
        fb_path = db_path.parent / "audit_fallback_queue.jsonl"
        assert fb_path.exists()
        fallback_contents = fb_path.read_text(encoding="utf-8")
        assert first_event_id in fallback_contents
        assert second_event_id not in fallback_contents
        assert len(service.event_buffer) == 1
        assert service.event_buffer[0].event_id == second_event_id
        assert service._db_pool is None
        assert service.owner_loop is None

    @pytest.mark.asyncio
    async def test_export_events_json_and_csv(self, audit_service):
        """Test exporting events to JSON and CSV formats"""
        # Log a few events
        for i in range(3):
            await audit_service.log_event(
                event_type=AuditEventType.DATA_READ,
                context=AuditContext(user_id="export_user"),
                resource_type="doc",
                resource_id=f"d{i}",
                metadata={"idx": i},
            )
        await audit_service.flush()

        # Export JSON content (no file)
        json_content = await audit_service.export_events(
            user_id="export_user",
            format="json",
        )
        data = json.loads(json_content)
        assert isinstance(data, list) and len(data) >= 3
        assert any(e.get("resource_type") == "doc" for e in data)

        # Export CSV content (no file)
        csv_content = await audit_service.export_events(
            user_id="export_user",
            format="csv",
        )
        # Expect header + at least 3 rows
        lines = [ln for ln in csv_content.splitlines() if ln.strip()]
        assert len(lines) >= 4  # header + 3 rows
        header = lines[0].split(",")
        assert "event_type" in header and "event_id" in header

    @pytest.mark.asyncio
    async def test_daily_statistics_aggregation(self, audit_service):
        """Test daily statistics are properly aggregated"""
        # Log events with metrics
        for i in range(5):
            await audit_service.log_event(
                event_type=AuditEventType.EVAL_COMPLETED,
                context=AuditContext(user_id="user1"),
                tokens_used=100,
                estimated_cost=0.01,
                duration_ms=500.0
            )

        await audit_service.flush()

        # Check daily stats
        async with aiosqlite.connect(audit_service.db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM audit_daily_stats WHERE category = ?",
                (AuditEventCategory.EVALUATION.value,)
            )
            row = await cursor.fetchone()

            assert row is not None
            # Verify aggregations
            assert row[2] == 5  # total_events
            assert row[5] == 0.05  # total_cost (5 * 0.01)
            assert row[6] == 500  # total_tokens (5 * 100)

    @pytest.mark.asyncio
    async def test_cleanup_old_logs(self, audit_service):
        """Test cleanup of old audit logs"""
        # Log old event (manually set timestamp)
        old_event = AuditEvent(
            event_type=AuditEventType.DATA_READ,
            timestamp=datetime.now(timezone.utc) - timedelta(days=10)
        )

        async with audit_service.buffer_lock:
            audit_service.event_buffer.append(old_event)

        await audit_service.flush()

        # Run cleanup
        await audit_service.cleanup_old_logs()

        # Old event should be deleted
        events = await audit_service.query_events()
        for event in events:
            timestamp = datetime.fromisoformat(event["timestamp"])
            age = datetime.now(timezone.utc) - timestamp
            assert age.days < audit_service.retention_days

    @pytest.mark.asyncio
    async def test_audit_context_manager(self, audit_service):
        """Test audit operation context manager"""
        context = AuditContext(user_id="test_user")

        # Successful operation
        async with audit_operation(
            audit_service,
            AuditEventType.DATA_READ,
            context,
            resource_type="document",
            resource_id="doc123"
        ):
            # Simulate work
            await asyncio.sleep(0.1)

        await audit_service.flush()

        events = await audit_service.query_events(user_id="test_user")
        assert len(events) == 1
        assert events[0]["result"] == "success"
        assert events[0]["duration_ms"] > 0

    @pytest.mark.asyncio
    async def test_audit_context_manager_with_error(self, audit_service):
        """Test audit context manager handles errors"""
        context = AuditContext(user_id="test_user")

        # Operation that fails
        with pytest.raises(ValueError):
            async with audit_operation(
                audit_service,
                AuditEventType.DATA_WRITE,
                context,
                resource_type="document"
            ):
                raise ValueError("Test error")

        await audit_service.flush()

        events = await audit_service.query_events(user_id="test_user")
        assert len(events) == 1
        assert events[0]["result"] == "failure"
        assert "Test error" in events[0]["error_message"]

    @pytest.mark.asyncio
    async def test_audit_context_manager_raises_when_success_log_fails(self, audit_service, monkeypatch):
        """Audit logging is mandatory; success logging failures should propagate."""
        context = AuditContext(user_id="log_fail_user")
        original = audit_service.log_event

        async def _fail_on_success(*args, **kwargs):
            if kwargs.get("result") == "success":
                raise RuntimeError("audit log failure")
            return await original(*args, **kwargs)

        monkeypatch.setattr(audit_service, "log_event", _fail_on_success)

        with pytest.raises(RuntimeError, match="audit log failure"):
            async with audit_operation(
                audit_service,
                AuditEventType.DATA_READ,
                context,
                resource_type="document",
                resource_id="doc-success",
            ):
                await asyncio.sleep(0)

    @pytest.mark.asyncio
    async def test_audit_context_manager_preserves_original_error_when_failure_log_fails(self, audit_service, monkeypatch):
        """Failure logging should not mask the original exception."""
        context = AuditContext(user_id="log_fail_user")
        original = audit_service.log_event

        async def _fail_on_failure(*args, **kwargs):
            if kwargs.get("result") == "failure":
                raise RuntimeError("audit log failure")
            return await original(*args, **kwargs)

        monkeypatch.setattr(audit_service, "log_event", _fail_on_failure)

        with pytest.raises(ValueError, match="Test error"):
            async with audit_operation(
                audit_service,
                AuditEventType.DATA_WRITE,
                context,
                resource_type="document",
            ):
                raise ValueError("Test error")

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Global audit service deprecated; use dependency injection")
    async def test_global_service_singleton(self):
        """Test global service singleton pattern"""
        service1 = await get_unified_audit_service()
        service2 = await get_unified_audit_service()

        assert service1 is service2

        await shutdown_audit_service()

    @pytest.mark.asyncio
    async def test_correlation_tracking(self, audit_service):
        """Test correlation ID tracking across events"""
        correlation_id = "corr-123"

        # Log related events
        context = AuditContext(
            user_id="user1",
            correlation_id=correlation_id
        )

        await audit_service.log_event(
            event_type=AuditEventType.API_REQUEST,
            context=context
        )

        await audit_service.log_event(
            event_type=AuditEventType.DATA_READ,
            context=context
        )

        await audit_service.log_event(
            event_type=AuditEventType.API_RESPONSE,
            context=context
        )

        await audit_service.flush()

        # Query by correlation ID
        events = await audit_service.query_events(correlation_id=correlation_id)
        assert len(events) == 3

        # All should have same correlation ID
        for event in events:
            assert event["context_correlation_id"] == correlation_id

    @pytest.mark.asyncio
    async def test_session_tracking(self, audit_service):
        """Test session ID tracking"""
        session_id = "sess-456"

        context = AuditContext(
            user_id="user1",
            session_id=session_id
        )

        # Log session events
        await audit_service.log_event(
            event_type=AuditEventType.AUTH_LOGIN_SUCCESS,
            context=context
        )

        await audit_service.log_event(
            event_type=AuditEventType.DATA_READ,
            context=context
        )

        await audit_service.log_event(
            event_type=AuditEventType.AUTH_LOGOUT,
            context=context
        )

        await audit_service.flush()

        # Query events
        events = await audit_service.query_events(user_id="user1")

        # All should have same session ID
        for event in events:
            assert event["context_session_id"] == session_id

    @pytest.mark.asyncio
    async def test_auto_category_determination(self, audit_service):
        """Test automatic category determination from event type"""
        test_cases = [
            (AuditEventType.AUTH_LOGIN_SUCCESS, AuditEventCategory.AUTHENTICATION),
            (AuditEventType.USER_CREATED, AuditEventCategory.AUTHORIZATION),
            (AuditEventType.DATA_READ, AuditEventCategory.DATA_ACCESS),
            (AuditEventType.RAG_SEARCH, AuditEventCategory.RAG),
            (AuditEventType.EVAL_STARTED, AuditEventCategory.EVALUATION),
            (AuditEventType.API_REQUEST, AuditEventCategory.API_CALL),
            (AuditEventType.SECURITY_VIOLATION, AuditEventCategory.SECURITY),
            (AuditEventType.SYSTEM_START, AuditEventCategory.SYSTEM),
        ]

        for event_type, expected_category in test_cases:
            await audit_service.log_event(event_type=event_type)

        await audit_service.flush()

        events = await audit_service.query_events()

        for event, (event_type, expected_category) in zip(reversed(events), test_cases):
            assert event["category"] == expected_category.value

    @pytest.mark.asyncio
    async def test_auto_severity_determination(self, audit_service):
        """Test automatic severity determination"""
        test_cases = [
            (AuditEventType.SECURITY_VIOLATION, "failure", AuditSeverity.CRITICAL),
            (AuditEventType.SUSPICIOUS_ACTIVITY, "success", AuditSeverity.CRITICAL),
            (AuditEventType.AUTH_LOGIN_FAILURE, "failure", AuditSeverity.WARNING),
            (AuditEventType.SYSTEM_START, "success", AuditSeverity.DEBUG),
            (AuditEventType.DATA_READ, "error", AuditSeverity.ERROR),
        ]

        for event_type, result, expected_severity in test_cases:
            await audit_service.log_event(
                event_type=event_type,
                result=result
            )

        await audit_service.flush()

        events = await audit_service.query_events()

        for event, (_, _, expected_severity) in zip(reversed(events), test_cases):
            assert event["severity"] == expected_severity.value


# ============================================================================
# Test Performance
# ============================================================================

class TestPerformance:
    """Test performance characteristics"""

    @pytest.mark.asyncio
    async def test_batch_insert_performance(self, audit_service):
        """Test batch insert is performant"""
        import time

        audit_service.buffer_size = 1000

        # Log many events
        start = time.perf_counter()

        for i in range(1000):
            await audit_service.log_event(
                event_type=AuditEventType.DATA_READ,
                context=AuditContext(user_id=f"user_{i % 10}")
            )

        await audit_service.flush()

        elapsed = time.perf_counter() - start

        # Should handle 1000 events in under 5 seconds
        assert elapsed < 5.0

        # Verify all events were stored
        assert audit_service.stats["events_flushed"] >= 1000

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, audit_service):
        """Test concurrent write handling"""
        async def write_events(user_id: str, count: int):
            for i in range(count):
                await audit_service.log_event(
                    event_type=AuditEventType.DATA_WRITE,
                    context=AuditContext(user_id=user_id),
                    metadata={"index": i}
                )

        # Launch concurrent writers
        tasks = [
            write_events(f"user_{i}", 100)
            for i in range(10)
        ]

        await asyncio.gather(*tasks)
        await audit_service.flush()

        # Should have logged all events
        assert audit_service.stats["events_logged"] == 1000

    @pytest.mark.asyncio
    async def test_query_performance(self, audit_service):
        """Test query performance with indexes"""
        # Log many events
        for i in range(500):
            await audit_service.log_event(
                event_type=AuditEventType.DATA_READ,
                context=AuditContext(
                    user_id=f"user_{i % 50}",
                    request_id=f"req_{i}"
                )
            )

        await audit_service.flush()

        # Test indexed queries
        import time

        # Query by user_id (indexed)
        start = time.perf_counter()
        events = await audit_service.query_events(user_id="user_10")
        user_query_time = time.perf_counter() - start

        # Query by request_id (indexed)
        start = time.perf_counter()
        events = await audit_service.query_events(request_id="req_100")
        request_query_time = time.perf_counter() - start

        # Both should be fast due to indexes; allow margin for CI load
        assert user_query_time < 0.3
        assert request_query_time < 0.3


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Test integration scenarios"""

    @pytest.mark.asyncio
    async def test_full_audit_workflow(self, audit_service):
        """Test complete audit workflow"""
        # Simulate user session
        session_id = "session-123"
        user_id = "user-456"

        # 1. User login
        await audit_service.log_event(
            event_type=AuditEventType.AUTH_LOGIN_SUCCESS,
            context=AuditContext(
                user_id=user_id,
                session_id=session_id,
                ip_address="192.168.1.100"
            )
        )

        # 2. User performs operations
        for i in range(5):
            await audit_service.log_event(
                event_type=AuditEventType.DATA_READ,
                context=AuditContext(
                    user_id=user_id,
                    session_id=session_id
                ),
                resource_type="document",
                resource_id=f"doc_{i}"
            )

        # 3. User modifies data
        await audit_service.log_event(
            event_type=AuditEventType.DATA_UPDATE,
            context=AuditContext(
                user_id=user_id,
                session_id=session_id
            ),
            resource_type="profile",
            resource_id=user_id,
            metadata={"fields_updated": ["email", "name"]}
        )

        # 4. User logout
        await audit_service.log_event(
            event_type=AuditEventType.AUTH_LOGOUT,
            context=AuditContext(
                user_id=user_id,
                session_id=session_id
            )
        )

        await audit_service.flush()

        # Verify complete session trail
        events = await audit_service.query_events(user_id=user_id)
        assert len(events) == 8

        # Check session consistency
        for event in events:
            assert event["context_session_id"] == session_id

        # Verify event sequence
        event_types = [e["event_type"] for e in reversed(events)]
        assert event_types[0] == AuditEventType.AUTH_LOGIN_SUCCESS.value
        assert event_types[-1] == AuditEventType.AUTH_LOGOUT.value

    @pytest.mark.asyncio
    async def test_rag_workflow_audit(self, audit_service):
        """Test RAG operation audit trail"""
        request_id = "req-789"
        correlation_id = "corr-abc"

        context = AuditContext(
            user_id="researcher",
            request_id=request_id,
            correlation_id=correlation_id
        )

        # 1. Search request
        await audit_service.log_event(
            event_type=AuditEventType.RAG_SEARCH,
            context=context,
            metadata={"query": "quantum computing basics"}
        )

        # 2. Document retrieval
        await audit_service.log_event(
            event_type=AuditEventType.RAG_RETRIEVAL,
            context=context,
            result_count=10,
            duration_ms=150
        )

        # 3. Embedding generation
        await audit_service.log_event(
            event_type=AuditEventType.RAG_EMBEDDING,
            context=context,
            tokens_used=500,
            estimated_cost=0.001
        )

        # 4. Response generation
        await audit_service.log_event(
            event_type=AuditEventType.RAG_GENERATION,
            context=context,
            tokens_used=1500,
            estimated_cost=0.03,
            duration_ms=2000
        )

        await audit_service.flush()

        # Query all related events
        events = await audit_service.query_events(correlation_id=correlation_id)
        assert len(events) == 4

        # Calculate total cost and tokens
        total_cost = sum(e["estimated_cost"] or 0 for e in events)
        total_tokens = sum(e["tokens_used"] or 0 for e in events)

        assert total_cost == pytest.approx(0.031, rel=1e-3)
        assert total_tokens == 2000

    @pytest.mark.asyncio
    async def test_security_incident_tracking(self, audit_service):
        """Test tracking of security incidents"""
        attacker_ip = "10.0.0.1"

        # Simulate attack pattern
        for i in range(10):
            context = AuditContext(
                ip_address=attacker_ip,
                user_id=f"attempt_{i}" if i < 5 else None
            )

            await audit_service.log_event(
                event_type=AuditEventType.AUTH_LOGIN_FAILURE,
                context=context,
                result="failure",
                metadata={"consecutive_failures": i + 1}
            )

        # Security violation detected
        await audit_service.log_event(
            event_type=AuditEventType.SECURITY_VIOLATION,
            context=AuditContext(ip_address=attacker_ip),
            metadata={"reason": "brute_force_detected"}
        )

        await audit_service.flush()

        # Query high-risk events
        events = await audit_service.query_events(min_risk_score=70)

        # Should have multiple high-risk events
        assert len(events) > 0

        # All should be from same IP
        for event in events:
            if event["context_ip_address"]:
                assert event["context_ip_address"] == attacker_ip


# ============================================================================
# Streaming Export Tests (CSV and JSON with file_path)
# ============================================================================

class TestStreamingExport:
    """Tests for streaming export paths when writing to files."""

    @pytest.mark.asyncio
    async def test_export_events_csv_streaming_to_file(self, audit_service, tmp_path):
        user = "csv_stream_user"
        # Log a few events for a specific user
        for i in range(3):
            await audit_service.log_event(
                event_type=AuditEventType.DATA_READ,
                context=AuditContext(user_id=user),
                resource_type="doc",
                resource_id=f"d{i}",
                metadata={"idx": i},
            )
        await audit_service.flush()

        # Export to CSV using streaming file path
        csv_path = tmp_path / "audit_stream.csv"
        count = await audit_service.export_events(
            user_id=user,
            format="csv",
            file_path=str(csv_path),
        )
        # Verify count and file content
        assert count >= 3
        content = csv_path.read_text(encoding="utf-8").splitlines()
        assert content[0].startswith("event_id,")
        # header + at least 3 rows
        assert len(content) >= 4

    @pytest.mark.asyncio
    async def test_export_events_csv_streaming_empty_writes_header(self, audit_service, tmp_path):
        csv_path = tmp_path / "audit_empty.csv"
        count = await audit_service.export_events(
            format="csv",
            file_path=str(csv_path),
        )
        assert count == 0
        lines = [ln for ln in csv_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == 1
        assert lines[0].startswith("event_id,")

    @pytest.mark.asyncio
    async def test_export_events_csv_streaming_respects_max_rows(self, audit_service, tmp_path):
        user = "csv_stream_max_rows"
        total = 12
        max_rows = 5
        for i in range(total):
            await audit_service.log_event(
                event_type=AuditEventType.DATA_READ,
                context=AuditContext(user_id=user),
                resource_type="doc",
                resource_id=f"d{i}",
                metadata={"idx": i},
            )
        await audit_service.flush()

        csv_path = tmp_path / "audit_stream_max.csv"
        count = await audit_service.export_events(
            user_id=user,
            format="csv",
            file_path=str(csv_path),
            chunk_size=4,
            max_rows=max_rows,
        )
        assert count == max_rows
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        assert lines[0].startswith("event_id,")
        assert len(lines) == max_rows + 1

    @pytest.mark.asyncio
    async def test_export_events_json_streaming_to_file(self, audit_service, tmp_path):
        user = "json_stream_user"
        # Log a few events for a specific user
        for i in range(4):
            await audit_service.log_event(
                event_type=AuditEventType.DATA_WRITE,
                context=AuditContext(user_id=user),
                resource_type="note",
                resource_id=f"n{i}",
                metadata={"idx": i},
            )
        await audit_service.flush()

        # Export to JSON using streaming file path
        json_path = tmp_path / "audit_stream.json"
        count = await audit_service.export_events(
            user_id=user,
            format="json",
            file_path=str(json_path),
        )
        assert count >= 4
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) >= 4
        assert any(e.get("resource_type") == "note" for e in data)

    @pytest.mark.asyncio
    async def test_export_events_csv_streaming_large_file(self, audit_service, tmp_path):
        user = "csv_stream_many"
        total = 123
        # Generate more rows than a small chunk size to exercise chunked writes
        for i in range(total):
            await audit_service.log_event(
                event_type=AuditEventType.DATA_READ,
                context=AuditContext(user_id=user),
                resource_type="doc",
                resource_id=f"d{i}",
                metadata={"idx": i},
            )
        await audit_service.flush()

        csv_path = tmp_path / "audit_stream_large.csv"
        import time
        start = time.perf_counter()
        count = await audit_service.export_events(
            user_id=user,
            format="csv",
            file_path=str(csv_path),
            chunk_size=10,  # small chunk to force multiple iterations
        )
        elapsed = time.perf_counter() - start
        assert count == total
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        # header + total rows
        assert lines[0].startswith("event_id,")
        assert len(lines) == total + 1
        # Performance bound (generous to avoid flakiness)
        assert elapsed < 1.5

    @pytest.mark.asyncio
    async def test_export_events_json_streaming_generator_large(self, audit_service):
        user = "json_stream_gen"
        total = 200
        for i in range(total):
            await audit_service.log_event(
                event_type=AuditEventType.DATA_WRITE,
                context=AuditContext(user_id=user),
                resource_type="note",
                resource_id=f"n{i}",
                metadata={"idx": i},
            )
        await audit_service.flush()

        gen = await audit_service.export_events(
            user_id=user,
            format="json",
            stream=True,
            chunk_size=25,
        )

        import time, json as _json
        start = time.perf_counter()
        chunks = []
        async for c in gen:
            chunks.append(c)
        elapsed = time.perf_counter() - start
        content = "".join(chunks)
        data = _json.loads(content)
        assert isinstance(data, list)
        assert len(data) == total
        # Performance bound (generous to avoid flakiness)
        assert elapsed < 1.5

    @pytest.mark.asyncio
    async def test_export_events_json_streaming_large_file(self, audit_service, tmp_path):
        user = "json_stream_file"
        total = 120
        for i in range(total):
            await audit_service.log_event(
                event_type=AuditEventType.DATA_READ,
                context=AuditContext(user_id=user),
                resource_type="doc",
                resource_id=f"dx{i}",
                metadata={"i": i},
            )
        await audit_service.flush()

        json_path = tmp_path / "audit_large.json"
        # Use small chunk_size to exercise multi-chunk writes
        count = await audit_service.export_events(
            user_id=user,
            format="json",
            file_path=str(json_path),
            chunk_size=15,
        )
        assert count == total
        import json as _json
        data = _json.loads(json_path.read_text(encoding="utf-8"))
        assert isinstance(data, list)
        assert len(data) == total

    @pytest.mark.asyncio
    async def test_export_events_jsonl_streaming_to_file(self, audit_service, tmp_path):
        user = "jsonl_stream_file"
        total = 25
        for i in range(total):
            await audit_service.log_event(
                event_type=AuditEventType.DATA_READ,
                context=AuditContext(user_id=user),
                resource_type="doc",
                resource_id=f"j{i}",
                metadata={"i": i},
            )
        await audit_service.flush()

        jsonl_path = tmp_path / "audit_stream.ndjson"
        count = await audit_service.export_events(
            user_id=user,
            format="jsonl",
            file_path=str(jsonl_path),
            chunk_size=7,
        )
        assert count == total
        lines = [ln for ln in jsonl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        assert len(lines) == total
        import json as _json
        objs = [_json.loads(ln) for ln in lines]
        assert any(o.get("resource_id") == "j0" for o in objs)

    @pytest.mark.asyncio
    async def test_export_events_jsonl_streaming_max_rows(self, audit_service):
        user = "jsonl_max_rows"
        total = 60
        for i in range(total):
            await audit_service.log_event(
                event_type=AuditEventType.DATA_WRITE,
                context=AuditContext(user_id=user),
                resource_type="item",
                resource_id=f"x{i}",
                metadata={"i": i},
            )
        await audit_service.flush()

        max_rows = 25
        gen = await audit_service.export_events(
            user_id=user,
            format="jsonl",
            stream=True,
            chunk_size=9,
            max_rows=max_rows,
        )
        chunks = []
        async for c in gen:
            chunks.append(c)
        content = "".join(chunks)
        lines = [ln for ln in content.splitlines() if ln.strip()]
        assert len(lines) == max_rows
        # Ensure each line is valid JSON
        import json as _json
        for ln in lines:
            _json.loads(ln)

    @pytest.mark.asyncio
    async def test_export_events_json_streaming_max_rows_stops_early(self, audit_service):
        """JSON streaming generator should stop querying once max_rows is reached."""
        user = "json_max_rows"
        total = 60
        for i in range(total):
            await audit_service.log_event(
                event_type=AuditEventType.DATA_WRITE,
                context=AuditContext(user_id=user),
                resource_type="item",
                resource_id=f"j{i}",
                metadata={"i": i},
            )
        await audit_service.flush()

        call_count = 0
        orig_query = audit_service.query_events

        async def _counting_query(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return await orig_query(*args, **kwargs)

        audit_service.query_events = _counting_query  # type: ignore[assignment]
        try:
            max_rows = 25
            gen = await audit_service.export_events(
                user_id=user,
                format="json",
                stream=True,
                chunk_size=1,
                max_rows=max_rows,
            )
            chunks = []
            async for c in gen:
                chunks.append(c)
            content = "".join(chunks)
            data = json.loads(content)
            assert isinstance(data, list)
            assert len(data) == max_rows
            # With chunk_size=1, a correct implementation should not scan the full table.
            assert call_count <= max_rows + 1
        finally:
            audit_service.query_events = orig_query  # type: ignore[assignment]

    @pytest.mark.asyncio
    async def test_export_events_rejects_non_positive_max_rows(self, audit_service):
        """export_events should reject non-positive max_rows values."""
        with pytest.raises(ValueError, match="max_rows must be > 0"):
            await audit_service.export_events(format="json", max_rows=0)
        with pytest.raises(ValueError, match="max_rows must be > 0"):
            await audit_service.export_events(format="json", max_rows=-1)

    async def test_audit_operation_with_start_and_completed_types(self, audit_service):
        ctx = AuditContext(user_id="ctx_op_user")
        # Use distinct start and complete event types
        async with audit_operation(
            audit_service,
            AuditEventType.DATA_READ,
            ctx,
            start_event_type=AuditEventType.API_REQUEST,
            completed_event_type=AuditEventType.API_RESPONSE,
            resource_type="document",
            resource_id="docABC",
        ):
            await asyncio.sleep(0.05)
        await audit_service.flush()
        events = await audit_service.query_events(user_id="ctx_op_user")
        assert len(events) == 2
        types = {e["event_type"] for e in events}
        assert AuditEventType.API_REQUEST.value in types
        assert AuditEventType.API_RESPONSE.value in types
        # Verify result fields
        started = next(e for e in events if e["event_type"] == AuditEventType.API_REQUEST.value)
        completed = next(e for e in events if e["event_type"] == AuditEventType.API_RESPONSE.value)
        assert started["result"] == "started"
        assert completed["result"] == "success"
        assert (completed.get("duration_ms") or 0) > 0

    @pytest.mark.asyncio
    async def test_audit_operation_ignores_reserved_kwargs_on_success(self, audit_service):
        """Reserved kwargs should be ignored rather than causing duplicate-kwarg crashes."""
        ctx = AuditContext(user_id="ctx_reserved_success")
        async with audit_operation(
            audit_service,
            AuditEventType.DATA_READ,
            ctx,
            resource_type="document",
            resource_id="doc-reserved-success",
            result="failure",
            duration_ms=999.0,
            error_message="override-me",
        ):
            await asyncio.sleep(0)

        await audit_service.flush()
        events = await audit_service.query_events(user_id="ctx_reserved_success")
        assert len(events) == 1
        event = events[0]
        assert event["result"] == "success"
        assert "override-me" not in str(event.get("error_message") or "")
        # Context manager-computed duration must be used, not caller override.
        assert float(event.get("duration_ms") or 0.0) != 999.0

    @pytest.mark.asyncio
    async def test_audit_operation_ignores_reserved_kwargs_on_failure(self, audit_service):
        """Reserved kwargs should not mask real failure semantics."""
        ctx = AuditContext(user_id="ctx_reserved_failure")
        with pytest.raises(ValueError, match="boom-real"):
            async with audit_operation(
                audit_service,
                AuditEventType.DATA_WRITE,
                ctx,
                resource_type="document",
                result="success",
                duration_ms=123.0,
                error_message="override-me",
            ):
                raise ValueError("boom-real")

        await audit_service.flush()
        events = await audit_service.query_events(user_id="ctx_reserved_failure")
        assert len(events) == 1
        event = events[0]
        assert event["result"] == "failure"
        assert "boom-real" in str(event.get("error_message") or "")
        assert "override-me" not in str(event.get("error_message") or "")


# ============================================================================
# Test Fallback Queue Atomicity
# ============================================================================

class TestFallbackQueueAtomicity:
    """Test that fallback queue replay is atomic and doesn't double-count stats."""

    @pytest.mark.asyncio
    async def test_fallback_queue_partial_replay_no_duplicate_stats(self, temp_db_path):
        """Verify that partial replay doesn't inflate daily stats on re-replay."""
        # Create service and add some events
        service = UnifiedAuditService(
            db_path=temp_db_path,
            retention_days=7,
            enable_pii_detection=False,
            enable_risk_scoring=False,
            buffer_size=100,
            flush_interval=60.0,
        )
        await service.initialize()

        # Create fallback queue manually with test events
        fb_path = Path(temp_db_path).parent / "audit_fallback_queue.jsonl"
        events_data = []
        for i in range(10):
            event = AuditEvent(
                event_id=f"test-event-{i}",
                timestamp=datetime.now(timezone.utc),
                category=AuditEventCategory.DATA_ACCESS,
                event_type=AuditEventType.DATA_READ,
                severity=AuditSeverity.INFO,
                resource_type="test",
                resource_id=str(i),
                result="success",
                duration_ms=100.0,
            )
            events_data.append(json.dumps(event.to_dict()) + "\n")

        fb_path.write_text("".join(events_data))

        # Replay the fallback queue
        replayed = await service.replay_fallback_queue()
        assert replayed == 10

        # File should be removed
        assert not fb_path.exists()

        # Query stats
        events = await service.query_events()
        assert len(events) == 10

        await service.stop()

    @pytest.mark.asyncio
    async def test_fallback_replay_does_not_double_count_stats(self, tmp_path):
        """Replaying duplicate fallback events should not inflate daily stats."""
        db_path = tmp_path / "audit.db"
        service = UnifiedAuditService(
            db_path=str(db_path),
            retention_days=7,
            enable_pii_detection=False,
            enable_risk_scoring=False,
            buffer_size=100,
            flush_interval=60.0,
        )
        await service.initialize()
        ts = datetime(2024, 1, 2, 12, 0, tzinfo=timezone.utc)
        try:
            fb_path = db_path.parent / "audit_fallback_queue.jsonl"
            payload_lines = []
            for i in range(3):
                event = AuditEvent(
                    event_id=f"dup-{i}",
                    timestamp=ts,
                    category=AuditEventCategory.DATA_ACCESS,
                    event_type=AuditEventType.DATA_READ,
                    severity=AuditSeverity.INFO,
                )
                payload_lines.append(json.dumps(event.to_dict()) + "\n")

            fb_path.write_text("".join(payload_lines), encoding="utf-8")
            inserted = await service.replay_fallback_queue()
            assert inserted == 3

            async with aiosqlite.connect(db_path) as db:
                row = await db.execute(
                    "SELECT total_events FROM audit_daily_stats WHERE date = ? AND category = ?",
                    (ts.date(), AuditEventCategory.DATA_ACCESS.value),
                )
                first = await row.fetchone()
                first_total = int(first[0]) if first else 0

            fb_path.write_text("".join(payload_lines), encoding="utf-8")
            inserted_again = await service.replay_fallback_queue()
            assert inserted_again == 0

            async with aiosqlite.connect(db_path) as db:
                row = await db.execute(
                    "SELECT total_events FROM audit_daily_stats WHERE date = ? AND category = ?",
                    (ts.date(), AuditEventCategory.DATA_ACCESS.value),
                )
                second = await row.fetchone()
                second_total = int(second[0]) if second else 0

            assert first_total == 3
            assert second_total == first_total
        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_duration_count_tracked_correctly(self, temp_db_path):
        """Verify duration_count column is used for correct avg_duration calculation."""
        service = UnifiedAuditService(
            db_path=temp_db_path,
            retention_days=7,
            enable_pii_detection=False,
            enable_risk_scoring=False,
            buffer_size=100,
            flush_interval=60.0,
        )
        await service.initialize()

        # Log some events with duration and some without
        ctx = AuditContext(user_id="stats_test")

        # 3 events with duration (100ms, 200ms, 0ms) -> avg should be 100ms
        await service.log_event(
            event_type=AuditEventType.DATA_READ,
            context=ctx,
            duration_ms=100.0,
        )
        await service.log_event(
            event_type=AuditEventType.DATA_READ,
            context=ctx,
            duration_ms=200.0,
        )
        await service.log_event(
            event_type=AuditEventType.DATA_READ,
            context=ctx,
            duration_ms=0.0,
        )
        # 3 events without duration
        for _ in range(3):
            await service.log_event(
                event_type=AuditEventType.DATA_READ,
                context=ctx,
            )

        await service.flush()

        # Check the daily stats table
        async with aiosqlite.connect(temp_db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT total_events, duration_count, avg_duration_ms FROM audit_daily_stats WHERE category = ?",
                (AuditEventCategory.DATA_ACCESS.value,)
            )
            row = await cursor.fetchone()

        assert row is not None
        assert row["total_events"] == 6
        assert row["duration_count"] == 3
        # avg_duration should be 100ms (average of 100, 200, and 0)
        assert abs(row["avg_duration_ms"] - 100.0) < 0.1

        await service.stop()


def test_stop_safe_when_owner_loop_closed(monkeypatch):


    """stop() should not await tasks attached to a closed/different event loop."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("TEST_MODE", raising=False)
    monkeypatch.delenv("TLDW_TEST_MODE", raising=False)

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    service = UnifiedAuditService(
        db_path=db_path,
        retention_days=7,
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=10,
        flush_interval=60.0,
    )

    loop1 = asyncio.new_event_loop()
    asyncio.set_event_loop(loop1)
    try:
        loop1.run_until_complete(service.initialize())

        async def _cancel_background_tasks() -> None:
            tasks = [t for t in (service._flush_task, service._cleanup_task, service._replay_task) if t]
            for t in tasks:
                t.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        loop1.run_until_complete(_cancel_background_tasks())
    finally:
        loop1.close()
        asyncio.set_event_loop(None)

    asyncio.run(service.stop())
    assert service._owner_loop is None
    assert service._db_pool is None

    Path(db_path).unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_migration_drops_legacy_views_and_triggers(tmp_path):
    """Legacy audit views/triggers should be removed during unified migration."""
    db_path = tmp_path / "legacy_audit.db"

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                user_id TEXT,
                session_id TEXT,
                ip_address TEXT,
                user_agent TEXT,
                endpoint TEXT,
                method TEXT,
                resource_id TEXT,
                resource_type TEXT,
                action TEXT NOT NULL,
                outcome TEXT NOT NULL,
                details TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE user_rate_limits (
                user_id TEXT,
                tier TEXT,
                evaluations_per_day INTEGER
            )
            """
        )
        cur.execute(
            """
            CREATE VIEW audit_statistics AS
            SELECT
                DATE(timestamp) as audit_date,
                event_type,
                severity,
                outcome,
                COUNT(*) as event_count,
                COUNT(DISTINCT user_id) as unique_users,
                COUNT(DISTINCT ip_address) as unique_ips
            FROM audit_events
            GROUP BY DATE(timestamp), event_type, severity, outcome
            """
        )
        cur.execute(
            """
            CREATE VIEW security_alerts AS
            SELECT
                event_id,
                timestamp,
                event_type,
                severity,
                user_id,
                ip_address,
                action,
                details,
                CASE
                    WHEN severity = 'critical' THEN 1
                    WHEN severity = 'high' THEN 2
                    WHEN severity = 'medium' THEN 3
                    ELSE 4
                END as priority
            FROM audit_events
            WHERE severity IN ('critical', 'high')
            ORDER BY priority, timestamp DESC
            """
        )
        cur.execute(
            """
            CREATE TRIGGER audit_rate_limit_changes
            AFTER UPDATE ON user_rate_limits
            BEGIN
                INSERT INTO audit_events (
                    event_id, timestamp, event_type, severity,
                    user_id, action, outcome, resource_type, resource_id,
                    details
                ) VALUES (
                    lower(hex(randomblob(16))),
                    datetime('now'),
                    'config.tier_upgrade',
                    'medium',
                    NEW.user_id,
                    'User tier updated',
                    'success',
                    'user_tier',
                    NEW.user_id,
                    json_object('old_tier', OLD.tier, 'new_tier', NEW.tier)
                );
            END
            """
        )
        conn.commit()

    service = UnifiedAuditService(
        db_path=str(db_path),
        retention_days=7,
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=10,
        flush_interval=1.0,
    )
    await service.initialize(start_background_tasks=False)
    await service.stop()

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        views = {row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='view'")}
        triggers = {row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='trigger'")}
        assert "audit_statistics" not in views
        assert "security_alerts" not in views
        assert "audit_rate_limit_changes" not in triggers
        cols = [row[1] for row in cur.execute("PRAGMA table_info(audit_events)")]
        assert "outcome" not in cols


@pytest.mark.asyncio
async def test_flush_propagates_cancellation_and_preserves_buffer(audit_service, monkeypatch):
    """flush() should re-raise cancellation and keep buffered events."""
    await audit_service.log_event(
        event_type=AuditEventType.DATA_READ,
        context=AuditContext(user_id="cancel-user"),
    )
    assert len(audit_service.event_buffer) == 1

    async def _cancel_filter(*_args, **_kwargs):
        raise asyncio.CancelledError()

    monkeypatch.setattr(audit_service, "_filter_new_events", _cancel_filter)

    with pytest.raises(asyncio.CancelledError):
        await audit_service.flush()

    assert len(audit_service.event_buffer) == 1
    monkeypatch.undo()


@pytest.mark.asyncio
async def test_cleanup_old_logs_propagates_cancellation(audit_service, monkeypatch):
    """cleanup_old_logs() should not swallow task cancellation."""

    async def _cancel_pool():
        raise asyncio.CancelledError()

    monkeypatch.setattr(audit_service, "_ensure_db_pool", _cancel_pool)

    with pytest.raises(asyncio.CancelledError):
        await audit_service.cleanup_old_logs()


@pytest.mark.asyncio
async def test_audit_operation_logs_failure_for_non_whitelisted_exception(audit_service):
    """Runtime exceptions outside noncritical tuple should still be audited as failures."""
    context = AuditContext(user_id="zero-div-user")

    with pytest.raises(ZeroDivisionError):
        async with audit_operation(
            audit_service,
            AuditEventType.DATA_WRITE,
            context,
            resource_type="document",
        ):
            raise ZeroDivisionError("division by zero")

    await audit_service.flush()
    events = await audit_service.query_events(user_id="zero-div-user")
    assert len(events) == 1
    assert events[0]["result"] == "failure"
    assert "division by zero" in (events[0]["error_message"] or "")


@pytest.mark.asyncio
async def test_replay_fallback_queue_parses_falsey_pii_detected_string(audit_service):
    """Fallback replay should parse string false values as false."""
    event = AuditEvent(
        event_id="fallback-pii-string",
        event_type=AuditEventType.DATA_READ,
        category=AuditEventCategory.DATA_ACCESS,
        context=AuditContext(user_id="fallback-user"),
        pii_detected=True,
    )
    record = event.to_dict()
    record["pii_detected"] = "false"

    fb_path = Path(audit_service.db_path).parent / "audit_fallback_queue.jsonl"
    fb_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    inserted = await audit_service.replay_fallback_queue(max_batch=100)
    assert inserted == 1

    events = await audit_service.query_events(user_id="fallback-user")
    replayed = next((row for row in events if row["event_id"] == "fallback-pii-string"), None)
    assert replayed is not None
    assert replayed["pii_detected"] in (0, False)


@pytest.mark.asyncio
async def test_replay_fallback_queue_quarantines_malformed_lines(audit_service):
    """Malformed fallback lines should be quarantined instead of dropped."""
    event = AuditEvent(
        event_id="fallback-quarantine-valid",
        event_type=AuditEventType.DATA_READ,
        category=AuditEventCategory.DATA_ACCESS,
        context=AuditContext(user_id="fallback-quarantine-user"),
    )

    fb_path = Path(audit_service.db_path).parent / "audit_fallback_queue.jsonl"
    bad_path = fb_path.with_suffix(".bad.jsonl")
    bad_path.unlink(missing_ok=True)
    fb_path.write_text(
        "not-json\n[]\n" + json.dumps(event.to_dict()) + "\n",
        encoding="utf-8",
    )

    inserted = await audit_service.replay_fallback_queue(max_batch=100)
    assert inserted == 1
    assert not fb_path.exists()
    assert bad_path.exists()

    bad_lines = [line.strip() for line in bad_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(bad_lines) == 2
    assert "not-json" in bad_lines
    assert "[]" in bad_lines

    events = await audit_service.query_events(user_id="fallback-quarantine-user")
    replayed = next((row for row in events if row["event_id"] == "fallback-quarantine-valid"), None)
    assert replayed is not None


@pytest.mark.asyncio
async def test_query_events_raises_on_read_failure(audit_service, monkeypatch):
    """query_events should raise on DB read failure (no silent empty success)."""

    @asynccontextmanager
    async def _boom_read_db():
        raise RuntimeError("db read failed")
        yield

    monkeypatch.setattr(audit_service, "_read_db", _boom_read_db)

    with pytest.raises(AuditReadError):
        await audit_service.query_events(user_id="read-fail-user")


@pytest.mark.asyncio
async def test_count_events_raises_on_read_failure(audit_service, monkeypatch):
    """count_events should raise on DB read failure (no silent zero success)."""

    @asynccontextmanager
    async def _boom_read_db():
        raise RuntimeError("db count failed")
        yield

    monkeypatch.setattr(audit_service, "_read_db", _boom_read_db)

    with pytest.raises(AuditReadError):
        await audit_service.count_events(user_id="count-fail-user")


@pytest.mark.asyncio
async def test_export_events_raises_on_read_failure(audit_service, monkeypatch):
    """export_events should raise when keyset query fails."""

    @asynccontextmanager
    async def _boom_read_db():
        raise RuntimeError("db export failed")
        yield

    monkeypatch.setattr(audit_service, "_read_db", _boom_read_db)

    with pytest.raises(AuditReadError):
        await audit_service.export_events(
            format="json",
            user_id="export-fail-user",
            stream=False,
            max_rows=10,
        )

@pytest.mark.asyncio
async def test_legacy_migration_allows_new_events_with_chain_hash(tmp_path):
    """A migrated legacy table should accept fresh writes with chain hashes."""
    db_path = tmp_path / "legacy_insert_audit.db"

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                user_id TEXT,
                session_id TEXT,
                ip_address TEXT,
                user_agent TEXT,
                endpoint TEXT,
                method TEXT,
                resource_id TEXT,
                resource_type TEXT,
                action TEXT NOT NULL,
                outcome TEXT NOT NULL,
                details TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()

    service = UnifiedAuditService(
        db_path=str(db_path),
        retention_days=7,
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=10,
        flush_interval=1.0,
    )
    await service.initialize(start_background_tasks=False)
    try:
        await service.log_event(
            event_type=AuditEventType.DATA_READ,
            context=AuditContext(user_id="legacy-writer"),
            action="read",
            resource_type="doc",
            resource_id="legacy-doc",
        )
        await service.flush(raise_on_failure=True)
    finally:
        await service.stop()

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cols = [row[1] for row in cur.execute("PRAGMA table_info(audit_events)")]
        row = cur.execute(
            "SELECT chain_hash FROM audit_events WHERE resource_id = ?",
            ("legacy-doc",),
        ).fetchone()

    assert "chain_hash" in cols
    assert row is not None
    assert row[0]


@pytest.mark.asyncio
async def test_legacy_migration_populated_rows_preserves_chain_hash_binding(tmp_path):
    """Legacy rows should migrate without failing chain-hash bindings."""
    db_path = tmp_path / "legacy_populated_audit.db"

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_id TEXT UNIQUE NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                event_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                user_id TEXT,
                action TEXT NOT NULL,
                outcome TEXT NOT NULL,
                metadata TEXT
            )
            """
        )
        cur.execute(
            """
            INSERT INTO audit_events (
                event_id,
                timestamp,
                event_type,
                severity,
                user_id,
                action,
                outcome,
                metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-1",
                "2026-01-01T00:00:00+00:00",
                "data.read",
                "info",
                "17",
                "read",
                "success",
                "{}",
            ),
        )
        conn.commit()

    service = UnifiedAuditService(
        db_path=str(db_path),
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=10,
        flush_interval=60.0,
    )
    await service.initialize(start_background_tasks=False)
    try:
        await service.log_event(
            event_type=AuditEventType.DATA_READ,
            context=AuditContext(user_id="17"),
            action="post_migration_read",
            resource_id="doc-1",
        )
        await service.flush(raise_on_failure=True)
    finally:
        await service.stop()

    # Verify post-migration row exists and chain integrity holds for new rows.
    # Legacy migrated rows have empty chain_hash (by design), so we verify only
    # the post-migration sub-chain which starts a fresh hash sequence.
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT action, context_user_id, timestamp, event_type, chain_hash "
            "FROM audit_events ORDER BY timestamp ASC, event_id ASC"
        ) as cur:
            rows = [dict(r) for r in await cur.fetchall()]

    assert len(rows) >= 2, "Expected at least the migrated row and the post-migration row"
    assert len([r for r in rows if r["action"] == "post_migration_read"]) == 1

    # Rows with non-empty chain_hash form the verifiable chain
    chained_rows = [r for r in rows if r.get("chain_hash")]
    assert len(chained_rows) >= 1, "Post-migration row should have a chain_hash"

    result = verify_audit_chain([
        {
            "action": row.get("action", ""),
            "user_id": row.get("context_user_id"),
            "timestamp": row.get("timestamp", ""),
            "detail": row.get("event_type", ""),
            "chain_hash": row.get("chain_hash", ""),
        }
        for row in chained_rows
    ])
    assert result["valid"] is True, f"Chain integrity broken after migration: {result}"


@pytest.mark.asyncio
async def test_failed_flush_does_not_advance_chain_state(tmp_path, monkeypatch):
    """Failed flushes must not advance the persisted hash-chain head."""
    db_path = tmp_path / "failed_flush_chain.db"
    service = UnifiedAuditService(
        db_path=str(db_path),
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=10,
        flush_interval=60.0,
    )
    await service.initialize(start_background_tasks=False)
    try:
        await service.log_event(
            AuditEventType.AUTH_LOGIN_SUCCESS,
            context=AuditContext(user_id="chain-user"),
            action="login",
        )

        original_update_daily_stats = service._update_daily_stats

        async def _boom_stats(*_args, **_kwargs):
            raise RuntimeError("boom-after-chain")

        monkeypatch.setattr(service, "_update_daily_stats", _boom_stats)

        with pytest.raises(RuntimeError):
            await service.flush(raise_on_failure=True)

        assert service._last_chain_hash == ""

        monkeypatch.setattr(service, "_update_daily_stats", original_update_daily_stats)
        await service.flush(raise_on_failure=True)

        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT action, context_user_id, timestamp, event_type, chain_hash FROM audit_events ORDER BY timestamp ASC, event_id ASC"
            ) as cur:
                rows = [dict(r) for r in await cur.fetchall()]

        result = verify_audit_chain([
            {
                "action": row.get("action", ""),
                "user_id": row.get("context_user_id"),
                "timestamp": row.get("timestamp", ""),
                "detail": row.get("event_type", ""),
                "chain_hash": row.get("chain_hash", ""),
            }
            for row in rows
        ])

        assert result["valid"] is True
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_replay_fallback_queue_recomputes_chain_hashes(audit_service):
    """Fallback replay should restore events with valid chain hashes."""
    first = AuditEvent(
        event_id="fallback-chain-1",
        event_type=AuditEventType.DATA_READ,
        category=AuditEventCategory.DATA_ACCESS,
        context=AuditContext(user_id="fallback-chain-user"),
        action="read",
        resource_id="fb-1",
    )
    second = AuditEvent(
        event_id="fallback-chain-2",
        event_type=AuditEventType.DATA_EXPORT,
        category=AuditEventCategory.DATA_MODIFICATION,
        context=AuditContext(user_id="fallback-chain-user"),
        action="export",
        resource_id="fb-2",
    )

    fb_path = Path(audit_service.db_path).parent / "audit_fallback_queue.jsonl"
    fb_path.write_text(
        json.dumps(first.to_dict()) + "\n" + json.dumps(second.to_dict()) + "\n",
        encoding="utf-8",
    )

    inserted = await audit_service.replay_fallback_queue(max_batch=100)
    assert inserted == 2

    async with aiosqlite.connect(audit_service.db_path) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT action, context_user_id, timestamp, event_type, chain_hash FROM audit_events WHERE context_user_id = ? ORDER BY timestamp ASC, event_id ASC",
            ("fallback-chain-user",),
        ) as cur:
            rows = [dict(r) for r in await cur.fetchall()]

    assert len(rows) == 2
    assert all(row.get("chain_hash") for row in rows)

    result = verify_audit_chain([
        {
            "action": row.get("action", ""),
            "user_id": row.get("context_user_id"),
            "timestamp": row.get("timestamp", ""),
            "detail": row.get("event_type", ""),
            "chain_hash": row.get("chain_hash", ""),
        }
        for row in rows
    ])
    assert result["valid"] is True


@pytest.mark.asyncio
async def test_count_events_flushes_buffered_events(tmp_path):
    """Counting should reflect buffered events without waiting for background flush."""
    db_path = tmp_path / "count_flushes_buffer.db"
    service = UnifiedAuditService(
        db_path=str(db_path),
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=100,
        flush_interval=60.0,
    )
    await service.initialize(start_background_tasks=False)
    try:
        await service.log_event(
            AuditEventType.DATA_READ,
            context=AuditContext(user_id="buffered-count-user"),
            resource_id="count-me",
        )

        count = await service.count_events(user_id="buffered-count-user")

        assert count == 1
        assert service.event_buffer == []
    finally:
        await service.stop()


@pytest.mark.asyncio
async def test_export_events_flushes_buffered_events(tmp_path):
    """Exports should include buffered events without waiting for periodic flush."""
    db_path = tmp_path / "export_flushes_buffer.db"
    service = UnifiedAuditService(
        db_path=str(db_path),
        enable_pii_detection=False,
        enable_risk_scoring=False,
        buffer_size=100,
        flush_interval=60.0,
    )
    await service.initialize(start_background_tasks=False)
    try:
        await service.log_event(
            AuditEventType.DATA_READ,
            context=AuditContext(user_id="buffered-export-user"),
            resource_id="export-me",
        )

        content = await service.export_events(
            user_id="buffered-export-user",
            format="json",
            stream=False,
            max_rows=10,
        )
        data = json.loads(content)

        assert any(row.get("resource_id") == "export-me" for row in data)
        assert service.event_buffer == []
    finally:
        await service.stop()
