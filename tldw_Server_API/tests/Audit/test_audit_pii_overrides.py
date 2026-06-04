import json
import tempfile
from pathlib import Path

import pytest


@pytest.mark.asyncio
async def test_audit_pii_overrides_and_scan_fields(monkeypatch):
    # Configure custom PII pattern and an extra scan field
    from tldw_Server_API.app.core.config import settings
    from tldw_Server_API.app.core.Audit import unified_audit_service as audit_mod

    UnifiedAuditService = audit_mod.UnifiedAuditService
    AuditEventType = audit_mod.AuditEventType
    AuditContext = audit_mod.AuditContext

    # Custom pattern to match HELLO followed by 3 digits
    pii_patterns = {"custom": r"HELLO\d{3}"}
    monkeypatch.setitem(settings, "AUDIT_PII_PATTERNS", pii_patterns)
    monkeypatch.setitem(audit_mod._app_settings, "AUDIT_PII_PATTERNS", pii_patterns)
    # Ensure we scan a context field not in defaults to prove override works
    scan_fields = ["context_endpoint"]
    monkeypatch.setitem(settings, "AUDIT_PII_SCAN_FIELDS", scan_fields)
    monkeypatch.setitem(audit_mod._app_settings, "AUDIT_PII_SCAN_FIELDS", scan_fields)

    # Use a temp DB for isolation
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    svc = UnifiedAuditService(
        db_path=db_path,
        enable_pii_detection=True,
        enable_risk_scoring=False,
        buffer_size=10,
        flush_interval=0.1,
    )
    try:
        await svc.initialize()

        # Include the custom PII string both in metadata and in a context field
        ctx = AuditContext(
            user_id="u-ovr",
            endpoint="/v1/foo/HELLO123",
        )

        await svc.log_event(
            event_type=AuditEventType.DATA_WRITE,
            context=ctx,
            metadata={"payload": "prefix-HELLO123-suffix"},
        )
        await svc.flush()

        rows = await svc.query_events(user_id="u-ovr")
        assert rows, "Expected at least one audit row"
        row = rows[0]

        # PII detection flag and compliance flag should be set
        assert row.get("pii_detected") in (True, 1, "1")
        flags = row.get("compliance_flags")
        if isinstance(flags, str):
            try:
                flags = json.loads(flags)
            except Exception:
                _ = None
        assert isinstance(flags, (list, tuple)) and ("pii_detected" in flags)

        # Metadata should be redacted
        meta = row.get("metadata")
        if isinstance(meta, str):
            meta = json.loads(meta)
        # The custom token should be redacted with placeholder
        assert "HELLO123" not in json.dumps(meta)
        assert "[CUSTOM_REDACTED]" in json.dumps(meta)

        # The extra scan field on context (endpoint) should be redacted as well
        assert row.get("context_endpoint") and "HELLO123" not in row.get("context_endpoint")
        assert "[CUSTOM_REDACTED]" in row.get("context_endpoint")
    finally:
        try:
            await svc.stop()
        except Exception:
            _ = None
        try:
            Path(db_path).unlink(missing_ok=True)
        except Exception:
            _ = None
