# Monitoring Safe-First Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the approved safe-first Monitoring defects, lock current lifecycle/seam behavior into regression coverage, align docs with shipped behavior, and capture deferred contract work without changing the external monitoring API contract.

**Architecture:** Keep the remediation local to Monitoring backend paths. Correct the `TopicMonitoringService.reload()` config refresh bug, harden `NotificationService` safe wrappers and document its current generic/digest split, then make the public alert lifecycle and admin overlay seam explicit through tests, lightweight schema/docs clarification, and non-throwing diagnostics. Finish with docs alignment, follow-up issue capture, and the reviewed monitoring/admin/auth verification slices.

**Tech Stack:** Python 3.14, FastAPI, Pydantic v2, SQLite/AuthNZ repos, pytest, tenacity, Bandit

---

## Stages

## Stage 1: Topic Monitoring Correctness
**Goal**: Fix the reload dedupe-config bug and prove the runtime settings refresh correctly.
**Success Criteria**: `reload()` updates `_dedup_window_seconds` and `_simhash_distance` without regressing existing path reload behavior.
**Tests**: `python -m pytest -q tldw_Server_API/tests/Monitoring/test_topic_monitoring.py -k "reload or dedupe"`
**Status**: Not Started

## Stage 2: Notification Guardrails
**Goal**: Harden best-effort notification helpers and lock current generic/digest semantics into tests.
**Success Criteria**: retry exhaustion no longer leaks through safe wrappers, generic notifications stay non-email, digest flush remains clear-count-only, and Guardian batching behavior still passes.
**Tests**: `python -m pytest -q tldw_Server_API/tests/Monitoring/test_notification_service.py tldw_Server_API/tests/Guardian/test_notification_batching.py`
**Status**: Not Started

## Stage 3: Lifecycle And Overlay Contract Coverage
**Goal**: Make the public alert lifecycle and overlay-only admin seam explicit without changing route responses.
**Success Criteria**: minimal mutation responses are described in schema/tests, current read/acknowledge/dismiss state effects are covered, and overlay-only admin actions are explicit plus diagnostic.
**Tests**: `python -m pytest -q tldw_Server_API/tests/Monitoring/test_monitoring_contract_schema.py tldw_Server_API/tests/Admin/test_monitoring_alerts_overlay_integration.py tldw_Server_API/tests/Admin/test_admin_monitoring_api.py tldw_Server_API/tests/Admin/test_admin_monitoring_overlay_diagnostics.py`
**Status**: Not Started

## Stage 4: Documentation And Follow-Up Tracking
**Goal**: Align Monitoring docs with actual shipped behavior and capture deferred contract work in a durable artifact.
**Success Criteria**: product/README language matches runtime behavior and the umbrella follow-up issue content exists remotely or as issue-ready markdown.
**Tests**: `python -m pytest -q tldw_Server_API/tests/Monitoring/test_monitoring_docs_contract.py`
**Status**: Not Started

## Stage 5: Verification And Hardening
**Goal**: Re-run the reviewed monitoring/admin/auth slices, run Bandit on the touched scope, and optionally re-check Postgres parity.
**Success Criteria**: targeted pytest slices pass, Bandit finds no new issues in touched paths, and Postgres parity is either passed or explicitly recorded as skipped due to fixture availability.
**Tests**: targeted pytest slices + `python -m bandit`
**Status**: Not Started

## File Map

- Modify: `tldw_Server_API/app/core/Monitoring/topic_monitoring_service.py`
  - Refresh dedupe window and simhash distance during `reload()`.
- Modify: `tldw_Server_API/app/core/Monitoring/notification_service.py`
  - Swallow retry exhaustion in safe wrappers and tighten helper docstrings around generic/digest behavior.
- Modify: `tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py`
  - Add a read-only alert lookup helper for non-throwing admin overlay diagnostics.
- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/admin_monitoring.py`
  - Add non-breaking overlay identity diagnostics before persisted admin overlay mutations.
- Modify: `tldw_Server_API/app/api/v1/schemas/monitoring_schemas.py`
  - Describe the current minimal mutation response contract directly in the response schema.
- Modify: `tldw_Server_API/tests/Monitoring/test_topic_monitoring.py`
  - Add the reload dedupe regression.
- Modify: `tldw_Server_API/tests/Monitoring/test_notification_service.py`
  - Add retry-exhaustion, generic-path, and digest-flush coverage.
- Create: `tldw_Server_API/tests/Monitoring/test_monitoring_contract_schema.py`
  - Assert the public mutation schema documents the re-read contract.
- Create: `tldw_Server_API/tests/Admin/test_admin_monitoring_overlay_diagnostics.py`
  - Unit-test the new non-throwing overlay diagnostics helper.
- Modify: `tldw_Server_API/tests/Admin/test_admin_monitoring_api.py`
  - Make overlay-only admin behavior explicit at the API layer and isolate the public alerts DB path.
- Modify: `tldw_Server_API/tests/Admin/test_monitoring_alerts_overlay_integration.py`
  - Lock the current read/acknowledge/dismiss effects and minimal responses.
- Create: `tldw_Server_API/tests/Monitoring/test_monitoring_docs_contract.py`
  - Assert docs describe the shipped notification/lifecycle contract.
- Modify: `Docs/Product/Completed/Topic_Monitoring_Watchlists.md`
  - Replace stale “all local/no external calls” and placeholder notification wording with current best-effort behavior.
- Modify: `tldw_Server_API/app/core/Monitoring/README.md`
  - Clarify topic-alert vs generic notification behavior, digest flush semantics, and the shared Guardian dependency.
- Create: `Docs/superpowers/reviews/2026-04-07-monitoring-safe-first-followup-issues.md`
  - Store issue-ready umbrella/subtask content when direct GitHub issue creation is unavailable.

### Task 1: Fix Reload Dedupe Config Refresh

**Files:**
- Modify: `tldw_Server_API/app/core/Monitoring/topic_monitoring_service.py`
- Modify: `tldw_Server_API/tests/Monitoring/test_topic_monitoring.py`

- [ ] **Step 1: Write the failing dedupe-reload regression**

```python
def test_topic_monitoring_reload_refreshes_dedupe_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_file = tmp_path / "alerts.db"
    wl_file = tmp_path / "watchlists.json"
    wl_file.write_text(json.dumps({"watchlists": []}), encoding="utf-8")

    monkeypatch.setenv("MONITORING_ALERTS_DB", str(db_file))
    monkeypatch.setenv("MONITORING_WATCHLISTS_FILE", str(wl_file))
    monkeypatch.setenv("MONITORING_ENABLED", "true")
    monkeypatch.setenv("TOPIC_MONITOR_DEDUP_SECONDS", "300")
    monkeypatch.setenv("TOPIC_MONITOR_SIMHASH_DISTANCE", "3")

    _reset_topic_monitoring_service()
    svc = get_topic_monitoring_service()

    assert svc._dedup_window_seconds == 300
    assert svc._simhash_distance == 3

    monkeypatch.setenv("TOPIC_MONITOR_DEDUP_SECONDS", "30")
    monkeypatch.setenv("TOPIC_MONITOR_SIMHASH_DISTANCE", "1")

    svc.reload()

    assert svc._dedup_window_seconds == 30
    assert svc._simhash_distance == 1
```

- [ ] **Step 2: Run the focused reload tests to verify they fail**

Run:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Monitoring/test_topic_monitoring.py -k "reload_updates_paths or reload_refreshes_dedupe_settings"
```

Expected:

- the existing path reload test still passes
- the new regression fails because `reload()` leaves `_dedup_window_seconds` and `_simhash_distance` at their constructor values

- [ ] **Step 3: Write the minimal implementation**

```python
# tldw_Server_API/app/core/Monitoring/topic_monitoring_service.py
def reload(
    self,
    *,
    delete_missing: bool = False,
    disable_missing: bool = False,
    include_unmanaged: bool = False,
) -> None:
    with self._lock:
        self._config = load_and_log_configs() or {}
        monitoring_cfg = (
            self._config.get("monitoring") if isinstance(self._config, dict) else None
        ) or {}
        self._enabled = self._resolve_enabled(monitoring_cfg)
        self._max_scan_chars = self._resolve_max_scan_chars()
        self._dedup_window_seconds = self._coerce_int(
            os.getenv("TOPIC_MONITOR_DEDUP_SECONDS", monitoring_cfg.get("dedup_seconds", 300)),
            300,
        )
        self._simhash_distance = self._coerce_int(
            os.getenv("TOPIC_MONITOR_SIMHASH_DISTANCE", monitoring_cfg.get("simhash_distance", 3)),
            3,
        )
        self._dedupe_state = {}
        self._dedupe_stream_last_seen = {}
        self._dedupe_last_cleanup = 0.0
        self._watchlists_path, self._db_path = self._resolve_paths(monitoring_cfg)
        self._db = TopicMonitoringDB(db_path=self._db_path)
        self._seed_watchlists_from_file(
            delete_missing=delete_missing,
            disable_missing=disable_missing,
            include_unmanaged=include_unmanaged,
        )
        self._load_watchlists_from_db()
```

- [ ] **Step 4: Run the reload regression slice again**

Run:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Monitoring/test_topic_monitoring.py -k "reload_updates_paths or reload_refreshes_dedupe_settings"
```

Expected: PASS for both reload-focused tests.

- [ ] **Step 5: Commit the bugfix**

```bash
git add \
  tldw_Server_API/app/core/Monitoring/topic_monitoring_service.py \
  tldw_Server_API/tests/Monitoring/test_topic_monitoring.py
git commit -m "fix: refresh monitoring dedupe settings on reload"
```

### Task 2: Harden Notification Safe Wrappers And Lock Current Helper Semantics

**Files:**
- Modify: `tldw_Server_API/app/core/Monitoring/notification_service.py`
- Modify: `tldw_Server_API/tests/Monitoring/test_notification_service.py`
- Test: `tldw_Server_API/tests/Guardian/test_notification_batching.py`

- [ ] **Step 1: Write the failing notification regressions**

```python
from tenacity import Future, RetryError


def _retry_error(message: str) -> RetryError:
    attempt = Future(3)
    attempt.set_exception(RuntimeError(message))
    return RetryError(attempt)


def test_send_webhook_safe_swallows_retry_exhaustion(monkeypatch) -> None:
    svc = NotificationService()
    monkeypatch.setattr(
        svc,
        "_send_webhook",
        lambda payload: (_ for _ in ()).throw(_retry_error("webhook boom")),
    )

    svc._send_webhook_safe({"event": "test"})


def test_send_email_safe_swallows_retry_exhaustion(monkeypatch) -> None:
    svc = NotificationService()
    alert = TopicAlert(
        user_id="u1",
        scope_type="user",
        scope_id="u1",
        source="chat.input",
        watchlist_id="watch-1",
        rule_category="system",
        rule_severity="critical",
        pattern="cpu high",
        text_snippet="CPU at 95%",
    )
    monkeypatch.setattr(
        svc,
        "_send_email",
        lambda payload: (_ for _ in ()).throw(_retry_error("email boom")),
    )

    svc._send_email_safe(alert)


def test_notify_generic_only_schedules_webhook_path(monkeypatch, tmp_path) -> None:
    svc = NotificationService()
    svc.enabled = True
    svc.min_severity = "info"
    svc.file_path = str(tmp_path / "notifications.jsonl")
    svc.webhook_url = "https://example.com/hook"
    svc.email_to = "alerts@example.com"
    svc.smtp_host = "smtp.example.com"
    svc.email_from = "sender@example.com"

    targets: list[object] = []

    class _FakeThread:
        def __init__(self, *, target=None, args=(), daemon=None):  # noqa: ANN001, ANN002
            _ = (args, daemon)
            targets.append(target)

        def start(self) -> None:
            return None

    monkeypatch.setattr(notification_service.threading, "Thread", _FakeThread)

    result = svc.notify_generic({"type": "guardian_alert", "severity": "warning", "user_id": "u1"})

    assert result == "logged"
    assert svc._send_webhook_safe in targets
    assert svc._send_email_safe not in targets


def test_flush_digest_returns_count_without_dispatch(monkeypatch) -> None:
    svc = NotificationService()
    svc.enabled = True
    svc.min_severity = "info"
    svc.digest_mode = "hourly"
    svc.notify_or_batch({"type": "guardian_alert", "severity": "info", "user_id": "u1"})
    svc.notify_or_batch({"type": "guardian_alert", "severity": "info", "user_id": "u1"})
    monkeypatch.setattr(
        svc,
        "notify_generic",
        lambda payload: (_ for _ in ()).throw(AssertionError("flush_digest must not dispatch")),
    )

    assert svc.flush_digest("u1") == 2
    assert svc.get_pending_digest_count("u1") == 0
```

- [ ] **Step 2: Run the focused notification tests and verify the right failure**

Run:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Monitoring/test_notification_service.py \
  tldw_Server_API/tests/Guardian/test_notification_batching.py
```

Expected:

- the new webhook/email safe-wrapper tests fail because `RetryError` is not caught today
- the generic-path and digest-path tests either pass immediately or expose any unintended coupling before code changes
- Guardian batching tests remain green and become the shared-behavior guardrail for the implementation

- [ ] **Step 3: Write the minimal implementation**

```python
# tldw_Server_API/app/core/Monitoring/notification_service.py
from tenacity import RetryError, retry, stop_after_attempt, wait_exponential


def _send_webhook_safe(self, payload: dict[str, Any]) -> None:
    try:
        self._send_webhook(payload)
    except RetryError as exc:
        logger.info("Webhook notify failed after retries: {}", exc)
    except Exception as exc:  # noqa: BLE001 - safe wrapper must stay best-effort
        logger.info("Webhook notify failed: {}", exc)


def flush_digest(self, recipient: str | None = None) -> int:
    """Clear pending digest items and return the number cleared."""
    with self._lock:
        if recipient is not None:
            items = self._pending_digests.pop(recipient, [])
            count = len(items)
        else:
            count = sum(len(v) for v in self._pending_digests.values())
            self._pending_digests.clear()
    return count


def _send_email_safe(self, alert: TopicAlert) -> None:
    try:
        self._send_email(alert)
    except RetryError as exc:
        logger.info("Email notify failed after retries: {}", exc)
    except Exception as exc:  # noqa: BLE001 - safe wrapper must stay best-effort
        logger.info("Email notify failed: {}", exc)
```

- [ ] **Step 4: Re-run the notification slices**

Run:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Monitoring/test_notification_service.py \
  tldw_Server_API/tests/Guardian/test_notification_batching.py
```

Expected:

- all monitoring notification tests pass
- Guardian digest/batching tests still pass unchanged

- [ ] **Step 5: Commit the notification hardening**

```bash
git add \
  tldw_Server_API/app/core/Monitoring/notification_service.py \
  tldw_Server_API/tests/Monitoring/test_notification_service.py
git commit -m "fix: harden monitoring notification safe wrappers"
```

### Task 3: Make The Public Alert Lifecycle Contract Explicit

**Files:**
- Create: `tldw_Server_API/tests/Monitoring/test_monitoring_contract_schema.py`
- Modify: `tldw_Server_API/tests/Admin/test_monitoring_alerts_overlay_integration.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/monitoring_schemas.py`

- [ ] **Step 1: Write the contract-clarification tests**

```python
# tldw_Server_API/tests/Monitoring/test_monitoring_contract_schema.py
from tldw_Server_API.app.api.v1.schemas.monitoring_schemas import MarkReadResponse


def test_mark_read_response_schema_describes_minimal_mutation_contract() -> None:
    schema = MarkReadResponse.model_json_schema()
    status_description = schema["properties"]["status"]["description"].lower()
    id_description = schema["properties"]["id"]["description"].lower()

    assert "minimal mutation acknowledgement" in status_description
    assert "re-list alerts" in status_description
    assert "runtime alert row id" in id_description
```

```python
# tldw_Server_API/tests/Admin/test_monitoring_alerts_overlay_integration.py
read_resp = client.post(f"/api/v1/monitoring/alerts/{alert_id}/read")
assert read_resp.status_code == 200, read_resp.text
assert read_resp.json() == {"status": "ok", "id": alert_id}

acknowledge_resp = client.post(f"/api/v1/monitoring/alerts/{alert_id}/acknowledge")
assert acknowledge_resp.status_code == 200, acknowledge_resp.text
assert acknowledge_resp.json() == {"status": "ok", "id": alert_id}

dismiss_resp = client.delete(f"/api/v1/monitoring/alerts/{alert_id}")
assert dismiss_resp.status_code == 200, dismiss_resp.text
assert dismiss_resp.json() == {"status": "ok", "id": alert_id}

refreshed_item = refreshed_resp.json()["items"][0]
assert refreshed_item["is_read"] is True
assert refreshed_item["read_at"] is not None
assert refreshed_item["acknowledged_at"] is not None
assert refreshed_item["dismissed_at"] is not None
```

- [ ] **Step 2: Run the lifecycle contract slice**

Run:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Monitoring/test_monitoring_contract_schema.py \
  tldw_Server_API/tests/Admin/test_monitoring_alerts_overlay_integration.py
```

Expected:

- the schema test fails because `MarkReadResponse` does not currently describe the minimal-response/re-read contract
- the integration test either passes immediately or exposes any lifecycle drift while you are already in this slice

- [ ] **Step 3: Write the minimal implementation**

```python
# tldw_Server_API/app/api/v1/schemas/monitoring_schemas.py
class MarkReadResponse(BaseModel):
    status: str = Field(
        description="Minimal mutation acknowledgement only; re-list alerts for authoritative merged state."
    )
    id: int = Field(description="Runtime alert row id for the mutated monitoring alert")
```

- [ ] **Step 4: Re-run the lifecycle slice**

Run:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Monitoring/test_monitoring_contract_schema.py \
  tldw_Server_API/tests/Admin/test_monitoring_alerts_overlay_integration.py
```

Expected: PASS with the schema contract explicit and the current lifecycle behavior still preserved.

- [ ] **Step 5: Commit the lifecycle clarification**

```bash
git add \
  tldw_Server_API/app/api/v1/schemas/monitoring_schemas.py \
  tldw_Server_API/tests/Monitoring/test_monitoring_contract_schema.py \
  tldw_Server_API/tests/Admin/test_monitoring_alerts_overlay_integration.py
git commit -m "test: lock monitoring alert lifecycle contract"
```

### Task 4: Add Non-Throwing Overlay Diagnostics And Explicit Overlay-Only Tests

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/admin/admin_monitoring.py`
- Create: `tldw_Server_API/tests/Admin/test_admin_monitoring_overlay_diagnostics.py`
- Modify: `tldw_Server_API/tests/Admin/test_admin_monitoring_api.py`

- [ ] **Step 1: Write the failing overlay-diagnostic tests**

```python
# tldw_Server_API/tests/Admin/test_admin_monitoring_overlay_diagnostics.py
from tldw_Server_API.app.api.v1.endpoints.admin import admin_monitoring as admin_monitoring_mod


class _StubMonitoringDb:
    def __init__(self, row):
        self.row = row
        self.lookups: list[int] = []

    def get_alert(self, alert_id: int):
        self.lookups.append(alert_id)
        return self.row


def test_warn_if_runtime_alert_identity_missing_logs_warning(monkeypatch) -> None:
    db = _StubMonitoringDb(row=None)
    warnings: list[str] = []
    monkeypatch.setattr(admin_monitoring_mod.logger, "warning", lambda message, *args, **kwargs: warnings.append(str(message)))

    admin_monitoring_mod._warn_if_overlay_identity_has_no_runtime_row("alert:77", db)

    assert db.lookups == [77]
    assert any("missing runtime alert" in msg for msg in warnings)


def test_warn_if_overlay_only_identity_logs_info_without_lookup(monkeypatch) -> None:
    db = _StubMonitoringDb(row=None)
    infos: list[str] = []
    monkeypatch.setattr(admin_monitoring_mod.logger, "info", lambda message, *args, **kwargs: infos.append(str(message)))

    admin_monitoring_mod._warn_if_overlay_identity_has_no_runtime_row("fingerprint:abc", db)

    assert db.lookups == []
    assert any("overlay-only identity" in msg for msg in infos)
```

```python
# tldw_Server_API/tests/Admin/test_admin_monitoring_api.py
os.environ["MONITORING_ALERTS_DB"] = str(tmp_path / "monitoring_alerts.db")
monitoring_endpoints._TOPIC_MONITORING_DB = None

public_alerts_resp = client.get("/api/v1/monitoring/alerts")
assert public_alerts_resp.status_code == 200, public_alerts_resp.text
assert all(item["alert_identity"] != "alert:7" for item in public_alerts_resp.json()["items"])

history_resp = client.get(
    "/api/v1/admin/monitoring/alerts/history",
    params={"alert_identity": "alert:7"},
)
assert history_resp.status_code == 200, history_resp.text
assert [item["action"] for item in history_resp.json()["items"][:4]] == [
    "escalated",
    "snoozed",
    "unassigned",
    "assigned",
]
```

- [ ] **Step 2: Run the overlay seam slice and verify the expected failure**

Run:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Admin/test_admin_monitoring_overlay_diagnostics.py \
  tldw_Server_API/tests/Admin/test_admin_monitoring_api.py
```

Expected:

- the new diagnostics tests fail because `_warn_if_overlay_identity_has_no_runtime_row()` and `TopicMonitoringDB.get_alert()` do not exist yet
- the API overlay-only assertions pass or expose any unexpected coupling between admin overlay state and the public alert list

- [ ] **Step 3: Write the minimal implementation**

```python
# tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py
def get_alert(self, alert_id: int) -> dict[str, Any] | None:
    with self._lock:
        conn = self._connect()
        try:
            cur = conn.execute(
                """
                SELECT id, created_at, user_id, scope_type, scope_id, source,
                       watchlist_id, rule_id, rule_category, rule_severity, pattern,
                       source_id, chunk_id, chunk_seq, text_snippet, metadata, is_read, read_at
                FROM topic_alerts
                WHERE id = ?
                """,
                (int(alert_id),),
            )
            row = cur.fetchone()
            return {key: row[key] for key in row.keys()} if row else None
        finally:
            conn.close()
```

```python
# tldw_Server_API/app/api/v1/endpoints/admin/admin_monitoring.py
import os

from tldw_Server_API.app.core.DB_Management.TopicMonitoring_DB import TopicMonitoringDB

_RUNTIME_ALERT_ID_PREFIX = "alert:"


def _warn_if_overlay_identity_has_no_runtime_row(
    alert_identity: str,
    monitoring_db: TopicMonitoringDB,
) -> None:
    if not alert_identity.startswith(_RUNTIME_ALERT_ID_PREFIX):
        logger.info("monitoring admin overlay mutation for overlay-only identity {}", alert_identity)
        return

    raw_alert_id = alert_identity[len(_RUNTIME_ALERT_ID_PREFIX):]
    try:
        alert_id = int(raw_alert_id)
    except ValueError:
        logger.warning("monitoring admin overlay mutation uses malformed runtime alert identity {}", alert_identity)
        return

    if monitoring_db.get_alert(alert_id) is None:
        logger.warning("monitoring admin overlay mutation references missing runtime alert {}", alert_identity)


async def _emit_overlay_identity_diagnostic(alert_identity: str) -> None:
    try:
        db = TopicMonitoringDB(os.getenv("MONITORING_ALERTS_DB", "Databases/monitoring_alerts.db"))
        await asyncio.to_thread(_warn_if_overlay_identity_has_no_runtime_row, alert_identity, db)
    except _MONITORING_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("monitoring overlay diagnostic skipped for {}: {}", alert_identity, exc)
```

```python
# Call this before each persisted overlay mutation in assign_alert/snooze_alert/escalate_alert
await _emit_overlay_identity_diagnostic(alert_identity)
```

- [ ] **Step 4: Re-run the overlay slice**

Run:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Admin/test_admin_monitoring_overlay_diagnostics.py \
  tldw_Server_API/tests/Admin/test_admin_monitoring_api.py \
  tldw_Server_API/tests/Admin/test_admin_monitoring_repo.py
```

Expected: PASS with explicit overlay-only behavior and non-throwing diagnostics in place.

- [ ] **Step 5: Commit the overlay seam work**

```bash
git add \
  tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_monitoring.py \
  tldw_Server_API/tests/Admin/test_admin_monitoring_overlay_diagnostics.py \
  tldw_Server_API/tests/Admin/test_admin_monitoring_api.py
git commit -m "test: make monitoring overlay seam explicit"
```

### Task 5: Align Docs, Capture Follow-Up Issues, And Run Full Verification

**Files:**
- Create: `tldw_Server_API/tests/Monitoring/test_monitoring_docs_contract.py`
- Modify: `Docs/Product/Completed/Topic_Monitoring_Watchlists.md`
- Modify: `tldw_Server_API/app/core/Monitoring/README.md`
- Create: `Docs/superpowers/reviews/2026-04-07-monitoring-safe-first-followup-issues.md`

- [ ] **Step 1: Write the failing docs-contract regression**

```python
from pathlib import Path


def test_monitoring_docs_describe_current_notification_contract() -> None:
    product_doc = Path("Docs/Product/Completed/Topic_Monitoring_Watchlists.md").read_text(encoding="utf-8")
    readme_doc = Path("tldw_Server_API/app/core/Monitoring/README.md").read_text(encoding="utf-8")

    assert "best-effort webhook/email attempts" in product_doc
    assert "generic notifications only use the jsonl sink plus optional webhook dispatch" in readme_doc.lower()
    assert "flush_digest() currently clears buffered items and returns the count only" in readme_doc
    assert "re-list alerts for authoritative merged state" in product_doc
```

- [ ] **Step 2: Run the docs-contract test and confirm it fails**

Run:

```bash
source .venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Monitoring/test_monitoring_docs_contract.py
```

Expected: FAIL because the current docs still describe the notification path as fully local/placeholder and do not spell out the minimal alert mutation contract.

- [ ] **Step 3: Update the docs and write the follow-up issue artifact**

```markdown
<!-- Docs/Product/Completed/Topic_Monitoring_Watchlists.md -->
## Notifications (Phase 1 scaffolding)
- Local JSONL file sink gated by severity threshold.
- Topic-alert notifications may also attempt best-effort webhook/email delivery when configured.
- Generic notifications use the JSONL sink and optional webhook only; they do not send email in the current batch.
- Digest modes buffer items in memory; `flush_digest()` clears buffered entries and returns the count, but does not emit a compiled digest notification in this batch.

## Alert Lifecycle
- `POST /api/v1/monitoring/alerts/{id}/read` and `POST /api/v1/monitoring/alerts/{id}/acknowledge` currently share the same minimal `{status, id}` response contract.
- `DELETE /api/v1/monitoring/alerts/{id}` dismisses the alert and leaves the authoritative merged state to be observed by re-listing alerts.
```

```markdown
<!-- tldw_Server_API/app/core/Monitoring/README.md -->
- NotificationService: JSONL sink for topic alerts with optional best-effort webhook/email attempts.
- `notify_generic()` writes JSONL and optionally starts webhook delivery; it does not send email.
- `notify_or_batch()` batches payloads in `hourly`/`daily` digest modes.
- `flush_digest()` currently clears buffered items and returns the count only; it does not dispatch a compiled digest.
- Guardian dispatch reuses `notify_or_batch()`, so Monitoring helper hardening must preserve current Guardian-facing behavior.
```

```markdown
<!-- Docs/superpowers/reviews/2026-04-07-monitoring-safe-first-followup-issues.md -->
# Monitoring Safe-First Follow-Up Issues

## Umbrella Issue
Title: Monitoring follow-ups: tighten alert identity, lifecycle, and digest semantics

Body:
- Background: the safe-first remediation intentionally preserved current public Monitoring behavior while fixing internal defects and documenting the current contract.
- Goal: track the contract changes that should not ship silently in a compatibility batch.
- Scope:
  - stricter overlay identity validation or a first-class overlay-only contract
  - public alert lifecycle/response redesign
  - real digest delivery semantics
  - admin/public permission-model clarification if needed

## Suggested Subtasks

### Subtask 1
Title: Define the authoritative contract for overlay-only monitoring identities

### Subtask 2
Title: Redesign public monitoring alert mutation responses around merged state

### Subtask 3
Title: Decide whether Monitoring digest mode should send compiled deliveries

### Subtask 4
Title: Clarify long-term Monitoring admin/public permission boundaries
```

If `gh` is installed and authenticated, create the remote umbrella issue from the same artifact:

```bash
gh issue create \
  --title "Monitoring follow-ups: tighten alert identity, lifecycle, and digest semantics" \
  --body-file Docs/superpowers/reviews/2026-04-07-monitoring-safe-first-followup-issues.md
```

If `gh` is unavailable, keep the markdown artifact in-repo and report that fallback explicitly in the execution summary.

- [ ] **Step 4: Re-run docs tests, full monitoring/admin/auth verification, optional Postgres parity, and Bandit**

Run:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Monitoring/test_topic_monitoring.py \
  tldw_Server_API/tests/Monitoring/test_notification_service.py \
  tldw_Server_API/tests/Monitoring/test_notification_endpoint.py \
  tldw_Server_API/tests/Monitoring/test_monitoring_notifications_settings.py \
  tldw_Server_API/tests/Monitoring/test_monitoring_contract_schema.py \
  tldw_Server_API/tests/Monitoring/test_monitoring_docs_contract.py \
  tldw_Server_API/tests/Admin/test_admin_monitoring_api.py \
  tldw_Server_API/tests/Admin/test_admin_monitoring_overlay_diagnostics.py \
  tldw_Server_API/tests/Admin/test_admin_monitoring_repo.py \
  tldw_Server_API/tests/Admin/test_monitoring_alerts_overlay_integration.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_monitoring_permissions_claims.py \
  tldw_Server_API/tests/AuthNZ/unit/test_authnz_monitoring_repo_backend_selection.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_monitoring_repo_sqlite.py \
  tldw_Server_API/tests/Guardian/test_notification_batching.py
```

Optional Postgres parity rerun when the fixture is available:

```bash
source .venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_monitoring_repo_postgres.py \
  tldw_Server_API/tests/AuthNZ/integration/test_authnz_admin_monitoring_repo_postgres.py
```

Run Bandit on the touched backend scope:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Monitoring \
  tldw_Server_API/app/core/DB_Management/TopicMonitoring_DB.py \
  tldw_Server_API/app/api/v1/endpoints/admin/admin_monitoring.py \
  tldw_Server_API/app/api/v1/schemas/monitoring_schemas.py \
  -f json -o /tmp/bandit_monitoring_safe_first.json
```

Expected:

- all targeted monitoring/admin/auth tests pass
- Guardian batching still passes
- Postgres parity either passes or skips only because the fixture is unavailable
- Bandit reports no new actionable findings in the touched scope

- [ ] **Step 5: Commit the docs and verification-backed remediation finish**

```bash
git add \
  Docs/Product/Completed/Topic_Monitoring_Watchlists.md \
  Docs/superpowers/reviews/2026-04-07-monitoring-safe-first-followup-issues.md \
  tldw_Server_API/app/core/Monitoring/README.md \
  tldw_Server_API/tests/Monitoring/test_monitoring_docs_contract.py
git commit -m "docs: align monitoring remediation contract"
```
