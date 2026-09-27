# Claims Jobs Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move Stage 1 Claims rebuild, review-notification delivery, and alert delivery work onto the existing core Jobs module without adding Claims-owned queue or lifecycle mechanics.

**Architecture:** Claims owns domain contracts, payload validation, enqueue helpers, and job handlers. Jobs owns persistence, queues, leases, retries, backoff, status, cancellation, quarantine, and admin controls through `JobManager` and `WorkerSDK`. Existing Claims synchronous business logic is refactored into callable seams so the legacy bounded paths and the new Jobs handlers reuse the same delivery/rebuild behavior.

**Tech Stack:** Python, FastAPI service layer, Loguru, SQLite/PostgreSQL Media DB helpers, core Jobs `JobManager`, core Jobs `WorkerSDK`, pytest, Bandit.

---

Backlog task for this plan: `TASK-9936`.

Design spec: `Docs/superpowers/specs/2026-06-24-claims-jobs-operational-control-plane-design.md`.

Before implementing this plan, create or reuse a separate Backlog task for the code work. Keep `TASK-9936` for the plan artifact only.

## File Structure

Create:

- `tldw_Server_API/app/core/Claims_Extraction/claims_job_contracts.py`  
  Owns Claims Jobs constants, payload validation, ID-only result helpers, and `ClaimsJobError` with `WorkerSDK`-compatible attributes.

- `tldw_Server_API/app/core/Claims_Extraction/claims_jobs.py`  
  Owns configuration reads, queue resolution, max-retry reads, idempotency-key builders, `JobManager.create_job(...)` calls, and the read-only Claims Jobs dashboard summary.

- `tldw_Server_API/app/core/Claims_Extraction/claims_job_handlers.py`  
  Owns dispatch for one acquired Jobs row and calls existing Claims rebuild/notification/alert business logic.

- `tldw_Server_API/app/services/claims_jobs_worker.py`  
  Owns `WorkerConfig`, `WorkerSDK` startup, stop-event bridging, and lifecycle worker spec registration.

- `tldw_Server_API/tests/Claims/test_claims_jobs_contracts.py`
- `tldw_Server_API/tests/Claims/test_claims_jobs_enqueue.py`
- `tldw_Server_API/tests/Claims/test_claims_jobs_handlers.py`
- `tldw_Server_API/tests/Services/test_claims_jobs_worker.py`

Modify:

- `tldw_Server_API/app/core/Claims_Extraction/claims_rebuild_service.py`  
  Extract one-media rebuild business logic into a return-value helper used by both the legacy queue and Jobs handler.

- `tldw_Server_API/app/core/Claims_Extraction/claims_notifications.py`  
  Extract review notification delivery into a synchronous return-value helper used by both the legacy bounded dispatch path and Jobs handler.

- `tldw_Server_API/app/core/Claims_Extraction/claims_service.py`  
  Route rebuild/review/alert enqueue points through Jobs when `CLAIMS_JOBS_ENABLED=true`, keep legacy bounded paths when false, and add a read-only Claims Jobs summary to the dashboard payload.

- `tldw_Server_API/app/core/DB_Management/media_db/runtime/claims_monitoring_event_ops.py`  
  Return the inserted monitoring event and add a `get_claims_monitoring_event(...)` helper so alert Jobs can reload by ID.

- `tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py`  
  Bind the new `get_claims_monitoring_event` helper.

- `tldw_Server_API/app/core/DB_Management/media_db/schema/sqlite_claims_extensions.py`  
  No schema change is planned for Stage 1 because `claims_monitoring_events.delivered_at` already exists and has bootstrap repair coverage.

- `tldw_Server_API/app/services/startup_worker_groups.py`  
  Add the Claims Jobs worker spec provider to the lifecycle catalog.

- Existing tests under `tldw_Server_API/tests/Claims`, `tldw_Server_API/tests/DB_Management`, and `tldw_Server_API/tests/Services` as listed in the task steps.

Do not modify core Jobs queue mechanics, lifecycle state machines, RBAC controls, retry behavior, leases, or admin routes for this Stage 1 slice.

## Stage 1 Scope

In scope:

- `claims_rebuild_media`
- `claims_deliver_review_notification`
- `claims_deliver_alert` for `slack` and `webhook`
- Opt-in routing with `CLAIMS_JOBS_ENABLED`
- Worker startup with `CLAIMS_JOBS_WORKER_ENABLED`
- Dashboard read-only status summary from Jobs
- Existing legacy bounded paths when Jobs mode is disabled

Out of scope:

- New Claims queue-control APIs
- New Claims-owned retry loops, lease loops, or pause/resume/drain behavior
- Alert email delivery as a per-event Jobs handler
- Stage 2 analytics exports, review metrics aggregation, cluster rebuilds
- Stage 3 recurring Scheduler/APScheduler orchestration changes

## Task 1: Contracts And Payload Validation

**Files:**
- Create: `tldw_Server_API/app/core/Claims_Extraction/claims_job_contracts.py`
- Test: `tldw_Server_API/tests/Claims/test_claims_jobs_contracts.py`

- [ ] **Step 1: Write failing contract tests**

Add this test file:

```python
import pytest

from tldw_Server_API.app.core.Claims_Extraction import claims_job_contracts as contracts


pytestmark = pytest.mark.unit


def test_rebuild_payload_validation_accepts_id_only_payload() -> None:
    payload = contracts.validate_rebuild_media_payload(
        {"version": 1, "owner_user_id": "1", "media_id": 42}
    )

    assert payload == {"version": 1, "owner_user_id": "1", "media_id": 42}


def test_payload_validation_rejects_paths_and_synthetic_owner() -> None:
    with pytest.raises(contracts.ClaimsJobError) as excinfo:
        contracts.validate_rebuild_media_payload(
            {
                "version": 1,
                "owner_user_id": "0",
                "media_id": 42,
                "db_path": "/tmp/Media_DB_v2.db",
            }
        )

    exc = excinfo.value
    assert exc.retryable is False
    assert exc.failure_code == "claims_invalid_payload"


def test_review_payload_sorts_and_dedupes_notification_ids() -> None:
    payload = contracts.validate_review_notification_payload(
        {"version": 1, "owner_user_id": "7", "notification_ids": [3, "2", 3]}
    )

    assert payload["notification_ids"] == [2, 3]


def test_alert_payload_rejects_unsupported_channel() -> None:
    with pytest.raises(contracts.ClaimsJobError) as excinfo:
        contracts.validate_alert_delivery_payload(
            {
                "version": 1,
                "owner_user_id": "7",
                "event_id": 55,
                "alert_id": 9,
                "channel": "email",
            }
        )

    assert excinfo.value.failure_code == "claims_unsupported_channel"


def test_claims_job_error_exposes_worker_sdk_attributes() -> None:
    exc = contracts.ClaimsJobError(
        "locked",
        retryable=True,
        failure_code="claims_db_locked",
        backoff_seconds=13,
    )

    assert str(exc) == "locked"
    assert exc.retryable is True
    assert exc.failure_code == "claims_db_locked"
    assert exc.backoff_seconds == 13
```

- [ ] **Step 2: Run contract tests and verify they fail for missing module**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Claims/test_claims_jobs_contracts.py -q
```

Expected: fail during import because `claims_job_contracts.py` does not exist yet.

- [ ] **Step 3: Add the contracts module**

Create `claims_job_contracts.py` with this shape:

```python
from __future__ import annotations

import json
from typing import Any

CLAIMS_JOBS_DOMAIN = "claims"
CLAIMS_JOBS_DEFAULT_QUEUE = "default"

CLAIMS_REBUILD_MEDIA_JOB_TYPE = "claims_rebuild_media"
CLAIMS_DELIVER_REVIEW_NOTIFICATION_JOB_TYPE = "claims_deliver_review_notification"
CLAIMS_DELIVER_ALERT_JOB_TYPE = "claims_deliver_alert"

CLAIMS_JOB_PAYLOAD_VERSION = 1
CLAIMS_ALERT_JOB_CHANNELS = {"slack", "webhook"}
SENSITIVE_PAYLOAD_KEYS = {
    "db_path",
    "path",
    "webhook_url",
    "slack_webhook_url",
    "email_recipients",
    "recipient",
    "recipients",
    "claim_text",
    "notification_body",
    "alert_payload",
    "api_key",
    "secret",
    "token",
}


class ClaimsJobError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        failure_code: str = "claims_job_failed",
        backoff_seconds: int | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = bool(retryable)
        self.failure_code = str(failure_code)
        if backoff_seconds is not None:
            self.backoff_seconds = int(backoff_seconds)


def _normalize_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ClaimsJobError(
                "claims job payload is not valid JSON",
                retryable=False,
                failure_code="claims_invalid_payload",
            ) from exc
        if isinstance(parsed, dict):
            return dict(parsed)
    raise ClaimsJobError(
        "claims job payload must be an object",
        retryable=False,
        failure_code="claims_invalid_payload",
    )


def _reject_sensitive_keys(payload: dict[str, Any]) -> None:
    present = sorted(SENSITIVE_PAYLOAD_KEYS.intersection(payload))
    if present:
        raise ClaimsJobError(
            f"claims job payload contains disallowed keys: {', '.join(present)}",
            retryable=False,
            failure_code="claims_invalid_payload",
        )


def _owner_user_id(value: Any) -> str:
    owner = str(value or "").strip()
    if not owner or owner == "0":
        raise ClaimsJobError(
            "claims job payload missing real owner_user_id",
            retryable=False,
            failure_code="claims_missing_owner",
        )
    return owner


def _positive_int(value: Any, field: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ClaimsJobError(
            f"claims job payload has invalid {field}",
            retryable=False,
            failure_code="claims_invalid_payload",
        ) from exc
    if parsed <= 0:
        raise ClaimsJobError(
            f"claims job payload has invalid {field}",
            retryable=False,
            failure_code="claims_invalid_payload",
        )
    return parsed


def _version(payload: dict[str, Any]) -> int:
    version = _positive_int(payload.get("version"), "version")
    if version != CLAIMS_JOB_PAYLOAD_VERSION:
        raise ClaimsJobError(
            "unsupported claims job payload version",
            retryable=False,
            failure_code="claims_unsupported_payload_version",
        )
    return version


def validate_rebuild_media_payload(value: Any) -> dict[str, Any]:
    payload = _normalize_dict(value)
    _reject_sensitive_keys(payload)
    return {
        "version": _version(payload),
        "owner_user_id": _owner_user_id(payload.get("owner_user_id")),
        "media_id": _positive_int(payload.get("media_id"), "media_id"),
    }


def validate_review_notification_payload(value: Any) -> dict[str, Any]:
    payload = _normalize_dict(value)
    _reject_sensitive_keys(payload)
    raw_ids = payload.get("notification_ids")
    if not isinstance(raw_ids, list):
        raise ClaimsJobError(
            "claims review notification payload requires notification_ids",
            retryable=False,
            failure_code="claims_invalid_payload",
        )
    ids = sorted({_positive_int(item, "notification_id") for item in raw_ids})
    if not ids:
        raise ClaimsJobError(
            "claims review notification payload requires notification_ids",
            retryable=False,
            failure_code="claims_invalid_payload",
        )
    return {
        "version": _version(payload),
        "owner_user_id": _owner_user_id(payload.get("owner_user_id")),
        "notification_ids": ids,
    }


def validate_alert_delivery_payload(value: Any) -> dict[str, Any]:
    payload = _normalize_dict(value)
    _reject_sensitive_keys(payload)
    channel = str(payload.get("channel") or "").strip().lower()
    if channel not in CLAIMS_ALERT_JOB_CHANNELS:
        raise ClaimsJobError(
            "unsupported claims alert channel",
            retryable=False,
            failure_code="claims_unsupported_channel",
        )
    return {
        "version": _version(payload),
        "owner_user_id": _owner_user_id(payload.get("owner_user_id")),
        "event_id": _positive_int(payload.get("event_id"), "event_id"),
        "alert_id": _positive_int(payload.get("alert_id"), "alert_id"),
        "channel": channel,
    }


def skipped_result(reason: str, **extra: Any) -> dict[str, Any]:
    return {"outcome": "skipped", "reason": str(reason), **extra}


def ok_result(**extra: Any) -> dict[str, Any]:
    return {"outcome": "ok", **extra}
```

- [ ] **Step 4: Run contract tests and verify they pass**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Claims/test_claims_jobs_contracts.py -q
```

Expected: all tests in `test_claims_jobs_contracts.py` pass.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add tldw_Server_API/app/core/Claims_Extraction/claims_job_contracts.py tldw_Server_API/tests/Claims/test_claims_jobs_contracts.py
git commit -m "feat: add Claims Jobs contracts"
```

## Task 2: Enqueue Helpers And Read-Only Jobs Summary

**Files:**
- Create: `tldw_Server_API/app/core/Claims_Extraction/claims_jobs.py`
- Test: `tldw_Server_API/tests/Claims/test_claims_jobs_enqueue.py`

- [ ] **Step 1: Write failing enqueue tests**

Add this test file:

```python
import pytest

from tldw_Server_API.app.core.Claims_Extraction import claims_jobs
from tldw_Server_API.app.core.Claims_Extraction.claims_job_contracts import (
    CLAIMS_DELIVER_ALERT_JOB_TYPE,
    CLAIMS_DELIVER_REVIEW_NOTIFICATION_JOB_TYPE,
    CLAIMS_JOBS_DOMAIN,
    CLAIMS_REBUILD_MEDIA_JOB_TYPE,
)


pytestmark = pytest.mark.unit


class FakeJobManager:
    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.status_counts = {"queued": 2, "processing": 1, "failed": 1}

    def create_job(self, **kwargs):
        self.created.append(kwargs)
        return {"id": len(self.created), **kwargs, "status": "queued"}

    def get_job(self, job_id: int):
        return {"id": job_id, **self.created[job_id - 1], "status": "queued"}

    def summarize_by_status(self, **kwargs):
        assert kwargs == {"domain": CLAIMS_JOBS_DOMAIN, "owner_user_id": "1"}
        return dict(self.status_counts)


def test_enqueue_rebuild_media_creates_id_only_jobs_payload(monkeypatch) -> None:
    monkeypatch.setitem(claims_jobs.settings, "CLAIMS_JOBS_QUEUE", "default")
    monkeypatch.setitem(claims_jobs.settings, "CLAIMS_JOBS_MAX_RETRIES_REBUILD", 4)
    fake = FakeJobManager()

    job = claims_jobs.enqueue_claims_rebuild_media(
        media_id=42,
        owner_user_id="1",
        job_manager=fake,
    )

    assert job["id"] == 1
    created = fake.created[0]
    assert created["domain"] == CLAIMS_JOBS_DOMAIN
    assert created["queue"] == "default"
    assert created["job_type"] == CLAIMS_REBUILD_MEDIA_JOB_TYPE
    assert created["owner_user_id"] == "1"
    assert created["payload"] == {"version": 1, "owner_user_id": "1", "media_id": 42}
    assert "db_path" not in created["payload"]
    assert created["idempotency_key"] == "claims:rebuild:1:42"
    assert created["max_retries"] == 4


def test_enqueue_review_notification_uses_sorted_idempotency(monkeypatch) -> None:
    monkeypatch.setitem(claims_jobs.settings, "CLAIMS_JOBS_QUEUE", "default")
    fake = FakeJobManager()

    claims_jobs.enqueue_claims_review_notification(
        owner_user_id="1",
        notification_ids=[9, 3, 9],
        job_manager=fake,
    )

    created = fake.created[0]
    assert created["job_type"] == CLAIMS_DELIVER_REVIEW_NOTIFICATION_JOB_TYPE
    assert created["payload"]["notification_ids"] == [3, 9]
    assert created["idempotency_key"].startswith("claims:notify_review:1:")


def test_enqueue_alert_delivery_rejects_email_channel() -> None:
    fake = FakeJobManager()

    with pytest.raises(Exception):
        claims_jobs.enqueue_claims_alert_delivery(
            owner_user_id="1",
            event_id=10,
            alert_id=5,
            channel="email",
            job_manager=fake,
        )

    assert fake.created == []


def test_claims_jobs_summary_is_read_only() -> None:
    fake = FakeJobManager()

    summary = claims_jobs.claims_jobs_summary(job_manager=fake, owner_user_id="1")

    assert summary == {
        "domain": CLAIMS_JOBS_DOMAIN,
        "counts": {"queued": 2, "processing": 1, "failed": 1},
    }
```

- [ ] **Step 2: Run enqueue tests and verify they fail for missing module**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Claims/test_claims_jobs_enqueue.py -q
```

Expected: fail during import because `claims_jobs.py` does not exist yet.

- [ ] **Step 3: Add enqueue helpers**

Create `claims_jobs.py` with this shape:

```python
from __future__ import annotations

import hashlib
from typing import Any, Mapping

from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.worker_utils import coerce_int, jobs_manager_from_env

from .claims_job_contracts import (
    CLAIMS_DELIVER_ALERT_JOB_TYPE,
    CLAIMS_DELIVER_REVIEW_NOTIFICATION_JOB_TYPE,
    CLAIMS_JOB_PAYLOAD_VERSION,
    CLAIMS_JOBS_DEFAULT_QUEUE,
    CLAIMS_JOBS_DOMAIN,
    CLAIMS_REBUILD_MEDIA_JOB_TYPE,
    validate_alert_delivery_payload,
    validate_rebuild_media_payload,
    validate_review_notification_payload,
)


def _settings_map(settings_obj: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
    return settings_obj if settings_obj is not None else settings


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def claims_jobs_enabled(settings_obj: Mapping[str, Any] | None = None) -> bool:
    return _truthy(_settings_map(settings_obj).get("CLAIMS_JOBS_ENABLED", False))


def claims_jobs_worker_enabled(settings_obj: Mapping[str, Any] | None = None) -> bool:
    return _truthy(_settings_map(settings_obj).get("CLAIMS_JOBS_WORKER_ENABLED", False))


def claims_jobs_queue(settings_obj: Mapping[str, Any] | None = None) -> str:
    queue = str(_settings_map(settings_obj).get("CLAIMS_JOBS_QUEUE", CLAIMS_JOBS_DEFAULT_QUEUE)).strip()
    return queue or CLAIMS_JOBS_DEFAULT_QUEUE


def _max_retries(key: str, default: int = 3, settings_obj: Mapping[str, Any] | None = None) -> int:
    return coerce_int(_settings_map(settings_obj).get(key), default)


def _manager(job_manager: JobManager | None = None) -> JobManager:
    return job_manager or jobs_manager_from_env()


def _refresh(manager: JobManager, created: dict[str, Any]) -> dict[str, Any]:
    job_id = created.get("id")
    if job_id is None:
        return created
    return manager.get_job(int(job_id)) or created


def _hash_ids(values: list[int]) -> str:
    joined = ",".join(str(v) for v in sorted(set(values)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def enqueue_claims_rebuild_media(
    *,
    media_id: int,
    owner_user_id: str,
    job_manager: JobManager | None = None,
    settings_obj: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = validate_rebuild_media_payload(
        {
            "version": CLAIMS_JOB_PAYLOAD_VERSION,
            "owner_user_id": owner_user_id,
            "media_id": media_id,
        }
    )
    manager = _manager(job_manager)
    created = manager.create_job(
        domain=CLAIMS_JOBS_DOMAIN,
        queue=claims_jobs_queue(settings_obj),
        job_type=CLAIMS_REBUILD_MEDIA_JOB_TYPE,
        payload=payload,
        owner_user_id=payload["owner_user_id"],
        priority=5,
        max_retries=_max_retries("CLAIMS_JOBS_MAX_RETRIES_REBUILD", 3, settings_obj),
        idempotency_key=f"claims:rebuild:{payload['owner_user_id']}:{payload['media_id']}",
    )
    return _refresh(manager, created)


def enqueue_claims_review_notification(
    *,
    owner_user_id: str,
    notification_ids: list[int],
    job_manager: JobManager | None = None,
    settings_obj: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = validate_review_notification_payload(
        {
            "version": CLAIMS_JOB_PAYLOAD_VERSION,
            "owner_user_id": owner_user_id,
            "notification_ids": notification_ids,
        }
    )
    manager = _manager(job_manager)
    created = manager.create_job(
        domain=CLAIMS_JOBS_DOMAIN,
        queue=claims_jobs_queue(settings_obj),
        job_type=CLAIMS_DELIVER_REVIEW_NOTIFICATION_JOB_TYPE,
        payload=payload,
        owner_user_id=payload["owner_user_id"],
        priority=5,
        max_retries=_max_retries("CLAIMS_JOBS_MAX_RETRIES_REVIEW_NOTIFICATION", 3, settings_obj),
        idempotency_key=f"claims:notify_review:{payload['owner_user_id']}:{_hash_ids(payload['notification_ids'])}",
    )
    return _refresh(manager, created)


def enqueue_claims_alert_delivery(
    *,
    owner_user_id: str,
    event_id: int,
    alert_id: int,
    channel: str,
    job_manager: JobManager | None = None,
    settings_obj: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = validate_alert_delivery_payload(
        {
            "version": CLAIMS_JOB_PAYLOAD_VERSION,
            "owner_user_id": owner_user_id,
            "event_id": event_id,
            "alert_id": alert_id,
            "channel": channel,
        }
    )
    manager = _manager(job_manager)
    created = manager.create_job(
        domain=CLAIMS_JOBS_DOMAIN,
        queue=claims_jobs_queue(settings_obj),
        job_type=CLAIMS_DELIVER_ALERT_JOB_TYPE,
        payload=payload,
        owner_user_id=payload["owner_user_id"],
        priority=5,
        max_retries=_max_retries("CLAIMS_JOBS_MAX_RETRIES_ALERT", 3, settings_obj),
        idempotency_key=(
            f"claims:alert:{payload['owner_user_id']}:{payload['event_id']}:"
            f"{payload['alert_id']}:{payload['channel']}"
        ),
    )
    return _refresh(manager, created)


def claims_jobs_summary(
    *,
    job_manager: JobManager | None = None,
    owner_user_id: str | None = None,
) -> dict[str, Any]:
    manager = _manager(job_manager)
    return {
        "domain": CLAIMS_JOBS_DOMAIN,
        "counts": manager.summarize_by_status(
            domain=CLAIMS_JOBS_DOMAIN,
            owner_user_id=str(owner_user_id) if owner_user_id else None,
        ),
    }
```

- [ ] **Step 4: Run enqueue tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Claims/test_claims_jobs_enqueue.py -q
```

Expected: all tests in `test_claims_jobs_enqueue.py` pass.

- [ ] **Step 5: Commit Task 2**

Run:

```bash
git add tldw_Server_API/app/core/Claims_Extraction/claims_jobs.py tldw_Server_API/tests/Claims/test_claims_jobs_enqueue.py
git commit -m "feat: add Claims Jobs enqueue helpers"
```

## Task 3: Rebuild Business Seam

**Files:**
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_rebuild_service.py`
- Test: `tldw_Server_API/tests/Claims/test_claims_rebuild_service_failure.py`
- Test: `tldw_Server_API/tests/Claims/test_claims_jobs_handlers.py`

- [ ] **Step 1: Write failing tests for a return-value rebuild helper**

Append focused tests to `test_claims_rebuild_service_failure.py`:

```python
def test_rebuild_claims_for_media_returns_skipped_for_missing_media(tmp_path):
    from tldw_Server_API.app.core.Claims_Extraction.claims_rebuild_service import (
        rebuild_claims_for_media,
    )
    from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase

    db_path = tmp_path / "missing-media.db"
    db = MediaDatabase(db_path=str(db_path), client_id="test")
    db.initialize_db()
    db.close_connection()

    result = rebuild_claims_for_media(db_path=str(db_path), media_id=404)

    assert result == {"outcome": "skipped", "reason": "media_missing", "media_id": 404}
```

- [ ] **Step 2: Run the rebuild test and verify it fails for missing helper**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Claims/test_claims_rebuild_service_failure.py::test_rebuild_claims_for_media_returns_skipped_for_missing_media -q
```

Expected: fail because `rebuild_claims_for_media` is not defined.

- [ ] **Step 3: Extract the helper from `_process_task`**

In `claims_rebuild_service.py`, add `rebuild_claims_for_media` above `ClaimsRebuildService`, then make `_process_task` call it:

```python
def rebuild_claims_for_media(*, db_path: str, media_id: int) -> dict[str, Any]:
    media_id = int(media_id)
    with managed_media_database(
        client_id=str(settings.get("SERVER_CLIENT_ID", "SERVER_API_V1")),
        db_path=str(db_path),
        initialize=False,
        suppress_close_exceptions=_CLAIMS_REBUILD_NONCRITICAL_EXCEPTIONS,
    ) as db:
        media = db.get_media_by_id(media_id, include_deleted=False, include_trash=False)
        if not media:
            logger.warning(f"Claims rebuild: media_id={media_id} not found")
            return {"outcome": "skipped", "reason": "media_missing", "media_id": media_id}
        content = media.get("content") or ""
        title = media.get("title") or f"media_{media_id}.txt"
        chunks = chunk_for_embedding(content, file_name=title)
        max_per = int(settings.get("CLAIMS_MAX_PER_CHUNK", 3))
        mode = str(settings.get("CLAIM_EXTRACTOR_MODE", "heuristic"))
        budget = resolve_claims_job_budget(settings=settings)
        claims = extract_claims_for_chunks(
            chunks,
            extractor_mode=mode,
            max_per_chunk=max_per,
            budget=budget,
        )
        if not claims:
            logger.info(f"Claims rebuild: no claims extracted for media_id={media_id}")
            return {"outcome": "skipped", "reason": "no_claims_extracted", "media_id": media_id}
        chunk_text_map: dict[int, str] = {}
        for ch in chunks:
            meta = (ch or {}).get("metadata", {}) or {}
            idx = int(meta.get("chunk_index") or meta.get("index") or 0)
            chunk_text_map[idx] = (ch or {}).get("text") or (ch or {}).get("content") or ""
        with db.transaction():
            deleted = db.soft_delete_claims_for_media(media_id)
            inserted = store_claims(db, media_id=media_id, chunk_texts_by_index=chunk_text_map, claims=claims)
            if inserted <= 0:
                raise RuntimeError(f"Claims rebuild stored zero replacement claims for media_id={media_id}")
        logger.info(f"Claims rebuild: media_id={media_id} deleted={deleted} inserted={inserted}")
        return {
            "outcome": "ok",
            "media_id": media_id,
            "deleted": int(deleted),
            "inserted": int(inserted),
        }
```

Then replace the body of `_process_task` with:

```python
    def _process_task(self, task: ClaimsRebuildTask) -> None:
        rebuild_claims_for_media(db_path=task.db_path, media_id=task.media_id)
```

- [ ] **Step 4: Run rebuild service tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Claims/test_claims_rebuild_service_failure.py -q
```

Expected: all tests in that file pass.

- [ ] **Step 5: Commit Task 3**

Run:

```bash
git add tldw_Server_API/app/core/Claims_Extraction/claims_rebuild_service.py tldw_Server_API/tests/Claims/test_claims_rebuild_service_failure.py
git commit -m "refactor: expose Claims rebuild result helper"
```

## Task 4: Monitoring Event Reload Support For Alert Jobs

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/runtime/claims_monitoring_event_ops.py`
- Modify: `tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py`
- Test: `tldw_Server_API/tests/DB_Management/test_media_db_claims_monitoring_event_ops.py`

- [ ] **Step 1: Write failing DB helper tests**

Append to `test_media_db_claims_monitoring_event_ops.py`:

```python
def test_insert_claims_monitoring_event_returns_inserted_row_and_gets_by_id(tmp_path: Path) -> None:
    db = _make_db(tmp_path, "claims-monitoring-event-get.db")
    try:
        created = db.insert_claims_monitoring_event(
            user_id="1",
            event_type="unsupported_ratio",
            severity="warning",
            payload_json='{"alert_id":9}',
        )

        assert isinstance(created["id"], int)
        loaded = db.get_claims_monitoring_event(int(created["id"]))
        assert loaded["id"] == created["id"]
        assert loaded["user_id"] == "1"
        assert loaded["event_type"] == "unsupported_ratio"
        assert loaded["payload_json"] == '{"alert_id":9}'
    finally:
        db.close_connection()
```

- [ ] **Step 2: Run the DB helper test and verify it fails**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_media_db_claims_monitoring_event_ops.py::test_insert_claims_monitoring_event_returns_inserted_row_and_gets_by_id -q
```

Expected: fail because `insert_claims_monitoring_event` returns `None` and `get_claims_monitoring_event` is not bound.

- [ ] **Step 3: Return inserted event rows and bind `get_claims_monitoring_event`**

In `claims_monitoring_event_ops.py`, update `insert_claims_monitoring_event` to return the inserted row:

```python
def insert_claims_monitoring_event(
    self,
    *,
    user_id: str,
    event_type: str,
    severity: str | None = None,
    payload_json: str | None = None,
) -> dict[str, Any]:
    now = self._get_current_utc_timestamp_str()
    insert_sql = (
        "INSERT INTO claims_monitoring_events "
        "(user_id, event_type, severity, payload_json, created_at, delivered_at) "
        "VALUES (?, ?, ?, ?, ?, ?)"
    )
    if self.backend_type == BackendType.POSTGRESQL:
        insert_sql += " RETURNING id"
    cursor = self.execute_query(
        insert_sql,
        (
            str(user_id),
            str(event_type),
            severity,
            payload_json,
            now,
            None,
        ),
        commit=True,
    )
    if self.backend_type == BackendType.POSTGRESQL:
        row = cursor.fetchone()
        event_id = int(row["id"]) if row else 0
    else:
        event_id = int(getattr(cursor, "lastrowid", 0) or 0)
    return get_claims_monitoring_event(self, event_id) if event_id else {}
```

Also add:

```python
from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
```

Add:

```python
def get_claims_monitoring_event(self, event_id: int) -> dict[str, Any]:
    row = self.execute_query(
        (
            "SELECT id, user_id, event_type, severity, payload_json, created_at, delivered_at "
            "FROM claims_monitoring_events WHERE id = ?"
        ),
        (int(event_id),),
    ).fetchone()
    return dict(row) if row else {}
```

In `media_database_impl.py`, import and bind the helper:

```python
from tldw_Server_API.app.core.DB_Management.media_db.runtime.claims_monitoring_event_ops import (
    get_claims_monitoring_event,
    get_latest_claims_monitoring_event_delivery,
    insert_claims_monitoring_event,
    list_claims_monitoring_events,
    list_undelivered_claims_monitoring_events,
    mark_claims_monitoring_events_delivered,
)

MediaDatabase.get_claims_monitoring_event = get_claims_monitoring_event
```

- [ ] **Step 4: Run DB monitoring event tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/DB_Management/test_media_db_claims_monitoring_event_ops.py -q
```

Expected: all tests in that file pass.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add tldw_Server_API/app/core/DB_Management/media_db/runtime/claims_monitoring_event_ops.py tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py tldw_Server_API/tests/DB_Management/test_media_db_claims_monitoring_event_ops.py
git commit -m "feat: expose Claims monitoring event lookup"
```

## Task 5: Review Notification Delivery Seam

**Files:**
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_notifications.py`
- Test: `tldw_Server_API/tests/Claims/test_claims_review_notifications.py`

- [ ] **Step 1: Write failing tests for synchronous delivery**

Append to `test_claims_review_notifications.py`:

```python
def test_deliver_claim_review_notifications_now_returns_skipped_when_disabled(monkeypatch):
    class _FakeDb:
        def get_claims_monitoring_settings(self, user_id):
            assert user_id == "1"
            return {"enabled": False}

        def close_connection(self) -> None:
            pass

    @contextmanager
    def _fake_managed_media_database(*_args, **_kwargs):
        yield _FakeDb()

    monkeypatch.setattr(claims_notifications, "managed_media_database", _fake_managed_media_database)

    result = claims_notifications.deliver_claim_review_notifications_now(
        db_path="/tmp/claims-review.db",
        owner_user_id="1",
        notification_ids=[7],
    )

    assert result == {"outcome": "skipped", "reason": "settings_disabled", "notification_ids": [7]}
```

- [ ] **Step 2: Run the new notification test and verify it fails**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Claims/test_claims_review_notifications.py::test_deliver_claim_review_notifications_now_returns_skipped_when_disabled -q
```

Expected: fail because `deliver_claim_review_notifications_now` is not defined.

- [ ] **Step 3: Extract synchronous delivery and keep bounded legacy dispatch**

In `claims_notifications.py`, add:

```python
def deliver_claim_review_notifications_now(
    *,
    db_path: str,
    owner_user_id: str,
    notification_ids: list[int],
) -> dict[str, Any]:
    normalized_ids = sorted({int(v) for v in notification_ids if int(v) > 0})
    if not normalized_ids:
        return {"outcome": "skipped", "reason": "no_notification_ids", "notification_ids": []}
    with managed_media_database(
        client_id=str(settings.get("SERVER_CLIENT_ID", "SERVER_API_V1")),
        db_path=str(db_path),
        initialize=False,
        suppress_init_exceptions=_CLAIMS_NOTIFICATION_NONCRITICAL_EXCEPTIONS,
        suppress_close_exceptions=_CLAIMS_NOTIFICATION_NONCRITICAL_EXCEPTIONS,
    ) as db:
        config_row = db.get_claims_monitoring_settings(str(owner_user_id)) or {}
        if config_row and not bool(config_row.get("enabled", True)):
            return {"outcome": "skipped", "reason": "settings_disabled", "notification_ids": normalized_ids}
        channels = _normalize_review_channels(config_row)
        if not any(channels.values()):
            return {"outcome": "skipped", "reason": "no_channels", "notification_ids": normalized_ids}
        rows = db.get_claim_notifications_by_ids(normalized_ids)
        if not rows:
            return {"outcome": "skipped", "reason": "notifications_missing", "notification_ids": normalized_ids}
        notifications = [
            _normalize_notification_row(row)
            for row in rows
            if not row.get("delivered_at")
        ]
        if not notifications:
            return {"outcome": "skipped", "reason": "already_delivered", "notification_ids": normalized_ids}
        payload = _build_review_digest_payload(user_id=str(owner_user_id), notifications=notifications)
        delivered = False
        slack_url = config_row.get("slack_webhook_url")
        webhook_url = config_row.get("webhook_url")
        recipients = _parse_email_recipients(config_row.get("email_recipients"))
        if channels.get("slack") and slack_url:
            delivered = _deliver_review_webhook(
                url=str(slack_url),
                payload={"text": f"Claims review notifications: {len(notifications)} items"},
                channel="slack",
            ) or delivered
        if channels.get("webhook") and webhook_url:
            delivered = _deliver_review_webhook(
                url=str(webhook_url),
                payload=payload,
                channel="webhook",
            ) or delivered
        if channels.get("email") and recipients:
            html_body, text_body = _build_review_email_bodies(notifications)
            delivered = _deliver_review_email_sync(
                recipients=recipients,
                subject=f"Claims review notifications ({len(notifications)})",
                html_body=html_body,
                text_body=text_body,
            ) or delivered
        if not delivered:
            return {"outcome": "failed", "reason": "delivery_failed", "notification_ids": normalized_ids}
        marked = db.mark_claim_notifications_delivered(normalized_ids)
        return {"outcome": "ok", "notification_ids": normalized_ids, "delivered": int(marked)}
```

Then simplify `dispatch_claim_review_notifications` to call the helper inside the existing bounded submission:

```python
def dispatch_claim_review_notifications(
    *,
    db_path: str,
    owner_user_id: str,
    notification_ids: list[int],
) -> None:
    if not notification_ids:
        return

    def _deliver() -> None:
        try:
            deliver_claim_review_notifications_now(
                db_path=db_path,
                owner_user_id=owner_user_id,
                notification_ids=notification_ids,
            )
        except _CLAIMS_NOTIFICATION_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug(f"Claims review notification delivery failed: {exc}")

    submit_claims_notification_delivery(_deliver)
```

- [ ] **Step 4: Run review notification tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Claims/test_claims_review_notifications.py -q
```

Expected: all tests in that file pass.

- [ ] **Step 5: Commit Task 5**

Run:

```bash
git add tldw_Server_API/app/core/Claims_Extraction/claims_notifications.py tldw_Server_API/tests/Claims/test_claims_review_notifications.py
git commit -m "refactor: expose Claims review notification delivery"
```

## Task 6: Claims Job Handlers

**Files:**
- Create: `tldw_Server_API/app/core/Claims_Extraction/claims_job_handlers.py`
- Create: `tldw_Server_API/app/core/Claims_Extraction/claims_alert_delivery.py`
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_service.py`
- Test: `tldw_Server_API/tests/Claims/test_claims_jobs_handlers.py`
- Test: `tldw_Server_API/tests/Claims/test_claims_webhook_delivery.py`

- [ ] **Step 1: Write failing handler tests**

Create `test_claims_jobs_handlers.py`:

```python
import json
from contextlib import contextmanager

import pytest

from tldw_Server_API.app.core.Claims_Extraction import claims_job_handlers
from tldw_Server_API.app.core.Claims_Extraction.claims_job_contracts import (
    CLAIMS_DELIVER_ALERT_JOB_TYPE,
    CLAIMS_DELIVER_REVIEW_NOTIFICATION_JOB_TYPE,
    CLAIMS_REBUILD_MEDIA_JOB_TYPE,
    ClaimsJobError,
)


pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_rebuild_handler_uses_owner_db_path_and_returns_result(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        claims_job_handlers,
        "get_user_media_db_path",
        lambda owner: f"/tmp/user-{owner}/Media_DB_v2.db",
    )
    monkeypatch.setattr(
        claims_job_handlers,
        "rebuild_claims_for_media",
        lambda **kwargs: calls.append(kwargs) or {"outcome": "ok", "media_id": 42, "deleted": 1, "inserted": 2},
    )

    result = await claims_job_handlers.process_claims_job(
        {
            "id": 1,
            "job_type": CLAIMS_REBUILD_MEDIA_JOB_TYPE,
            "owner_user_id": "7",
            "payload": {"version": 1, "owner_user_id": "7", "media_id": 42},
        }
    )

    assert result["outcome"] == "ok"
    assert calls == [{"db_path": "/tmp/user-7/Media_DB_v2.db", "media_id": 42}]


@pytest.mark.asyncio
async def test_handler_rejects_owner_mismatch() -> None:
    with pytest.raises(ClaimsJobError) as excinfo:
        await claims_job_handlers.process_claims_job(
            {
                "id": 1,
                "job_type": CLAIMS_REBUILD_MEDIA_JOB_TYPE,
                "owner_user_id": "8",
                "payload": {"version": 1, "owner_user_id": "7", "media_id": 42},
            }
        )

    assert excinfo.value.retryable is False
    assert excinfo.value.failure_code == "claims_owner_scope_violation"


@pytest.mark.asyncio
async def test_review_notification_delivery_failure_is_retryable(monkeypatch) -> None:
    monkeypatch.setattr(
        claims_job_handlers,
        "get_user_media_db_path",
        lambda owner: f"/tmp/user-{owner}/Media_DB_v2.db",
    )
    monkeypatch.setattr(
        claims_job_handlers,
        "deliver_claim_review_notifications_now",
        lambda **_kwargs: {"outcome": "failed", "reason": "delivery_failed"},
    )

    with pytest.raises(ClaimsJobError) as excinfo:
        await claims_job_handlers.process_claims_job(
            {
                "id": 1,
                "job_type": CLAIMS_DELIVER_REVIEW_NOTIFICATION_JOB_TYPE,
                "owner_user_id": "7",
                "payload": {"version": 1, "owner_user_id": "7", "notification_ids": [5]},
            }
        )

    assert excinfo.value.retryable is True
    assert excinfo.value.failure_code == "claims_review_notification_delivery_failed"


@pytest.mark.asyncio
async def test_alert_delivery_uses_existing_db_and_preserves_slack_payload(monkeypatch) -> None:
    open_kwargs: dict[str, object] = {}
    delivered: list[dict[str, object]] = []

    class _Db:
        def get_claims_monitoring_event(self, event_id: int) -> dict[str, object]:
            assert event_id == 9
            return {
                "id": 9,
                "user_id": "7",
                "payload_json": json.dumps(
                    {"window_ratio": 0.42, "threshold": 0.25, "baseline_ratio": 0.10}
                ),
            }

        def get_claims_monitoring_alert(self, alert_id: int) -> dict[str, object]:
            assert alert_id == 3
            return {
                "id": 3,
                "user_id": "7",
                "enabled": True,
                "channels_json": json.dumps({"slack": True}),
                "slack_webhook_url": "https://example.test/slack",
            }

        def list_claims_monitoring_events(self, **_kwargs) -> list[dict[str, object]]:
            return []

    @contextmanager
    def _fake_managed_media_database(*_args, **kwargs):
        open_kwargs.update(kwargs)
        yield _Db()

    monkeypatch.setattr(claims_job_handlers, "get_user_media_db_path", lambda owner: f"/tmp/user-{owner}/Media_DB_v2.db")
    monkeypatch.setattr(claims_job_handlers, "managed_media_database", _fake_managed_media_database)
    monkeypatch.setattr(
        claims_job_handlers,
        "deliver_claims_alert_webhook",
        lambda **kwargs: delivered.append(kwargs) or True,
    )

    result = await claims_job_handlers.process_claims_job(
        {
            "id": 1,
            "job_type": CLAIMS_DELIVER_ALERT_JOB_TYPE,
            "owner_user_id": "7",
            "payload": {"version": 1, "owner_user_id": "7", "event_id": 9, "alert_id": 3, "channel": "slack"},
        }
    )

    assert result["outcome"] == "ok"
    assert open_kwargs["initialize"] is False
    assert delivered[0]["payload"]["text"] == "Claims alert: unsupported ratio 42.0% (threshold 25.0%, baseline 10.0%)"
```

- [ ] **Step 2: Run handler tests and verify they fail for missing module**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Claims/test_claims_jobs_handlers.py -q
```

Expected: fail during import because `claims_job_handlers.py` does not exist yet.

- [ ] **Step 3: Add handler dispatch**

Create `claims_job_handlers.py`:

```python
from __future__ import annotations

import json
from typing import Any

from tldw_Server_API.app.core.DB_Management.db_path_utils import get_user_media_db_path

from .claims_job_contracts import (
    CLAIMS_DELIVER_ALERT_JOB_TYPE,
    CLAIMS_DELIVER_REVIEW_NOTIFICATION_JOB_TYPE,
    CLAIMS_REBUILD_MEDIA_JOB_TYPE,
    ClaimsJobError,
    validate_alert_delivery_payload,
    validate_rebuild_media_payload,
    validate_review_notification_payload,
)
from .claims_notifications import deliver_claim_review_notifications_now
from .claims_rebuild_service import rebuild_claims_for_media
from .claims_alert_delivery import (
    build_claims_alert_delivery_payload,
    deliver_claims_alert_webhook,
    normalize_claims_alert_channels,
)


def _payload(job: dict[str, Any]) -> dict[str, Any]:
    value = job.get("payload") or {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ClaimsJobError(
                "claims job payload is not valid JSON",
                retryable=False,
                failure_code="claims_invalid_payload",
            ) from exc
        if isinstance(parsed, dict):
            return parsed
    raise ClaimsJobError(
        "claims job payload must be an object",
        retryable=False,
        failure_code="claims_invalid_payload",
    )


def _assert_owner(job: dict[str, Any], owner_user_id: str) -> None:
    row_owner = str(job.get("owner_user_id") or "").strip()
    if row_owner and row_owner != str(owner_user_id):
        raise ClaimsJobError(
            "claims job owner mismatch",
            retryable=False,
            failure_code="claims_owner_scope_violation",
        )


def _db_path(owner_user_id: str) -> str:
    return str(get_user_media_db_path(int(owner_user_id)))


async def process_claims_job(job: dict[str, Any]) -> dict[str, Any]:
    job_type = str(job.get("job_type") or "").strip()
    if job_type == CLAIMS_REBUILD_MEDIA_JOB_TYPE:
        payload = validate_rebuild_media_payload(_payload(job))
        _assert_owner(job, payload["owner_user_id"])
        return rebuild_claims_for_media(
            db_path=_db_path(payload["owner_user_id"]),
            media_id=payload["media_id"],
        )
    if job_type == CLAIMS_DELIVER_REVIEW_NOTIFICATION_JOB_TYPE:
        payload = validate_review_notification_payload(_payload(job))
        _assert_owner(job, payload["owner_user_id"])
        result = deliver_claim_review_notifications_now(
            db_path=_db_path(payload["owner_user_id"]),
            owner_user_id=payload["owner_user_id"],
            notification_ids=payload["notification_ids"],
        )
        if result.get("outcome") == "failed":
            raise ClaimsJobError(
                str(result.get("reason") or "claims review notification delivery failed"),
                retryable=True,
                failure_code="claims_review_notification_delivery_failed",
            )
        return result
    if job_type == CLAIMS_DELIVER_ALERT_JOB_TYPE:
        payload = validate_alert_delivery_payload(_payload(job))
        _assert_owner(job, payload["owner_user_id"])
        return _deliver_alert(payload)
    raise ClaimsJobError(
        "unsupported claims job type",
        retryable=False,
        failure_code="claims_unsupported_job_type",
    )
```

- [ ] **Step 4: Extract and reuse the alert delivery seam**

Create `claims_alert_delivery.py` for alert-delivery-only behavior that can be used by both `claims_service.py` and `claims_job_handlers.py`. Move the existing webhook retry/event-recording behavior there and expose public helpers:

```python
def normalize_claims_alert_channels(raw_value: Any | None) -> dict[str, bool]: ...

def build_claims_alert_delivery_payload(*, channel: str, event_payload: dict[str, Any]) -> dict[str, Any]:
    if channel == "slack":
        return {
            "text": (
                "Claims alert: unsupported ratio "
                f"{_format_ratio(event_payload.get('window_ratio'))} "
                f"(threshold {_format_ratio(event_payload.get('threshold'))}, "
                f"baseline {_format_ratio(event_payload.get('baseline_ratio'))})"
            )
        }
    return dict(event_payload)

def deliver_claims_alert_webhook(
    *,
    url: str,
    payload: dict[str, Any],
    channel: str,
    db_path: str,
    user_id: str,
    alert_id: int | None = None,
    event_id: int | None = None,
) -> bool: ...
```

Keep `claims_service.py` importing these helpers instead of owning the delivery implementation. `claims_job_handlers.py` must not import private helpers from `claims_service.py`; alert delivery is domain delivery logic, not Jobs lifecycle logic.

When moving the webhook helper, change it to return `bool` and persist `event_id` when present:

```python
def deliver_claims_alert_webhook(
    *,
    url: str,
    payload: dict[str, Any],
    channel: str,
    db_path: str,
    user_id: str,
    alert_id: int | None = None,
    event_id: int | None = None,
) -> bool:
    ...
                _record_webhook_event(
                    db_path=db_path,
                    user_id=user_id,
                    channel=channel,
                    status="success",
                    attempt=attempt,
                    status_code=status_code,
                    alert_id=alert_id,
                    event_id=event_id,
                )
                return True
    ...
        if attempt >= max_attempts:
            return False
    return False
```

Update `_record_webhook_event` to persist `event_id` when provided:

```python
            if event_id is not None:
                payload["event_id"] = int(event_id)
```

Add `managed_media_database` and a local noncritical exception tuple to `claims_job_handlers.py` after imports. Do not import `claims_service.py` from this module.

```python
from tldw_Server_API.app.core.DB_Management.media_db.api import managed_media_database
from tldw_Server_API.app.core.DB_Management.media_db.runtime.noncritical import MEDIA_NONCRITICAL_EXCEPTIONS

_CLAIMS_HANDLER_NONCRITICAL_EXCEPTIONS = MEDIA_NONCRITICAL_EXCEPTIONS
```

Then add:

```python
def _payload_dict(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("payload_json") or "{}"
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _already_delivered(db: Any, *, owner_user_id: str, event_id: int, alert_id: int, channel: str) -> bool:
    rows = db.list_claims_monitoring_events(
        user_id=str(owner_user_id),
        event_type="webhook_delivery",
    )
    for row in rows:
        payload = _payload_dict(dict(row))
        if (
            str(payload.get("status")) == "success"
            and int(payload.get("event_id") or 0) == int(event_id)
            and int(payload.get("alert_id") or 0) == int(alert_id)
            and str(payload.get("channel") or "") == str(channel)
        ):
            return True
    return False


def _deliver_alert(payload: dict[str, Any]) -> dict[str, Any]:
    owner_user_id = payload["owner_user_id"]
    db_path = _db_path(owner_user_id)
    with managed_media_database(
        client_id="claims_jobs_worker",
        db_path=db_path,
        initialize=False,
        suppress_init_exceptions=_CLAIMS_HANDLER_NONCRITICAL_EXCEPTIONS,
        suppress_close_exceptions=_CLAIMS_HANDLER_NONCRITICAL_EXCEPTIONS,
    ) as db:
        event = db.get_claims_monitoring_event(int(payload["event_id"]))
        if not event or str(event.get("user_id")) != str(owner_user_id):
            return {"outcome": "skipped", "reason": "event_missing", "event_id": payload["event_id"]}
        alert = db.get_claims_monitoring_alert(int(payload["alert_id"]))
        if not alert or str(alert.get("user_id")) != str(owner_user_id):
            return {"outcome": "skipped", "reason": "alert_missing", "alert_id": payload["alert_id"]}
        if not bool(alert.get("enabled", True)):
            return {"outcome": "skipped", "reason": "alert_disabled", "alert_id": payload["alert_id"]}
        if _already_delivered(
            db,
            owner_user_id=owner_user_id,
            event_id=payload["event_id"],
            alert_id=payload["alert_id"],
            channel=payload["channel"],
        ):
            return {"outcome": "skipped", "reason": "already_delivered", "alert_id": payload["alert_id"]}
        channels = normalize_claims_alert_channels(alert.get("channels_json") or alert.get("channels"))
        if not channels.get(payload["channel"]):
            return {"outcome": "skipped", "reason": "channel_disabled", "channel": payload["channel"]}
        event_payload = _payload_dict(event)
        if payload["channel"] == "slack":
            url = alert.get("slack_webhook_url")
        else:
            url = alert.get("webhook_url")
        if not url:
            return {"outcome": "skipped", "reason": "channel_missing_url", "channel": payload["channel"]}
        body = build_claims_alert_delivery_payload(channel=payload["channel"], event_payload=event_payload)
        delivered = deliver_claims_alert_webhook(
            url=str(url),
            payload=body,
            channel=payload["channel"],
            db_path=db_path,
            user_id=owner_user_id,
            alert_id=payload["alert_id"],
            event_id=payload["event_id"],
        )
        if not delivered:
            raise ClaimsJobError(
                "claims alert delivery failed",
                retryable=True,
                failure_code="claims_alert_delivery_failed",
            )
        return {
            "outcome": "ok",
            "event_id": payload["event_id"],
            "alert_id": payload["alert_id"],
            "channel": payload["channel"],
        }
```

- [ ] **Step 5: Run handler and webhook tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Claims/test_claims_jobs_handlers.py tldw_Server_API/tests/Claims/test_claims_webhook_delivery.py -q
```

Expected: all listed tests pass.

- [ ] **Step 6: Commit Task 6**

Run:

```bash
git add tldw_Server_API/app/core/Claims_Extraction/claims_job_handlers.py tldw_Server_API/app/core/Claims_Extraction/claims_alert_delivery.py tldw_Server_API/app/core/Claims_Extraction/claims_service.py tldw_Server_API/tests/Claims/test_claims_jobs_handlers.py tldw_Server_API/tests/Claims/test_claims_webhook_delivery.py
git commit -m "feat: add Claims Jobs handlers"
```

## Task 7: Service Routing To Jobs

**Files:**
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_service.py`
- Test: `tldw_Server_API/tests/Claims/test_claims_review_api.py`
- Test: `tldw_Server_API/tests/Claims/test_claims_rebuild_stale_policy.py`
- Test: `tldw_Server_API/tests/Claims/test_claims_dashboard_analytics.py`

- [ ] **Step 1: Add routing tests for explicit rebuild and best-effort notifications**

Add focused unit tests near the existing Claims service tests:

```python
def test_rebuild_claims_uses_jobs_when_enabled(monkeypatch):
    calls: list[dict[str, object]] = []

    class _User:
        id = 1
        is_admin = True

    class _Db:
        db_path_str = "/tmp/user-1/Media_DB_v2.db"

    monkeypatch.setattr(claims_service.claims_jobs, "claims_jobs_enabled", lambda: True)
    monkeypatch.setattr(
        claims_service.claims_jobs,
        "enqueue_claims_rebuild_media",
        lambda **kwargs: calls.append(kwargs) or {"id": 99},
    )

    result = claims_service.rebuild_claims(media_id=42, user_id=None, current_user=_User(), db=_Db())

    assert result == {"status": "accepted", "media_id": 42, "job_id": "99"}
    assert calls[0]["owner_user_id"] == "1"
    assert calls[0]["media_id"] == 42
```

For notification routing, assert primary review success remains success when enqueue raises:

```python
@pytest.mark.asyncio
async def test_review_notification_enqueue_failure_does_not_rollback_review(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
    from tldw_Server_API.app.core.AuthNZ.permissions import CLAIMS_ADMIN, CLAIMS_REVIEW

    class _User:
        id = 1
        username = "reviewer"
        is_admin = False

    db_path, claim_id, _media_id = _seed_review_db()
    db = MediaDatabase(db_path=db_path, client_id="1")
    principal = AuthPrincipal(
        kind="user",
        user_id=1,
        api_key_id=None,
        subject="reviewer",
        token_type="access",
        jti=None,
        roles=["reviewer"],
        permissions=[CLAIMS_REVIEW, CLAIMS_ADMIN],
        is_admin=False,
        org_ids=[],
        team_ids=[],
    )
    monkeypatch.setattr(claims_service.claims_jobs, "claims_jobs_enabled", lambda: True)
    monkeypatch.setattr(
        claims_service.claims_jobs,
        "enqueue_claims_review_notification",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("jobs unavailable")),
    )

    try:
        result = await claims_service.review_claim(
            claim_id=claim_id,
            payload={"status": "approved", "review_version": 1},
            user_id=None,
            principal=principal,
            current_user=_User(),
            db=db,
        )
    finally:
        db.close_connection()

    assert result["review_status"] == "approved"
```

- [ ] **Step 2: Run routing tests and verify they fail**

Run the exact tests added in Step 1:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Claims/test_claims_review_api.py tldw_Server_API/tests/Claims/test_claims_rebuild_stale_policy.py -q
```

Expected: tests fail because `claims_service` does not import `claims_jobs` and still calls legacy dispatch directly.

- [ ] **Step 3: Import `claims_jobs` and add narrow routing helpers**

In `claims_service.py`, add:

```python
from tldw_Server_API.app.core.Claims_Extraction import claims_jobs
```

Update `_enqueue_claim_rebuild_if_needed`:

```python
def _enqueue_claim_rebuild_if_needed(*, media_id: int, db_path: str, owner_user_id: str | None = None) -> None:
    try:
        if claims_jobs.claims_jobs_enabled():
            if not owner_user_id:
                logger.debug("Claims rebuild Jobs enqueue skipped: missing owner_user_id")
                return
            claims_jobs.enqueue_claims_rebuild_media(
                media_id=int(media_id),
                owner_user_id=str(owner_user_id),
            )
            return
        svc = get_claims_rebuild_service()
        svc.submit(media_id=int(media_id), db_path=str(db_path))
    except _CLAIMS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("Claims rebuild enqueue failed: {}", exc)
```

Update call sites in `review_claim` and `bulk_review_claims` to pass the resolved owner:

```python
_enqueue_claim_rebuild_if_needed(
    media_id=int(claim_row.get("media_id") or 0),
    db_path=str(target_db.db_path_str),
    owner_user_id=_resolve_claim_owner_user_id(
        claim_row,
        int(user_id) if user_id is not None else int(current_user.id),
    ),
)
```

For `bulk_review_claims`, do not keep only a `set[int]` of media ids after the claim loop. Jobs mode needs the resolved owner for each media id. Replace the collector with an owner map:

```python
rebuild_media_owners: dict[int, str] = {}
...
if desired_status in {"flagged", "reassigned"} and desired_status != current_status:
    with suppress(_CLAIMS_NONCRITICAL_EXCEPTIONS):
        media_id = int(claim_row.get("media_id") or 0)
        owner_for_rebuild = _resolve_claim_owner_user_id(
            claim_row,
            int(user_id) if user_id is not None else int(current_user.id),
        )
        if media_id > 0 and owner_for_rebuild:
            rebuild_media_owners[media_id] = str(owner_for_rebuild)
...
for media_id, owner_for_rebuild in rebuild_media_owners.items():
    _enqueue_claim_rebuild_if_needed(
        media_id=media_id,
        db_path=str(target_db.db_path_str),
        owner_user_id=owner_for_rebuild,
    )
```

Update notification dispatch call sites:

```python
if claims_jobs.claims_jobs_enabled():
    try:
        claims_jobs.enqueue_claims_review_notification(
            owner_user_id=str(owner_user_id),
            notification_ids=[int(notif_id)],
        )
    except _CLAIMS_NONCRITICAL_EXCEPTIONS as exc:
        logger.debug("Failed to enqueue claims review notification job: {}", exc)
else:
    dispatch_claim_review_notifications(
        db_path=str(target_db.db_path_str),
        owner_user_id=str(owner_user_id),
        notification_ids=[int(notif_id)],
    )
```

Update `rebuild_claims`:

```python
    owner_user_id = str(user_id) if user_id is not None and _legacy_user_has_platform_admin_claims(current_user) else str(current_user.id)
    if claims_jobs.claims_jobs_enabled():
        try:
            job = claims_jobs.enqueue_claims_rebuild_media(
                media_id=int(media_id),
                owner_user_id=owner_user_id,
            )
        except _CLAIMS_NONCRITICAL_EXCEPTIONS as exc:
            raise HTTPException(status_code=503, detail="Claims rebuild job enqueue failed") from exc
        return {"status": "accepted", "media_id": media_id, "job_id": str(job.get("id") or "")}
```

Update `rebuild_all_media` so Jobs mode loops over one-media Jobs:

```python
        if claims_jobs.claims_jobs_enabled():
            owner_user_id = str(user_id) if user_id is not None and _legacy_user_has_platform_admin_claims(current_user) else str(current_user.id)
            enqueued = 0
            for mid in mids:
                claims_jobs.enqueue_claims_rebuild_media(media_id=int(mid), owner_user_id=owner_user_id)
                enqueued += 1
            return {"status": "accepted", "enqueued": enqueued, "policy": normalized_policy}
```

- [ ] **Step 4: Enqueue alert delivery Jobs from alert evaluation**

In `_evaluate_claims_alerts_for_user`, capture the event row returned from `insert_claims_monitoring_event`:

```python
            event_row = db.insert_claims_monitoring_event(
                user_id=str(target_user_id),
                event_type="unsupported_ratio",
                severity="warning",
                payload_json=json.dumps(payload),
            )
```

Then route delivery:

```python
            if claims_jobs.claims_jobs_enabled() and event_row.get("id"):
                _enqueue_claims_alert_delivery_jobs(
                    config_row=dict(cfg),
                    event_id=int(event_row["id"]),
                    owner_user_id=target_user_id,
                )
            else:
                _dispatch_claims_alert_notifications(
                    config_row=dict(cfg),
                    payload=payload,
                    db_path=db.db_path_str,
                    user_id=target_user_id,
                )
```

Add helper:

```python
def _enqueue_claims_alert_delivery_jobs(
    *,
    config_row: dict[str, Any],
    event_id: int,
    owner_user_id: str,
) -> None:
    channels = normalize_claims_alert_channels(config_row.get("channels_json") or config_row.get("channels"))
    alert_id = int(config_row.get("id") or 0)
    for channel in ("slack", "webhook"):
        if not channels.get(channel):
            continue
        try:
            claims_jobs.enqueue_claims_alert_delivery(
                owner_user_id=str(owner_user_id),
                event_id=int(event_id),
                alert_id=alert_id,
                channel=channel,
            )
        except _CLAIMS_NONCRITICAL_EXCEPTIONS as exc:
            logger.debug("Failed to enqueue claims alert delivery job: {}", exc)
```

- [ ] **Step 5: Add dashboard Jobs summary**

In `claims_dashboard_analytics`, add:

```python
    try:
        payload["claims_jobs"] = claims_jobs.claims_jobs_summary(owner_user_id=owner_user_id)
    except _CLAIMS_NONCRITICAL_EXCEPTIONS:
        payload["claims_jobs"] = None
```

Add a dashboard test that monkeypatches `claims_jobs.claims_jobs_summary` and asserts the payload includes `claims_jobs` but no queue-control fields:

```python
assert "claims_jobs" in data
assert "pause" not in data["claims_jobs"]
assert "drain" not in data["claims_jobs"]
assert "requeue" not in data["claims_jobs"]
```

- [ ] **Step 6: Run Claims routing/dashboard tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Claims/test_claims_review_api.py tldw_Server_API/tests/Claims/test_claims_rebuild_stale_policy.py tldw_Server_API/tests/Claims/test_claims_dashboard_analytics.py -q
```

Expected: all listed tests pass.

- [ ] **Step 7: Commit Task 7**

Run:

```bash
git add tldw_Server_API/app/core/Claims_Extraction/claims_service.py tldw_Server_API/tests/Claims/test_claims_review_api.py tldw_Server_API/tests/Claims/test_claims_rebuild_stale_policy.py tldw_Server_API/tests/Claims/test_claims_dashboard_analytics.py
git commit -m "feat: route Claims background work through Jobs"
```

## Task 8: Claims Jobs Worker And Lifecycle Registration

**Files:**
- Create: `tldw_Server_API/app/services/claims_jobs_worker.py`
- Modify: `tldw_Server_API/app/services/startup_worker_groups.py`
- Test: `tldw_Server_API/tests/Services/test_claims_jobs_worker.py`
- Test: `tldw_Server_API/tests/Services/test_startup_worker_groups.py`

- [ ] **Step 1: Write failing worker tests**

Create `test_claims_jobs_worker.py`:

```python
import asyncio

import pytest

from tldw_Server_API.app.services import claims_jobs_worker
from tldw_Server_API.app.services.lifecycle_worker_specs import (
    ShutdownPhase,
    WorkerLifecycleContext,
)


pytestmark = pytest.mark.unit


def _context(settings=None):
    return WorkerLifecycleContext(
        app=object(),
        settings=settings or {},
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )


def test_claims_jobs_worker_spec_is_job_poller() -> None:
    [spec] = claims_jobs_worker.provide_claims_jobs_worker_specs()

    assert spec.name == "claims_jobs_task"
    assert spec.task_name == "claims_jobs_task"
    assert spec.category == "jobs"
    assert spec.phase is ShutdownPhase.JOB_POLLER_QUIESCE
    assert spec.enabled(_context({"CLAIMS_JOBS_WORKER_ENABLED": True})) is True
    assert spec.enabled(_context({"CLAIMS_JOBS_WORKER_ENABLED": False})) is False


@pytest.mark.asyncio
async def test_start_claims_jobs_worker_uses_worker_sdk_without_owner_filter(monkeypatch) -> None:
    observed = {}

    class _FakeSDK:
        def __init__(self, manager, config):
            observed["manager"] = manager
            observed["config"] = config
            self.stopped = False

        def stop(self):
            self.stopped = True
            observed["stopped"] = True

        async def run(self, *, handler, cancel_check=None):
            observed["handler"] = handler
            observed["cancel_check"] = cancel_check

    monkeypatch.setattr(claims_jobs_worker, "WorkerSDK", _FakeSDK)
    monkeypatch.setattr(claims_jobs_worker, "jobs_manager_from_env", lambda: "manager")
    stop_event = asyncio.Event()
    stop_event.set()

    await claims_jobs_worker.start_claims_jobs_worker(stop_event=stop_event)

    assert observed["config"].domain == "claims"
    assert observed["config"].queue == "default"
    assert observed["stopped"] is True
```

- [ ] **Step 2: Run worker tests and verify they fail for missing module**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_claims_jobs_worker.py -q
```

Expected: fail during import because `claims_jobs_worker.py` does not exist yet.

- [ ] **Step 3: Add the worker service**

Create `claims_jobs_worker.py`:

```python
from __future__ import annotations

import asyncio
import os
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Claims_Extraction.claims_job_contracts import CLAIMS_JOBS_DOMAIN
from tldw_Server_API.app.core.Claims_Extraction.claims_job_handlers import process_claims_job
from tldw_Server_API.app.core.Claims_Extraction.claims_jobs import (
    claims_jobs_queue,
    claims_jobs_worker_enabled,
)
from tldw_Server_API.app.core.Jobs.worker_sdk import WorkerConfig, WorkerSDK
from tldw_Server_API.app.core.Jobs.worker_utils import coerce_int, jobs_manager_from_env
from tldw_Server_API.app.services.lifecycle_worker_specs import (
    WorkerLifecycleContext,
    WorkerSpec,
)
from tldw_Server_API.app.services.lifecycle_workers import ShutdownPhase


def _worker_id() -> str:
    return (os.getenv("CLAIMS_JOBS_WORKER_ID") or "claims-jobs-worker").strip() or "claims-jobs-worker"


def build_claims_worker_config() -> WorkerConfig:
    return WorkerConfig(
        domain=CLAIMS_JOBS_DOMAIN,
        queue=claims_jobs_queue(),
        worker_id=_worker_id(),
        lease_seconds=coerce_int(os.getenv("CLAIMS_JOBS_LEASE_SECONDS"), 120),
        renew_jitter_seconds=coerce_int(os.getenv("CLAIMS_JOBS_RENEW_JITTER_SECONDS"), 5),
        renew_threshold_seconds=coerce_int(os.getenv("CLAIMS_JOBS_RENEW_THRESHOLD_SECONDS"), 15),
        backoff_base_seconds=coerce_int(os.getenv("CLAIMS_JOBS_BACKOFF_BASE_SECONDS"), 2),
        backoff_max_seconds=coerce_int(os.getenv("CLAIMS_JOBS_BACKOFF_MAX_SECONDS"), 30),
        retry_on_exception=True,
        retry_backoff_seconds=coerce_int(os.getenv("CLAIMS_JOBS_RETRY_BACKOFF_SECONDS"), 10),
    )


async def start_claims_jobs_worker(stop_event: asyncio.Event | None = None) -> None:
    manager = jobs_manager_from_env()
    sdk = WorkerSDK(manager, build_claims_worker_config())

    async def _wait_for_stop() -> None:
        if stop_event is None:
            return
        await stop_event.wait()
        sdk.stop()

    stopper = asyncio.create_task(_wait_for_stop())
    try:
        await sdk.run(handler=process_claims_job)
    except asyncio.CancelledError:
        sdk.stop()
        raise
    finally:
        sdk.stop()
        stopper.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stopper
        logger.info("Claims Jobs worker stopped")
```

Add `import contextlib` to the module.

Add lifecycle provider:

```python
def _claims_jobs_worker_enabled(context: WorkerLifecycleContext) -> bool:
    return claims_jobs_worker_enabled(context.settings)


def provide_claims_jobs_worker_specs(
    _context: WorkerLifecycleContext | None = None,
) -> tuple[WorkerSpec, ...]:
    return (
        WorkerSpec(
            name="claims_jobs_task",
            task_name="claims_jobs_task",
            category="jobs",
            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
            enabled=_claims_jobs_worker_enabled,
            factory=lambda _context, stop_event: start_claims_jobs_worker(stop_event=stop_event),
        ),
    )
```

Do not pass `owner_user_id` to `WorkerSDK.run(...)`. The Jobs SDK acquisition will process all owners in the Claims domain/queue.

- [ ] **Step 4: Register the provider in startup catalog**

In `startup_worker_groups.py`, import and include the provider before the legacy `provide_claims_rebuild_worker_specs` entry:

```python
    from tldw_Server_API.app.services.claims_jobs_worker import (
        provide_claims_jobs_worker_specs,
    )
```

Return order excerpt:

```python
        provide_content_jobs_worker_specs,
        provide_sidecar_owned_jobs_worker_specs,
        provide_notifications_abtest_worker_specs,
        provide_cleanup_worker_specs,
        provide_compactor_websub_worker_specs,
        provide_claims_jobs_worker_specs,
        provide_claims_rebuild_worker_specs,
```

Update `test_startup_worker_groups.py` expected names to include `claims_jobs_task`.

- [ ] **Step 5: Run worker lifecycle tests**

Run:

```bash
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Services/test_claims_jobs_worker.py tldw_Server_API/tests/Services/test_startup_worker_groups.py -q
```

Expected: all listed tests pass.

- [ ] **Step 6: Commit Task 8**

Run:

```bash
git add tldw_Server_API/app/services/claims_jobs_worker.py tldw_Server_API/app/services/startup_worker_groups.py tldw_Server_API/tests/Services/test_claims_jobs_worker.py tldw_Server_API/tests/Services/test_startup_worker_groups.py
git commit -m "feat: add Claims Jobs worker"
```

## Task 9: Integration Verification And Security Sweep

**Files:**
- Modify only files touched by previous tasks if verification exposes failures.
- Update Backlog implementation task with touched files and verification results.

- [ ] **Step 1: Run focused Claims Jobs test group**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Claims/test_claims_jobs_contracts.py \
  tldw_Server_API/tests/Claims/test_claims_jobs_enqueue.py \
  tldw_Server_API/tests/Claims/test_claims_jobs_handlers.py \
  tldw_Server_API/tests/Claims/test_claims_review_notifications.py \
  tldw_Server_API/tests/Claims/test_claims_webhook_delivery.py \
  tldw_Server_API/tests/Services/test_claims_jobs_worker.py \
  -q
```

Expected: all listed tests pass.

- [ ] **Step 2: Run related regression tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Claims/test_claims_review_api.py \
  tldw_Server_API/tests/Claims/test_claims_rebuild_stale_policy.py \
  tldw_Server_API/tests/Claims/test_claims_dashboard_analytics.py \
  tldw_Server_API/tests/Claims/test_claims_alerts_scheduler.py \
  tldw_Server_API/tests/DB_Management/test_media_db_claims_monitoring_event_ops.py \
  tldw_Server_API/tests/Services/test_startup_worker_groups.py \
  -q
```

Expected: all listed tests pass.

- [ ] **Step 3: Run Jobs owner/idempotency guard tests if the local environment supports them**

Run SQLite Jobs tests:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_idempotency_scope_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_rls_postgres.py \
  -q
```

Expected: SQLite test passes. PostgreSQL/RLS test passes when the project test database is available; if it skips for missing Postgres, record the skip reason in the Backlog task.

- [ ] **Step 4: Run formatting and security checks**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Claims_Extraction \
  tldw_Server_API/app/services/claims_jobs_worker.py \
  -f json -o /tmp/bandit_claims_jobs_stage1.json
```

Expected: no new Bandit findings in touched code. If Bandit reports existing unrelated findings, record them separately and fix any finding in code changed by this plan.

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 5: Run the focused suite after final fixes**

Run the commands from Steps 1 and 2 again after every verification-driven code change.

Expected: all focused and regression tests pass, or environment-only skips are recorded with exact skip text.

- [ ] **Step 6: Final commit**

Run:

```bash
git status --short
git add \
  tldw_Server_API/app/core/Claims_Extraction/claims_job_contracts.py \
  tldw_Server_API/app/core/Claims_Extraction/claims_jobs.py \
  tldw_Server_API/app/core/Claims_Extraction/claims_job_handlers.py \
  tldw_Server_API/app/core/Claims_Extraction/claims_rebuild_service.py \
  tldw_Server_API/app/core/Claims_Extraction/claims_notifications.py \
  tldw_Server_API/app/core/Claims_Extraction/claims_service.py \
  tldw_Server_API/app/core/DB_Management/media_db/runtime/claims_monitoring_event_ops.py \
  tldw_Server_API/app/core/DB_Management/media_db/media_database_impl.py \
  tldw_Server_API/app/services/claims_jobs_worker.py \
  tldw_Server_API/app/services/startup_worker_groups.py \
  tldw_Server_API/tests/Claims/test_claims_jobs_contracts.py \
  tldw_Server_API/tests/Claims/test_claims_jobs_enqueue.py \
  tldw_Server_API/tests/Claims/test_claims_jobs_handlers.py \
  tldw_Server_API/tests/Claims/test_claims_review_notifications.py \
  tldw_Server_API/tests/Claims/test_claims_webhook_delivery.py \
  tldw_Server_API/tests/Claims/test_claims_review_api.py \
  tldw_Server_API/tests/Claims/test_claims_rebuild_stale_policy.py \
  tldw_Server_API/tests/Claims/test_claims_dashboard_analytics.py \
  tldw_Server_API/tests/DB_Management/test_media_db_claims_monitoring_event_ops.py \
  tldw_Server_API/tests/Services/test_claims_jobs_worker.py \
  tldw_Server_API/tests/Services/test_startup_worker_groups.py
git commit -m "feat: move Claims Stage 1 work onto Jobs"
```

Expected: commit succeeds and no unrelated untracked files are staged.

## Self-Review Checklist

- Spec coverage: Tasks 1-2 cover contracts, payloads, enqueue helpers, idempotency, and queue settings. Tasks 3-7 cover rebuild, review notification, alert delivery, routing rules, enqueue failure behavior, and dashboard summary. Task 8 covers `WorkerSDK` startup and no owner acquisition filter. Task 9 covers verification, Bandit, and rollout evidence.
- Jobs ownership: The plan uses `JobManager.create_job(...)`, `JobManager.summarize_by_status(...)`, and `WorkerSDK`; it does not add Claims queue-control endpoints, custom leases, retry loops, or lifecycle mechanics.
- Payload safety: Contract tests reject raw paths, recipient/channel secrets, claim text, and synthetic owners.
- Alert persistence: The plan reuses `claims_monitoring_events` with `delivered_at` and adds event lookup/return values needed by ID-only alert Jobs.
- Type consistency: Job type constants are defined once in `claims_job_contracts.py` and imported by enqueue helpers and handlers.
- Verification: Focused pytest commands, related regressions, Jobs owner/idempotency checks, Bandit, and `git diff --check` are included.
