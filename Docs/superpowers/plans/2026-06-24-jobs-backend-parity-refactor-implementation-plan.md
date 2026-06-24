# Jobs Backend Parity Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first Jobs refactor slice: inventory the compatibility boundary, add backend parity and public contract coverage, and introduce settings/operation contracts before moving production SQL.

**Architecture:** Keep `JobManager` as the only public facade in this slice. Add shared parity scenarios under `tldw_Server_API/tests/Jobs/parity/`, add field-level public contract tests, and add standalone `JobsSettings` plus operation contract dataclasses without routing `JobManager` through them yet.

**Tech Stack:** Python, pytest, FastAPI `TestClient`, SQLite Jobs migrations, existing `pg_jobs`/`jobs_pg_dsn` Postgres fixtures, dataclasses, enums.

---

## Scope

This plan implements the first safety-net PR from `Docs/superpowers/specs/2026-06-24-jobs-backend-parity-refactor-design.md`.

This plan does not extract admission or lifecycle SQL from `tldw_Server_API/app/core/Jobs/manager.py`. Extraction starts only after these tests and contracts are merged.

## File Structure

- Create `Docs/Design/JOBS_BACKEND_PARITY_INVENTORY_2026_06_24.md`
  - Records direct Jobs SQL boundaries and domain status/id mappings.
- Create `tldw_Server_API/tests/Jobs/parity/__init__.py`
  - Marks the parity helpers package.
- Create `tldw_Server_API/tests/Jobs/parity/scenarios.py`
  - Shared scenario functions executed by both SQLite and Postgres wrappers.
- Create `tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py`
  - Runs fast shared scenarios against SQLite.
- Create `tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py`
  - Runs the same shared scenarios against the existing real Postgres fixture.
- Reference `tldw_Server_API/tests/Jobs/conftest.py`
  - Use the existing `pg_jobs` marker, `jobs_pg_dsn` fixture, and Jobs test environment defaults.
- Modify `tldw_Server_API/tests/Jobs/test_jobs_idempotency_scope_sqlite.py`
  - Delegate duplicate idempotency assertions to shared scenarios.
- Modify `tldw_Server_API/tests/Jobs/test_jobs_idempotency_scope_postgres.py`
  - Delegate duplicate idempotency assertions to shared scenarios and add request-id parity coverage.
- Modify `tldw_Server_API/tests/Jobs/test_jobs_completion_idempotent_sqlite.py`
  - Delegate duplicate completion idempotency assertions to shared scenarios.
- Modify `tldw_Server_API/tests/Jobs/test_jobs_completion_idempotent_postgres.py`
  - Delegate duplicate completion idempotency assertions to shared scenarios.
- Create `tldw_Server_API/tests/Jobs/test_jobs_admin_contract_sqlite.py`
  - Field-level public contract test for Jobs admin list/detail responses.
- Modify `tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_adapter.py`
  - Add status and payload-id mapping contract tests for the non-identity Chatbooks adapter boundary.
- Create `tldw_Server_API/app/core/Jobs/settings.py`
  - Defines explicit construction-time, operation-time, and refreshable setting semantics.
- Create `tldw_Server_API/tests/Jobs/test_jobs_settings.py`
  - Locks the settings snapshot/refresh behavior before manager integration.
- Create `tldw_Server_API/app/core/Jobs/operations/__init__.py`
  - Exposes operation contracts package.
- Create `tldw_Server_API/app/core/Jobs/operations/contracts.py`
  - Defines typed command/result objects and reason enums.
- Create `tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py`
  - Locks contract shape and ensures contracts do not import `JobManager`.
- Modify `backlog/tasks/task-12016 - Plan-Jobs-backend-parity-refactor-implementation.md`
  - Track implementation-plan completion and verification evidence.

---

### Task 1: Inventory Direct SQL And Domain Mapping Boundaries

**Files:**
- Create: `Docs/Design/JOBS_BACKEND_PARITY_INVENTORY_2026_06_24.md`
- Modify: `backlog/tasks/task-12016 - Plan-Jobs-backend-parity-refactor-implementation.md`

- [ ] **Step 1: Create the inventory document**

Create `Docs/Design/JOBS_BACKEND_PARITY_INVENTORY_2026_06_24.md` with this content:

```markdown
# Jobs Backend Parity Inventory

Date: 2026-06-24
Source spec: Docs/superpowers/specs/2026-06-24-jobs-backend-parity-refactor-design.md

## Purpose

This inventory defines the compatibility boundary for the first Jobs backend parity refactor PR. Production SQL extraction must not begin until each state-changing or public-facing path below is either covered by tests in this PR or explicitly assigned to a later extraction slice.

## Direct Runtime Jobs SQL

| Area | File | Representative SQL | Classification | First Slice Action |
| --- | --- | --- | --- | --- |
| Jobs admin SLA policies | `tldw_Server_API/app/api/v1/endpoints/jobs_admin.py` | `SELECT * FROM job_sla_policies` | read-only/status SQL | Defer as read model; existing SLA endpoint tests remain coverage. |
| Jobs admin SLA breaches | `tldw_Server_API/app/api/v1/endpoints/jobs_admin.py` | `SELECT id, domain, queue, job_type, status FROM jobs` | read-only/status SQL | Defer as read model; no extraction in first slice. |
| Jobs admin archive metadata | `tldw_Server_API/app/api/v1/endpoints/jobs_admin.py` | `SELECT payload, result, payload_compressed, result_compressed FROM jobs_archive` | read-only/status SQL | Defer as read model; no extraction in first slice. |
| Jobs admin batch cancel | `tldw_Server_API/app/api/v1/endpoints/jobs_admin.py` | `UPDATE jobs SET status='cancelled'` | state-changing SQL | Defer extraction; keep existing endpoint behavior and require separate operation slice. |
| Jobs admin batch reschedule | `tldw_Server_API/app/api/v1/endpoints/jobs_admin.py` | `UPDATE jobs SET available_at = NOW() + interval` | state-changing SQL | Defer extraction; keep existing endpoint behavior and require separate operation slice. |
| Jobs admin requeue quarantined | `tldw_Server_API/app/api/v1/endpoints/jobs_admin.py` | `UPDATE jobs SET status='queued'` | state-changing SQL | Defer extraction; keep existing endpoint behavior and require separate operation slice. |
| Prompt Studio status dashboard | `tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_status.py` | `SELECT status, COUNT(*) AS c FROM jobs` | read-only/status SQL | Defer as read model; include in mapping inventory. |
| Jobs metrics service | `tldw_Server_API/app/services/jobs_metrics_service.py` | `SELECT COUNT(*) FROM jobs` | service/worker operational SQL | Defer; keep service-specific metrics tests. |
| Audio jobs worker fairness scans | `tldw_Server_API/app/services/audio_jobs_worker.py` | `SELECT owner_user_id FROM jobs` | service/worker operational SQL | Defer; cover in worker-specific slice if acquire semantics move. |
| Jobs webhooks service | `tldw_Server_API/app/services/jobs_webhooks_service.py` | `SELECT id, event_type FROM job_events` | service/worker operational SQL | Defer; event outbox extraction owns this boundary. |
| External sources quota scan | `tldw_Server_API/app/core/External_Sources/connectors_service.py` | `SELECT COUNT(*) AS c FROM jobs` | service/worker operational SQL | Defer; not part of admission/lifecycle first slice. |

## Domain Status And Identifier Mappings

| Domain | Endpoint Or Adapter | Mapping | First Slice Action |
| --- | --- | --- | --- |
| Embeddings | `tldw_Server_API/app/core/Embeddings/jobs_adapter.py` | `quarantined -> failed`; unknown status derives as `processing`; public id prefers `jobs.uuid` | Defer endpoint contract; existing adapter tests stay active. |
| Chatbooks export | `tldw_Server_API/app/core/Chatbooks/jobs_adapter.py` | `queued -> pending`; `processing -> in_progress`; `quarantined -> failed`; payload `chatbooks_job_id` preferred over Jobs id | Add adapter contract tests in this PR. |
| Chatbooks import | `tldw_Server_API/app/core/Chatbooks/jobs_adapter.py` | `queued -> pending`; `processing -> in_progress`; `quarantined -> failed`; payload `chatbooks_job_id` preferred over Jobs id | Add adapter contract tests in this PR. |
| Prompt Studio optimization | `tldw_Server_API/app/core/Prompt_Management/prompt_studio/jobs_adapter.py` | `quarantined -> failed`; unknown status falls back to `queued` | Defer to domain adapter slice; no production extraction in first PR. |

## First PR Compatibility Gates

- Shared SQLite/Postgres scenarios cover idempotent create, acquire, renew stale/no-op behavior, complete idempotency, cancel terminal no-op, and events outbox behavior.
- Admin list/detail public responses are tested by required fields, not snapshots.
- Chatbooks adapter mapping is tested without FastAPI startup.
- `JobsSettings` documents snapshot, refresh, and operation-time setting groups before manager integration.
- Operation contract dataclasses exist and do not import `JobManager`.
```

- [ ] **Step 2: Verify inventory references real files**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python - <<'PY'
from pathlib import Path
doc = Path("Docs/Design/JOBS_BACKEND_PARITY_INVENTORY_2026_06_24.md").read_text()
paths = [
    "tldw_Server_API/app/api/v1/endpoints/jobs_admin.py",
    "tldw_Server_API/app/api/v1/endpoints/prompt_studio/prompt_studio_status.py",
    "tldw_Server_API/app/services/jobs_metrics_service.py",
    "tldw_Server_API/app/services/audio_jobs_worker.py",
    "tldw_Server_API/app/services/jobs_webhooks_service.py",
    "tldw_Server_API/app/core/External_Sources/connectors_service.py",
    "tldw_Server_API/app/core/Chatbooks/jobs_adapter.py",
    "tldw_Server_API/app/core/Embeddings/jobs_adapter.py",
]
missing = [path for path in paths if not Path(path).exists() or path not in doc]
assert missing == [], missing
PY
```

Expected: command exits with status `0`.

- [ ] **Step 3: Record task note**

Update `TASK-12016` implementation notes with:

```text
Inventory created at Docs/Design/JOBS_BACKEND_PARITY_INVENTORY_2026_06_24.md. It classifies admin direct SQL, read-model SQL, service/worker operational SQL, and first-slice domain mapping coverage.
```

- [ ] **Step 4: Commit**

Run:

```bash
git add Docs/Design/JOBS_BACKEND_PARITY_INVENTORY_2026_06_24.md "backlog/tasks/task-12016 - Plan-Jobs-backend-parity-refactor-implementation.md"
git commit -m "docs(jobs): inventory parity refactor boundaries"
```

Expected: commit succeeds.

---

### Task 2: Add Shared Parity Scenario Helpers

**Files:**
- Create: `tldw_Server_API/tests/Jobs/parity/__init__.py`
- Create: `tldw_Server_API/tests/Jobs/parity/scenarios.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_idempotency_scope_sqlite.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_idempotency_scope_postgres.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_completion_idempotent_sqlite.py`
- Modify: `tldw_Server_API/tests/Jobs/test_jobs_completion_idempotent_postgres.py`

- [ ] **Step 1: Create the package marker**

Create `tldw_Server_API/tests/Jobs/parity/__init__.py`:

```python
"""Shared Jobs backend parity scenarios."""
```

- [ ] **Step 2: Write the shared scenario file**

Create `tldw_Server_API/tests/Jobs/parity/scenarios.py`:

```python
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tldw_Server_API.app.core.Jobs.manager import JobManager

ManagerFactory = Callable[[], JobManager]


def run_idempotent_create_scope_scenario(make_manager: ManagerFactory) -> None:
    jm = make_manager()
    key = "idem-key-123"

    first = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="1",
        idempotency_key=key,
    )
    replay = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="1",
        idempotency_key=key,
    )
    assert int(first["id"]) == int(replay["id"])

    different_queue = jm.create_job(
        domain="chatbooks",
        queue="high",
        job_type="export",
        payload={},
        owner_user_id="1",
        idempotency_key=key,
    )
    different_type = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="import",
        payload={},
        owner_user_id="1",
        idempotency_key=key,
    )
    different_domain = jm.create_job(
        domain="other",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="2",
        idempotency_key=key,
    )

    assert int(different_queue["id"]) != int(first["id"])
    assert int(different_type["id"]) != int(first["id"])
    assert int(different_domain["id"]) != int(first["id"])


def run_idempotent_create_preserves_original_request_ids_scenario(make_manager: ManagerFactory) -> None:
    jm = make_manager()
    key = "idem-request-id-key"

    first = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="1",
        idempotency_key=key,
        request_id="request-first",
        trace_id="trace-first",
    )
    replay = jm.create_job(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id="1",
        idempotency_key=key,
        request_id="request-second",
        trace_id="trace-second",
    )

    assert int(first["id"]) == int(replay["id"])
    assert first["request_id"] == "request-first"
    assert replay["request_id"] == "request-first"
    assert first["trace_id"] == "trace-first"
    assert replay["trace_id"] == "trace-first"


def run_acquire_complete_lifecycle_scenario(make_manager: ManagerFactory) -> None:
    jm = make_manager()
    job = jm.create_job(
        domain="parity",
        queue="default",
        job_type="lifecycle",
        payload={"value": 1},
        owner_user_id="owner-1",
    )

    acquired = jm.acquire_next_job(
        domain="parity",
        queue="default",
        lease_seconds=10,
        worker_id="worker-1",
    )

    assert acquired is not None
    assert int(acquired["id"]) == int(job["id"])
    assert acquired["status"] == "processing"
    assert acquired["worker_id"] == "worker-1"
    assert acquired.get("lease_id")

    token = str(acquired["lease_id"])
    assert jm.complete_job(
        int(job["id"]),
        result={"ok": True},
        worker_id="worker-1",
        lease_id=token,
        completion_token=token,
    ) is True

    stored = jm.get_job(int(job["id"]))
    assert stored is not None
    assert stored["status"] == "completed"
    assert stored.get("leased_until") is None


def run_complete_idempotency_scenario(make_manager: ManagerFactory) -> None:
    jm = make_manager()
    job = jm.create_job(domain="test", queue="default", job_type="t", payload={}, owner_user_id="u")
    acquired = jm.acquire_next_job(domain="test", queue="default", lease_seconds=10, worker_id="w1")
    assert acquired is not None
    assert int(acquired["id"]) == int(job["id"])

    token = str(acquired["lease_id"])
    assert jm.complete_job(int(job["id"]), worker_id="w1", lease_id=token, completion_token=token) is True
    assert jm.complete_job(int(job["id"]), worker_id="w1", lease_id=token, completion_token=token) is True
    assert jm.complete_job(int(job["id"]), worker_id="w1", lease_id=token, completion_token="other-token") is False


def run_renew_stale_lease_noop_scenario(make_manager: ManagerFactory) -> None:
    jm = make_manager()
    job = jm.create_job(domain="parity", queue="default", job_type="renew", payload={}, owner_user_id="owner-1")
    acquired = jm.acquire_next_job(domain="parity", queue="default", lease_seconds=10, worker_id="worker-1")
    assert acquired is not None
    assert int(acquired["id"]) == int(job["id"])

    current_lease_id = str(acquired["lease_id"])
    assert jm.renew_job_lease(
        int(job["id"]),
        seconds=20,
        worker_id="worker-1",
        lease_id=current_lease_id,
        progress_percent=25.0,
        progress_message="still running",
        enforce=True,
    ) is True

    assert jm.renew_job_lease(
        int(job["id"]),
        seconds=20,
        worker_id="worker-1",
        lease_id="stale-lease",
        enforce=True,
    ) is False

    stored = jm.get_job(int(job["id"]))
    assert stored is not None
    assert stored["status"] == "processing"
    assert float(stored["progress_percent"]) == 25.0
    assert stored["progress_message"] == "still running"


def run_cancel_terminal_noop_scenario(make_manager: ManagerFactory) -> None:
    jm = make_manager()
    job = jm.create_job(domain="parity", queue="default", job_type="cancel", payload={}, owner_user_id="owner-1")

    assert jm.cancel_job(int(job["id"]), reason="user") is True
    stored = jm.get_job(int(job["id"]))
    assert stored is not None
    assert stored["status"] == "cancelled"

    assert jm.cancel_job(int(job["id"]), reason="again") is False
    stored_again = jm.get_job(int(job["id"]))
    assert stored_again is not None
    assert stored_again["status"] == "cancelled"


def run_events_outbox_create_complete_scenario(make_manager: ManagerFactory) -> None:
    jm = make_manager()
    job = jm.create_job(domain="parity", queue="default", job_type="events", payload={}, owner_user_id="owner-1")
    acquired = jm.acquire_next_job(domain="parity", queue="default", lease_seconds=10, worker_id="worker-1")
    assert acquired is not None
    token = str(acquired["lease_id"])
    assert jm.complete_job(int(job["id"]), worker_id="worker-1", lease_id=token, completion_token=token) is True

    events = jm.list_job_events_after(after_id=0, domain="parity", queue="default", job_type="events", limit=20)
    event_types = [str(row.get("event_type")) for row in events]
    assert "job.created" in event_types
    assert "job.completed" in event_types

    for event in events:
        assert event.get("attrs_json") is not None
        assert event.get("domain") == "parity"
        assert event.get("queue") == "default"
        assert event.get("job_type") == "events"
```

- [ ] **Step 3: Run the new scenario module import check**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m py_compile tldw_Server_API/tests/Jobs/parity/scenarios.py
```

Expected: command exits with status `0`.

- [ ] **Step 4: Refactor SQLite idempotency tests to shared scenarios**

Replace the body of `tldw_Server_API/tests/Jobs/test_jobs_idempotency_scope_sqlite.py` with:

```python
import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.tests.Jobs.parity.scenarios import (
    run_idempotent_create_preserves_original_request_ids_scenario,
    run_idempotent_create_scope_scenario,
)


@pytest.fixture()
def jobs_db(tmp_path):
    db_path = tmp_path / "jobs.db"
    ensure_jobs_tables(db_path)
    return db_path


def test_idempotency_scoped_to_domain_queue_type_sqlite(jobs_db):
    run_idempotent_create_scope_scenario(lambda: JobManager(jobs_db))


def test_idempotent_create_preserves_original_request_id_sqlite(jobs_db):
    run_idempotent_create_preserves_original_request_ids_scenario(lambda: JobManager(jobs_db))
```

- [ ] **Step 5: Refactor Postgres idempotency tests to shared scenarios**

Replace the body of `tldw_Server_API/tests/Jobs/test_jobs_idempotency_scope_postgres.py` with:

```python
import pytest

psycopg = pytest.importorskip("psycopg")
pytestmark = pytest.mark.pg_jobs

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.tests.Jobs.parity.scenarios import (
    run_idempotent_create_preserves_original_request_ids_scenario,
    run_idempotent_create_scope_scenario,
)


def test_idempotency_scoped_to_domain_queue_type_postgres(jobs_pg_dsn):
    run_idempotent_create_scope_scenario(lambda: JobManager(None, backend="postgres", db_url=jobs_pg_dsn))


def test_idempotent_create_preserves_original_request_id_postgres(jobs_pg_dsn):
    run_idempotent_create_preserves_original_request_ids_scenario(
        lambda: JobManager(None, backend="postgres", db_url=jobs_pg_dsn)
    )
```

- [ ] **Step 6: Refactor SQLite completion idempotency test to shared scenario**

Replace the body of `tldw_Server_API/tests/Jobs/test_jobs_completion_idempotent_sqlite.py` with:

```python
import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.tests.Jobs.parity.scenarios import run_complete_idempotency_scenario


@pytest.mark.unit
def test_completion_idempotent_sqlite(tmp_path, monkeypatch):
    db_path = tmp_path / "jobs.db"
    monkeypatch.setenv("JOBS_DB_PATH", str(db_path))
    run_complete_idempotency_scenario(lambda: JobManager())
```

- [ ] **Step 7: Refactor Postgres completion idempotency test to shared scenario**

Replace the body of `tldw_Server_API/tests/Jobs/test_jobs_completion_idempotent_postgres.py` with:

```python
import pytest

psycopg = pytest.importorskip("psycopg")
pytestmark = pytest.mark.pg_jobs

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.tests.Jobs.parity.scenarios import run_complete_idempotency_scenario


def test_completion_idempotent_postgres(jobs_pg_dsn):
    run_complete_idempotency_scenario(lambda: JobManager(None, backend="postgres", db_url=jobs_pg_dsn))
```

- [ ] **Step 8: Run the consolidated SQLite tests**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_idempotency_scope_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_completion_idempotent_sqlite.py \
  -q
```

Expected: all selected SQLite tests pass.

- [ ] **Step 9: Run the consolidated Postgres tests**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_idempotency_scope_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_completion_idempotent_postgres.py \
  -q
```

Expected: all selected Postgres tests pass when Postgres fixture prerequisites are available. If the fixture reports Postgres unavailable, record the skip reason in `TASK-12016`.

- [ ] **Step 10: Commit**

Run:

```bash
git add \
  tldw_Server_API/tests/Jobs/parity/__init__.py \
  tldw_Server_API/tests/Jobs/parity/scenarios.py \
  tldw_Server_API/tests/Jobs/test_jobs_idempotency_scope_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_idempotency_scope_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_completion_idempotent_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_completion_idempotent_postgres.py \
  "backlog/tasks/task-12016 - Plan-Jobs-backend-parity-refactor-implementation.md"
git commit -m "test(jobs): share backend parity scenarios"
```

Expected: commit succeeds.

---

### Task 3: Add First SQLite And Postgres Parity Wrappers

**Files:**
- Create: `tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py`
- Create: `tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py`

- [ ] **Step 1: Write SQLite parity tests**

Create `tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py`:

```python
from __future__ import annotations

import pytest

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables
from tldw_Server_API.tests.Jobs.parity.scenarios import (
    run_acquire_complete_lifecycle_scenario,
    run_cancel_terminal_noop_scenario,
    run_events_outbox_create_complete_scenario,
    run_idempotent_create_preserves_original_request_ids_scenario,
    run_idempotent_create_scope_scenario,
    run_renew_stale_lease_noop_scenario,
)


@pytest.fixture()
def sqlite_manager_factory(tmp_path, monkeypatch):
    db_path = tmp_path / "jobs.db"
    ensure_jobs_tables(db_path)
    monkeypatch.setenv("JOBS_DB_PATH", str(db_path))
    monkeypatch.setenv("JOBS_DISABLE_LEASE_ENFORCEMENT", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    return lambda: JobManager(db_path)


def test_sqlite_idempotent_create_scope(sqlite_manager_factory):
    run_idempotent_create_scope_scenario(sqlite_manager_factory)


def test_sqlite_idempotent_create_preserves_request_ids(sqlite_manager_factory):
    run_idempotent_create_preserves_original_request_ids_scenario(sqlite_manager_factory)


def test_sqlite_acquire_complete_lifecycle(sqlite_manager_factory):
    run_acquire_complete_lifecycle_scenario(sqlite_manager_factory)


def test_sqlite_renew_stale_lease_noop(sqlite_manager_factory):
    run_renew_stale_lease_noop_scenario(sqlite_manager_factory)


def test_sqlite_cancel_terminal_noop(sqlite_manager_factory):
    run_cancel_terminal_noop_scenario(sqlite_manager_factory)


def test_sqlite_events_outbox_create_complete(sqlite_manager_factory):
    run_events_outbox_create_complete_scenario(sqlite_manager_factory)
```

- [ ] **Step 2: Write Postgres parity tests**

Create `tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py`:

```python
from __future__ import annotations

import pytest

psycopg = pytest.importorskip("psycopg")
pytestmark = pytest.mark.pg_jobs

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.tests.Jobs.parity.scenarios import (
    run_acquire_complete_lifecycle_scenario,
    run_cancel_terminal_noop_scenario,
    run_events_outbox_create_complete_scenario,
    run_idempotent_create_preserves_original_request_ids_scenario,
    run_idempotent_create_scope_scenario,
    run_renew_stale_lease_noop_scenario,
)


@pytest.fixture()
def postgres_manager_factory(jobs_pg_dsn, monkeypatch):
    monkeypatch.setenv("JOBS_DB_URL", jobs_pg_dsn)
    monkeypatch.setenv("JOBS_DISABLE_LEASE_ENFORCEMENT", "true")
    monkeypatch.setenv("JOBS_EVENTS_OUTBOX", "true")
    return lambda: JobManager(None, backend="postgres", db_url=jobs_pg_dsn)


def test_postgres_idempotent_create_scope(postgres_manager_factory):
    run_idempotent_create_scope_scenario(postgres_manager_factory)


def test_postgres_idempotent_create_preserves_request_ids(postgres_manager_factory):
    run_idempotent_create_preserves_original_request_ids_scenario(postgres_manager_factory)


def test_postgres_acquire_complete_lifecycle(postgres_manager_factory):
    run_acquire_complete_lifecycle_scenario(postgres_manager_factory)


def test_postgres_renew_stale_lease_noop(postgres_manager_factory):
    run_renew_stale_lease_noop_scenario(postgres_manager_factory)


def test_postgres_cancel_terminal_noop(postgres_manager_factory):
    run_cancel_terminal_noop_scenario(postgres_manager_factory)


def test_postgres_events_outbox_create_complete(postgres_manager_factory):
    run_events_outbox_create_complete_scenario(postgres_manager_factory)
```

- [ ] **Step 3: Run SQLite parity wrapper**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py -q
```

Expected: all SQLite parity tests pass.

- [ ] **Step 4: Run Postgres parity wrapper**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py -q
```

Expected: all Postgres parity tests pass when fixture prerequisites are available. If Postgres is unavailable, record the skip reason in `TASK-12016`.

- [ ] **Step 5: Commit**

Run:

```bash
git add \
  tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py \
  tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py \
  "backlog/tasks/task-12016 - Plan-Jobs-backend-parity-refactor-implementation.md"
git commit -m "test(jobs): add first backend parity wrappers"
```

Expected: commit succeeds.

---

### Task 4: Add Public API And Domain Mapping Contract Tests

**Files:**
- Create: `tldw_Server_API/tests/Jobs/test_jobs_admin_contract_sqlite.py`
- Modify: `tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_adapter.py`

- [ ] **Step 1: Write admin list/detail field contract test**

Create `tldw_Server_API/tests/Jobs/test_jobs_admin_contract_sqlite.py`:

```python
from __future__ import annotations

import os
from pathlib import Path

from fastapi.testclient import TestClient

from tldw_Server_API.app.core.Jobs.manager import JobManager
from tldw_Server_API.app.core.Jobs.migrations import ensure_jobs_tables


def _setup_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TEST_MODE", "true")
    monkeypatch.setenv("AUTH_MODE", "single_user")
    monkeypatch.setenv("MINIMAL_TEST_APP", "1")
    monkeypatch.setenv("ROUTES_STABLE_ONLY", "0")
    monkeypatch.setenv("ROUTES_ENABLE", "jobs")
    monkeypatch.setenv("PRIVILEGE_METADATA_VALIDATE_ON_STARTUP", "0")
    monkeypatch.setenv("CHATBOOKS_CORE_WORKER_ENABLED", "false")
    monkeypatch.setenv("JOBS_METRICS_GAUGES_ENABLED", "false")
    monkeypatch.setenv("JOBS_METRICS_RECONCILE_ENABLE", "false")
    monkeypatch.setenv("JOBS_WEBHOOKS_ENABLED", "false")
    monkeypatch.setenv("JOBS_DB_PATH", os.path.join(os.getcwd(), "Databases", "jobs.db"))


def _client_headers():
    from tldw_Server_API.app.core.AuthNZ.settings import get_settings, reset_settings

    reset_settings()
    from tldw_Server_API.app.main import app

    app.dependency_overrides.clear()
    return app, {"X-API-KEY": get_settings().SINGLE_USER_API_KEY}


def test_jobs_admin_list_and_detail_public_field_contract_sqlite(monkeypatch, tmp_path):
    _setup_env(monkeypatch, tmp_path)
    ensure_jobs_tables(Path(os.environ["JOBS_DB_PATH"]))
    app, headers = _client_headers()

    manager = JobManager()
    job = manager.create_job(
        domain="ps",
        queue="default",
        job_type="contract",
        payload={"hello": "world"},
        owner_user_id="user-1",
        request_id="request-1",
        trace_id="trace-1",
    )

    with TestClient(app, headers=headers) as client:
        list_response = client.get(
            "/api/v1/jobs/list",
            params={"domain": "ps", "queue": "default", "job_type": "contract", "limit": 10},
        )
        detail_response = client.get(f"/api/v1/jobs/{int(job['id'])}", params={"domain": "ps"})

    assert list_response.status_code == 200, list_response.text
    listed = list_response.json()
    assert isinstance(listed, list)
    assert len(listed) == 1
    list_item = listed[0]
    for key in {
        "id",
        "uuid",
        "domain",
        "queue",
        "job_type",
        "status",
        "priority",
        "retry_count",
        "owner_user_id",
        "created_at",
        "updated_at",
    }:
        assert key in list_item
    assert list_item["id"] == int(job["id"])
    assert list_item["uuid"] == job["uuid"]
    assert list_item["status"] == "queued"

    assert detail_response.status_code == 200, detail_response.text
    detail = detail_response.json()
    for key in {
        "id",
        "uuid",
        "domain",
        "queue",
        "job_type",
        "status",
        "payload",
        "result",
        "archived",
        "created_at",
        "updated_at",
    }:
        assert key in detail
    assert detail["id"] == int(job["id"])
    assert detail["uuid"] == job["uuid"]
    assert detail["payload"]["hello"] == "world"
    assert detail["archived"] is False
```

- [ ] **Step 2: Extend Chatbooks adapter tests with status mapping contract**

Append these tests to `tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_adapter.py`:

```python
import pytest

from tldw_Server_API.app.core.Chatbooks import jobs_adapter as chatbooks_jobs_adapter


@pytest.mark.parametrize(
    ("core_status", "expected"),
    [
        ("queued", ExportStatus.PENDING),
        ("processing", ExportStatus.IN_PROGRESS),
        ("completed", ExportStatus.COMPLETED),
        ("failed", ExportStatus.FAILED),
        ("cancelled", ExportStatus.CANCELLED),
        ("quarantined", ExportStatus.FAILED),
    ],
)
def test_apply_export_status_mapping_contract(core_status, expected):
    adapter = ChatbooksJobsAdapter(owner_user_id=None)
    job = SimpleNamespace(status=ExportStatus.PENDING)

    adapter.apply_export_status(job, {"status": core_status})

    assert job.status is expected


@pytest.mark.parametrize(
    ("core_status", "expected"),
    [
        ("queued", ImportStatus.PENDING),
        ("processing", ImportStatus.IN_PROGRESS),
        ("completed", ImportStatus.COMPLETED),
        ("failed", ImportStatus.FAILED),
        ("cancelled", ImportStatus.CANCELLED),
        ("quarantined", ImportStatus.FAILED),
    ],
)
def test_apply_import_status_mapping_contract(core_status, expected):
    adapter = ChatbooksJobsAdapter(owner_user_id=None)
    job = SimpleNamespace(status=ImportStatus.PENDING)

    adapter.apply_import_status(job, {"status": core_status})

    assert job.status is expected


def test_map_jobs_prefers_payload_chatbooks_job_id(monkeypatch):
    class FakeJobManager:
        def list_jobs(self, **kwargs):
            assert kwargs["domain"] == "chatbooks"
            assert kwargs["job_type"] == "export"
            return [
                {
                    "id": 42,
                    "uuid": "jobs-uuid",
                    "domain": "chatbooks",
                    "job_type": "export",
                    "owner_user_id": "user-1",
                    "payload": {"chatbooks_job_id": "legacy-export-id"},
                    "status": "queued",
                }
            ]

    monkeypatch.setattr(chatbooks_jobs_adapter, "_jobs_manager", lambda: FakeJobManager())
    adapter = ChatbooksJobsAdapter(owner_user_id="user-1")

    mapped = adapter.map_jobs(job_ids=["legacy-export-id"], job_type="export", limit=1)

    assert set(mapped) == {"legacy-export-id"}
    assert mapped["legacy-export-id"]["uuid"] == "jobs-uuid"
```

- [ ] **Step 3: Run API/domain contract tests**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/test_jobs_admin_contract_sqlite.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_adapter.py \
  -q
```

Expected: all selected tests pass.

- [ ] **Step 4: Commit**

Run:

```bash
git add \
  tldw_Server_API/tests/Jobs/test_jobs_admin_contract_sqlite.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_adapter.py \
  "backlog/tasks/task-12016 - Plan-Jobs-backend-parity-refactor-implementation.md"
git commit -m "test(jobs): lock public jobs mapping contracts"
```

Expected: commit succeeds.

---

### Task 5: Introduce JobsSettings Snapshot And Refresh Semantics

**Files:**
- Create: `tldw_Server_API/app/core/Jobs/settings.py`
- Create: `tldw_Server_API/tests/Jobs/test_jobs_settings.py`

- [ ] **Step 1: Write failing settings tests**

Create `tldw_Server_API/tests/Jobs/test_jobs_settings.py`:

```python
from __future__ import annotations

from tldw_Server_API.app.core.Jobs.settings import JobsSettingMode, JobsSettings


def test_jobs_settings_snapshots_construction_time_values():
    env = {
        "JOBS_DB_URL": "postgresql://example/jobs",
        "JOBS_DB_PATH": "/tmp/jobs-a.db",
        "JOBS_MAX_JSON_BYTES": "123",
        "JOBS_LEASE_MAX_SECONDS": "45",
        "JOBS_EVENTS_OUTBOX": "true",
        "JOBS_COUNTERS_ENABLED": "false",
    }

    settings = JobsSettings.from_env(env)
    env["JOBS_MAX_JSON_BYTES"] = "999"

    assert settings.db_url == "postgresql://example/jobs"
    assert settings.db_path == "/tmp/jobs-a.db"
    assert settings.max_json_bytes == 123
    assert settings.lease_max_seconds == 45
    assert settings.events_outbox_enabled is True
    assert settings.counters_enabled is False


def test_jobs_settings_refresh_reads_new_environment_values():
    env = {"JOBS_MAX_JSON_BYTES": "123", "JOBS_LEASE_MAX_SECONDS": "45"}
    settings = JobsSettings.from_env(env)
    env["JOBS_MAX_JSON_BYTES"] = "456"

    refreshed = settings.refresh(env)

    assert settings.max_json_bytes == 123
    assert refreshed.max_json_bytes == 456
    assert refreshed.lease_max_seconds == 45


def test_jobs_settings_allowed_queues_are_domain_aware():
    settings = JobsSettings.from_env(
        {
            "JOBS_ALLOWED_QUEUES": "default,low",
            "JOBS_ALLOWED_QUEUES_CHATBOOKS": "export,import",
        }
    )

    assert settings.allowed_queues_for_domain(None) == ["default", "low"]
    assert settings.allowed_queues_for_domain("chatbooks") == ["default", "low", "export", "import"]


def test_jobs_settings_classifies_known_keys():
    assert JobsSettings.setting_mode("JOBS_DB_URL") is JobsSettingMode.CONSTRUCTION_TIME
    assert JobsSettings.setting_mode("JOBS_MAX_JSON_BYTES") is JobsSettingMode.SNAPSHOT_REFRESHABLE
    assert JobsSettings.setting_mode("JOBS_ALLOWED_QUEUES_CHATBOOKS") is JobsSettingMode.OPERATION_TIME
```

- [ ] **Step 2: Run settings tests and verify failure**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/test_jobs_settings.py -q
```

Expected: fails with `ModuleNotFoundError` or import error for `tldw_Server_API.app.core.Jobs.settings`.

- [ ] **Step 3: Implement settings module**

Create `tldw_Server_API/app/core/Jobs/settings.py`:

```python
from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from collections.abc import Mapping, Sequence

from tldw_Server_API.app.core.testing import is_truthy


class JobsSettingMode(str, Enum):
    CONSTRUCTION_TIME = "construction_time"
    SNAPSHOT_REFRESHABLE = "snapshot_refreshable"
    OPERATION_TIME = "operation_time"


def _env_value(env: Mapping[str, str], key: str, default: str | None = None) -> str | None:
    value = env.get(key)
    if value is None:
        return default
    return str(value)


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = _env_value(env, key)
    if raw is None or raw == "":
        return default
    return int(raw)


def _env_bool(env: Mapping[str, str], key: str, default: bool = False) -> bool:
    raw = _env_value(env, key)
    if raw is None:
        return default
    return is_truthy(str(raw))


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class JobsSettings:
    db_url: str | None = None
    db_path: str | None = None
    max_json_bytes: int = 1_048_576
    lease_max_seconds: int = 3_600
    events_outbox_enabled: bool = False
    counters_enabled: bool = False
    allowed_queues: Sequence[str] = ()
    allowed_queues_by_domain: Sequence[tuple[str, Sequence[str]]] = ()

    CONSTRUCTION_TIME_KEYS = frozenset({"JOBS_DB_URL", "JOBS_DB_PATH"})
    SNAPSHOT_REFRESHABLE_KEYS = frozenset(
        {
            "JOBS_MAX_JSON_BYTES",
            "JOBS_LEASE_MAX_SECONDS",
            "JOBS_EVENTS_OUTBOX",
            "JOBS_COUNTERS_ENABLED",
        }
    )
    OPERATION_TIME_PREFIXES = ("JOBS_ALLOWED_QUEUES_",)
    OPERATION_TIME_KEYS = frozenset({"JOBS_ALLOWED_QUEUES"})

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "JobsSettings":
        source = os.environ if env is None else env
        domain_queues: list[tuple[str, Sequence[str]]] = []
        for key, value in source.items():
            if key.startswith("JOBS_ALLOWED_QUEUES_"):
                domain = key.removeprefix("JOBS_ALLOWED_QUEUES_").lower()
                domain_queues.append((domain, tuple(_split_csv(value))))
        domain_queues.sort(key=lambda item: item[0])
        return cls(
            db_url=_env_value(source, "JOBS_DB_URL"),
            db_path=_env_value(source, "JOBS_DB_PATH"),
            max_json_bytes=_env_int(source, "JOBS_MAX_JSON_BYTES", 1_048_576),
            lease_max_seconds=_env_int(source, "JOBS_LEASE_MAX_SECONDS", 3_600),
            events_outbox_enabled=_env_bool(source, "JOBS_EVENTS_OUTBOX", False),
            counters_enabled=_env_bool(source, "JOBS_COUNTERS_ENABLED", False),
            allowed_queues=tuple(_split_csv(_env_value(source, "JOBS_ALLOWED_QUEUES", ""))),
            allowed_queues_by_domain=tuple(domain_queues),
        )

    def refresh(self, env: Mapping[str, str] | None = None) -> "JobsSettings":
        return type(self).from_env(env)

    def allowed_queues_for_domain(self, domain: str | None) -> list[str]:
        values = list(self.allowed_queues)
        if domain:
            wanted = str(domain).upper().lower()
            for key, queues in self.allowed_queues_by_domain:
                if key == wanted:
                    values.extend(queues)
                    break
        seen: set[str] = set()
        result: list[str] = []
        for queue in values:
            if queue not in seen:
                seen.add(queue)
                result.append(queue)
        return result

    @classmethod
    def setting_mode(cls, key: str) -> JobsSettingMode:
        normalized = str(key or "").strip().upper()
        if normalized in cls.CONSTRUCTION_TIME_KEYS:
            return JobsSettingMode.CONSTRUCTION_TIME
        if normalized in cls.SNAPSHOT_REFRESHABLE_KEYS:
            return JobsSettingMode.SNAPSHOT_REFRESHABLE
        if normalized in cls.OPERATION_TIME_KEYS:
            return JobsSettingMode.OPERATION_TIME
        if any(normalized.startswith(prefix) for prefix in cls.OPERATION_TIME_PREFIXES):
            return JobsSettingMode.OPERATION_TIME
        return JobsSettingMode.OPERATION_TIME


__all__ = ["JobsSettingMode", "JobsSettings"]
```

- [ ] **Step 4: Run settings tests and verify pass**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/test_jobs_settings.py -q
```

Expected: all settings tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add \
  tldw_Server_API/app/core/Jobs/settings.py \
  tldw_Server_API/tests/Jobs/test_jobs_settings.py \
  "backlog/tasks/task-12016 - Plan-Jobs-backend-parity-refactor-implementation.md"
git commit -m "feat(jobs): define settings refresh contract"
```

Expected: commit succeeds.

---

### Task 6: Introduce Operation Command And Result Contracts

**Files:**
- Create: `tldw_Server_API/app/core/Jobs/operations/__init__.py`
- Create: `tldw_Server_API/app/core/Jobs/operations/contracts.py`
- Create: `tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py`

- [ ] **Step 1: Write failing operation contract tests**

Create `tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py`:

```python
from __future__ import annotations

import ast
from pathlib import Path

from tldw_Server_API.app.core.Jobs.operations.contracts import (
    AdmissionRejectionReason,
    AdmissionResult,
    CreateJobCommand,
    LifecycleResult,
    NoTransitionReason,
    OperationOutcome,
)


def test_create_job_command_carries_public_job_facts():
    command = CreateJobCommand(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={"a": 1},
        owner_user_id="user-1",
        idempotency_key="same",
        priority=5,
        max_retries=2,
        request_id="request-1",
        trace_id="trace-1",
    )

    assert command.domain == "chatbooks"
    assert command.queue == "default"
    assert command.payload == {"a": 1}
    assert command.request_id == "request-1"
    assert command.trace_id == "trace-1"


def test_admission_result_distinguishes_inserted_and_existing_rows():
    inserted = AdmissionResult.inserted(row={"id": 1, "status": "queued"}, durable_events=({"event_type": "job.created"},))
    existing = AdmissionResult.existing(row={"id": 1, "status": "queued"})
    rejected = AdmissionResult.rejected(AdmissionRejectionReason.QUEUE_PAUSED)

    assert inserted.outcome is OperationOutcome.APPLIED
    assert inserted.inserted is True
    assert existing.outcome is OperationOutcome.NO_TRANSITION
    assert existing.inserted is False
    assert existing.no_transition_reason is NoTransitionReason.IDEMPOTENT_EXISTING
    assert rejected.outcome is OperationOutcome.ADMISSION_REJECTED
    assert rejected.admission_rejection_reason is AdmissionRejectionReason.QUEUE_PAUSED


def test_lifecycle_result_names_no_transition_reason():
    result = LifecycleResult.no_transition(NoTransitionReason.STALE_LEASE, row={"id": 1, "status": "processing"})

    assert result.outcome is OperationOutcome.NO_TRANSITION
    assert result.no_transition_reason is NoTransitionReason.STALE_LEASE
    assert result.transition_applied is False


def test_operation_contracts_do_not_import_job_manager():
    path = Path("tldw_Server_API/app/core/Jobs/operations/contracts.py")
    tree = ast.parse(path.read_text())
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)

    assert "tldw_Server_API.app.core.Jobs.manager" not in imports
    assert not any(import_name.endswith(".Jobs.manager") for import_name in imports)
```

- [ ] **Step 2: Run operation contract tests and verify failure**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py -q
```

Expected: fails with an import error for `tldw_Server_API.app.core.Jobs.operations.contracts`.

- [ ] **Step 3: Create operation package marker**

Create `tldw_Server_API/app/core/Jobs/operations/__init__.py`:

```python
"""Backend-specific Jobs operation contracts and implementations."""
```

- [ ] **Step 4: Implement operation contracts**

Create `tldw_Server_API/app/core/Jobs/operations/contracts.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections.abc import Sequence
from typing import Any


class OperationOutcome(str, Enum):
    APPLIED = "applied"
    NO_TRANSITION = "no_transition"
    ADMISSION_REJECTED = "admission_rejected"
    BACKEND_CONFLICT = "backend_conflict"
    BACKEND_SCHEMA_ERROR = "backend_schema_error"
    BACKEND_ERROR = "backend_error"


class NoTransitionReason(str, Enum):
    MISSING = "missing"
    WRONG_STATUS = "wrong_status"
    STALE_LEASE = "stale_lease"
    ALREADY_TERMINAL = "already_terminal"
    IDEMPOTENT_EXISTING = "idempotent_existing"
    RLS_FILTERED = "rls_filtered"


class AdmissionRejectionReason(str, Enum):
    QUEUE_PAUSED = "queue_paused"
    QUEUE_DRAINING = "queue_draining"
    QUOTA_EXCEEDED = "quota_exceeded"
    FAIR_SHARE_LIMIT = "fair_share_limit"
    DEPENDENCY_BLOCKED = "dependency_blocked"
    POLICY_REJECTED = "policy_rejected"


@dataclass(frozen=True)
class CreateJobCommand:
    domain: str
    queue: str
    job_type: str
    payload: dict[str, Any] | None
    owner_user_id: str | None
    idempotency_key: str | None = None
    priority: int = 100
    max_retries: int = 3
    available_at: datetime | None = None
    project_id: str | None = None
    batch_group: str | None = None
    request_id: str | None = None
    trace_id: str | None = None


@dataclass(frozen=True)
class AdmissionResult:
    outcome: OperationOutcome
    row: dict[str, Any] | None = None
    inserted: bool = False
    no_transition_reason: NoTransitionReason | None = None
    admission_rejection_reason: AdmissionRejectionReason | None = None
    durable_events: Sequence[dict[str, Any]] = field(default_factory=tuple)
    message: str | None = None

    @classmethod
    def inserted(cls, *, row: dict[str, Any], durable_events: Sequence[dict[str, Any]] = ()) -> "AdmissionResult":
        return cls(
            outcome=OperationOutcome.APPLIED,
            row=row,
            inserted=True,
            durable_events=durable_events,
        )

    @classmethod
    def existing(cls, *, row: dict[str, Any]) -> "AdmissionResult":
        return cls(
            outcome=OperationOutcome.NO_TRANSITION,
            row=row,
            inserted=False,
            no_transition_reason=NoTransitionReason.IDEMPOTENT_EXISTING,
        )

    @classmethod
    def rejected(cls, reason: AdmissionRejectionReason, *, message: str | None = None) -> "AdmissionResult":
        return cls(
            outcome=OperationOutcome.ADMISSION_REJECTED,
            admission_rejection_reason=reason,
            message=message,
        )


@dataclass(frozen=True)
class LifecycleResult:
    outcome: OperationOutcome
    transition_applied: bool
    row: dict[str, Any] | None = None
    no_transition_reason: NoTransitionReason | None = None
    durable_events: Sequence[dict[str, Any]] = field(default_factory=tuple)
    message: str | None = None

    @classmethod
    def applied(cls, *, row: dict[str, Any], durable_events: Sequence[dict[str, Any]] = ()) -> "LifecycleResult":
        return cls(
            outcome=OperationOutcome.APPLIED,
            transition_applied=True,
            row=row,
            durable_events=durable_events,
        )

    @classmethod
    def no_transition(
        cls,
        reason: NoTransitionReason,
        *,
        row: dict[str, Any] | None = None,
        message: str | None = None,
    ) -> "LifecycleResult":
        return cls(
            outcome=OperationOutcome.NO_TRANSITION,
            transition_applied=False,
            row=row,
            no_transition_reason=reason,
            message=message,
        )


__all__ = [
    "AdmissionRejectionReason",
    "AdmissionResult",
    "CreateJobCommand",
    "LifecycleResult",
    "NoTransitionReason",
    "OperationOutcome",
]
```

- [ ] **Step 5: Run operation contract tests and verify pass**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && RUN_JOBS=1 python -m pytest tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py -q
```

Expected: all operation contract tests pass.

- [ ] **Step 6: Verify contracts do not import manager**

Run:

```bash
rg -n "JobManager|Jobs\\.manager|from .*manager" tldw_Server_API/app/core/Jobs/operations
```

Expected: no matches.

- [ ] **Step 7: Commit**

Run:

```bash
git add \
  tldw_Server_API/app/core/Jobs/operations/__init__.py \
  tldw_Server_API/app/core/Jobs/operations/contracts.py \
  tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py \
  "backlog/tasks/task-12016 - Plan-Jobs-backend-parity-refactor-implementation.md"
git commit -m "feat(jobs): define operation result contracts"
```

Expected: commit succeeds.

---

### Task 7: Final Verification And Handoff

**Files:**
- Modify: `backlog/tasks/task-12016 - Plan-Jobs-backend-parity-refactor-implementation.md`

- [ ] **Step 1: Run focused SQLite and contract test suite**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/parity/test_sqlite_parity.py \
  tldw_Server_API/tests/Jobs/test_jobs_idempotency_scope_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_completion_idempotent_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_admin_contract_sqlite.py \
  tldw_Server_API/tests/Jobs/test_jobs_settings.py \
  tldw_Server_API/tests/Jobs/test_jobs_operation_contracts.py \
  tldw_Server_API/tests/Chatbooks/test_chatbooks_jobs_adapter.py \
  -q
```

Expected: all selected SQLite and contract tests pass.

- [ ] **Step 2: Run focused Postgres parity suite**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && RUN_JOBS=1 python -m pytest \
  tldw_Server_API/tests/Jobs/parity/test_postgres_parity.py \
  tldw_Server_API/tests/Jobs/test_jobs_idempotency_scope_postgres.py \
  tldw_Server_API/tests/Jobs/test_jobs_completion_idempotent_postgres.py \
  -q
```

Expected: all selected Postgres tests pass when fixture prerequisites are available. If Postgres is unavailable, record the fixture skip or failure exactly in `TASK-12016`.

- [ ] **Step 3: Run Bandit on touched implementation paths**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/Jobs/settings.py \
  tldw_Server_API/app/core/Jobs/operations \
  -f json -o /tmp/bandit_jobs_backend_parity_first_slice.json
```

Expected: Bandit exits `0` or reports no new findings in touched implementation code. If Bandit exits nonzero because the module is not installed, install/use the project dev environment or record the environment blocker in `TASK-12016`.

- [ ] **Step 4: Run whitespace and syntax checks**

Run:

```bash
git diff --check
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate && python -m py_compile \
  tldw_Server_API/app/core/Jobs/settings.py \
  tldw_Server_API/app/core/Jobs/operations/contracts.py \
  tldw_Server_API/tests/Jobs/parity/scenarios.py
```

Expected: both commands exit with status `0`.

- [ ] **Step 5: Update Backlog final summary**

Update `TASK-12016` with this final summary shape, replacing bracketed status text with actual verification results:

```text
Implemented the first Jobs backend parity refactor slice. Added a direct-SQL/domain mapping inventory, shared SQLite/Postgres parity scenarios, admin list/detail field contract coverage, Chatbooks status/id mapping contract coverage, explicit JobsSettings snapshot/refresh semantics, and operation command/result contracts. Verification: [focused SQLite tests result]; [focused Postgres tests result or fixture skip]; [Bandit result]; git diff --check passed.
```

- [ ] **Step 6: Final commit**

Run:

```bash
git add "backlog/tasks/task-12016 - Plan-Jobs-backend-parity-refactor-implementation.md"
git commit -m "chore(jobs): record parity slice verification"
```

Expected: commit succeeds if the Backlog task changed.

- [ ] **Step 7: Rebase and inspect PR diff**

Run:

```bash
git fetch origin dev
git rebase dev
git status --short
git rev-list --count HEAD..dev
git rev-list --count dev..HEAD
git diff --name-status dev..HEAD
```

Expected:

```text
git status --short prints no output
git rev-list --count HEAD..dev prints 0
git rev-list --count dev..HEAD prints N, where N is the number of commits on this branch
git diff --name-status dev..HEAD shows only the files named in this plan
```

---

## Review Checklist

- Parity tests are added before production SQL extraction.
- Existing paired SQLite/Postgres tests delegate to shared scenarios where this plan touches the same behavior.
- Public API tests assert fields and mapping behavior, not full snapshots.
- Chatbooks non-identity status/id mapping is covered in the first slice.
- Direct runtime Jobs SQL is inventoried and classified.
- `JobsSettings` makes env snapshot/refresh semantics explicit before manager integration.
- Operation contracts do not import `JobManager`.
- Bandit runs on touched implementation paths before completion.
