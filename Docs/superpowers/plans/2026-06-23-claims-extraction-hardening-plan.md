# Claims_Extraction Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the validated Claims_Extraction review findings from TASK-9934 with failing-first tests and keep the larger module refactor as a documented follow-up.

**Architecture:** Keep endpoint and public service entry points stable. Add narrowly scoped helpers only where they reduce risk: runtime bounds stay in `runtime_config`, rebuild replacement is guarded in `claims_rebuild_service`, notification dispatch uses one bounded semaphore helper from `claims_notifications`, and analytics owner scoping is centralized inside `claims_service` before later extraction to a dedicated analytics module.

**Tech Stack:** Python 3.14, FastAPI service layer, pytest, SQLite-backed MediaDatabase tests, Loguru, Bandit.

---

## File Structure

- Modify: `tldw_Server_API/app/core/Claims_Extraction/runtime_config.py`
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_service.py`
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_engine.py`
- Modify: `tldw_Server_API/app/core/Claims_Extraction/ingestion_claims.py`
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_rebuild_service.py`
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_notifications.py`
- Modify: `tldw_Server_API/app/core/Claims_Extraction/fva_pipeline.py`
- Create: `tldw_Server_API/tests/Claims/test_claims_cancellation_and_timeout.py`
- Modify: `tldw_Server_API/tests/Claims/test_claims_rebuild_service_failure.py`
- Modify: `tldw_Server_API/tests/Claims/test_claims_runtime_config.py`
- Modify: `tldw_Server_API/tests/Claims/test_claims_review_notifications.py`
- Modify: `tldw_Server_API/tests/Claims/test_claims_dashboard_analytics.py`
- Modify: `tldw_Server_API/tests/Claims_Extraction/test_fva_pipeline.py`
- Update: `backlog/tasks/task-9934 - Harden-Claims_Extraction-review-findings-and-refactor-design.md`

## Task 1: Runtime Bounds, Email Escaping, And FVA Metric Branch

**Files:**
- Modify: `tldw_Server_API/tests/Claims/test_claims_runtime_config.py`
- Modify: `tldw_Server_API/tests/Claims/test_claims_review_notifications.py`
- Modify: `tldw_Server_API/tests/Claims_Extraction/test_fva_pipeline.py`
- Modify: `tldw_Server_API/app/core/Claims_Extraction/runtime_config.py`
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_notifications.py`
- Modify: `tldw_Server_API/app/core/Claims_Extraction/fva_pipeline.py`

- [ ] **Step 1: Write failing runtime bounds tests**

Add assertions to `test_resolve_claims_context_window_chars_and_passes_are_bounded`:

```python
assert resolve_claims_context_window_chars({"CLAIMS_CONTEXT_WINDOW_CHARS": "999999"}) == 20000
assert resolve_claims_extraction_passes({"CLAIMS_EXTRACTION_PASSES": "999"}) == 10
```

- [ ] **Step 2: Write failing email escaping test**

Add this unit test in `test_claims_review_notifications.py`:

```python
def test_build_review_email_bodies_escapes_html() -> None:
    html_body, text_body = claims_notifications._build_review_email_bodies(
        [
            {
                "kind": "review_update",
                "created_at": "2026-06-23T00:00:00Z",
                "payload": {
                    "new_status": "approved",
                    "claim_text": "<script>alert(1)</script>",
                },
            }
        ]
    )

    assert "<script>" not in html_body
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html_body
    assert "<script>alert(1)</script>" in text_body
```

- [ ] **Step 3: Write failing FVA metric branch test**

In `test_fva_pipeline.py`, add a test that returns anti-context documents, patches `observe_histogram`, and asserts exactly three `fva_adjudication_scores` observations are made for support, contradict, and contestation scores. The test must fail on the current code because the histogram calls are in the `else` branch where `adjudication` is `None`.

- [ ] **Step 4: Run RED checks**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Claims/test_claims_runtime_config.py \
  tldw_Server_API/tests/Claims/test_claims_review_notifications.py \
  tldw_Server_API/tests/Claims_Extraction/test_fva_pipeline.py
```

Expected: failures for the new runtime max, HTML escaping, and FVA adjudication score assertions.

- [ ] **Step 5: Implement minimal production fixes**

In `runtime_config.py`, define exported constants:

```python
CLAIMS_CONTEXT_WINDOW_CHARS_MAX = 20000
CLAIMS_EXTRACTION_PASSES_MAX = 10
```

Clamp `resolve_claims_context_window_chars` with `min(CLAIMS_CONTEXT_WINDOW_CHARS_MAX, max(0, value))` and `resolve_claims_extraction_passes` with `min(CLAIMS_EXTRACTION_PASSES_MAX, max(1, value))`.

In `claims_notifications.py`, import `html` and escape `summary` before appending to `html_lines`:

```python
html_lines.append(f"<li>{html.escape(summary)}</li>")
```

In `fva_pipeline.py`, move the three adjudication score histogram calls from the `else` branch into the `if anti_docs` branch after `adjudication` is created.

- [ ] **Step 6: Run GREEN checks**

Run the same pytest command from Step 4. Expected: all selected tests pass.

## Task 2: Rebuild Replacement Safety

**Files:**
- Modify: `tldw_Server_API/tests/Claims/test_claims_rebuild_service_failure.py`
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_rebuild_service.py`

- [ ] **Step 1: Write failing SQLite rollback test**

Add a test that creates a real temporary `MediaDatabase`, inserts one existing active claim, patches `extract_claims_for_chunks` to return one replacement claim, patches `store_claims` to return `0`, and calls `ClaimsRebuildService._process_task`. The test asserts `_process_task` raises `RuntimeError` and the original claim remains active with `deleted = 0`.

- [ ] **Step 2: Update existing fake DB test for transaction support**

Add this method to the `_FakeDb` class used by `test_claims_rebuild_service_process_task_uses_managed_media_database`:

```python
@contextmanager
def transaction(self):
    yield self
```

- [ ] **Step 3: Run RED check**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Claims/test_claims_rebuild_service_failure.py
```

Expected: the new rollback test fails because the current code accepts `store_claims == 0` after soft-delete.

- [ ] **Step 4: Implement rebuild transaction guard**

In `_process_task`, wrap delete and store in `with db.transaction():`. Keep `deleted = db.soft_delete_claims_for_media(task.media_id)`, call `store_claims`, and if `inserted <= 0` for non-empty `claims`, raise:

```python
raise RuntimeError(f"Claims rebuild stored zero replacement claims for media_id={task.media_id}")
```

- [ ] **Step 5: Run GREEN check**

Run the pytest command from Step 3. Expected: all rebuild service tests pass.

## Task 3: Cancellation Propagation And LLM Timeout

**Files:**
- Create: `tldw_Server_API/tests/Claims/test_claims_cancellation_and_timeout.py`
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_engine.py`
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_service.py`
- Modify: `tldw_Server_API/app/core/Claims_Extraction/ingestion_claims.py`

- [ ] **Step 1: Write failing cancellation tuple tests**

Create tests that assert:

```python
assert asyncio.CancelledError not in claims_engine._CLAIMS_ENGINE_NONCRITICAL_EXCEPTIONS
assert asyncio.CancelledError not in claims_service._CLAIMS_NONCRITICAL_EXCEPTIONS
```

- [ ] **Step 2: Write failing timeout executor test**

Patch provider setup so `_llm_extract_claim_texts` reaches the provider call, and monkeypatch `concurrent.futures.ThreadPoolExecutor` because the function imports `concurrent.futures` locally. The fake future raises `concurrent.futures.TimeoutError` from `result`, and the fake executor records `shutdown(wait=False, cancel_futures=True)`. Assert `_llm_extract_claim_texts(...)` returns `[]` and does not call a context-manager `__exit__`.

- [ ] **Step 3: Run RED check**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Claims/test_claims_cancellation_and_timeout.py
```

Expected: cancellation tuple assertions fail and the timeout fake exposes the current context-manager shutdown behavior.

- [ ] **Step 4: Implement cancellation and timeout fixes**

Remove `asyncio.CancelledError` from the two noncritical exception tuples. In `ingestion_claims._llm_extract_claim_texts`, replace `with ThreadPoolExecutor(...) as _exec:` with explicit executor lifecycle:

```python
_exec = _futures.ThreadPoolExecutor(max_workers=1)
try:
    fut = _exec.submit(_call_provider)
    resp = fut.result(timeout=timeout_sec)
except _futures.TimeoutError:
    with contextlib.suppress(_CLAIMS_NONCRITICAL_EXCEPTIONS):
        fut.cancel()
    _exec.shutdown(wait=False, cancel_futures=True)
    ...
    return []
except _CLAIMS_NONCRITICAL_EXCEPTIONS:
    _exec.shutdown(wait=False, cancel_futures=True)
    raise
else:
    _exec.shutdown(wait=True)
```

Keep existing metrics and fallback recording unchanged.

- [ ] **Step 5: Run GREEN check**

Run the pytest command from Step 3. Expected: all new cancellation and timeout tests pass.

## Task 4: Dashboard Analytics Owner Scoping

**Files:**
- Modify: `tldw_Server_API/tests/Claims/test_claims_dashboard_analytics.py`
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_service.py`

- [ ] **Step 1: Write failing owner-scope analytics test**

Add a test that seeds two media rows with different owner identifiers and claims/review log entries for both. Call `_build_claims_analytics(db, owner_user_id="1", window_days=7)` and assert totals, per-media top rows, review throughput, status trends, and orphan claim count exclude owner `"2"`.

- [ ] **Step 2: Run RED check**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Claims/test_claims_dashboard_analytics.py
```

Expected: new owner-scope assertions fail because current helper SQL is mostly unscoped.

- [ ] **Step 3: Implement centralized owner-scope SQL helpers**

Add private helpers in `claims_service.py`:

```python
def _claims_owner_filter_sql(owner_user_id: str | None, *, media_alias: str = "m") -> tuple[str, tuple[Any, ...]]:
    if not owner_user_id:
        return "", ()
    return f" AND COALESCE(CAST({media_alias}.owner_user_id AS TEXT), {media_alias}.client_id) = ?", (str(owner_user_id),)
```

Join `Claims` to `Media` in status counts, review latency, throughput, status trends, claims-per-media stats, and cluster orphan/hotspot subqueries. Append the returned SQL fragment and parameters to each query.

- [ ] **Step 4: Run GREEN check**

Run the pytest command from Step 2. Expected: dashboard analytics tests pass.

## Task 5: Bounded Notification Dispatch

**Files:**
- Modify: `tldw_Server_API/tests/Claims/test_claims_review_notifications.py`
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_notifications.py`
- Modify: `tldw_Server_API/app/core/Claims_Extraction/claims_service.py`

- [ ] **Step 1: Write failing review dispatcher tests**

Add tests that patch a new `claims_notifications.submit_claims_notification_delivery` helper to run immediately for normal dispatch and to reject work when saturated. Assert normal dispatch marks notifications delivered and saturated dispatch logs/records failure without spawning a raw daemon thread.

- [ ] **Step 2: Write failing alert dispatcher test**

Add a claims service test that patches `claims_service.submit_claims_notification_delivery` to return `False` and patches `claims_service.threading.Thread` to raise `AssertionError` if used. Call `_dispatch_claims_alert_notifications` with webhook settings and assert no raw thread is created.

- [ ] **Step 3: Run RED checks**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Claims/test_claims_review_notifications.py
```

Expected: new dispatcher helper assertions fail because current code starts daemon threads directly.

- [ ] **Step 4: Implement shared bounded dispatcher helper**

In `claims_notifications.py`, add:

```python
from threading import BoundedSemaphore

_CLAIMS_NOTIFICATION_MAX_PENDING = 32
_notification_slots = BoundedSemaphore(_CLAIMS_NOTIFICATION_MAX_PENDING)

def submit_claims_notification_delivery(fn, *args, **kwargs) -> bool:
    if not _notification_slots.acquire(blocking=False):
        logger.warning("Claims notification dispatch queue is full")
        return False

    def _run() -> None:
        try:
            fn(*args, **kwargs)
        finally:
            _notification_slots.release()

    threading.Thread(target=_run, daemon=True).start()
    return True
```

This keeps the existing daemon-thread shutdown behavior but caps concurrent pending deliveries.

- [ ] **Step 5: Route dispatch through helper**

Change `dispatch_claim_review_notifications` to call `submit_claims_notification_delivery(_deliver)`. Import `submit_claims_notification_delivery` into `claims_service.py` and use it in `_dispatch_claims_alert_notifications` instead of `threading.Thread(...)`.

- [ ] **Step 6: Run GREEN checks**

Run the pytest command from Step 3 and the claims service alert test file that contains the new alert dispatcher test. Expected: notification dispatch tests pass.

## Task 6: Full Verification, Bandit, And Task Closeout

**Files:**
- Update: `backlog/tasks/task-9934 - Harden-Claims_Extraction-review-findings-and-refactor-design.md`

- [ ] **Step 1: Run targeted Claims_Extraction test suite**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Claims/test_claims_rebuild_service_failure.py \
  tldw_Server_API/tests/Claims/test_claims_runtime_config.py \
  tldw_Server_API/tests/Claims/test_claims_review_notifications.py \
  tldw_Server_API/tests/Claims/test_claims_dashboard_analytics.py \
  tldw_Server_API/tests/Claims/test_claims_cancellation_and_timeout.py \
  tldw_Server_API/tests/Claims_Extraction/test_fva_pipeline.py
```

Expected: all selected tests pass.

- [ ] **Step 2: Run Bandit on touched Claims_Extraction files**

Run:

```bash
source /Users/appledev/Documents/GitHub/tldw_server/.venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Claims_Extraction/runtime_config.py \
  tldw_Server_API/app/core/Claims_Extraction/claims_service.py \
  tldw_Server_API/app/core/Claims_Extraction/claims_engine.py \
  tldw_Server_API/app/core/Claims_Extraction/ingestion_claims.py \
  tldw_Server_API/app/core/Claims_Extraction/claims_rebuild_service.py \
  tldw_Server_API/app/core/Claims_Extraction/claims_notifications.py \
  tldw_Server_API/app/core/Claims_Extraction/fva_pipeline.py \
  -f json -o /tmp/bandit_claims_extraction_9934.json
```

Expected: no new high or medium findings in touched code.

- [ ] **Step 3: Update Backlog task**

Use `backlog task edit TASK-9934` to check completed acceptance criteria, append verification commands/results, and add the final summary.

- [ ] **Step 4: Review git diff**

Run:

```bash
git diff --stat
git diff -- tldw_Server_API/app/core/Claims_Extraction tldw_Server_API/tests/Claims tldw_Server_API/tests/Claims_Extraction Docs/superpowers/plans backlog/tasks
```

Expected: diff is limited to TASK-9934 plan, fixes, tests, and task metadata.

- [ ] **Step 5: Commit implementation**

Run:

```bash
git add \
  Docs/superpowers/plans/2026-06-23-claims-extraction-hardening-plan.md \
  "backlog/tasks/task-9934 - Harden-Claims_Extraction-review-findings-and-refactor-design.md" \
  tldw_Server_API/app/core/Claims_Extraction \
  tldw_Server_API/tests/Claims \
  tldw_Server_API/tests/Claims_Extraction
git commit -m "fix: harden Claims_Extraction review findings"
```
