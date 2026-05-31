# Watchlists P0 Demo Blockers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix only the newly verified `/watchlists` demo blockers so the demo can truthfully show digest generation plus audio briefing status from inside `/watchlists`.

**Architecture:** Keep the 2026-05-18 Watchlists PRD and implementation plan as the parent source of truth. Apply this plan as a P0 addendum: reuse the existing Watchlists API, Scheduler, Workflows artifact bridge, and shared UI, and patch the queue/status/selection/polling gaps that blocked the live dry run.

**Tech Stack:** FastAPI/Pydantic, existing Scheduler and Watchlists DB helpers, React/TypeScript shared UI in `apps/packages/ui`, Vitest, pytest, Bandit, and browser-observed smoke verification.

---

## Source Documents

- Parent PRD: `Docs/superpowers/specs/2026-05-18-watchlists-digest-audio-briefing-prd-design.md`
- Parent implementation plan: `Docs/superpowers/plans/2026-05-18-watchlists-digest-audio-briefing-implementation-plan.md`
- P0 addendum: `Docs/superpowers/specs/2026-05-22-watchlists-staged-demo-remediation-design.md`
- Readiness runbook: `Docs/Runbooks/watchlists_demo_readiness_2026_05_20.md`
- Backlog: `TASK-477`

## Scope Guard

Implement only:

1. Workflows queue worker availability for Watchlists audio tasks.
2. Structured audio trigger results for all non-submitted paths.
3. Meaningful `/runs/{run_id}/audio` status when Workflows DB/artifacts do not exist yet.
4. Live Reports audio polling for digest outputs with requested audio.
5. Better initial watchlist selection when the first API item is inactive/imported.
6. Focused tests plus demo smoke verification.

Do not reopen the completed 2026-05-18 PRD checklist unless one of these blockers requires a small touched-file correction.

## File Structure And Ownership

### Backend Audio Trigger Contract

- Modify: `tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py`
  - Owns `AudioBriefingTriggerResult`.
  - Owns TTS default resolution and configuration gate.
  - Owns `workflows` queue scaling before Scheduler submit.
- Modify: `tldw_Server_API/app/core/Watchlists/pipeline.py`
  - Consumes structured trigger result after monitor runs.
  - Persists run stats for task id and non-submission reason.
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
  - Consumes structured trigger result after explicit output creation.
  - Persists output metadata and run stats consistently.

### Backend Audio Status Bridge

- Modify: `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
  - Adds `queue_name` and any needed status fields to `WatchlistRunAudioResponse`.
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
  - Replaces `no_workflow_db` hard 404 with Scheduler-backed pending/queued/running/failed status when a task id exists.

### Frontend Status And Selection

- Modify: `apps/packages/ui/src/types/watchlists.ts`
  - Adds `queue_name` to `WatchlistRunAudioStatus`.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx`
  - Polls `getWatchlistRunAudio(run_id)` for requested audio on text digest outputs.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts`
  - Merges live audio status over stale output metadata.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx`
  - Uses a pure fallback selector instead of `items[0]`.
- Create: `apps/packages/ui/src/components/Option/Watchlists/watchlist-selection.ts`
  - Owns deterministic selection scoring.
- Test: `apps/packages/ui/src/components/Option/Watchlists/__tests__/watchlist-selection.test.ts`
  - Covers active/imported/current-selection fallback rules without rendering the full page.

---

## Task 1: Structured Audio Trigger Result And Workflows Queue Worker

**Files:**
- Modify: `tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py`
- Modify: `tldw_Server_API/app/core/Watchlists/pipeline.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Test: `tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py`
- Test: `tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py`

- [ ] **Step 1: Write failing trigger-result tests**

In `test_audio_briefing_workflow.py`, update existing `None` expectations and add coverage for:

```python
result = await trigger_audio_briefing(
    user_id=1,
    job_id=42,
    run_id=7,
    output_prefs={"generate_audio": True, "audio_model": "kitten", "audio_voice": "expr-voice-2-f"},
    db=db,
    scheduler=mock_scheduler,
)
assert result.status == "submitted"
assert result.task_id == "task_abc123"
```

Add queue failure coverage:

```python
mock_scheduler.scale_workers = AsyncMock(return_value=0)
result = await trigger_audio_briefing(...)
assert result.status == "queue_unavailable"
assert result.task_id is None
mock_scheduler.submit.assert_not_called()
```

Add no-items and scheduler-submit-failure coverage:

```python
assert result.status == "skipped_no_items"
assert result.reason == "no_ingested_items"

assert result.status == "enqueue_failed"
assert result.reason == "scheduler_submit_failed"
```

Add default-resolution coverage so the hardcoded Kokoro fallback cannot return:

```python
result = await trigger_audio_briefing(
    user_id=1,
    job_id=42,
    run_id=7,
    output_prefs={"generate_audio": True, "audio_voice": "Bella"},
    db=db,
    scheduler=mock_scheduler,
)
payload = mock_scheduler.submit.call_args.kwargs["payload"]
assert payload["inputs"]["tts_model"] != "kokoro"
assert payload["inputs"]["tts_model"] == "KittenML/kitten-tts-nano-0.8"
assert payload["inputs"]["tts_voice"] == "Bella"
```

- [ ] **Step 2: Run backend trigger tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py -q
```

Expected: FAIL because `trigger_audio_briefing()` currently returns `str | None`, does not call `scale_workers()`, and treats queue failures as `None`.

- [ ] **Step 3: Add structured result object**

Add near the top of `audio_briefing_workflow.py`:

```python
from dataclasses import dataclass
from typing import Literal

AudioBriefingTriggerStatus = Literal[
    "disabled",
    "submitted",
    "skipped_no_items",
    "configuration_required",
    "queue_unavailable",
    "enqueue_failed",
]

@dataclass(frozen=True)
class AudioBriefingTriggerResult:
    status: AudioBriefingTriggerStatus
    task_id: str | None = None
    reason: str | None = None

    @property
    def submitted(self) -> bool:
        return self.status == "submitted" and bool(self.task_id)
```

Update `trigger_audio_briefing()` to return `AudioBriefingTriggerResult` for every path.

- [ ] **Step 4: Replace hardcoded Kokoro defaults with shared TTS default resolution**

Add a small helper in `audio_briefing_workflow.py` that uses the existing TTS resolver instead of `output_prefs.get("audio_model") or "kokoro"`:

```python
from tldw_Server_API.app.core.TTS.tts_request_resolution import resolve_tts_request_defaults

def _resolve_audio_briefing_tts_defaults(output_prefs: dict[str, Any]) -> tuple[str, str] | None:
    try:
        resolved_tts = resolve_tts_request_defaults(
            provider=output_prefs.get("audio_provider"),
            model=output_prefs.get("audio_model"),
            voice=output_prefs.get("audio_voice"),
        )
    except Exception:
        return None
    if not resolved_tts.model or not resolved_tts.voice:
        return None
    return resolved_tts.model, resolved_tts.voice
```

Call that helper inside `trigger_audio_briefing()` before building workflow inputs:

```python
resolved_tts = _resolve_audio_briefing_tts_defaults(output_prefs)
if resolved_tts is None:
    return AudioBriefingTriggerResult(
        status="configuration_required",
        reason="tts_defaults_unavailable",
    )
workflow_inputs = _build_workflow_inputs(items, output_prefs, resolved_tts=resolved_tts)
```

Update `_build_workflow_inputs(..., resolved_tts: tuple[str, str])` to use the resolved model and voice:

```python
tts_model, tts_voice = resolved_tts
...
"tts_model": tts_model,
"tts_voice": tts_voice,
```

Keep explicitly configured model and voice values unchanged. This matches the existing `/api/v1/audio/speech` default-resolution path, removes the broken Kokoro assumption, and avoids forcing current `/watchlists` audio toggles into `configuration_required` merely because the UI has historically saved voice but not model.

Do not add a full provider catalog/preflight UI in this P0 task. That belongs in the follow-up status UX PR unless the implementation finds the existing resolver is insufficient to make the demo path runnable.

- [ ] **Step 5: Ensure a `workflows` worker before submit**

After `scheduler = await get_global_scheduler()` or injected scheduler resolution:

```python
try:
    worker_count = await scheduler.scale_workers(1, "workflows")
except Exception as exc:
    logger.warning("Audio briefing: workflows queue unavailable for run {} (error_type={})", run_id, type(exc).__name__)
    return AudioBriefingTriggerResult(status="queue_unavailable", reason="workflows_queue_scale_failed")

if worker_count < 1:
    return AudioBriefingTriggerResult(status="queue_unavailable", reason="workflows_queue_has_no_workers")
```

Then call `scheduler.submit(...)` exactly as today. On success:

```python
return AudioBriefingTriggerResult(status="submitted", task_id=task_id)
```

On submit failure:

```python
return AudioBriefingTriggerResult(status="enqueue_failed", reason="scheduler_submit_failed")
```

- [ ] **Step 6: Update pipeline call site**

In `pipeline.py`, replace truthiness of `audio_task_id` with structured result handling:

```python
audio_result = await trigger_audio_briefing(...)
stats["audio_briefing_status"] = audio_result.status
if audio_result.task_id:
    stats["audio_briefing_task_id"] = audio_result.task_id
if audio_result.reason:
    stats["audio_briefing_reason"] = audio_result.reason
```

Only write `audio_briefing_task_id` when `audio_result.submitted` is true.

- [ ] **Step 7: Update explicit output creation call site**

In `watchlists.py`, replace `audio_task_id` handling with:

```python
audio_result = await trigger_audio_briefing(...)
metadata["audio_briefing_status"] = audio_result.status
if audio_result.task_id:
    metadata["audio_briefing_task_id"] = audio_result.task_id
if audio_result.reason:
    metadata["audio_briefing_reason"] = audio_result.reason
```

Update run stats the same way, preserving the task id only when present. Do not collapse `queue_unavailable`, `configuration_required`, or `enqueue_failed` into `skipped`.

- [ ] **Step 8: Run focused backend tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py -q
```

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py tldw_Server_API/app/core/Watchlists/pipeline.py tldw_Server_API/app/api/v1/endpoints/watchlists.py tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py
git commit -m "fix: make watchlists audio trigger status explicit"
```

---

## Task 2: Run Audio Status Fallback When Workflows DB Is Missing

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Modify: `apps/packages/ui/src/types/watchlists.ts`
- Test: `tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py`
- Test: `apps/packages/ui/src/services/__tests__/watchlists-audio.test.ts`

- [ ] **Step 1: Write failing backend endpoint tests**

In `test_audio_output_delivery.py`, add:

```python
async def test_returns_queued_when_workflows_db_missing_but_scheduler_task_exists(...):
    run.stats_json = json.dumps({"audio_briefing_task_id": "task_queued"})
    scheduler = MagicMock()
    scheduler.get_task = AsyncMock(return_value=SimpleNamespace(
        id="task_queued",
        status=TaskStatus.QUEUED,
        queue_name="workflows",
        error=None,
    ))
    ...
    result = await get_run_audio(...)
    assert result["status"] == "queued"
    assert result["task_id"] == "task_queued"
    assert result["queue_name"] == "workflows"
    assert result["audio_uri"] is None
```

Add unavailable scheduler coverage:

```python
assert result["status"] == "pending"
assert result["fallback_reason"] == "workflow_run_not_started"
```

- [ ] **Step 2: Write failing frontend type/service test**

In `watchlists-audio.test.ts`, add a fixture that includes `queue_name: "workflows"` and assert the helper returns it.

- [ ] **Step 3: Run tests to verify failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py -q

cd apps/packages/ui
bunx vitest run src/services/__tests__/watchlists-audio.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL because `no_workflow_db` is a hard 404 and `queue_name` is not in the backend/frontend response types.

- [ ] **Step 4: Add `queue_name` to schemas and types**

In `WatchlistRunAudioResponse`:

```python
queue_name: str | None = None
```

In `WatchlistRunAudioStatus`:

```ts
queue_name?: string | null
```

- [ ] **Step 5: Add a safe Scheduler lookup helper**

In `watchlists.py`, add a small private helper near the audio endpoint helpers:

```python
_SCHEDULER_AUDIO_STATUS_MAP = {
    "queued": "queued",
    "pending": "pending",
    "running": "running",
    "completed": "completed",
    "failed": "failed",
    "dead": "dead",
    "cancelled": "cancelled",
    "canceled": "cancelled",
}

async def _get_audio_scheduler_task_status(task_id: str) -> dict[str, Any] | None:
    try:
        from tldw_Server_API.app.core.Scheduler import get_global_scheduler
        scheduler = await get_global_scheduler(start_workers=False)
        task = await scheduler.get_task(task_id)
    except _WATCHLISTS_NONCRITICAL_EXCEPTIONS:
        return None
    if task is None:
        return None
    raw_status = getattr(getattr(task, "status", None), "value", None) or str(getattr(task, "status", "unknown"))
    raw_status = raw_status.rsplit(".", 1)[-1].lower()
    status_value = _SCHEDULER_AUDIO_STATUS_MAP.get(raw_status, "unknown")
    fallback_reason = "scheduler_task_error" if getattr(task, "error", None) else None
    return {
        "task_id": task_id,
        "status": status_value,
        "queue_name": getattr(task, "queue_name", None),
        "fallback_reason": fallback_reason,
    }
```

Use `start_workers=False` so a read-only status endpoint does not start a worker pool just because a user opened Reports.

- [ ] **Step 6: Replace `no_workflow_db` hard 404**

When `wf_db_path` is missing and a task id exists:

```python
scheduler_status = await _get_audio_scheduler_task_status(str(task_id))
if scheduler_status:
    return {
        "run_id": run_id,
        **scheduler_status,
        "audio_uri": None,
        "download_url": None,
    }
return {
    "run_id": run_id,
    "task_id": task_id,
    "status": "pending",
    "queue_name": "workflows",
    "audio_uri": None,
    "download_url": None,
    "fallback_reason": "workflow_run_not_started",
}
```

Keep `404 no_audio_briefing_for_run` when no task id exists.

- [ ] **Step 7: Run focused backend/frontend tests**

Run the commands from Step 3 again.

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py tldw_Server_API/app/api/v1/endpoints/watchlists.py tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py apps/packages/ui/src/types/watchlists.ts apps/packages/ui/src/services/__tests__/watchlists-audio.test.ts
git commit -m "fix: report queued watchlist audio tasks"
```

---

## Task 3: Reports Output Preview Live Audio Polling

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx`

- [ ] **Step 1: Write failing drawer polling tests**

Update the `@/services/watchlists` mock in `OutputPreviewDrawer.audio.test.tsx` to include `getWatchlistRunAudio`.

Add a test:

```tsx
serviceMocks.getWatchlistRunAudio.mockResolvedValue({
  run_id: 9,
  task_id: "task_audio_pending",
  queue_name: "workflows",
  status: "queued",
  audio_uri: null,
  download_url: null
})

render(<OutputPreviewDrawer open output={markdownOutputWithAudioRequested} onClose={vi.fn()} />)

await waitFor(() => expect(serviceMocks.getWatchlistRunAudio).toHaveBeenCalledWith(9))
expect(screen.getByText("Queued")).toBeInTheDocument()
expect(screen.getByText(/workflows/)).toBeInTheDocument()
```

Add a terminal-state test for completed response with final artifact, and a close/unmount test proving polling stops.

- [ ] **Step 2: Run frontend tests to verify failure**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL because the drawer only reads static output metadata.

- [ ] **Step 3: Add live audio state to drawer**

In `OutputPreviewDrawer.tsx`, import `getWatchlistRunAudio` and add:

```tsx
const [liveAudioStatus, setLiveAudioStatus] = useState<WatchlistRunAudioStatus | null>(null)
```

When `open && output?.run_id && audioSummary.requested`, fetch status immediately.

- [ ] **Step 4: Add bounded polling**

Poll only while status is pending, queued, or running:

```tsx
const shouldPollAudio =
  open &&
  output?.run_id != null &&
  audioSummary.requested &&
  ["pending", "queued", "running"].includes(liveAudioSummary.status)
```

Use `setTimeout` or `setInterval` cleanup in `useEffect`. Do not poll for output artifacts that have no requested audio.

- [ ] **Step 5: Merge live status over metadata**

In `outputMetadata.ts`, add a helper:

```ts
export const getMergedOutputAudioStatusSummary = (
  metadata: unknown,
  liveStatus: WatchlistRunAudioStatus | null | undefined,
  labels?: OutputMetadataLabels
): AudioStatusSummary => {
  const live = liveStatus ? getAudioStatusSummary(liveStatus, labels) : null
  if (live?.requested) return live
  return getOutputAudioStatusSummary(metadata, labels)
}
```

Also add `queueName?: string` to `AudioStatusSummary` and populate it from `queue_name` / `queueName`. Use the merged summary in the drawer. Keep binary downloads only for actual audio outputs.

- [ ] **Step 6: Render queue name when present**

Add a small diagnostic line in the audio artifact panel:

```tsx
{audioSummary.queueName && (
  <div className="text-xs text-text-muted">
    {t("watchlists:outputs.audioQueueName", "Queue: {{queue}}", { queue: audioSummary.queueName })}
  </div>
)}
```

- [ ] **Step 7: Run focused frontend tests**

Run the command from Step 2 again.

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx
git commit -m "fix: poll watchlist audio status in reports"
```

---

## Task 4: Prefer Active Watchlist Selection Over Imported Placeholder

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx`
- Create: `apps/packages/ui/src/components/Option/Watchlists/watchlist-selection.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/__tests__/watchlist-selection.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.experimental-ia.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.orientation-guidance.test.tsx`

- [ ] **Step 1: Write failing selection helper tests**

Create a colocated pure helper:

```ts
export const resolvePreferredWatchlistId = (
  items: WatchlistContainer[],
  selectedWatchlistId: number | null
): number | null => { ... }
```

Test cases:

```ts
expect(resolvePreferredWatchlistId(
  [
    importedInactive,
    activeNewsWatchlist
  ],
  null
)).toBe(activeNewsWatchlist.id)

expect(resolvePreferredWatchlistId(items, activeExisting.id)).toBe(activeExisting.id)
expect(resolvePreferredWatchlistId([], null)).toBeNull()
```

- [ ] **Step 2: Run focused selection tests to verify failure**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Watchlists/__tests__/watchlist-selection.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL once the new helper assertions are added, because current fallback uses `items[0]`.

- [ ] **Step 3: Implement deterministic selection scoring**

Use a pure helper:

```ts
const isImportedPlaceholder = (watchlist: WatchlistContainer): boolean =>
  watchlist.name.trim().toLowerCase() === "imported watchlist"

const watchlistScore = (watchlist: WatchlistContainer): number => {
  let score = 0
  if (watchlist.status === "active") score += 100
  if (!watchlist.deleted_at && !watchlist.archived_at) score += 40
  if (!isImportedPlaceholder(watchlist)) score += 20
  if (watchlist.domain === "news" || watchlist.domain === "cti_osint") score += 5
  return score
}
```

Break ties by `updated_at`, then `created_at`, then lower array index. Keep the current selected id if it is still valid.

- [ ] **Step 4: Replace `items[0]` fallback**

In `loadWatchlists()`:

```ts
const nextSelectedWatchlistId = resolvePreferredWatchlistId(items, selectedWatchlistId)
```

Do not change setup wizard completion, manual create, or query-param selection behavior.

- [ ] **Step 5: Run focused frontend tests**

Run the command from Step 2 again.

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.experimental-ia.test.tsx apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.orientation-guidance.test.tsx
git add apps/packages/ui/src/components/Option/Watchlists/watchlist-selection.ts apps/packages/ui/src/components/Option/Watchlists/__tests__/watchlist-selection.test.ts
git commit -m "fix: prefer active watchlist selection"
```

---

## Task 5: Focused Verification And Demo Smoke

**Files:**
- Modify: `Docs/Runbooks/watchlists_demo_readiness_2026_05_20.md` only if verification steps need clarification.
- Update: relevant Backlog task final summaries.

- [ ] **Step 1: Run backend audio/watchlists tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py \
  tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py \
  tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run frontend watchlists tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/services/__tests__/watchlists-audio.test.ts \
  src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx \
  src/components/Option/Watchlists/__tests__/watchlist-selection.test.ts \
  src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.experimental-ia.test.tsx \
  src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.orientation-guidance.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

Expected: PASS.

- [ ] **Step 3: Run Bandit on touched backend code**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py \
  tldw_Server_API/app/core/Watchlists/pipeline.py \
  tldw_Server_API/app/api/v1/endpoints/watchlists.py \
  tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py \
  -f json -o /tmp/bandit_watchlists_p0_demo_blockers.json
```

Expected: exit code 0 or only pre-existing/touched-test-safe findings documented.

- [ ] **Step 4: Run diff hygiene**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 5: Run browser-observed smoke**

Start services according to the repo runbook, then verify:

1. `/watchlists` selects an active non-imported watchlist when available.
2. Create or select a news watchlist with a reachable RSS/source.
3. Run a monitor and create a digest.
4. Enable audio with an explicitly verified working provider/model/voice.
5. Confirm `/api/v1/watchlists/runs/{run_id}/audio` returns `queued`, `running`, `failed`, or `completed`, not `no_workflow_db`.
6. Confirm Reports output preview polls and updates audio status.
7. Confirm final playable audio only if the configured provider completes.

Record screenshots or terminal artifacts under `/tmp` and summarize paths in the Backlog final summary. Do not commit runtime DBs, screenshots, or generated audio.

- [ ] **Step 6: Update Backlog and commit verification docs only if needed**

If the runbook needed changes:

```bash
git add Docs/Runbooks/watchlists_demo_readiness_2026_05_20.md
git commit -m "docs: clarify watchlists demo smoke gate"
```

Otherwise, only update the Backlog task final summaries before PR creation.

---

## Recommended PR Shape

Use one PR if all five tasks remain small and tests are clean. Split only if backend audio status grows beyond the planned touched files:

- PR A: Tasks 1-2, backend queue/status contract.
- PR B: Tasks 3-4, frontend Reports polling and selection.
- PR C: Task 5, verification/runbook only if needed.

Do not merge a PR that claims final playable audio unless the browser smoke verifies a provider-backed final artifact.

## Plan Review Note

The required plan-review subagent has not been dispatched in this session because the current agent policy requires explicit user authorization before spawning subagents. This plan should be reviewed before execution if the user authorizes that review step.
