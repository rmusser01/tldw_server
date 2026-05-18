# Watchlists Digest And Audio Briefing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the `/watchlists` digest/newsletter and optional 1-4 speaker audio briefing workflow from the hardened PRD without removing existing watchlists, OSINT, CTI, or power-user flows.

**Architecture:** Reuse the existing Watchlists WebUI, shared UI package, Watchlists API, Scheduler pipeline, output artifacts, notifications service, and audio briefing workflow. Start with contract alignment and observable states, then add guided source/monitor/digest/audio setup inside `/watchlists`, then add reuse, batch operations, and operator recovery.

**Tech Stack:** Next.js route wrapper, shared React/Ant Design UI in `apps/packages/ui`, TypeScript service/types, FastAPI/Pydantic Watchlists endpoints, SQLite-backed Watchlists DB, Scheduler/Workflow audio tasks, Vitest, Pytest, Playwright.

---

## Source Documents

- PRD: `Docs/superpowers/specs/2026-05-18-watchlists-digest-audio-briefing-prd-design.md`
- Backlog: `TASK-425`
- Key WebUI route: `apps/tldw-frontend/pages/watchlists.tsx`
- Shared route/component entry: `apps/packages/ui/src/routes/option-watchlists.tsx`
- Watchlists shell: `apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx`
- Watchlists API: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Watchlists schemas: `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
- Watchlists pipeline: `tldw_Server_API/app/core/Watchlists/pipeline.py`
- Audio workflow bridge: `tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py`

## File Structure And Ownership

### Frontend Contract And Services

- Modify: `apps/packages/ui/src/types/watchlists.ts`
  - Add typed source settings/scrape rules.
  - Add output audio fields missing from `WatchlistOutputCreate`.
  - Add `auto_output` job output preference typing.
  - Add run audio status/artifact typing.
  - Add structured `audio_cast` typing while preserving `voice_map`.
- Modify: `apps/packages/ui/src/services/watchlists.ts`
  - Add `getWatchlistRunAudio(runId)`.
  - Keep existing source/job/output functions stable.
  - Optionally add `validateWatchlistSourceDraftRules` only if the backend route is added.
- Test: `apps/packages/ui/src/services/__tests__/watchlists-overview.test.ts` or new `apps/packages/ui/src/services/__tests__/watchlists-audio.test.ts`.

### Source Setup

- Modify: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourceFormModal.tsx`
  - Preserve and submit `settings`.
  - Pass draft `settings` to source test.
  - Accept capability-driven forum enablement instead of hard-coded disabled forum UI.
  - Add typed website extraction/discovery controls behind advanced disclosure.
  - Show validation diagnostics returned by backend.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesTab.tsx`
  - Load `/watchlists/settings` capability data and pass `forums_enabled` into `SourceFormModal`.
- Create: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/source-settings.ts`
  - Pure helpers for parsing, normalizing, merging, and serializing source settings without deleting unknown keys.
- Test: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourceFormModal.test-source.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourceFormModal.forum-help.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/source-settings.test.ts`
- Backend modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
  - Return selector validation diagnostics from draft/saved source test, or add a specific validation endpoint.
- Backend test: `tldw_Server_API/tests/Watchlists/test_fetchers_scrape_rules.py`
- Backend test: `tldw_Server_API/tests/Watchlists/test_preview_endpoint.py`

### Monitor Cadence, Output, Delivery

- Modify: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/schedule-utils.ts`
  - Support every N minutes and every N hours as first-class presets.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/schedule-frequency.ts`
  - Preserve minimum interval validation.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/SchedulePicker.tsx`
  - Replace fixed hourly/every-6-hour-only UI with variable interval controls.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobFormModal.tsx`
  - Surface `auto_output.enabled`.
  - Explain scheduled output/delivery behavior before save.
  - Keep raw cron and existing advanced controls.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-contract.ts`
  - Ensure scheduled pipeline payloads set `output_prefs.auto_output.enabled`.
  - Keep manual/test run output creation explicit.
- Test: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/schedule-utils.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/SchedulePicker.help.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/JobFormModal.live-summary.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts`
- Backend test: `tldw_Server_API/tests/Watchlists/test_job_output_prefs_roundtrip.py`
- Backend test: `tldw_Server_API/tests/Watchlists/test_newsletter_briefing_gaps.py`

### Run, Output, Audio Artifacts

- Modify: `apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx`
  - Show output, delivery, audio pending, audio failed, fallback, and final artifact states.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunsTab.tsx`
  - Add compact audio/output status indicator.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx`
  - Show delivery result, script, per-speaker artifacts, fallback reason, and final player when metadata exists.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts`
  - Add pure metadata extraction helpers for delivery/audio status.
- Backend modify: `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
  - Add typed response schema for run audio status.
  - Add typed output metadata structures where practical.
- Backend modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
  - Return script/per-speaker/final audio artifacts, not just one final audio candidate.
- Backend modify: `tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py`
  - Accept structured speaker config.
  - Persist script, per-speaker artifacts, final mix, and fallback reason as workflow/watchlist output metadata.
- Test: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx`
- Backend test: `tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py`
- Backend test: `tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py`

### Guided Pipeline MVP

- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx`
  - Replace partial quick setup with a full additive guided entry point.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-contract.ts`
  - Extend draft model to include source settings preview, output/delivery expectations, and audio cast.
- Create: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/PipelineWizard.tsx`
  - Source, Monitor, Digest, Optional Audio, Review steps.
- Create: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-wizard-state.ts`
  - Pure state transitions and validation helpers.
- Test: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/PipelineWizard.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-wizard-state.test.ts`
- Regression test: `apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.experimental-ia.test.tsx`

### Power-User Throughput

- Modify: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobsTab.tsx`
  - Clone monitor.
  - Batch activation/schedule/output changes.
  - Batch retry entry points if backend supports them.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesTab.tsx`
  - Clone source rules.
  - Batch source rule test.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/shared/WatchlistsCommandPalette.tsx`
  - Commands for create pipeline, clone, validate, run, retry, export.
- Backend modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
  - Add safe batch endpoints only where existing single-operation APIs are insufficient.
- Test: existing source/jobs batch tests plus new focused clone/batch tests.

### Operator/Admin Reliability

- Modify: `apps/packages/ui/src/components/Option/Watchlists/shared/WatchlistsHealthBar.tsx`
  - Surface source/scheduler/delivery/audio health.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx`
  - Diagnostic bundle export and stage-specific retry controls.
- Backend modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
  - Add delivery-only retry and audio-only retry if existing workflow APIs are too generic.
  - Add diagnostic bundle endpoint if not already available through run logs/exports.
- Backend test: `tldw_Server_API/tests/Watchlists/test_delivery_integrations.py`
- Backend test: `tldw_Server_API/tests/Watchlists/test_runs_csv_export.py`
- Backend test: new `tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py`

---

## Stage Gates

- Stage 0 is complete when types/services can represent existing backend capability without changing the user journey.
- Stage 1A is complete when users can save source settings and variable cadence without raw JSON or cron.
- Stage 1B is complete when scheduled monitors can produce digest/newsletter output through `auto_output.enabled` and show delivery state.
- Stage 1C is complete when optional audio shows script, per-speaker artifacts, final mix, fallback reason, and final player inside `/watchlists`.
- Stage 2 is complete when expert users can clone/reuse/batch the core workflow.
- Stage 3 is complete when operators can diagnose and retry delivery/audio without rerunning ingestion.

## Recommended PR Slices

- PR 1: Tasks 1-3. Contract alignment, variable cadence, source settings preservation, and forum capability gating. This creates immediate user value without backend artifact changes.
- PR 2: Tasks 4-5. Source validation diagnostics plus scheduled output/delivery contract.
- PR 3: Tasks 6-7. Audio status display plus backend script/per-speaker/final artifact persistence.
- PR 4: Task 8. Guided pipeline MVP, built only after the contracts it promises are present.
- PR 5: Tasks 9-10. Power-user reuse/batch operations and operator recovery.
- PR 6: Task 11. End-to-end verification, browser QA, and release hardening.

---

## Task 1: Frontend Watchlists Contract Alignment

**Files:**
- Modify: `apps/packages/ui/src/types/watchlists.ts`
- Modify: `apps/packages/ui/src/services/watchlists.ts`
- Test: `apps/packages/ui/src/services/__tests__/watchlists-overview.test.ts`
- Test: new `apps/packages/ui/src/services/__tests__/watchlists-audio.test.ts`

- [x] **Step 1: Write failing type/service tests**

Add service tests proving `getWatchlistRunAudio(123)` calls `/api/v1/watchlists/runs/123/audio`, and compile-time fixtures proving `WatchlistOutputCreate` accepts backend-supported audio fields.

Representative fixture:

```ts
const output: WatchlistOutputCreate = {
  run_id: 10,
  generate_audio: true,
  target_audio_minutes: 12,
  audio_model: "kokoro",
  audio_voice: "af_heart",
  audio_speed: 1.05,
  background_audio_uri: "file:///tmp/bed.mp3",
  background_volume: 0.15,
  audio_language: "en",
  llm_provider: "openai",
  llm_model: "gpt-4.1-mini",
  persona_summarize: true,
  voice_map: { HOST: "af_bella", ANALYST: "am_adam" }
}
```

- [x] **Step 2: Run tests to verify failure**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/services/__tests__/watchlists-audio.test.ts src/components/Option/Watchlists/__tests__/watchlists-static-guard.typecheck.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL because the service helper and several output audio fields are missing or incomplete.

- [x] **Step 3: Add types and service helper**

Add these types to `watchlists.ts`:

```ts
export interface WatchlistAudioCastSpeaker {
  id: string
  label: string
  role?: string
  voice: string
  persona?: string
}

export interface WatchlistAudioCast {
  speaker_count: 1 | 2 | 3 | 4
  speakers: WatchlistAudioCastSpeaker[]
}

export interface WatchlistRunAudioStatus {
  run_id: number
  task_id?: string | null
  status: "pending" | "running" | "completed" | "failed" | "unknown" | string
  audio_uri?: string | null
  download_url?: string | null
  artifact_id?: string | number | null
  size_bytes?: number | null
  mime_type?: string | null
  script_artifact?: Record<string, unknown> | null
  speaker_artifacts?: Array<Record<string, unknown>>
  final_artifact?: Record<string, unknown> | null
  fallback_reason?: string | null
  error?: string | null
}
```

Extend `JobOutputPrefs` and `WatchlistOutputCreate` with backend-supported audio fields and `auto_output`.

Add this helper in `watchlists.ts` service:

```ts
export const getWatchlistRunAudio = async (
  runId: number
): Promise<WatchlistRunAudioStatus> => {
  return bgRequest<WatchlistRunAudioStatus>({
    path: `/api/v1/watchlists/runs/${runId}/audio` as any,
    method: "GET"
  })
}
```

- [x] **Step 4: Run focused tests**

Run the command from Step 2 again.

Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add apps/packages/ui/src/types/watchlists.ts apps/packages/ui/src/services/watchlists.ts apps/packages/ui/src/services/__tests__/watchlists-audio.test.ts
git commit -m "feat: align watchlists audio contracts"
```

---

## Task 2: Variable Cadence Controls

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/schedule-utils.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/schedule-frequency.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/SchedulePicker.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/schedule-utils.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/SchedulePicker.help.test.tsx`

- [x] **Step 1: Write failing schedule tests**

Add coverage for:

- Every 5 hours -> `0 */5 * * *`
- Every 15 minutes -> `*/15 * * * *`
- Every 4 minutes blocked by existing minimum interval
- Existing `0 */6 * * *` parses back to every 6 hours
- Weekly still works
- Raw cron still available

- [x] **Step 2: Run tests to verify failure**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Watchlists/JobsTab/__tests__/schedule-utils.test.ts src/components/Option/Watchlists/JobsTab/__tests__/SchedulePicker.help.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL because presets are fixed to hourly/every6/daily/weekly.

- [x] **Step 3: Implement interval preset state**

Update `SchedulePresetKey` and state:

```ts
export type SchedulePresetKey = "interval" | "daily" | "weekdays" | "weekly"
export type ScheduleIntervalUnit = "minutes" | "hours"

export interface PresetScheduleState {
  preset: SchedulePresetKey
  intervalValue: number
  intervalUnit: ScheduleIntervalUnit
  hour: number
  minute: number
  weekday: WeekdayToken
}
```

Build cron as:

```ts
if (state.preset === "interval" && state.intervalUnit === "minutes") {
  return `*/${clampInteger(state.intervalValue, 5, 59)} * * * *`
}
if (state.preset === "interval" && state.intervalUnit === "hours") {
  return `${minute} */${clampInteger(state.intervalValue, 1, 23)} * * *`
}
```

Keep custom cron as the escape hatch for schedules cron can express but the preset UI cannot safely model.

Update `parsePresetFromCron` so existing `*/6` schedules map into the interval model instead of appearing as raw cron after upgrade.

- [x] **Step 4: Update UI copy and controls**

In `SchedulePicker.tsx`, use segmented/select controls for:

- Manual/no schedule via existing clear/null behavior.
- Every N minutes.
- Every N hours.
- Daily.
- Weekdays.
- Weekly.
- Advanced cron.

Always show generated cron and a human-readable preview via `CronDisplay`.

- [x] **Step 5: Run focused tests**

Run the command from Step 2 again.

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/Watchlists/JobsTab/schedule-utils.ts apps/packages/ui/src/components/Option/Watchlists/JobsTab/schedule-frequency.ts apps/packages/ui/src/components/Option/Watchlists/JobsTab/SchedulePicker.tsx apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/schedule-utils.test.ts apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/SchedulePicker.help.test.tsx
git commit -m "feat: add variable watchlist cadence controls"
```

---

## Task 3: Source Settings Preservation And Rule Preview

**Files:**
- Create: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/source-settings.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourceFormModal.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesTab.tsx`
- Test: new `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/source-settings.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourceFormModal.test-source.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourceFormModal.forum-help.test.tsx`

- [x] **Step 1: Write failing source settings tests**

Cover:

- Editing a source preserves unknown `settings` keys.
- Website scrape rules are serialized under `settings.scrape_rules`.
- Draft source test sends `settings`.
- Empty advanced fields do not create noisy settings.
- Forum source option is enabled only when watchlists settings report `forums_enabled`.

- [x] **Step 2: Run tests to verify failure**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Watchlists/SourcesTab/__tests__/source-settings.test.ts src/components/Option/Watchlists/SourcesTab/__tests__/SourceFormModal.test-source.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL because settings are not exposed or submitted.

- [x] **Step 3: Implement pure merge helpers**

Add helpers:

```ts
export const mergeSourceSettings = (
  existing: Record<string, unknown> | null | undefined,
  patch: Record<string, unknown>
): Record<string, unknown> => ({
  ...(existing || {}),
  ...patch
})
```

Use more specific functions for scrape rules, so fields can be deleted intentionally without wiping unrelated advanced settings.

- [x] **Step 4: Update `SourceFormModal`**

Change `onSubmit` values to include `settings`.

Load `initialValues.settings` into form state.

Add a `forumsEnabled` prop to `SourceFormModal`, defaulting to `false`, and pass it from `SourcesTab` using the existing settings endpoint/service. Keep the disabled explanatory copy when the capability is false.

For saved and draft tests, pass settings:

```ts
await testWatchlistSourceDraft(
  {
    url: draftUrl,
    source_type: draftType as SourceType,
    settings: buildSourceSettingsPayload(initialValues?.settings, values)
  },
  { limit: 10 }
)
```

- [x] **Step 5: Add validation display placeholders**

Show a compact diagnostics block when preview response includes:

- fetch mode
- selector errors
- no-match warnings
- non-unique warnings
- fragile selector warnings
- dedupe preview key

Do not invent those fields in the UI if the backend does not return them yet; render only when present.

- [x] **Step 6: Run focused tests**

Run the command from Step 2 again.

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add apps/packages/ui/src/components/Option/Watchlists/SourcesTab/source-settings.ts apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourceFormModal.tsx apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesTab.tsx apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/source-settings.test.ts apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourceFormModal.test-source.test.tsx apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourceFormModal.forum-help.test.tsx
git commit -m "feat: preserve watchlist source settings"
```

---

## Task 4: Source Validation Diagnostics API

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Test: `tldw_Server_API/tests/Watchlists/test_fetchers_scrape_rules.py`
- Test: `tldw_Server_API/tests/Watchlists/test_preview_endpoint.py`

- [x] **Step 1: Write failing backend tests**

Add tests proving draft/saved source test returns diagnostics for site sources with scrape rules:

```python
assert response.json()["diagnostics"]["fetch_mode"] == "scrape_rules"
assert "selector_warnings" in response.json()["diagnostics"]
```

- [x] **Step 2: Run tests to verify failure**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists/test_fetchers_scrape_rules.py tldw_Server_API/tests/Watchlists/test_preview_endpoint.py -q
```

Expected: FAIL because preview result currently exposes only item counts/items to the frontend contract.

- [x] **Step 3: Add response schema**

Extend preview response schema with optional diagnostics:

```python
class SourcePreviewDiagnostics(BaseModel):
    fetch_mode: str | None = None
    selector_errors: list[str] = Field(default_factory=list)
    selector_warnings: list[str] = Field(default_factory=list)
    dedupe_preview_key: str | None = None
```

Keep fields optional to avoid breaking existing callers.

- [x] **Step 4: Populate diagnostics**

Reuse existing `validate_selector_rules` output for site/forum scrape rules. For RSS and discovery fallback, report `fetch_mode` only if no selector diagnostics exist.

- [x] **Step 5: Run backend tests**

Run the command from Step 2 again.

Expected: PASS.

- [x] **Step 6: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py tldw_Server_API/app/api/v1/endpoints/watchlists.py tldw_Server_API/tests/Watchlists/test_fetchers_scrape_rules.py tldw_Server_API/tests/Watchlists/test_preview_endpoint.py
git commit -m "feat: expose watchlist source validation diagnostics"
```

---

## Task 5: Auto-Output And Delivery Contract

**Files:**
- Modify: `apps/packages/ui/src/types/watchlists.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobFormModal.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-contract.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/JobFormModal.live-summary.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts`
- Backend test: `tldw_Server_API/tests/Watchlists/test_job_output_prefs_roundtrip.py`
- Backend test: `tldw_Server_API/tests/Watchlists/test_newsletter_briefing_gaps.py`

- [x] **Step 1: Write failing frontend tests**

Cover:

- Scheduled monitor payload includes `output_prefs.auto_output.enabled: true`.
- Manual/test output creation remains explicit.
- Delivery status helper distinguishes `sent`, `skipped`, `failed`, and `pending`.

- [x] **Step 2: Run frontend tests to verify failure**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts src/components/Option/Watchlists/JobsTab/__tests__/JobFormModal.live-summary.test.tsx src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts --maxWorkers=1 --no-file-parallelism
```

- [x] **Step 3: Add `auto_output` type and UI**

Add to `JobOutputPrefs`:

```ts
auto_output?: {
  enabled?: boolean
  type?: string
  format?: "md" | "html"
  template_name?: string
  template_version?: number
}
```

In `JobFormModal`, make scheduled output explicit: when delivery or audio is enabled for recurring monitors, require or default `auto_output.enabled` and show review copy that output artifacts will be generated each run.

- [x] **Step 4: Update pipeline payload helpers**

In `pipeline-contract.ts`, set `auto_output.enabled` when the user chooses scheduled digest/newsletter output. Do not set it for manual/test-only flows.

- [x] **Step 5: Confirm backend roundtrip**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists/test_job_output_prefs_roundtrip.py tldw_Server_API/tests/Watchlists/test_newsletter_briefing_gaps.py -q
```

Expected: PASS or only frontend-driven behavior needs update. If backend drops the field, fix schema/persistence before proceeding.

- [x] **Step 6: Run focused frontend tests**

Run the command from Step 2 again.

Expected: PASS.

- [x] **Step 7: Commit**

```bash
git add apps/packages/ui/src/types/watchlists.ts apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobFormModal.tsx apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-contract.ts apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/JobFormModal.live-summary.test.tsx apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts
git commit -m "feat: make watchlist scheduled outputs explicit"
```

---

## Task 6: Run Audio Status In Activity And Reports

**Files:**
- Modify: `apps/packages/ui/src/services/watchlists.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunsTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx`

- [ ] **Step 1: Write failing UI tests**

Cover:

- Run drawer polls or loads audio status when `stats.audio_briefing_task_id` exists.
- Pending status is visible.
- Final audio renders a player/download link.
- Output preview shows fallback reason when metadata has fallback.

- [ ] **Step 2: Run tests to verify failure**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx --maxWorkers=1 --no-file-parallelism
```

- [ ] **Step 3: Implement status helpers**

Add pure helpers in `outputMetadata.ts` for:

- delivery status summary
- audio requested/pending/final/failed/fallback summary
- script/per-speaker/final artifact extraction

- [ ] **Step 4: Render states**

In run/output surfaces, distinguish:

- audio not requested
- audio queued/pending
- audio running
- final audio available
- failed audio
- fallback single-voice audio
- status unknown

- [ ] **Step 5: Run focused tests**

Run the command from Step 2 again.

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/services/watchlists.ts apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunsTab.tsx apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx
git commit -m "feat: surface watchlist audio run status"
```

---

## Task 7: Backend Audio Artifact Persistence

**Files:**
- Modify: `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Modify: `tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py`
- Test: `tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py`
- Test: `tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py`

- [ ] **Step 1: Write failing backend tests**

Add tests for `GET /runs/{run_id}/audio` returning:

- script artifact
- per-speaker artifacts
- final artifact
- fallback reason when multi-voice fails

- [ ] **Step 2: Run tests to verify failure**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py -q
```

Expected: FAIL because the endpoint currently chooses a final audio candidate and does not expose the full artifact graph.

- [ ] **Step 3: Add structured audio cast schema**

Add optional `audio_cast` to output create/job prefs schemas. Preserve `voice_map` compatibility.

```python
class WatchlistAudioCastSpeaker(BaseModel):
    id: str
    label: str
    role: str | None = None
    voice: str
    persona: str | None = None

class WatchlistAudioCast(BaseModel):
    speaker_count: int = Field(..., ge=1, le=4)
    speakers: list[WatchlistAudioCastSpeaker]
```

- [ ] **Step 4: Persist intermediate artifacts**

Extend workflow metadata/artifact naming so script, per-speaker clips, final mix, and fallback reason can be retrieved by run ID. Do not create a new podcast job system.

- [ ] **Step 5: Expand run audio endpoint**

Return a stable shape:

```python
{
    "run_id": run_id,
    "task_id": task_id,
    "status": matching_run_status,
    "script_artifact": script_artifact,
    "speaker_artifacts": speaker_artifacts,
    "final_artifact": final_artifact,
    "fallback_reason": fallback_reason,
    "download_url": final_download_url,
}
```

- [ ] **Step 6: Run backend tests**

Run the command from Step 2 again.

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py tldw_Server_API/app/api/v1/endpoints/watchlists.py tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py
git commit -m "feat: persist watchlist audio briefing artifacts"
```

---

## Task 8: Guided Pipeline MVP

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-contract.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/quick-setup.ts`
- Create: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/PipelineWizard.tsx`
- Create: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-wizard-state.ts`
- Test: new `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/PipelineWizard.test.tsx`
- Test: new `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-wizard-state.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts`
- Regression: `apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.experimental-ia.test.tsx`

- [ ] **Step 1: Write failing wizard state tests**

Cover:

- Requires at least one source.
- Requires monitor name.
- Supports manual/every-N-hours/daily/weekly schedule.
- Requires template for digest/newsletter.
- Optional audio supports 0-4 speakers.
- Review summary names source, cadence, filters, output, delivery, and audio.

- [ ] **Step 2: Run tests to verify failure**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-wizard-state.test.ts src/components/Option/Watchlists/OverviewTab/__tests__/PipelineWizard.test.tsx src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts --maxWorkers=1 --no-file-parallelism
```

- [ ] **Step 3: Implement wizard state helpers**

Keep state pure. Use existing service calls from the component, not from helper files.

- [ ] **Step 4: Implement additive `PipelineWizard`**

Wizard steps:

1. Sources.
2. Monitor.
3. Digest.
4. Optional Audio.
5. Review.

The wizard should call existing `createWatchlistSource`, `createWatchlistJob`, `triggerWatchlistRun`, and `createWatchlistOutput` helpers. It must not bypass existing API contracts.

- [ ] **Step 5: Wire into overview without removing full controls**

Add "Create briefing pipeline" as an entry point. Preserve current Sources/Items/Outputs primary tabs and "Show all views".

- [ ] **Step 6: Run focused tests**

Run the command from Step 2 again.

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-contract.ts apps/packages/ui/src/components/Option/Watchlists/OverviewTab/quick-setup.ts apps/packages/ui/src/components/Option/Watchlists/OverviewTab/PipelineWizard.tsx apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-wizard-state.ts apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/PipelineWizard.test.tsx apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-wizard-state.test.ts apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.experimental-ia.test.tsx
git commit -m "feat: add watchlists briefing pipeline wizard"
```

---

## Task 9: Power-User Reuse And Batch Operations

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobsTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/shared/WatchlistsCommandPalette.tsx`
- Modify as needed: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Test: existing Jobs/Sources batch tests plus new focused tests.

- [ ] **Step 1: Write failing clone/reuse tests**

Cover:

- Clone monitor preserves scope, schedule, filters, output prefs, delivery, audio cast.
- Clone source rules preserves settings but resets status/seen state.
- Command palette exposes create, clone, run, preview, retry, export.

- [ ] **Step 2: Implement frontend-only clone where possible**

Prefer composing existing create APIs with copied payloads. Add backend batch APIs only where frontend composition is unsafe or inefficient.

- [ ] **Step 3: Add batch test/validation actions**

Source batch rule test can reuse `checkWatchlistSourcesNow` initially, then move to a richer validation endpoint when Task 4 exists.

- [ ] **Step 4: Run watchlists scale and batch tests**

```bash
cd apps/packages/ui
bun run test:watchlists:scale
```

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobsTab.tsx apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesTab.tsx apps/packages/ui/src/components/Option/Watchlists/shared/WatchlistsCommandPalette.tsx
git commit -m "feat: add watchlists reuse and batch controls"
```

---

## Task 10: Operator Recovery And Diagnostics

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Watchlists/shared/WatchlistsHealthBar.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunsTab.tsx`
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Test: new `tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py`
- Test: `tldw_Server_API/tests/Watchlists/test_delivery_integrations.py`
- Test: `tldw_Server_API/tests/Watchlists/test_runs_csv_export.py`

- [ ] **Step 1: Write failing operator tests**

Cover:

- Delivery-only retry does not rerun ingestion.
- Audio-only retry does not rerun ingestion.
- Diagnostic bundle contains source statuses, filter tallies, output metadata, delivery statuses, audio task metadata, and logs.

- [ ] **Step 2: Run backend tests to verify failure**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py tldw_Server_API/tests/Watchlists/test_delivery_integrations.py tldw_Server_API/tests/Watchlists/test_runs_csv_export.py -q
```

- [ ] **Step 3: Add backend retry and diagnostics endpoints**

Only add endpoints that cannot safely be represented with existing APIs. Keep them stage-specific:

- output delivery retry
- audio retry
- diagnostic bundle export

- [ ] **Step 4: Render operator controls**

Controls must show confirmation copy that names what will and will not rerun.

- [ ] **Step 5: Run backend and frontend focused tests**

Run backend command from Step 2 and focused `RunsTab` Vitest tests.

- [ ] **Step 6: Commit**

```bash
git add tldw_Server_API/app/api/v1/endpoints/watchlists.py tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py tldw_Server_API/tests/Watchlists/test_delivery_integrations.py tldw_Server_API/tests/Watchlists/test_runs_csv_export.py apps/packages/ui/src/components/Option/Watchlists/shared/WatchlistsHealthBar.tsx apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunsTab.tsx
git commit -m "feat: add watchlists operator recovery controls"
```

---

## Task 11: End-To-End Verification And Browser QA

**Files:**
- Create or modify: `apps/tldw-frontend/e2e/...` only if an existing watchlists workflow spec is present and suitable.
- Otherwise rely on component/API tests plus browser-observed manual QA.

- [ ] **Step 1: Run focused frontend verification**

```bash
cd apps/packages/ui
bun run test:watchlists:a11y
bun run test:watchlists:scale
bun run test:watchlists:typecheck
```

- [ ] **Step 2: Run focused backend verification**

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists/test_watchlists_api.py tldw_Server_API/tests/Watchlists/test_full_pipeline_integration.py tldw_Server_API/tests/Watchlists/test_job_output_prefs_roundtrip.py tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py -q
```

- [ ] **Step 3: Run Bandit on touched backend code**

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/api/v1/endpoints/watchlists.py tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py tldw_Server_API/app/core/Watchlists -f json -o /tmp/bandit_watchlists_digest_audio.json
```

- [ ] **Step 4: Browser QA `/watchlists`**

Start local services per repo conventions, then verify:

- First-time empty/near-empty state offers guided pipeline and full controls.
- Source test displays RSS/site mode, preview counts, and diagnostics.
- Every 5 hours can be selected without raw cron.
- Weekly can be selected.
- Scheduled digest shows `auto_output` expectation before save.
- Email delivery setup distinguishes configured/unavailable/skipped states.
- Optional audio supports no audio and 1-4 speakers.
- Run detail shows pending/final/failed audio state.
- Output preview shows digest, delivery status, final audio, and fallback when applicable.
- Full tab workflow and "Show all views" still work.

- [ ] **Step 5: Final regression sweep**

```bash
git diff --check
```

If frontend and backend tests are too expensive for the final patch, document exact skipped commands and why in the Backlog task and PR body.

---

## Deferred Or Explicitly Non-MVP Work

- True every-N-weeks recurrence beyond cron-supported weekly patterns. Use custom cron or add a backend recurrence abstraction later.
- Full podcast studio outside `/watchlists`.
- Removing raw cron, raw `voice_map`, OPML, templates, item review, or advanced tabs.
- Forum source default enablement before backend settings report `forums_enabled`.
- Large IA redesign outside `/watchlists`.

## Plan Review Notes

- This plan intentionally starts with contract work because the current UI cannot honestly promise scheduled digest delivery or multi-speaker artifact recovery until the frontend/backend contracts can represent those states.
- The first implementation PR should stop after Tasks 1-3 or Tasks 1-5. Shipping all tasks in one PR would create review risk and make regression source hard to isolate.
- A plan-document-reviewer subagent was not dispatched in this session because the user did not explicitly authorize subagents. Before large-scale execution, run an independent review against this plan and the PRD if subagent delegation is authorized.
