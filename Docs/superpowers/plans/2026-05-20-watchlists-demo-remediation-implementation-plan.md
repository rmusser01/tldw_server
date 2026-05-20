# Watchlists Demo Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the verified `/watchlists` demo blockers first, then complete the durable first-time, audio, operator, and power-user workflow improvements without removing existing news, OSINT, CTI, or advanced Watchlists flows.

**Architecture:** Reuse the existing shared Watchlists UI, Watchlists API, Scheduler, pipeline, templates, output artifacts, and extension route. The first PR slice makes current behavior truthful and demonstrable; later slices expand the same contracts instead of adding a parallel podcast or digest product outside `/watchlists`.

**Tech Stack:** React/TypeScript shared UI in `apps/packages/ui`, Next.js WebUI route shim, WXT/browser-extension shared route, FastAPI/Pydantic Watchlists endpoints, SQLite-backed Watchlists DB, Scheduler `submit(...)`, Watchlists pipeline/audio workflow, Vitest, Playwright, Pytest, Bandit.

---

## Source Documents

- Demo remediation spec: `Docs/superpowers/specs/2026-05-20-watchlists-demo-remediation-staged-plans-design.md`
- Existing digest/audio PRD: `Docs/superpowers/specs/2026-05-18-watchlists-digest-audio-briefing-prd-design.md`
- Existing digest/audio implementation plan: `Docs/superpowers/plans/2026-05-18-watchlists-digest-audio-briefing-implementation-plan.md`
- Backlog: `TASK-441`

## Verified Current-State Evidence

- WebUI route: `apps/tldw-frontend/pages/watchlists.tsx` dynamically imports `@/routes/option-watchlists`.
- Extension route: `apps/tldw-frontend/extension/routes/route-registry.tsx` registers `/watchlists` to the same route component.
- Shared route: `apps/packages/ui/src/routes/option-watchlists.tsx` renders `WatchlistsPlaygroundPage`.
- Backend builtin template: `tldw_Server_API/app/core/Watchlists/template_store.py` registers `briefing_markdown`, not `briefing_md`.
- Quick setup sends the wrong template: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/quick-setup.ts` currently sends `template_name: "briefing_md"`.
- Pipeline payloads send the wrong template: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-contract.ts` forwards `draft.templateName` as `template_name` and `template.default_name`.
- Jobs already contains the correct mapping pattern: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobFormModal.tsx` maps recipe `briefing_md` to backend template `briefing_markdown`.
- Audio enqueue bug: `tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py` creates a Scheduler `Task` and calls `scheduler.enqueue(task)`.
- Scheduler public API: `tldw_Server_API/app/core/Scheduler/scheduler.py` exposes `submit(handler, payload, priority, queue_name, depends_on, idempotency_key, metadata, auth_context)`.
- Run audio endpoint exists: `tldw_Server_API/app/api/v1/endpoints/watchlists.py` exposes `GET /api/v1/watchlists/runs/{run_id}/audio` and currently returns `no_audio_briefing_for_run` when run stats lack an audio task id.
- Output creation already records `audio_briefing_status` as `pending`, `skipped`, or `enqueue_failed` in output metadata when `generate_audio` is requested.

## File Structure And Ownership

### Demo Rescue Contract

- Create: `Docs/Runbooks/watchlists_demo_readiness_2026_05_20.md`
  - Single owner-readable demo contract: environment, demo sources, commands, expected states, fallbacks, and hard stops.
- Modify: `apps/tldw-frontend/e2e/workflows/watchlists-demo-readiness.spec.ts`
  - WebUI same-origin demo smoke for source create, monitor create, run, output creation, and truthful audio status.
- Modify: `apps/extension/tests/e2e/watchlists.spec.ts`
  - Add one strict mount/workflow smoke or extend the existing strict suite with output error/status assertions.

### Frontend Template Contract

- Create: `apps/packages/ui/src/components/Option/Watchlists/shared/templateNames.ts`
  - Pure mapping from UI recipe ids to backend template names.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/quick-setup.ts`
  - Send backend template names and nested `template.default_name`.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-contract.ts`
  - Normalize template names before job/output payloads.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx`
  - Keep UI labels stable, but render recoverable output-generation errors.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/shared/scale-benchmark.ts`
  - Stop scale fixtures from sending backend-invalid `briefing_md`.
- Test: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/quick-setup.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/OverviewTab.quick-setup.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/TemplatesTab/__tests__/template-recipes.test.ts`

### Backend Audio Enqueue

- Modify: `tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py`
  - Replace `scheduler.enqueue(Task(...))` with `scheduler.submit("workflow_run", payload=..., queue_name="workflows", idempotency_key=..., metadata=...)`.
- Test: `tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py`
  - Mock `submit`, not `enqueue`.
- Test: `tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py`
  - Keep pending/running/completed/failed retrieval coverage green.
- Test: `tldw_Server_API/tests/Watchlists/test_watchlists_api.py`
  - Keep explicit output `generate_audio=true` metadata coverage green.

### Frontend Audio/Status Truthfulness

- Modify: `apps/packages/ui/src/types/watchlists.ts`
  - Add typed run audio status, audio output metadata fields, and explicit `WatchlistOutputCreate` audio request fields.
- Modify: `apps/packages/ui/src/services/watchlists.ts`
  - Add `getWatchlistRunAudio(runId)`.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-contract.ts`
  - Ensure manual/test output creation sends `generate_audio=true` and audio settings when the user requested audio.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts`
  - Normalize audio requested/pending/skipped/enqueue_failed/completed/failed states.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx`
  - Show output/audio status as partial, failed, pending, or complete.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx`
  - Show audio pending/failed/skipped details without implying a playable artifact exists.
- Test: `apps/packages/ui/src/services/__tests__/watchlists-audio.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx`

### Digest And Newsletter Output Contract

- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-contract.ts`
  - Set scheduled `output_prefs.auto_output.enabled` only when the monitor is supposed to create scheduled reports.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobFormModal.tsx`
  - Explain scheduled output/delivery behavior before save while preserving advanced controls.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/job-summaries.ts`
  - Show template, report type, delivery, and audio linkage consistently.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputsTab.tsx`
  - Keep generated digest/newsletter artifacts discoverable from Reports.
- Test: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/JobFormModal.live-summary.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/JobsTab.scope-filter-summary.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.regenerate-modal.test.tsx`
- Backend test: `tldw_Server_API/tests/Watchlists/test_job_output_prefs_roundtrip.py`
- Backend test: `tldw_Server_API/tests/Watchlists/test_newsletter_briefing_gaps.py`

### Source/Run Health Truthfulness

- Modify: `tldw_Server_API/app/core/Watchlists/pipeline.py`
  - Persist source fetch/extraction failures into run stats without necessarily making a mixed-source run a hard failure.
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
  - Return run/source warning fields through details/list endpoints if the pipeline already persists them.
- Modify: `apps/packages/ui/src/services/watchlists-overview.ts`
  - Aggregate source/run/output/audio warnings into Overview health.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/shared/WatchlistsHealthBar.tsx`
  - Block `System healthy` while unresolved active source, recent run, output, delivery, or audio hard failures exist.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx`
  - Ensure health copy matches warning/partial state.
- Test: `tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py`
- Test: `apps/packages/ui/src/services/__tests__/watchlists-overview.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.health.test.tsx`

### First-Time Workflow Completion

- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/quick-setup.ts`
  - Support every N minutes, every N hours, daily, weekdays, weekly, and manual presets.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/PipelineWizard.tsx`
  - Show variable cadence controls and accurate review summary.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-contract.ts`
  - Convert variable cadence to the existing schedule expression contract.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/SchedulePicker.tsx`
  - Reuse existing advanced schedule controls where possible.
- Test: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/PipelineWizard.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/schedule-utils.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/SchedulePicker.help.test.tsx`

### Source Confidence And Dedupe Explanation

- Create: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/source-settings.ts`
  - Preserve unknown source `settings` keys while exposing typed controls.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourceFormModal.tsx`
  - Show test result, sample items, extraction diagnostics, and dedupe identity preview.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesTab.tsx`
  - Surface source error badges and links to seen/dedupe tools.
- Modify: `tldw_Server_API/app/core/Watchlists/fetchers.py`
  - Return selector validation diagnostics already available from `validate_selector_rules`.
- Test: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/source-settings.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourceFormModal.test-source.test.tsx`
- Test: `tldw_Server_API/tests/Watchlists/test_fetchers_scrape_rules.py`

### Durable Audio Artifacts And Recovery

- Modify: `tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py`
  - Persist script, per-speaker script/audio, final mix, fallback reason, provider/voice provenance, and stage errors.
- Modify: `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
  - Add typed response schemas for run audio status and audio artifacts.
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
  - Return audio artifact stages and add safe retry endpoint only for audio generation.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx`
  - Render script, per-speaker status, final player/download, and retry controls.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx`
  - Link audio task/output stages to the same artifact record.
- Test: `tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py`
- Test: `tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx`

### Power-User And Operator Hardening

- Modify: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobsTab.tsx`
  - Clone monitor, batch activation/schedule/output changes, and batch retry entry points where backend support exists.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesTab.tsx`
  - Clone source rules and batch source validation.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/shared/WatchlistsCommandPalette.tsx`
  - Add commands for create pipeline, clone, validate, run, retry, export, and reports.
- Modify: `apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx`
  - Stage-level fetch/extraction/dedupe/filter/output/delivery/audio diagnostics.
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
  - Add delivery-only retry and diagnostic bundle endpoints only if existing endpoints cannot express the operation safely.
- Test: `apps/extension/tests/e2e/watchlists.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/watchlists-items.spec.ts`
- Test: `tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py`

## Recommended PR Slices

1. **PR A, Demo Rescue:** Tasks 1-5. Fix template payloads, Scheduler audio submit, truthful minimal audio/source health, demo runbook, and live WebUI/extension verification.
2. **PR B, First-Time Workflow:** Tasks 6-8. Variable cadence, review summary, digest/newsletter contract, source validation/dedupe explanation.
3. **PR C, Durable Audio Product:** Task 9. Persist and display script, per-speaker artifacts, final mix, fallback, and retry.
4. **PR D, Operator Recovery:** Task 10 focused on health model, stage-level diagnostics, retry controls, and diagnostic bundles.
5. **PR E, Power-User Throughput:** Task 11 focused on clone, presets, batch validation, batch operations, command palette, and preservation gates.
6. **PR F, Final Verification:** Task 12. Full browser, extension, backend, frontend, Bandit, and demo-readiness closeout.

## Task 1: Template Contract Hotfix

**Files:**
- Create: `apps/packages/ui/src/components/Option/Watchlists/shared/templateNames.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/quick-setup.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-contract.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/shared/scale-benchmark.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/quick-setup.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/OverviewTab.quick-setup.test.tsx`

- [ ] **Step 1: Write failing tests for quick setup**

Update `quick-setup.test.ts` so briefing mode expects backend template names:

```ts
expect(payload.output_prefs).toMatchObject({
  template_name: "briefing_markdown",
  template: { default_name: "briefing_markdown" },
  generate_audio: true
})
```

- [ ] **Step 2: Write failing tests for pipeline payloads**

Update `pipeline-contract.test.ts` so a draft with `templateName: "briefing_md"` produces:

```ts
expect(jobPayload.output_prefs).toMatchObject({
  template_name: "briefing_markdown",
  template: { default_name: "briefing_markdown" }
})
expect(outputPayload.template_name).toBe("briefing_markdown")
```

- [ ] **Step 3: Run the focused frontend tests and confirm failure**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Watchlists/OverviewTab/__tests__/quick-setup.test.ts \
  src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL with `briefing_md` where tests expect `briefing_markdown`.

- [ ] **Step 4: Add the shared template mapping helper**

Create `shared/templateNames.ts`:

```ts
export type WatchlistTemplateRecipeId = "briefing_md" | "newsletter_html" | "mece_md"

const BACKEND_TEMPLATE_BY_RECIPE: Record<WatchlistTemplateRecipeId, string> = {
  briefing_md: "briefing_markdown",
  newsletter_html: "newsletter_html",
  mece_md: "mece_markdown"
}

export const normalizeWatchlistTemplateName = (value?: string | null): string => {
  const trimmed = String(value || "").trim()
  if (!trimmed) return ""
  return BACKEND_TEMPLATE_BY_RECIPE[trimmed as WatchlistTemplateRecipeId] || trimmed
}
```

- [ ] **Step 5: Use the helper in quick setup**

In `quick-setup.ts`, convert the briefing output prefs to:

```ts
const templateName = normalizeWatchlistTemplateName("briefing_md")
payload.output_prefs = {
  template_name: templateName,
  template: { default_name: templateName },
  generate_audio: Boolean(values.includeAudioBriefing)
}
```

- [ ] **Step 6: Use the helper in pipeline payloads**

In `pipeline-contract.ts`, derive one normalized template name:

```ts
const templateName = normalizeWatchlistTemplateName(draft.templateName)
```

Use `templateName` for `output_prefs.template_name`, `output_prefs.template.default_name`, and `WatchlistOutputCreate.template_name`.

- [ ] **Step 7: Keep UI recipe labels unchanged**

In `OverviewTab.tsx`, keep display strings such as `briefing_md` only where they identify the UI recipe. Do not show `briefing_markdown` as a new user-facing product concept unless the surrounding UI is explicitly template-admin UI.

- [ ] **Step 8: Add recoverable output-creation error handling**

Wrap explicit output generation calls in `OverviewTab.tsx` so backend `template_not_found` renders an in-page error:

```ts
try {
  await createWatchlistOutput(payload)
} catch (error) {
  setPipelineError(formatWatchlistError(error, "Could not create the digest output"))
  return
}
```

The message must include the failed template name when present, and must not let the Next.js runtime overlay become the only feedback.

- [ ] **Step 9: Update scale fixtures**

Change `shared/scale-benchmark.ts` fixture output prefs from `template_name: "briefing_md"` to `template_name: "briefing_markdown"`.

- [ ] **Step 10: Run focused frontend tests and verify pass**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Watchlists/OverviewTab/__tests__/quick-setup.test.ts \
  src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts \
  src/components/Option/Watchlists/OverviewTab/__tests__/OverviewTab.quick-setup.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

Expected: PASS.

- [ ] **Step 11: Run template regression tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Watchlists/TemplatesTab/__tests__/template-recipes.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: PASS, with recipe IDs still valid for template creation UI.

- [ ] **Step 12: Commit**

Run:

```bash
git add \
  apps/packages/ui/src/components/Option/Watchlists/shared/templateNames.ts \
  apps/packages/ui/src/components/Option/Watchlists/OverviewTab/quick-setup.ts \
  apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-contract.ts \
  apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx \
  apps/packages/ui/src/components/Option/Watchlists/shared/scale-benchmark.ts \
  apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/quick-setup.test.ts \
  apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts \
  apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/OverviewTab.quick-setup.test.tsx
git commit -m "fix: normalize watchlists digest template payloads"
```

## Task 2: Audio Scheduler Submit Hotfix

**Files:**
- Modify: `tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py`
- Modify: `tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py`
- Test: `tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py`
- Test: `tldw_Server_API/tests/Watchlists/test_watchlists_api.py`

- [x] **Step 1: Write failing audio enqueue test**

In `test_audio_briefing_workflow.py`, replace enqueue-oriented mocks with a scheduler double that only exposes `submit`:

```python
class FakeScheduler:
    def __init__(self) -> None:
        self.calls = []

    async def submit(self, handler, payload=None, priority=None, queue_name=None,
                     depends_on=None, idempotency_key=None, metadata=None, auth_context=None):
        user_id = metadata.get("user_id") if isinstance(metadata, dict) else None
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("Task metadata must include a non-empty 'user_id'")
        self.calls.append({
            "handler": handler,
            "payload": payload,
            "queue_name": queue_name,
            "idempotency_key": idempotency_key,
            "metadata": metadata,
        })
        return "task_audio_123"
```

Assert:

```python
assert result == "task_audio_123"
assert scheduler.calls[0]["handler"] == "workflow_run"
assert scheduler.calls[0]["payload"]["definition_snapshot"]["name"] == "audio_briefing"
assert scheduler.calls[0]["payload"]["metadata"]["watchlist_run_id"] == 123
assert scheduler.calls[0]["queue_name"] == "workflows"
assert scheduler.calls[0]["idempotency_key"] == "watchlist-audio-briefing:1:42:123"
assert scheduler.calls[0]["metadata"]["user_id"] == "1"
```

- [x] **Step 2: Run the failing backend test**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py -q
```

Expected: FAIL because production code calls `enqueue`.

- [x] **Step 3: Replace enqueue with submit**

In `audio_briefing_workflow.py`, remove the `Task` import and replace the enqueue block with:

```python
scheduler = await get_global_scheduler()
task_id = await scheduler.submit(
    "workflow_run",
    payload={
        "user_id": user_id,
        "definition_snapshot": AUDIO_BRIEFING_WORKFLOW_DEF,
        "inputs": workflow_inputs,
        "mode": "async",
        "metadata": {
            "source": "watchlist_audio_briefing",
            "watchlist_job_id": job_id,
            "watchlist_run_id": run_id,
        },
    },
    queue_name="workflows",
    idempotency_key=f"watchlist-audio-briefing:{user_id}:{job_id}:{run_id}",
    metadata={
        "source": "watchlist_audio_briefing",
        "watchlist_job_id": job_id,
        "watchlist_run_id": run_id,
        "user_id": str(user_id),
    },
)
```

Do not fabricate an audio artifact at enqueue time. This task only proves the request enters Scheduler.

- [x] **Step 4: Preserve skip behavior**

Keep existing returns of `None` for `generate_audio=false`, no ingested items, item load failure, or Scheduler submission failure. The caller already maps `None` to `skipped`; later tasks improve the visible reason.

- [x] **Step 5: Run audio workflow tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py -q
```

Expected: PASS.

- [x] **Step 6: Run endpoint metadata tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Watchlists/test_watchlists_api.py::test_outputs_generate_audio_payload_triggers_workflow_and_updates_run_stats \
  tldw_Server_API/tests/Watchlists/test_watchlists_api.py::test_outputs_generate_audio_trigger_returns_none_marks_skipped_metadata \
  tldw_Server_API/tests/Watchlists/test_watchlists_api.py::test_outputs_generate_audio_trigger_failure_marks_enqueue_failed_metadata \
  -q
```

Expected: PASS.

- [x] **Step 7: Run audio delivery endpoint tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py -q
```

Expected: PASS.

- [x] **Step 8: Commit**

Run:

```bash
git add \
  Docs/superpowers/plans/2026-05-20-watchlists-demo-remediation-implementation-plan.md \
  tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py \
  tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py
git commit -m "fix: submit watchlist audio briefings through scheduler"
```

## Task 3: Minimal Audio Status UI And Service Contract

**Files:**
- Modify: `apps/packages/ui/src/types/watchlists.ts`
- Modify: `apps/packages/ui/src/services/watchlists.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-contract.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx`
- Test: `apps/packages/ui/src/services/__tests__/watchlists-audio.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx`

- [ ] **Step 1: Write service test for run audio endpoint**

Create `watchlists-audio.test.ts` with a `bgRequest` mock proving:

```ts
await getWatchlistRunAudio(42)
expect(bgRequest).toHaveBeenCalledWith({
  path: "/api/v1/watchlists/runs/42/audio",
  method: "GET"
})
```

- [ ] **Step 2: Write metadata tests for pending/skipped/enqueue_failed**

In `outputMetadata.test.ts`, add cases:

```ts
expect(getOutputAudioStatus({ audio_briefing_requested: true, audio_briefing_status: "pending" }))
  .toMatchObject({ kind: "pending" })

expect(getOutputAudioStatus({ audio_briefing_requested: true, audio_briefing_status: "skipped" }))
  .toMatchObject({ kind: "skipped" })

expect(getOutputAudioStatus({
  audio_briefing_requested: true,
  audio_briefing_status: "enqueue_failed",
  audio_briefing_error: "Scheduler not started"
})).toMatchObject({ kind: "failed", reason: "Scheduler not started" })
```

- [ ] **Step 3: Write pipeline output audio request test**

In `pipeline-contract.test.ts`, assert manual/test output creation sends backend audio trigger fields when audio is selected:

```ts
const payload = toPipelineOutputCreatePayload(123, {
  ...draft,
  includeAudio: true,
  audioVoice: "af_heart",
  targetAudioMinutes: 8
})

expect(payload).toMatchObject({
  run_id: 123,
  generate_audio: true,
  audio_voice: "af_heart",
  target_audio_minutes: 8
})
```

This is separate from `metadata.audio`; backend output creation only triggers the audio bridge from explicit audio request fields.

- [ ] **Step 4: Run focused tests and confirm failure**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/services/__tests__/watchlists-audio.test.ts \
  src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts \
  src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL because helper/types/status parser do not exist yet and output creation does not send audio trigger fields.

- [ ] **Step 5: Add run audio status and output request types**

In `types/watchlists.ts`, add:

```ts
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

Extend `WatchlistOutputCreate` with explicit audio request fields:

```ts
  generate_audio?: boolean
  target_audio_minutes?: number
  audio_voice?: string
  audio_speed?: number
  audio_language?: string
  llm_provider?: string
  llm_model?: string
  voice_map?: Record<string, string>
```

- [ ] **Step 6: Add frontend service helper**

In `services/watchlists.ts`, import the new type and add:

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

- [ ] **Step 7: Send audio trigger fields from pipeline output creation**

In `pipeline-contract.ts`, update `toPipelineOutputCreatePayload`:

```ts
if (draft.includeAudio) {
  payload.generate_audio = true
  payload.audio_voice = String(draft.audioVoice || "").trim() || undefined
  payload.target_audio_minutes = Number(draft.targetAudioMinutes)
  payload.metadata = {
    ...(payload.metadata || {}),
    audio: {
      enabled: true,
      voice: payload.audio_voice || null,
      target_minutes: payload.target_audio_minutes
    }
  }
}
```

- [ ] **Step 8: Add pure output metadata parser**

In `outputMetadata.ts`, add a pure helper similar to:

```ts
export type OutputAudioStatus =
  | { kind: "none"; requested: false }
  | { kind: "pending"; requested: true; taskId?: string }
  | { kind: "skipped"; requested: true; reason?: string }
  | { kind: "failed"; requested: true; reason?: string }
  | { kind: "completed"; requested: true; taskId?: string; audioUri?: string }

export const getOutputAudioStatus = (metadata: unknown): OutputAudioStatus => {
  const record = isRecord(metadata) ? metadata : {}
  if (record.audio_briefing_requested !== true) return { kind: "none", requested: false }
  const status = asNonEmptyString(record.audio_briefing_status)
  if (status === "pending") {
    return { kind: "pending", requested: true, taskId: asNonEmptyString(record.audio_briefing_task_id) }
  }
  if (status === "enqueue_failed") {
    return { kind: "failed", requested: true, reason: asNonEmptyString(record.audio_briefing_error) }
  }
  if (status === "skipped") {
    return { kind: "skipped", requested: true, reason: asNonEmptyString(record.audio_briefing_error) }
  }
  if (status === "completed") {
    return { kind: "completed", requested: true, taskId: asNonEmptyString(record.audio_briefing_task_id) }
  }
  return { kind: "pending", requested: true, taskId: asNonEmptyString(record.audio_briefing_task_id) }
}
```

- [ ] **Step 9: Show audio status in output preview**

In `OutputPreviewDrawer.tsx`, render:

- pending: "Audio briefing queued" plus task id when present.
- skipped: "Audio briefing skipped" plus reason when present.
- failed: "Audio briefing failed" plus safe reason.
- completed with no artifact URL: "Audio briefing complete; artifact details unavailable".
- none: do not render the audio block.

Do not show a player unless a verified URI/download URL exists.

- [ ] **Step 10: Show audio status in run detail**

In `RunDetailDrawer.tsx`, show the same status language in the run timeline/summary. If `/runs/{run_id}/audio` returns 404, render "No audio briefing requested for this run" only when output metadata also lacks an audio request; otherwise render a warning that the output requested audio but run audio is missing.

- [ ] **Step 11: Run focused frontend tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/services/__tests__/watchlists-audio.test.ts \
  src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts \
  src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts \
  src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx \
  src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

Expected: PASS.

- [ ] **Step 12: Commit**

Run:

```bash
git add \
  apps/packages/ui/src/types/watchlists.ts \
  apps/packages/ui/src/services/watchlists.ts \
  apps/packages/ui/src/services/__tests__/watchlists-audio.test.ts \
  apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-contract.ts \
  apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/outputMetadata.ts \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/outputMetadata.test.ts \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx \
  apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx \
  apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx
git commit -m "feat: surface watchlist audio briefing status"
```

## Task 4: Source And Run Health Truthfulness

**Files:**
- Modify: `tldw_Server_API/app/core/Watchlists/pipeline.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Modify: `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
- Modify: `apps/packages/ui/src/types/watchlists.ts`
- Modify: `apps/packages/ui/src/services/watchlists-overview.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/shared/WatchlistsHealthBar.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx`
- Test: `tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py`
- Test: `apps/packages/ui/src/services/__tests__/watchlists-overview.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.health.test.tsx`

- [x] **Step 1: Write backend failing test for failed source with zero items**

Create or extend `test_watchlists_operator_recovery.py` with a case where one active source returns `error:403`, the run ingests zero items, and run stats include source failure information:

```python
assert run["status"] in {"completed", "succeeded", "partial", "warning"}
assert run["stats"]["source_errors"] >= 1
assert run["stats"]["source_statuses"][0]["status"].startswith("error:")
```

The exact run status may remain `completed` for backward compatibility, but stats must carry warning evidence.

- [x] **Step 2: Run backend test and confirm failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py -q
```

Expected: FAIL if source failures are not persisted or exposed.

- [x] **Step 3: Persist source failure stats in pipeline**

In `pipeline.py`, add source-level stats fields when fetch/extraction fails:

```python
stats.setdefault("source_statuses", []).append({
    "source_id": source_id,
    "name": source_name,
    "status": source_status,
    "error": safe_error_message,
    "items_found": found_count,
    "items_ingested": ingested_count,
})
stats["source_errors"] = sum(
    1 for source in stats.get("source_statuses", [])
    if str(source.get("status", "")).startswith("error")
)
```

Keep the error message safe: no secrets, tokens, or full credentials.

- [x] **Step 4: Expose stats without breaking existing run response shape**

In `watchlists.py`, ensure list/detail endpoints include the persisted stats object already returned for runs. Add schema fields only if current schemas strip unknown stats keys.

- [x] **Step 5: Run backend tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py \
  tldw_Server_API/tests/Watchlists/test_watchlists_pipeline.py \
  tldw_Server_API/tests/Watchlists/test_run_detail_filters_totals.py \
  -q
```

Expected: PASS.

- [x] **Step 6: Write frontend health aggregation tests**

In `watchlists-overview.test.ts`, add cases:

```ts
expect(buildWatchlistsOverviewHealth({
  sources: [{ id: 1, status: "error:403", active: true }],
  runs: [{ id: 9, status: "succeeded", stats: { source_errors: 1, items_ingested: 0 } }],
  outputs: []
})).toMatchObject({ level: "warning" })
```

Also assert the title is not `System healthy`.

- [x] **Step 7: Implement health aggregation**

In `watchlists-overview.ts`, count:

- active sources with `status` beginning `error:`
- recent runs with `stats.source_errors > 0`
- zero-item recent runs with source errors
- outputs with `audio_briefing_status` of `enqueue_failed` or `skipped` when audio was requested
- delivery/output errors already exposed by current metadata

- [x] **Step 8: Render warning state**

In `WatchlistsHealthBar.tsx` and `OverviewTab.tsx`, render warning/partial state with links to affected Source, Activity, or Reports tab. Preserve the existing healthy state when no unresolved warnings exist.

- [x] **Step 9: Run focused frontend tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/services/__tests__/watchlists-overview.test.ts \
  src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.health.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

Expected: PASS.

- [x] **Step 10: Commit**

Run:

```bash
git add \
  tldw_Server_API/app/core/Watchlists/pipeline.py \
  tldw_Server_API/app/api/v1/endpoints/watchlists.py \
  tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py \
  tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py \
  apps/packages/ui/src/types/watchlists.ts \
  apps/packages/ui/src/services/watchlists-overview.ts \
  apps/packages/ui/src/services/__tests__/watchlists-overview.test.ts \
  apps/packages/ui/src/components/Option/Watchlists/shared/WatchlistsHealthBar.tsx \
  apps/packages/ui/src/components/Option/Watchlists/OverviewTab/OverviewTab.tsx \
  apps/packages/ui/src/components/Option/Watchlists/__tests__/WatchlistsPlaygroundPage.health.test.tsx
git commit -m "fix: reflect watchlist source failures in health"
```

## Task 5: Demo Runbook And Live Verification Gate

**Files:**
- Create: `Docs/Runbooks/watchlists_demo_readiness_2026_05_20.md`
- Create: `apps/tldw-frontend/e2e/workflows/watchlists-demo-readiness.spec.ts`
- Modify: `apps/extension/tests/e2e/watchlists.spec.ts`

- [ ] **Step 1: Write demo runbook skeleton**

Create `Docs/Runbooks/watchlists_demo_readiness_2026_05_20.md` with sections:

```markdown
# Watchlists Demo Readiness Runbook

## Environment
## Preflight Sources
## Provider And Voice Preflight
## WebUI Same-Origin Path
## Extension Path
## Demo Script Safe Claims
## Hard Stops
## Known Degradations
```

- [ ] **Step 2: Add source preflight commands**

Record the source-test command shape:

```bash
curl -sf \
  -H "X-API-Key: $SINGLE_USER_API_KEY" \
  -H "Content-Type: application/json" \
  -X POST http://127.0.0.1:8000/api/v1/watchlists/sources/test \
  --data '{"url":"https://example.com/rss.xml","source_type":"rss"}'
```

The runbook must say local loopback feeds are not valid demo sources unless backend policy explicitly allows them.

- [ ] **Step 3: Add audio preflight distinction**

Document two separate gates:

- Scheduler enqueue gate: `generate_audio=true` produces a task id and `/runs/{run_id}/audio` returns a meaningful status.
- Final playback gate: provider, model, voice, script, per-speaker audio, and final mix all complete and produce a playable/downloadable artifact.

- [ ] **Step 4: Write WebUI Playwright smoke**

Create `watchlists-demo-readiness.spec.ts` that uses mocked or live-configured endpoints to assert:

- `/watchlists` loads.
- creating source/monitor sends `briefing_markdown`.
- output creation failure renders in-app error, not a runtime overlay.
- audio status renders pending/failed/skipped truthfully.

- [ ] **Step 5: Extend extension strict smoke**

In `apps/extension/tests/e2e/watchlists.spec.ts`, add or update a test to verify:

- extension `/watchlists` route mounts.
- the shared route can render Activity/Reports status.
- output generation errors do not crash the extension page.

- [ ] **Step 6: Run WebUI smoke**

Run:

```bash
cd apps/tldw-frontend
npx playwright test e2e/workflows/watchlists-demo-readiness.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 7: Run extension strict watchlists smoke**

Run the existing strict command used by the repo for extension watchlists. If Chrome launch is blocked by sandboxing, rerun with the approved escalated Playwright command:

```bash
cd apps/extension
npx playwright test tests/e2e/watchlists.spec.ts --reporter=line
```

Expected: PASS with no skipped watchlists tests.

- [ ] **Step 8: Run backend demo-scope tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Watchlists/test_e2e_rss_briefing.py \
  tldw_Server_API/tests/Watchlists/test_full_pipeline_integration.py \
  tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py \
  tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py \
  -q
```

Expected: PASS.

- [ ] **Step 9: Run touched Python Bandit**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py \
  tldw_Server_API/app/core/Watchlists/pipeline.py \
  tldw_Server_API/app/api/v1/endpoints/watchlists.py \
  -f json -o /tmp/bandit_watchlists_demo_rescue.json
```

Expected: no new findings in touched code.

- [ ] **Step 10: Commit**

Run:

```bash
git add \
  Docs/Runbooks/watchlists_demo_readiness_2026_05_20.md \
  apps/tldw-frontend/e2e/workflows/watchlists-demo-readiness.spec.ts \
  apps/extension/tests/e2e/watchlists.spec.ts
git commit -m "test: add watchlists demo readiness gate"
```

## Task 6: First-Time Cadence And Review Cleanup

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/quick-setup.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/PipelineWizard.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-contract.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/schedule-utils.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/SchedulePicker.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/PipelineWizard.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/schedule-utils.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/SchedulePicker.help.test.tsx`

- [ ] **Step 1: Write schedule utility tests**

Add cases for:

- every 5 hours
- every 30 minutes
- weekly on selected weekday
- manual/no schedule
- existing raw cron preservation

Expected payload examples:

```ts
expect(resolveQuickSetupSchedule({ kind: "interval", unit: "hour", every: 5 }))
  .toMatchObject({ schedule_type: "interval", interval_hours: 5 })

expect(resolveQuickSetupSchedule({ kind: "weekly", weekday: "mon", time: "08:00" }))
  .toMatchObject({ schedule_type: "cron" })
```

- [ ] **Step 2: Write review summary tests**

In `PipelineWizard.test.tsx`, assert:

- one source displays as one source, not zero feeds.
- audio off displays only text digest/report copy.
- audio on displays "1 speaker", "2 speakers", "3 speakers", or "4 speakers" based on the cast, not a fixed 3-person podcast assumption.

- [ ] **Step 3: Run tests and confirm failure**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Watchlists/OverviewTab/__tests__/PipelineWizard.test.tsx \
  src/components/Option/Watchlists/JobsTab/__tests__/schedule-utils.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL for unsupported variable cadence and existing summary contradictions.

- [ ] **Step 4: Extend schedule draft model**

In `quick-setup.ts` and `pipeline-contract.ts`, represent cadence as:

```ts
type WatchlistCadenceDraft =
  | { kind: "manual" }
  | { kind: "interval"; every: number; unit: "minute" | "hour" }
  | { kind: "daily"; time?: string }
  | { kind: "weekdays"; time?: string }
  | { kind: "weekly"; weekday: string; time?: string }
  | { kind: "advanced"; cron: string }
```

- [ ] **Step 5: Implement variable cadence conversion**

Map interval and weekly drafts into the existing job schedule payload shape already accepted by `JobFormModal`/backend. Do not invent a new backend schedule contract unless current fields cannot express the cadence.

- [ ] **Step 6: Update wizard controls**

In `PipelineWizard.tsx`, add variable cadence controls using the existing SchedulePicker patterns. Keep advanced cron behind an explicit advanced disclosure.

- [ ] **Step 7: Fix review summary**

Use normalized draft state, not stale form labels, for:

- source count
- cadence label
- output template/report type
- delivery
- optional audio speaker count
- first-run behavior

- [ ] **Step 8: Run focused tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Watchlists/OverviewTab/__tests__/PipelineWizard.test.tsx \
  src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts \
  src/components/Option/Watchlists/JobsTab/__tests__/schedule-utils.test.ts \
  src/components/Option/Watchlists/JobsTab/__tests__/SchedulePicker.help.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

Expected: PASS.

- [ ] **Step 9: Commit**

Run:

```bash
git add \
  apps/packages/ui/src/components/Option/Watchlists/OverviewTab/quick-setup.ts \
  apps/packages/ui/src/components/Option/Watchlists/OverviewTab/PipelineWizard.tsx \
  apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-contract.ts \
  apps/packages/ui/src/components/Option/Watchlists/JobsTab/schedule-utils.ts \
  apps/packages/ui/src/components/Option/Watchlists/JobsTab/SchedulePicker.tsx \
  apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/PipelineWizard.test.tsx \
  apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts \
  apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/schedule-utils.test.ts \
  apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/SchedulePicker.help.test.tsx
git commit -m "feat: support variable watchlist cadence setup"
```

## Task 7: Digest And Newsletter Output Contract

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-contract.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobFormModal.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/job-summaries.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputsTab.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/JobFormModal.live-summary.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/JobsTab.scope-filter-summary.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.regenerate-modal.test.tsx`
- Test: `tldw_Server_API/tests/Watchlists/test_job_output_prefs_roundtrip.py`
- Test: `tldw_Server_API/tests/Watchlists/test_newsletter_briefing_gaps.py`

- [ ] **Step 1: Write scheduled output contract tests**

In `pipeline-contract.test.ts`, add two cases:

```ts
expect(toPipelineJobCreatePayload({
  ...draft,
  createScheduledOutput: true,
  templateName: "briefing_md"
}).output_prefs).toMatchObject({
  auto_output: {
    enabled: true,
    type: "briefing_markdown",
    template_name: "briefing_markdown"
  },
  template: { default_name: "briefing_markdown" }
})

expect(toPipelineJobCreatePayload({
  ...draft,
  createScheduledOutput: false
}).output_prefs?.auto_output?.enabled).not.toBe(true)
```

If the draft model does not yet have `createScheduledOutput`, add it in the same test as the explicit user choice.

- [ ] **Step 2: Write job form summary tests**

In `JobFormModal.live-summary.test.tsx`, assert the save review distinguishes:

- scheduled report creation enabled
- manual/test generation only
- delivery enabled versus in-app Reports only
- audio enabled versus text digest only

- [ ] **Step 3: Write Reports discoverability test**

In `OutputsTab.regenerate-modal.test.tsx`, assert regenerated digest/newsletter output shows:

- template name
- run id or source run linkage
- item count when present
- delivery status
- audio status if audio requested

- [ ] **Step 4: Run frontend tests and confirm failure**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts \
  src/components/Option/Watchlists/JobsTab/__tests__/JobFormModal.live-summary.test.tsx \
  src/components/Option/Watchlists/JobsTab/__tests__/JobsTab.scope-filter-summary.test.tsx \
  src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.regenerate-modal.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL where scheduled output intent and report state are implicit or mislabeled.

- [ ] **Step 5: Implement explicit scheduled output preference**

In `pipeline-contract.ts`, set scheduled output only when requested:

```ts
if (draft.createScheduledOutput) {
  outputPrefs.auto_output = {
    enabled: true,
    type: "briefing_markdown",
    template_name: templateName
  }
}
```

Keep one-off test output creation in `toPipelineOutputCreatePayload`; do not conflate it with scheduled `auto_output`.

- [ ] **Step 6: Update job form copy and summaries**

In `JobFormModal.tsx` and `job-summaries.ts`, render explicit phrases:

- "Create a report after each scheduled run"
- "Manual/test reports only"
- "Deliver by email"
- "Save to Chatbook"
- "Reports tab only"
- "Audio briefing requested"

Keep raw cron, template selector, and existing advanced controls reachable.

- [ ] **Step 7: Keep Reports state grounded in backend artifacts**

In `OutputsTab.tsx`, show output state from actual output records and metadata. Do not show "newsletter sent" or "audio ready" unless the output metadata says delivery/audio succeeded.

- [ ] **Step 8: Run backend output contract tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Watchlists/test_job_output_prefs_roundtrip.py \
  tldw_Server_API/tests/Watchlists/test_newsletter_briefing_gaps.py \
  -q
```

Expected: PASS.

- [ ] **Step 9: Run frontend output contract tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts \
  src/components/Option/Watchlists/JobsTab/__tests__/JobFormModal.live-summary.test.tsx \
  src/components/Option/Watchlists/JobsTab/__tests__/JobsTab.scope-filter-summary.test.tsx \
  src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.regenerate-modal.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

Expected: PASS.

- [ ] **Step 10: Commit**

Run:

```bash
git add \
  apps/packages/ui/src/components/Option/Watchlists/OverviewTab/pipeline-contract.ts \
  apps/packages/ui/src/components/Option/Watchlists/OverviewTab/__tests__/pipeline-contract.test.ts \
  apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobFormModal.tsx \
  apps/packages/ui/src/components/Option/Watchlists/JobsTab/job-summaries.ts \
  apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/JobFormModal.live-summary.test.tsx \
  apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/JobsTab.scope-filter-summary.test.tsx \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputsTab.tsx \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputsTab.regenerate-modal.test.tsx
git commit -m "feat: clarify watchlists digest output contract"
```

## Task 8: Source Validation And Dedupe Confidence

**Files:**
- Create: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/source-settings.ts`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourceFormModal.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesTab.tsx`
- Modify: `tldw_Server_API/app/core/Watchlists/fetchers.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Test: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/source-settings.test.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourceFormModal.test-source.test.tsx`
- Test: `tldw_Server_API/tests/Watchlists/test_fetchers_scrape_rules.py`
- Test: `tldw_Server_API/tests/Watchlists/test_preview_endpoint.py`

- [ ] **Step 1: Write source settings helper tests**

Create tests proving unknown keys survive:

```ts
const merged = mergeSourceSettings(
  { vendor_specific: true, scrape: { selector: ".old" } },
  { scrape: { selector: ".article" } }
)
expect(merged.vendor_specific).toBe(true)
expect(merged.scrape.selector).toBe(".article")
```

- [ ] **Step 2: Write source test UI assertions**

In `SourceFormModal.test-source.test.tsx`, assert a failed source test shows:

- HTTP/fetch status
- selector diagnostics when available
- sample item count
- dedupe identity preview

- [ ] **Step 3: Run frontend tests and confirm failure**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Watchlists/SourcesTab/__tests__/source-settings.test.ts \
  src/components/Option/Watchlists/SourcesTab/__tests__/SourceFormModal.test-source.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL until helpers/UI are added.

- [ ] **Step 4: Implement source settings helpers**

Create `source-settings.ts` with pure helpers:

```ts
export const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

export const mergeSourceSettings = (
  current: unknown,
  patch: Record<string, unknown>
): Record<string, unknown> => ({
  ...(isRecord(current) ? current : {}),
  ...patch
})
```

Add typed parse/serialize helpers for scrape selectors, extraction mode, dedupe key, and advanced JSON. Invalid advanced JSON should block save with an inline error.

- [ ] **Step 5: Pass draft settings to source test**

In `SourceFormModal.tsx`, include normalized draft `settings` in source-test calls so test results reflect what the user is saving.

- [ ] **Step 6: Surface diagnostics**

Render:

- fetch result/status
- sample item titles or count
- selector diagnostics from backend
- dedupe identity preview such as "URL + canonical URL" or the configured custom identity
- warning when no items are found

- [ ] **Step 7: Extend backend diagnostics only where missing**

In `fetchers.py`/`watchlists.py`, expose `validate_selector_rules` diagnostics from source test responses. Do not change the stored source contract unless the UI needs a new persisted setting.

- [ ] **Step 8: Run backend source tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Watchlists/test_fetchers_scrape_rules.py \
  tldw_Server_API/tests/Watchlists/test_preview_endpoint.py \
  -q
```

Expected: PASS.

- [ ] **Step 9: Run frontend source tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Watchlists/SourcesTab/__tests__/source-settings.test.ts \
  src/components/Option/Watchlists/SourcesTab/__tests__/SourceFormModal.test-source.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

Expected: PASS.

- [ ] **Step 10: Commit**

Run:

```bash
git add \
  apps/packages/ui/src/components/Option/Watchlists/SourcesTab/source-settings.ts \
  apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourceFormModal.tsx \
  apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesTab.tsx \
  apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/source-settings.test.ts \
  apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourceFormModal.test-source.test.tsx \
  tldw_Server_API/app/core/Watchlists/fetchers.py \
  tldw_Server_API/app/api/v1/endpoints/watchlists.py \
  tldw_Server_API/tests/Watchlists/test_fetchers_scrape_rules.py \
  tldw_Server_API/tests/Watchlists/test_preview_endpoint.py
git commit -m "feat: explain watchlist source validation and dedupe"
```

## Task 9: Durable 1-4 Speaker Audio Artifacts

**Files:**
- Modify: `apps/packages/ui/src/types/watchlists.ts`
- Modify: `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`
- Modify: `tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx`
- Test: `tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py`
- Test: `tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py`
- Test: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx`

- [ ] **Step 1: Write backend artifact response tests**

In `test_audio_output_delivery.py`, add a completed task fixture that returns:

```python
{
    "script_artifact": {"id": "script_1", "text": "..."},
    "speaker_artifacts": [
        {"speaker_id": "host", "script": "...", "audio_uri": "file:///tmp/host.wav"}
    ],
    "final_artifact": {"audio_uri": "file:///tmp/final.mp3", "mime_type": "audio/mpeg"},
}
```

Assert `/runs/{run_id}/audio` returns those fields and status `completed`.

- [ ] **Step 2: Write frontend rendering tests**

In `OutputPreviewDrawer.audio.test.tsx`, assert:

- 1-speaker, 2-speaker, 3-speaker, and 4-speaker labels render from metadata.
- per-speaker pending/failed/completed status is visible.
- final player/download appears only when final artifact URL exists.
- fallback reason renders when no final mix exists.

- [ ] **Step 3: Run tests and confirm failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py -q
cd apps/packages/ui
bunx vitest run src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: FAIL where artifact schema/UI is incomplete.

- [ ] **Step 4: Add backend schema fields**

In `watchlists_schemas.py`, add typed models:

```python
class WatchlistAudioArtifact(BaseModel):
    id: str | int | None = None
    speaker_id: str | None = None
    script: str | None = None
    audio_uri: str | None = None
    download_url: str | None = None
    mime_type: str | None = None
    size_bytes: int | None = None
    status: str | None = None
    error: str | None = None

class WatchlistRunAudioStatus(BaseModel):
    run_id: int
    task_id: str | None = None
    status: str
    script_artifact: WatchlistAudioArtifact | None = None
    speaker_artifacts: list[WatchlistAudioArtifact] = Field(default_factory=list)
    final_artifact: WatchlistAudioArtifact | None = None
    fallback_reason: str | None = None
    error: str | None = None
```

- [ ] **Step 5: Persist artifact metadata**

In `audio_briefing_workflow.py`, when workflow output data is available, persist stage metadata to the associated watchlist output/run stats instead of relying on transient task state only.

Do not block Task 2's enqueue hotfix on final artifact generation. This task is the durable artifact work.

- [ ] **Step 6: Return artifacts from run audio endpoint**

In `watchlists.py`, update `/runs/{run_id}/audio` to prefer final audio artifact when present, but also return script and speaker artifacts. Preserve current 404 behavior only for true "no audio requested" runs.

- [ ] **Step 7: Render artifact UI**

In `OutputPreviewDrawer.tsx` and `RunDetailDrawer.tsx`, render:

- script preview/copy
- per-speaker row with voice, script status, audio status
- final player/download
- retry action placeholder only when retry endpoint exists

- [ ] **Step 8: Run backend audio tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py \
  tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py \
  -q
```

Expected: PASS.

- [ ] **Step 9: Run frontend audio tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx \
  src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

Expected: PASS.

- [ ] **Step 10: Commit**

Run:

```bash
git add \
  apps/packages/ui/src/types/watchlists.ts \
  tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py \
  tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py \
  tldw_Server_API/app/api/v1/endpoints/watchlists.py \
  tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py \
  tldw_Server_API/tests/Watchlists/test_audio_output_delivery.py \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/__tests__/OutputPreviewDrawer.audio.test.tsx \
  apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx
git commit -m "feat: persist watchlist audio briefing artifacts"
```

## Task 10: Operator Recovery And Stage Diagnostics

**Files:**
- Modify: `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- Modify: `tldw_Server_API/app/core/Watchlists/pipeline.py`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/shared/WatchlistsCommandPalette.tsx`
- Test: `tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py`
- Test: `apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx`

- [ ] **Step 1: Write backend retry contract tests**

Add tests for:

- retry output render without rerunning ingestion
- retry delivery without recreating output
- retry audio without scraping sources
- diagnostic bundle includes run id, job id, source statuses, output ids, delivery status, audio status, and safe errors

- [ ] **Step 2: Run backend tests and confirm failure**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py -q
```

Expected: FAIL until endpoints/diagnostics are added.

- [ ] **Step 3: Add minimal safe retry endpoints**

Only add endpoints that are not already expressible:

```python
POST /api/v1/watchlists/runs/{run_id}/outputs:retry
POST /api/v1/watchlists/outputs/{output_id}/delivery:retry
POST /api/v1/watchlists/runs/{run_id}/audio:retry
GET  /api/v1/watchlists/runs/{run_id}/diagnostics
```

Each endpoint must check ownership/auth with existing Watchlists dependencies and must be idempotent or clearly duplicate-safe.

- [ ] **Step 4: Add stage-level diagnostics**

Build diagnostics from existing run stats/output metadata first. Include:

- fetch
- extraction
- dedupe
- filters
- output render
- delivery
- audio script
- per-speaker audio
- final mix

- [ ] **Step 5: Run backend recovery tests**

Run:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py \
  tldw_Server_API/tests/Watchlists/test_delivery_integrations.py \
  tldw_Server_API/tests/Watchlists/test_runs_csv_export.py \
  -q
```

Expected: PASS.

- [ ] **Step 6: Write frontend retry/diagnostic tests**

In `RunDetailDrawer.stream-lifecycle.test.tsx`, assert stage rows and retry actions appear for failed output, delivery, or audio. Actions must use clear labels and confirmation when duplicate-producing.

- [ ] **Step 7: Implement frontend controls**

Render separate actions:

- Retry output
- Retry delivery
- Retry audio
- Rerun ingestion
- Download diagnostics

Do not collapse them into one "Try again" button.

- [ ] **Step 8: Run focused frontend tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: PASS.

- [ ] **Step 9: Commit**

Run:

```bash
git add \
  tldw_Server_API/app/api/v1/endpoints/watchlists.py \
  tldw_Server_API/app/core/Watchlists/pipeline.py \
  tldw_Server_API/tests/Watchlists/test_watchlists_operator_recovery.py \
  apps/packages/ui/src/components/Option/Watchlists/RunsTab/RunDetailDrawer.tsx \
  apps/packages/ui/src/components/Option/Watchlists/RunsTab/__tests__/RunDetailDrawer.stream-lifecycle.test.tsx \
  apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx \
  apps/packages/ui/src/components/Option/Watchlists/shared/WatchlistsCommandPalette.tsx
git commit -m "feat: add watchlists recovery diagnostics"
```

## Task 11: Power-User Throughput And Preservation

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobsTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesTab.tsx`
- Modify: `apps/packages/ui/src/components/Option/Watchlists/shared/WatchlistsCommandPalette.tsx`
- Modify: `apps/tldw-frontend/e2e/workflows/watchlists-items.spec.ts`
- Modify: `apps/extension/tests/e2e/watchlists.spec.ts`
- Test: `apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/JobsTab.advanced-details.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourceSeenDrawer.test.tsx`

- [ ] **Step 1: Write preservation smoke checklist in tests**

Add WebUI/extension smoke assertions for:

- full-view tabs
- OPML import/export
- raw cron
- raw JSON/source settings preservation
- batch source/item controls
- source seen/dedupe controls
- template listing/editing
- report preview/download
- diagnostics/export
- command palette access

- [ ] **Step 2: Run preservation tests and confirm current baseline**

Run:

```bash
cd apps/tldw-frontend
npx playwright test e2e/workflows/watchlists-items.spec.ts --reporter=line
cd ../extension
npx playwright test tests/e2e/watchlists.spec.ts --reporter=line
```

Expected: existing smoke should pass or reveal true current gaps. Do not hide failures by deleting coverage.

- [ ] **Step 3: Add clone monitor action**

In `JobsTab.tsx`, add a clone action that opens `JobFormModal` with copied values:

- name suffixed with "copy"
- active defaults false unless user explicitly activates
- schedule, filters, output prefs, scope, and settings preserved

- [ ] **Step 4: Add clone source rules action**

In `SourcesTab.tsx`, add clone source action:

- URL remains editable
- source type, settings, tags, groups, extraction rules, dedupe rules preserved
- active defaults false until validated

- [ ] **Step 5: Add command palette commands**

In `WatchlistsCommandPalette.tsx`, add commands:

- Create pipeline
- Clone selected monitor
- Clone selected source
- Validate selected sources
- Run selected monitor
- Retry failed output/delivery/audio
- Export diagnostics

- [ ] **Step 6: Add batch operation summaries**

Before committing batch changes, render an impact summary:

- count affected
- active/inactive impact
- cadence/output/delivery changes
- whether operation can duplicate ingestion or delivery

- [ ] **Step 7: Run focused component tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Watchlists/JobsTab/__tests__/JobsTab.advanced-details.test.tsx \
  src/components/Option/Watchlists/SourcesTab/__tests__/SourceSeenDrawer.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

Expected: PASS.

- [ ] **Step 8: Run WebUI and extension preservation smoke**

Run:

```bash
cd apps/tldw-frontend
npx playwright test e2e/workflows/watchlists-items.spec.ts --reporter=line
cd ../extension
npx playwright test tests/e2e/watchlists.spec.ts --reporter=line
```

Expected: PASS with no regression in existing advanced workflows.

- [ ] **Step 9: Commit**

Run:

```bash
git add \
  apps/packages/ui/src/components/Option/Watchlists/JobsTab/JobsTab.tsx \
  apps/packages/ui/src/components/Option/Watchlists/SourcesTab/SourcesTab.tsx \
  apps/packages/ui/src/components/Option/Watchlists/shared/WatchlistsCommandPalette.tsx \
  apps/tldw-frontend/e2e/workflows/watchlists-items.spec.ts \
  apps/extension/tests/e2e/watchlists.spec.ts \
  apps/packages/ui/src/components/Option/Watchlists/JobsTab/__tests__/JobsTab.advanced-details.test.tsx \
  apps/packages/ui/src/components/Option/Watchlists/SourcesTab/__tests__/SourceSeenDrawer.test.tsx
git commit -m "feat: improve watchlists power-user throughput"
```

## Task 12: Final Verification And Release Gate

**Files:**
- Modify: `Docs/Runbooks/watchlists_demo_readiness_2026_05_20.md`
- Modify: `backlog/tasks/task-441 - Create-implementation-plan-for-Watchlists-demo-remediation-tracks.md`

- [ ] **Step 1: Run full frontend watchlists gates**

Run:

```bash
cd apps/packages/ui
bun run test:watchlists:typecheck
bun run test:watchlists:scale
bun run test:watchlists:a11y
```

Expected: PASS.

- [ ] **Step 2: Run full backend watchlists gate**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Watchlists -q
```

Expected: PASS. Existing expected xfail/xpass behavior must be explained if present.

- [ ] **Step 3: Run WebUI browser smoke**

Start backend and WebUI in the same-origin mode chosen in the runbook, then run:

```bash
cd apps/tldw-frontend
npx playwright test e2e/workflows/watchlists-demo-readiness.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 4: Run extension strict watchlists gate**

Run:

```bash
cd apps/extension
npx playwright test tests/e2e/watchlists.spec.ts --reporter=line
```

Expected: PASS with no skipped watchlists tests. If sandbox blocks Chrome launch, rerun with the already approved escalated Playwright path and record that in the runbook.

- [ ] **Step 5: Run Bandit on touched Python scope**

Run:

```bash
source .venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/api/v1/endpoints/watchlists.py \
  tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py \
  tldw_Server_API/app/core/Watchlists \
  -f json -o /tmp/bandit_watchlists_remediation.json
```

Expected: no new findings in touched code.

- [ ] **Step 6: Do manual live demo dry run**

Using the runbook, verify:

- source-test preflight passes for demo source and fallback source
- guided path creates source and monitor
- demonstrated cadence works, including every 5 hours or weekly if shown
- run ingests real items
- digest report appears
- audio task status appears truthfully
- final playable audio is claimed only if artifact exists
- Activity and Reports agree
- active source failure blocks `System healthy`
- extension claim matches what was tested

- [ ] **Step 7: Update task final summary**

Update `TASK-441` with:

- plan path
- review status
- verification commands run for plan-only work
- note that Bandit is not applicable to this plan-only task unless code tasks have also been executed

- [ ] **Step 8: Commit final docs/task update**

Run:

```bash
git add \
  Docs/Runbooks/watchlists_demo_readiness_2026_05_20.md \
  'backlog/tasks/task-441 - Create-implementation-plan-for-Watchlists-demo-remediation-tracks.md'
git commit -m "docs: finalize watchlists remediation verification gate"
```

## Execution Notes

- Do not remove existing Watchlists tabs, OPML workflows, raw cron, raw JSON/settings, templates, batch item review, source seen controls, diagnostics exports, or command palette paths.
- Keep the MVP digest/audio workflow inside `/watchlists`; advanced editing handoffs may exist, but users must not have to leave `/watchlists` to create script, per-speaker audio, and final podcast-style output once Task 9 is complete.
- Treat 3-person podcast as one example. The product contract is optional 1-4 speaker audio briefing/recording.
- Do not claim final audio playback unless a playable or downloadable artifact is verified. A queued Scheduler task is not a finished podcast.
- Do not hide source fetch failures behind `System healthy`.
- Prefer extending existing API fields and metadata over adding new endpoint families. Add endpoints only when existing operations cannot safely express stage-specific retry or diagnostics.
- For PRs materially authored by AI, preserve the repo's human-owned `Change summary` requirement; do not fabricate that text.
