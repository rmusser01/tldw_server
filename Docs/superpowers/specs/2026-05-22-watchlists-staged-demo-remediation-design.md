# Watchlists Staged Demo Remediation Addendum

## Status

Approved direction: use the existing Watchlists PRD and implementation plan as the parent product source of truth, then apply this document as a narrow demo-remediation addendum.

Backlog task: `TASK-476`.

## Relationship To Existing PRDs

This document is not a replacement PRD.

Parent product source of truth:

- `Docs/superpowers/specs/2026-05-18-watchlists-digest-audio-briefing-prd-design.md`
- `Docs/superpowers/plans/2026-05-18-watchlists-digest-audio-briefing-implementation-plan.md`

Existing rescue/hardening layer:

- `Docs/superpowers/specs/2026-05-20-watchlists-demo-remediation-staged-plans-design.md`
- `Docs/Runbooks/watchlists_demo_readiness_2026_05_20.md`

This addendum only updates the near-term remediation plan with the latest verified post-merge/demo dry-run blockers:

- `workflow_run` tasks can be submitted to the `workflows` queue without a guaranteed worker.
- `/api/v1/watchlists/runs/{run_id}/audio` can return `no_workflow_db` even when an audio Scheduler task exists.
- Reports can show stale audio status because output preview does not poll the run-audio endpoint.
- The UI can default to an inactive/imported watchlist rather than the active watchlist with demo data.
- Audio defaults can imply Kokoro is usable when the local model is not installed.

When implementation planning resumes, the 2026-05-18 PRD remains the product baseline. This addendum supplies the P0 correction set and acceptance criteria that should be applied before continuing broader durable-audio, first-time workflow, and power-user work.

## Scope

This design covers `/watchlists` and directly connected flows required to demonstrate:

- News scraping watchlists with per-source fetch, extraction, and dedupe identity rules.
- Per-monitor inclusion/exclusion filters, cadence, output, and delivery settings.
- Digest/newsletter output from scraped items.
- Optional 1-4 speaker audio briefing/podcast generation from the digest, including script, per-speaker audio, and final mixed output.
- Repeat use by news-heavy, OSINT/CTI, and power-user personas.

The MVP must keep the core workflow inside `/watchlists`. Advanced handoffs can exist for deeper workflow editing, but creating the script, per-speaker audio, and final audio output must not require leaving `/watchlists`.

This design does not remove existing news-junkie or OSINT/CTI researcher workflows.

## Current Evidence

The current `origin/dev` implementation already contains much of the Watchlists product surface, but the live audio path is not demo-safe.

Relevant implementation seams:

- `tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py`
  - `trigger_audio_briefing()` submits a Scheduler task named `workflow_run` to queue `workflows`.
  - `_build_workflow_inputs()` supports `audio_cast`, `voice_map`, `target_audio_minutes`, TTS model, voice, speed, background audio, and persona summarization.
  - Defaults currently fall back to `kokoro` and `af_heart` when no audio model or voice is configured.
- `tldw_Server_API/app/core/Scheduler/core/worker_pool.py`
  - Worker pool startup only ensures minimum workers on the default queue.
  - Dynamic scaling only iterates queues that already have workers, so a task on a new `workflows` queue can remain queued indefinitely.
- `tldw_Server_API/app/core/Scheduler/scheduler.py`
  - Existing APIs already support `get_task(task_id)` and `scale_workers(target, queue_name)`.
- `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
  - `GET /api/v1/watchlists/runs/{run_id}/audio` reads `audio_briefing_task_id` from run stats.
  - It then tries to locate a per-user `workflows/workflows.db`.
  - If the Workflows DB does not exist, it currently raises `404 no_workflow_db` even though a Scheduler task id exists.
  - If the Workflows DB exists but no matching workflow run exists yet, it returns pending.
  - It already normalizes script, speaker, final audio, fallback, and artifact metadata when artifacts exist.
- `apps/packages/ui/src/components/Option/Watchlists/WatchlistsPlaygroundPage.tsx`
  - `loadWatchlists()` sets selection to `items[0]` when the current selection is invalid.
  - In live demo data, this selected an imported/inactive watchlist instead of the active demo watchlist with runs and outputs.
- `apps/packages/ui/src/components/Option/Watchlists/OutputsTab/OutputPreviewDrawer.tsx`
  - The drawer renders audio artifact metadata already embedded in output metadata.
  - It does not poll `getWatchlistRunAudio(run_id)` for live task/artifact state when audio was requested on a text digest output.
- `apps/packages/ui/src/services/watchlists.ts`
  - The frontend already exposes `getWatchlistRunAudio(runId)`.
- `Docs/Runbooks/watchlists_demo_readiness_2026_05_20.md`
  - Demo readiness requires provider/voice preflight, Scheduler enqueue proof, a meaningful `/runs/{run_id}/audio` response, and final playable audio proof.

Live dry-run findings from the latest demo rehearsal:

- The digest path is viable.
- The audio task was submitted to Scheduler, but stayed queued because no worker served the `workflows` queue.
- `GET /api/v1/watchlists/runs/{run_id}/audio` returned `404 {"detail":"no_workflow_db"}` while the task was queued.
- The UI initially selected an inactive/imported watchlist, making Reports and Updates appear empty.
- Direct TTS worked with KittenTTS; Kokoro failed because the ONNX model was missing in that environment.

## Problem Statement

The system can create watchlists, scrape sources, run monitors, and generate digest output, but it cannot reliably prove the optional audio briefing workflow end to end from `/watchlists`.

The failure mode is especially bad for demos and first-time users because the UI suggests audio can be generated, but the backend can silently leave work queued, the status endpoint can return a misleading 404, and the UI can show the wrong watchlist by default. Power users also lack enough status visibility to diagnose stuck work without leaving the product surface.

## Goals

- Make the P0 demo path truthful and runnable: source setup, monitor run, digest, audio request, live queued/running/failed/completed status, and final artifact if the configured provider succeeds.
- Keep P0 implementation narrow by reusing existing Scheduler, Workflows, Watchlists, and frontend service contracts.
- Preserve variable cadence support, including sub-daily intervals such as every 5 hours and longer schedules such as weekly.
- Preserve source/monitor ownership:
  - Per-source: fetch rules, extraction rules, dedupe identity rules.
  - Per-monitor: inclusion/exclusion filters, cadence, output, delivery, and audio settings.
- Treat 1-4 speaker audio briefing as the product contract; a 3-person podcast is only an example configuration.
- Keep core digest-to-audio creation inside `/watchlists`.
- Add clear follow-up stages for durable artifact persistence, first-time/status UX, and power-user throughput.

## Non-Goals

- No generic WebUI redesign.
- No replacement of Scheduler or Workflows.
- No removal of advanced tabs, OSINT/CTI affordances, templates, filters, or existing Watchlists expert controls.
- No guarantee that every local TTS provider is installed or healthy. The product must detect and explain unavailable providers.
- No requirement that P0 deliver studio-grade audio mixing. P0 must make the actual state visible and allow the configured working provider to complete.

## Approaches Considered

### A. Staged Remediation With P0 Demo Rescue

P0 fixes the verified blockers: `workflows` queue workers, `/runs/{run_id}/audio` status fallback, safer audio configuration behavior, watchlist default selection, and live output preview status. Follow-up PRs harden artifact durability, status UX, and power-user speed.

This is the recommended approach because it fixes the path that failed in the dry run without overbuilding a parallel audio system.

### B. Audio Platform First

Rework workflow artifact persistence, per-speaker outputs, final mixing, and provenance before touching demo UX.

This is more complete long-term, but it is too risky for demo readiness because it delays the known operational blockers.

### C. UI Status First

Patch Reports and copy so users better understand pending audio while leaving Scheduler behavior mostly unchanged.

This would reduce confusion but still allow the demo to fail because the audio task may never run.

## Recommended Architecture

### Current Path

1. A monitor run completes and a digest output is created.
2. If audio is enabled, Watchlists calls `trigger_audio_briefing()`.
3. `trigger_audio_briefing()` gathers ingested run items, builds workflow inputs, and submits `workflow_run` to Scheduler queue `workflows`.
4. The workflow handler should create a workflow run, compose script, generate audio, persist artifacts, and expose artifacts through Workflows.
5. `/api/v1/watchlists/runs/{run_id}/audio` bridges Watchlists runs to workflow artifacts.
6. `/watchlists` Reports should show the digest and the audio status/artifacts.

### Required P0 Contract

The P0 contract is:

- If audio is requested and a Scheduler task id exists, `/runs/{run_id}/audio` must return a meaningful state instead of `no_workflow_db`.
- If the task is queued because no Workflows DB exists yet, the state should be `queued` or `pending` with `task_id`, `queue_name`, and no audio URI.
- If the task is running, failed, cancelled, dead, or completed in Scheduler, that state should be surfaced.
- If workflow artifacts exist, artifact metadata remains the source of truth for script, speaker audio, final audio, fallback reason, and download URL.
- The UI must poll the run audio endpoint when output metadata says audio was requested, even if the visible output artifact is a Markdown or HTML digest.
- Any new fields exposed by the run-audio endpoint must be added to both the backend response schema and frontend TypeScript type so FastAPI response-model filtering and frontend consumers preserve them.

This keeps the bridge inside existing APIs and avoids a second audio job table.

## PR 1: P0 Demo Rescue

### Backend: Ensure the Workflows Queue Can Run

Change `trigger_audio_briefing()` to ensure a `workflows` worker exists before submitting the task:

- Use existing `scheduler.scale_workers(1, "workflows")`.
- Treat a thrown scale failure, or a returned worker count below `1`, as an enqueue-prevention error represented through the structured trigger result contract, not as a generic `None`.
- Do not modify unrelated Scheduler queues.
- Do not introduce a new Jobs system for this path.

Acceptance:

- A unit test injects a scheduler double and verifies `scale_workers(1, "workflows")` is called before `submit()`.
- If scaling fails, `trigger_audio_briefing()` returns a structured `queue_unavailable` or `enqueue_failed` result and logs a sanitized warning.
- If scaling returns `0`, `trigger_audio_briefing()` returns a structured `queue_unavailable` result and does not submit.
- If scaling succeeds, the task is submitted to `queue_name="workflows"`.
- Idempotent resubmission still works: if `scheduler.submit()` returns an existing task id through the existing idempotency key, the caller records that task id normally.

### Backend: Make Audio Status Useful Before Workflows DB Exists

Update `GET /api/v1/watchlists/runs/{run_id}/audio`:

- Keep `404 run_not_found` and `404 no_audio_briefing_for_run`.
- Replace `404 no_workflow_db` with a status response when `audio_briefing_task_id` exists.
- Attempt Scheduler lookup through a helper that does not start a worker pool merely because a user opened a read-only status endpoint. If the only available lookup path would start Scheduler, the implementation must either document and accept that side effect explicitly in the PR or fall back to a safe pending/unknown response.
- Map Scheduler statuses into API statuses such as `queued`, `running`, `completed`, `failed`, `cancelled`, `dead`, or `unknown`.
- Include `task_id`, `queue_name`, `audio_uri: null`, `download_url: null`, and safe `fallback_reason` when relevant.
- Add `queue_name` to `WatchlistRunAudioResponse` and `WatchlistRunAudioStatus`; otherwise FastAPI response-model filtering and frontend typing will silently discard the field.
- If Scheduler lookup fails, return `pending` with `fallback_reason: "workflow_run_not_started"` rather than exposing filesystem paths or stack traces.

Acceptance:

- A unit test covers missing Workflows DB plus queued Scheduler task.
- A unit test covers missing Workflows DB plus Scheduler lookup unavailable.
- Existing artifact lookup tests continue to pass.
- Error responses do not leak local paths, API keys, bearer tokens, or raw exception text.

### Backend: Avoid Broken Audio Defaults

Current fallback to `kokoro`/`af_heart` is unsafe in demo environments where Kokoro models are not installed.

P0 should not silently enqueue a likely-broken model/voice combination. The narrowest acceptable fix is:

- Preserve explicitly configured `audio_model` and `audio_voice`.
- If no model/voice is configured, resolve from a known healthy default only if the app can prove it is available through existing TTS settings/catalog/preflight helpers.
- If availability cannot be proven, mark audio as skipped or configuration-required with a user-safe reason instead of enqueueing work that will predictably fail.
- Replace the current `str | None` trigger result with a small structured result, or another equally explicit contract, so output creation can distinguish `submitted`, `skipped_no_items`, `configuration_required`, `queue_unavailable`, and `enqueue_failed` without parsing logs. This contract applies to all non-submitted paths, including no items, invalid/missing TTS configuration, Scheduler queue scaling failure, and Scheduler submission failure.

Acceptance:

- The output creation path records an actionable audio status and reason when audio is requested without a usable TTS configuration.
- The output creation path records actionable metadata for queue and enqueue failures, such as `audio_briefing_status: "queue_unavailable"` or `"enqueue_failed"` plus a user-safe reason, instead of collapsing them into `"skipped"`.
- Existing explicit model/voice settings still pass through unchanged.
- Tests do not assert Kokoro as the universal default.
- Tests cover the structured no-usable-configuration result and verify output metadata exposes a user-safe `configuration_required` reason.
- Tests cover structured `queue_unavailable` and `enqueue_failed` results and verify they remain distinguishable from `skipped_no_items`.

### Frontend: Prefer a Useful Watchlist Selection

Update `WatchlistsPlaygroundPage.loadWatchlists()` selection fallback:

- Preserve valid current selection.
- Preserve newly created setup/manual watchlists.
- When there is no valid selection, prefer a watchlist that is active and has recent runs, outputs, sources, or jobs if the API response includes enough information.
- If list metadata is insufficient, prefer an active non-imported watchlist before an imported/inactive placeholder.
- Fall back to the first item only when no better signal exists.

Acceptance:

- A frontend test covers an imported/inactive first item and an active demo watchlist later in the list.
- The selected watchlist after setup wizard completion remains the created watchlist.
- No query-param handoff behavior regresses.

### Frontend: Poll Live Audio Status From Reports

Update `OutputPreviewDrawer`:

- When the drawer is open and output metadata has `audio_briefing_requested` or `audio_briefing_task_id`, call `getWatchlistRunAudio(output.run_id)`.
- Poll while status is pending, queued, or running.
- Merge live endpoint data over stale output metadata for the audio artifact panel.
- Show the task id, queue/status, fallback reason, script artifact, speaker artifacts, final artifact, and final download link when present.
- Avoid binary audio download attempts unless the selected output itself is an audio artifact.

Acceptance:

- A frontend test verifies a Markdown digest with requested audio polls `getWatchlistRunAudio()`.
- A queued response renders a queued/pending state instead of "No content available" alone.
- A completed response renders final audio and speaker/script artifact links.
- Polling stops when the drawer closes or status is terminal.

### P0 Verification

Run focused gates:

- Backend watchlists audio tests.
- Backend scheduler/audio workflow unit tests touched by the change.
- Frontend OutputPreviewDrawer tests.
- Frontend Watchlists selection tests.
- A live smoke run:
  - Start backend and WebUI.
  - Create/select an active news watchlist.
  - Add a known reachable source.
  - Create a monitor with variable cadence and digest output.
  - Run it now.
  - Generate digest with audio enabled using a verified working TTS provider.
  - Confirm `/runs/{run_id}/audio` returns queued/running/completed rather than `no_workflow_db`.
  - Confirm Reports show digest and live audio status.
  - Confirm final playable audio only if the provider actually completes.

## PR 2: Durable Audio Artifacts

P0 makes status truthful. PR 2 makes the audio product durable.

Backend requirements:

- Persist a script artifact for the generated briefing.
- Persist per-speaker script segments where the workflow can separate speakers.
- Persist per-speaker audio artifacts for 1-4 speakers.
- Persist a final mixed audio artifact.
- Mark final vs intermediate artifacts explicitly in metadata.
- Include provider, model, voice, speed, language, cast configuration, source digest id, run id, job id, fallback reason, and generation timestamps.
- Preserve partial artifacts if a later stage fails.
- Make failed stages retryable without rerunning successful earlier stages when practical.

API requirements:

- `/runs/{run_id}/audio` should return a stable, typed summary for script, per-speaker artifacts, final artifact, and fallback state.
- Artifact download URLs must route through existing Workflows artifact download endpoints or a documented Watchlists-owned proxy.
- Partial completion must be distinguishable from total failure.

Frontend requirements:

- Reports preview should show a compact artifact graph: digest, script, speaker tracks, final mix.
- Users should be able to regenerate audio without regenerating the digest.
- Users should be able to download script, each speaker track, and final mix.

## PR 3: Status UX and Setup Hardening

PR 3 improves first-time and demo/operator confidence.

Add a Watchlists-local audio readiness/preflight surface:

- TTS provider selected.
- Model selected.
- Voice selected.
- Voice availability.
- Missing local model assets.
- Sample generation result.
- Whether multi-speaker generation is supported by the selected backend.
- What will happen next when the user saves or runs the monitor.

Improve user-visible states:

- `Audio not configured`
- `Queued`
- `Running`
- `Generating script`
- `Generating speaker tracks`
- `Mixing final audio`
- `Completed`
- `Completed with fallback`
- `Failed; retry available`
- `Skipped; no ingested items`

The UI should keep explanations local to the relevant setup/status surface. It should not add generic onboarding copy across the page.

## PR 4: Power-User Hardening

Power users need speed, control, reuse, batch operations, and observability.

Add or harden:

- Output/audio presets reusable across monitors.
- Batch regenerate digest.
- Batch regenerate audio.
- Batch retry failed runs.
- Batch export outputs and artifacts.
- Saved filter presets.
- Copy monitor settings.
- Observability details: queue name, task id, worker queue count, current stage, retry count, failure reason, artifact count, and log link.
- Dense table controls that do not hide advanced state from OSINT/CTI users.

Do not remove existing expert surfaces. Progressive disclosure can make defaults safer, but expert controls must remain reachable.

## Error Handling and Recovery

P0 must distinguish:

- `missing`: no audio was requested for this run.
- `pending`: audio requested, but no workflow run/artifacts exist yet.
- `queued`: Scheduler has the task queued.
- `running`: Scheduler or workflow is processing.
- `failed`: Scheduler/workflow failed.
- `skipped`: no items, missing usable configuration, or user-disabled audio.
- `completed`: final artifact exists.
- `unknown`: lookup failed without a safe specific state.

PR 2 must add a first-class `partial` state for cases where script or speaker artifacts exist but the final mix does not. P0 can surface available partial artifacts if the current endpoint already returns them, but it should not overclaim a durable partial-artifact contract until PR 2 normalizes and tests it.

The frontend should render each state with an action:

- Configure audio.
- Wait/poll.
- Retry.
- Open logs/details.
- Download artifact.
- Regenerate audio.

## Accessibility and UX Constraints

- Status must be visible without opening browser dev tools.
- Status updates should use existing live-region patterns where available.
- The Reports preview drawer must remain keyboard usable and preserve focus restoration.
- Polling must stop on drawer close.
- Audio controls must use native browser audio controls for playback.
- Error copy must not expose secrets or local filesystem paths.
- Dense power-user surfaces are acceptable, but first-time users should not need to decode Scheduler or Workflows terminology to know what happened.

## Data Ownership Boundaries

Per-source:

- Fetch URL and fetch method.
- Extraction rules.
- Normalization and dedupe identity rules.
- Source validation status.

Per-monitor:

- Source membership or source set.
- Cadence, including every-N-hours and weekly schedules.
- Inclusion/exclusion filters.
- Output template and delivery.
- Digest/newsletter settings.
- Audio briefing settings, including speaker count/cast, voices, model, speed, language, background audio, and delivery.

Per-run:

- Scrape status.
- Item tallies.
- Digest output artifacts.
- Audio task id.
- Workflow/audio artifacts.
- Logs and retry state.

## Implementation Notes for Later Planning

The implementation plan should split PR 1 into small commits:

1. Backend audio status fallback tests.
2. Backend audio status fallback implementation.
3. Backend workflow queue scaling tests.
4. Backend workflow queue scaling implementation.
5. Backend TTS configuration/default behavior tests and implementation.
6. Frontend watchlist selection test and implementation.
7. Frontend audio polling test and implementation.
8. Focused verification and live smoke.

PR 2-4 should remain separate unless PR 1 exposes a blocker that must be fixed first.

## Open Risks

- The exact existing TTS availability API may not provide a single synchronous "known healthy default" helper. If so, P0 should prefer configuration-required status over guessing.
- Workflows artifact persistence may already produce some artifact metadata, but PR 2 must verify actual persisted records rather than assuming the shape from endpoint normalization.
- Scheduler task status may be unavailable if the scheduler has not started in the current process. P0 should handle that as a safe pending/unknown status, not a hard error.
- Live smoke may depend on local models or provider credentials. The demo script must name the verified provider and model before claiming final audio support.

## Definition of Done

For this design task:

- The staged remediation design is written and committed.
- The spec review loop finds no blocking issues or all found issues are addressed.
- The user reviews and approves the written spec before implementation planning starts.

For P0 implementation later:

- The dry-run failure modes listed in this design are either fixed or converted into accurate user-visible states.
- The demo can show a digest and truthful audio status from `/watchlists`.
- A final playable audio artifact is claimed only after provider-backed generation completes and is verified.
