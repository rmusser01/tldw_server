# Watchlists Demo Remediation Staged Plans

Status: Draft for review
Date: 2026-05-20
Backlog: TASK-464
Scope: `/watchlists` WebUI and browser-extension shared UI, plus directly connected Watchlists API, Scheduler, output, delivery, and audio briefing flows needed to make the news scraping to digest to optional audio workflow demonstrable and durable.

## 1. Context

Live walkthrough verification of `/watchlists` found that the current implementation is not ready for a full end-to-end demonstration of the promised workflow:

1. Create tracked news sources.
2. Configure monitor cadence, source rules, filters, dedupe, and output expectations.
3. Turn scraped items into a digest/newsletter.
4. Optionally create a 1-4 speaker audio briefing or podcast-style output from that digest.
5. Repeat the workflow as a power user or operator with speed, reuse, batch controls, and observability.

The route and shared UI are wired correctly: the WebUI `/watchlists` page dynamically loads the shared Watchlists route, and the extension route renders the same `WatchlistsPlaygroundPage`. Shared component tests and backend Watchlists tests pass, and the extension watchlists E2E suite can pass when Chrome is allowed to launch outside the sandbox. The live workflow still has blocking product/runtime defects.

This document converts the observed issues into staged remediation plans. It is not a replacement for the existing Watchlists PRD or implementation plan. It is a rescue and hardening layer that prioritizes live demo readiness while preserving the broader product plan.

## 2. Verified Issues To Address

### P0: Pipeline Digest Generation Breaks By Default

The guided pipeline and quick setup send `briefing_md` as the default template name, but the backend exposes `briefing_markdown`. In live browser testing, the pipeline run scraped items but crashed during output creation with `template_not_found (POST /api/v1/watchlists/outputs)`.

Impact:
- First-time users cannot complete the promised digest workflow.
- The page can show a framework runtime overlay instead of a recoverable in-app error.
- The demo path is brittle even when scraping works.

### P0: Audio Briefing Requests Do Not Enqueue

The audio bridge calls `scheduler.enqueue(task)`, but the live Scheduler object exposes `submit(...)`. Live logs showed: `Audio briefing: failed to enqueue workflow for run 2: 'Scheduler' object has no attribute 'enqueue'`. Output metadata recorded `audio_briefing_requested: true` and `audio_briefing_status: skipped`; `/runs/{run_id}/audio` returned `no_audio_briefing_for_run`.

Impact:
- Optional 1-4 speaker audio cannot be truthfully demoed end to end.
- Users see an audio request accepted but no artifact or actionable reason.
- Tests mock the wrong Scheduler method, so the regression is not caught.

### P1: Health And Run Status Can Be Misleading

A source can have `status=error:403` while its run is marked `Succeeded` with zero items and zero errors, and Overview can still report `System healthy`.

Impact:
- Operators cannot tell what failed.
- A demo can appear to succeed while producing no useful content.
- First-time users lose trust because the system does not explain what happened next.

### P1: First-Time Cadence Setup Is Incomplete

The first-time quick setup offers manual, hourly, daily, and weekdays, while the user requirement includes variable intervals such as every 5 hours and weekly. Variable cadence exists in the pipeline/advanced controls, but not consistently in the beginner path.

Impact:
- The first-time workflow does not match the product requirement.
- Users must discover a separate advanced flow to express common schedules.

### P2: Review And Summary States Are Inconsistent

The guided review can show `Total feeds: 0` after entering a feed and can show both disabled and enabled audio language in the same summary.

Impact:
- Users cannot verify what they are about to create.
- The final review step is not credible for a demo or production workflow.

### P2/P3: Power-User And Operator Workflows Need Completion

The current UI has many strong advanced surfaces: full tabs, command palette, Activity, diagnostics, OPML import, batch source/item controls, run detail drawers, and Reports. The gaps are around reliable stage status, reuse, batch monitor/output operations, source validation at scale, delivery/audio retry, and failure drill-down.

Impact:
- News junkies, OSINT users, CTI researchers, and operators retain basic control but lose speed and visibility at the exact points where automation fails.

## 3. Delivery Strategy

Use three parallel tracks:

1. Track A, Demo Rescue: same-day or near-term fixes that make the demonstrated workflow truthful.
2. Track B, Product Workflow Completion: durable first-time and regular-user workflow completion inside `/watchlists`.
3. Track C, Power User And Operator Hardening: throughput, reuse, diagnostics, and recovery for expert and admin workflows.

Track A is allowed to make tactical fixes, but it must not introduce fake success states or parallel systems. Tracks B and C convert the rescue fixes into durable product contracts.

## 4. Track A: Demo Rescue

### A0. Reproduce And Freeze Demo Contract

Goal: create a repeatable live verification contract for the demonstration path.

Scope:
- Document the exact test setup: backend mode, API key, WebUI deployment mode, demo source URLs, and extension route verification.
- Select demo source URLs through preflight, not convenience. Local loopback fixtures are not valid demo sources unless the backend SSRF/loopback policy explicitly allows them; otherwise use a stable external RSS/source fixture and verify it through `/watchlists/sources/test`.
- Decide whether the demo runs in quickstart/same-origin mode or advanced API-base mode. Quickstart/same-origin is the default for the demo until advanced API-base routing is separately verified.
- If final audio playback is part of the demo, preflight the required Scheduler, workflow handler, LLM/script provider, TTS provider, and voice configuration before the demo script claims final audio output.
- Define the expected user path: create source, create monitor, run scrape, generate digest, request audio, inspect Activity and Reports.
- Capture expected success, acceptable degradation, and hard-fail conditions.

Fixes covered:
- Prevents subjective readiness claims.
- Gives devs a stable reproduction loop for tomorrow's demo.

Gate:
- One command list and one browser script/path can reproduce the demo.
- Demo source URLs pass source-test preflight and have a named fallback source.
- The selected WebUI deployment mode is verified before the demo. Advanced API-base mode cannot be used for the demo unless `/watchlists` requests hit the configured API origin rather than falling back to broken relative API paths.
- If final audio playback is in scope, provider and voice readiness is verified separately from Scheduler enqueue readiness.
- Each observed failure has exact expected/actual output.
- Demo owners know which features are safe to show and which must be avoided until fixed.

### A1. Template Contract Hotfix

Goal: stop `/watchlists` from sending non-existent default output template names.

Scope:
- Replace or map UI recipe ID `briefing_md` to backend template name `briefing_markdown` before any output creation request.
- Audit quick setup, pipeline wizard, overview summaries, template recipes, tests, and scale fixtures for places where recipe IDs are being sent as backend template names.
- Add in-app error handling for output creation failures so a bad template cannot trigger a Next runtime overlay.

Fixes covered:
- `template_not_found` during pipeline test generation.
- Broken quick setup digest/audio defaults.
- Unrecoverable browser overlay during demo.

Gate:
- UI-created output requests use `briefing_markdown`.
- Reports show the output created by pipeline test generation.
- Invalid template names render a recoverable in-app error with the failing template name and next step.

### A2. Audio Enqueue Hotfix

Goal: make watchlist audio requests use the actual Scheduler API.

Scope:
- Replace `scheduler.enqueue(task)` usage in the Watchlists audio bridge with `scheduler.submit("workflow_run", payload=..., queue_name="workflows", ...)`, following existing Scheduler usage.
- Update tests that currently mock `scheduler.enqueue`.
- Preserve current workflow definition and output metadata contract.
- Confirm that `generate_audio=true` stores a task id and pending status.

Fixes covered:
- `'Scheduler' object has no attribute 'enqueue'`.
- Audio requests silently becoming `skipped`.
- `/runs/{run_id}/audio` returning `no_audio_briefing_for_run` immediately after a valid audio request.

Gate:
- A live output creation request with `generate_audio=true` records `audio_briefing_status=pending` and a task id.
- Scheduler DB has a queued/running/completed `workflow_run` task.
- `/runs/{run_id}/audio` returns a meaningful pending/running/completed/failed state.

### A3. Demo Truthfulness Fallback

Goal: if a provider, model, voice, scheduler, or workflow stage is unavailable, the UI must explain that state plainly.

Scope:
- Surface skipped/enqueue_failed/audio failed states in Activity and Reports.
- Include the error reason where safe.
- Distinguish "audio unavailable" from "audio generation in progress" and "audio complete".
- Ensure source/output/audio warnings affect visible health summaries.

Fixes covered:
- Audio requested but skipped.
- Source failure hidden behind healthy status.
- Digest/audio demo states that imply success without artifact output.

Gate:
- No skipped audio or failed output appears as healthy/successful.
- Active source fetch errors and zero-item runs with source errors surface as warning or partial-success states.
- `System healthy` is blocked while an active source, recent run, output, delivery, or audio task has an unresolved hard failure.
- Activity and Reports expose enough status to diagnose the failure without reading server logs.
- The demo can truthfully show either completed audio or a clear "audio unavailable because..." state.

## 5. Track B: Product Workflow Completion

### B1. First-Time Pipeline Cleanup

Goal: make beginner setup match the first-time persona and variable-cadence requirement.

Scope:
- Add manual, every N minutes, every N hours, daily, weekdays, weekly, and advanced cron options to beginner setup where appropriate.
- Fix review summary counts and audio copy.
- Keep the existing advanced flow available for power users.

Gate:
- A first-time user can configure every 5 hours and weekly without leaving `/watchlists`.
- Review summary accurately states sources, cadence, output, delivery, audio, and first-run behavior.

### B2. Digest/Newsletter Output Contract

Goal: make scheduled and manual digest outputs predictable.

Scope:
- Clarify when `auto_output.enabled` is set for scheduled monitor output.
- Keep one-off test generation explicit.
- Show where generated reports will appear and whether email/chatbook delivery is enabled.
- Validate output templates before save or before run/test generation.

Gate:
- Monitor review states exactly when reports will be created.
- Reports tab shows generated digest/newsletter artifacts with template, run, item count, delivery state, and warnings.

### B3. Source Validation And Dedupe Explanation

Goal: make source setup explain fetch, extraction, and dedupe before save.

Scope:
- Show sample items, fetch status, source status, selector diagnostics where available, and dedupe identity preview.
- Preserve unknown source `settings` keys when editing.
- Expose advanced source rules without making first-time users write JSON.
- Make failed source status affect Overview and Activity.

Gate:
- Users can tell whether a source will use RSS, scrape rules, or top-link discovery.
- Users can tell what key will mark an item as already seen.
- A failed source is visible without opening developer logs.

### B4. 1-4 Speaker Audio UX Completion

Goal: make optional audio credible inside `/watchlists`.

Scope:
- Keep 1-4 speaker cast controls.
- Persist script, per-speaker script/audio states, final mix state, fallback reason, and final playable/downloadable audio so Reports, Activity, retry, and diagnostics can all reference the same artifacts.
- Avoid requiring users to leave `/watchlists` for MVP creation or review.
- Keep advanced handoffs available for deeper editing later.

Gate:
- A completed audio briefing has visible script, speaker artifacts, final artifact/player, and provenance.
- A failed or fallback audio briefing has a stage-specific reason and retry path.

## 6. Track C: Power User And Operator Hardening

### C0. Existing Workflow Preservation Baseline

Goal: prevent the rescue and guided-workflow changes from regressing existing news, OSINT, CTI, and advanced Watchlists workflows.

Scope:
- Preserve OPML import/export.
- Preserve raw cron entry.
- Preserve raw JSON/settings escape hatches and unknown source `settings` keys.
- Preserve batch item/source controls.
- Preserve source seen/dedupe inspection and reset controls.
- Preserve run diagnostics and CSV/diagnostic exports.
- Preserve template listing/editing, output regeneration, and report preview/download flows.
- Preserve command palette access and full-view tabs.

Gate:
- A golden-path preservation checklist passes before each PR that changes shared Watchlists state, forms, or navigation.
- Existing advanced workflows remain reachable without using the new guided pipeline.
- Any intentionally changed power-user behavior is called out in the PR and has a replacement path.

### C1. Health Model Correction

Goal: make Overview health reflect real source, output, delivery, and audio state.

Scope:
- Include source failures, zero-item runs with source errors, output render failures, delivery failures/skips, and audio failures/skips.
- Provide drill-down links to the affected source, run, output, or audio task.

Gate:
- `System healthy` cannot appear while active sources or recent runs have unresolved errors.
- Health cards distinguish warnings from hard failures.

### C2. Run Detail Observability

Goal: make Activity useful for debugging.

Scope:
- Show stage-level status for fetch, extraction, dedupe, filters, output render, delivery, and audio.
- Show item found/ingested/duplicate/filtered/error counts.
- Show output linkage and audio task linkage.

Gate:
- A user can answer what happened in a run without reading server logs.
- A succeeded run with failed output/audio is clearly marked as partially successful or warning state.

### C3. Recovery Controls

Goal: retry only the failed stage where safe.

Scope:
- Expose retry output, retry delivery, retry audio, rerun ingestion, and clear source backoff/dedupe as separate actions.
- Add confirmations for destructive or duplicate-producing actions.
- Avoid duplicating ingested items unless the user explicitly reruns ingestion.

Gate:
- Operators can recover output, delivery, or audio failures without rerunning source scraping.
- Retry results are visible in Activity and Reports.

### C4. Power-User Throughput

Goal: preserve and improve dense workflows for news, OSINT, and CTI users.

Scope:
- Clone monitor.
- Clone source rules.
- Save pipeline presets.
- Save filter/output/audio/delivery presets.
- Batch source validation.
- Batch monitor/output retry or activation where backend support exists.
- Add command palette entries for create pipeline, clone, validate, run, retry, and export.

Gate:
- Experienced users can reuse a proven setup without recreating every form field.
- Batch operations show impact summaries before commit.
- Existing OPML, raw cron, raw JSON, templates, and full-view tabs remain available.

## 7. Suggested PR Slices

### PR A: Demo Rescue

Includes:
- A0 demo contract.
- A1 template mapping/default fix.
- A2 audio Scheduler submit fix.
- Minimal A3 skipped/failed state visibility.
- A minimal C0 preservation smoke check for existing advanced controls touched by the rescue work.

Why this grouping:
- These issues block the public demo path.
- They are tightly coupled to the live failure evidence.
- They should be tested together against a real backend and browser.

### PR B: First-Time Workflow

Includes:
- B1 first-time cadence and review cleanup.
- B2 digest/newsletter output contract validation.

Why this grouping:
- These are user-facing workflow correctness fixes.
- They stabilize the guided path after the demo blockers are removed.

### PR C: Source Confidence

Includes:
- B3 source validation, dedupe explanation, and health linkage.
- C1 source-health integration where needed.

Why this grouping:
- Source confidence is the foundation for both first-time and operator trust.

### PR D: Audio Product Completion

Includes:
- B4 script/per-speaker/final artifact visibility.
- C2/C3 audio observability and retry controls.

Why this grouping:
- It moves audio from "enqueued task" to "usable product workflow."

### PR E: Operator Recovery

Includes:
- C1 complete health model.
- C2 stage-level run detail.
- C3 retry controls and diagnostic summaries.

Why this grouping:
- Operators need a coherent failure model rather than isolated badges.

### PR F: Power-User Throughput

Includes:
- C0 preservation baseline expansion.
- C4 presets, clone, batch validation, batch retry, and command palette additions.

Why this grouping:
- These are high-value but should not delay correctness and recovery.

## 8. Ownership And Dependencies

### Frontend Shared UI

Owns:
- Pipeline wizard, quick setup, review summaries, error states, Activity/Reports/Overview display, command palette, and browser-extension parity.

Primary paths:
- `apps/packages/ui/src/components/Option/Watchlists`
- `apps/packages/ui/src/services/watchlists.ts`
- `apps/packages/ui/src/types/watchlists.ts`

### Backend Watchlists API

Owns:
- Template validation responses, source test diagnostics, output creation, run status, audio status, retry endpoints, diagnostics, and safe error contracts.

Primary paths:
- `tldw_Server_API/app/api/v1/endpoints/watchlists.py`
- `tldw_Server_API/app/api/v1/schemas/watchlists_schemas.py`

### Scheduler And Workflow

Owns:
- Audio task enqueue, workflow execution, task status, script/per-speaker/final artifact creation, and fallback behavior.

Primary paths:
- `tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py`
- `tldw_Server_API/app/core/Scheduler`
- `tldw_Server_API/app/core/Workflows`

### Watchlists Pipeline And DB

Owns:
- Source run status, item counts, dedupe/seen state, automatic output creation, and persisted run/output metadata.

Primary paths:
- `tldw_Server_API/app/core/Watchlists/pipeline.py`
- `tldw_Server_API/app/core/DB_Management/Watchlists_DB.py`

## 9. Verification Strategy

### Automated Gates

Frontend:
- Watchlists typecheck/static guard.
- Focused Vitest for PipelineWizard, quick setup, schedule picker, output metadata, run detail drawer, and Reports audio display.
- Extension watchlists E2E strict mode with no skips when Chrome can launch.

Backend:
- Watchlists API tests for output creation, template validation, job output preferences, audio enqueue, audio retrieval, retry, and operator diagnostics.
- Scheduler integration test proving `workflow_run` is submitted through `submit(...)`.
- Bandit on touched Python paths.

Browser/live:
- WebUI `/watchlists` live walkthrough in the selected, preflighted deployment mode.
- External or allowlisted demo RSS/source scrape that has passed `/watchlists/sources/test`.
- Digest generation.
- Audio request and status/artifact check. If final playback is claimed, the check must verify a playable/downloadable artifact, not only a queued task.
- Reports and Activity verification.

### Demo Readiness Gate

The demo is green only if:

1. WebUI `/watchlists` loads without API fallback errors in the selected deployment mode.
2. Demo source URLs pass source-test preflight and have a fallback source. Local loopback sources are excluded unless explicitly allowed by backend policy.
3. Creating a source and monitor works from the guided path.
4. Running the monitor ingests real external or allowlisted demo-source items.
5. Running test generation produces a visible digest report.
6. Audio enqueue plumbing is proven: a valid `generate_audio=true` request creates a Scheduler `workflow_run` task id and `/runs/{run_id}/audio` returns visible pending/running/completed/failed status. Provider, voice, or model unavailability may degrade only after the task/status plumbing is proven.
7. Final audio playback is claimed only if provider/voice readiness is preflighted and a playable/downloadable artifact is verified.
8. Reports and Activity agree about output/audio status.
9. Source fetch errors and zero-item runs cannot appear as clean success or `System healthy`.
10. If the demo uses quick setup, quick setup must support the demonstrated cadence, including every 5 hours or weekly when shown, and its review step must show the correct feed count and non-contradictory audio copy. If the demo intentionally uses the pipeline builder instead, quick setup cadence/review cleanup is explicitly excluded from the initial demo script and remains in Track B.
11. Extension readiness is explicit. If the extension is part of the demo, pass an extension-path smoke covering the Watchlists shared route, guided setup or equivalent creation path, output generation error handling, Activity/Reports status, and no runtime overlay. If the extension is not part of the demo, label it mount-only and do not claim extension workflow readiness.
12. Existing advanced workflows touched by the rescue work still pass a preservation smoke: full-view tabs, raw cron, OPML import/export, raw JSON/settings preservation, batch source/item controls, source seen/dedupe controls, template listing/editing, report preview/download, diagnostics/export, and command palette access.

If item 6 is not green, the demo script must omit all audio claims. If item 7 is not green, the demo script may show audio task status but must omit final podcast/playback claims.

## 10. Non-Goals

- Do not remove existing Watchlists tabs, OPML workflows, templates, raw cron, raw JSON, batch item review, source seen controls, or OSINT/CTI workflows.
- Do not move the MVP workflow out of `/watchlists`.
- Do not build a separate podcast studio for MVP.
- Do not mask failures as success for demo optics.
- Do not add broad unrelated WebUI redesign work to this remediation series.

## 11. Open Implementation Notes

- The UI can keep recipe IDs such as `briefing_md` internally, but API payloads must send backend template names such as `briefing_markdown`.
- The audio hotfix should follow the Scheduler public API rather than reaching into backend queue implementations.
- Source failures should not necessarily make ingestion "failed" if some sources succeed, but the run must become warning/partial rather than clean success when active sources fail.
- Live browser verification should use quickstart/same-origin mode unless the advanced deployment API base URL behavior is separately fixed and verified.
