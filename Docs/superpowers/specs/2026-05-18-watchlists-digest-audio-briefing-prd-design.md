# Watchlists Digest And Audio Briefing PRD

Status: Draft for implementation planning
Date: 2026-05-18
Backlog: TASK-424
Scope: `/watchlists` WebUI and browser-extension shared UI, plus directly connected watchlists APIs, scheduler/workflow jobs, output artifacts, notifications, and run observability.

## 1. Product Summary

`/watchlists` should become the primary place to create, run, monitor, and recover a news scraping to digest to optional audio briefing workflow. The goal is not to replace existing watchlists behavior. The goal is to add a workflow-first path that sits on top of the existing Feeds, Monitors, Activity, Articles, Reports, Templates, and Settings surfaces.

The target workflow must support:

1. Creating news scraping sources for RSS feeds and websites.
2. Configuring source-level fetch, extraction, and dedupe identity rules.
3. Configuring monitor-level cadence, scope, inclusion/exclusion filters, output format, and delivery.
4. Producing a digest or newsletter as an in-app report and optional email/chatbook delivery.
5. Optionally producing a 1-4 person audio briefing or podcast-style recording inside `/watchlists`.
6. Reviewing script, per-speaker script/audio artifacts, and final combined audio output without requiring users to leave `/watchlists` for the MVP.
7. Letting experienced users and operators reuse setups, batch-edit, inspect failures, retry work, and understand what happened.

## 2. Code-Grounded Current State

These are verified implementation facts that the PRD must build on.

- Route and shared UI: `apps/tldw-frontend/pages/watchlists.tsx` dynamically loads `@/routes/option-watchlists`; `option-watchlists.tsx` renders `WatchlistsPlaygroundPage` and supports deep links for tabs, source/job/run/output IDs.
- Current IA: `WatchlistsPlaygroundPage.tsx` uses a progressive 3-primary-tab layout: Sources/Items/Outputs, with Jobs, Runs, and Templates as secondary inline sections. A full 8-tab layout is available behind "Show all views".
- Sources: backend source schemas already include `settings: dict[str, Any] | None`; create/update persists `settings_json`. Source types are `rss`, `site`, and feature-flagged `forum`.
- Source test behavior: RSS sources call `fetch_rss_feed`; site/forum sources use `settings.scrape_rules` when present, otherwise fall back to top-link discovery (`top_n`, `discover_method`).
- Scrape rules: fetchers already support selector fields such as `entry_selector`, `title_selector`, `summary_selector`, `content_selector`, `author_selector`, `date_selector`, `guid_xpath`, pagination selectors, alternates, and a schema DSL. `validate_selector_rules` already returns selector errors, no-match warnings, non-unique warnings, fragile selector warnings, and optional selector counts.
- Dedupe: per-source seen state already exists in `source_seen_items`, keyed by `(source_id, item_key)`. RSS/site pipeline keys default to `guid`, normalized URL, title, or content hash fallback. APIs exist to inspect and clear per-source seen state.
- Monitors/jobs: `scrape_jobs` already stores `scope_json`, `schedule_expr`, `schedule_timezone`, `output_prefs_json`, and `job_filters_json`.
- Cadence: backend uses APScheduler cron parsing. Minimum interval defaults to 5 minutes through `WATCHLIST_MIN_SCHEDULE_INTERVAL_MINUTES`. Frontend validates the same rule but only exposes hourly, every 6 hours, daily, weekly, and raw cron.
- Filters: monitor filters already support `keyword`, `author`, `date_range`, `regex`, and `all` with `include`, `exclude`, and `flag`; include-only gating exists through `require_include`.
- Outputs: output creation supports templates, Markdown/HTML formats, retention, generated variants, delivery plans, email delivery, and Chatbook delivery.
- Email: `NotificationsService` is used at output creation. If notifications are unavailable, email/chatbook delivery returns a skipped status rather than silently pretending delivery happened.
- Auto-output: the pipeline can generate a run output automatically when `job_output_prefs.auto_output.enabled` is set.
- Audio: backend schemas already include `generate_audio`, `target_audio_minutes`, audio model/voice/speed, language, LLM provider/model, persona pre-summarization fields, background audio settings, and `voice_map`.
- Audio workflow: `audio_briefing_workflow.py` defines a Scheduler workflow with `compose_script`, `clean_script`, `multi_voice_tts`, and single-voice fallback. It is triggered after completed runs or explicit output creation when `generate_audio` is true.
- Audio retrieval: `GET /api/v1/watchlists/runs/{run_id}/audio` returns task status and final audio artifact/download metadata. The frontend service layer does not currently expose this helper.
- Frontend gap: `WatchlistOutputCreate` TypeScript type omits many backend audio fields (`generate_audio`, `target_audio_minutes`, `audio_voice`, `voice_map`, etc.), so some valid backend output payloads are not first-class in the shared UI contract.
- Current source modal gap: `SourceFormModal` only captures name, URL, source type, and tags, and tests only URL/type. It does not expose `settings`, `scrape_rules`, dedupe identity, selector validation, top-link discovery, history, or per-source backoff/seen controls.
- Current audio UI gap: `JobFormModal` exposes a simple audio toggle, voice, speed, duration, test sample, background URI, and raw voice-map JSON. It does not expose speaker count, speaker roles, script review, per-speaker generation, per-speaker audio artifacts, or final mix status.

## 2.1 Design Review Corrections Added Before Implementation Planning

The review found the PRD directionally correct, but several requirements needed tighter implementation boundaries before being converted into an engineering plan.

- Configurable dedupe identity is a required product/API improvement, not a verified current capability. The current system has per-source seen state and a fixed key fallback order; user-editable identity rules need backend work.
- Typed source settings must be additive. The UI must read and write known fields without deleting unknown `settings` keys already used by advanced users or future backend features.
- Scheduled digest/newsletter delivery depends on monitor output preferences. A run only creates recurring reports when `job_output_prefs.auto_output.enabled` or an equivalent backend contract is set; one-off test runs can still use explicit output creation.
- Email delivery state belongs to the output artifact. Setup can validate availability and recipients, but success/failure/skipped status must be attached to the generated output or delivery attempt.
- Multi-speaker audio requires persisted intermediate artifacts. The UI cannot credibly show script review, per-speaker generation, retry, or final mixing unless the workflow persists script, per-speaker clips, final mix, and fallback reason.
- The guided pipeline must be additive. It should launch from an empty-state or primary CTA and reuse existing tabs/components; it must not remove the current full-control watchlists workflow used by news, OSINT, and CTI users.

## 3. Personas

### First-Time News User

Knows the sites they want to track and wants a daily/weekly/every-N-hours digest without learning cron, JSON, template internals, or workflow terminology. They need confidence that the system found real items, will run when expected, and will produce a digest/newsletter and optional audio briefing.

### Power User / News Junkie / OSINT / CTI Researcher

Already understands sources, scraping, filtering, dedupe, saved views, outputs, and templates. They need speed, batch operations, explicit controls, reusable configurations, filtering precision, failure visibility, and no loss of the existing dense workflows.

### Operator / Admin

Owns reliability. They need to answer what ran, what failed, what is queued, whether a schedule is active, whether email/audio was skipped or failed, when to retry, and which source, selector, filter, provider, or delivery channel caused the problem.

## 4. Goals And Non-Goals

### Goals

- Make the end-to-end digest/audio workflow completable inside `/watchlists`.
- Preserve all existing tabs, deep links, batch behaviors, templates, OPML, item review, and OSINT/CTI-style workflows.
- Replace recall-heavy setup with a guided workflow that still reveals advanced controls.
- Surface the existing backend power instead of building parallel source, delivery, or audio systems.
- Make variable cadence recognizable: minutes, every N hours, daily, weekdays, weekly, and custom cron.
- Make source rules and monitor rules separable and understandable.
- Make every generated digest/audio artifact traceable to source, monitor, run, filters, template, and delivery status.

### Non-Goals

- Do not redesign the whole WebUI.
- Do not move watchlists into Chat, TTS, Workflow Studio, or another top-level route for the MVP.
- Do not remove raw cron, raw voice map JSON, templates, OPML, or existing advanced tabs.
- Do not make forums a default source type until the existing feature flag says they are enabled.
- Do not create a separate audio product or podcast studio for MVP; use watchlists plus the existing audio/workflow systems.

## 5. Ownership Model

### Per-Source Ownership

Sources own fetch/extraction/dedupe identity:

- URL and source type.
- RSS normalization and conditional fetch metadata.
- Website extraction rules: selectors, schema DSL, pagination, alternates, discovery settings.
- Dedupe identity configuration: default key order is existing behavior; advanced user-editable rules for GUID, canonical URL, selector-derived ID, title, or content hash fallback require backend/API support before the UI exposes them as more than preview text.
- Per-source seen state, backoff, disabled/deferred status, test results, and selector validation.

### Per-Monitor Ownership

Monitors own workflow intent:

- Source/group/tag scope.
- Cadence and timezone.
- Inclusion/exclusion/flag filters and include-only behavior.
- Output expectations: digest/newsletter/template, auto-output, retention, delivery.
- Optional audio briefing settings: speaker count, cast, target length, voice mapping, script review mode, background audio, final output.
- Run-now behavior, retry policy, concurrency, and run/delivery recovery.

## 6. Target Workflow

### 6.1 First-Time Workflow

1. User opens `/watchlists` and chooses "Create briefing pipeline".
2. Step 1, Sources:
   - Paste one or more RSS/site URLs.
   - The UI detects RSS vs website where possible.
   - For RSS, run a feed preview.
   - For website, offer "simple discovery" first, then "custom extraction rules" if preview is weak.
   - Show sample items, selector warnings, dedupe identity preview, and "what will count as already seen".
3. Step 2, Monitor:
   - Choose sources/groups/tags.
   - Choose cadence through plain controls: manual, every N minutes, every N hours, daily, weekdays, weekly, or advanced cron.
   - Add optional include/exclude/flag filters with live preview counts.
4. Step 3, Digest:
   - Choose digest/report type: Markdown briefing, HTML newsletter, structured review.
   - Enable email delivery and enter recipients if desired.
   - Preview expected digest output using sample or latest completed run.
5. Step 4, Optional Audio:
   - Choose "No audio", "1 speaker", "2 speakers", "3 speakers", or "4 speakers".
   - Select speaker roles and voices.
   - Generate or preview script inside `/watchlists`.
   - Generate per-speaker script/audio artifacts.
   - Combine voices into final audio.
   - Show fallback behavior if multi-voice generation fails.
6. Step 5, Review and Run:
   - Show plain-language summary: sources, dedupe key, cadence, filters, output, delivery, audio, first run behavior.
   - User can create only, create and test run, or create and schedule.
7. After completion:
   - If a run starts, land on Activity with live logs and next-step status.
   - When output exists, deep-link to Reports with digest preview, delivery status, audio status/player, and retry controls.

### 6.2 Power-User Workflow

1. User opens `/watchlists` in full-view or command-palette mode.
2. User imports or batch-selects sources.
3. User applies source rule templates or edits source settings directly.
4. User batch-tests sources and sees extraction/dedupe failures inline.
5. User creates or clones a monitor from an existing template.
6. User sets cadence, filters, output, delivery, and audio cast from saved presets.
7. User runs a preview, inspects filter/dedupe tallies, and saves.
8. User monitors run queues, retries failed stages, exports evidence, and reuses generated artifacts.

### 6.3 Operator Workflow

1. User opens `/watchlists` overview or Activity.
2. User sees failed/deferred/disabled sources, failed runs, skipped deliveries, pending audio tasks, and stale schedules.
3. User drills into run detail: source fetch, extraction, dedupe, filters, output render, delivery, audio workflow.
4. User retries a failed stage where safe, clears dedupe/backoff when intentional, or disables a bad source.
5. User can export run logs/tallies and inspect scheduler linkage when authorized.

## 7. Requirements

### P0: Contract And Surface Fixes

- R1: Add typed frontend support for all backend watchlist output audio fields, including `generate_audio`, target minutes, voice/model/speed, language, provider/model, persona fields, background options, and `voice_map`.
- R2: Add a typed frontend service for `GET /watchlists/runs/{run_id}/audio`.
- R3: Add source settings types for common scrape rules and discovery/dedupe options while keeping raw advanced JSON escape hatches and preserving unknown `settings` keys on edit.
- R4: Make `SourceFormModal` submit source `settings` instead of dropping them.
- R5: Make forum source UI capability-driven from `/watchlists/settings`; disabled only when `forums_enabled` is false.
- R6: Add recognition-friendly variable cadence controls that generate cron and honor the existing backend minimum interval.
- R7: Expose `auto_output.enabled` in job output settings so scheduled monitors can create digest reports automatically; keep explicit output creation for test runs and manual previews.
- R8: Surface email/chatbook delivery availability in setup review and attach skipped/failed/sent statuses to output artifacts after generation.
- R9: Keep existing tabs, aliases, deep links, and "Show all views" behavior intact.

### P1: Guided Pipeline MVP Inside `/watchlists`

- R10: Replace the current partial "Briefing pipeline builder" with a complete create pipeline flow covering sources, monitor, digest/newsletter, optional 1-4 speaker audio, and review.
- R11: Source setup must show fetch/extraction test results, selector validation warnings, dedupe identity preview, and whether a source will use RSS, scrape rules, or top-link discovery. If selector validation is not externally reachable today, add a route or expand source test responses to return those diagnostics.
- R12: Monitor setup must show filter impact preview, include-only behavior, cadence summary, next run time, and delivery/audio expectations.
- R13: Digest setup must preview rendered Markdown/HTML where run context exists, and show a credible sample otherwise.
- R14: Email setup must validate recipients before save and explain fallback-to-user-email behavior only when relevant.
- R15: Audio setup must support 1-4 speakers through structured controls, not raw JSON only.
- R16: Audio MVP must produce visible script, per-speaker script/audio status, final mixed output status, and playable final audio inside `/watchlists`; this requires workflow/output artifact persistence, not just polling final audio status.
- R17: On run completion, Reports must show digest output, delivery status, audio task status, final audio player, and retry entry points.
- R18: Activity must show when a run has an output, pending audio task, skipped audio, failed audio, or final audio artifact.

### P2: Power-User And Reuse

- R19: Add clone monitor, clone source rules, saved pipeline presets, saved audio cast presets, and reusable delivery presets.
- R20: Add batch operations for monitors and outputs, not only sources/items.
- R21: Add batch source rule testing and validation for selected feeds/sites.
- R22: Add saved filter sets and template/output presets for repeated daily workflows.
- R23: Add bulk retry for failed runs/deliveries/audio tasks with clear limits and progress.
- R24: Add compact observability columns: next run, last run, new/duplicate/filtered counts, output status, delivery status, audio status.
- R25: Preserve raw JSON/cron/power-user controls behind advanced disclosure.

### P3: Operator/Admin Hardening

- R26: Add operator-oriented failure dashboard sections inside `/watchlists`: source health, scheduler health, delivery health, audio workflow health.
- R27: Expose scheduler/workflow IDs only for authorized users and label them as diagnostic details.
- R28: Add safe source dedupe/backoff reset controls with confirmation and audit-friendly result summaries.
- R29: Add downloadable run diagnostic bundle: run metadata, source statuses, filter tallies, output metadata, delivery statuses, audio task metadata, and logs.
- R30: Add delivery retry and audio retry semantics that do not duplicate source ingestion unless the user explicitly reruns ingestion.
- R31: Add capability-driven forum setup when the backend feature flag is enabled.

## 8. Phased Delivery Plan

### Phase 0: Contract Alignment And Code Truth

Goal: Remove the current frontend/backend mismatch without changing the full user journey yet.

Deliverables:

- TypeScript contract updates for output audio fields and source settings, including a safe merge path that preserves unknown advanced `settings` keys.
- `getWatchlistRunAudio(runId)` service helper.
- Source form pass-through for `settings`.
- Selector/source-test response contract that returns validation diagnostics, or a new validation endpoint if the existing test route cannot return them cleanly.
- Forum UI reads `/watchlists/settings`.
- Schedule picker supports every N minutes/hours/days/weeks and custom cron.
- Job output prefs UI exposes `auto_output.enabled`.
- Reports and Activity can show pending/final audio status from existing APIs.
- Structured audio cast payload shape for 1-4 speakers, with raw `voice_map` retained as an advanced escape hatch.

Success criteria:

- Existing watchlists tests still pass.
- Users can configure every 5 hours without writing cron.
- Frontend can create an output with `generate_audio: true` using typed fields.
- Editing typed source settings does not delete unknown settings keys.
- A run with `audio_briefing_task_id` displays pending/final audio state.

### Phase 1: End-To-End Guided MVP

Goal: A first-time user can complete the core workflow inside `/watchlists`.

Phase 1 should be built in dependency order. Do not start by building the full wizard shell if the output/audio contracts cannot yet support the promises shown in the UI.

#### Phase 1A: Guided Source And Monitor Setup

- Additive "Create pipeline" entry point from empty state and current overview.
- Source preview panel with extraction warnings, dedupe key preview, source settings preservation, and fallback explanation.
- Monitor preview panel with filter and include-only impact, cadence summary, next run time, and explicit create/run/schedule options.

#### Phase 1B: Digest And Newsletter Output

- Digest/newsletter setup with template selection, `auto_output.enabled` for scheduled monitors, and explicit output creation for one-off test runs.
- Email setup with recipient validation, availability status, and clear post-run delivery status.
- Reports view showing digest content, source/filter/output provenance, and delivery status.

#### Phase 1C: Optional Audio Briefing

- Optional 1-4 speaker audio setup with structured speaker controls.
- Script review and per-speaker generation state inside `/watchlists`.
- Workflow/output metadata that persists script, per-speaker artifacts, final audio, and fallback reason.
- Reports view showing script, per-speaker artifacts, final audio player, and audio retry entry points.

Shared recovery states:

- Recovery states for source test failure, empty preview, invalid schedule, invalid email, output render failure, skipped email, pending audio, failed audio, and fallback single-voice audio.

Success criteria:

- First-time user can create a source-backed monitor, run it, and review a digest without leaving `/watchlists`.
- If audio is enabled, the user can review the script, inspect per-speaker generation state, and play the final audio without leaving `/watchlists`.
- The UI tells the user what will happen next before save.
- Failures identify source, monitor, delivery, output, or audio stage.

### Phase 2: Power-User Throughput

Goal: Make repeated daily use fast and controllable.

Deliverables:

- Clone monitor/source rule/audio cast/delivery preset.
- Saved source rule templates and saved monitor templates.
- Batch source validation.
- Batch monitor activation, schedule change, delivery change, and retry.
- Advanced table columns for new/duplicate/filtered/output/delivery/audio status.
- Saved views for common OSINT/CTI workflows.
- Keyboard/command palette actions for create, clone, run, preview, retry, export.

Success criteria:

- Power users can create a new related pipeline from an existing one in under two minutes.
- Batch operations have progress and partial-failure recovery.
- Existing dense review workflows remain available.

### Phase 3: Operator And Admin Reliability

Goal: Make production-like operation explainable and recoverable.

Deliverables:

- Operator dashboard inside `/watchlists`.
- Diagnostic run bundle export.
- Delivery and audio retry controls.
- Source dedupe/backoff reset with confirmation and audit text.
- Capability-driven forum source setup.
- Scheduler/workflow diagnostic details for authorized users.

Success criteria:

- Operators can identify and recover common failures without shell access.
- Retrying delivery/audio does not accidentally rerun scraping.
- Forum setup appears only when backend capability is enabled.

## 9. Issue-To-Requirement Mapping

| Severity | User Type | Issue | Evidence | Requirement |
| --- | --- | --- | --- | --- |
| P0 | Both | Core audio output fields exist in backend but not in frontend `WatchlistOutputCreate` type. | Backend schema has `generate_audio` and audio fields; frontend output create type does not. | R1 |
| P0 | Both | Run audio artifact endpoint exists but shared UI service does not expose it. | Backend `GET /runs/{run_id}/audio`; frontend search only finds `/api/v1/audio/speech`. | R2, R17, R18 |
| P0 | First-time | Source settings exist but source modal drops them. | Backend source create/update persists `settings_json`; `SourceFormModal` onSubmit only includes name/url/type/tags. | R3, R4, R11 |
| P0 | Power user | Typed source settings could accidentally erase advanced raw settings if implemented as replacement JSON. | Source `settings_json` is arbitrary today; advanced users may already rely on keys the first typed UI does not understand. | R3, R4, R25 |
| P0 | First-time | Arbitrary cadence is possible but hidden behind cron. | Backend cron/min interval support; frontend presets only hourly/every6/daily/weekly. | R6 |
| P0 | Both | Scheduled digest auto-output exists but is not first-class in current setup. | Pipeline reads `job_output_prefs.auto_output.enabled`; current builder mostly creates output after run or sets template defaults. | R7, R13 |
| P0 | Both | Email delivery exists but availability/skipped states are not setup-visible enough. | `NotificationsService` may return `skipped: notifications_unavailable`; output preview shows statuses only after output. | R8, R14, R17 |
| P1 | First-time | Website scraping setup lacks extraction-rule guidance and validation. | `validate_selector_rules` exists; Source UI does not expose it. | R11 |
| P1 | Both | User-editable dedupe identity is desired but not verified as configurable in the current pipeline. | Current code has per-source seen state and fixed fallback key derivation; the UI cannot honestly offer custom identity rules until API support exists. | R11, R28 |
| P1 | Both | Audio UI is too low-level and does not support user mental model of 1-4 speakers. | Current UI has voice/speed/duration plus raw voice-map JSON; backend workflow composes multi-voice script. | R15, R16 |
| P1 | Operator | Audio workflow status is disconnected from watchlists run recovery. | Run stats store `audio_briefing_task_id`; endpoint scans Workflow artifacts. UI does not expose full lifecycle. | R17, R18, R30 |
| P1 | Power user | Source dedupe seen state exists but is not usable as setup/recovery control. | Per-source seen APIs exist; source UI does not expose dedupe identity/reset as part of setup. | R11, R28 |
| P2 | Power user | Existing batch operations are uneven. | Sources/items have batch controls; monitors/outputs have weaker batch/reuse. | R19-R24 |
| P2 | Both | Progressive IA hides some direct controls for users who know the system. | Full layout exists behind "Show all views"; PRD must preserve and improve expert access. | R9, R25 |

## 10. UX And Interaction Requirements

- Empty state must offer two paths: "Create briefing pipeline" and "Open full controls".
- Every setup step must show what data will be created or changed.
- Preview states must distinguish: not tested, loading, found items, no items, partial warning, failed.
- Schedule UI must always show cron output and next run time after selection.
- Source testing must label which fetch mode was used: RSS, scrape rules, discovery fallback, or forum.
- Filter preview must show ingestable, filtered, flagged, include-only gated, and sample matched items.
- Output preview must distinguish sample preview from real run output.
- Email delivery must show recipient count, validation state, and delivery availability.
- Audio must use a cast metaphor: speaker role, voice, style/persona, script section, generated audio, final mix.
- Advanced controls must remain available but not required for the first successful pipeline.
- Accessibility: wizard steps, drawers, audio players, validation errors, and live run status must have keyboard and screen-reader support. Existing focus restoration patterns should be reused.

## 11. API, Backend, And Scheduler Recommendations

### Frontend-Owned

- Update `apps/packages/ui/src/types/watchlists.ts`.
- Update `apps/packages/ui/src/services/watchlists.ts`.
- Extend `SourceFormModal`, `SourcesTab`, `JobFormModal`, `SchedulePicker`, `OverviewTab`, `RunsTab`, `OutputsTab`, and `OutputPreviewDrawer`.
- Add a source-rule editor with simple fields first, raw JSON fallback, and read-modify-write preservation of unknown settings keys.
- Add a speaker/cast editor that writes structured output prefs and can derive `voice_map`.
- Keep "Create pipeline" as an additive guided path; do not remove or simplify the current full-tab workflow to ship the wizard.

### Backend/API-Owned

- Add explicit Pydantic schemas for common `source.settings` and scrape-rule validation responses while preserving arbitrary advanced keys.
- Expose selector validation through a route if not already externally reachable.
- Add or document API support for configurable source dedupe identity before exposing editable dedupe-key controls beyond preview/reset.
- Add a structured `audio_cast` or equivalent speaker schema while preserving `voice_map` for advanced/manual use.
- Add first-class audio status shape to run detail responses so the frontend does not need to stitch run stats, workflow scans, and output artifacts manually.
- Add safe retry endpoints for delivery/audio-only retry if existing workflow/task APIs are too generic for `/watchlists`.
- Add output metadata for script, per-speaker artifacts, final mix, and fallback reason.

### Scheduler/Workflow-Owned

- Keep using the existing Watchlists/Scheduler/Workflow path for audio; do not create a separate podcast job system.
- Extend `audio_briefing_workflow` inputs to accept structured speaker config, script review mode, and per-speaker artifact naming.
- Persist script and per-speaker artifacts as workflow/watchlist output artifacts.
- Ensure fallback single-voice audio is visible as fallback, not as full multi-speaker success.

### Notifications-Owned

- Surface whether email delivery is configured before users save a recurring newsletter.
- Preserve skipped/partial/failed delivery status details in output metadata.
- Add retry semantics for delivery without regenerating output content unless explicitly requested.

## 12. Measurement

- First-time source setup success rate.
- Time from empty `/watchlists` to first successful run.
- Time from first successful run to first digest output.
- Time from first digest output to playable audio output when audio is enabled.
- Percentage of users who can configure non-daily cadence without raw cron.
- Source preview failure rate by source type.
- Selector validation warning rate and fix-through rate.
- Delivery success/skipped/failed rate.
- Audio compose/TTS/fallback/failure rate.
- Power-user batch operation success and partial failure recovery rate.

## 13. Testing Requirements

- Unit tests for schedule preset generation, every-N cadence validation, source settings serialization, source rule validation UI helpers, and audio cast payload generation.
- Frontend component tests for source modal settings, unknown settings preservation, pipeline wizard steps, schedule picker, output delivery statuses, run audio status, and audio preview.
- Backend tests for source settings validation, dedupe identity configuration once added, run audio status, output delivery metadata, `auto_output.enabled`, and audio workflow metadata.
- Workflow tests proving script, per-speaker audio, final mix, and fallback reason are persisted when audio is enabled.
- E2E tests for empty-state first-time pipeline creation, digest generation, scheduled auto-output, email configuration validation, optional 1-speaker audio, optional 3-speaker audio, and power-user full-layout clone/run/preview flow.
- Regression tests to ensure existing tabs, deep links, OPML import/export, templates, item review, and command palette behavior still work.

## 14. Real Risks And Resolved Non-Questions

Resolved by code inspection:

- Cadence is not an open backend question. Cron plus minimum interval validation already exists.
- Email delivery is not an open backend question. It exists through `NotificationsService`, with skipped status when unavailable.
- Audio briefing is not an open backend question. There is an existing Scheduler workflow with compose, clean, multi-voice TTS, and fallback.
- Source settings are not an open storage question. Sources already persist arbitrary settings JSON.
- Source dedupe ownership is not an open DB question. Dedupe seen state is per source.
- Filters are not an open rule-engine question. The existing monitor filter engine is the baseline.

Real risks:

- Existing source settings are untyped and could become another raw JSON trap unless the PRD creates a typed simple path without deleting unknown advanced keys.
- Configurable dedupe identity is not proven as an existing backend setting; exposing it as editable UI before API support would be misleading.
- The existing audio workflow may not currently persist all intermediate artifacts needed for script review and per-speaker recovery; that requires workflow/output metadata work.
- Email availability may depend on AuthNZ email configuration; setup must expose skipped/unavailable honestly, and recurring email requires generated outputs from auto-output or equivalent scheduler support.
- Adding guided workflow must not hide existing OSINT/CTI workflows or slow expert paths.
- The UI must not imply scheduled outputs will be created unless `auto_output.enabled` or an equivalent backend path is actually set.

## 15. Implementation Priority

1. Phase 0 contract alignment.
2. Variable cadence UI and source settings pass-through with unknown-key preservation.
3. Source validation diagnostics and honest dedupe identity preview.
4. Auto-output and delivery state surfacing.
5. Run audio service/status in Activity and Reports.
6. Guided source/monitor/digest flow.
7. Structured 1-4 speaker audio controls and persisted script/per-speaker/final artifacts.
8. Power-user reuse and batch operations.
9. Operator diagnostics and retry controls.
10. Feature-flagged forum source setup.

## 16. Acceptance Criteria

- A first-time user can create a source-backed monitor, choose a variable cadence, configure digest/newsletter output, enable email, enable optional 1-4 speaker audio, run once, preview the digest, and play the final audio inside `/watchlists`.
- Scheduled digest/newsletter creation uses monitor output preferences and does not rely on a manual post-run output action.
- Optional audio shows script, per-speaker generation state/artifacts, final mix, and fallback reason when applicable.
- A power user can keep using the full tabbed interface, raw cron, templates, source imports, batch source operations, item review, and advanced controls.
- An operator can identify source, filter, output, email, and audio failures from `/watchlists` and retry or recover the correct stage.
- Existing watchlists APIs and UI flows are extended, not replaced.
- The implementation plan can split work by frontend contract, source setup, monitor/output setup, audio workflow/artifacts, delivery/notifications, and observability.
