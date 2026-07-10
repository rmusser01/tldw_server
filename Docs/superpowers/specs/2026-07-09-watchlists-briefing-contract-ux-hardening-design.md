# Watchlists Briefing Contract and UX Hardening Design

Status: Confirmed for implementation planning
Date: 2026-07-09
Backlog: TASK-12105
Register: Product
Scope: Shared Watchlists UI used by the WebUI and browser extension, directly connected Watchlists APIs, scheduled pipeline execution, generated text and audio artifacts, delivery state, and recovery.

## 1. Problem

Watchlists currently exposes several overlapping setup paths that create materially different monitor payloads. The guided pipeline builder can enable scheduled output, while Quick Setup and the broader Watchlist Setup flow can create a schedule with text or audio intent but omit `auto_output.enabled`. The backend then marks collection runs successful before downstream output and audio work has completed, skips output when no new items exist, suppresses some downstream failures, and uses different item limits for text and audio.

The resulting product can say a monitor is ready while its configured scheduled briefing is not guaranteed to be attempted, persisted, or surfaced truthfully. Users must also reconstruct state across Overview, Activity, Reports, Jobs, and setup drawers. Live UAT exposed inconsistent counts and timestamps, missing audio and delivery state, an output download failure under mismatched revisions, repeated accessible names, unlabeled controls, and background transitions that were not reliably announced.

## 2. Goal

Make one user promise reliable and inspectable:

> Turn these sources into this briefing, at this cadence, and deliver it here.

Every setup entry point must produce the same versioned pipeline contract. Every scheduled occurrence must expose a durable fulfillment lifecycle for collection, selection, text, audio, persistence, and delivery. The shared UI must guide setup through Sources, Cadence, Briefing, Delivery, and Test, then keep the latest outcome playable and recoverable without requiring the user to navigate operational tables.

## 3. Users and Context

### First-time news researcher

Knows which publications or sites matter, but should not need to understand cron, output preference nesting, Scheduler tasks, or workflow artifacts. Success means leaving setup with an exact receipt and finding a text and optional audio briefing at the promised time.

### Returning researcher or analyst

Checks the newest briefing quickly, often before a meeting. Success means seeing readiness, pressing Play, understanding delivery state, and recovering a failed stage without rerunning successful work.

### Power user or operator

Needs filters, templates, raw cron, source diagnostics, stage identifiers, and retry controls. Success means the simplified default flow preserves advanced controls and provides accurate provenance rather than hiding operational truth.

The primary scene is a researcher in a normally lit home office checking before their first meeting whether overnight sources became a trustworthy, playable briefing, then returning in dim light if a failed stage needs diagnosis.

## 4. Confirmed Product Decisions

1. A selected text or audio artifact is part of the occurrence contract, not a best-effort side effect.
2. An occurrence is not fulfilled until every selected artifact is persisted.
3. A collection run may succeed while briefing fulfillment remains running, partial, or failed. The UI must not collapse these meanings into one success label.
4. No-new-item occurrences still persist a short `no_material_updates` text briefing and, when selected, an audio equivalent.
5. All qualifying newly collected items are candidates after monitor filters and deduplication. One deterministic backend-configured safety cap applies to both text and audio. The result discloses included and omitted counts.
6. Reports is the required storage destination. Existing supported delivery adapters remain optional. No new delivery integration is introduced by this work.
7. Delivery starts after all selected artifacts are ready. A delivery failure does not delete or hide the artifacts.
8. Retries resume failed stages and must not duplicate reports, audio artifacts, Chatbooks, or external messages.
9. Test exercises the real collection, selection, rendering, persistence, and optional audio path. It does not activate a schedule or contact external recipients by default.
10. Existing advanced fields and unknown output preference keys survive normalization and edits.

## 5. Approaches Considered

### A. Canonical contract and shared fulfillment service, selected

Evolve the existing Watchlists pipeline and artifact systems. Add one versioned normalizer, route all setup builders through it, and centralize output, audio, and delivery orchestration behind a shared fulfillment service. Keep collection-run status and briefing-fulfillment status distinct.

This approach fixes the root cause once, reuses existing storage and workflows, preserves current deep links and advanced controls, and can migrate legacy payloads without a parallel product.

### B. Frontend-only consolidation, rejected

Replace the visible setup UI but keep the current backend execution split. This reduces visible duplication but cannot prevent false success, skipped zero-item output, delivery ordering problems, or duplicate retries.

### C. New Watchlists V2 workflow, rejected

Create a separate set of schemas, routes, jobs, and tables. This provides conceptual cleanliness but duplicates mature ingestion, Scheduler, output, notification, and audio systems. Migration risk and maintenance cost exceed the value.

## 6. Canonical Pipeline Contract

### 6.1 Ownership

The canonical contract is the only supported serialization target for new setup flows. It is represented in frontend types and normalized into backend `output_prefs` without discarding compatible legacy or unknown fields.

The normalized contract contains these logical sections:

- `contract_version`: integer version, initially `1`.
- `sources`: source or group scope plus source-owned fetch and extraction settings.
- `cadence`: manual or scheduled expression, IANA timezone, activation state, and next occurrence projection.
- `selection`: monitor filters, deduplication behavior, deterministic cap, and manual-curation override.
- `briefing.text`: enabled state, type, format, template name and version, title policy, and retention.
- `briefing.audio`: enabled state, target duration, language, provider/model/voice defaults, cast, and fallback policy.
- `delivery.reports`: always enabled.
- `delivery.email` and `delivery.chatbook`: existing adapters with explicit enabled state and validated configuration.
- `test_policy`: delivery disabled by default and a marker that identifies test artifacts.

The frontend may use a task-specific draft shape, but every submit path calls the same `normalizeBriefingPipelineContract` and `toWatchlistJobPayload` functions. SetupWizard, Quick Setup, Pipeline Builder, monitor creation, cloning, and edit flows must not hand-author divergent `output_prefs` objects.

### 6.2 Normalization

Normalization must:

1. Preserve unknown top-level and nested advanced fields.
2. Convert recognized legacy keys into canonical sections.
3. Set scheduled text output intent whenever a scheduled briefing is selected.
4. Couple selected audio to a text briefing and scheduled output occurrence.
5. Never silently enable email, Chatbook, or another external delivery.
6. Reject contradictory required settings with field-specific errors.
7. Return a normalized contract plus compatibility warnings that can be shown in advanced details.

Saved legacy monitors remain runnable. Opening one in the editor displays its normalized meaning, and saving writes the current version while preserving unknown fields.

### 6.3 Exact receipt

The same normalized contract drives review copy and persistence. The receipt includes the next absolute occurrence, timezone abbreviation and IANA timezone, source count, artifact selection, target audio length, Reports storage, and enabled deliveries.

Example:

> Tomorrow at 8:00 AM PT (America/Los_Angeles), collect new items from 3 sources, generate a text report and an 8-minute audio briefing, save both in Reports, and email the briefing to name@example.com.

For daylight-saving boundaries, the receipt adds a concise note when the UTC offset of the following occurrence differs from the current offset.

## 7. Fulfillment Lifecycle

### 7.1 Separate status domains

Collection run status remains responsible for source fetch, extraction, filtering, deduplication, and ingestion. Briefing fulfillment is a durable projection attached to the scheduled or manual occurrence and output artifact.

Fulfillment stages are:

1. `collect`
2. `select`
3. `render_text`
4. `persist_text`
5. `generate_audio`, when selected
6. `persist_audio`, when selected
7. `deliver`, for each enabled adapter

Each stage records `not_started`, `queued`, `running`, `ready`, `failed`, `skipped`, or `cancelled`, plus timestamps, stable diagnostic code, retryability, and artifact identifiers where applicable.

Overall artifact readiness is:

- `running`: a required artifact stage is queued or running.
- `ready`: every selected artifact is persisted.
- `failed`: at least one required artifact stage failed.
- `cancelled`: the occurrence was cancelled before required artifacts were ready.

Delivery state is separate:

- `not_configured`
- `waiting_for_artifacts`
- `delivering`
- `delivered`
- `partially_delivered`
- `failed`

The UI may describe a ready briefing with failed delivery as “Briefing ready, email failed.” It must not call the occurrence fully delivered.

### 7.2 No material updates

When selection produces zero items, the system creates a deterministic text artifact that states:

- no qualifying new material was found;
- when the sources were checked;
- how many sources succeeded, failed, or were deferred;
- when the next run is expected.

If audio is selected, this text becomes the short audio script. It does not require an LLM. TTS capability is still required and failures remain retryable.

### 7.3 Selection cap

Text and audio use the same ordered selection result. Ordering is deterministic and grounded in existing monitor rules, recency, and stable identifiers. The limit is backend-configured rather than hard-coded independently in multiple consumers.

When qualifying items exceed the cap, metadata records `candidate_count`, `included_count`, and `omitted_count`. The report and run detail disclose the omission. Manual curation can replace the automatic selection for an explicit preview or rerun, but is never required for scheduled output.

### 7.4 Idempotency

Each occurrence receives a stable key derived from user, monitor, scheduled fire time or manual invocation ID, and contract version. Each stage derives an idempotency key from occurrence, stage, and artifact role.

Retries must:

- reuse ready artifacts;
- resume only failed or missing stages;
- reuse the occurrence key;
- avoid repeating successful deliveries;
- create a new attempt record without creating a new logical artifact unless the user explicitly chooses Regenerate as a new version.

### 7.5 Failure handling

The fulfillment service must not catch a downstream failure and leave only a debug log. It persists a stable failed state and diagnostic code. User-facing messages name the failed stage, retain completed work, and offer the narrowest safe recovery action.

Provider or queue unavailability discovered during setup appears as a warning or blocker according to whether the selected artifact could run. Runtime changes remain recoverable and cannot be promised away by setup validation.

## 8. Outcome-First Setup Flow

### 8.1 Sources

Users add or select feeds and sites, test them, and see sample results. The default view shows source identity, type, and readiness. Extraction rules, deduplication details, and raw settings remain behind advanced disclosure.

### 8.2 Cadence

Users select manual, interval, daily, weekdays, weekly, or advanced cron and an IANA timezone. The UI shows the next exact occurrence and validates the existing backend minimum interval.

### 8.3 Briefing

Text is selected by default for briefing outcomes. Users choose format/template and may enable audio, target length, voice, and advanced cast settings. Audio readiness validation checks provider/model/voice resolution without claiming future availability.

### 8.4 Delivery

Reports is shown as required and always enabled. Existing email and Chatbook adapters are optional. The UI validates address and adapter configuration, explains that delivery waits for selected artifacts, and never silently falls back to a recipient the user did not review.

### 8.5 Test

The final step shows the natural-language receipt and runs a real test occurrence. External delivery is off by default. Users can separately choose Send test for a reviewed recipient or destination.

Test progress shows collection, selection, text, audio, persistence, and delivery-test state. Activation becomes available after contract validation; a failed runtime test does not destroy the draft and provides stage-specific recovery.

### 8.6 Entry-point consolidation

The primary creation action opens this one flow. Existing Quick Setup, Create Watchlist, Pipeline Builder, empty-state onboarding, and monitor-add entry points either open the canonical flow at the appropriate step or act as advanced editors over the same normalized contract.

No separate abbreviated builder may persist a reduced contract. Duplicate tours, repeated setup cards, and competing primary calls to action are removed. Documentation and command-palette access remain secondary.

## 9. Latest Briefing Surface

The selected watchlist’s Overview begins with one latest-briefing region when an occurrence exists. It is a semantic section rather than a nested card grid.

The surface includes:

- briefing title and exact generation time;
- Play, Pause, Resume, seek, duration, and playback state when audio is ready;
- text readiness and Open report action;
- audio readiness and Regenerate action when failed;
- delivery state and Retry delivery action when failed;
- source/item provenance summary;
- last run and next exact run;
- a link to all Reports;
- Inspect run for detailed diagnostics.

Empty state copy explains when the first briefing will run and offers Test now. Running state presents stage progress without shifting the primary controls. Partial failure retains Play or Open report for ready artifacts.

The surface adapts structurally:

- wide screens use a dominant content column with a compact status/action rail;
- medium screens stack status beneath playback without hiding recovery;
- narrow extension/side-panel layouts use one column, full-width primary actions, and no fixed-width drawers.

## 10. Information Architecture and Visual Direction

The color strategy is Restrained. Existing warm neutrals carry the surface, command blue identifies the primary next action, source teal identifies provenance, and semantic colors communicate status. Light and dark themes retain equivalent contrast and hierarchy.

Linear is the reference for compact hierarchy, GitHub Actions for stage state and retry semantics, and Pocket Casts for legible playback controls. These are behavioral anchors, not styling copies.

The Overview hierarchy is:

1. Latest briefing and its primary action.
2. Next run and current fulfillment state.
3. Source health or blockers that affect the next outcome.
4. Recent activity and all-history navigation.
5. Advanced setup and operational diagnostics.

Cards are used only for independent bounded objects. Status rows and stage progress use semantic lists or timelines. The first viewport does not repeat headings, documentation prompts, tours, mode switches, or orientation copy.

## 11. Language and Terminology

Use these nouns consistently:

- Sources: configured feeds and sites.
- Schedule: when collection occurs.
- Briefing: the user outcome containing selected artifacts.
- Report: persisted text artifact.
- Audio: persisted playable artifact.
- Delivery: movement or notification outside Reports.
- Run: collection execution.
- Occurrence: one scheduled or manual fulfillment attempt.

Copy identifies the object, action, outcome, and recovery:

- “Generating the 8-minute audio briefing.”
- “Text report ready. Audio generation failed.”
- “Briefing ready. Email delivery failed. Retry email.”
- “No qualifying updates were found. A status briefing was saved.”

Avoid “included in briefing” unless the item belongs to the persisted selection. Counts must name their definition, such as New, Unread, Selected, or Included.

## 12. Accessibility and Background State

### 12.1 Accessible names

Every interactive control receives a unique, record-derived accessible name:

- source active switch names its own source;
- item selection and row actions name their own item;
- report, audio, and delivery switches name the setting they control;
- icon buttons name the exact action and object;
- view-mode and health controls expose their purpose and current state.

Tests must render multiple records with different names and assert that accessible names do not repeat incorrectly. This guards the live interpolation defect that unit tests with a single record missed.

### 12.2 Announcements

One polite live region announces non-urgent stage transitions, completion, and updated next-run state. An assertive region is reserved for blocking failures that require immediate action. Announcements are deduplicated and contain a stable object name.

Polling and AbortError cancellation must not emit false failures. A user-triggered run gives immediate acknowledgment, continues reporting background state, and announces completion or failure even if the initiating view changes.

### 12.3 Keyboard and focus

The five setup steps are keyboard navigable with visible focus. Validation moves focus to a concise error summary that links to invalid fields. Drawers restore focus to their trigger. Playback and retry actions remain reachable at 200% zoom and in narrow extension layouts.

## 13. Responsive, Localization, and Edge Cases

The implementation must handle:

- zero, one, typical, and hundreds of sources;
- zero, typical, and capped qualifying items;
- very long source, monitor, article, template, and recipient names;
- RTL layout and CJK text without fixed character assumptions;
- browser extension side-panel widths;
- 200% zoom and reduced motion;
- offline, timeout, server validation, queue unavailable, and provider unavailable states;
- concurrent Run now, scheduled occurrence, retry, and regeneration actions;
- stale polling responses arriving after a newer state;
- DST transitions and timezone changes;
- missing or expired artifact download URLs;
- legacy monitors with unknown output preference fields;
- partial source failure where a briefing can still be generated.

## 14. Security and Privacy

- Do not include secrets, provider credentials, raw filesystem paths, or private recipient data in logs or broad live-region messages.
- Validate recipient and template inputs at API boundaries.
- Preserve per-user and organization artifact ownership on every readiness, download, and retry route.
- Recovery actions must authorize the underlying run, output, audio task, and delivery destination.
- Idempotency keys are opaque externally and scoped to the owning user or organization.

## 15. Compatibility and Migration

The migration is additive:

1. Read current and legacy monitor preferences through the versioned normalizer.
2. Preserve unknown fields during edits.
3. Do not change existing manual-only monitors into scheduled-output monitors without reviewed user intent.
4. Show a compatibility warning when a legacy configuration has audio intent without scheduled text output.
5. Saving through the canonical editor writes version `1` and the normalized output contract.
6. Existing tabs, deep links, command-palette actions, templates, raw cron, and advanced controls remain available.

## 16. Testing Strategy

### Contract tests

- Every setup builder produces the same normalized job payload for equivalent intent.
- Legacy and unknown fields round-trip.
- Scheduled text and audio intent sets required output fields.
- Manual-only configurations do not silently enable scheduled output.
- Receipt text is derived from the normalized contract and exact next occurrence.

### Backend unit and integration tests

- A selected artifact failure persists failed fulfillment rather than silent success.
- Zero-item runs persist the no-material-update text and optional audio request.
- Text and audio consume the same bounded selection.
- Delivery waits for selected artifacts.
- Repeated stage retries are idempotent.
- Regenerate creates a new version only when explicitly requested.
- Legacy preferences normalize without data loss.
- User and organization authorization protects status, artifacts, and retries.

### Frontend component tests

- The setup sequence, validation, receipt, Test safety, and recovery states.
- Latest briefing ready, running, empty, partial, failed, offline, and stale states.
- Multiple-record accessible-name assertions.
- Live-region transition and AbortError suppression assertions.
- Narrow, medium, wide, RTL, long-copy, and 200%-zoom layout behavior.

### Browser acceptance

- Start matched frontend and backend revisions from the implementation worktree.
- Complete the daily news-source scenario in WebUI through CDP.
- Verify scheduled contract persistence, Test behavior, text report, audio readiness/player, delivery state, next run, and recovery.
- Repeat the shared flow in the browser extension build at side-panel width.
- Confirm the generated artifact download endpoints and playback URLs from the same revision.
- Run automated accessibility checks and manually inspect names, focus, announcements, contrast, and zoom.

### Security and quality gates

- Focused and affected frontend/backend suites pass.
- Type checking, linting, build, and extension build pass for the touched scope.
- Bandit passes on touched Python paths.
- Impeccable audit reports no unresolved P0 or P1 findings in the scoped surface.

## 17. Acceptance Criteria

1. Equivalent intent from every setup entry point produces an equivalent versioned pipeline contract.
2. Scheduled text and selected audio are required fulfillment stages, including no-material-update occurrences.
3. A collection run cannot create a false user-facing briefing success when required output work failed or was skipped.
4. Text, audio, and delivery state are durable, independently visible, and narrowly retryable.
5. Repeating a retry does not duplicate reports, audio, Chatbooks, or email.
6. The primary setup flow is Sources, Cadence, Briefing, Delivery, Test and ends with an exact natural-language receipt.
7. Test uses the production contract without activating the schedule or sending external delivery by default.
8. Latest briefing exposes playback, text/audio readiness, delivery, provenance, next run, and recovery on WebUI and extension layouts.
9. Record-specific accessible names, keyboard focus, and background announcements work with multiple live records.
10. Matched-revision CDP UAT completes the daily news briefing use case with text and selected audio artifacts available in Reports.

## 18. Non-Goals

- Building a new podcast studio or audio workflow engine.
- Adding new delivery providers.
- Replacing existing source extraction, deduplication, Scheduler, output templates, Notifications, or TTS provider systems.
- Removing power-user tabs, raw cron, template editing, raw voice maps, or operator diagnostics.
- Redesigning unrelated WebUI routes.
- Claiming external providers can never be unavailable. The guarantee is truthful orchestration, durable state, and recovery, not impossible uptime.

## 19. Implementation References

Use the existing shared patterns and the following Impeccable references during implementation:

- `reference/spatial-design.md`
- `reference/interaction-design.md`
- `reference/accessibility.md`
- `reference/responsive-design.md`
- `reference/ux-writing.md`
- `reference/motion-design.md`
- `reference/distill.md`
- `reference/harden.md`
- `reference/clarify.md`
- `reference/onboard.md`
- `reference/layout.md`
- `reference/audit.md`
- `reference/polish.md`

Visual direction probes were intentionally skipped because this is a refinement of an established product surface with a confirmed visual direction, not a directionally ambiguous new interface.
