# Quick Ingest UX Remediation Stages Design

Date: 2026-05-16
Status: Approved for planning
Owner: Codex brainstorming session
Backlog: TASK-392

## Summary

This spec turns the quick-ingest WebUI and browser-extension UX audit into a
risk-first staged remediation plan. The work stays scoped to the quick-ingest
launch path, wizard flow, processing and recovery states, results handoff,
shared WebUI/extension behavior, and verification coverage.

The active product path is the shared quick-ingest wizard, not the older tabbed
quick-ingest modal. The staged plan should therefore focus on the wizard and its
shared services first, while explicitly identifying any legacy modal or stale
test expectations before implementation starts.

## Goals

- Make quick ingest understandable for first-time users without slowing down
  returning users.
- Make terminal results actionable, especially the handoff to Media, Knowledge,
  Workspace, and Chat where supported.
- Make offline, failed, cancelled, minimized, and interrupted states honest and
  recoverable.
- Reduce preventable URL and file-input failures.
- Keep WebUI and browser-extension behavior aligned through shared UI and
  shared quick-ingest services.
- Replace unresolved audit notes with concrete verification gaps to close.

## Non-Goals

- Redesign the broader WebUI shell, Media page, Knowledge page, Chat page, or
  backend architecture.
- Change backend ingest APIs unless a stage proves it is required for the
  quick-ingest user experience.
- Rebuild the browser extension as a separate product surface.
- Implement the remediation in this design phase.
- Preserve stale legacy-modal behavior if the wizard is confirmed as the only
  active quick-ingest path.

## Current State Evidence

### Active shared wizard

The active quick-ingest flow is the shared wizard modal:

- [QuickIngestWizardModal.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/QuickIngestWizardModal.tsx)
- [AddContentStep.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/QuickIngest/AddContentStep.tsx)
- [WizardConfigureStep.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/QuickIngest/WizardConfigureStep.tsx)
- [ReviewStep.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/QuickIngest/ReviewStep.tsx)
- [ProcessingStep.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/QuickIngest/ProcessingStep.tsx)
- [WizardResultsStep.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/QuickIngest/WizardResultsStep.tsx)
- [FloatingProgressWidget.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/QuickIngest/FloatingProgressWidget.tsx)

### Launch and runtime paths

Quick ingest is launched from shared WebUI and extension surfaces:

- [QuickIngestButton.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Layouts/QuickIngestButton.tsx)
- [WebLayout.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/components/layout/WebLayout.tsx)
- [sidepanel-chat.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/extension/routes/sidepanel-chat.tsx)
- [form.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Sidepanel/Chat/form.tsx)
- [background.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/entries/background.ts)
- [quick-ingest-batch.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/services/tldw/quick-ingest-batch.ts)
- [quick-ingest-session-reattach.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/services/tldw/quick-ingest-session-reattach.ts)

### Legacy and potentially stale surfaces

The older quick-ingest modal and result-panel code still exist:

- [QuickIngestModal.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/QuickIngestModal.tsx)
- [ResultsPanel.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/QuickIngest/ResultsPanel.tsx)

Some e2e selectors and expectations still reference legacy concepts such as
`quick-ingest-run`, `quick-ingest-cancel`, tabbed results, and primary Media CTA
test ids. Stage 1 must decide whether those tests should migrate to the wizard
or remain tied to a reachable legacy path.

## UX Principles

- Prefer recognition over recall: users should not need to know where ingest
  results go before using the modal.
- Prefer truthful system status: failed, cancelled, interrupted, and completed
  states must be visually and semantically distinct.
- Preserve power-user speed: the quick path should remain one paste plus one
  processing action.
- Keep shared behavior shared: WebUI and extension should not fork UX logic
  unless the extension runtime truly requires a different path.
- Fix recovery paths before adding new capabilities.

## Stage 1: Foundation And Evidence Alignment

**Goal:** establish the active quick-ingest implementation path and prevent
future remediation work from fixing stale or unreachable surfaces.

**Findings addressed:**

- Stale or legacy test expectations.
- Unclear active surface between wizard and older tabbed modal.
- WebUI/extension parity risk.

**User outcome:** implementation work starts on the flow users actually see.

**Scope:**

- Confirm all active launchers route to `QuickIngestWizardModal`.
- Identify whether `QuickIngestModal.tsx`, `ResultsPanel.tsx`, and legacy
  tabbed quick-ingest tests are still reachable.
- Build a stage-to-file ownership map for the wizard, services, and e2e helpers.
- Rename the audit's prior unresolved verification notes into
  `Verification Gaps To Close`.

**Required planning artifacts:**

- Active-path map from each launcher to modal, runtime, transport, and result
  surface.
- Legacy reachability decision for the older modal and result panel.
- Test classification table: current wizard coverage, legacy coverage, stale
  selector coverage, and missing coverage.

**Implementation notes for planning:**

- Treat legacy code as either "active and needs parity" or "deprecated and
  should not drive new expectations."
- Do not delete legacy code in this stage unless a later implementation plan
  explicitly scopes that cleanup.

**Verification:**

- A focused code audit records the active launch path from WebUI and extension.
- Existing tests are classified as current wizard coverage, legacy coverage, or
  stale selectors requiring migration.

**Dependencies:** none.

## Stage 2: First-Time Clarity And Entry Consistency

**Goal:** make the first quick-ingest open understandable without slowing down
returning users.

**Findings addressed:**

- Weak first-time mental model.
- Inconsistent terminology between "Add Content" and "Quick Ingest."
- Storage and destination explanation appears too late for first-time users.

**User outcome:** a first-time user can answer what quick ingest does, what will
happen to their input, and where results appear.

**Scope:**

- Align launcher labels, aria labels, tooltips, and modal title language.
- Add one compact purpose/destination line to the Add step.
- Preserve the quick path: paste or drop content, then use defaults or configure.
- Keep destination copy concrete: Media for persisted items, Knowledge when
  chunking/search is enabled, Workspace only for supported document originals.

**Implementation notes for planning:**

- Avoid tutorial-heavy content in the modal body.
- Put the first-time explanation near the input area, not behind Advanced.
- Use existing i18n patterns.

**Verification:**

- Browser check on WebUI Add step.
- Extension-side check that the same shared copy appears in the sidepanel modal.
- Accessibility check for launcher label, modal name, and input labels.

**Dependencies:** Stage 1.

## Stage 3: Results Handoff And Recovery Actions

**Goal:** make terminal results actionable and honest.

**Findings addressed:**

- Completed results lack a primary Media handoff in the active wizard.
- Result rows expose a Remove action that currently does not do anything.
- Duplicate and skipped recovery copy is ambiguous.

**User outcome:** after ingest, users know where the item went and only see
actions that work.

**Scope:**

- Define result-action priority:
  1. Open in Media for persisted items with a resolvable media id or item target.
  2. Search in Knowledge when content was chunked or indexed for retrieval.
  3. Open in Workspace when the original file type is supported there.
  4. Chat when a supported media or document target is available.
- Remove or implement unavailable actions.
- Clarify duplicate/skipped messaging:
  - "Already queued" for local queue duplicates.
  - "Already in library" for backend duplicate/skipped results.
  - Clear overwrite path through the existing overwrite setting or Deep preset.

**Implementation notes for planning:**

- Prefer one shared result-action mapper over per-component conditionals.
- Do not invent a new destination if the backend result does not expose a stable
  target id.
- If Media handoff needs backend result normalization, scope that explicitly in
  the implementation plan.

**Verification:**

- Unit tests for success, skipped, and error rows.
- Browser/e2e coverage for successful URL ingest result handoff.
- Error-row test proves every visible action changes UI state or dispatches a
  real callback.

**Dependencies:** Stage 1.

## Stage 4: Offline, Cancel, Progress, And Background Status Correctness

**Goal:** make quick ingest resilient and truthful when the server is
unavailable, work is cancelled, work fails, or work continues in the background.

**Findings addressed:**

- Offline/server failure is not integrated into the active wizard's Add step.
- Processing copy is generic and can describe the wrong task.
- Minimized widget treats complete, failed, and cancelled as "Done."
- Cancel, close, minimize, reattach, and recovery states need current-flow
  verification.

**User outcome:** users fail early with recovery guidance, can safely minimize or
cancel, and can trust terminal status.

**Scope:**

- Pass connection state into the active wizard.
- Define disconnected behavior:
  - inputs can remain editable when useful
  - processing actions are disabled or routed to setup/retry guidance
  - health diagnostics remains available where already supported
- Make processing copy match item type or use neutral language.
- Split terminal widget states into Done, Failed, Cancelled, and Interrupted.
- Preserve reattach behavior for direct WebUI jobs and extension runtime jobs.

**Implementation notes for planning:**

- Keep cancellation best-effort wording honest if backend cancellation cannot be
  guaranteed instantly.
- Prefer neutral status copy such as "Processing and indexing content" when a
  batch mixes web, document, audio, and video.
- Do not add new notification systems in this stage.

**Verification:**

- Tests for disconnected Add step and failed start.
- Tests for cancel during processing and stale completion after cancel.
- Tests for minimized widget terminal states.
- Browser check for close while processing confirmation.

**Dependencies:** Stage 1.

## Stage 5: Input Hardening For URL And File Paths

**Goal:** reduce preventable input failures and make file-size promises
technically credible.

**Findings addressed:**

- URL duplicate detection only compares raw strings.
- Invalid pasted URLs become queue cleanup work.
- Supported file copy, accepted file extensions, and detected file types do not
  fully match.
- The UI advertises a 500 MB max while the wizard buffers files into JS arrays.

**User outcome:** users can paste batches and add files with fewer avoidable
errors, and large-file behavior matches what the UI promises.

**Scope:**

- Reuse the existing URL normalization helper for wizard queue dedupe.
- Improve pasted batch feedback with valid and invalid counts.
- Reconcile supported-file copy, file input accept string, detected file types,
  and backend capability.
- Decide the large-file strategy:
  - stream or pass `File` objects through direct upload without full array
    serialization, or
  - lower the displayed limit until large files are safe.

**Required planning decision:**

Large-file handling is a formal decision point, not a copy-only fix. The
implementation plan must choose one path before coding:

- **Transport fix:** preserve the 500 MB promise by avoiding full client-side
  serialization.
- **Truthful limit fix:** lower the advertised limit and add preflight warnings
  until the transport is safe.

**Implementation notes for planning:**

- URL normalization should not silently change the submitted URL in a way that
  surprises users; display can show the original while dedupe uses normalized
  keys.
- File-size remediation may need a transport-level design if extension runtime
  messaging cannot carry large files safely.
- Keep unsupported file handling explicit rather than relying only on the file
  picker accept attribute.

**Verification:**

- Unit tests for URL normalization and duplicate queue behavior.
- Tests for invalid/mixed URL paste.
- Tests for supported and unsupported file classification.
- Manual or automated large-file preflight check appropriate to the chosen
  file strategy.

**Dependencies:** Stage 1. Large-file transport work may depend on Stage 4
runtime/session clarity.

## Stage 6: Verification And Test Gates

**Goal:** close the audit's verification gaps with current wizard coverage.

**Findings addressed:**

- Success, progress, cancel, extension, and mobile behavior were partly
  code-inferred in the audit.
- Several e2e helpers and tests appear to reference older quick-ingest labels,
  tabs, or test ids.

**User outcome:** future regressions are caught in the active quick-ingest flow,
not in stale selectors.

**Scope:**

- Update or add current wizard tests for:
  - empty/default state
  - invalid URL
  - valid URL
  - duplicate URL
  - unsupported file
  - offline/server unavailable
  - start failure
  - processing
  - cancel
  - minimize and reopen
  - success handoff
  - skipped duplicate
  - error retry/remove recovery
  - constrained viewport
  - WebUI and extension launch parity
- Retire or re-scope legacy-modal assertions after Stage 1 classification.
- Record verification commands in the implementation plan and task notes.

**Implementation notes for planning:**

- Prefer shared journey helpers that can target both WebUI and extension shell
  where possible.
- Do not preserve legacy test ids only to satisfy stale tests.
- Keep live-ingest e2e deterministic by using local fixtures and mocked or
  controlled backend responses where appropriate.
- Add focused regression tests inside Stages 2 through 5 when those stages
  change behavior. Use this stage for final parity, stale-selector cleanup, and
  end-to-end confidence rather than deferring all tests until the end.

**Verification:**

- Focused Vitest for shared wizard state and row actions.
- Focused Playwright/e2e for WebUI and extension launch paths.
- Constrained viewport check for the modal and configure options.
- Manual browser smoke only where automation cannot reliably cover the state.

**Dependencies:** Stages 2 through 5 define the expected behavior that this
stage locks down.

## Verification Gaps To Close

These are not open product questions; they are validation tasks that should be
closed during implementation planning or execution.

- Live successful ingest was not submitted during the audit to avoid mutating
  local media data. A deterministic test or local fixture should cover it.
- Extension packaged UI was not live-run during the audit. Shared-code evidence
  is strong, but extension shell behavior should be verified directly.
- Mobile and narrow viewport behavior was code-inferred. The active modal uses a
  fixed Ant Design width with internal scrolling, so constrained viewport must
  be tested.
- Legacy `QuickIngestModal.tsx` and older result-panel code still exist. Stage 1
  must confirm whether they are reachable before implementation treats them as
  dead or active.

## Suggested Implementation Sequence

1. Stage 1 first, because every later stage depends on knowing the active
   surface and current test status.
2. Stage 2 next, because it is low-risk and improves first-time comprehension.
3. Stage 3 next, because results handoff and recovery affect trust after every
   run.
4. Stage 4 next, because truthful progress and cancellation protect long-running
   workflows.
5. Stage 5 next, because file transport and duplicate normalization may require
   deeper coordination.
6. Stage 6 last as the hardening pass, with interim tests added inside each
   earlier stage where practical.

## Completion Criteria For The Remediation Program

- Quick ingest can be launched from WebUI and extension shells with consistent
  shared behavior.
- First-time users understand the purpose and destination before processing.
- Returning users can still use the fast path without added friction.
- Completed, skipped, failed, cancelled, interrupted, minimized, and offline
  states are visually and semantically distinct.
- Every visible result/recovery action works.
- URL/file validation prevents avoidable mistakes without blocking legitimate
  use.
- Current tests cover the active wizard rather than stale legacy selectors.
