# Knowledge QA Follow-On Trust Remediation Design

## Purpose

This design defines a follow-on remediation program for `/knowledge` after the
`TASK-528` recovery and guardrail work. It is based on live WebUI and browser
extension testing against a running backend, plus the current `origin/dev`
baseline at `1d752415d3`.

The goal is to make Knowledge QA trustworthy as a workflow for searching a
personal library and reviewing grounded answers with citations. The work should
not duplicate the readiness, empty-state, and baseline UAT improvements already
landed in `TASK-528`. It should close the deeper trust and reliability gaps that
remain when the page is exercised live.

`/knowledge` remains a Knowledge QA page. It must not become a flashcards page,
deck manager, spaced-repetition workflow, study-set surface, or general
knowledge-management CRUD hub. Flashcards are out of scope unless a separate
flashcards route owns that behavior.

## Current Baseline

The latest `origin/dev` already includes:

- `TASK-528` and child tasks for Knowledge QA WebUI and extension remediation.
- `Docs/Plans/2026-06-07-knowledge-qa-uat-checklist.md`.
- Deterministic Knowledge QA route-state tests.
- Readiness, setup diagnostics, first-run, no-source, no-results, settings,
  evidence, export, and extension guardrail improvements.

Live QA still found important gaps:

- A successful WebUI search returned five sources but zero citations.
- A scoped query could produce a general uncited answer while web fallback was
  disabled.
- Evidence source preview could say full source content was unavailable.
- Source rows could report zero percent match while still supporting a generated
  answer.
- The extension could show setup/offline state changes before recovering.
- Extension search could complete while thread sync failed with an extension
  messaging timeout.
- Export and history can preserve answers whose trust status is degraded unless
  trust status is carried consistently.
- `TASK-528` closeout recorded extension runtime E2E as blocked by the WXT
  production build stall before browser launch. Follow-on extension verification
  must treat that as a real release risk, not as a generic skipped test.

## Product Boundary

Knowledge QA is for:

- Asking questions over selected personal-library sources.
- Reviewing answers with citations and inspectable evidence.
- Narrowing source scope by category, document, note, or saved profile.
- Using web fallback only when the user enables it.
- Handing off to Research Workspace, Chat, Notes, or Media when appropriate.

Knowledge QA is not for:

- Creating or reviewing flashcards.
- Managing decks, study sets, spaced repetition, or quiz workflows.
- Replacing source-owner CRUD screens.
- Hiding unsupported answers behind normal success styling.

## Design Principles

1. Trust status is a product contract, not only UI copy.
2. A citation marker is not enough. Citations must map to inspectable evidence.
3. Missing metadata must fail closed. Older or partial RAG responses should not
   be interpreted as grounded success.
4. Extension failures must be split by setup, reachability, and sync causes.
5. Live UAT must use deterministic fixture data, not whatever happens to be in a
   developer's local database.
6. Longer-term evidence improvements are valuable but must not block core trust
   fixes.

## Core Contracts

### Answer Trust Contract

The RAG response and Knowledge QA UI should classify each search into one of the
following states:

- `cited_answer`: the generated answer has citations that pass the citation
  validity rules below.
- `uncited_degraded_answer`: the answer exists, but it is not fully grounded.
  It must be visibly degraded and must not be exported or revisited as if it
  were grounded.
- `no_answer_insufficient_evidence`: evidence was missing or too weak, so the
  system abstains instead of generating a normal answer.
- `no_results`: the selected sources produced no usable local evidence.
- `failed_search`: the request failed before a reliable answer or no-results
  state could be established.
- `unsynced_local_result`: extension search results are visible, but thread or
  message persistence did not complete.
- `unknown_trust`: an older or partial response is missing the metadata needed
  to classify trust safely.

The frontend must treat missing trust metadata as degraded or unknown, not as
grounded. `unknown_trust` is the canonical state for that case.

### Trust Classification Precedence

Implementation plans should normalize responses in this order so WebUI,
extension, history, and export do not diverge:

1. Transport, auth, backend readiness, or parsing failure becomes
   `failed_search`.
2. Extension search success with thread/message persistence failure becomes
   `unsynced_local_result`, even if local answer content is visible.
3. Older or partial payloads missing required trust metadata become
   `unknown_trust`, never inferred grounded success.
4. No evidence from selected local sources, and no enabled web-fallback
   evidence, becomes `no_results`.
5. Evidence that is present but empty, too weak, unavailable, filtered, deleted,
   or permission-limited becomes `no_answer_insufficient_evidence` unless the
   backend intentionally returns a degraded answer.
6. Answer text with no valid citations becomes `uncited_degraded_answer`.
7. Answer text with valid citations and inspectable evidence becomes
   `cited_answer`.

### Citation Validity Contract

A valid Knowledge QA citation must satisfy all of these requirements:

1. The answer contains a citation marker or structured citation anchor.
2. The marker maps to a returned source.
3. The returned source has inspectable evidence, such as an excerpt, chunk text,
   matched quote, or explicit source-open target.
4. The cited evidence satisfies the release-scope relevance rule below.
5. The citation remains available in answer view, evidence view, history, and
   export output.

If only the marker exists, the answer is not grounded. If the source exists but
the evidence preview is unavailable, the answer may be partially supported but
cannot be treated as a fully cited answer.

For stages 1 through 6, the release-scope relevance rule is intentionally
bounded and deterministic enough for implementation planning: the citation must
anchor to an answer sentence or claim span, map to a returned source, and that
source must expose a retrieved excerpt, chunk, or quote selected by the RAG
pipeline for the same query and source scope. The implementation may use
retrieval score, source status, chunk id, and excerpt presence to validate this
minimum relevance. A stronger semantic claim-to-source adjudicator, such as
claim extraction or model-based evidence judging, belongs to Stage 7 and must
not block the core release gates.

Zero or near-zero relevance, missing evidence text, missing chunk/source ids, or
an unavailable source status cannot produce `cited_answer` unless the backend
also returns a deterministic reason that the source still satisfies the selected
query and source scope. Stage 1B implementation planning must define the weak
evidence threshold policy and the surfaced weak-evidence reason codes.

### Evidence Materialization Contract

Every source row should expose:

- Source id and canonical source type.
- User-readable title.
- Relevance or ranking diagnostics.
- Matched excerpt, chunk text, quote, or unavailable reason.
- Original/open target when available.
- Source status, such as searched, empty, unavailable, filtered, deleted, or
  permission-limited.
- Whether the source is cited by the answer.

Source preview must never show only `Full source content is unavailable` without
explaining why. If full content is unavailable, the UI should still show the
matched excerpt when the backend has it.

### Evidence Origin Contract

Trust state and evidence origin are separate dimensions. Each answer and source
should preserve whether supporting evidence came from:

- `local_library`: selected personal documents, notes, or media.
- `web_fallback`: external/web results used only after explicit opt-in.
- `mixed`: both local-library and web-fallback evidence.
- `unknown_origin`: older or partial payloads missing origin metadata.

Web fallback evidence must be labeled in answer view, evidence view, history,
and export. External evidence must not be presented as searched personal-library
evidence, and personal-library scoped searches must not silently broaden to web
or workspace sources without visible user consent or response metadata.

### Extension Reliability Contract

Extension state must distinguish:

- `setup_missing`: server URL or credentials are missing.
- `setup_invalid`: saved configuration is malformed.
- `backend_unreachable`: the backend health or live endpoint cannot be reached.
- `backend_auth_failed`: backend is reachable but credentials are invalid.
- `api_allowlist_blocked`: extension networking blocks an absolute or host API
  request.
- `search_succeeded_sync_failed`: search completed but thread/message
  persistence failed.
- `search_failed`: search itself failed.

The extension may show local results after a sync failure, but it must label
them as unsynced and offer retry. Console-only failures are not acceptable for
core workflow reliability.

### Scoped Search Contract

Source category selection, exact source selection, saved profiles, request
payload, response source status, and rendered result list must round-trip.

Excluded sources must not appear unless:

- Web fallback is enabled and the source is clearly external.
- The user intentionally broadens the source scope.
- The response reports that a workspace or profile changed the effective scope.

### Export And History Contract

Exports, recent sessions, and history entries must preserve trust state:

- Cited and grounded.
- Degraded or uncited.
- Unknown trust or backwards-compatible payload.
- No results.
- Failed.
- Unsynced.

Export of degraded, unknown, or uncited answers should require explicit user
acknowledgement by default and should produce output that is clearly labeled as
unsupported draft material. Fully blocking export may still be used for failed
searches, no-results states, or missing answer content.

## Staged Remediation Program

### Stage 0: Baseline Reconciliation

Goal: compare live QA findings against the current `origin/dev` baseline,
`TASK-528`, existing tests, and docs.

Deliverables:

- Gap matrix from live finding to current owner.
- Child task map for this follow-on program.
- List of existing tests that already cover each state.
- List of tests that must be added or converted to live-backend coverage.

Acceptance criteria:

- No completed `TASK-528` work is duplicated.
- Every remaining live issue has a target stage.
- The spec explicitly preserves the `/knowledge` QA-only boundary.

### Stage 1A: Trust Taxonomy And Safe Interpretation

Goal: define and render trust states before enforcing strict backend behavior.

Scope:

- Response normalization for existing and older RAG payloads.
- Frontend trust classification.
- Degraded, unknown, no-answer, no-results, failed, and unsynced states.
- Early propagation of trust status into history and export surfaces.

Acceptance criteria:

- Missing citation/source metadata is treated as degraded or unknown.
- Older or partial payloads normalize to `unknown_trust` instead of grounded
  success.
- UI no longer styles uncited answers as normal success.
- History and export dialogs can display trust status before deeper backend
  changes land.

### Stage 1B: Citation Enforcement And Abstention

Goal: enforce answer-generation rules once evidence payloads are available.

Dependencies:

- Requires Stage 2 evidence materialization for strict grounded-answer
  classification.

Scope:

- Backend or shared RAG trust metadata.
- Citation coverage rules.
- Abstention behavior for empty, weak, or uncitable evidence.
- Web fallback disclosure when an answer uses non-local evidence.
- Release-scope citation validity using answer-sentence anchors, returned
  source mapping, and inspectable retrieved excerpts or chunks.

Acceptance criteria:

- A normal successful answer requires valid citations.
- Empty retrieval cannot produce a normal general-answer success when web
  fallback is disabled.
- Weak evidence either produces an abstention or an explicitly degraded answer.
- Semantic claim-to-source judging is not required for the core release gate.

### Stage 2: Evidence Materialization

Goal: make retrieved sources inspectable.

Scope:

- Backend response excerpts, chunks, quotes, source ids, source types, and
  unavailable reasons.
- Evidence rail source rows and source previews.
- Open-original behavior.
- Details view diagnostics such as score, reranking, source status, and why the
  source was retrieved.

Acceptance criteria:

- Every source row has inspectable evidence or a specific unavailable reason.
- Source preview explains missing full content.
- Source rows expose stable source, chunk, and excerpt identifiers needed for
  later citation jump targets.

### Stage 3: Extension Reliability

Goal: make extension setup, search, and sync states trustworthy.

Scope:

- Setup and host/API allowlisting diagnostics.
- Backend health and auth checks.
- Runtime E2E harness readiness, including the known WXT production build stall
  before browser launch.
- Offline/retry state transitions.
- Service-worker messaging and thread creation.
- Unsynced local-result recovery.

Acceptance criteria:

- Extension setup does not conflate missing config, offline backend, auth
  failure, and allowlist failure.
- A successful extension search either syncs or shows an actionable unsynced
  state.
- Extension console errors that affect the workflow have visible UI
  equivalents.
- Extension signoff requires a runtime E2E harness that launches the options
  route. If the WXT/runtime blocker remains, it must be recorded as a
  release-blocking extension verification gap and extension signoff cannot be
  treated as passed.

### Stage 4: Scoped Search Reliability

Goal: make exact source selection and saved profiles auditable.

Scope:

- Category and exact document/note selection persistence.
- Saved source/search profiles.
- Request payload validation.
- Result-source validation against scope.
- Web fallback and workspace scope exceptions.

Acceptance criteria:

- Saved profiles restore exact source categories and exact source ids.
- Results reflect the selected source scope.
- Scope changes are visible before the search runs and in result details after
  the search completes.

### Stage 5: Export, History, And Recovery Hardening

Goal: prevent unsupported answers from leaking into durable user workflows as if
they were grounded.

Scope:

- Export dialog trust warnings.
- Markdown, PDF, and chatbook output trust labeling.
- Search history and recent-session trust labels.
- Failed, degraded, no-results, and unsynced recovery actions.

Acceptance criteria:

- Degraded and uncited answers are labeled in history and export.
- Export either blocks or clearly labels unsupported answers.
- History restoration preserves citations, source status, and trust state.

### Stage 6: Live UAT Gates

Goal: make the live workflow hard to regress.

Scope:

- Deterministic fixture library for live backend tests.
- WebUI live backend tests.
- Extension live backend tests.
- Console and network assertions.
- Release checklist updates.

Fixture requirements:

- One indexed source with a known cited answer.
- One distractor source that should not be cited.
- One no-match query.
- One scoped-search query that excludes a tempting source.
- One fixture that produces a deliberately uncited or degraded response through
  mocks or test mode.

Acceptance criteria:

- WebUI and extension both run against seeded content.
- Tests cover backend unavailable, setup required, ready search, no source,
  no results, cited result, uncited degraded result, scoped result, export, and
  extension sync failure.
- Live cited, no-results, and scoped-search paths must exercise seeded backend
  data. Mock or test-mode forcing is allowed only for the deliberate degraded or
  uncited response fixture, and must be labeled as such.
- Final release gate includes rendered verification, not unit tests alone.

### Stage 7: Longer-Term Evidence Workflow

Goal: improve evidence review beyond basic citations as separate non-blocking
enhancements after the core trust fixes.

Scope:

- Claim-to-source mapping.
- Evidence confidence and coverage summaries.
- Better Research Workspace handoff.
- Relationship to Chat, Notes, and Media.
- Evidence audit view for power users.

Acceptance criteria:

- Stage 7 items are tracked as non-blocking enhancements.
- Stage 7 is outside the release scope for stages 1 through 6.
- `/knowledge` remains QA-only.
- Research Workspace remains the deeper workspace for expanded research
  workflows.

## Dependency Model

- Stage 0 runs first.
- Stage 1A can begin immediately after Stage 0.
- Stage 2 can run in parallel with Stage 1A.
- Stage 1B depends on Stage 2.
- Stage 3 can run in parallel after Stage 0.
- Stage 4 can run after Stage 0, with final UAT coverage in Stage 6.
- Stage 5 can begin with Stage 1A status propagation and finish after Stage 1B.
- Stage 6 should be the release gate for the full series.
- Stage 7 must not block stages 1 through 6 and should be planned as separate
  backlog enhancements after the core remediation series.

## Affected Surfaces

Backend:

- Unified RAG request and response schemas.
- RAG pipeline citation metadata.
- Source status and excerpt payloads.
- Test fixture or test-mode data seeding.

Shared frontend UI:

- Knowledge QA provider and response normalization.
- Answer panel.
- Evidence rail, source list, source preview, and details panel.
- Source scope controls and saved profiles.
- Export dialog.
- History and recent sessions.

WebUI:

- `/knowledge` route.
- Readiness gate integration where needed.
- Live backend E2E scripts.

Extension:

- Options route `#/knowledge`.
- Setup diagnostics and health state.
- Extension networking and allowlisting.
- Service-worker messaging and thread sync.
- Extension E2E harness.

Docs:

- UAT checklist.
- User guide.
- Stage implementation plans.
- Backlog task records.

## Testing Strategy

Deterministic tests:

- Mocked WebUI and extension route states for readiness blocked, setup required,
  ready search, no source, no result, cited result, degraded uncited result,
  failed search, unsynced extension result, and export warning.
- Component tests for answer trust classification, citation validity display,
  evidence source preview, source picker persistence, export labeling, and
  history trust badges.
- Contract tests for RAG response normalization, source status, excerpts,
  unavailable reasons, and backwards-compatible payload handling.

Live backend tests:

- WebUI `/knowledge` against a running backend with seeded fixture content.
- Extension `options.html#/knowledge` against the same backend.
- Console and network assertions for OpenAPI, health, RAG, chat/thread sync, and
  extension messaging.
- Known cited answer, known no-result query, scoped query, and extension sync
  flow.
- Explicit WXT/runtime harness health check before treating extension browser
  coverage as passed or skipped.

Security and privacy checks:

- Web fallback remains disabled unless the user enables it.
- External/web evidence is labeled distinctly from personal-library evidence.
- No secret values appear in logs, exports, or visible diagnostics.
- Backend Bandit runs are required for touched Python production code.

## Release Gates

The remediation series is not complete until:

- A normal answer cannot be marked grounded unless valid citations map to
  inspectable evidence.
- Every source row has evidence text or a specific unavailable reason.
- Empty or weak evidence does not produce a normal success answer.
- Web fallback origin is preserved and labeled anywhere the answer, evidence,
  history, or export is shown.
- Extension setup, reachability, and sync failures have visible recovery states.
- Extension runtime E2E passes for the options route and Knowledge QA workflow.
  If WXT or the runtime harness still blocks browser launch, the remediation
  series is release-blocked with a tracked owner, command, timeout, and failure
  artifact.
- Export and history retain trust status.
- WebUI live-backend UAT passes with deterministic fixture data, and extension
  live-backend UAT passes after the runtime harness health gate is green.
- `/knowledge` remains separate from flashcards and study workflows.

## Implementation Planning Notes

The implementation-planning step created reviewable child tasks and plan files
from the stages above. Continuation should start from these task slices and keep
each Backlog record current as implementation proceeds:

- `TASK-2279.1`: Reconcile Knowledge QA follow-on trust baseline.
- `TASK-2279.2`: Define Knowledge QA trust taxonomy and safe response handling.
- `TASK-2279.3`: Materialize Knowledge QA evidence excerpts and source previews.
- `TASK-2279.4`: Enforce Knowledge QA citation validity and abstention.
- `TASK-2279.5`: Harden Knowledge QA extension setup and sync reliability.
- `TASK-2279.6`: Verify Knowledge QA scoped search and saved profile round-trip.
- `TASK-2279.7`: Propagate Knowledge QA trust status into export and history.
- `TASK-2279.8`: Add Knowledge QA live UAT fixtures and release gates.
- `TASK-2279.9`: Plan non-blocking Knowledge QA evidence workflow improvements.

Each implementation plan should include:

- Files and ownership boundaries.
- Failing tests first where practical.
- WebUI and extension verification expectations.
- Backwards-compatibility behavior for older RAG responses.
- Evidence origin and web fallback labeling expectations.
- Extension runtime E2E harness preconditions when the stage touches extension
  signoff.
- Explicit statement that flashcards are out of scope for `/knowledge`.
