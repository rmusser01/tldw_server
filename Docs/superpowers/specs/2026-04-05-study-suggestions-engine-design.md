# Study Suggestions Engine Design

Date: 2026-04-05
Status: Approved for planning
Owner: Codex brainstorming session

## Summary

Add a shared `Study Suggestions` capability that recommends what a user should do next after finishing a quiz attempt or a flashcard study session.

The first product slice is intentionally narrow:

- generate source-aware follow-up suggestions for `quizzes` and `flashcards`
- show an immediate post-session summary with suggested next topics
- let users edit the suggested topics before generating the next artifact
- persist frozen suggestion snapshots so users can reopen them later from history
- support `Refresh recommendations` without overwriting the original snapshot

The design should not dead-end inside study features. It should establish a generalized internal suggestion-engine shape that can later support other services such as notes, chat, research, and presentations.

## Problem

The current flashcards and quizzes experience already supports study actions, but it does not yet support session-level continuation guidance.

Today:

- Quizzes have durable attempts, detailed results, remediation actions, citations, and generation paths, but they do not produce a reusable “study next” recommendation snapshot. See [`ResultsTab.tsx`](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Quiz/tabs/ResultsTab.tsx) and [`QuizRemediationPanel.tsx`](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Quiz/components/QuizRemediationPanel.tsx).
- Flashcards have mature review history for `SM-2+` and `FSRS`, plus aggregated analytics, but the current “session” concept in the review UI is local React state rather than a persisted entity. See [`ChaChaNotes_DB.py`](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py) and [`ReviewTab.tsx`](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx).
- Both generation surfaces already accept `focus_topics`, which means the repo has a useful continuation hook, but not a shared recommendation layer. See [`quizzes.ts`](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/services/quizzes.ts) and [`flashcards.py`](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/schemas/flashcards.py).

This leaves a product gap:

- users can complete study work
- users can manually generate more material
- but the system does not yet summarize what was covered, identify weak or adjacent topics, and guide the user into the next focused step

## Goals

- Add a shared recommendation layer for `quizzes` and `flashcards`.
- Show a post-session summary with covered topics, weakness-oriented follow-up topics, and next-step actions.
- Keep recommendations source-aware whenever evidence exists.
- Let users edit, add, remove, and rename suggested topics before generating the next artifact.
- Support both same-format and cross-format continuation:
  - quiz -> quiz
  - quiz -> flashcards
  - flashcards -> flashcards
  - flashcards -> quiz
- Persist frozen suggestion snapshots and allow reopening them from history.
- Support `Refresh recommendations` by creating a new snapshot rather than mutating the original.
- Reuse existing generation infrastructure, especially `focus_topics`, instead of creating new study generators.
- Establish a generalized engine shape that can later support notes, chat, research, presentation, and similar activity-driven suggestions.

## Non-Goals

- Build a universal recommendation platform for every service in the first implementation plan.
- Replace existing quiz remediation workflows.
- Replace existing flashcard scheduler logic or review-history storage.
- Introduce a new global taxonomy system for all topics in v1.
- Add real-time recommendations during a session.
- Add third-party integrations or external recommendation providers.
- Build a public-facing recommendation feed or homepage dashboard in v1.
- Guarantee perfect semantic deduplication of all topic labels in the first release.

## Requirements Confirmed With User

- The same conceptual recommendation model should work in both flashcards and quizzes.
- Recommendation ranking should be `hybrid`:
  - use session performance signals
  - but prefer recommendations that can still be traced to original source material
- Both modules should expose:
  - a same-format primary action
  - a cross-format secondary action
- Suggestions should be `weakness + adjacent`, not weakness-only and not broad coverage.
- The feature should appear immediately after activity completion and should also be available later from history.
- Topic recommendation identity should prefer:
  - source citation labels and structured source metadata
  - then existing stored tags
  - then model-derived topic labels when metadata is weak
- Users should be able to edit the topic builder substantially:
  - add
  - remove
  - rename
  - revise topics before generation
- History reopen should support both:
  - the original frozen snapshot
  - an explicit `Refresh recommendations` action
- Flashcards should support:
  - auto-summary on queue completion
  - manual early session end
- Manual topics are allowed, even when they are not tied to the original source bundle.
- The design should leave room for a later generalized suggestion engine for other internal services such as notes, chat, research, and presentations.

## Current State

### Quiz already has a strong native activity anchor

Quizzes already persist attempt records, question-level answer breakdowns, remediation actions, and citation metadata.

Relevant anchors:

- [`ResultsTab.tsx`](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Quiz/tabs/ResultsTab.tsx)
- [`QuizRemediationPanel.tsx`](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Quiz/components/QuizRemediationPanel.tsx)
- [`quizzes.ts`](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/services/quizzes.ts)
- [`quizzes.py`](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/schemas/quizzes.py)

Important existing properties:

- `QuizAttempt` is already a durable unit.
- quiz questions and answers can carry `source_citations`
- quizzes can carry `source_bundle_json`
- quiz generation already accepts `focus_topics`

Design implication:

- quiz suggestion snapshots should anchor to the existing attempt record rather than duplicating quiz lifecycle state in a second session table

### Flashcards already persist review history, but not a first-class study session

Flashcards already persist the data needed for `SM-2+` and `FSRS`.

Relevant anchors:

- [`ChaChaNotes_DB.py`](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py)
- [`ReviewTab.tsx`](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Flashcards/tabs/ReviewTab.tsx)
- [`ReviewAnalyticsSummary.tsx`](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Flashcards/components/ReviewAnalyticsSummary.tsx)

Important current facts:

- flashcard review history is persisted in `flashcard_reviews`
- each review write updates per-card scheduler state and review timestamps
- analytics summary is aggregated from review rows, not from a persisted session entity
- the current “this session” UX in `ReviewTab` is local component state and resets when review scope changes
- existing review rows are not yet grouped under a persisted session identifier

Design implication:

- flashcards need a new persisted review-session boundary if the product requires durable end-of-session summaries and history reopen

### Existing generators already expose the right integration seam

The current generation surfaces already accept user-provided focus topic lists.

Relevant anchors:

- [`quizzes.ts`](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/services/quizzes.ts)
- [`flashcards.py`](/Users/macbook-dev/Documents/GitHub/tldw_server2/tldw_Server_API/app/api/v1/schemas/flashcards.py)
- [`GenerateTab.tsx`](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Quiz/tabs/GenerateTab.tsx)

Design implication:

- v1 should feed normalized topic selections into existing generation endpoints rather than creating a new generation surface from scratch

## Design Constraints Discovered During Review

### Snapshot Safety Constraint

Frozen suggestion snapshots must not become a stale content cache that leaks private or deleted data after permissions change.

V1 should store stable refs, labels, evidence classes, counts, and safe display metadata by default. Detailed evidence should be fetched live when the snapshot is reopened.

### Native Anchor Constraint

The design should not duplicate activity models that already exist.

Quiz attempts are already durable and should remain the native activity anchor. Only services that lack a durable completion entity should require a new session table.

### Flashcard Session Boundary Constraint

Flashcard review history is not the same as a flashcard study session.

If the user wants immediate summaries plus history reopen, flashcards need an explicit persisted session boundary instead of deriving session behavior only from UI state.

### Grounding Constraint

Allowing manual topic entry must not blur the difference between source-grounded and exploratory suggestions.

The UI and persistence model should preserve topic evidence classes so the product can distinguish provenance-backed recommendations from open-ended user intent.

### Canonicalization Constraint

Refresh behavior will drift unless topic labels are normalized.

The design needs a canonicalization step to reduce duplicate or slightly varied labels across sessions and refreshes.

### Duplicate Artifact Constraint

Once snapshots can generate quizzes and flashcard artifacts, the system must detect prior outputs from the same snapshot and topic selection.

Otherwise reopened history will easily create duplicate decks and quizzes.

### Async Completion Constraint

Suggestion generation must not block quiz completion or flashcard review completion.

The recommendation pipeline may need model-assisted topic derivation when metadata is thin. That work should be asynchronous and best-effort.

## Approaches Considered

### Approach 1: Shared Suggestion Engine With Service Adapters

Build one shared recommendation engine and connect it to:

- quiz attempts
- flashcard review sessions

Pros:

- matches the user’s “same model in both modules” request
- aligns with the repo’s existing `focus_topics` and quiz/flashcard handoff capabilities
- creates a clean path to future notes/chat/research/presentation suggestion surfaces
- avoids duplicated recommendation logic

Cons:

- requires one new durable flashcard session model
- requires new persistence for suggestion snapshots and output links
- requires careful boundary design so the engine is generic enough without becoming over-abstract

### Approach 2: Separate Module-Specific Recommendation Implementations

Build a quiz-only continuation feature and a flashcards-only continuation feature that happen to look similar.

Pros:

- simpler to land one module at a time
- quiz path can ship quickly because attempts already exist

Cons:

- recommendation logic will drift across modules
- harder to generalize later
- duplicates topic ranking, refresh, and duplicate-output handling

### Approach 3: Thin Frontend Layer On Existing Data

Add only a UI panel and compute recommendations ad hoc in the frontend from currently available results or review data.

Pros:

- lowest initial scope
- minimal schema work

Cons:

- weak fit for persisted history and refresh lineage
- poor fit for permission-safe snapshots
- does not solve the flashcard session-boundary problem

## Recommendation

Use Approach 1.

Build a shared suggestion engine with native service anchors:

- quizzes anchor to `quiz_attempt`
- flashcards anchor to a new `flashcard_review_session`

The engine should own ranking, canonicalization, snapshot persistence, refresh lineage, and duplicate-output detection. The services should own their activity adapters and presentation surfaces.

## Proposed Architecture

### Product Shape

V1 adds a `Study Suggestions` experience that appears after:

- a completed quiz attempt
- a completed or manually ended flashcard review session

The experience should produce:

- a post-session summary
- a ranked topic list
- an editable topic builder
- one primary same-format action
- one secondary cross-format action
- a persisted frozen snapshot that can later be reopened

This is a study-specific feature in the UI, but it should be implemented on top of a generalized internal suggestion contract.

### Generalized Internal Suggestion Contract

Recommended conceptual units:

- `SuggestionContext`
- `SuggestionSnapshot`
- `SuggestionCandidate`
- `SuggestionOutputLink`

Recommended context concepts:

- `service`
- `activity_type`
- `anchor_type`
- `anchor_id`
- `workspace_id`
- `source_bundle`
- `summary_metrics`
- `performance_signals`

The first implementation plan should only support study suggestion contexts from quizzes and flashcards, but these concepts should remain service-agnostic so later services can add adapters rather than replace the engine.

### Recommended Units

- `QuizAttemptSuggestionAdapter`
- `FlashcardReviewSessionAdapter`
- `TopicEvidenceResolver`
- `TopicNormalizer`
- `SuggestionRanker`
- `SuggestionSnapshotService`
- `SuggestionRefreshService`
- `SuggestionOutputLinkStore`

Responsibilities:

- `QuizAttemptSuggestionAdapter`
  - reads quiz attempt, quiz metadata, answer correctness, source bundle, and citations
  - emits a normalized suggestion context

- `FlashcardReviewSessionAdapter`
  - reads persisted flashcard session summary plus reviewed-card evidence
  - emits a normalized suggestion context

- `TopicEvidenceResolver`
  - resolves topic candidates from strongest to weakest evidence
  - source metadata and citation labels first
  - existing tags next
  - model-derived labels from source-backed content last

- `TopicNormalizer`
  - canonicalizes labels
  - dedupes obvious near-duplicates
  - keeps alternate labels for display and auditability

- `SuggestionRanker`
  - ranks weakness-first
  - adds adjacent topics only when they remain tied to the same source bundle or cited material
  - preserves evidence class and confidence

- `SuggestionSnapshotService`
  - persists frozen snapshots
  - exposes reopen-ready summary payloads

- `SuggestionRefreshService`
  - recomputes suggestions into a new snapshot
  - never mutates the prior snapshot in place

- `SuggestionOutputLinkStore`
  - links generated follow-up artifacts back to the exact snapshot version that created them
  - powers duplicate detection and `Open existing` behavior

## Persistence Model

### Table 1: `suggestion_snapshots`

This is the primary persisted unit the user sees later in history.

Suggested fields:

- `id`
- `service`
- `activity_type`
- `anchor_type`
- `anchor_id`
- `workspace_id`
- `suggestion_type`
- `payload_json`
- `user_selection_json`
- `status`
- `refreshed_from_snapshot_id`
- `created_at`
- `last_modified`
- `client_id`
- `version`
- `deleted`

Suggested v1 values:

- `service`
  - `quiz`
  - `flashcards`
- `activity_type`
  - `quiz_attempt`
  - `flashcard_review_session`
- `anchor_type`
  - `quiz_attempt`
  - `flashcard_review_session`
- `suggestion_type`
  - `study_suggestions`
- `status`
  - `active`
  - `superseded`

Recommended `payload_json` sections:

- `summary`
  - score, counts, streak, reviewed-card counts, and similar safe metrics
- `topics`
  - canonical label
  - display label
  - evidence class
  - evidence strength
  - weakness and adjacency flags
  - safe source refs
  - source availability flags
- `actions`
  - primary and secondary follow-up actions
- `metadata`
  - generation version
  - refresh reason
  - safe service-specific display context

### Table 2: `suggestion_generation_links`

This records what a snapshot produced.

Suggested fields:

- `id`
- `snapshot_id`
- `target_service`
- `target_type`
- `target_id`
- `selection_fingerprint`
- `created_at`

This allows:

- `Open existing` instead of silent duplicate generation
- auditability for history reopen
- future analytics on recommendation conversion

### Table 3: `flashcard_review_sessions`

This table exists because flashcards need a native session anchor for summary/history behavior.

Suggested fields:

- `id`
- `workspace_id`
- `deck_id`
- `review_mode`
- `tag_filter`
- `scope_key`
- `started_at`
- `completed_at`
- `last_activity_at`
- `status`
- `review_count`
- `lapse_count`
- `summary_metrics_json`
- `source_bundle_json`
- `client_id`
- `version`
- `last_modified`
- `deleted`

Suggested `status` values:

- `active`
- `completed`
- `abandoned`

### Existing Table Extension: `flashcard_reviews.review_session_id`

The new flashcard session model must be able to anchor individual review events.

Recommended change:

- add nullable `review_session_id` to `flashcard_reviews`
- reference `flashcard_review_sessions.id`
- write the active session id on every new review row
- allow legacy rows to remain null for backward compatibility

This keeps existing review-history semantics intact while making session-scoped analytics and suggestion derivation precise.

Implementation-plan scope rule:

- do not create a generalized `activity_sessions` table in the first plan
- use native quiz attempts as-is
- introduce only the flashcard session table that is required for parity

Sync/version rule:

- `suggestion_snapshots` and `flashcard_review_sessions` should follow the repo’s normal sync/version conventions, including `client_id`, `version`, `last_modified`, and sync-log participation
- `flashcard_reviews.review_session_id` is an additive linkage field, not a replacement for existing review-history storage

## Permission-Safe Snapshot Behavior

Suggestion snapshots should be safe to reopen later even when underlying content visibility has changed.

Store by default:

- stable source refs
- lightweight labels
- evidence class
- topic flags
- counts and scores
- safe route hints

Do not store by default:

- large source excerpts
- rich note or chat content
- long quotes from source content
- content that would bypass later permission checks

History reopen behavior:

- render the frozen snapshot immediately from stored payload
- fetch richer evidence live under current permissions
- if evidence cannot be loaded:
  - show `Unavailable source`
  - keep the topic and summary row visible
  - mark the evidence as orphaned or inaccessible

This preserves trust, keeps history useful, and avoids turning the snapshot table into a second content store.

## Flashcard Session Lifecycle

### Session Start

A flashcard review session starts on the first submitted review after:

- opening the review surface
- switching scope
- or completing/abandoning the previous session

### Scope Key

One session is defined by a scope key composed of:

- review mode
  - `due`
  - `cram`
- selected deck or global scope
- cram tag filter when present

### Session End

A session ends when:

- the queue is exhausted
- the user explicitly clicks `End Session`
- or inactivity exceeds the configured timeout

Recommended inactivity timeout:

- 30 minutes without a submitted review

### Resume Behavior

- if the user returns before timeout and the scope key is unchanged, continue the active session
- if the scope changes, close the prior session and start a new one on the next submitted review

### Overlap Rule

V1 should not allow overlapping active flashcard sessions across different scopes.

Closing the prior session on scope change is simpler, easier to reason about, and produces cleaner history.

### Same-Scope Multi-Client Rule

V1 also needs deterministic behavior when the same user reviews the same scope from multiple clients or tabs.

Recommended rule:

- at most one active session per user per `scope_key`
- when a review is submitted, the backend should attach it to the newest active session for that same user and scope
- if no active session exists, create one
- if multiple active sessions somehow exist for the same scope because of legacy rows or a race condition:
  - pick the newest active session as authoritative
  - mark older duplicates as `abandoned`
  - continue writing new reviews to the authoritative session

This keeps `review_session_id` assignment deterministic and avoids fragmenting one logical study run across multiple session rows.

## Topic Model

### Topic Evidence Classes

Each topic should carry one of three explicit evidence classes:

- `grounded`
  - derived from source metadata, citation labels, or strongly source-backed structure
- `derived`
  - inferred from source-backed session content when metadata is weak
- `exploratory`
  - fully user-added or manual

### Ranking Rules

- rank weakness-first
- add adjacency only from the same source bundle or clearly related cited material
- do not present adjacent topics as already-covered topics
- only `grounded` and `derived` topics participate in source-aware adjacency ranking
- `exploratory` topics are allowed for generation but do not participate in source-backed ranking claims

### Topic Normalization

The design needs a stable normalization step.

Recommended responsibilities:

- normalize case and punctuation
- trim obvious filler wording
- dedupe near-identical labels
- choose a canonical label
- preserve alternate display labels when useful

This reduces refresh churn and keeps history understandable.

## Suggestion Generation Flow

### Triggering

Suggestion generation should run asynchronously after:

- quiz attempt completion
- flashcard session completion

### Async Model

Recommended flow:

1. activity completes immediately
2. suggestion snapshot generation is enqueued
3. the UI shows a loading or pending state in the suggestions panel
4. the snapshot appears when generation succeeds
5. failure is non-fatal and retryable

This ensures suggestions never block:

- quiz completion
- flashcard review completion
- history rendering

Backend orchestration rule:

- use `Jobs` for snapshot generation and refresh execution because this is a user-visible async feature that should support reliable status, retries, and later operational visibility
- do not introduce a separate Scheduler-based path for this v1 user-facing flow

### Client Status Contract

The async job path needs a concrete client contract so quiz and flashcard surfaces can render pending, ready, and failed states consistently.

Recommended v1 contract:

- expose suggestion status by `anchor_type` and `anchor_id`, not only by raw `job_id`
- the status envelope should include:
  - `anchor_type`
  - `anchor_id`
  - `status`
    - `none`
    - `pending`
    - `ready`
    - `failed`
  - `job_id` when work is pending
  - `snapshot_id` when a snapshot is ready
  - `error` when the latest job failed
- the client should poll or refetch by anchor until a terminal state is reached
- once `snapshot_id` exists, the client should load the frozen snapshot through the normal snapshot read endpoint

This keeps low-level job orchestration out of the UI while still giving the client a stable pending-state contract.

### Ranking Inputs

Recommended ranking inputs for quiz:

- incorrect answers
- low score questions
- hint usage
- source bundle coverage
- per-question citations and tags

Recommended ranking inputs for flashcards:

- `Again` and `Hard` ratings
- lapse events
- concentration of weak reviews around source refs, tags, or topic labels
- answer-time strain as a secondary signal

### Flashcard Grounding Eligibility

Flashcard sessions will not always have enough provenance to support strong source-aware adjacency.

Recommended eligibility rules:

- `source-aware adjacency` is allowed only when a meaningful portion of the reviewed cards can be tied to stable evidence through:
  - provenance-backed study-pack or citation data
  - durable source references with resolvable source metadata
  - a session or deck with coherent source lineage
- `weakly grounded suggestions` are allowed when only coarse tags or partial source refs exist
- `exploratory-only follow-up` should be used when a session is mostly manual or mixed-provenance with no stable shared source bundle

Implementation-plan rule:

- define an explicit backend predicate such as `is_source_grounded_session`
- when that predicate is false, the UI should still render the panel but suppress source-aware adjacency claims and use weaker explanatory copy

This prevents the system from overstating provenance for manual or mixed decks.

## User Flows

### Quiz Surface

Entry point:

- post-attempt results experience, above or alongside detailed answer review

Panel content:

- score summary
- topics covered
- weak topics
- adjacent study-next topics
- editable topic builder
- primary action: `Generate Follow-up Quiz`
- secondary action: `Generate Flashcards`

History reopen:

- from prior attempt history
- show frozen snapshot first
- offer `Refresh recommendations`

### Flashcards Surface

Entry points:

- queue completion in due mode
- queue completion in cram mode
- manual `End Session`

Panel content:

- reviewed-count summary
- deck/scope context
- weak topics inferred from session evidence
- adjacent study-next topics
- editable topic builder
- primary action: `Generate Focused Flashcards`
- secondary action: `Generate Quiz`

History reopen:

- from persisted flashcard review session history or session-linked suggestion snapshot history

Recommended v1 entry point:

- add a lightweight `Recent study sessions` section inside the Flashcards review surface
- list completed flashcard review sessions with their suggestion state
- let users reopen the linked snapshot from that same review area

Implementation-plan rule:

- do not build a new global history page in the first implementation plan
- keep flashcard history access anchored inside the existing review workspace

### Editable Topic Builder

Required user actions:

- remove a topic
- rename a topic
- add a manual topic
- optionally restore suggested defaults

Each topic row should show:

- display label
- evidence class
- whether it is weakness-driven or adjacent
- whether it is currently selected for generation

Manual topics should be visually distinct so users understand they are exploratory rather than source-grounded.

## Generation Integration

### Same-Format And Cross-Format Actions

Each module should expose:

- one primary same-format action
- one secondary cross-format action

V1 mapping:

- quiz -> follow-up quiz primary
- quiz -> flashcards secondary
- flashcards -> flashcards primary
- flashcards -> quiz secondary

### Existing Generation Reuse

Reuse the current generation endpoints and pass normalized selected topics into existing `focus_topics` inputs.

Do not create new standalone generation APIs in the first implementation plan unless an adapter layer is needed for orchestration and duplicate detection.

### Duplicate Handling

If a snapshot has already produced an artifact for the same selected topics and target format:

- default to `Open existing`
- still allow `Generate again` explicitly

This should be powered by:

- `suggestion_generation_links`
- a stable selection fingerprint

Recommended v1 fingerprint inputs:

- `snapshot_id`
- `target_service`
- `target_type`
- normalized selected topic set
- generator action kind
  - for example `follow_up_quiz` versus `follow_up_flashcards`
- generator version when prompt or ranking contracts materially change

The fingerprint should not collapse outputs from different snapshots, even when topic labels happen to match.

## Refresh Behavior

History reopen should support both stability and evolution.

Rules:

- original snapshot remains frozen
- `Refresh recommendations` creates a new snapshot row
- new snapshot should point to `refreshed_from_snapshot_id`
- refreshed snapshot may have:
  - new ranking
  - new adjacent topics
  - updated evidence availability
- prior outputs should stay linked to the exact original snapshot that created them

This keeps history auditable and avoids confusing mutation of prior decisions.

## Error Handling And Degradation

### Suggestion Generation Failure

- do not block the base activity completion flow
- show a non-fatal error state in the suggestions panel
- allow retry
- preserve the rest of the result/review UI

### Weak Or Missing Evidence

- still allow the panel to render
- degrade to weaker topic evidence classes
- if no adjacent topics are reliable, show only weakness-oriented suggestions
- if no meaningful topics can be inferred, fall back to simple continuation actions using the same source scope

### Source Deletion Or Visibility Change

- keep the snapshot summary visible
- mark detailed evidence unavailable
- do not fail the whole panel because one source is inaccessible

### Cross-Format Generation Failure

- preserve the topic builder state
- preserve the snapshot
- allow retry or switching to the other generation path

## Testing Strategy

### Unit Tests

- topic evidence resolution
- topic normalization and dedupe
- weakness-first ranking
- adjacency filtering
- exploratory-topic handling
- snapshot lineage and refresh behavior
- duplicate-output fingerprinting
- flashcard session lifecycle transitions

### Integration Tests

- quiz attempt -> suggestion snapshot creation
- flashcard review session -> suggestion snapshot creation
- history reopen of frozen snapshot
- refresh creating a superseding snapshot
- snapshot-linked generation in both same-format and cross-format flows
- permission-safe fallback when source details are unavailable

### UI Tests

- quiz post-attempt suggestions panel
- flashcard end-session suggestions panel
- editable topic builder add/remove/rename
- evidence class badges
- loading, failure, and retry states
- `Open existing` versus `Generate again`

### Regression Tests

- existing quiz remediation behavior remains intact
- existing flashcard review and analytics behavior remains intact
- existing `focus_topics` generation behavior remains intact

## Future Expansion

The first implementation plan should stop at study suggestions for quizzes and flashcards.

However, the internal design should allow future services to add adapters such as:

- notes
- chat
- research
- presentations

The extension path should be:

1. add a new service-specific activity adapter
2. emit a normalized suggestion context
3. reuse snapshot, refresh, and output-link infrastructure
4. add a service-specific surface and action policy

This keeps the engine generalizable without requiring the first implementation plan to solve every future service immediately.

## Implementation-Plan Scope Rule

The first implementation plan should include:

- suggestion snapshots
- suggestion generation links
- flashcard review sessions
- quiz and flashcard adapters
- async suggestion generation
- quiz and flashcard UI surfaces
- duplicate-output handling
- refresh lineage

The first implementation plan should not include:

- notes, chat, research, or presentation adapters
- a universal activity-event platform
- a global topic ontology
- live in-session suggestions
- a shared dashboard outside the quiz and flashcard surfaces

## Success Criteria

This design is successful if it delivers:

- a trustworthy post-session recommendation experience in quizzes and flashcards
- topic suggestions that remain explicit about their evidence class
- durable, reopenable frozen snapshots
- refreshable recommendations without destructive mutation
- reuse of current generation infrastructure through `focus_topics`
- a clean internal path for future suggestion surfaces in other services
