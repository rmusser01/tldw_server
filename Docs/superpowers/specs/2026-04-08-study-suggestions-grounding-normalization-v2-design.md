# Study Suggestions Grounding And Normalization V2 Design

Date: 2026-04-08
Status: Proposed
Owner: Codex brainstorming session

## Summary

Build a deterministic `grounding + normalization v2` upgrade for the existing `StudySuggestions` engine.

This phase is intentionally narrow:

- improve topic grounding quality for quiz attempts and flashcard review sessions
- introduce a stable `topic_key` model without breaking existing frozen snapshots
- refactor topic extraction ownership so adapters emit structured evidence and the pipeline owns canonicalization
- keep snapshots permission-safe and backward-compatible
- prepare the engine for later analytics and cross-service expansion without introducing LLM dependency in this phase

This is the first follow-up phase after the merged quiz/flashcard study-suggestions MVP. It should improve recommendation quality inside the existing study loop before adding analytics or expanding to other services.

## Problem

The current engine works, but the quality ceiling is limited by four structural issues:

1. Topic identity is too label-driven.
2. Extraction logic is concentrated in snapshot assembly rather than clearly split between adapters and the topic pipeline.
3. Flashcard suggestions currently do not feed rich enough provenance or performance signals into the ranking path.
4. The current system has no explicit normalization-version contract, which makes future analytics and canonicalization upgrades risky.

The result is a v1 that can produce useful recommendations, but is still vulnerable to:

- duplicate or drifting topic labels
- unstable semantics when labels are renamed or refreshed
- weak flashcard grounding quality
- difficulty measuring whether the same topic recurs across sessions

## Goals

- Add stable semantic topic identity through `topic_key`.
- Preserve separate snapshot-local topic identity for UI editing and reopening history.
- Move extraction boundaries toward:
  - adapters produce structured evidence inputs
  - pipeline performs normalization, canonicalization, merge, and ranking
  - snapshot service orchestrates and serializes
- Improve flashcard suggestion quality by feeding real provenance and performance signals into the pipeline.
- Keep snapshot payloads permission-safe.
- Maintain backward compatibility for existing snapshots and existing follow-up action flows.
- Introduce `normalization_version` so canonicalization rules can evolve safely.
- Prepare phase 2 analytics to measure user interaction by stable topic identity.

## Non-Goals

- Add LLM-based topic extraction or enrichment.
- Add embeddings-based topic similarity.
- Introduce a persistent global topic catalog table in this phase.
- Expand the suggestion engine to notes, chat, research, or presentations yet.
- Redesign the study-suggestions UI beyond minor evidence/copy improvements.
- Replace the existing suggestion snapshot model or action model from scratch.

## Current State

### Current Topic Pipeline

The current topic pipeline is intentionally minimal and label-driven.

See:

- `tldw_Server_API/app/core/StudySuggestions/topic_pipeline.py`
- `tldw_Server_API/app/core/StudySuggestions/types.py`

Notable properties today:

- canonicalization is based on a small inline synonym map
- identity is effectively the canonical label string
- there is no stable `topic_key`
- there is no `normalization_version`
- evidence classes are limited to `grounded`, `weakly_grounded`, and `derived`

### Current Snapshot Assembly

Topic extraction logic currently lives largely in snapshot assembly.

See:

- `tldw_Server_API/app/core/StudySuggestions/snapshot_service.py`

Notable properties today:

- quiz extraction, label collection, weakness/adjacency assignment, and topic payload construction happen directly inside the snapshot path
- flashcard snapshots are assembled from shallow derived labels
- snapshot topic rows use a snapshot-local `"id"` such as `topic-1`

### Current Follow-Up Action Model

Follow-up actions currently resolve selected topics from the frozen snapshot payload using snapshot-local topic ids and labels.

See:

- `tldw_Server_API/app/core/StudySuggestions/actions.py`

Notable properties today:

- selection fingerprints are label-based
- renamed topics change the labels used downstream
- there is no stable semantic topic identifier in fingerprints or future analytics

### Current UI Contract

The UI currently expects topic rows with:

- `id`
- `display_label`
- `type`
- `status`
- `selected`

See:

- `apps/packages/ui/src/components/StudySuggestions/StudySuggestionsPanel.tsx`

This contract must continue to work for old snapshots after phase 1 lands.

## Design Principles

### 1. Dual Identity, Not Single Identity

Each topic must have two separate identities:

- `snapshot_topic_id`
  - frozen row identity local to one snapshot
  - used by the UI for selection, editing, and reset behavior
- `topic_key`
  - stable semantic identity derived from deterministic normalization
  - encoded in a namespaced format such as `<namespace>:<canonical_slug>`
  - `namespace` comes from the normalization dictionary or domain family, not from the calling service
  - default namespace is `general` when no stronger domain is available
  - used for dedupe, analytics, and future cross-service reuse

`display_label` remains user-facing presentation text and may change without changing `topic_key`.

### 2. Version Canonicalization

Every computed topic row must carry a `normalization_version`.

This version identifies the alias dictionary and canonicalization rules that produced the row. It prevents future upgrades from silently changing semantic identity without traceability.

### 3. Adapters Produce Evidence, Pipeline Produces Topics

Phase 1 should clarify ownership:

- `quiz_adapter.py`
  - emits structured quiz evidence inputs
- `flashcard_adapter.py`
  - emits structured flashcard evidence inputs
- `topic_pipeline.py`
  - cleans labels
  - resolves aliases
  - generates `topic_key`
  - merges evidence
  - assigns evidence classes
  - ranks topics
- `snapshot_service.py`
  - loads anchor data
  - invokes adapters and pipeline
  - serializes permission-safe payloads

This removes the current ambiguity where extraction and ranking logic are mixed into snapshot assembly.

### 4. Permission-Safe Frozen Snapshots

Frozen snapshots must remain lightweight and permission-safe.

Allowed in snapshot payload:

- topic ids
- topic keys
- canonical and display labels
- evidence class
- rank reason
- counts
- light source refs
- safe evidence reasons

Frozen labels must come from a strict allowlist:

- user-authored tags already visible elsewhere in product UI
- deck names, study-pack titles, workspace-visible labels
- source titles or headings that already exist as explicit metadata fields
- deterministic canonical labels produced from allowlisted labels plus alias dictionaries

If a candidate label can only be derived from restricted body text, transcript text, question text, chat content, note content, or LLM summarization of those bodies, it must stay live-only and must not be serialized into the frozen snapshot.

Not allowed in snapshot payload:

- source excerpts
- question text
- source summaries
- note body fragments
- media transcript content

Detailed evidence remains live-resolved when the snapshot is reopened.

### 5. Compatibility First

Phase 1 must not break:

- reopened legacy snapshots
- current UI rendering
- follow-up action flows for old snapshots
- duplicate detection for pre-phase-1 outputs

Compatibility behavior must be explicit rather than incidental.

## Proposed Data Model

### Topic Payload V2

Each topic row in a new snapshot payload should carry:

- `id`
  - snapshot-local identifier, for example `topic-1`
- `topic_key`
  - stable deterministic semantic key in the form `<namespace>:<canonical_slug>`
- `normalization_version`
  - deterministic canonicalization version, starting with something like `norm-v2`
- `canonical_label`
  - canonical normalized label
- `display_label`
  - user-facing label
- `type`
  - evidence class
- `status`
  - rank reason
- `selected`
  - default selection state
- `source_count`
  - number of lightweight source refs contributing to this topic
- `evidence_reasons`
  - compact structured reasons such as `["missed_question", "source_citation"]`
- optional `source_type`
- optional `source_id`

### Identity Rules

- `id` must remain snapshot-local and stable for the lifetime of that snapshot.
- `topic_key` must be stable across snapshots as long as the canonicalization rules and source semantics remain unchanged.
- `topic_key` identity is defined by the full `<namespace>:<canonical_slug>` pair plus `normalization_version`.
- service name must not be used as the namespace boundary.
- different services may reuse the same `topic_key` only when they resolve to the same namespace and canonical slug.
- `display_label` may change due to user edits or improved formatting without changing `topic_key`.

## Proposed Backend Changes

### `types.py`

Extend topic-related types to include:

- `topic_key`
- `normalization_version`
- `source_count`
- `evidence_reasons`

Add an explicit `exploratory` evidence class if needed by the serialized payload contract. If the code keeps `exploratory` only as a rank reason, the spec should still require a consistent mapping from rank/evidence to UI copy.

### `topic_pipeline.py`

Refactor the pipeline into explicit stages:

1. label cleaning
2. token and phrase normalization
3. alias resolution
4. canonical label selection
5. `topic_key` generation
6. evidence merge across sources
7. ranking

New deterministic inputs:

- alias dictionary module
- phrase-level replacements
- safe singular/plural collapsing where supported
- stopword trimming
- domain synonym groups

New deterministic outputs:

- canonical label
- topic key
- normalization version
- merged evidence metadata

The pipeline must treat namespace resolution as part of canonicalization, not as an adapter-side afterthought.

### `quiz_adapter.py`

Refactor quiz evidence extraction out of `snapshot_service.py` and into the adapter.

The adapter should emit structured inputs such as:

- source-backed labels from citations
- tag-backed labels from questions or quiz metadata
- weakness evidence from incorrect answers
- adjacency evidence from correct/covered related topics
- lightweight source refs associated with evidence

The adapter should prefer:

1. citation/source labels that resolve to allowlisted metadata fields
2. structured tags
3. deterministic derived labels only when needed

If citation-derived wording is only available from question text or explanation bodies, the adapter may use it as live evidence for ranking but must not serialize it into the frozen payload label set.

### `flashcard_review_sessions` and `flashcard_reviews`

Phase 1 must make flashcard evidence acquisition concrete instead of reconstructing everything ad hoc.

Chosen contract:

- add persisted session-level aggregates to `flashcard_review_sessions`
  - `cards_reviewed`
  - a deterministic recall-success metric, exposed as `correct_count` for compatibility if needed
  - lightweight `source_bundle_json`
  - optional lineage references such as `study_pack_id` when the deck/session came from one
- update those aggregates at review time so completed sessions can be summarized without replaying the entire review log on the hot path
- keep `flashcard_reviews.review_session_id` as the row-level source of truth for reconstruction and legacy backfill

Query strategy:

- primary path for new sessions
  - load aggregates and lightweight source refs directly from `flashcard_review_sessions`
- fallback path for legacy or partially populated sessions
  - reconstruct reviewed-card coverage from `flashcard_reviews.review_session_id`
  - hydrate card/deck/study-pack provenance live from the existing tables

This is intentionally hybrid:

- persisted session aggregates keep snapshot generation cheap and deterministic
- review-log reconstruction preserves compatibility with sessions created before the new columns exist
- card-level metadata stays recoverable without bloating the session row

If the implementation keeps the compatibility name `correct_count`, the spec should require one deterministic mapping from stored flashcard ratings into that value and use it consistently across backend and UI summaries.

### `flashcard_adapter.py`

This phase must explicitly improve flashcard evidence quality before expecting better ranking.

The adapter should consume or derive:

- real `cards_reviewed`
- real `correct_count` or similar performance signals from the persisted session aggregate or review-log reconstruction path above
- session source bundle when available
- deck provenance and study-pack lineage
- card-level source metadata when recoverable
- tag-filter context

The adapter should distinguish:

- genuinely grounded sessions
- weakly grounded tag/deck sessions
- exploratory/manual sessions

If provenance is thin, the adapter should mark the session exploratory rather than over-claiming adjacency.

### `snapshot_service.py`

Refactor snapshot service responsibilities to:

- load raw anchor data
- request structured evidence from the appropriate adapter
- invoke the topic pipeline
- serialize the V2 topic rows

The snapshot service should no longer own the majority of quiz extraction logic directly.

### `actions.py`

Update topic selection and fingerprint resolution rules:

- prefer `topic_key` when present
- fall back to legacy label normalization when it is absent
- preserve snapshot-local `id` for selection targeting

Action handling must preserve two separate concepts:

- semantic identity
  - resolved server-side from the selected snapshot rows as `selected_topic_keys`
  - used for dedupe, analytics, and refreshed-snapshot equivalence
- prompt text
  - resolved from `selected_topic_edits` plus `manual_topic_labels`
  - used for generation instructions and user-visible follow-up copy

The external request contract may continue to send `selected_topic_ids`, `selected_topic_edits`, and `manual_topic_labels`.
The server must resolve both semantic and prompt-oriented representations before dispatching generation.

New follow-up fingerprints should include:

- `snapshot_id`
- `target_service`
- `target_type`
- selected `topic_key`s when present
- normalized manual-topic labels for exploratory additions that do not have a `topic_key`
- `action_kind`
- `generator_version`
- `normalization_version`

For legacy snapshots:

- continue using normalized labels as the fallback fingerprint input

Edited labels for snapshot-backed topics must not change duplicate identity on their own. If the user wants a semantically identical selection regenerated with different phrasing, that must go through `force_regenerate`.

## UI Impact

This phase should keep the UI mostly stable.

Expected UI changes:

- compatibility mapper for topic rows that may or may not contain `topic_key`
- optional copy/badge improvements for stronger provenance explanation
- no major workflow changes

The `StudySuggestionsPanel` should continue to work with:

- old snapshots that only have `id` and `display_label`
- new snapshots that also include `topic_key`, `canonical_label`, and richer evidence metadata

## Backward Compatibility

### Legacy Snapshots

Old snapshots should continue to render without forced refresh.

Compatibility rules:

- if `topic_key` is missing, UI falls back to the old contract
- if `normalization_version` is missing, treat it as legacy
- follow-up actions should resolve selected topics from `id` and label as they do today

### Legacy Fingerprints

Do not rewrite old generation links or old fingerprints.

Instead:

- existing links remain valid
- new snapshots use the new fingerprint composition
- duplicate detection continues to work within each generation style

### Refreshed Snapshot Equivalence

Refreshing a legacy snapshot into a V2 child snapshot should prefer reuse over duplicate generation.

Chosen behavior:

- direct lookup still checks the current snapshot id plus its native fingerprint first
- if that misses and the snapshot has an ancestor chain, action resolution should perform a second equivalence lookup across the refreshed-from lineage
- that lineage lookup should compare action contract plus semantic topic identity
  - use `topic_key` when both sides have it
  - fall back to normalized legacy labels when an ancestor snapshot predates V2
- if an ancestor hit is found, return `opened_existing` instead of creating a duplicate artifact
- the system may persist a new child-snapshot alias link pointing at the reused target so later lookups stay fast

This lineage-equivalence reuse should apply only to semantic selections. Manual-only additions or `force_regenerate` requests should continue to create new outputs.

### Refresh Behavior

Refreshing an old snapshot may produce a new child snapshot with the V2 payload shape.

This is acceptable and desirable, as long as:

- the original snapshot remains unchanged
- the new snapshot explicitly carries the new normalization version
- lineage-aware duplicate lookup can still reopen an equivalent existing artifact from the ancestor chain

## Testing Strategy

Add or extend tests for:

- deterministic `topic_key` stability under raw-label drift
- namespace collision protection, for example identical slugs in different domain dictionaries
- alias collapse across source labels, tags, and derived labels
- `normalization_version` propagation into snapshot payloads and fingerprints
- legacy snapshot compatibility in follow-up action resolution
- refreshed-child lineage equivalence opening an existing ancestor artifact instead of generating a duplicate
- semantic fingerprint stability when display labels are edited but selected `topic_key`s stay constant
- flashcard provenance downgrade behavior
- quiz weakness vs adjacency ordering
- snapshot payload safety guarantees
- frozen-label allowlist enforcement
- UI compatibility with mixed old/new snapshot shapes
- fixture-based grounding audit for flashcard sessions

## Rollout Plan

Phase 1 should land in this order:

1. introduce new topic types and compatibility-safe payload schema
2. add flashcard session aggregate and provenance storage plus legacy reconstruction helpers
3. refactor adapters to emit structured evidence
4. refactor topic pipeline to emit `topic_key` and `normalization_version`
5. upgrade snapshot serialization
6. upgrade action fingerprinting with legacy and refreshed-lineage fallback
7. add UI compatibility handling
8. add targeted tests and grounding audit fixtures

This order keeps the system working at every step and avoids breaking frozen history.

## Success Criteria

Phase 1 is successful when:

- repeated semantically equivalent labels map to the same `topic_key`
- old snapshots still render and generate follow-up artifacts successfully
- new snapshots carry richer, stable topic identity
- a checked-in flashcard grounding audit fixture set exists and passes
- grounded flashcard fixtures produce at least one grounded or weakly grounded topic in the top 3
- exploratory/manual flashcard fixtures produce zero falsely grounded topics
- future analytics can key on `topic_key` instead of only display labels

## Follow-On Phases

### Phase 2

Add feedback and analytics around:

- selected topics
- ignored topics
- generated vs opened-existing follow-ups
- downstream completion outcomes

Phase 2 should use `topic_key` as the primary analytic identity.

### Phase 3

Generalize the suggestion engine for other internal services such as:

- notes
- chat
- research
- presentations

That later expansion should reuse:

- the adapter boundary
- namespaced `topic_key`
- `normalization_version`
- permission-safe frozen snapshots
