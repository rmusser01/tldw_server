# Study Suggestions Grounding And Normalization V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement deterministic `StudySuggestions` grounding/normalization v2: namespaced `topic_key`s, safe persisted V2 snapshots, richer flashcard provenance/performance inputs, and stable dedupe/reopen behavior for quiz and flashcard follow-ups.

**Architecture:** Keep the existing shared `StudySuggestions` backend domain, but split responsibilities more cleanly: adapters emit structured evidence, `topic_pipeline.py` owns canonicalization and ranking, `snapshot_service.py` orchestrates payload creation, and `actions.py` owns semantic duplicate identity plus refreshed-lineage reopen behavior. Extend `flashcard_review_sessions` with lightweight aggregates while treating `flashcard_reviews.review_session_id` rows as canonical, and preserve backward compatibility by layering V2 fields onto the current topic payload contract instead of replacing it.

**Tech Stack:** FastAPI, Pydantic, SQLite/PostgreSQL via `ChaChaNotes_DB`, pytest, Bandit, React, TanStack Query, Ant Design, Vitest

---

## Scope Lock

Keep these decisions fixed during implementation:

- deterministic only:
  - no LLM enrichment
  - no embeddings-based similarity
- no global topic catalog table in this phase
- `flashcard_reviews.review_session_id` is canonical when aggregates and reconstructed rollups disagree
- persisted snapshot payloads stay deny-by-default and permission-safe
- at most one active direct generation link exists per:
  - `snapshot_id`
  - `target_service`
  - `target_type`
  - `selection_fingerprint`
- refreshed-lineage reuse stays read-only in phase 1:
  - no persisted child alias rows in `suggestion_generation_links`
- old snapshots and old label-based fingerprints must continue to work
- UI changes stay minimal:
  - compatibility mapping
  - badge/copy improvements
  - no new workflow

## File Structure

- `tldw_Server_API/app/core/StudySuggestions/types.py`
  Purpose: add V2 topic dataclasses and internal action-resolution structures such as semantic-topic selection and richer evidence metadata.
- `tldw_Server_API/app/core/StudySuggestions/topic_aliases.py`
  Purpose: hold the deterministic alias dictionary, namespace rules, and `NORMALIZATION_VERSION` constant for V2 canonicalization.
- `tldw_Server_API/app/core/StudySuggestions/topic_pipeline.py`
  Purpose: refactor normalization into explicit V2 stages, emit namespaced `topic_key`s, and rank topics with stable evidence metadata.
- `tldw_Server_API/app/core/StudySuggestions/quiz_adapter.py`
  Purpose: emit structured quiz evidence inputs from citations, tags, and incorrect/correct question outcomes.
- `tldw_Server_API/app/core/StudySuggestions/flashcard_adapter.py`
  Purpose: emit flashcard evidence from persisted session aggregates, reconstructed review rollups, deck lineage, and citation/source metadata.
- `tldw_Server_API/app/core/StudySuggestions/snapshot_service.py`
  Purpose: orchestrate anchor reads, call adapters + pipeline, serialize V2 payloads, and keep legacy snapshot behavior intact.
- `tldw_Server_API/app/core/StudySuggestions/actions.py`
  Purpose: separate semantic identity from edited prompt text, enforce direct-link uniqueness, validate reopen targets, and walk refreshed snapshot ancestry.
- `tldw_Server_API/app/api/v1/endpoints/study_suggestions.py`
  Purpose: wire the updated action-preparation path and preserve the external request/response contract.
- `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  Purpose: extend `flashcard_review_sessions`, update suggestion payload sanitization, tighten generation-link uniqueness, and add rollup/lineage helpers.
- `tldw_Server_API/tests/StudySuggestions/test_study_suggestion_storage.py`
  Purpose: lock persisted V2 payload round-trip, sanitizer behavior, generation-link uniqueness, and refreshed-lineage reopen storage behavior.
- `tldw_Server_API/tests/StudySuggestions/test_flashcard_review_sessions.py`
  Purpose: verify aggregate writes, reconstruction fallback, and drift reconciliation for flashcard sessions.
- `tldw_Server_API/tests/StudySuggestions/test_topic_pipeline.py`
  Purpose: verify namespaced `topic_key` generation, normalization versioning, alias collapse, and namespace-collision protection.
- `tldw_Server_API/tests/StudySuggestions/test_study_suggestion_adapters.py`
  Purpose: verify quiz and flashcard evidence extraction, provenance downgrade rules, and grounded/manual session classification.
- `tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py`
  Purpose: verify action dedupe, `force_regenerate`, stale-target rejection, and legacy snapshot compatibility.
- `tldw_Server_API/tests/StudySuggestions/fixtures/flashcard_grounding_audit_cases.json`
  Purpose: fixture-based audit cases for grounded vs exploratory flashcard sessions.
- `apps/packages/ui/src/services/studySuggestions.ts`
  Purpose: extend frontend topic types with optional V2 fields while preserving the current request contract.
- `apps/packages/ui/src/components/StudySuggestions/hooks/useStudySuggestions.ts`
  Purpose: keep action payload construction stable while accepting richer snapshot payloads.
- `apps/packages/ui/src/components/StudySuggestions/StudySuggestionsPanel.tsx`
  Purpose: map V2 topic rows into the existing editor state, preserve snapshot-local IDs, and show minimal evidence-copy improvements.
- `apps/packages/ui/src/components/StudySuggestions/hooks/__tests__/useStudySuggestions.test.tsx`
  Purpose: verify the unchanged action request shape and response handling with richer snapshots.
- `apps/packages/ui/src/components/StudySuggestions/components/__tests__/StudySuggestionsPanel.test.tsx`
  Purpose: verify mixed legacy/V2 snapshots, topic-key-aware rendering, and no regression in manual-topic handling.

## Task 1: Persist V2 Snapshot Fields Safely

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/core/StudySuggestions/types.py`
- Modify: `tldw_Server_API/tests/StudySuggestions/test_study_suggestion_storage.py`
- Modify: `tldw_Server_API/tests/StudySuggestions/test_study_suggestion_schemas.py`

- [ ] **Step 1: Write the failing storage tests**

Add assertions like:

```python
payload = {
    "topics": [
        {
            "id": "topic-1",
            "topic_key": "renal:renal-physiology",
            "normalization_version": "norm-v2",
            "display_label": "Renal Physiology",
            "canonical_label": "renal physiology",
            "evidence_reasons": ["missed_question"],
        }
    ]
}
```

Verify that `create_suggestion_snapshot(...)` + `get_suggestion_snapshot(...)` preserves `topic_key`, `normalization_version`, and `evidence_reasons`, while still dropping unsafe text keys.

- [ ] **Step 2: Run the targeted backend tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_storage.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_schemas.py -v
```

- [ ] **Step 3: Implement the minimal persistence contract**

Make these changes:

- extend the safe suggestion-payload allowlist in `ChaChaNotes_DB.py`
- keep the sanitizer deny-by-default for unknown keys
- extend internal StudySuggestions types so V2 topic rows have explicit fields for:
  - `topic_key`
  - `normalization_version`
  - `source_count`
  - `evidence_reasons`

- [ ] **Step 4: Re-run the targeted tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_storage.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_schemas.py -v
```

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/app/core/StudySuggestions/types.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_storage.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_schemas.py
git commit -m "feat: persist study suggestion v2 payload fields"
```

## Task 2: Add Flashcard Session Aggregates And Reconciliation

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/tests/StudySuggestions/test_flashcard_review_sessions.py`
- Modify: `tldw_Server_API/tests/StudySuggestions/test_study_suggestion_storage.py`

- [ ] **Step 1: Write the failing session-rollup tests**

Cover:

- `review_flashcard(...)` updates `cards_reviewed` and compatibility `correct_count`
- legacy sessions without aggregate fields still reconstruct from `flashcard_reviews.review_session_id`
- impossible aggregates such as `correct_count > cards_reviewed` fall back to reconstruction
- reconciliation prefers reconstructed values and may repair the aggregate row

Example assertion:

```python
rollup = db.get_flashcard_review_session_rollup(session_id)
assert rollup["cards_reviewed"] == 3
assert rollup["aggregate_source"] in {"session", "reconstructed"}
```

- [ ] **Step 2: Run the targeted backend tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/StudySuggestions/test_flashcard_review_sessions.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_storage.py -v
```

- [ ] **Step 3: Implement aggregate writes and reconciliation helpers**

Make these changes:

- add `flashcard_review_sessions` columns for:
  - `cards_reviewed`
  - compatibility `correct_count`
  - `source_bundle_json`
  - optional `study_pack_id`
- update `review_flashcard(...)` to write review rows and aggregate counters in the same transaction
- add one DB helper that returns the merged rollup used by suggestions and encapsulates:
  - primary aggregate read
  - fallback reconstruction from `flashcard_reviews`
  - aggregate repair when cached values drift

- [ ] **Step 4: Re-run the targeted tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/StudySuggestions/test_flashcard_review_sessions.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_storage.py -v
```

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/tests/StudySuggestions/test_flashcard_review_sessions.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_storage.py
git commit -m "feat: add flashcard session rollups for study suggestions"
```

## Task 3: Build The V2 Topic Pipeline

**Files:**
- Create: `tldw_Server_API/app/core/StudySuggestions/topic_aliases.py`
- Modify: `tldw_Server_API/app/core/StudySuggestions/types.py`
- Modify: `tldw_Server_API/app/core/StudySuggestions/topic_pipeline.py`
- Modify: `tldw_Server_API/tests/StudySuggestions/test_topic_pipeline.py`

- [ ] **Step 1: Write the failing normalization tests**

Cover:

- semantically equivalent labels collapse to one namespaced `topic_key`
- different namespaces can reuse the same slug without collision
- `normalization_version` is attached to every ranked topic
- `display_label` drift does not alter semantic identity

Example assertion:

```python
assert ranked[0].topic_key == "renal:renal-physiology"
assert ranked[0].normalization_version == "norm-v2"
```

- [ ] **Step 2: Run the targeted backend tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/StudySuggestions/test_topic_pipeline.py -v
```

- [ ] **Step 3: Implement the deterministic V2 pipeline**

Make these changes:

- move alias and namespace rules into `topic_aliases.py`
- split `topic_pipeline.py` into explicit stages:
  - clean label
  - phrase/token normalization
  - alias resolution
  - canonical label selection
  - namespace + `topic_key` generation
  - evidence merge
  - ranking
- extend `RankedTopic`/`TopicCandidate` dataclasses with:
  - `topic_key`
  - `normalization_version`
  - `source_count`
  - `evidence_reasons`

- [ ] **Step 4: Re-run the targeted tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/StudySuggestions/test_topic_pipeline.py -v
```

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/StudySuggestions/topic_aliases.py \
  tldw_Server_API/app/core/StudySuggestions/types.py \
  tldw_Server_API/app/core/StudySuggestions/topic_pipeline.py \
  tldw_Server_API/tests/StudySuggestions/test_topic_pipeline.py
git commit -m "feat: add study suggestion topic normalization v2"
```

## Task 4: Refactor Adapters And Snapshot Serialization

**Files:**
- Modify: `tldw_Server_API/app/core/StudySuggestions/quiz_adapter.py`
- Modify: `tldw_Server_API/app/core/StudySuggestions/flashcard_adapter.py`
- Modify: `tldw_Server_API/app/core/StudySuggestions/snapshot_service.py`
- Modify: `tldw_Server_API/tests/StudySuggestions/test_study_suggestion_adapters.py`
- Modify: `tldw_Server_API/tests/StudySuggestions/test_study_suggestion_storage.py`
- Create: `tldw_Server_API/tests/StudySuggestions/fixtures/flashcard_grounding_audit_cases.json`

- [ ] **Step 1: Write the failing adapter/snapshot tests**

Cover:

- quiz snapshots serialize `topic_key`, `normalization_version`, `canonical_label`, and `evidence_reasons`
- flashcard snapshots consume merged session rollups rather than hardcoded zeros
- grounded flashcard fixtures produce grounded or weakly grounded topics in the top 3
- exploratory/manual fixtures do not produce falsely grounded topics

- [ ] **Step 2: Run the targeted backend tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_adapters.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_storage.py -v
```

- [ ] **Step 3: Implement adapter-owned evidence extraction**

Make these changes:

- move quiz extraction details out of `snapshot_service.py` and into `quiz_adapter.py`
- have `flashcard_adapter.py` consume the new merged session rollup and card/deck/study-pack provenance
- update `snapshot_service.py` to serialize V2 topic rows while keeping legacy-safe snapshot shapes readable
- keep frozen payloads lightweight:
  - no excerpts
  - no question text
  - only allowlisted labels and light refs

- [ ] **Step 4: Re-run the targeted tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_adapters.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_storage.py -v
```

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/StudySuggestions/quiz_adapter.py \
  tldw_Server_API/app/core/StudySuggestions/flashcard_adapter.py \
  tldw_Server_API/app/core/StudySuggestions/snapshot_service.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_adapters.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_storage.py \
  tldw_Server_API/tests/StudySuggestions/fixtures/flashcard_grounding_audit_cases.json
git commit -m "feat: serialize study suggestion snapshots with v2 grounding"
```

## Task 5: Rework Action Semantics And Refreshed-Lineage Reopen

**Files:**
- Modify: `tldw_Server_API/app/core/StudySuggestions/actions.py`
- Modify: `tldw_Server_API/app/api/v1/endpoints/study_suggestions.py`
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/tests/StudySuggestions/test_study_suggestion_storage.py`
- Modify: `tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py`

- [ ] **Step 1: Write the failing action/dedupe tests**

Cover:

- semantic identity is derived from selected snapshot rows and not from edited display labels
- `force_regenerate` replaces the active direct link for a snapshot/fingerprint instead of leaving multiple active rows
- refreshed child snapshots reopen validated ancestor artifacts
- stale/deleted ancestor targets are ignored instead of reopening broken links

Example assertion:

```python
assert response["disposition"] == "opened_existing"
assert response["target_id"] == str(existing_quiz_id)
```

- [ ] **Step 2: Run the targeted backend tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_storage.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py -v
```

- [ ] **Step 3: Implement the action-resolution changes**

Make these changes:

- split internal action resolution into:
  - semantic keys used for fingerprints/dedupe
  - prompt labels used for generation instructions
- tighten DB uniqueness for active direct generation links so `target_id` is not part of the canonical duplicate identity
- walk only the `refreshed_from_snapshot_id` ancestor chain for reopen lookup
- validate target liveness before returning `opened_existing`
- keep refreshed-lineage reuse read-only:
  - do not write child alias rows in phase 1

- [ ] **Step 4: Re-run the targeted tests**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_storage.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py -v
```

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/StudySuggestions/actions.py \
  tldw_Server_API/app/api/v1/endpoints/study_suggestions.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_storage.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py
git commit -m "feat: stabilize study suggestion action dedupe semantics"
```

## Task 6: Update Frontend Compatibility For V2 Topics

**Files:**
- Modify: `apps/packages/ui/src/services/studySuggestions.ts`
- Modify: `apps/packages/ui/src/components/StudySuggestions/hooks/useStudySuggestions.ts`
- Modify: `apps/packages/ui/src/components/StudySuggestions/StudySuggestionsPanel.tsx`
- Modify: `apps/packages/ui/src/components/StudySuggestions/hooks/__tests__/useStudySuggestions.test.tsx`
- Modify: `apps/packages/ui/src/components/StudySuggestions/components/__tests__/StudySuggestionsPanel.test.tsx`

- [ ] **Step 1: Write the failing frontend tests**

Cover:

- V2 snapshot topics with `topic_key`, `canonical_label`, and `evidence_reasons` still render through the current panel
- legacy snapshot topics still work unchanged
- action requests still submit `selected_topic_ids`, `selected_topic_edits`, and `manual_topic_labels`
- edited labels remain user-visible without changing selection IDs

- [ ] **Step 2: Run the targeted frontend tests to verify they fail**

Run:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui && bunx vitest run \
  src/components/StudySuggestions/hooks/__tests__/useStudySuggestions.test.tsx \
  src/components/StudySuggestions/components/__tests__/StudySuggestionsPanel.test.tsx
```

- [ ] **Step 3: Implement the minimal compatibility pass**

Make these changes:

- extend `StudySuggestionSnapshotTopic` with optional V2 fields
- keep request payload shape unchanged
- map V2 topic rows into the existing `TopicBuilderTopic` model by preserving:
  - snapshot-local `id`
  - editable `display_label`
  - optional provenance/evidence copy
- do not add new UI workflow elements in this task

- [ ] **Step 4: Re-run the targeted frontend tests**

Run:

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui && bunx vitest run \
  src/components/StudySuggestions/hooks/__tests__/useStudySuggestions.test.tsx \
  src/components/StudySuggestions/components/__tests__/StudySuggestionsPanel.test.tsx
```

- [ ] **Step 5: Commit**

```bash
git add \
  apps/packages/ui/src/services/studySuggestions.ts \
  apps/packages/ui/src/components/StudySuggestions/hooks/useStudySuggestions.ts \
  apps/packages/ui/src/components/StudySuggestions/StudySuggestionsPanel.tsx \
  apps/packages/ui/src/components/StudySuggestions/hooks/__tests__/useStudySuggestions.test.tsx \
  apps/packages/ui/src/components/StudySuggestions/components/__tests__/StudySuggestionsPanel.test.tsx
git commit -m "feat: add study suggestion v2 frontend compatibility"
```

## Task 7: Final Verification

**Files:**
- Modify: `Docs/superpowers/specs/2026-04-08-study-suggestions-grounding-normalization-v2-design.md` only if implementation requires an approved spec correction

- [ ] **Step 1: Run the full touched backend suite**

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/StudySuggestions/test_topic_pipeline.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_adapters.py \
  tldw_Server_API/tests/StudySuggestions/test_flashcard_review_sessions.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_storage.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestion_schemas.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestions_endpoints_api.py \
  tldw_Server_API/tests/StudySuggestions/test_study_suggestions_jobs_worker.py -v
```

- [ ] **Step 2: Run the touched frontend tests**

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui && bunx vitest run \
  src/components/StudySuggestions/hooks/__tests__/useStudySuggestions.test.tsx \
  src/components/StudySuggestions/components/__tests__/StudySuggestionsPanel.test.tsx
```

- [ ] **Step 3: Run security and diff hygiene checks**

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/StudySuggestions \
  tldw_Server_API/app/api/v1/endpoints/study_suggestions.py \
  tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py \
  -f json -o /tmp/bandit_study_suggestions_grounding_v2.json
git diff --check
```

- [ ] **Step 4: Commit the final integration pass if needed**

```bash
git add <touched files>
git commit -m "fix: finalize study suggestion grounding normalization v2"
```

