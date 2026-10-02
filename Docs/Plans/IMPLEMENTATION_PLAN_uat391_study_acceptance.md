# UAT391 — exact Note-to-study acceptance

Task: TASK-13260.278.4. Approved scope: RELEASE_UAT_PLAYBOOK A-08 creation/reload, A-10 source handoff/generation/edit/save/reload, A-11 five Easy reviews/persisted scheduling/analytics. This bounded harness repair does not certify the remaining independent recovery/control cases.

## Stage 1: Trace the real handoff and contracts
**Goal**: Identify causal coverage gaps and existing UI/API paths.
**Success Criteria**: Use saved Note identity, source handoff, deck, generated draft/save responses, review preview/session/analytics contracts.
**Tests**: Read current UI handlers, API schemas and existing acceptance fixtures.
**Status**: Complete

## Stage 2: Exact semantic and cardinality oracle
**Goal**: Reject partial, duplicate, mismatched and unsupported F-BIOLOGY card sets.
**Success Criteria**: Explicit supported question/answer forms cover each of the five facts exactly once; unsupported wording fails for inspection rather than passing on keywords.
**Tests**: Focused Vitest unit tests with valid cards and negative mutations.
**Status**: Complete

## Stage 3: Real UI causal journey
**Goal**: Replace skip/no-op/paste/global counts with the Note handoff, five generated/saved cards and five distinct reviews.
**Success Criteria**: Canonical Note/deck/card/session IDs, edited content, source links, preview-to-schedule agreement, deck-scoped counts/analytics persist after reload; no browser API fulfillment.
**Tests**: Playwright discovery, ESLint, TypeScript; parent-owned native deterministic and live-provider runs on SQLite and official PostgreSQL.
**Status**: Complete

## Stage 4: Verify and report limits
**Goal**: Record actual results and remaining native obligations.
**Success Criteria**: Focused checks pass and native evidence is reported honestly.
**Tests**: Focused oracle tests, lint/types, Bandit applicability and native gate handoff.
**Status**: In Progress


## Verification and native obligations

- The oracle tests use fixed independent QA fixtures and mutations for 0/1/4/6 cards, duplicated facts, contradictory/unsupported answers, wrong subject, swapped answers and numeric punctuation. Unknown paraphrases fail for inspection rather than being certified by matching keywords.
- The journey retains `notes-study-lineage.json` on success or failure with original generation, canonical Note/deck/card/session IDs, edited draft, requests/responses, scheduling, and before/after analytics. Records remain for the parent-owned disposable runtime/fixture cleanup after evidence collection.
- Run with `TLDW_LIVE_TIER_UAT=1`, `TLDW_WEB_AUTOSTART=false`, owned `TLDW_WEB_URL`, `TLDW_SERVER_URL`, and `TLDW_API_KEY`; optional `UAT_STUDY_PROVIDER`/`UAT_STUDY_MODEL` explicitly select the provider. Run the journeys project, this spec only, workers=1 and retries=0. The fixture currently authenticates using single-user/API-key setup; this is not multi-user authentication coverage.
- Run separately against a controlled provider behind the real generation backend and a configured real text provider, with owned SQLite and officially provisioned PostgreSQL. Do not substitute browser fulfillment or seed cards for generation. Retain first live output/failure and report wording outside the frozen semantic rubric for inspection.
- New deck default scheduling is recorded in evidence. Easy uses API rating5 (UI shortcut4), server-authoritative preview days, exactly one repetition/event per card, and due minus server review time with a two-second serialization tolerance. Practice, re-rate, true lapses, alternate scheduler settings, interrupted batch saves, imports/export and deck maintenance are independent A-10/A-11 cases, not certified by this narrow linked journey.
- Root owns the final combined TypeScript check and native integrated runtime results. Bandit on the touched e2e directories reports zero Python LOC, zero findings/errors; this is applicability evidence, not TypeScript security analysis.

### Final scoped checks (2026-09-21)

- Focused Vitest: 16/16 pass. ESLint on all three changed TypeScript files: exit0. Playwright `--project=journeys --list`: one discoverable linked study test, exit0. Existing Node localStorage/module-register warnings remain tooling warnings.
- Initial full frontend TypeScript run found the repaired local page-object method typo and two concurrent root live-tier argument diagnostics. Final combined type verification is deferred to the parent as requested while shared harness files change.
- No browser/native integrated run was performed by this repair agent. Native SQLite/PostgreSQL and controlled/live generation acceptance remain pending; no AC/native-pass claim is made.
- Review-session lifecycle was checked: `useFlashcardReviewRun.create` initializes a null session ID; cleanup does not call End for a null ID; submit records the ID from the first review response. Backend `review_flashcard` creates the session only while processing a rating. Following the source Note link before the first rating therefore does not create an empty session; exactly one completed persisted session remains the expected result.
