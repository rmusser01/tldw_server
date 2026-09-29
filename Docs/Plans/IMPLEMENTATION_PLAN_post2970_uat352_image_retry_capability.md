# UAT352: refresh image capability on explicit Retry

Task: TASK13260.277.2. Bounded design approved by root on 2026-09-20.

## Diagnosis and design

The model factory reads TldwModelsService's persisted 15-minute cache. Retry
reconstructs its client from the same negative result even when current server
metadata confirms vision. Ordinary forced refresh also has a 30-second cooldown.
Backend optional llama.cpp properties discovery separately caches misses for 30
seconds; a current negative remains a legitimate refusal.

Explicit failed-turn Retry with actual outgoing image parts requests fresh model
capabilities when cached vision is unconfirmed. This intent is separate from the
backend retry flag (false for a never-dispatched local refusal). Fresh discovery
bypasses cached data/cooldown and older in-flight reads, coalesces fresh reads,
and retains old durable catalog records on failure. Existing image and request
scope guards remain authoritative. Text-only/OCR requests do not force discovery.

## Stage 1: causal tests
**Goal**: Reproduce stale negative Retry using real cache/factory and pipeline.
**Success Criteria**: Tests fail for absent freshness, including forced cooldown,
late fetch, and local-refusal Retry identity.
**Tests**: TldwModels cache, factory image recovery, saved normal Chat integration.
**Status**: Complete

## Stage 2: bounded implementation
**Goal**: Add explicit fresh catalog read and image Retry intent.
**Success Criteria**: Positive recovery sends original bytes/IDs; current false,
missing model, outage, cancellation and replaced owner fail closed.
**Tests**: Focused suites and surrounding Chat behavior.
**Status**: Complete

## Stage 3: review and verification
**Goal**: Independent review, scoped type/lint and parent native handoff.
**Success Criteria**: Causal tests pass; no new compiler/lint diagnostics; review
findings resolved. Bandit is inapplicable to TypeScript-only changes.
**Tests**: Recorded final suite commands and logs; native SQLite/Postgres remains
root-owned, with no edits to frozen profiles or shared services.
**Status**: In Progress

## Verification and review record

- Initial causal run: 11 failures / 136 passes across real cache, factory and
  saved Chat tests (`/private/tmp/uat352-red.log`). Freshness repaired those
  failures. The saved-image fixture was corrected to contain a valid PNG so the
  existing MIME detector did not legitimately normalize it to JPEG.
- Held capability reads: Stop, A-to-B and A-to-B-to-A cannot dispatch after
  cancellation. The final harness uses real local abort-controller state.
  Removing only the post-refresh cancellation guard via a temporary Vite
  transform reproduces all three failures
  (`/private/tmp/uat352-final-cancel-mutation-red.log`).
- Provider review: explicit/default provider plus unqualified duplicate IDs
  reproduced two fresh-positive failures, then root review reproduced two
  foreign cached-positive failures. Both initial and refreshed entries now
  match the normalized provider actually used for dispatch. Qualified provider
  precedence remains covered. Existing incomplete mock records now include the
  required model ID/provider. The unqualified routing control uses plain text;
  it previously borrowed llama vision while asserting OpenAI dispatch.
- Fresh cache invalidation during a pending read and a pending durable write
  each reproduced a failure; obsolete results now return no capabilities.
  Fresh discovery failure retains
  the previous durable catalog, and concurrent fresh requests coalesce.
- Final related suites: 15 suites / 258 tests pass
  (`/private/tmp/uat352-final-suites.log`). Scope includes all model tests, model
  cache and wrapper caches, saved Chat, pipeline abort/error/conversation cases,
  and provider resolution.
- ESLint: zero errors, eight existing warnings, zero additions
  (`/private/tmp/uat352-lint-comparison.json`). Scoped diff whitespace is clean.
- Final TypeScript comparison: 93 baseline / 93 current / zero new or touched
  diagnostics (`/private/tmp/uat352-type-comparison.json`). This uses the same
  compiler/configuration with read-only HEAD source overlays for the three
  production files; no checkout changes were made.
- Bandit is inapplicable to this TypeScript-only patch. No Python was changed.
- Exact owned-file manifest: `/private/tmp/uat352-owned-files.txt` (10 files).
  Root owns independent final review and native SQLite/PostgreSQL acceptance.

## Native acceptance boundary

Use a fresh integrated candidate without changing frozen evidence profiles.
Retain a genuine image user turn under a discovery outage. After restoring the
provider, prove that current metadata for the same provider/model reports vision
before Retry; an optional backend properties miss can remain unconfirmed for
30 seconds and should still be refused. Retry without page reload or model
change, verify exact image bytes and client/conversation identity, and reload to
confirm one canonical user/assistant result. Also exercise a never-dispatched
local refusal (server Retry remains false), a dispatched failed turn (server
Retry stays true), a genuine current negative and a continuing discovery outage.
No backend, native browser, frozen profile or shared-service changes were made
by this implementation task.
