# Knowledge follow-ups — TASK13511–13514

## Stage 1: CI diagnosis
**Goal**: Trace the offline email failure and validate the existing repair.
**Success Criteria**: Actual forbidden caller identified without weakening the tripwire; existing PR ownership documented.
**Tests**: Same email cases under SQLite and blocked PostgreSQL; existing accounting boundary isolation.
**Status**: Complete

## Stage 2: Complete canonical-note source snapshots
**Goal**: Research handoffs retain complete available Notes content and original source version.
**Success Criteria**: Exact owner-scoped reads, honest labels, reusable successful snapshots and preserved lifecycle fences.
**Tests**: Mounted import regressions and provenance round-trip checks; both client type checks.
**Status**: Complete

Final proof includes 152 affected frontend tests, exact canonical-note snapshots, server readiness, overlapping source IDs and real canonical save/readback.

## Stage 3: Browser accessibility and real-provider validation
**Goal**: Extend workflow evidence with available native browser controls and live generation.
**Success Criteria**: Retained credential-free results, exact source mapping, honest device/participant coverage and task protocol.
**Tests**: Available Safari/native extension controls, browser keyboard checks, curated retrieval/generation cases and latency observations.
**Status**: In Progress

## Stage 4: Provenance compatibility and closeout
**Goal**: Make the coordinated backend provenance design reviewable and publish verified follow-up changes.
**Success Criteria**: Notes/Sync compatibility choices documented, remaining required inputs explicit, tests/hooks/security checks complete.
**Tests**: Review against ADR031 and canonical Notes contracts; touched-scope validation.
**Status**: In Progress

The coordinated Notes/Sync proposal is reviewable. Publish the bounded fixes as a draft PR; independent backend provenance, real device/participant testing and the existing Email CI integration remain separately open. See Docs/Reviews/KNOWLEDGE_FOLLOWUP_RESULTS_2026_10_06.md for evidence and limits.
