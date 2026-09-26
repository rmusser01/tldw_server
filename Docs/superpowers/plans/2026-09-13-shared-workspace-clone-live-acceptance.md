# Shared Workspace Clone Live Acceptance Plan

**Goal:** Certify TASK-12020.50 with real backend, Jobs, auth and WebUI/CDP evidence.
**Architecture:** Run the unchanged production FastAPI app and clone worker against
isolated databases. Use owner/recipient/outsider accounts and public APIs for normal
flows; label controlled fault injection separately. Do not mock acceptance results.
**Tech Stack:** FastAPI, SQLite, Jobs WorkerSDK, Next.js, Playwright over CDP.
**Spec:** ../specs/2026-08-25-shared-workspace-clone-jobs-design.md

## Stage 1: Isolated Runtime
**Goal:** Start a real backend and current WebUI without touching user data.
**Success Criteria:** Health/auth endpoints work; owner, recipient and outsider are
distinct; runtime paths and exact revision/diff fingerprints are recorded.
**Tests:** Health and authenticated profile reads; confirm configured DB locations.
**Status:** Complete

## Stage 2: Research Copy
**Goal:** Ingest uniquely identifiable text and clone through `/shared`.
**Success Criteria:** No enqueue-time success; real worker completion; source content
and notes survive; exact target opens; text retrieval/citations and vector readiness
are verified through APIs rather than inferred from copy labels.
**Tests:** CDP screenshots/network evidence, recipient workspace/source reads,
grounded retrieval, and owner-source independence.
**Status:** In Progress

## Stage 3: Recovery and Boundaries
**Goal:** Exercise same-key replay, concurrent tabs, lost responses, archival,
owner isolation, revocation, fatal/interrupted work, retry and cleanup.
**Success Criteria:** One logical copy per key; no unrelated-user visibility;
staged targets remain hidden; failures are bounded; cleanup is observable.
**Tests:** Real HTTP and browser scenarios; controlled queue/worker intervention
only inside the isolated environment, explicitly identified in the matrix.
**Status:** Not Started

## Stage 4: Evidence and Review
**Goal:** Produce a durable acceptance matrix and preserve honest limitations.
**Success Criteria:** Every AC maps to evidence or an explicit unresolved blocker;
no fixture result is described as live; new fixes have focused regressions/review.
**Tests:** Changed-scope lint/tests/Bandit where applicable, diff check, matrix
cross-check, cleanup of owned processes and credentials.
**Status:** Not Started

## Constraints
- Use CDP, never computer control; do not cancel CI or unrelated processes.
- Preserve TASK-12020.49 uncommitted work and existing route ownership.
- Local/self-hosted data and credentials remain outside tracked artifacts.
- Vector generation is TASK-12020.45; explicit needs_indexing/not_configured is valid.
- No merge/PR is implied by this acceptance task.

## Live Findings And Checkpoint

See `../../Development/shared-workspace-clone-live-acceptance-2026-09-13.md` for
the current acceptance matrix. Live validation found and fixed non-ISO SQLite
operation timestamps, missing FTS publication and the fresh-role permission/UI
capability mismatch. Live revoke/re-grant recovery now passes. Target-page
inspection found clone-UUID initialization and operator-only health probe
failures. The JWT health-probe failure is now repaired and live CDP confirms
Connected without operator permissions. Explicit clone-target hydration remains
unresolved and blocks UI chat acceptance. PostgreSQL was unavailable. Do not mark
the task complete or equate mocked integration coverage with live fault scenarios.

## Target-Opening Design Review

All five identified design findings are incorporated in
[the revised opening spec](../specs/2026-09-13-owned-workspace-opening-design.md)
and [its implementation plan](2026-09-13-owned-workspace-opening.md): read-only
readiness, atomic draft-preserving activation, canonical workspace notes,
account/server scope throughout activation and persistence, and complete typed
loading with explicit failures. The read-only loader checkpoint is implemented
and tested; scoped activation and route wiring remain pending. Acceptance Stage 2
remains blocked on the integrated repair and subsequent live verification.
