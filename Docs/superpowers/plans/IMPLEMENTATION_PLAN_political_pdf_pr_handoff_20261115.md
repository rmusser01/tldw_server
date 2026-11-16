# Political PDF PR handoff

Tasks: TASK-15505.8 (integration/delivery), TASK-15505.9 (legacy compatibility). User requested continued work and a PR against dev.

ADR assessment: no new durable rule. Retain ADR-003/007 Jobs and Workspace, ADR-008 cache persistence, ADR-022/060 embedding ownership, ADR-025 provider gateway, ADR-049 chat ownership and ADR-065 Notes provenance. Respect the current core-to-API boundary and preserve API schema re-exports.

## Stage 1: Qualify the full PR scope
**Goal:** Check all earlier and follow-through commits, publication contents and current dev integration.
**Success Criteria:** Independent scope/security checks, broad frontend/backend outcomes and remaining failures are explicit.
**Tests:** Full touched-scope Bandit; connected UI/Python suites; publication secret scan; task-ID union; nonmutating merge simulation.
**Status:** Complete

## Stage 2: Repair compatibility and integrate dev
**Goal:** Preserve valid old snapshots and upstream refactors without weakening ownership or inventing original metadata.
**Success Criteria:** Original fidelity test remains intact and passes; missing-ID aliases/parents remain safe; actual legacy UI path has an honest outcome for missing timestamps. Token-safe embedding callers use dev's core owner. Core models no longer import API schemas; API imports remain compatible.
**Tests:** Regression-first legacy remap/service tests, full workspace suite; changed Media/Embeddings callers and layering/endpoint contracts; TypeScript, focused lint/Bandit and independent review.
**Status:** In Progress

## Stage 3: Publish draft and record checks
**Goal:** Open the authorized PR against dev with a concrete review packet and honest acceptance limits.
**Success Criteria:** Branch pushed without force, PR attached, checks inspected, task records linked. Human Change summary and semantic labels remain explicitly pending; no merge or deployment.
**Tests:** Final diff/task/secret checks, GitHub base/head/draft verification and initial CI status.
**Status:** Not Started

Remove only this plan after completion; preserve external acceptance work in TASK-15505.6.
