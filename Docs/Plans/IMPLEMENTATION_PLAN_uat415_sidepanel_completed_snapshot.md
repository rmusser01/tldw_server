# UAT415 completed sidepanel reply restoration

Task: TASK13260.278.13. Approved repair scope follows the user's ongoing UAT repair instruction.

## Stage 1: Reproduce the stale saved tab
**Goal:** Reproduce the native completed-reply/stale-cursor mismatch in both shipped route callers.
**Success Criteria:** A same-owner saved reply with the exact local message ID/parent and matching streamed prefix replaces the older tab snapshot; current implementation fails.
**Tests:** Existing real-store sidepanel ownership harness, shared and legacy routes.
**Status:** Complete

## Stage 2: Recover the completed local reply
**Goal:** Reuse the owned local message database during saved-tab restoration.
**Success Criteria:** Recover only unacknowledged assistant snapshots ending in the existing streaming cursor when a completed row has the same ID, parent and prefix. Preserve genuinely interrupted output, edited messages, settings, history, drafts and queue. Reject foreign/unowned/mismatched history and cancelled reads. Do not add requests to the provider or server.
**Tests:** Same-ID positive controls; missing/foreign/changed-prefix/changed-parent/pending/error controls and account/tab/unmount invalidation.
**Status:** Complete

## Stage 3: Verify and accept
**Goal:** Verify the narrow repair against retained native SQLite/PostgreSQL evidence.
**Success Criteria:** Relevant route/resume/queue tests and scoped lint/type comparisons pass; a new immutable extension recovers both actual completed replies and preserves unsaved interrupted output and account boundaries. Record exact provenance; original evidence stays unchanged.
**Tests:** Focused Vitest, ESLint, matched TypeScript; native packaged extension and canonical ID readback. Bandit does not parse TypeScript and is inapplicable to this TS-only repair.
**Status:** In Progress

Verification: two causal failures/46controls, then109passes/4suites. Scoped ESLint0; shared UI354matched baseline/current with0added/removed/touched diagnostics; frontend TypeScript exit0. Root review found no additional blocker; independent agent capacity unavailable. Native frozen package acceptance remains outstanding.
