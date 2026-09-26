---
id: TASK-13245
title: Persist Workspace Persona opt-out and conversation provenance
status: In Progress
assignee: []
created_date: '2026-09-13 18:15'
updated_date: '2026-09-14 00:09'
labels:
  - persona
  - workspaces
  - parity
dependencies:
  - TASK-13244
references:
  - 'https://github.com/rmusser01/tldw_server/issues/2950'
documentation:
  - >-
    Docs/superpowers/plans/2026-09-13-persona-workspace-parity-implementation-plan.md
  - Docs/Design/2026-09-13-persona-workspace-choice-provenance-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Distinguish unset defaults from an explicit None choice, preserve creation-time assistant provenance, and define opt-in server resolution without changing legacy chat-create behavior. Issue 2950 Stage 2.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Implement the linked stage contract with focused regression coverage and recorded verification; do not claim completion from documentation alone.
- [ ] #2 Explicit None survives restart and provisioning; opt-in inheritance persists bounded provenance atomically; stale versions and accepted retries cannot change identity.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stage 2 contract design underway in child TASK-13245.1; runtime not started. Corrected prior assessment: existing resolve_new_conversation_assistant already handles omitted identity inheritance and explicit-null opt-out;9 existing startup tests pass. Workspace chat create bypasses Sync v2; strict replay needs a DB-owned atomic path. Proposed design preserves legacy semantics and keeps Workspace Sync/Buddy/UI out of scope.

Stage 2 prompt/memory baseline: 13 passed, 9 failed with HTTP 503 missing_provider_credentials before mocked dispatch. Persona fixture patches chat.API_KEYS but runtime now uses provider_credential_runtime.load_server_config_snapshot. Diagnostic-only in-memory fixture supplying the same dummy credential via that loader yielded 22 passed. No runtime/test edits made. Repair the existing fixture and rerun ordinary suites during implementation; tracked in the staged plan and design validation record.

Design prerequisite TASK-13245.1 completed and published as draft PR #2958 (stacked on #2957). Independent review findings resolved at contract level. Parent remains In Progress: all runtime acceptance criteria are unchecked; requester approval is needed before slice 2A implementation.

Requester-requested second design review completed in TASK-13245.1/PR #2958. Verified/amended silent strict-selector downgrade on older servers (dedicated route), mixed-version cached-writer hazard (offline migration), lifecycle activation ordering, and unbounded permanent receipts (finite owner-scoped budget including tombstones). Independent re-review found no remaining material contract issues. Runtime untouched; revised contract and implementation verification gates still apply.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
