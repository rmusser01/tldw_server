---
id: TASK-12018
title: Plan RPG campaign/session runtime implementation
status: Done
created_date: 2026-06-25 02:30
labels:
- planning
- rpg
- ttrpg
- backend
references:
- TASK-12017
documentation:
- Docs/superpowers/specs/2026-06-25-rpg-campaign-session-runtime-design.md
- Docs/superpowers/plans/2026-06-25-rpg-campaign-session-runtime-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-06-25-rpg-campaign-session-runtime-implementation-plan.md
updated_date: 2026-06-25 03:11
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the implementation plan for the approved RPG/TTRPG campaign/session runtime design. Scope is planning only: translate the approved spec into staged TDD tasks with file responsibilities, verification commands, commit checkpoints, and handoff options. No implementation code in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Implementation plan is written to Docs/superpowers/plans/2026-06-25-rpg-campaign-session-runtime-implementation-plan.md
- [x] #2 Plan maps approved spec to reviewable TDD tasks covering core ledger, adapters, REST, proposals/rules lookup, MCP tools, tests, permissions, and verification
- [x] #3 Plan self-review finds no placeholders, missing spec requirements, or inconsistent type/function names
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect adjacent module patterns needed for exact file/task planning.
2. Write the implementation plan document from the approved spec.
3. Self-review the plan against the spec and placeholder/type consistency checks.
4. Record verification and commit the plan plus Backlog task update.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Wrote implementation plan for approved RPG campaign/session runtime design. Verified required plan header, ran red-flag scan for incomplete markers, and ran git diff --check on the plan and task file. Bandit is not applicable to this planning-only change because no Python implementation code was modified.
Reopened for pre-execution review amendments. Local review found that the plan's idempotency uniqueness would block multi-event commits, append plus snapshot/proposal updates were not described as one atomic transaction, and several API/MCP snippets used __dict__ on slots dataclasses.
Pre-execution review amendments completed. Updated the plan to require optimistic concurrency (`expected_last_event_sequence` plus snapshot version checks), operation-scoped idempotency records for all mutating REST/MCP operations, atomic event/snapshot/session/proposal/idempotency commits, server-derived event sources, explicit REST permission dependencies, MCP DB access via `context.db_paths["chacha"]`, protocol-level MCP authorization tests, adapter-owned check resolution, full V1 reducer event coverage, and corrected schema/service/API/MCP signatures. Ran `git diff --check` and markdown fence balance check on the amended plan.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implementation plan amended after subagent-driven pre-execution review. The remaining plan is aligned around atomic ledger commits, operation-scoped idempotency, explicit API/MCP authorization, adapter boundaries, and complete V1 event validation/reducer coverage.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
