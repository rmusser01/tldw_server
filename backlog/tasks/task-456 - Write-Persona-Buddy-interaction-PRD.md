---
id: TASK-456
title: Write Persona Buddy interaction PRD
status: Done
labels:
- persona
- buddy
- prd
references:
- 'PR #1895'
- 'issue #1510'
documentation:
- Docs/superpowers/specs/2026-05-20-persona-buddy-interaction-prd-design.md
modified_files:
- Docs/superpowers/specs/2026-05-20-persona-buddy-interaction-prd-design.md
- backlog/tasks/task-456 - Write-Persona-Buddy-interaction-PRD.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the PRD for the Buddy interaction scope in the Persona/Buddy system, explicitly avoiding overlap with PR #1895 Persona Visual sectioned workspace. The PRD should define backend Persona Live Control API scope, shared frontend controller scope, Buddy dock/popover behavior, error handling, safety, and acceptance criteria.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 PRD documents non-overlap with Persona Visual sectioned workspace PR #1895.
- [x] #2 PRD defines backend-first Persona Live Control API scope for multi-session Buddy interaction.
- [x] #3 PRD defines shared frontend controller and web desktop Buddy dock/popover behavior.
- [x] #4 PRD includes error handling, safety, and test/acceptance criteria.
- [x] #5 Spec review is run before user handoff.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Wrote `Docs/superpowers/specs/2026-05-20-persona-buddy-interaction-prd-design.md`.
- Verified PR #1895 is merged and scoped to `VisualPackEditor`, its tests, and its Backlog task.
- Verified currently open PRs are unrelated to this Buddy interaction PRD scope.
- Incorporated spec-review feedback by clarifying backend-owned user-scoped Buddy focus, retry-safe create/resume/send semantics with `idempotency_key` and `client_message_id`, and frontend-mediated browser microphone capture.
- Per user-requested design critique, hardened the PRD before implementation planning: clarified the first slice as text-first, routed approval-needed handling to the full Live view, specified stale focus cleanup, added a runtime lifecycle vocabulary, required capability flags for deferred voice controls, and split implementation staging into smaller reviewable PRs.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created the Persona Buddy Interaction PRD for the next Buddy interaction scope. The PRD defines the non-overlap boundary with merged PR #1895, a backend Persona Live Control API, a shared frontend Persona Live controller, the compact web desktop Buddy dock/popover behavior, visual-state priority, safety/privacy requirements, recovery behavior, and backend/frontend/browser acceptance criteria.

Spec review result: first pass found focus ownership, send/idempotency, and microphone-authority gaps; the PRD was patched and the second review approved it.

Additional design review before implementation planning found and fixed over-broad staging and underspecified edge cases around default Buddy dependency, stale focus cleanup, approval handling, voice deferral, and lifecycle/capability testing.

Verification:
- `git diff --check` passed.
- Bandit was not run because this change only adds documentation and Backlog task text.

Known skips:
- No code, unit, integration, or browser tests were run because this is a PRD-only task.
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
