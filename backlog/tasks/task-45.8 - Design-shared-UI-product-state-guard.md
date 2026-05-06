---
id: TASK-45.8
title: Design shared UI product-state guard
status: Done
assignee: []
created_date: '2026-05-06 00:04'
updated_date: '2026-05-06 00:11'
labels:
  - design-system
  - frontend
  - docs
  - guardrails
dependencies: []
documentation:
  - Docs/Design/tldw_web_design_system_contract.md
  - Docs/Design/tldw_web_design_system_inventory.md
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the approved design spec for a baseline product-state guard over
apps/packages/ui/src. The guard should protect design-system product-state
primitives during the full migration to the design framework while allowing
documented legacy exceptions and migration notes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A spec document is written under Docs/superpowers/specs with the approved baseline product-state guard design.
- [x] #2 The spec covers scope, migration model, guard rules, baseline
  behavior, architecture, testing, rollout, and non-goals.
- [x] #3 The spec is reviewed and updated for clarity before user review.
- [x] #4 The design spec is committed on a clean branch for the next
  implementation-plan step.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Wrote Docs/superpowers/specs/2026-05-06-design-system-product-state-guard-design.md
in clean worktree .worktrees/design-system-product-state-guard-spec. Spec review
loop took three passes: first review found baseline identity, owner metadata,
canonical root exemption, and active_migration_target behavior gaps; second
found one stale path+rule testing bullet; third review approved with no blocking
issues. Advisory recommendations are deferred to implementation planning:
decide whether adapter exceptions live in baseline or separate allowlist,
consider stale baseline warnings, and update lifecycle status after user review.

Verification: spec review loop approved on the third pass. git diff --check
passed. Line-length scan passed for the spec and task file after wrapping the
Backlog notes. Runtime tests and Bandit were skipped because this task only
adds documentation and Backlog metadata.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Drafted and reviewed the design spec for a guarded shared UI product-state
foundation. The spec defines the apps/packages/ui/src boundary, migration model,
product-state definition, guard rules, canonical implementation/adaptor
exemptions, stable finding identities, baseline schema and behavior,
architecture, report format, testing strategy, rollout, success criteria, risks,
and implementation-planning questions. Spec review is approved with only
advisory implementation-planning recommendations remaining.

Verification is documentation-focused: spec review approved, git diff --check
passed, and no executable code paths were touched. Runtime tests and Bandit are
not applicable for this docs-only design task.
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
