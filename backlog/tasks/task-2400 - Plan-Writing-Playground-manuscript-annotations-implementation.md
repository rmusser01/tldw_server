---
id: TASK-2400
title: Plan Writing Playground manuscript annotations implementation
status: Done
labels:
- planning
- webui
- extension
- writing-playground
- manuscripts
documentation:
- Docs/superpowers/plans/2026-06-23-writing-playground-manuscript-annotations-implementation-plan.md
- Docs/superpowers/specs/2026-05-24-writing-playground-manuscript-annotations-design.md
modified_files:
- Docs/superpowers/plans/2026-06-23-writing-playground-manuscript-annotations-implementation-plan.md
- Docs/superpowers/specs/2026-05-24-writing-playground-manuscript-annotations-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->

<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Created an implementation plan for the approved Writing Playground manuscript annotations design. The plan decomposes backend schema/helper/API work, frontend saved-scene binding, inspector annotations, desktop margin rail, selected-text review, Jobs-backed scene review, suggested-fix revision handoff, tests, accessibility, and final verification into task-sized TDD stages.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implementation plan reviewed and hardened before subagent-driven execution. Updates cover exact provider/model contract, Unicode code-point offset handling, per-user Jobs worker ownership through owner_user_id, startup poller regression coverage, PostgreSQL migration verification, mandatory dirty-scene navigation behavior, hook barrel exports, browser margin-rail smoke checks, package-local dev/e2e commands, and expanded Bandit touched scope. Verification: git diff --check passed for plan/spec/task files; ASCII scan returned no matches; stale-contract/placeholder scan returned no matches. Bandit was not run because this task only edits Markdown planning/spec/backlog artifacts.
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
