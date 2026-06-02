---
id: TASK-505
title: Plan repeatable solo onboarding UAT harness
status: Done
documentation:
- Docs/superpowers/specs/2026-06-02-solo-onboarding-v2-roadmap-design.md
- Docs/superpowers/plans/2026-06-02-repeatable-onboarding-uat-harness-implementation-plan.md
modified_files:
- Docs/superpowers/plans/2026-06-02-repeatable-onboarding-uat-harness-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the implementation plan for PR1 of the solo onboarding V2 roadmap: a repeatable manual/dev UAT harness that runs the real WebUI, real backend, and repository mock OpenAI-compatible API server against an isolated runtime profile with screenshots, logs, and JSON summaries.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Plan is saved under Docs/superpowers/plans with exact files, steps, commands, tests, and expected outcomes.
- [x] #2 Plan scopes PR1 only and defers diagnostics, starter questions, and local model V2 product behavior to later PRs.
- [x] #3 Plan uses the repo mock_openai_server for provider behavior, not Playwright route mocks.
- [x] #4 Plan defines isolated temp runtime profiles, artifact layout, cleanup behavior, and secret redaction checks.
- [x] #5 Plan includes verification commands and a staged commit strategy for implementation.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Created `Docs/superpowers/plans/2026-06-02-repeatable-onboarding-uat-harness-implementation-plan.md`.
- Scoped PR1 to harness and evidence work only; PR2 diagnostics, PR3 starter questions, and PR4 local model UX remain deferred.
- Planned repo `mock_openai_server` deterministic scenario controls so provider/chat/model behavior comes from static mock-server config files instead of Playwright route mocks.
- Planned isolated temp runtime profiles with synthetic secrets, temp config/env/databases/uploads/logs, cleanup, redacted manifest, and leak scanning.
- Included staged implementation commits, exact commands, and verification gates for mock-server pytest, frontend Vitest guards, manual UAT, existing onboarding E2E, Bandit for touched Python, and diff checks.
- Plan-document-reviewer subagent review was not dispatched because the available `multi_agent_v1.spawn_agent` tool is restricted to cases where the user explicitly asks for subagents, delegation, or parallel agent work. Performed a local plan review and patched findings instead.
- Verification: plan/task-only change. `git diff --check -- Docs/superpowers/plans/2026-06-02-repeatable-onboarding-uat-harness-implementation-plan.md 'backlog/tasks/task-505 - Plan-repeatable-solo-onboarding-UAT-harness.md'` passed. ASCII check passed. Bandit is not applicable because no executable code was changed in this task.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PR1 implementation plan is complete and ready for user review. The plan defines the repeatable solo onboarding UAT harness architecture, file map, Tier A scenarios, artifact contract, cleanup/redaction requirements, staged task breakdown, and verification strategy.
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
