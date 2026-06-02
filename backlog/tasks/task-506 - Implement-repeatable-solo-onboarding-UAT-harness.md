---
id: TASK-506
title: Implement repeatable solo onboarding UAT harness
status: In Progress
priority: high
references:
- TASK-504
- TASK-505
documentation:
- Docs/superpowers/plans/2026-06-02-repeatable-onboarding-uat-harness-implementation-plan.md
- Docs/superpowers/specs/2026-06-02-solo-onboarding-v2-roadmap-design.md
- 'Baseline 2026-06-02: RUN_MOCK_OPENAI=1 python -m pytest mock_openai_server/tests/test_server.py
  -q => 17 passed'
- '2 failed. Failures are pre-existing: test_chat_with_parameters returns model gpt-4
  instead of requested gpt-3.5-turbo; async streaming client attempts real network
  because no ASGI transport is configured. Frontend baseline bunx vitest run __tests__/e2e-harness-readiness.guard.test.ts
  failed before tests because fresh worktree lacks local node dependencies (cannot
  resolve vitest/config and @vitejs/plugin-react).'
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first PR from the solo onboarding V2 roadmap: a repeatable Playwright/CDP-driven onboarding UAT harness that uses the repo mock_openai_server, isolated runtime profile, deterministic scenarios, screenshots, JSON summary, and backend/frontend/mock logs. This task tracks the implementation branch work and verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
