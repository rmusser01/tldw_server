---
id: TASK-12952
title: Rebase PR 2719 and address auth-refresh review feedback
status: In Progress
labels:
- auth
- webui
- browser-extension
- review
priority: high
references:
- https://github.com/rmusser01/tldw_server/pull/2719
- TASK-12953
documentation:
- Docs/superpowers/specs/2026-07-12-legacy-api-key-refresh-migration-design.md
modified_files:
- apps/packages/ui/src/services/__tests__/tldw-api-client.quickstart-auth.test.ts
- apps/tldw-frontend/e2e/extension-api-key-persistence.spec.ts
- apps/tldw-frontend/e2e/helpers/manual-api-key-fixture.ts
- apps/tldw-frontend/e2e/manual-api-key-persistence.spec.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Rebase PR #2719 onto latest dev, evaluate and address every actionable review comment, verify the shared WebUI/browser-extension auth-refresh fix, resolve review threads, and update the PR branch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 PR #2719 is rebased onto the latest origin/dev without unresolved conflicts.
- [ ] #2 Every actionable inline and out-of-diff review comment is either fixed or answered with verified technical reasoning.
- [ ] #3 Focused unit, browser, build, compile, diff, and applicable security checks pass.
- [ ] #4 The rebased branch is pushed and all addressed review threads are resolved.
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
