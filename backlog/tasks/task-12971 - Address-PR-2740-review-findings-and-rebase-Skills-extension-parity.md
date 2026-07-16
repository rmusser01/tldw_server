---
id: TASK-12971
title: Address PR 2740 review findings and rebase Skills extension parity
status: In Progress
labels:
- skills
- extension
- code-review
- reliability
priority: high
references:
- 'PR #2740'
documentation:
- backlog/completed/task-12970 - Certify-Skills-browser-extension-parity-and-fix-shell-specific-regressions.md
modified_files:
- IMPLEMENTATION_PLAN_skills_extension_parity_pr_2740_review.md
- apps/extension/tests/e2e/skills.parity.spec.ts
- apps/extension/tests/e2e/utils/extension-build.test.ts
- apps/extension/tests/e2e/utils/extension-build.ts
- apps/extension/tests/unit/options-theme-bootstrap.test.ts
- apps/extension/tests/unit/skills-fixture-request-contract.test.ts
- apps/packages/ui/src/public/theme-bootstrap.js
- apps/tldw-frontend/e2e/utils/skills-fixtures.ts
- backlog/tasks/task-12971 - Address-PR-2740-review-findings-and-rebase-Skills-extension-parity.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Resolve every actionable review finding on PR #2740, reply to and resolve review threads, rebase the branch onto the latest dev, rerun focused and release verification, and merge only after all repository merge gates are satisfied.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Download archive verification fails with a clear error when Playwright cannot create a stream.
- [x] #2 Theme bootstrap handles expected blocked localStorage access without silently swallowing unrelated exceptions.
- [x] #3 Skills fixture request validation normalizes HTTP methods and reports malformed URLs with a bounded contract-specific error.
- [ ] #4 All actionable PR threads are replied to and resolved, and all relevant focused/full verification passes.
- [ ] #5 Branch is rebased onto latest origin/dev and merged only when repository merge gates are satisfied.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See IMPLEMENTATION_PLAN_skills_extension_parity_pr_2740_review.md.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
2026-07-15: Verified all PR feedback. Implemented the nullable Playwright download-stream guard, a narrow SecurityError-only theme storage fallback that rethrows unexpected errors, and case-normalized fixture methods with serialized/bounded malformed-URL diagnostics. TDD evidence: initial focused run failed 4 intended tests; escaped-input hardening then failed at 621 characters; final focused tests passed 15/15. Extension touched unit tests passed 19/19, TypeScript compile passed, production Chrome build and token sync passed with documented baseline warnings, strict packaged parity passed 6/6 with skipped=0 unexpected=0 flaky=0, WebUI Skills Playwright passed 13 mocked with 3 pre-existing live-server skips. The first eight-file shared Skills aggregate had two load-sensitive 5-second timeouts; both passed individually and the complete Manager file then passed 83/83. Bandit is not applicable because no Python files changed.
2026-07-15: CodeRabbit added six threads after the initial review. Implemented four confirmed fixes/evidence: unknown-based runtime shim narrowing plus nullish diagnostics fallback; persistent-context cleanup when targeted page preparation/navigation/readiness fails; a non-overlapping /skills/context fixture route; and a packaged focus-handoff regression. The focus regression passed against the unchanged Drawer after its close transition, so no speculative focusable prop was added. The environment-cleanup suggestion was not applied because afterEach already restores the complete environment and both setup callers clean filesystem state in finally. The shared focus-hook extraction was not applied because only two effects share the fallback and abstracting them would broaden a targeted fix. New TDD tests failed for context leak and route overlap, then passed. Final current evidence: touched extension tests 21/21, compile passed, strict packaged parity 6/6 with zero skips/unexpected/flaky, and WebUI Skills Playwright 13 mocked passed with the same three unavailable live-server skips.
2026-07-15: Independent review found that a rejected context.close() could mask the original extension setup error. Added a regression that first failed with the cleanup error, then updated cleanup to preserve and rethrow the setup error while attaching the cleanup failure as its cause. Final touched extension verification now passes 22/22; TypeScript compile passes; strict packaged parity passes 6/6 with skipped=0 unexpected=0 flaky=0.
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
