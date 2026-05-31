---
id: TASK-66
title: Make tldw-frontend TypeScript check green
status: Done
assignee: []
created_date: '2026-05-05 05:23'
updated_date: '2026-05-05 15:06'
labels:
  - frontend
  - typescript
  - tests
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1302'
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Investigate and reduce local apps/tldw-frontend TypeScript check failures after the design-system merge. Separate PR-caused failures from existing baseline errors, preserve focused Playground state tests, and avoid broad unrelated rewrites unless needed to get the frontend gate green.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 apps/tldw-frontend tsc failure surface is documented with current error count
- [x] #2 PR-local Playground guard tests pass
- [x] #3 Any committed fixes are scoped and explained
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Current local tsc baseline for apps/tldw-frontend was reproduced in /private/tmp/tldw-dev-tsc. Initial current dev surface was 388 primary TypeScript errors; after scoped fixes, node node_modules/typescript/bin/tsc --noEmit --pretty false -p tsconfig.json exits 0 from apps/tldw-frontend.

Focused Playground guard tests pass: bunx vitest run the six Playground guard specs -> 6 files passed, 6 tests passed.

Fixes are intentionally type-surface cleanup and stale-code removal: Playground compare/unreachable cluster, tldw API client/domain typings, i18n TFunction dependency types, callback adapters, generated API envelopes, strict route/path helpers, and test/shim type adapters. Bandit is not applicable because touched scope is TypeScript/frontend/test configuration rather than Python security-sensitive code.

git diff --check exits 0.

Draft PR opened: https://github.com/rmusser01/tldw_server/pull/1302. Human requester still needs to add the required human-written Change summary before merge.

Review-fix pass completed for PR #1302. Addressed structured chat version-conflict detection and getChat reuse; scoped chat metadata/scope precedence; command palette query callback consistency; sidepanel edit send flag; prompt ID/content separation; persisted quick-ingest/account-mode validation; navigation shim delimiter and promise handling; test setup shims; and targeted cast/type cleanup. Verification after review fixes: tsc in apps/tldw-frontend exited 0; focused vitest batch with --testTimeout=10000 in apps/packages/ui passed 4 files and 62 tests; git diff --check exited 0. The same focused vitest batch without increased timeout had two FamilyGuardrailsWizard cases exceed the default 5000ms, and those two passed when isolated.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Made the tldw frontend TypeScript gate green from a current dev baseline by tightening frontend/domain typings, removing stale Playground compare code, preserving abort signals, normalizing API envelopes, adapting callback arities, and cleaning test/shim type surfaces. Verified apps/tldw-frontend tsc exits 0, focused Playground guard vitest tests pass, and git diff --check is clean.
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
