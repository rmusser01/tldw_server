---
id: TASK-100
title: Design WebUI dependency trimming sequence for issue 1346
status: Done
assignee: []
created_date: '2026-05-07 01:05'
updated_date: '2026-05-07 02:49'
labels:
  - webui
  - dependencies
  - design
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1346'
documentation:
  - Docs/superpowers/specs/2026-05-07-webui-dependency-trimming-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the design/spec for GitHub issue #1346, which asks to reduce WebUI package usage by first auditing dependencies, then removing unused or tiny packages, then replacing axios with platform-native fetch helpers. This task is for the design/spec review unit only; implementation work should be split into follow-up tasks or subtasks after the spec is approved.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec documents the approved A -> C -> B sequence: dependency audit, quick cleanup, then axios replacement.
- [x] #2 Spec identifies target package surfaces: apps/tldw-frontend/package.json, apps/packages/ui/package.json, and apps/bun.lock.
- [x] #3 Spec includes guardrails for security-sensitive and complex-domain packages that should not be hand-rolled.
- [x] #4 Spec defines verification expectations for audit, quick cleanup, and axios replacement slices.
- [x] #5 Spec is linked to GitHub issue #1346 and ready for implementation planning after review.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Wrote approved design spec for issue #1346 at Docs/superpowers/specs/2026-05-07-webui-dependency-trimming-design.md. No runtime or dependency files changed in this design slice.

Spec review loop result: Approved. Reviewer confirmed the spec preserves audit -> quick cleanup -> axios replacement order, splits work into reviewable units, names target package surfaces and issue/task references, and includes guardrails plus verification expectations.

Verification for design slice: git diff --check passed. Bandit skipped because this slice changes docs/backlog task metadata only and touches no Python code.

Manual design review before implementation planning found and fixed three risks: default api/baseURL compatibility for axios replacement, per-request config compatibility, and extension impact checks for shared @tldw/ui dependency changes. Also clarified that clsx should split into its own PR if compatibility work is non-mechanical.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the WebUI dependency trimming design for issue #1346. The spec documents the approved audit -> quick cleanup -> axios replacement sequence, identifies the WebUI/shared UI package surfaces and extension impact checks, records guardrails for complex/security-sensitive packages, and defines verification expectations for follow-up slices. Verification was docs-scope git diff --check; Bandit skipped because no Python/runtime code changed.
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
