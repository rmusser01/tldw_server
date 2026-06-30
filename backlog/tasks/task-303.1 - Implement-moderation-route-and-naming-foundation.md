---
id: TASK-303.1
title: Implement moderation route and naming foundation
status: Done
assignee: []
created_date: '2026-05-12 16:13'
updated_date: '2026-05-12 18:06'
labels:
  - moderation
  - webui
  - routes
dependencies:
  - TASK-305
documentation:
  - >-
    Docs/superpowers/specs/2026-05-12-moderation-review-rules-remediation-design.md
  - >-
    Docs/superpowers/plans/2026-05-12-moderation-review-rules-remediation-implementation-plan.md
parent_task_id: TASK-303
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Stage 1 of the moderation remediation plan. Establish the durable route and naming foundation: /moderation is the review destination, /moderation/rules is the rule configuration destination, and /moderation-playground becomes a legacy redirect. Keep this slice route/navigation focused and avoid building the full review queue data model.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 /moderation renders an honest Moderation Review first-slice shell in shared WebUI and extension route registries.
- [x] #2 /moderation/rules renders the existing rule configuration experience with user-facing Content Rules naming.
- [x] #3 /moderation-playground is preserved as a legacy redirect to /moderation/rules in Next, shared, and extension routing surfaces.
- [x] #4 Header shortcuts, settings navigation, tutorials, locale strings, and smoke inventory no longer present moderation-playground as the primary destination.
- [x] #5 Focused route, navigation, and redirect tests cover the new paths and legacy alias behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Stage 1 execution plan:
1. Add failing focused tests for route constants, settings navigation, route alias redirects, and moderation route smoke expectations.
2. Implement shared route constants for /moderation, /moderation/rules, and legacy /moderation-playground.
3. Add the honest first-slice Moderation Review shell and shared/extension route wrappers.
4. Repoint Content Rules wrappers to the existing ModerationPlayground shell and convert legacy playground wrappers/pages to redirects.
5. Update route registries, header shortcuts/default migration, settings nav, tutorials/locales, and E2E inventory/mapping.
6. Run focused Vitest verification and Playwright route workflow if available; document Bandit as not applicable to TS/route-only frontend changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-05-12: Implemented Stage 1 route foundation. Focused Vitest route/nav/shortcut tests pass; targeted Playwright moderation-routes workflow passes with mocked health/moderation endpoints. Full tsc --noEmit currently fails on pre-existing unrelated Evaluation/persona/VN type errors outside this slice; no Bandit run because touched implementation is frontend TS/route/test code only.

2026-05-12: Reviewer findings addressed: aligned ModerationReviewShell copy/test expectations, made settings-nav route source assertion cwd-stable, preserved Next legacy redirect params, and migrated persisted header moderation-playground shortcuts to both moderation-review and moderation-rules.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented Stage 1 route/naming foundation for moderation: /moderation now renders an honest review shell, /moderation/rules renders the existing Content Rules configuration surface, and /moderation-playground remains as a legacy redirect. Updated shared and extension route registries, Next pages, shortcut defaults/migration, settings navigation, tutorials/locales, smoke inventory, and focused route tests. Verification: focused Vitest suite passed; targeted Playwright moderation-routes workflow passed; full tsc remains blocked by unrelated baseline errors outside this slice.
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
