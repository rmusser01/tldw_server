---
id: TASK-418.10.1
title: Implement WP12 route governance metadata and inventory coverage
status: Done
labels:
- ux
- webui
- extension
- governance
- tests
priority: high
parent_task_id: TASK-418.10
references:
- TASK-418.10
documentation:
- Docs/superpowers/plans/2026-05-17-webui-route-governance-qa-implementation-plan.md
- Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md
- Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md
modified_files:
- apps/packages/ui/src/routes/__tests__/route-governance.metadata-coverage.test.ts
- apps/packages/ui/src/routes/route-metadata.ts
- apps/tldw-frontend/e2e/smoke/page-inventory.ts
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the first implementation slice from the WebUI/extension UX remediation Task 12 plan. Add deterministic route governance coverage that checks route metadata against shared route registries, extension routes, sidepanel availability, and smoke inventory decisions. Keep scope frontend-governance/test focused; do not change backend APIs or route-family UX unless a trivial test harness alignment is required.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Route governance metadata coverage test exists and covers duplicate inventory paths, shared option routes, active and skipped page inventory metadata, included smoke route activity, and excluded alias activity.
- [x] #2 Missing Next-only smoke inventory routes have explicit route metadata with user-facing classification and rationale.
- [x] #3 Redirect alias rows in page inventory are skipped with metadata-backed reasons, and stale `/chat/settings` inventory rows are removed.
- [x] #4 Focused route Vitest and Playwright route metadata smoke checks pass; full TypeScript baseline result is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Added route-governance metadata coverage as a deterministic Vitest guard for shared option-route metadata, active smoke inventory metadata ownership, skipped inventory reasons, included smoke route inventory presence, duplicate page inventory paths, and smoke-excluded alias handling.

The red phase exposed 23 active inventory routes without metadata, two stale `/chat/settings` entries, and five active redirect aliases. Added metadata for Next-only admin/auth/billing/connector/marketing/agent/media redirect pages and marked redirect aliases as skipped in `page-inventory.ts`.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented the first WP12 route-governance slice. Added deterministic metadata/inventory coverage for duplicate smoke inventory paths, shared option-route metadata ownership, active and skipped smoke inventory metadata ownership, active inclusion for `smoke: "include"` web routes, and prevention of active smoke runs for `smoke: "exclude"` alias rows. Added missing route metadata for Next-only admin, hosted auth/billing, connector placeholder, marketing landing, legacy agent, and dynamic media redirect pages. Removed stale `/chat/settings` smoke entries and marked redirect aliases as skipped in page inventory. Verification: focused route Vitest suite passed (21 tests), Playwright route metadata smoke contract passed (3 tests), `git diff --check` passed, targeted ESLint exited 0 with package-base ignore warnings for shared UI files, and full `bunx tsc --noEmit` was attempted but failed on existing unrelated baseline errors outside the touched route governance files. Bandit skipped because this slice touched TypeScript/Markdown only and no Python code.
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
