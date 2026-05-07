---
id: TASK-101
title: Create WebUI dependency audit for issue 1346
status: In Progress
assignee: []
created_date: '2026-05-07 01:36'
updated_date: '2026-05-07 01:50'
labels:
  - webui
  - dependencies
  - audit
dependencies:
  - TASK-100
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1346'
documentation:
  - Docs/superpowers/specs/2026-05-07-webui-dependency-trimming-design.md
  - >-
    Docs/superpowers/plans/2026-05-07-webui-dependency-audit-implementation-plan.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the first approved work unit from the WebUI dependency trimming design: create a reviewable dependency audit artifact before removing packages. The audit should cover apps/tldw-frontend/package.json, apps/packages/ui/package.json, apps/bun.lock, and apps/extension/package.json as an impact-check surface for shared @tldw/ui dependencies. This task should not remove packages or rewrite axios.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Docs/Design/WebUI_Dependency_Audit.md exists and links to issue #1346, TASK-100, and the approved design spec.
- [ ] #2 Audit covers apps/tldw-frontend/package.json, apps/packages/ui/package.json, apps/bun.lock, and extension impact checks for shared UI candidates.
- [ ] #3 Audit table records package, declared locations, import count, representative sites, consumer surface, category, decision, risk, expected impact, and follow-up slice.
- [ ] #4 Security-sensitive and complex-domain packages are explicitly marked keep or defer-design with rationale.
- [ ] #5 Quick cleanup and axios replacement candidates are ranked for follow-up tasks without changing package manifests in this audit slice.
- [ ] #6 Verification commands and Bandit docs-only skip are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Wrote implementation plan for the first issue #1346 work unit: create Docs/Design/WebUI_Dependency_Audit.md without package or runtime edits.

Plan review before commit found and fixed two issues: usage-scan commands now exclude manifests/docs/generated artifacts so declaration mentions are not counted as usage, and Backlog documentation commands now preserve both spec and plan links. Final task closeout command now checks DoD and marks TASK-101 Done after audit completion.

Started dependency audit implementation plan. First slice is docs-only audit artifact; package removals and axios replacement are follow-up work.

Quality fix: added TASK-100 parent design task reference to Docs/Design/WebUI_Dependency_Audit.md so the audit traces both design approval and execution work.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
