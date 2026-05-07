---
id: TASK-101
title: Create WebUI dependency audit for issue 1346
status: Done
assignee: []
created_date: '2026-05-07 01:36'
updated_date: '2026-05-07 02:11'
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
- [x] #1 Docs/Design/WebUI_Dependency_Audit.md exists and links to issue #1346, TASK-100, and the approved design spec.
- [x] #2 Audit covers apps/tldw-frontend/package.json, apps/packages/ui/package.json, apps/bun.lock, and extension impact checks for shared UI candidates.
- [x] #3 Audit table records package, declared locations, import count, representative sites, consumer surface, category, decision, risk, expected impact, and follow-up slice.
- [x] #4 Security-sensitive and complex-domain packages are explicitly marked keep or defer-design with rationale.
- [x] #5 Quick cleanup and axios replacement candidates are ranked for follow-up tasks without changing package manifests in this audit slice.
- [x] #6 Verification commands and Bandit docs-only skip are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Task 3 docs-only classification plan: regenerate declaration and usage JSON, inspect pubsub-js/buffer/stream-browserify/clsx/axios with import-aware evidence, fill every audit table classification cell, update ranked follow-up and verification notes, run git diff --check and git status --short, then commit only Docs/Design/WebUI_Dependency_Audit.md and TASK-101.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Wrote implementation plan for the first issue #1346 work unit: create Docs/Design/WebUI_Dependency_Audit.md without package or runtime edits.

Plan review before commit found and fixed two issues: usage-scan commands now exclude manifests/docs/generated artifacts so declaration mentions are not counted as usage, and Backlog documentation commands now preserve both spec and plan links. Final task closeout command now checks DoD and marks TASK-101 Done after audit completion.

Started dependency audit implementation plan. First slice is docs-only audit artifact; package removals and axios replacement are follow-up work.

Quality fix: added TASK-100 parent design task reference to Docs/Design/WebUI_Dependency_Audit.md so the audit traces both design approval and execution work.

Task 2 progress generated the temporary declaration JSON from the three manifests. Observed 138 unique declarations total. Populated the audit table with 125 WebUI and shared UI declarations. Left Task 3 fields as TBD. Excluded 13 extension only declarations because extension remains an impact check surface.

Task 3 started in worktree branch codex/webui-dependency-trim-1346. Scope remains docs-only: audit document and TASK-101 updates only.

Task 3 progress generated /tmp/tldw-webui-dependency-usage.json from 4605 scanned files across apps/tldw-frontend, apps/packages/ui, and apps/extension. Classified all 125 audited WebUI/shared UI rows, ranked quick cleanup and replacement candidates, and recorded quick-candidate precision caveats and Bandit docs-only skip in the audit.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the docs-only WebUI dependency audit classification for issue #1346. The audit now classifies all 125 WebUI/shared UI package rows with import counts, representative sites, consumer surface, category, decision, risk, expected impact, and follow-up slice. It ranks quick cleanup candidates, keeps axios and clsx as dedicated replacement work, defers icon-stack and complex rendering/parser/editor/security/schema packages, and records the usage scan, 4605 scanned file count, quick-candidate inspection, precision caveats, and Bandit docs-only skip. Verification run: table shape/no-TBD check and git diff --check.
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
