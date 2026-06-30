---
id: TASK-45.40
title: Design design-system remaining-work issue tracker
status: In Progress
assignee: []
created_date: '2026-05-14 01:41'
updated_date: '2026-05-14 01:57'
labels:
  - design-system
  - webui
  - product-state
dependencies: []
references:
  - Docs/Design/tldw_web_design_system_contract.md
  - Docs/Design/tldw_web_design_system_inventory.md
  - apps/packages/ui/scripts/design-system-product-state-baseline.json
parent_task_id: TASK-45
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design the GitHub epic plus Backlog.md mirror structure for tracking the remaining tldw WebUI and extension design-system migration and governance program. The design must capture the approved tracker model: one public GitHub epic, mirrored Backlog parent, product-area sub-issues, governance sub-issues, hybrid priority order, and area-complete closure based on product-state baseline burn-down.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A written design spec defines the GitHub epic and Backlog.md mirror relationship for the remaining design-system program.
- [x] #2 The spec defines product-area and governance sub-issue templates including scope, baseline debt, done criteria, verification, and tracking links.
- [x] #3 The spec defines how current product-state baseline counts are snapshotted and refreshed after migration PRs.
- [x] #4 The spec includes rollout and maintenance rules that prevent the tracker from becoming stale or duplicating line-level baseline entries.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Wrote Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md for the approved GitHub epic plus Backlog mirror tracker. The spec includes product-area and governance issue templates, baseline snapshot rules, ordered path ownership, mutable-state source-of-truth rules, baseline issue-reference format, and close/reopen behavior.

Spec review loop: first review found four issues around reproducible path ownership, duplicated mutable state, baseline issue-reference format, and closed-issue drift. Revised the spec to address all four. Second review approved with no blocker or medium-severity issues.

Verification: git diff --check passed. Bandit is not applicable because this task only writes Markdown and Backlog metadata.

User-requested design review after initial approval: self-review found three maintainability risks. The spec now clarifies that GitHub sub-issues are regular linked issues unless native sub-issues are available, adds a human-reviewed issue-body draft gate before public GitHub mutation, and adds long-tail split rules for path groups with five or more findings.

Review verification: git diff --check passed after the self-review patch.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
