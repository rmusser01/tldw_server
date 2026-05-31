---
id: TASK-45.44.3
title: 'Migrate design-system product state: Jobs, Scheduler, and Watchlists'
status: Done
assignee: []
created_date: 2026-05-14 03:19
updated_date: 2026-05-24 01:50
labels:
- design-system
- webui
- extension
- product-state
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/issues/1660
- Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
- apps/packages/ui/scripts/design-system-product-state-baseline.json
- https://github.com/rmusser01/tldw_server/pull/2012
- https://github.com/rmusser01/tldw_server/pull/2013
- https://github.com/rmusser01/tldw_server/pull/2016
- https://github.com/rmusser01/tldw_server/pull/2029
- https://github.com/rmusser01/tldw_server/pull/2037
- https://github.com/rmusser01/tldw_server/pull/2039
- https://github.com/rmusser01/tldw_server/pull/2044
documentation:
  - |-
    TASK-45.44.3.6 / PR #2012 migrated OutputsTab delivery-issues Alert to the design-system Alert primitive.
    Baseline evidence:
      - total product-state exceptions: 259 -> 258
      - Jobs/Scheduler/Watchlists exceptions: 24 -> 23
      - OutputsTab target rows: 1 -> 0
    Verification recorded in TASK-45.44.3.6.
  - |-
    TASK-45.44.3.7 / PR #2013 migrated RunsTab load-error and reliability-attention banners to the design-system Alert primitive.
    Baseline evidence:
      - total product-state exceptions: 258 -> 257
      - Jobs/Scheduler/Watchlists exceptions: 23 -> 22
      - RunsTab target rows: 1 -> 0
    Verification recorded in TASK-45.44.3.7.
  - |-
    TASK-45.44.3.8 / PR #2016 migrated SourcesBulkImport loading, preflight, and import summary banners to the design-system Alert primitive.
    Baseline evidence:
      - total product-state exceptions: 257 -> 256
      - Jobs/Scheduler/Watchlists exceptions: 22 -> 21
      - SourcesBulkImport target rows: 1 -> 0
    Verification recorded in TASK-45.44.3.8.
  - |-
    TASK-45.44.3.9 migrated ReportBuilderDrawer run-required and preflight warning notices to the design-system Alert primitive.
    Baseline evidence:
      - total product-state exceptions: 256 -> 255
      - Jobs/Scheduler/Watchlists exceptions: 21 -> 20
      - ReportBuilderDrawer target rows: 1 -> 0
    Verification recorded in TASK-45.44.3.9.
  - |-
    TASK-45.44.3.10 / PR #2029 migrated AlertsTab boundary guidance and load-error callouts to the design-system Alert primitive.
    Baseline evidence:
      - total product-state exceptions: 253 -> 251
      - Jobs/Scheduler/Watchlists exceptions: 20 -> 18
      - AlertsTab target rows: 2 -> 0
    Verification recorded in TASK-45.44.3.10.
  - |-
    TASK-45.44.3.11 / PR #2037 migrated TemplateEditor version-drift and visual-repair warning callouts to the design-system Alert primitive.
    Baseline evidence:
      - total product-state exceptions: 251 -> 249
      - Jobs/Scheduler/Watchlists exceptions: 18 -> 16
      - TemplateEditor target rows: 2 -> 0
    Verification recorded in TASK-45.44.3.11.
  - |-
    TASK-45.44.3.12 / PR #2039 migrated WatchlistSetupWizard collection-scope guidance and validation error callouts to the design-system Alert primitive.
    Baseline evidence:
      - total product-state exceptions: 249 -> 247
      - Jobs/Scheduler/Watchlists exceptions: 16 -> 14
      - WatchlistSetupWizard target rows: 2 -> 0
    Verification recorded in TASK-45.44.3.12.
  - |-
    TASK-45.44.3.13 / PR #2044 migrated Watchlists SettingsTab settings guidance, diagnostics, cluster subscription, cluster error, and unavailable callouts to the design-system Alert primitive.
    Baseline evidence:
      - total product-state exceptions: 247 -> 242
      - Jobs/Scheduler/Watchlists exceptions: 14 -> 9
      - SettingsTab target rows: 5 -> 0
    Verification recorded in TASK-45.44.3.13.
  - |-
    TASK-45.44.3.14 / PR #2044 migrated the remaining Common Workflow AnalyzeBookWorkflow and AgentTasks product-state UI to design-system primitives.
    Baseline evidence:
      - total product-state exceptions: 242 -> 233
      - Jobs/Scheduler/Watchlists exceptions: 9 -> 0
      - AnalyzeBookWorkflow target rows: 3 -> 0
      - AgentTasks target rows: 6 -> 0
    Verification recorded in TASK-45.44.3.14.
parent_task_id: TASK-45.44
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Mirror the linked GitHub product-area migration issue. Closure requires zero current product-state baseline exceptions for the owned path map area and the verification gates recorded in the GitHub issue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The linked GitHub issue owns current count and public status.
- [x] #2 Implementation PR tasks are created under this child when the area is too broad for one PR.
- [x] #3 Backlog notes record PR links and before/after count evidence.
<!-- AC:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the Jobs/Scheduler/Watchlists product-state migration tracker. The area moved from 24 baseline exceptions at the start of this tracker to zero after TASK-45.44.3.14, with PR/task notes recording each migration slice, before/after counts, and focused verification. Remaining product-state exceptions are outside this tracker area.
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
