---
id: TASK-45.44.15
title: Define design-system CI gate tightening path
status: Done
assignee: []
created_date: 2026-05-14 03:20
labels:
- design-system
- webui
- extension
- governance
dependencies: []
references:
- https://github.com/rmusser01/tldw_server/issues/1672
- Docs/superpowers/specs/2026-05-14-design-system-remaining-work-tracker-design.md
- Docs/Design/tldw_web_design_system_ci_gate_path.md
- https://github.com/rmusser01/tldw_server/pull/2626
parent_task_id: TASK-45.44
priority: medium
modified_files:
- Docs/Design/tldw_web_design_system_ci_gate_path.md
- Docs/Design/tldw_web_design_system_baseline_reporting.md
- backlog/tasks/task-45.44.15 - Define-design-system-CI-gate-tightening-path.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Mirror the linked GitHub governance issue. Closure requires a durable guard, documented policy, CI path, component ownership decision, documentation artifact, or visual QA checklist as specified by the GitHub issue.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The linked GitHub issue owns public status.
- [x] #2 Backlog notes record PR links and verification evidence.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- Added `Docs/Design/tldw_web_design_system_ci_gate_path.md` defining the staged path from report-only PR signal to required new-finding gate, stale-baseline cleanup, and area-zero enforcement.
- Chose `frontend-required` as the future CI owner instead of adding a new workflow, because it already detects frontend changes and installs frontend dependencies.
- Linked the CI gate path from `tldw_web_design_system_baseline_reporting.md` so migration PR instructions point at the durable artifact.
- Rebased PR #2626 onto `origin/dev` at `83428eff33`.
- Addressed review feedback by tying the report-only trigger to the current `tldw_frontend_changed` classifier paths, documenting that docs-only changes do not trigger it, linking the concrete verifier script, and replacing plain document references with relative Markdown links.
- Verification: `git diff --check` passed and all linked repository targets exist.
- `bun run verify:design-system-state` runs but exits 1 on current `dev` debt: unbaselined product-state findings plus 14 stale baseline rows. This docs-only PR does not alter product-state source or baseline data.
- Bandit is not applicable because this task changes only markdown documentation and Backlog metadata.
- PR: https://github.com/rmusser01/tldw_server/pull/2626
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Defined the design-system CI gate tightening path in a durable design doc and linked it from the baseline reporting workflow. Review remediation now anchors trigger behavior to the current path classifier, names the concrete verifier owner, and uses navigable document links. The branch was rebased onto current `dev`; the verifier's existing `dev` debt is recorded as an out-of-scope baseline blocker.
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
