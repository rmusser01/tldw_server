---
id: TASK-104
title: Create WebUI dependency audit for issue 1346
status: Done
assignee: []
created_date: '2026-05-07 01:36'
updated_date: '2026-05-07 03:04'
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
Task 4 closeout plan: verify the existing audit follow-up sections against the final ranking requirements; make only narrow wording edits where needed; remove the temporary Task 4 pending blocker; record final verification and docs-only Bandit skip; close TASK-104 as Done; commit only Docs/Design/WebUI_Dependency_Audit.md and this Backlog task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Wrote implementation plan for the first issue #1346 work unit: create Docs/Design/WebUI_Dependency_Audit.md without package or runtime edits.

Plan review before commit found and fixed two issues: usage-scan commands now exclude manifests/docs/generated artifacts so declaration mentions are not counted as usage, and Backlog documentation commands now preserve both spec and plan links. Final task closeout command now checks DoD and marks TASK-104 Done after audit completion.

Started dependency audit implementation plan. First slice is docs-only audit artifact; package removals and axios replacement are follow-up work.

Quality fix: added TASK-100 parent design task reference to Docs/Design/WebUI_Dependency_Audit.md so the audit traces both design approval and execution work.

Task 2 progress generated the temporary declaration JSON from the three manifests. Observed 138 unique declarations total. Populated the audit table with 125 WebUI and shared UI declarations. Left Task 3 fields as TBD. Excluded 13 extension only declarations because extension remains an impact check surface.

Task 3 started in worktree branch codex/webui-dependency-trim-1346. Scope remains docs-only: audit document and TASK-104 updates only.

Task 3 progress generated /tmp/tldw-webui-dependency-usage.json from 4605 scanned files across apps/tldw-frontend, apps/packages/ui, and apps/extension. Classified all 125 audited WebUI/shared UI rows, ranked quick cleanup and replacement candidates, and recorded quick-candidate precision caveats and Bandit docs-only skip in the audit.

Review correction started: spec review found false-positive package counts from broad text/config-key matching, especially generic names like next. Reopening task status while the import/config-aware table correction is applied.

Review correction progress regenerated the table from strict import/config/package-script evidence. Corrected generic-name false positives across the table; next now has WebUI-only evidence and no shared UI local-variable matches. Quick candidates remain pubsub-js/buffer/stream-browserify remove-now, clsx and axios remain replace-later.

Task 3 data-quality correction started: review found postcss-import plugin-key evidence was missing and pdfjs-dist had a count/site mismatch between direct script evidence and runtime worker/version reference. Keeping TASK-104 In Progress for Task 4/final closeout.

Task 3 data-quality correction completed: postcss-import now counts the direct PostCSS plugin key in apps/tldw-frontend/postcss.config.mjs, and pdfjs-dist now aligns import count, representative sites, and consumer surface across copy-pdf-worker.mjs plus the shared PdfDocument runtime worker/version reference. TASK-104 remains In Progress for Task 4/final closeout.

Task 4 closeout completed: confirmed the ranked follow-up queue already satisfied the plan, made only narrow wording edits to clarify cleanup attempt order and follow-up verification, and removed the temporary Task 4 pending blocker from Known Skips. Final verification run: rg -n "pubsub-js|buffer|stream-browserify|clsx|axios|dompurify|defer-design|remove-now|replace-later" Docs/Design/WebUI_Dependency_Audit.md; git diff --check; git status --short; merge-policy gate checked. Bandit remains skipped because this slice changed only documentation and Backlog task metadata, with no Python or runtime code changes. A human-authored Change summary is not present in this AI-authored closeout, so the PR must remain draft/not merge-ready until the requester supplies it.

PR review correction completed: restacked the branch on current origin/dev to remove unrelated CodeGraph/workflow documentation from the PR diff, renamed this task to TASK-104 to avoid task IDs already present on dev and sibling review work, reclassified zero-evidence complex package declarations as investigate-lockfile rather than defer-design, and recorded the AI-authored PR merge gate. A human-authored Change summary is still required before this PR is merge-ready; this task records the gate but does not satisfy it.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed the WebUI dependency audit for issue #1346. The audit links the issue, TASK-100, and the approved design/plan; covers WebUI/shared UI declarations plus extension impact checks; records the required package evidence table; explicitly keeps, defers, or routes complex/security-sensitive stacks to lockfile investigation; ranks quick cleanup and replacement follow-ups without changing package manifests; records final verification plus the docs-only Bandit skip; and records that a human-authored Change summary is required before the AI-authored PR is merge-ready.
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
