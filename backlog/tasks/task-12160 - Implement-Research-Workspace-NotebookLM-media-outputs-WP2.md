---
id: TASK-12160
title: Implement Research Workspace NotebookLM media outputs WP2
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-05 05:57'
labels: []
dependencies: []
references:
  - >-
    Docs/superpowers/specs/2026-07-05-research-workspace-notebooklm-media-outputs-wp2-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track WP2 work for real NotebookLM-style media outputs in Research Workspace: backend jobs for narrated slideshow Video Overview and image-backend generated Infographic artifacts. Supersedes the earlier cheap text/storyboard WP2 idea; outputs must be real media artifacts.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Video Overview generates a real backend-rendered narrated slideshow artifact with durable preview/download.
- [ ] #2 Infographic generates a real image-backend PNG artifact with durable preview/download.
- [ ] #3 Research Workspace output jobs are submitted, drained by a registered worker, and expose status/progress/errors.
- [ ] #4 Final media and per-slide narration assets use durable output artifacts, not TTL file-artifact export URLs or generated-file ids alone.
- [ ] #5 UI capability gates, pending states, completed previews, and unavailable states are covered by tests.
- [ ] #6 Backend validation, worker, capability, and persistence paths are covered by tests.
- [ ] #7 Bandit and targeted frontend/backend verification are recorded before completion.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Implementation plan approved: Docs/superpowers/plans/2026-07-05-research-workspace-notebooklm-media-outputs-wp2-plan.md. Plan review status: Approved.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
