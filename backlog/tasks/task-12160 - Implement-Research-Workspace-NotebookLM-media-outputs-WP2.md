---
id: TASK-12160
title: Implement Research Workspace NotebookLM media outputs WP2
status: In Progress
assignee: []
created_date: ''
updated_date: 2026-07-05 05:57
labels: []
dependencies: []
references:
- Docs/superpowers/specs/2026-07-05-research-workspace-notebooklm-media-outputs-wp2-design.md
modified_files:
- tldw_Server_API/app/api/v1/schemas/research_workspace_capabilities.py
- tldw_Server_API/app/core/Research_Workspace/capabilities.py
- tldw_Server_API/tests/Research_Workspace/test_capability_derivation.py
- tldw_Server_API/app/api/v1/schemas/research_workspace_outputs.py
- tldw_Server_API/app/core/Research_Workspace/output_jobs.py
- tldw_Server_API/app/api/v1/endpoints/workspaces.py
- tldw_Server_API/tests/Research_Workspace/test_output_jobs_api.py
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Task 2 started: adding Research Workspace output job schemas/API skeleton only; worker/media generation/persistence remain for later plan tasks.
Task 2 complete: added output submit/status schemas, minimal Research Workspace output job helpers, POST/GET workspace output routes, and focused tests for validation, pending artifact/job creation, status projection, and status ownership isolation. Verification: red pytest failed on missing research_workspace_outputs module; green pytest tldw_Server_API/tests/Research_Workspace/test_output_jobs_api.py -v passed 7 tests; git diff --check passed; Bandit on touched backend implementation files passed with 0 results.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
