---
id: TASK-12070
title: Enable verbose TestPyPI publish diagnostics
status: Done
assignee: []
created_date: '2026-06-29 19:36'
updated_date: '2026-06-29 19:41'
labels:
  - packaging
  - pypi
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Temporarily enable verbose output for the backend/API PyPI publish workflow's TestPyPI upload step so the TestPyPI 400 Bad Request response body can be diagnosed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 TestPyPI publish action emits verbose upload failure details.
- [x] #2 Workflow is rerun against the diagnostic branch and the resulting TestPyPI failure/success is recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Changed .github/workflows/publish-pypi.yml to pass verbose: true to the TestPyPI publish action, pushed codex/testpypi-verbose-diagnostics, and reran publish-pypi.yml with target=testpypi. Run 28397789395 built distributions successfully but TestPyPI upload failed with a verbose 400 response.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Verbose TestPyPI diagnostics captured. The publish action validates both dist files, then TestPyPI rejects tldw_server-0.1.32 because package metadata contains a direct VCS dependency: dots-ocr @ git+https://github.com/rednote-hilab/dots.ocr.git ; extra == "ocr-dots". This is a package metadata issue, not a trusted-publishing/environment issue. Bandit was skipped because this task only changes GitHub Actions YAML and Backlog metadata.
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
