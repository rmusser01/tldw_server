---
id: TASK-12070
title: Enable verbose TestPyPI publish diagnostics
status: To Do
assignee: []
created_date: '2026-06-29 19:36'
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
- [ ] #1 TestPyPI publish action emits verbose upload failure details.
- [ ] #2 Workflow is rerun against the diagnostic branch and the resulting TestPyPI failure/success is recorded.
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
