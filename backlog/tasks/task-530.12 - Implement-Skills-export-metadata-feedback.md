---
id: TASK-530.12
title: Implement Skills export metadata feedback
status: In Progress
labels:
- skills
- webui
- safe-operations
- frontend
priority: high
parent_task_id: TASK-530
documentation:
- Docs/superpowers/specs/2026-06-29-skills-export-metadata-feedback-design.md
- Docs/superpowers/plans/2026-06-29-skills-export-metadata-feedback.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue TASK-530 Safe Operations after TASK-530.11 by preserving Skills export response metadata through the frontend client and using the server-provided filename for downloads and success feedback. Keep scope limited to export metadata, filename fallback/safety, and user feedback; do not add bulk export or permission/model metadata panels.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Skills export API client returns both Blob and filename metadata parsed from Content-Disposition when available.
- [ ] #2 Skills export API client falls back to a safe `<skill>.zip` filename when the header is missing, malformed, or unsafe.
- [ ] #3 Skills manager uses the returned filename for the browser download and shows success feedback naming the actual file.
- [ ] #4 Existing sanitized export failure feedback remains unchanged.
- [ ] #5 Focused frontend tests cover metadata filename, fallback filename, success feedback, and error feedback.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

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
