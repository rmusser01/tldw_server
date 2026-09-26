---
id: TASK-13376
title: Qualify complete-app provider setup and first-document workflow
status: To Do
assignee: []
created_date: '2026-09-26 16:38'
labels:
  - distribution
  - qualification
  - webui
dependencies:
  - TASK-13343
references:
  - Docs/Design/2026-09-20-complete-app-distribution-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Explicit unresolved product acceptance checkpoint from the user-approved review corrections. TASK-13265 is the completed design task and cannot stand in for pending implementation or qualification. Initial wizard progression in WP1 does not prove that a newcomer can configure a provider and use documents. This task qualifies that ordinary browser workflow before an installer is advertised as a complete usable application; it does not authorize publication or change native-platform/core-format requirements.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 From a fresh signed extracted paired candidate outside a checkout, complete ordinary provider setup with a deterministic mock using the real WebUI and no manual backend master key or frontend-server URL wiring.
- [ ] #2 Use ordinary UI controls to ingest a Markdown document, find its content through search, and complete a chat with the configured mock provider; fail on setup or application errors.
- [ ] #3 Stop/start preserves provider configuration and document data; record exact candidate source, platform, browser and novice instructions with full-setup evidence distinct from initial-wizard checks.
- [ ] #4 Keep full-provider/document qualification false until all required workflows pass; retain the complete native platform and core-format matrix as separate required product gates.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Created as an explicit acceptance checkpoint while correcting TASK-13343. No test success is claimed and no requirement is waived.
<!-- SECTION:NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
