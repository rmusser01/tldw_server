---
id: TASK-12957
title: Migrate summarization media and audio service prompts
status: To Do
assignee: []
created_date: '2026-07-14 00:13'
updated_date: '2026-07-14 00:17'
labels:
  - prompts
  - service-prompts
  - domain-migration
  - backend
dependencies:
  - TASK-12958
references:
  - TASK-12956
  - TASK-12958
documentation:
  - >-
    Docs/superpowers/specs/2026-07-12-user-customizable-service-prompts-design.md
  - Docs/Design/service-prompt-inventory.md
  - >-
    Docs/superpowers/plans/2026-07-13-service-prompts-domain-summarization-media-audio.md
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 All 16 approved summarization/media/audio IDs are registered and every named consumer resolves one immutable atomic bundle.
- [ ] #2 Tests prove byte-equivalent defaults, precedence, literal/template behavior, locked fragments, provenance, and exact size limits.
- [ ] #3 Direct, async, and protected Jobs call sites satisfy the approved inventory contract and all implementation gates pass.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Execute Docs/superpowers/plans/2026-07-13-service-prompts-domain-summarization-media-audio.md after the Service Prompts foundation and TASK-12958 inventory are complete.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
First rollout domain; implementation remains To Do. The planning-time CI-shard skip does not waive any implementation gate.
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
