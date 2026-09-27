---
id: TASK-13373
title: Move provider-adapter SSE loops onto streaming.iter_sse_lines_requests
status: To Do
assignee: []
created_date: '2026-09-24 02:13'
labels:
  - tech-debt
  - streaming
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Nine provider adapters keep their own SSE read loops instead of streaming.iter_sse_lines_requests / aiter helpers. Migrating changes decode (errors='replace') and error-frame behaviour, so it needs per-adapter tests pinning the current frames first. Split from TASK-13370.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each migrated adapter has a test pinning its error frame and decode behaviour before the switch
- [ ] #2 No adapter keeps a hand-rolled SSE loop, or the exceptions are documented
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
