---
id: TASK-243
title: Design VN scripted model expansion runtime
status: In Progress
assignee: []
created_date: '2026-05-10 19:52'
labels:
  - vn
  - design
  - scripted-generation
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/issues/1535'
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create the design spec for GitHub issue #1535: backend-owned model expansion and regeneration for VN scripted_story sessions, plus the follow-on WebUI generation-history inspector path. This is design-first before implementation and should split backend/runtime and WebUI work into separate PR-sized paths.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Design defines backend generation requests, immutable revisions, active revision semantics, confirmation/cancel/regenerate/activate commands, idempotency, failures, moderation, checkpoints, and generated-choice behavior.
- [ ] #2 Design defines the dedicated generation-history and debug APIs plus the WebUI generation-history route expectations.
- [ ] #3 Design explicitly splits backend runtime/API implementation from WebUI inspector implementation.
- [ ] #4 Spec is written under Docs/superpowers/specs and reviewed before implementation planning.
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
