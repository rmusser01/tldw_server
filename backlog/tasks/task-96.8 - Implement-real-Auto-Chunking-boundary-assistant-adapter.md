---
id: TASK-96.8
title: Implement real Auto Chunking boundary assistant adapter
status: To Do
assignee: []
created_date: '2026-05-06 17:53'
labels:
  - backend
  - chunking
  - auto-chunking
  - llm
dependencies:
  - TASK-96.7
documentation:
  - Docs/superpowers/specs/2026-05-06-auto-chunking-design.md
  - Docs/superpowers/plans/2026-05-06-auto-chunking-implementation-plan.md
parent_task_id: TASK-96
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-up for Auto Chunking V1. Add a real LLM-backed boundary assistant only after defining an explicit adapter interface and availability checks. The adapter should refine boundaries or labels from deterministic candidate plans and must fall back deterministically on provider, timeout, config, or runtime errors.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Define a narrow AutoChunkBoundaryAssistant interface and result type before adding provider calls.
- [ ] #2 Availability checks are explicit and do not rely only on provider keys being configured.
- [ ] #3 Adapter is invoked only when auto_chunking_use_llm=true.
- [ ] #4 Timeout, provider error, and invalid response paths preserve deterministic Auto plans with fallback metadata.
- [ ] #5 Tests cover default no-call behavior, explicit opt-in success, timeout/error fallback, and metadata used_llm semantics.
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
