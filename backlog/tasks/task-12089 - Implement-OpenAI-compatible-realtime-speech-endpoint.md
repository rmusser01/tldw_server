---
id: TASK-12089
title: Implement OpenAI-compatible realtime speech endpoint
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-07-01 21:20'
labels:
  - audio
  - realtime
  - implementation
dependencies: []
references:
  - TASK-12088
  - 'https://github.com/huggingface/speech-to-speech'
  - 'https://developers.openai.com/api/docs/guides/realtime'
  - >-
    https://developers.openai.com/api/docs/guides/realtime-conversations#handling-audio-with-websockets
documentation:
  - Docs/superpowers/specs/2026-07-01-openai-realtime-speech-endpoint-design.md
  - >-
    Docs/superpowers/plans/2026-07-01-openai-realtime-speech-endpoint-implementation-plan.md
modified_files:
  - backlog/tasks/task-12089 - Implement-OpenAI-compatible-realtime-speech-endpoint.md
  - Docs/superpowers/plans/2026-07-01-openai-realtime-speech-endpoint-implementation-plan.md
  - tldw_Server_API/app/core/Audio/Realtime/__init__.py
  - tldw_Server_API/app/core/Audio/Realtime/constants.py
  - tldw_Server_API/app/core/Audio/Realtime/models.py
  - tldw_Server_API/app/core/Audio/Realtime/protocol.py
  - tldw_Server_API/app/core/Audio/Realtime/capabilities.py
  - tldw_Server_API/app/core/Audio/Realtime/pipeline.py
  - tldw_Server_API/app/core/Audio/Realtime/session.py
  - tldw_Server_API/app/core/Audio/Realtime/persistence.py
  - tldw_Server_API/tests/Audio/test_realtime_protocol_adapter.py
  - tldw_Server_API/tests/Audio/test_realtime_capabilities.py
  - tldw_Server_API/tests/Audio/test_realtime_session.py
  - tldw_Server_API/tests/Audio/test_realtime_persistence.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the adapter-first OpenAI GA Realtime-compatible speech-to-speech WebSocket endpoint plan, including protocol/capabilities, session orchestration, auth and route integration, default pipeline adapter, docs, tests, and verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-01-openai-realtime-speech-endpoint-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Stage 1 complete. Implemented protocol constants, dataclass command/event models, OpenAI GA protocol parser/serializer, capabilities metadata, and provider-free tests. Verification: baseline focused tests passed before implementation (21 passed); Stage 1 tests passed after fixes (43 passed, 3 warnings); spec compliance review passed; code-quality review passed with no Critical or Important findings. Bandit production Realtime package reported errors=0 results=0. Minor hardening candidate: reject stray top-level beta audio fields consistently across event types.

Stage 2 complete. Implemented provider-free realtime pipeline event protocol, internal session orchestrator, manual audio turn lifecycle, response generation/cancellation guards, stale-output suppression, metadata merging, and optional persistence boundary. Verification: focused Stage 2 session+persistence tests passed locally (19 passed, 3 warnings); implementer reported expanded focused slice 72 passed; spec compliance review passed at HEAD 25c6a585; code-quality re-review passed with no Critical or Important findings; Bandit on tldw_Server_API/app/core/Audio/Realtime reported errors=0 results=0. The final persistence fix snapshots RealtimePersistenceConfig before yielding response.done so late session.update cannot misattribute a completed turn.
<!-- SECTION:NOTES:END -->

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
