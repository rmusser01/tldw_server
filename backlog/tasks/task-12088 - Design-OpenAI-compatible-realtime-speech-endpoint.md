---
id: TASK-12088
title: Design OpenAI-compatible realtime speech endpoint
status: Done
labels:
- audio
- realtime
- design
references:
- https://github.com/huggingface/speech-to-speech
- https://developers.openai.com/api/docs/guides/realtime
- https://developers.openai.com/api/docs/guides/realtime-conversations#handling-audio-with-websockets
documentation:
- Docs/superpowers/specs/2026-07-01-openai-realtime-speech-endpoint-design.md
- Docs/superpowers/plans/2026-07-01-openai-realtime-speech-endpoint-implementation-plan.md
modified_files:
- Docs/superpowers/specs/2026-07-01-openai-realtime-speech-endpoint-design.md
- Docs/superpowers/plans/2026-07-01-openai-realtime-speech-endpoint-implementation-plan.md
- backlog/tasks/task-12088 - Design-OpenAI-compatible-realtime-speech-endpoint.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write the design spec for an adapter-first OpenAI GA Realtime-compatible speech-to-speech WebSocket layer over the existing audio pipeline, including route strategy, auth, protocol boundaries, cancellation identifiers, testing, and rollout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-01-openai-realtime-speech-endpoint-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- 2026-07-01: Design accepted for implementation. Runtime work moved forward under TASK-12089 with the adapter-first route strategy, Stage 1 protocol boundary, identifier model, auth behavior, capability metadata, and default pipeline integration.
- 2026-07-01: Bandit is not applicable to this design-only task. Implementation security verification is recorded on TASK-12089.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Accepted the OpenAI-compatible realtime speech endpoint design and linked it to the implementation plan used by TASK-12089. The design remains the reference for the Stage 1 compatibility scope and the deferred Stage 2 latency/interruption work.
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
