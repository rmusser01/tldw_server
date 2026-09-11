---
id: TASK-13199
title: Keep Persona microphone cadence within the audio limit and recover on throttling
status: Done
assignee: []
created_date: 2026-09-06 00:30
updated_date: 2026-09-06 02:11
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Browser voice sends 4096 PCM samples at 16 kHz, exceeding the default 120 chunks per minute during ordinary continuous capture. A rejected stream keeps recording and floods the transcript with warnings.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Default admission supports at least a full minute of the shipped browser capture cadence while configured lower limits remain enforced.
- [x] #2 A matching audio rate-limit notice releases microphone ownership, prevents further audio sends, and gives a retry message.
- [x] #3 Focused regressions and source-bound UAT limitations are documented.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: amend existing Docs/ADR/046-persona-live-conversation-and-voice-runtime.md. Reason: default rate policy at the existing voice boundary; preserve explicit operator overrides. Reproduce browser cadence and rate-limit recovery in tests, set the bounded default to 300 chunks per minute, handle throttling through existing ownership teardown, verify and document. Raising the default is preferred over batching capture because batching adds latency and residual-buffer lifecycle complexity.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Raised the bounded default to 300 chunks/minute to cover the shipped 16 kHz/4096-sample browser cadence; explicit lower settings still apply. Owned throttling notices stop capture/playback, retire voice authority and present an explicit wait-and-retry message. Both regressions failed before repair. Final verification: 126 backend and 72 frontend tests pass; Bandit zero findings; Ruff four unchanged baseline findings; ESLint no rule findings (existing pages-directory configuration notice); scoped TypeScript zero owned errors and 27 existing dependency diagnostics. ADR046 and user guide/mirrors updated. Prior corrected browser run recorded provisional non-replayed words but was canceled after an operator timing overrun; actual short-window acceptance remains under TASK-13202/13198.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
