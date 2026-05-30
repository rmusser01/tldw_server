---
id: TASK-426
title: Design first-time model readiness setup flow
status: Done
labels:
- design
- setup
- webui
- embeddings
- audio
documentation:
- Docs/superpowers/specs/2026-05-18-first-time-model-readiness-setup-design.md
modified_files:
- Docs/superpowers/specs/2026-05-18-first-time-model-readiness-setup-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Create a design spec for a first-time setup/readiness flow that exposes curated choices for chat provider/endpoint/model, embedding model provisioning, transcription readiness, and secondary TTS readiness through backend /setup and native WebUI first-run setup using shared /api/v1/setup/* APIs.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Spec captures the unified first-time model readiness setup architecture across backend `/setup` and native WebUI setup.
- [x] #2 Spec covers chat, embeddings/RAG, speech readiness lanes, and secondary TTS readiness semantics.
- [x] #3 Spec records provisioning consent, permissions, error handling, secret handling, admin/local setup boundaries, and trusted custom-model safeguards.
- [x] #4 Verification and the implementation-deferred boundary are recorded.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Design-only task. Spec captures the approved first-time model readiness setup architecture, components, data flow, permissions, error handling, and testing plan. No runtime implementation in this task.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Closed this design-only task after confirming the spec and task summary already captured the approved first-time readiness setup direction.

- Design artifact: `Docs/superpowers/specs/2026-05-18-first-time-model-readiness-setup-design.md`.
- Updated the spec status from review-ready to design complete with implementation deferred.
- This closeout changes only Markdown documentation and Backlog task metadata. Bandit is not applicable.

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closed the first-time model readiness setup design task. The design spec records the approved unified readiness wizard direction, native WebUI backed by setup APIs, curated profiles, explicit Provision now gating, lane readiness semantics, permission boundaries, error handling, and test strategy. A follow-up critique pass hardened restart/admin/remote-blocked overlays, secondary TTS handling, WebUI fallback, pollable provisioning, secret handling, and trusted custom-model acknowledgement. Runtime implementation remains deferred to follow-up implementation tasks.
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
