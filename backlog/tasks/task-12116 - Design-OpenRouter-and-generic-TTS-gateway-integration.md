---
id: TASK-12116
title: Design OpenRouter and generic TTS gateway integration
status: In Progress
labels:
- tts
- openrouter
- design
priority: high
references:
- https://openrouter.ai/docs/guides/overview/multimodal/tts
- https://openrouter.ai/docs/api/api-reference/speech/create-audio-speech
- https://openrouter.ai/docs/guides/overview/models
- commit:5e0c199931
- commit:53fda5cd4f
- commit:5b8d951c26
- commit:13e3e3aceb
documentation:
- Docs/Design/TTS.md
- Docs/superpowers/specs/2026-07-15-openrouter-tts-gateway-design.md
modified_files:
- Docs/superpowers/specs/2026-07-15-openrouter-tts-gateway-design.md
- backlog/tasks/task-12116 - Design-OpenRouter-and-generic-TTS-gateway-integration.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Consolidate the approved design for first-class OpenRouter TTS plus multiple named OpenAI-compatible speech gateways. Scope is design and implementation planning only: explicit backend selection, config-first gateway definitions, discovery with overlays, admin-controlled URLs, optional user API keys, per-backend fallback, buffered-only conversion, per-gateway request-option allowlists, API/WebUI compatibility, testing, security, and rollout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Approved decisions are captured in a design spec under Docs/superpowers/specs
- [ ] #2 The design covers backend/API/config/runtime/discovery/WebUI/fallback/conversion/BYOK/security/testing/rollout contracts
- [ ] #3 A dedicated spec reviewer reports no blocking issues
- [ ] #4 The user reviews and approves the written spec before implementation planning begins
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
