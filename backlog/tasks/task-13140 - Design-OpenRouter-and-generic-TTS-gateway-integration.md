---
id: TASK-13140
title: Design OpenRouter and generic TTS gateway integration
status: Done
assignee: []
created_date: ''
updated_date: '2026-08-28 05:22'
labels:
  - tts
  - openrouter
  - design
dependencies: []
references:
  - 'https://openrouter.ai/docs/guides/overview/multimodal/tts'
  - 'https://openrouter.ai/docs/api/api-reference/speech/create-audio-speech'
  - 'https://openrouter.ai/docs/guides/overview/models'
  - 'commit:5e0c199931'
  - 'commit:53fda5cd4f'
  - 'commit:5b8d951c26'
  - 'commit:13e3e3aceb'
documentation:
  - Docs/Design/TTS.md
  - Docs/superpowers/specs/2026-07-15-openrouter-tts-gateway-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Consolidate the approved design for first-class OpenRouter TTS plus multiple named OpenAI-compatible speech gateways. Scope is design and implementation planning only: explicit backend selection, config-first gateway definitions, discovery with overlays, admin-controlled URLs, optional user API keys, per-backend fallback, buffered-only conversion, per-gateway request-option allowlists, API/WebUI compatibility, testing, security, and rollout.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Approved decisions are captured in a design spec under Docs/superpowers/specs
- [x] #2 The design covers backend/API/config/runtime/discovery/WebUI/fallback/conversion/BYOK/security/testing/rollout contracts
- [x] #3 A dedicated spec reviewer reports no blocking issues
- [x] #4 The user reviews and approves the written spec before implementation planning begins
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-07-16-openrouter-tts-gateway-implementation-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
User approved the written spec on 2026-07-16. Final spec review approved with no blocking findings. Implementation plan completed and independently reviewed through three whole-plan passes; final status Approved with no issues or recommendations. Created child implementation task TASK-12116.1 in To Do. Verification: git diff --check passed for planning artifacts. Bandit skipped because this task changes only Markdown/Backlog planning metadata and no executable code.

2026-08-28 TASK-13013.10 identity normalization: this completed design record moved from legacy TASK-12116 to canonical TASK-13140 so the active frontend release-safety task remains the sole TASK-12116 record. The implementation child remains TASK-12116.1 and now explicitly points to TASK-13140; design and implementation documentation use TASK-13140 for this parent. Historical commit subjects and immutable evidence may still mention legacy TASK-12116.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Completed and user-approved the OpenRouter plus generic named TTS gateway design. Produced an implementation-ready TDD plan covering config normalization, dynamic registry, transport/audio validation, credential-scoped discovery, BYOK, explicit routing, bounded fallback/conversion, API/catalogs, history/presets/jobs including server audiobooks, WebUI capability negotiation and persistence, security validation, documentation, and final review. Implementation is tracked separately in TASK-12116.1.
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
