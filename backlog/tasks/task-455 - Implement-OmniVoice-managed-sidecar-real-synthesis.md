---
id: TASK-455
title: Implement OmniVoice managed sidecar real synthesis
status: In Progress
assignee: []
created_date: ''
updated_date: '2026-05-22 20:21'
labels:
  - tts
  - omnivoice
  - implementation
dependencies: []
references:
  - TASK-453
  - TASK-454
  - Docs/superpowers/specs/2026-05-22-omnivoice-real-sidecar-synthesis-design.md
documentation:
  - >-
    Docs/superpowers/plans/2026-05-22-omnivoice-real-sidecar-synthesis-implementation-plan.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Execute the approved implementation plan to finish the existing managed OmniVoice TTS sidecar so it uses the real OmniVoice Python API instead of returning stub silent WAV audio.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Task 2 implemented: normalized OmniVoice adapter sidecar payloads to canonical keys with design/clone conflict validation, generation object allowlist/coercion, scratch-dir direct reference materialization, native sample-rate header handling, structured sidecar error mapping, OmniVoice validation passthrough/parameter checks, and service no-fallback policy for explicit OmniVoice semantics.

Verification recorded for Task 2: red run failed 9 expected tests; focused suite later passed 29 tests; nearby OmniVoice protocol/registry/service sanitization checks passed 19 selected tests; Bandit code/tests returned 0 findings; scoped diff check passed. Full git diff --check is blocked by unrelated pre-existing trailing whitespace in Docs/Design/Agents.md.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->
<!-- SECTION:FINAL_SUMMARY:END -->

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
