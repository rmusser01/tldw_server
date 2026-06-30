---
id: TASK-453
title: Design OmniVoice managed sidecar real synthesis completion
status: Done
labels:
- tts
- omnivoice
- design
references:
- https://github.com/k2-fsa/OmniVoice
documentation:
- Docs/superpowers/specs/2026-05-22-omnivoice-real-sidecar-synthesis-design.md
modified_files:
- Docs/superpowers/specs/2026-05-22-omnivoice-real-sidecar-synthesis-design.md
- backlog/tasks/task-453 - Design-OmniVoice-managed-sidecar-real-synthesis-completion.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Write a design spec for finishing the existing managed OmniVoice TTS sidecar integration so it produces real OmniVoice audio instead of the current contract-stub silent WAV behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Created and reviewed the OmniVoice real sidecar synthesis design spec. The approved direction is to finish the existing managed sidecar using OmniVoice's Python API, explicit install/verify provisioning, auto/design/clone support, language passthrough, one configured runtime, required reference_text for cloning, structured errors, and opt-in real-runtime tests. A post-review hardening pass fixed two planning risks: the no-download policy now requires a resolved local model directory before from_pretrained, and the sidecar language_id field must map to OmniVoice's Python API language argument.
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
