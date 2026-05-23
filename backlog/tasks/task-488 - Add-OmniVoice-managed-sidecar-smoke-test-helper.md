---
id: TASK-488
title: Add OmniVoice managed sidecar smoke test helper
status: Done
assignee: []
created_date: ''
updated_date: '2026-05-23 01:10'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1958'
documentation:
  - Docs/STT-TTS/TTS-SETUP-GUIDE.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a repo-native operator smoke utility for real OmniVoice managed sidecar synthesis, with focused tests and TTS setup guide updates documenting model/runtime configuration and verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-22-omnivoice-sidecar-smoke-helper-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a repo-native OmniVoice managed sidecar smoke helper that exercises OmniVoiceSidecarSupervisor plus OmniVoiceAdapter, preserves sidecar Python symlinks when building config, validates parseable non-silent 24 kHz mono WAV output, and always shuts down the supervisor. Updated the TTS setup guide with model path examples, provider config, smoke command, opt-in pytest caveats, and common failure notes.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added Helper_Scripts/TTS_Installers/smoke_test_omnivoice_sidecar.py with focused unit coverage in tldw_Server_API/tests/TTS_NEW/unit/test_omnivoice_sidecar_smoke.py and documentation updates in Docs/STT-TTS/TTS-SETUP-GUIDE.md. Verification after rebasing onto origin/dev: smoke helper unit suite 10 passed; adjacent OmniVoice installer/supervisor tests 40 passed; helper --help exited 0; git diff --check exited 0; Bandit helper report had 0 findings; Bandit test report had 0 findings with pytest assertion check B101 skipped; real managed sidecar smoke synthesized /private/tmp/omnivoice-helper-sidecar-smoke-rebase.wav as 158924 bytes, 24000 Hz mono, 79440 frames, RMS 2425.56, peak 16384. Known note: the underlying runtime still emits a multiprocessing resource_tracker semaphore warning after successful synthesis.
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
