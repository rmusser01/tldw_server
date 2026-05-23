---
id: TASK-488
title: Add OmniVoice managed sidecar smoke test helper
status: Done
assignee: []
created_date: '2026-05-23T01:10:00Z'
updated_date: '2026-05-23T01:55:00Z'
labels: []
dependencies: []
references:
  - 'https://github.com/rmusser01/tldw_server/pull/1969'
documentation:
  - Docs/STT-TTS/TTS-SETUP-GUIDE.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add a repo-native operator smoke utility for real OmniVoice managed sidecar synthesis, with focused tests and TTS setup guide updates documenting model/runtime configuration and verification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Smoke helper exercises the managed OmniVoice sidecar path through OmniVoiceSidecarSupervisor and OmniVoiceAdapter.
- [x] #2 Smoke helper validates parseable non-silent 24 kHz mono WAV output and records real synthesis evidence.
- [x] #3 Focused unit tests, adjacent OmniVoice tests, Bandit, helper --help, and diff checks are recorded before completion.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-22-omnivoice-sidecar-smoke-helper-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented a repo-native OmniVoice managed sidecar smoke helper that exercises OmniVoiceSidecarSupervisor plus OmniVoiceAdapter, preserves sidecar Python symlinks when building config, validates parseable non-silent 24 kHz mono WAV output, and always shuts down the supervisor. Updated the TTS setup guide with model path examples, provider config, smoke command, opt-in pytest caveats, and common failure notes. Addressed PR #1969 review fixes with typed/docstring cleanup, bounded WAV sample analysis, executable sidecar-python validation, positive speed/num_step validation, shutdown error preservation, and Backlog metadata corrections.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added review fixes for PR #1969 on top of the OmniVoice managed sidecar smoke helper. The helper now validates executable sidecar Python plus positive speed/num_step controls, limits WAV sample analysis to a bounded window, preserves the primary failure if supervisor shutdown also fails, reports shutdown-only failures explicitly, and includes the requested typing/docstring/line-length cleanups. Verification: smoke helper unit suite 16 passed; adjacent OmniVoice installer/supervisor tests 40 passed; helper --help exited 0; py_compile exited 0; git diff --check exited 0; line-length scan found no >100 character lines in touched helper/test files; Bandit helper and test reports had 0 findings; real managed sidecar smoke synthesized /private/tmp/omnivoice-helper-sidecar-smoke-review.wav as 157964 bytes, 24000 Hz mono, 78960 frames, RMS 3922.93, peak 16384. Known note: the underlying runtime still emits a multiprocessing resource_tracker semaphore warning after successful synthesis.
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
