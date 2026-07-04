---
id: TASK-12147
title: Harden audio briefing generation before Research Workspace reuse
status: Done
priority: High
modified_files:
- tldw_Server_API/app/core/Workflows/adapters/audio/multi_voice_tts.py
- tldw_Server_API/app/core/Workflows/adapters/audio/_config.py
- tldw_Server_API/app/core/Watchlists/audio_briefing_workflow.py
- tldw_Server_API/tests/Workflows/adapters/test_audio_adapters.py
- tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py
references:
- TASK-12148
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the existing audio briefing/TTS workflow so Research Workspace can reuse it without reporting partial assets as successful, while preserving explicit TTS provider selection for future briefing flows.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 multi_voice_tts fails visibly when final audio concatenation fails instead of promoting a single segment as final output
- [ ] #2 explicit TTS provider selection is propagated through Watchlists audio briefing inputs into the internal TTS generation call
- [ ] #3 tests cover concat failure behavior and provider propagation without external HTTP calls
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented shared audio briefing hardening before Research Workspace reuse. multi_voice_tts now fails the workflow step on final concat failure instead of promoting a single section as the final artifact, allowing the existing single-voice fallback route to run. Watchlists audio briefing now preserves the resolved TTS provider through workflow inputs and into both multi-voice and fallback TTS steps. Verification: targeted red tests failed before implementation; focused regression tests passed; `python -m pytest tldw_Server_API/tests/Workflows/adapters/test_audio_adapters.py -q` passed 82 tests; `python -m pytest tldw_Server_API/tests/Workflows/adapters/test_audio_adapters.py::TestMultiVoiceTTSAdapter tldw_Server_API/tests/Watchlists/test_audio_briefing_workflow.py -q` passed 44 tests; Bandit on touched implementation files reported zero findings.
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
