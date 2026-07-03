---
id: TASK-12126
title: Harden audio transcription pipeline input handling
status: Done
labels:
- audio
- transcription
- stt
- bugfix
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address follow-up bugs in the audio transcription pipeline where compressed or non-canonical audio can reach WAV-only loaders after conversion failure, and where in-memory NeMo Parakeet/Canary paths can ignore the caller-provided sample rate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] `/audio/transcriptions` no longer falls back to the original upload when canonical WAV conversion fails.
- [x] Soundfile-backed local providers do not receive compressed originals through the endpoint conversion fallback.
- [x] Direct Canary and Parakeet MLX buffered paths either receive canonical WAV input or fail with a clear provider error.
- [x] In-memory Parakeet and Canary transcription normalizes non-16 kHz NumPy audio before direct model calls.
- [x] Focused regression tests, Ruff, and Bandit are run for the touched scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Using a scoped worktree on `codex/audio-transcription-pipeline-hardening`. Backlog CLI/MCP is unavailable in this environment, so this task was created as a manual repository-policy exception approved in-thread.

Implemented minimal hardening for the five verified issues:
- `/audio/transcriptions` rejects conversion import failures, conversion errors, empty conversion results, non-WAV conversion results, and missing conversion outputs with `invalid_audio`.
- Canary, Qwen3 ASR, and VibeVoice adapter paths canonicalize compressed direct-call inputs to WAV before soundfile/provider loading.
- Parakeet MLX buffered path canonicalizes compressed direct-call inputs to WAV before duration/buffered loading and rejects converted paths outside `base_dir`.
- NeMo Parakeet and Canary direct NumPy paths fold channels, handle scalar/empty arrays, resample non-16 kHz input to 16 kHz, and fall back without SciPy.

Verification:
- `python -m pytest tldw_Server_API/tests/Audio/test_audio_transcriptions_adapter_path.py tldw_Server_API/tests/Audio/test_stt_provider_adapter.py tldw_Server_API/tests/Media_Ingestion_Modification/test_nemo_transcription.py tldw_Server_API/tests/Media_Ingestion_Modification/test_parakeet_mlx.py -q --basetemp C:\Users\GDesktop-1\AppData\Local\Temp\tldw_pytest_tmp` -> `79 passed, 2 skipped`.
- `python -m ruff check ... --select E9,F821,F822,F823,B904,BLE001` -> passed.
- `python -m compileall -q` on touched production files -> passed.
- `git diff --check` -> passed; Git reported expected Windows LF-to-CRLF warnings only.
- `python -m bandit -r ... -f json -o C:\Users\GDesktop-1\AppData\Local\Temp\bandit_audio_transcription_pipeline.json` -> ran; nonzero due existing low-severity subprocess findings in `Audio_Transcription_Lib.py` outside this diff.
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
