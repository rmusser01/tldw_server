---
id: TASK-15644
title: Support OGM video transcription
status: Done
labels:
- media
- transcription
references:
- Docs/Code_Documentation/Ingestion_Pipeline_Video.md
- Docs/ADR/022-embeddings-api-and-media-pipeline.md
- https://github.com/rmusser01/tldw_server/pull/3624
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Recognize Ogg Media (.ogm) inputs in video ingestion and route them through the existing transcription pipeline.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Video uploads accept valid OGM files and retain MIME validation.
- [x] #2 Shared ingestion clients recognize .ogm as video without browser MIME metadata.
- [x] #3 Regression tests and an FFmpeg check cover OGM upload and audio extraction.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR required: no. This extends the input format allowlist for the existing media ingestion and transcription path. ADR-022 continues to govern media job ownership.
Verify upload validation and UI routing with focused regressions, then validate a real OGM sample with the existing FFmpeg WAV conversion. Preserve video size and content-type controls.
A real MPEG-4/Vorbis OGM sample passed staging, FFmpeg extraction (48,000 frames of 16 kHz mono PCM for a three-second window), and video transcript handling with stubbed speech recognition. Nine unrelated HTML validation failures were independently reproduced using the unmodified Git HEAD modules.
Verification passed for 85 backend tests (one expected failure; 11 HTML cases omitted from the rerun) and 116 frontend tests. ESLint passed on touched UI files, Ruff showed no new diagnostics, and Bandit found no issues in touched production code.
Renumbered for the dev PR because open PR #3615 already holds TASK-15642. The dev port preserves the shared standard video upload size cap.
Verified the dev port with 16 focused backend tests, 170 shared UI tests, and a real OGM-to-PCM smoke test with stubbed speech recognition. Bandit and Python syntax checks passed on the touched production scope.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Enabled .ogm uploads and video routing in the existing media API and shared clients. Ogg MIME aliases apply to OGM uploads, and FFmpeg extracts PCM for the existing transcription service. Documented support and added regressions.
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
