---
id: TASK-12949
title: Fix Parakeet ONNX 80/128 feature mismatch for incomplete model bundles
status: In Progress
labels:
- bug
- audio
- stt
- parakeet
priority: high
modified_files:
- tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Parakeet_ONNX.py
- tldw_Server_API/tests/Media_Ingestion_Modification/test_parakeet_onnx.py
references:
- 'User report: Parakeet ONNX encoder rejected 80 features; expected 128.'
documentation:
- Docs/superpowers/specs/2026-07-12-parakeet-onnx-bundle-metadata-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Require the Parakeet ONNX model metadata needed by onnx-asr, refresh incomplete remote caches, reject invalid local bundles clearly, and add regression coverage for the reported 80-vs-128 feature mismatch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Remote Parakeet ONNX caches missing config.json are refreshed before model load.
- [ ] #2 Local Parakeet TDT bundles missing config.json fail closed with an actionable error.
- [ ] #3 Bundle config feature size is validated against the encoder graph input.
- [ ] #4 Focused Parakeet ONNX tests pass.
- [ ] #5 Bandit reports no new findings in the touched Python scope.
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
