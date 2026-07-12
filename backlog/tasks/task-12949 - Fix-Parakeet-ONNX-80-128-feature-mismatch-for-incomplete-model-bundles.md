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
- Docs/superpowers/specs/2026-07-12-parakeet-onnx-bundle-metadata-design.md
- backlog/tasks/task-12949 - Fix-Parakeet-ONNX-80-128-feature-mismatch-for-incomplete-model-bundles.md
- tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Parakeet_ONNX.py
- tldw_Server_API/tests/Media_Ingestion_Modification/test_parakeet_onnx.py
references:
- 'User report: Parakeet ONNX encoder rejected 80 features; expected 128.'
- https://github.com/rmusser01/tldw_server/pull/2715
documentation:
- Docs/superpowers/specs/2026-07-12-parakeet-onnx-bundle-metadata-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Require the Parakeet ONNX model metadata needed by onnx-asr, refresh incomplete remote caches, reject invalid local bundles clearly, and add regression coverage for the reported 80-vs-128 feature mismatch.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Remote Parakeet ONNX caches missing config.json are refreshed before model load.
- [x] #2 Local Parakeet TDT bundles missing config.json fail closed with an actionable error.
- [x] #3 Bundle config feature size is validated against the encoder graph input.
- [x] #4 Focused Parakeet ONNX tests pass.
- [x] #5 Bandit reports no new findings in the touched Python scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- TDD evidence: new cache/local/config/encoder regression cases failed against the original loader and passed after the fix.
- Verification: focused suite 34 passed, 3 skipped; Ruff passed on production and tests; Bandit JSON reported 0 results and 0 errors; git diff --check passed; real cached INT8 encoder reported a 128-feature match.
- Independent code review found no critical production issues. The cache-repair test was strengthened to prove the refreshed bundle completes loading, and dynamic feature axes are covered.
- PR #2715 is a draft pending the repository-required human-written Change summary.
Review follow-up reopened: rebase PR #2715 onto latest dev, inspect all review threads and GitHub Actions checks, address verified issues, rerun validation, and update the PR.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the Parakeet ONNX 80/128 feature mismatch at model-load time. The loader now treats config.json as required metadata, refreshes stale remote caches, rejects incomplete or invalid local bundles, validates positive integer features_size values, and checks the selected encoder's static audio_signal feature axis before onnx-asr loads or the runtime is cached. Added regression coverage for cache repair, malformed metadata, quantized/unquantized mismatches, and dynamic axes.
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->
