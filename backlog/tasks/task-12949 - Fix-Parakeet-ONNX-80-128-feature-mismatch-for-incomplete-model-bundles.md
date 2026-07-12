---
id: TASK-12949
title: Fix Parakeet ONNX 80/128 feature mismatch for incomplete model bundles
status: Done
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
- TDD evidence: cache metadata, graph, external-data, provider, explicit-local-path, and incomplete-repair regressions failed against the prior loader and passed after the fixes.
- Review follow-up: rebased PR #2715 onto origin/dev at 8601d41f80; addressed and resolved both inline review threads.
- Final verification: 40 passed, 3 dependency-gated skips; Ruff passed; Bandit JSON reported 0 findings; git diff --check passed.
- Independent final review found no remaining Critical or Important issues.
- PR head: 20e8b04e3c. Refreshed GitHub Actions runs were still queued with no logs or failures as of 2026-07-12 12:23 PDT.
- Merge gate: repository policy still requires the human requester to write the PR Change summary explaining what changed and why.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Rebased PR #2715 onto the latest dev and hardened Parakeet ONNX bundle loading. Metadata inspection now stays CPU-only while inference preserves requested providers. Remote TDT caches repair missing metadata, graphs, and external-data failures once, then fail closed if still incomplete; explicit local paths are never downloaded. Added red-green regression coverage for all reviewed edge cases, resolved both inline threads, and completed local test, lint, security, and independent review gates.
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
