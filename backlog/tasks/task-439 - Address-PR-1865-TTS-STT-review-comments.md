---
id: TASK-439
title: Address PR 1865 TTS STT review comments
status: Done
labels:
- audio
- tts
- stt
- review-fix
- pr-1865
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Address still-actionable automated review comments on PR #1865 for the TTS/STT workflow branch. Verify each finding against code, fix valid issues with focused changes, document non-actionable items, run frontend/backend/Bandit/diff verification, push the branch, and resolve or reply to review threads where appropriate.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Valid PR #1865 review findings from Gemini, Qodo, CodeRabbit, and cubic are fixed or explicitly documented as non-actionable with technical rationale.
- [x] #2 Backend fixes cover audio preset async blocking, preset limits, secret-key normalization, DB uniqueness/default constraints, sanitized STT capability metadata, response model/rate scope, and schema verification gaps.
- [x] #3 Frontend fixes cover preset apply async behavior, name/duplicate handling, stale connection mirror clearing, preset config normalization, stale error/abort cleanup, and readiness/provenance correctness.
- [x] #4 Focused backend, frontend, Bandit, and git diff verification is recorded before pushing.
- [x] #5 PR branch is pushed and actionable review threads are resolved or replied to.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect touched files and tests for each PR review finding. 2. Implement backend fixes for STT capabilities and audio presets. 3. Implement frontend fixes for preset controls, config normalization, readiness/provenance, stale errors, and connection mirror clearing. 4. Add or extend focused tests. 5. Run focused frontend/backend/Bandit/diff verification. 6. Push fixes and resolve/reply to PR threads.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented PR #1865 review fixes across backend audio presets/STT capabilities and shared WebUI/extension TTS/STT surfaces. Verification: backend pytest tldw_Server_API/tests/Audio/test_audio_presets_endpoint.py tldw_Server_API/tests/Audio/test_stt_capabilities_endpoint.py passed 9 tests; package UI Vitest targeted suite passed 89 tests; extension parity Vitest passed 2 tests; Bandit on touched backend audio/storage files reported 0 findings; git diff --check passed; touched-file TypeScript filter produced no errors. Old PR Full Suite failures were inspected; the named Audio/Audit failures reproduce as passing locally and are not in this slice's touched code.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed the automated review surface for PR #1865 with focused backend and frontend fixes. Backend changes add guarded/sanitized STT capability metadata, response models, rate/scope enforcement, caching for synchronous health probes, audio preset limits, stricter secret-key rejection, update validation, and DB-level uniqueness/default constraints. Frontend changes improve async preset apply handling, duplicate naming, edit-state preservation, cleared server URL mirrors, config normalization, STT provenance reuse, stale error cleanup, render abort races, and WebUI/extension route parity tests.

Verification recorded: focused backend pytest passed 9 tests, package UI Vitest passed 89 tests, extension parity Vitest passed 2 tests, Bandit reported 0 findings on touched backend files, `git diff --check` passed, and the touched-file TypeScript filter returned no errors. The old PR Full Suite Audio/Audit failures were inspected and the named tests passed locally; they were not caused by this review-fix slice.
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
