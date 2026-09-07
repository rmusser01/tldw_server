---
id: TASK-13214
title: Honor configured Persona Live speech provider without Kokoro lock-in
status: Done
assignee: []
created_date: '2026-09-07 16:06'
updated_date: '2026-09-07 17:45'
labels:
  - buddy
  - voice
  - bug
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove the UAT-specific Kokoro restriction so Persona Live prepares and speaks through the configured TTS provider. Audit related UAT assumptions and retain explicit errors, provider choice, and Stop/cleanup behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Configured registered TTS providers prepare and synthesize without silently substituting Kokoro or its model/voice defaults.
- [x] #2 Unsupported or unavailable configurations fail clearly before recording where detectable; preparation does not synthesize user content.
- [x] #3 Regression coverage verifies a non-Kokoro provider through preparation, audio delivery and stop/cleanup, including provider failure and no fallback.
- [x] #4 Audit related voice assumptions, correct relevant guides and record remaining limitations and actual UAT evidence.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no new ADR; existing backlog/decisions/004-persona-live-tts-provider-selection.md applies. 1. Reproduce Qodo backend and frontend defects with focused regression tests. 2. Preserve configured Kitten defaults and truthful readiness; guard each native utterance identity; show catalog failure and retry. 3. Correct function contracts and validate existing category markers. 4. Run focused tests, lint/Bandit, review fixes, respond to all findings, and verify PR checks.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented provider/model/voice selection, shared authenticated TTS credential scope, browser playback recovery and temporary adapter cleanup. Kitten preparation warms the requested model through its cache; Chat preparation no longer requires a static server key. Guides corrected in both repositories. Final verification/evidence recording in progress.
Verification complete: 265 focused Python and 111 frontend tests passed; Bandit found zero issues in all seven changed production Python files. Targeted new Python Ruff/Black checks pass. Frontend ESLint has zero errors and 58 confirmed baseline warnings; large legacy Python formatting/lint debt remains unchanged. Real restarted server prepared Parakeet + browser TTS, dispatched the authorized DeepSeek Persona reply, acknowledged voice Stop, and stopped the session with HTTP 200. Real native SpeechSynthesis through the production controller emitted start/end and stopped a long reply with no queued speech. The browser input/WS boundary was controlled; no microphone/human-audibility or paid remote-TTS claim. Independent review resolved credential-scope, selected Kitten model readiness, and browser explicit-voice substitution findings. Canonical/Published guide mirrors match and MkDocs build passed. ADR: backlog/decisions/004-persona-live-tts-provider-selection.md. Evidence and limitations: Docs/Reviews/PERSONA_TTS_PROVIDER_CHOICE_2026_09_07.md. Added incident-based lesson; temporary browser harness removed from product public files.
Reopened for four user-approved review corrections: consistent blank voice/default precedence, Kitten voice validation before readiness, speech generator cleanup inside credential scope, and documented reconnect with visible active model. Add failure-first regressions and scoped re-review before committing.
All four second-review findings addressed. Blank voices remain unset in the frontend and resolve the configured global voice only for its provider; explicit voices remain authoritative. Kitten validates the selected voice with the loaded runtime. Speech generators close within the credential scope, including partial-audio errors. Live displays model selection and reconnect guidance; both guides and Published mirrors require Disconnect then Connect after saving connected-session voice defaults. Failure-first reproduction: five backend and seven frontend failures. Final checks: 278 Python tests, 116 frontend tests, Ruff/Black clean on changed Python, Bandit zero findings, ESLint zero errors/two existing warnings, MkDocs build and links/mirrors pass. Two scoped independent re-reviews found no residual issues. Live UI labels checked, isolated backend restarted from matching source manifest and /health returned 200. No microphone or physical audio UAT repeated. Evidence appended in Docs/Reviews/PERSONA_TTS_PROVIDER_CHOICE_2026_09_07.md; ADR-004 clarified.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

PR #2928 Qodo review follow-up: validate all seven findings; fix Kitten shared model/revision state and readiness reporting, stale browser utterance callbacks, catalog error recovery, and missing annotations/docstrings. Module-level unit marker already exists; verify collection and respond with evidence. User authorizes all review fixes and merge.

Backend CI also identified the stale OpenAPI fingerprint for the new optional tts_model schema field. Regenerate canonical schema/types and verify drift gate; no additional API contract change.

PR #2928: resolved six confirmed Qodo findings (shared Kitten model/revision defaults, truthful public runtime health, stale native utterance callbacks, visible catalog failure/Retry, docstrings, annotations). Seventh marker claim rebutted: existing module pytestmark selects all 14 cases under -m unit. Six backend and five frontend red failures reproduced; final combined 284 Python/124 frontend tests passed, additional overlapping adapter/health suites 66 passed, Bandit zero, no new lint diagnostics, independent review clear. API fingerprint regenerated; sole schema delta is optional tts_model; drift check and frontend codegen pass. ADR-004 clarified lazy readiness versus routability and utterance ownership. Physical voice UAT not repeated. Details in Docs/Reviews/PERSONA_TTS_PROVIDER_CHOICE_2026_09_07.md.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Removed Kokoro lock-in and addressed all four follow-up findings: correct provider-owned default voice, pre-recording Kitten voice validation, deterministic speech cleanup before credential disposal, and actionable reconnect guidance with Live model display. Verified 278 Python and 116 frontend tests, zero Bandit findings, clean scoped re-review, built/synchronized guides and healthy restarted UAT server. Physical all-provider voice qualification remains outside this correction.

PR #2928 Qodo findings and docs/OpenAPI CI integration issues addressed with failure-first tests and independent review; final combined 284 Python and 124 frontend tests pass.
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
