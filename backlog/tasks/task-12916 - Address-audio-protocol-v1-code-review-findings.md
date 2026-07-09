---
id: TASK-12916
title: Address audio protocol v1 code review findings
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-08 03:04'
labels:
  - webui
  - audio
  - review
dependencies: []
documentation:
  - Docs/superpowers/plans/2026-07-08-chat-audio-streaming-protocol-v1.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix validated review findings from the strict chat audio streaming protocol v1 code review: dictation final frame/stop handling, STT redaction for final frames, post-config protocol violation closes, frontend push-to-talk path, and composer dictation insertion behavior where applicable.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Validated Critical/Important review findings are fixed or explicitly rejected with evidence.
- [x] #2 Focused backend/frontend regression tests cover the fixed behavior.
- [x] #3 Bandit/diff checks and verification results are recorded.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

Implemented review fixes:
- WebUI server dictation now accepts backend final frames, waits after stop for final/done, and appends only the missing suffix from cumulative full_transcript frames.
- STT redaction policy and audio stream redaction metrics now include final frames.
- Unified transcription and audio chat WebSockets close with 4400 on malformed JSON after strict config.
- useVoiceChatStream exposes push_to_talk mode and push_to_talk_release control frame while preserving voice_chat default.
- Playground and Sidepanel dictation transcript insertion now appends to current composer text instead of replacing it.

Verification:
- Backend focused audio suite: 61 passed.
- Frontend/WebUI voice suite: 52 passed.
- Bandit touched backend audio scope: 0 findings.
- Scoped git diff --check: passed.
- apps/tldw-frontend typecheck still fails on unrelated pre-existing files: AudioStudio TimelineEditor, ScheduledTasks, Skills Manager, scheduled-tasks service, mcp-hub, voice-cloning, knowledge QA fixtures, flashcards spec.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Addressed all validated critical/important review findings for audio protocol v1 follow-up. Dictation handles final frames and deferred stop completion, STT redaction/metrics include final frames, post-config malformed JSON closes 4400, push-to-talk framing is exposed in the frontend hook, and composer dictation appends instead of overwriting user text. Verification: backend audio focused suite 61 passed; frontend/WebUI voice focused suite 52 passed; Bandit touched backend audio scope 0 findings; staged diff check passed. apps/tldw-frontend typecheck remains blocked by unrelated pre-existing errors outside the touched files.
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
