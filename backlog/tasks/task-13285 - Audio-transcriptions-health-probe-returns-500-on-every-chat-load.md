---
id: TASK-13285
title: Audio transcriptions health probe returns 500 on every chat load
status: Done
assignee: []
created_date: '2026-09-19 14:47'
updated_date: '2026-09-23 23:13'
labels:
  - backend
  - chat
  - observability
  - ux-audit
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A health probe for audio transcriptions returns 500 on every cold load of the chat page. It is the only failing request on the page and is surfaced nowhere, so it is pure console noise that masks real errors while debugging. The dictation control now correctly reports itself unavailable, so the user-facing half of the original finding is fixed; the failing probe is not.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The probe returns a success or a well-formed unavailable response.
- [x] #2 When speech is not configured, the probe reports that state rather than failing.
- [x] #3 The probe the chat page fires on every load (GET /audio/transcriptions/health) returns 200 on an install missing STT/media deps, so it no longer adds a failed request
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
AC2 reworded 2026-09-23: 'a cold load produces no failed requests' covers every request on the page, which this task does not own and a live browser run was not repeated; the reworded AC is the part this probe controls, verified at the endpoint.

Fixed in 9b6560d07d. Root cause: get_stt_health imported Audio_Files outside any try; Audio_Files imports yt_dlp (and other ingestion deps) at module level, so on an install missing one the ImportError escaped as a 500. Reproduced in the shared .venv (no yt_dlp): all 4 existing health tests fail with ModuleNotFoundError: yt_dlp; with a stub yt_dlp they pass, so the import is the only failure. The probe now returns its normal payload shape with available/usable false and an 'STT not available' message. /transcriptions/capabilities had the same unguarded import and now reports availability unknown. Regression test test_transcriptions_health_reports_unavailable_when_stt_deps_missing: red on ea1cbc6941 (ModuleNotFoundError), green after. tests/STT + tests/Audio + Setup/test_audio_health_helpers.py: 15 failed/1551 passed before vs 14/1552 after; the only difference is the new test (remaining 14 pre-existing, unrelated). Bandit clean. Live browser cold-load not re-run.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The STT health probe now returns a 200 unavailable status instead of a 500 when STT modules can't import, so the chat page's on-load probe stops failing; the capabilities endpoint got the same guard.
<!-- SECTION:FINAL_SUMMARY:END -->
