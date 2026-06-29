---
id: TASK-2370
title: Add commercial Fish Audio S2 TTS API backend
status: Done
assignee: []
created_date: ''
updated_date: 2026-06-28 01:07
labels:
- tts
- audio
- provider
- fish-audio
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add hosted Fish Audio S2 commercial API support to the existing fish_s2 TTS provider without replacing the self-hosted Fish Speech native_http backend. The backend should call https://api.fish.audio/v1/tts with the required model header, support hosted /model voice creation, preserve local voice metadata mappings, and keep tests/docs/config aligned.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 fish_s2 can be configured with backend: commercial_api and FISH_AUDIO_API_KEY/FISH_API_KEY
- [x] #2 Commercial TTS payload sends the model header and Fish-compatible JSON fields including prosody, sample rate, bitrates, latency, and multi-speaker reference IDs
- [x] #3 Commercial voice/model creation uses Fish /model and stores returned remote model IDs in local voice metadata
- [x] #4 Existing native_http Fish Speech behavior remains covered and unchanged except for shared adapter plumbing
- [x] #5 Focused unit and endpoint tests pass
- [x] #6 Bandit runs on touched backend/API TTS scope with no new findings
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented hosted Fish Audio commercial_api support behind the existing fish_s2 provider while preserving native_http.

Verification:
- Fish-focused pytest suite: 34 passed.
- Service Fish subset: 8 passed.
- Bandit touched production TTS scope: 0 findings, report at /tmp/bandit_fish_s2_commercial_api.json.

Known skips/blockers: no live Fish API call was run; automated coverage uses mocked HTTP transports.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fish Audio S2 now supports a hosted commercial API backend via `backend: commercial_api` under the existing `fish_s2` provider key. The implementation covers hosted TTS payload/header construction, streaming, private hosted voice model creation/deletion, local-to-remote voice metadata mapping, Fish OPUS validation, env-based API key configuration, and updated setup documentation. Verification: Fish-focused pytest suite passed (34 tests), service Fish subset passed (8 tests), and Bandit on touched production TTS files reported zero findings.
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

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR #2540 rebase/review pass: rebased Fish S2 branch onto latest origin/dev and addressed review comments for native retry-after parsing, response JSON logging, aggregate audio route mounting, BYOK fallback logging, and Fish S2 endpoint response models. Verification so far: Fish S2 focused pytest selection passed (69 selected), aggregate/route tests passed (22), audio OAuth retry tests passed (18), presets/voice-conversion selection passed (13), Bandit touched production scope reported zero findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
