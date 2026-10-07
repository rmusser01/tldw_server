---
id: TASK-13416
title: >-
  Chatbook: tokenless clients now get 401 from the server audio STT-health and
  streaming-status probes (#3058)
status: Done
assignee: []
created_date: '2026-10-01 23:47'
updated_date: '2026-10-03 11:31'
labels:
  - cross-repo
  - chatbook
  - audio
  - auth
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
#3058 (merged 2026-10-01) requires an authenticated user on GET /api/v1/audio/transcriptions/health and GET /api/v1/audio/stream/status. tldw_chatbook allows clients without an API token: its tldw_api client adds auth headers only when credentials exist, and Audio_Services_Interop/audio_services_scope_service.py calls get_stt_health and get_audio_streaming_status in server mode. Tokenless Chatbook installs now get 401 from these probes. The server routes stay protected (owner decision). This is a Chatbook-side follow-up flagged by Qodo's cross-repo review on #3058.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 tldw_chatbook either requires configured server credentials before making these server-mode audio probes, or treats 401 as an explicit 'authentication required' state instead of a generic failure
- [x] #2 Chatbook tests cover the tokenless/401 path for both probes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Fixed in tldw_chatbook PR #2954 (merged into chatbook dev 2026-10-03 11:29Z, merge a26d6a9c6c). Approach (b): ServerAudioServicesService maps a 401 from get_stt_health (including its warm-up capability lookup) and get_audio_streaming_status to a typed PolicyDeniedError with reason code auth_required, the same code Chatbook already uses in runtime_policy/server_context.py and Research_Workspace/server_adapter.py. Chosen over skipping probes without a token because it also covers server-rejected and late-set tokens, and older servers that still answer anonymously. The API client already sent no credentials without a token. Tests: tokenless client sends no X-API-KEY/Authorization to either probe; 401 -> auth_required for plain STT, warm-up lookup and streaming status; end-to-end through the scope service via an httpx mock transport. Targeted run: 76 passed. No Chatbook screen calls these probes yet. Known gap: test_audio_streaming (admin diagnostic) can still raise a raw AuthenticationError for tokenless clients; it is outside the two routes #3058 changed. Chatbook's Perf Guard check was red, as it is on chatbook dev itself, and is not required.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Tokenless Chatbook installs now get an explicit auth_required result instead of a raw error from the two server audio probes that #3058 put behind auth (tldw_chatbook #2954). The server routes stay protected.
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
