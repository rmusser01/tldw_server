---
id: TASK-13416
title: >-
  Chatbook: tokenless clients now get 401 from the server audio STT-health and
  streaming-status probes (#3058)
status: To Do
assignee: []
created_date: '2026-10-01 23:47'
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
- [ ] #1 tldw_chatbook either requires configured server credentials before making these server-mode audio probes, or treats 401 as an explicit 'authentication required' state instead of a generic failure
- [ ] #2 Chatbook tests cover the tokenless/401 path for both probes
<!-- AC:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
