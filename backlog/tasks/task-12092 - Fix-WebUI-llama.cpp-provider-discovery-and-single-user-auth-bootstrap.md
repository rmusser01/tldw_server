---
id: TASK-12092
title: Fix WebUI llama.cpp provider discovery and single-user auth bootstrap
status: Done
assignee: []
created_date: ''
updated_date: '2026-07-02 05:01'
labels:
  - bug
  - webui
  - llm-providers
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Direct chat to llama.cpp through CUSTOM_OPENAI_API_URL works, but the WebUI model catalog ignores env-resolved custom OpenAI settings and filters every model out. The frontend shared TldwApiClient also misses the Next public API key unless runtime auth is exposed. Fix discovery/auth so latest dev can use local llama.cpp from the WebUI and then capture the requested screenshots.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 /api/v1/llm/providers and /api/v1/llm/models/metadata expose env-configured custom OpenAI model/endpoint as usable when direct chat works
- [x] #2 WebUI requests include the configured single-user API key without manual localStorage seeding in advanced mode
- [x] #3 Existing provider filtering continues to hide genuinely unavailable models
- [x] #4 Regression tests cover provider env resolution and auth bootstrap behavior
- [x] #5 Requested chat and character-card screenshots are captured against the fixed instance
- [x] #6 Character-card image_base64 is converted to a chat avatar data URL when hydrating selected character assistants
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
IMPLEMENTATION_PLAN_llamacpp_webui_catalog_auth.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Root causes investigated:
- Backend provider catalog readiness was not using the same env-resolved custom OpenAI endpoint/model path as chat completions, so latest dev could chat directly while WebUI discovery still saw stale/default custom OpenAI settings.
- WebUI advanced-mode single-user auth bootstrapped NEXT_PUBLIC_X_API_KEY into runtime request auth, but shared connection/model/stream paths did not consistently treat that runtime key as configured auth.
- A post-send /messages load-failed label reproduced as a frontend stale/race state; the exact endpoint returned HTTP 200 and a reload displayed the persisted Miku chat cleanly.
- Character-card import/storage is not the avatar failing layer: GET /api/v1/characters/3 returns image_base64 for Miku, but chat/server-session hydration used characterToAssistantSelection(), which normalized only avatar_url and ignored image_base64. Reloaded character chats therefore lost the imported PNG avatar and rendered the fallback placeholder.

Implemented fixes:
- Env-first custom OpenAI endpoint/model/API-key resolution in llm_providers catalog metadata.
- Runtime single-user key propagation through runtime bootstrap, connection health, model cache, and direct stream fallback.
- Character assistant hydration now converts embedded image_base64 into a data:image URL when avatar_url is absent, matching the character-card manager path.
- Regression tests for provider env resolution, shared runtime auth behavior, and embedded character avatar hydration.

Screenshots captured:
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.playwright-mcp/normal-chat-llamacpp.png
- /Users/macbook-dev/Documents/GitHub/tldw_server2/.playwright-mcp/character-chat-miku-roleplay.png

Verification:
- pytest provider/model suites: 20 passed.
- vitest WebUI auth/model/stream/avatar suites: 131 passed.
- Bandit on llm_providers.py: no findings.
- git diff --check: clean.
- Live probes after draining dev-server output: API 200, llama.cpp 200, WebUI 200.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed latest-dev WebUI llama.cpp setup blockers by aligning provider discovery with env-resolved custom OpenAI config and making NEXT_PUBLIC_X_API_KEY runtime auth work across shared WebUI request paths. Imported/used the available Miku character card, verified normal and character chat through the running llama.cpp server, captured the requested screenshots, and fixed the character-chat avatar path so embedded card image_base64 is shown instead of the generic placeholder.

PR: https://github.com/rmusser01/tldw_server/pull/2573
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
