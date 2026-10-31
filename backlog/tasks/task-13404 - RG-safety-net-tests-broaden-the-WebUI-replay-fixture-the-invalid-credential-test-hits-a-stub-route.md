---
id: TASK-13404
title: >-
  RG safety-net tests: broaden the WebUI replay fixture; the invalid-credential
  test hits a stub route
status: Done
assignee: []
created_date: '2026-09-30 09:46'
updated_date: '2026-10-02 03:36'
labels:
  - tests
  - rate-limit
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
From the final review of plan 2026-09-29-rg-ingress-safety-net:
- The WebUI replay fixture (tests/Resource_Governance/fixtures/webui_session_requests.json) is a 132-request smoke tour on an empty DB. It covers 9 paths and 4 policies, with no chat, media, notes or RAG pages, and runs on the memory backend only.
- test_middleware_identity.py::test_invalid_credentials_reach_the_route asserts 200 on a stub route with no auth, not the real route's 401.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The replay fixture includes chat, media, notes and RAG traffic from a populated session
- [x] #2 The replay test also runs on the Redis backend (InMemoryAsyncRedis)
- [x] #3 The invalid-credential test asserts the real route's 401
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Fixture broadened from 132->528 requests (methods: GET 250, OPTIONS 257, POST 21), recorded from a populated single-user session (AUTH_MODE=single_user, RG_ENABLED=false) on port 18765 vs a Next.js WebUI on port 18080. Populate step (curl, X-API-KEY auth): 5 notes via POST /api/v1/notes/; 3 media docs via POST /api/v1/media/add (media_type=document, perform_analysis=false, small .md files); 3 chat conversations (6 turns total) via POST /api/v1/chat/completions with api_provider=custom-openai-api pointed at mock_openai_server/run_server.py --port 18081 (CUSTOM_OPENAI_API_IP=http://127.0.0.1:18081/v1, CUSTOM_OPENAI_API_KEY=sk-mock-key-12345, model=gpt-4.1-2025-04-14, save_to_db=true); 5 RAG queries via POST /api/v1/rag/search (confirmed real hits against the media docs); plus GET list calls for notes/media/chat/conversations. WebUI tour: cd apps/tldw-frontend && TLDW_LIVE_TIER_UAT=1 TLDW_SERVER_URL=http://127.0.0.1:18765 TLDW_E2E_API_KEY=... TLDW_WEB_URL=http://localhost:18080 TLDW_WEB_CMD='bun run dev -- -p 18080' node_modules/.bin/playwright test e2e/smoke/all-pages.spec.ts --reporter=line (192 passed, 2 unrelated UI failures - acceptable, only the access log mattered). Backend stdout (Loguru-wrapped uvicorn.access) from both the pre-restart and post-restart (custom-openai env added) server runs was concatenated, then converted with Helper_Scripts/rg_webui_fixture_from_access_log.py. Final per-path-family counts (3rd path segment): persona 411, config 16, media 11, chat 10, moderation 10, notes 7, prompts 7, llm 6, slides 6, chunking 6, rag 5, health 4, explainer 4, quizzes 4, kanban 4, ingestion-sources 4, users 3, chat-workflows 2, items 2, evaluations 2, prompt-studio 2, data-tables 1, audio 1. All four required families (chat, media, notes, rag) present. AC2: parametrized test_webui_replay_no_429.py over backend=[memory,redis] (RedisResourceGovernor + InMemoryAsyncRedis stub, unique rg_t_webui_replay_<n> namespace per run, same Clock/FakeTime source as memory) per the _gov() injection pattern in test_governor_safety_net.py. Replay result: 0 governor 429s on both memory and redis backends with the shipped resource_governor_policies.yaml (two full passes of the session plus a 1 req/s extension stream layered on top) - no denying policy/peak rate to report since nothing was denied; limits were not touched. AC3: test_invalid_credentials_reach_the_route rewritten to wire the real Depends(get_request_user) auth dependency into its own dedicated route/app (AUTH_MODE=single_user, SINGLE_USER_API_KEY=the-configured-real-key, generous RG policy so RG itself never interferes, RG's own identity-charging path still faked via auth_principal_resolver.get_auth_principal as in the sibling tests in this file). X-API-KEY: fake now asserts a real 401 from authenticate_api_key_user's single-user branch (no DB touched); a matching key asserts 200 as a sanity check against a blanket-401 stub. Verification: ruff check clean on both changed test files; tldw_Server_API/tests/Resource_Governance full suite: 363 passed, 5 skipped, 2 xfailed (TLDW_TEST_NO_DOCKER=1, -n 4). Both the tldw_server backend and mock_openai_server were started and stopped by this task (PIDs tracked and killed directly, never --no-docker-affecting or ambient processes).
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
