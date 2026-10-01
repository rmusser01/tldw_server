---
id: TASK-13404
title: >-
  RG safety-net tests: broaden the WebUI replay fixture; the invalid-credential
  test hits a stub route
status: To Do
assignee: []
created_date: '2026-09-30 09:46'
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
- [ ] #1 The replay fixture includes chat, media, notes and RAG traffic from a populated session
- [ ] #2 The replay test also runs on the Redis backend (InMemoryAsyncRedis)
- [ ] #3 The invalid-credential test asserts the real route's 401
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
