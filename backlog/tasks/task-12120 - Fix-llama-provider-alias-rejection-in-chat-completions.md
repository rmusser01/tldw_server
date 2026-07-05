---
id: TASK-12120
title: Fix llama provider alias rejection in chat completions
status: Done
labels:
- bug
- webui
- chat
priority: High
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The WebUI provider catalog exposes the configured llama.cpp endpoint as provider name `llama`, but chat completion request validation rejects `api_provider: "llama"` before adapter alias resolution. This breaks WebUI chat sends against a configured llama.cpp server with a 422 validation error.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `api_provider: "llama"` is accepted for chat completions and normalized to the llama.cpp provider used by backend routing.
- [x] #2 Existing canonical `api_provider: "llama.cpp"` behavior remains unchanged.
- [x] #3 A regression test covers the alias path that the WebUI uses.
- [x] #4 Focused tests and security checks pass for the touched scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- `origin/dev` already contains provider alias normalization via `normalize_catalog_provider_for_chat`.
- Added focused request-schema coverage for the WebUI `api_provider: "llama"` alias path so the regression is caught before the endpoint returns HTTP 422.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Added schema regression coverage for the WebUI `api_provider: "llama"` request path.
- Verified `api_provider: "llama"` returns HTTP 200 against the PR worktree backend and live llama.cpp server at `127.0.0.1:9099`, with response content `alias ok`.
- Verification:
  - `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Chat/unit/test_chat_request_schemas.py -q` (`25 passed`)
  - `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m bandit -r tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py -f json -o /tmp/bandit_task_12120.json` (`0 results`)
  - Local curl to `POST /api/v1/chat/completions` with `api_provider: "llama"` returned `HTTP/1.1 200 OK`.
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
