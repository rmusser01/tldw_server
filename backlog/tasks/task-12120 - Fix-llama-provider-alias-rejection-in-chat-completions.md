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
The WebUI provider catalog exposes the configured llama.cpp endpoint as provider name `llama`, but chat completion request validation rejects `api_provider: "llama"` before adapter alias resolution. A second frontend path sends provider-qualified model ids such as `llama:<model>` without `api_provider`; backend execution previously treated that as the default provider and returned a generation failure instead of routing to llama.cpp. Both paths break WebUI chat sends against a configured llama.cpp server.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `api_provider: "llama"` is accepted for chat completions and normalized to the llama.cpp provider used by backend routing.
- [x] #2 Frontend provider-qualified `model: "llama:<model>"` requests are normalized to provider `llama.cpp` and model `<model>`.
- [x] #3 Existing canonical `api_provider: "llama.cpp"` behavior remains unchanged.
- [x] #4 Regression tests cover the alias paths that the WebUI uses.
- [x] #5 Focused tests and security checks pass for the touched scope.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
- `origin/dev` already contains provider alias normalization via `normalize_catalog_provider_for_chat`.
- Added focused request-schema coverage for the WebUI `api_provider: "llama"` alias path so the regression is caught before the endpoint returns HTTP 422.
- Added backend provider/model normalization for provider-qualified model ids. Slash-qualified model ids keep the existing behavior; colon-qualified model ids are split only when the prefix normalizes to a registered chat provider.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
- Added schema regression coverage for the WebUI `api_provider: "llama"` request path.
- Added provider/model resolution coverage for the WebUI `model: "llama:<model>"` request path and normalized it to the llama.cpp adapter.
- Verified `api_provider: "llama"` returns HTTP 200 against the PR worktree backend and live llama.cpp server at `127.0.0.1:9099`, with response content `alias ok`.
- Verification:
  - `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat_NEW/unit/test_provider_model_resolution.py::test_resolve_provider_and_model_normalizes_llama_colon_model_prefix -q` failed before the fix with `metrics_provider == "openai"`.
  - `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Chat_NEW/unit/test_provider_model_resolution.py tldw_Server_API/tests/Chat/unit/test_chat_request_schemas.py -q` (`31 passed`)
  - `source .venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Chat/chat_service.py tldw_Server_API/app/api/v1/schemas/chat_request_schemas.py -f json -o /tmp/bandit_task_12120.json` (`0 results`)
  - Local curl to `POST /api/v1/chat/completions` with `api_provider: "llama"` returned `HTTP/1.1 200 OK`.
  - Local curl to `POST /api/v1/chat/completions` with `model: "llama:<gguf>"` and no `api_provider` returned `HTTP/1.1 200 OK` against the PR worktree backend and live llama.cpp server, with response content `colon alias ok`.
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
