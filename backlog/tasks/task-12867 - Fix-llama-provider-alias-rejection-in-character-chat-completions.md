---
id: TASK-12867
title: Fix llama provider alias rejection in character chat completions
status: Done
labels:
- bug
- chat
- llm-providers
- webui
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Character chat completion can receive the WebUI catalog provider id `llama`, but the backend shared provider resolver treated it as a separate credentialed provider instead of canonicalizing it to `llama.cpp`. This caused `/api/v1/chats/{id}/complete-v2` to return `missing_provider_credentials` even though the llama.cpp server endpoint was configured and normal chat could stream successfully.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 `api_provider: llama` normalizes to `llama.cpp` for `/api/v1/chat/completions`.
- [x] #2 Character chat `/complete-v2` normalizes raw `provider: llama` before credentials/provider dispatch.
- [x] #3 Regression tests cover schema validation and shared provider/model resolution aliases.
- [x] #4 Live localhost verification against configured llama.cpp returns 200 streaming response.
- [x] #5 Bandit reports zero findings for touched backend source files.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Root cause: standard chat normalized WebUI catalog provider id `llama` via ChatCompletionRequest, but character chat `/complete-v2` passed raw `provider: llama` into the shared resolver. The resolver treated `llama` as a separate credentialed provider instead of canonical `llama.cpp`, producing `missing_provider_credentials` despite the configured llama.cpp endpoint. Implemented normalization in both ChatCompletionRequest validation and shared chat_service provider/model resolution. Live verification: created a fresh Miku chat and POSTed `/api/v1/chats/3ed7b614-922c-4161-807c-d8a7c048d15b/complete-v2?scope_type=global` with `provider: "llama"`; response was HTTP 200 text/event-stream from `gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf`. Focused pytest: `tldw_Server_API/tests/Chat_NEW/unit/test_chat_schemas.py tldw_Server_API/tests/Chat_NEW/unit/test_provider_model_resolution.py -q` passed 40 tests. Bandit: `/tmp/bandit_task_12120_final.json` reported 0 findings for `chat_request_schemas.py` and `chat_service.py`. Screenshots regenerated in `/tmp/tldw-github-showcase/`. Known non-blocking warning: WebUI still logs per-chat settings 404 warnings, which are unrelated to this provider alias regression.
PR #2586 review follow-up in progress: address Qodo type-hint/style/test-stability comments, Qodo unreachable warning logging comment, and Gemini defensive strip/lower normalization comments; then rebase onto latest dev and repush.
PR #2586 review follow-up addressed: added explicit type hints and `-> None` to new tests; shortened/wrapped the schema regression test signature; replaced the Pydantic post-construction mutation with a `SimpleNamespace` request stub matching the character-chat resolver input shape; preserved defensive `.strip().lower()` before provider alias normalization; removed the unreachable `pass` before the alias-resolution warning log. Verification after review fixes: focused pytest passed 40 tests with 5 warnings; Bandit `/tmp/bandit_task_12118_review.json` reported 0 findings; `git diff --check` passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed the llama.cpp provider alias regression and addressed PR review follow-ups. The WebUI catalog id `llama` now canonicalizes to `llama.cpp` in standard chat validation and shared character-chat provider/model resolution, while preserving prior trim/lower defensive behavior. Review follow-up tightened the regression tests, removed brittle Pydantic mutation, and restored the intended warning log in alias-resolution exception handling.
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
