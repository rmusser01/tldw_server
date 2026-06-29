# Provider Preflight

- Run id: `pre-main-uat-20260629054510`
- UAT task id: `TASK-12064`
- Provider rerun task id: `TASK-12065`
- Status: Passed

## Summary

| Provider | Endpoint | Model | Timestamp (UTC) | Result | Expected token present | Raw artifact |
| --- | --- | --- | --- | --- | --- | --- |
| OpenAI | `https://api.openai.com/v1/chat/completions` | `gpt-4o-mini` (`gpt-4o-mini-2024-07-18` returned) | `2026-06-29T06:31:50Z` | Pass - direct chat completion returned HTTP 200 | Yes | `/tmp/tldw-pre-main-uat/pre-main-uat-20260629054510/provider/openai_response.redacted.txt` |
| llama.cpp | `http://127.0.0.1:9099/v1/models` and `/v1/chat/completions` | `gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf` | `2026-06-29T06:36:15Z` | Pass - direct models and chat completion returned HTTP 200 | Yes | `/tmp/tldw-pre-main-uat/pre-main-uat-20260629054510/provider/llamacpp_chat_response.redacted.txt` |

## OpenAI Direct Preflight

- Credential check: `OPENAI_API_KEY` is present after `source /tmp/tldw-pre-main-uat/pre-main-uat-20260629054510/uat.env`.
- Secret handling: no credential value was printed or stored.
- Chat completion: `curl` exit `0`, HTTP `200`.
- Expected token: `ok-pre-main-uat-20260629054510`
- Pass/fail: pass.
- Redacted response excerpt: `model=gpt-4o-mini-2024-07-18`; `content=ok-pre-main-uat-20260629054510`.
- Note: an earlier run stopped at a missing `OPENAI_API_KEY`; that was resolved by loading the credential from the base checkout's config `.env` into the temporary UAT env without printing the value.

## llama.cpp Direct Preflight

- Selected `LLAMA_CPP_MODEL`: `gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf`.
- Persisted run state: `LLAMA_CPP_MODEL` is recorded in `/tmp/tldw-pre-main-uat/pre-main-uat-20260629054510/uat.env`.
- `/v1/models` query: `curl` exit `0`, HTTP `200`.
- Chat completion: `curl` exit `0`, HTTP `200`.
- Expected token: `ok-pre-main-uat-20260629054510`
- Redacted response excerpt: `model=gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf`; `content=ok-pre-main-uat-20260629054510`.
- Note: sandboxed localhost curl initially failed with connection refused even though `lsof` showed `llama-server` listening on `*:9099`; the bounded host-access rerun reached the listener successfully.

## Next-Step Status

- Status for Task 2: `DONE`.
- Blocker: none for direct provider preflight.
- Next UAT step: configure and start the isolated local single-user WebUI runtime, then verify backend-mediated answer paths against OpenAI and llama.cpp.
