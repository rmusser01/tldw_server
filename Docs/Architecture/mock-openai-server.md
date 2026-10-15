# The mock OpenAI server

This document describes the standalone OpenAI-compatible test double at the repo root: what it simulates, its config/pattern system, who launches it (Evaluations integration tests, frontend onboarding UAT, governed CI), and its determinism guarantees and limits. **Naming trap**: the Chat/Character_Chat suites' `mock_openai_server` fixture is an unrelated inline `http.server` — see gotcha 1.

## Authoritative files

| Concern | File | Key symbols |
| --- | --- | --- |
| Server | `mock_openai_server/mock_openai/server.py` | FastAPI app; `/v1/chat/completions` (JSON + SSE), `/v1/embeddings`, `/v1/completions`, `/v1/models`, `/health`; any `Bearer sk-*` auth (unless auth disabled); optional random error simulation; scenario failures capped by `times` per process |
| Config | `mock_openai_server/mock_openai/config.py` | `MockConfig` (file/dict/env); `ResponsePattern` matching — model equality, `content_regex` against the **last** message (text-part aware), system-prompt substring, arbitrary field equality; priority-ordered selection |
| Responses | `mock_openai_server/mock_openai/responses.py` | `ResponseManager` — fixture files with `$timestamp/$request_id/$chat_id/$model` substitution, cached post-substitution; defaults: fixed chat string, **random** 1536-dim embeddings |
| Streaming | `mock_openai_server/mock_openai/streaming.py` | role chunk → word-group content chunks (5 words / 50 ms) → finish → `[DONE]`; function/tool-call deltas |
| Entry | `mock_openai_server/run_server.py`, `config.json` | defaults: port 8080, streaming enabled |
| Consumers | `tldw_Server_API/tests/Evaluations/conftest.py` (subprocess on :8080); `apps/tldw-frontend/scripts/onboarding-uat/run.mjs` + `e2e/onboarding-uat/mock-openai/configs/*.json` (per-port, committed fixtures); `.github/workflows/ci.yml` (the ci-journeys config for governed E2E) | |

## Usage (dataflow)

1. A launcher starts the server with a config (repo default or a UAT config).
2. The system under test points at it via base-URL env or patched config.
3. A request hits `/v1/chat/completions` → auth check → optional simulated error → optional scenario failure (pattern-matched, counted, capped) → pattern match (priority desc) selects a fixture → loaded, templated, cached → JSON, or re-chunked to SSE when streaming.

## Boundaries

- A separate installable package, deliberately outside `tldw_Server_API`; the MCP runtime packaging test explicitly forbids pulling it in.
- It simulates the OpenAI HTTP surface only — no token-accurate accounting, no tool execution, no audio.

## Failure behaviors (verified)

| Case | Behavior |
| --- | --- |
| Missing/invalid Authorization | 401 `authentication_error` (skipped when auth disabled) |
| Error-simulation draw | 500 `simulated_error` (random) |
| Scenario failure match | Configured status (e.g. 503), at most `times` occurrences per process |
| No pattern match / missing fixture | Built-in default response — never an error |

## Config keys (exact)

File: `server.{host,port,cors_origins,log_requests,simulate_errors,error_rate,require_auth}`; `streaming.{enabled,chunk_delay_ms,words_per_chunk}` (legacy `tokens_per_chunk` accepted); `responses.<endpoint>.patterns[].{match{model|content_regex|system_prompt|<any-field>},response_file,priority}` + `.default`; `models[]`; `response_base_dir`; `scenario_failures.<endpoint>[].{match,status_code,message,type,code,times}`. Env: `MOCK_OPENAI_{HOST,PORT,CORS_ORIGINS,LOG_REQUESTS,STREAMING_ENABLED,CHUNK_DELAY}`; test-side `MOCK_OPENAI_HEALTH_TIMEOUT_SECONDS`.

## Verified gotchas

1. **Two unrelated "mock openai" things**: this package (Evaluations/UAT/CI) vs. the inline `BaseHTTPRequestHandler` in `tldw_Server_API/tests/_plugins/chat_fixtures.py` used by the Chat/Character_Chat suites — the similarly-named `test_complete_v2*_with_mock_openai.py` tests use the inline one.
2. The Evaluations fixture hardcodes `localhost:8080` — parallel runs collide; the frontend runner parameterizes ports instead.
3. Scenario-failure counters are process-global and never reset in-process; tests must clear them.
4. Response caching freezes template substitutions at first load per process.
5. Auth accepts *any* `sk-*` key — auth-failure paths are only testable via malformed headers.
6. The package README is upstream-generic and omits scenario failures and the UAT wiring — partially stale for in-repo use.

## Governing decisions and docs

No ADR and no `Code_Documentation` entry — this doc is the reference alongside the package README (accurate on endpoints/config, stale on in-repo usage). Its own test suite includes strict-match regression guards pairing the mock with the frontend's ci-journeys fixtures.

## Related docs

- [evaluations.md](./evaluations.md) — the integration tests that launch it
- [chat-character.md](./chat-character.md) — the suites using the *other* inline mock
