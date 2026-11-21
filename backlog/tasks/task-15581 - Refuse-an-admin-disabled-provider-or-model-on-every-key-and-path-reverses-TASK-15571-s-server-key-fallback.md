---
id: TASK-15581
title: Refuse an admin-disabled provider or model on every key and path (reverses
  TASK-15571's server-key fallback)
status: Done
labels:
- llm
- byok
- policy
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The owner's rule, 2026-11-19, verbatim: "if a user tries to use a model/service thats been explicitly disabled by the admin, they shouldn't be able to use it; if its just been disabled as in 'no credits/key enabled' then tehy should be able to use it via BYOK". An explicit disable is the provider override's is_enabled=False (provider_disabled), or a model not on its allowed_models when that list is set (model_not_allowed); the owner confirmed the second on 2026-11-19. Either one is refused on every key (the user's own, a team or org shared key, and the server's) and on every path. A provider the server simply has no key or credits for is not a disable: a user's own key must still work. This reverses TASK-15571's rule (a denied model falls back to the server key), which PR #3523 shipped (b1a32fbba4).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The analyze path (_summarize_via_adapter) refuses with 'configuration' on every key and with no handle, whatever the model's source (configured, DEFAULT_MODEL_<PROVIDER>, model_override); the evaluation judges and Prompt Studio's direct adapter calls refuse a disallowed DEFAULT_MODEL_<PROVIDER>; the MCP model_completion authoritative path refuses on every key after its scope check, with tests
- [x] #2 An override store that cannot be read refuses (credential_store_unavailable) on each path above, with tests
- [x] #3 A provider the server has no key or credits for still serves a user's own key: with no override, or one that only leaves it on, an own-key call runs and is not checked against the LLM limit, with a test
- [x] #4 PR #3523's fallback is removed (server_key_fallback, log_server_key_fallback, override_outcome, the quota gate's strict parameters, the media ingest worker's answered marking, the judges' and MCP's server-key branches) and llm_quota_gate.py, media_ingest_jobs_worker.py, eval_runner.py and ms_g_eval.py are as before #3523
- [x] #5 ADR-086 is amended with the reversal and the owner's rule; Usage_Quotas.md, the BYOK guide (and its Docs/Published mirror) and Docs/Architecture/llm-providers.md say a disabled provider or model is refused on every key
- [x] #6 Every provider adapter call (chat, achat, stream, astream) refuses an admin-disabled provider or model, and each OpenRouter fallback model, on every key (own, team, org, server, no handle) before the quota check, with ChatConfigurationError carrying the reason as error_code, with tests
- [x] #7 Every standalone worker process and CLI that can make an LLM call loads and refreshes the provider-override store at start, so a provider with no override is unaffected outside the app process, with tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Measured on the branch (merged with origin/dev 56e7ea11dc): RED/GREEN per file matched the plan (llm_provider_overrides 21F->1F, AF_UNIX path-length env artifact; byok_calls 54F->132P; boundary 22F->111P; MCP integration 9F->91P; llamacpp 2F->7P; Slides provider same 84 baseline ids); wider set 419 files, only env failures (PostgreSQL cluster down, xdist ordering, probe-blocked baseline); tests/Docs 214 passed; tests/lint 331 passed; tests/CI 2 known failures; triage/shard/ids/scope gates exit 0; Bandit 2 pre-existing Low B105; ruff 25 = 25 vs origin/dev.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
An admin-disabled provider or model is refused on every key and path: the provider base class checks every adapter call before the quota check; the analyze path, the MCP completion tool, Slides standalone-HTML generation and the llama.cpp inference route keep their own check; each OpenRouter fallback model is checked too; RAG, evaluations, MCP and an endpoint that maps nothing keep the reason, 403 (503 for an unreadable store), and a store outage is retryable on analyze; PR #3523's fallback is removed. Every standalone worker and evaluation CLI that can make an LLM call loads the override store at start, and an AuthNZ pool reset keeps it loaded. A provider with no server key still serves a user's own key. ADR-086 and ADR-067 amended; docs updated; the WebUI listing gap is TASK-15583, an explicit disallowed state TASK-15582.
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
