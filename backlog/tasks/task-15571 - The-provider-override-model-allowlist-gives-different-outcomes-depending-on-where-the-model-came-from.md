---
id: TASK-15571
title: The provider-override model allowlist gives different outcomes depending on
  where the model came from
status: Done
labels:
- llm
- byok
- policy
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the final review of PR #3500 (TASK-15496), pre-existing. The same admin allowlist (provider-override model policy) gives one denied model four outcomes, depending on where the model came from: (1) a denied configured own-key model: users_own_key_credentials returns None on the resolve-time denial, so the call runs on the server's key with no check (usage_quota_deps.py:119-121); (2) a denied DEFAULT_MODEL_<PROVIDER> fallback model for an own-key handle fails 'configuration' at dispatch (since PR #3500); (3) a request's model_override is never checked: media_analysis_credentials resolves the handle without a model (media_processing_deps.py:704) and _summarize_via_adapter uses model_override as given (Summarization_General_Lib.py:319-320); (4) for server-key callers (no handle) a denied model runs, because _summarize_via_adapter has no dispatch-time policy check on that path (Summarization_General_Lib.py:337-338). Separately, an authoritative-scope resolution skips the resolve-time override capture (core/Provider_Credentials/provider_credential_runtime.py:520-535, override_snapshot=None); its only caller is the MCP model_completion adapter (core/MCP_unified/adapters/model_completion/adapter.py:242), which has no model-policy check of its own. Chat calls validate_provider_override for every call (chat.py:4468); the analyze path does not. Also take the PR #3500 Minor: a denied fallback model writes no server log line, so an operator whose DEFAULT_MODEL_* is outside the allowlist sees every own-key analysis fail with 'Provider credential context is invalid.' and no reason.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Decide and record (owner) one rule for a model the override policy denies, whatever its source (configured, DEFAULT_MODEL_* fallback, model_override) and whatever the key (own key, server key, team/org): refuse, or fall back to the server key
- [x] #2 _summarize_via_adapter enforces that rule at dispatch for every call, handle or not, as chat does; tests pin a denied model from each source (configured, DEFAULT_MODEL_* fallback, model_override) on the own-key and server-key paths
- [x] #3 The MCP model_completion adapter's authoritative-scope path applies the same rule, with a test
- [x] #4 A denied model writes one server log line naming the provider and the reason (no secrets)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Owner decision (AC #1), 2026-11-19: a model the provider-override policy denies falls back to the server key, whatever its source (configured model, DEFAULT_MODEL_<PROVIDER> fallback, model_override, the MCP authoritative path) and whatever the key (the user's own, a team's, an org's); the call is not refused for the denial itself. This reverses PR #3500 (TASK-15496), which failed a denied DEFAULT_MODEL_* fallback with 'configuration'. Coordinator rulings, 2026-11-19, on the plan review: only model_not_allowed falls back; a disabled provider refuses ('configuration' on analyze, the policy error on MCP); an override store that cannot be read fails closed ('configuration' on analyze, credential_store_unavailable on MCP); on analyze the fallback needs a quota check that answers, and refuses when none does (TASK-15481's fail-closed rule), with 'configuration' and the message 'The LLM usage limit could not be checked.'; a user_to_check failure is no answer; the media ingest worker marks a job admitted only when its start check answered; MCP's reservation already fails closed. With no handle, or a server_default handle, nothing changes and no check applies; AC #4 is read per checked call. AC #2's 'as chat does' is read under this rule: chat refuses a denied model with a 403; the analyze path falls back instead. Plan: Docs/superpowers/plans/2026-11-19-denied-model-server-key-fallback.md.
Implemented per plan (PR #3523). RED to GREEN: test_provider_adapter_runtime_boundary 13 failed/91 passed -> 104; test_summarization_adapter 1 failed/127 -> 128; test_llm_quota_byok_calls 11 failed/73 -> 84; test_llm_quota_decision 1 failed/35 -> 36; test_model_completion_integration 5 failed/80 -> 85; test_model_completion_quota_compatibility 1 failed/7 -> 8; test_provider_credential_runtime 59 (pin). Unchanged: gate 12, job refusal 8, adapter 136, transport 187, factory 89, byok_runtime 198+1 skipped, sqlite 25, and the wider set at its baseline counts. tests/Docs 214, tests/lint 326 (ignoring test_runtime_package_discovery.py, no setuptools), tests/CI 6 failed, all environment-only (no python on PATH for 4, no coverage module for the 2 jobs-postgres cases); with python on PATH only the 2 coverage cases fail. test_quality_triage, check_shard_coverage, check_backlog_task_ids and scope_predicate_ratchet exit 0; Bandit (uvx) reports nothing on the six source files. Rulings as applied: only model_not_allowed falls back; disabled provider and unreadable store refuse; analyze fallback fails closed when no quota check answers; the worker marks a job admitted only when its start check answered (its docstring now says a mid-item refusal can land for an unanswered start check). Follow-ups: TASK-15576 (the 19 resolve-time runtimes) and TASK-15577 (users_own_key_credentials' catch-all). The Flashcards Docs/Published mirror drift (TASK-15569) is committed with the refresh.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
One rule: a model the override denies falls back to the server key at analyze dispatch and on the MCP authoritative path, after a quota check that answers (fail-closed); a disabled provider or an unreadable store refuses; the resolve-time path logs provider and reason; ADR-086 amended; docs updated; follow-ups TASK-15576, TASK-15577.
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
