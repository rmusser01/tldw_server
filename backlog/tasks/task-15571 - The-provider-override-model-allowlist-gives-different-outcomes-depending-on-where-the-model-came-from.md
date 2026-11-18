---
id: TASK-15571
title: The provider-override model allowlist gives different outcomes depending on
  where the model came from
status: To Do
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
- [ ] #1 Decide and record (owner) one rule for a model the override policy denies, whatever its source (configured, DEFAULT_MODEL_* fallback, model_override) and whatever the key (own key, server key, team/org): refuse, or fall back to the server key
- [ ] #2 _summarize_via_adapter enforces that rule at dispatch for every call, handle or not, as chat does; tests pin a denied model from each source (configured, DEFAULT_MODEL_* fallback, model_override) on the own-key and server-key paths
- [ ] #3 The MCP model_completion adapter's authoritative-scope path applies the same rule, with a test
- [ ] #4 A denied model writes one server log line naming the provider and the reason (no secrets)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [ ] #1 Acceptance criteria completed
- [ ] #2 Tests or verification recorded
- [ ] #3 Documentation updated when relevant
- [ ] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [ ] #5 Final summary added
- [ ] #6 Known skips or blockers documented
<!-- DOD:END -->
