---
id: TASK-15618
title: 'Pin the mixed provider listing: server-key entries next to an own-key entry'
status: Done
labels:
- llm
- byok
- tests
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred minor M1 of PR #3554's final review (TASK-15583); needs #3554. In tldw_Server_API/app/api/v1/endpoints/llm_providers.py (about lines 1871-1877), when a signed-in user has own keys, every other provider entry goes through the per-entry apply_llm_provider_overrides_to_listing. No test has an override on a second provider, so replacing that else branch with providers.append(entry) passes all 12 own-key tests, and a signed-in user with any own key would then see another provider's denied models and a disabled provider's server models (TASK-15583 AC #2).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A test gives a user with an own key for one provider a server key for a second provider whose override has denied_models, and asserts that user's listing omits the denied model
- [x] #2 Replacing the per-entry override application with a plain append fails these tests
- [x] #3 A test asserts that user's listing shows a server-key provider the admin disabled as unusable (enabled and provider_enabled false, no models)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Measured: origin/dev held #3554 (merge commit d0964c7b63); t4base 13 passed; t4green 16 passed; t4red (else branch replaced by providers.append(entry)) 3 failed, 13 passed; netprobe 0 throughout. AC #2 (the disabled row) was amended per RR5: the test asserts the provider listed as unusable, not hidden. Docs: none relevant. Bandit: not run (not installed), test-only change. Known limit: for a disabled server-key provider only 'enabled' differs under the mutation; the builder already lists it unusable with no listed models.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
The mixed provider listing is pinned. A user with their own OpenAI key, next to a Groq entry on the server's key, still sees Groq held to the admin override: a model on Groq's denied_models and a model off its allowed_models are not listed, and a Groq the admin disabled is listed unusable (enabled and provider_enabled false, no models). Replacing _users_listing's per-entry override application with a plain append fails all three rows. The branch merges dev after #3554.
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
