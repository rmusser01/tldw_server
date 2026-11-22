---
id: TASK-15618
title: 'Pin the mixed provider listing: server-key entries next to an own-key entry'
status: To Do
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
- [ ] #1 A test gives a user with an own key for one provider a server key for a second provider whose override has denied_models, and asserts that user's listing omits the denied model
- [ ] #2 A test asserts that user's listing hides a server-key provider the admin disabled (is_enabled=False)
- [ ] #3 Replacing the per-entry override application with a plain append fails these tests
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
