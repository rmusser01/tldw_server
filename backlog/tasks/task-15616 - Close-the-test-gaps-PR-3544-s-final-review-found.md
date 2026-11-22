---
id: TASK-15616
title: 'Close the test gaps PR #3544''s final review found'
status: To Do
labels:
- llm
- byok
- policy
- tests
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred minor M3 of PR #3544's final review (TASK-15582). (1) tests/Chat/integration/test_chat_endpoint_simplified.py (about line 1003): the validate_provider_override double swallows allowlist and still returns model_not_allowed, so the test pins a 403 from chat's first check that the real code can no longer produce (that check passes allowlist=False). (2) tests/Admin/test_admin_llm_provider_denied_models.py reads denied_models back from a stubbed get_llm_provider_override, so it proves only that build_override_response copies the field; the round trip through the repo and _parse_override_row is covered only in pieces. (3) Chat's failover check (chat.py about line 5303, allowlist=call_credential_source(refreshed_args) != USER_KEY_SOURCE) has no test: mutating it to allowlist=True would refuse an own-key failover to an off-list model unnoticed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The simplified chat test double refuses model_not_allowed only when called with allowlist true, so the test pins a refusal the real code can produce
- [ ] #2 A test sets denied_models through the admin API and reads it back through the real repo and _parse_override_row, with no stubbed get_llm_provider_override
- [ ] #3 A test pins chat's failover check: a fallback on the user's own key to a model off allowed_models dispatches, and a fallback on the server's key is refused model_not_allowed; mutating the check to allowlist=True fails it
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
