---
id: TASK-15616
title: 'Close the test gaps PR #3544''s final review found'
status: Done
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
- [x] #1 The simplified chat test double refuses model_not_allowed only when called with allowlist true, so the test pins a refusal the real code can produce
- [x] #2 A test sets denied_models through the admin API and reads it back through the real repo and _parse_override_row, with no stubbed get_llm_provider_override
- [x] #3 A test pins chat's failover check: a fallback on the user's own key to a model off allowed_models dispatches, and a fallback on the server's key is refused model_not_allowed; mutating the check to allowlist=True fails it
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Measured: t3green '5 passed'; M-a (the TASK-15615 first-check call reverted) fails item 1 (assert 200 == 403); M-b (_parse_override_row denied_models=None) fails item 2 '(None, None, None)' while the stubbed Admin test passes (1 failed, 1 passed); M-c (failover allowlist=True) fails [own-key-fallback] only (403 != 200; 1 failed, 1 passed); t3files 42 passed, 0 failed, comm empty, netprobe 0. AC #2: admin API = the admin service functions the routes call (R8); the HTTP route is not patched, because the endpoint-monkeypatch ratchet is at its ceiling. Docs: none relevant. Bandit: not run (not installed), test-only change. Known limit: the server-key fallback is refused by the credential runtime before the failover check, so only the own-key row pins that check.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Three test gaps from PR #3544's final review are closed. The simplified chat test's validate_provider_override double now refuses model_not_allowed only when called with the allowlist on, as the real check does, so the test pins chat's first-check refusal (it fails when the TASK-15615 call is reverted). A new SQLite test sets denied_models through the admin service (upsert_override, get_override) and reads it back through the real repo, the store refresh and _parse_override_row, padding and blanks dropped and an empty list clearing it. A new chat test pins the failover check: a fallback to a model off allowed_models dispatches on the user's own key and is refused model_not_allowed on the server's key; mutating the check to allowlist=True fails it.
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
