---
id: TASK-13959
title: 'BYOK quota exemption: the deferred review minors'
status: In Progress
labels:
- quotas
- backend
- byok
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Owner request 2026-10-31: 'address the deferred minor findings' of PR #3371's final review (TASK-13955, ADR-083). (1) A trusted caller's per-request base_url (admin, service principal, single user) reaches the billed adapters as request['base_url'] without classification, so a call to an operator host on the user's key reads as 'user'. (2) billed_credential_source does a hard PROVIDER_APP_CONFIG_KEYS[name] lookup that raises KeyError for a billed provider with no config section. (3) The Bedrock host pattern's * spans dots. (4) VENDOR_HOSTS lacks the vendors' documented alternate hosts (OpenAI's data-residency domains, api.cohere.com, api.moonshot.ai, Bedrock FIPS), and the docs say they count. (5) An own-key stream with no recording plan returns the provider's stream unwrapped, so a provider method nested in the stream body is checked and can refuse an over-limit user mid-stream. (6) A hosted over-limit own-key /chat/completions request runs users_own_key_serves twice. (7) No test drives users_own_key_serves against real key storage, and the allowlist stub ignores its argument. (8) usage_repo.insert_llm_usage_log's docstring has a line over 120 characters. Plan: Docs/superpowers/plans/2026-10-31-byok-quota-minors.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A per-request base_url or api_base_url off the vendor's hosts makes a user-key call user_unmetered, checked and counted; on the vendor's host it stays user, with tests at the adapter and at the gateway (sync and async)
- [ ] #2 A billed provider without a config section reads as user_unmetered instead of raising, and a test keeps every VENDOR_HOSTS provider in PROVIDER_APP_CONFIG_KEYS
- [ ] #3 A * in a VENDOR_HOSTS pattern matches one DNS label, with a test
- [ ] #4 VENDOR_HOSTS covers OpenAI's data-residency hosts (*.api.openai.com), api.cohere.com, api.moonshot.ai and Bedrock's FIPS endpoints, with a test, and a test keeps every billed adapter's default base on its vendor's host
- [ ] #5 A provider method nested in an own-key stream is not checked, sync and async, with a test
- [ ] #6 A hosted over-limit own-key /chat/completions request looks up the user's key once, with a test
- [ ] #7 users_own_key_serves is tested on real SQLite BYOK storage (stored, revoked, user deactivated), and the allowlist stub honours its argument
- [ ] #8 The long docstring line is wrapped; ADR-083 gets an amendment, Usage_Quotas.md is corrected, and the Docs/Published mirror is refreshed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-31-byok-quota-minors.md. Owner answers 2026-10-31: plan approved with the review's M1, M2 and M4 folded in; Decision 1 (OpenAI hosts) the wildcard *.api.openai.com (review M5) instead of ten literal hosts; Decision 2 (Change summary) 'Yes, waiver covers it'.
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
