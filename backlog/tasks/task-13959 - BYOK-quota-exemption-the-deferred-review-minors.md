---
id: TASK-13959
title: 'BYOK quota exemption: the deferred review minors'
status: Done
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
- [x] #1 A per-request base_url or api_base_url off the vendor's hosts makes a user-key call user_unmetered, checked and counted; on the vendor's host it stays user, with tests at the adapter and at the gateway (sync and async)
- [x] #2 A billed provider without a config section reads as user_unmetered instead of raising, and a test keeps every VENDOR_HOSTS provider in PROVIDER_APP_CONFIG_KEYS
- [x] #3 A * in a VENDOR_HOSTS pattern matches one DNS label, with a test
- [x] #4 VENDOR_HOSTS covers OpenAI's data-residency hosts (*.api.openai.com), api.cohere.com, api.moonshot.ai and Bedrock's FIPS endpoints, with a test, and a test keeps every billed adapter's default base on its vendor's host
- [x] #5 A provider method nested in an own-key stream is not checked, sync and async, with a test
- [x] #6 A hosted over-limit own-key /chat/completions request looks up the user's key once, with a test
- [x] #7 users_own_key_serves is tested on real SQLite BYOK storage (stored, revoked, user deactivated), and the allowlist stub honours its argument
- [x] #8 The long docstring line is wrapped; ADR-083 gets an amendment, Usage_Quotas.md is corrected, and the Docs/Published mirror is refreshed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: Docs/superpowers/plans/2026-10-31-byok-quota-minors.md. Owner answers 2026-10-31: plan approved with the review's M1, M2 and M4 folded in; Decision 1 (OpenAI hosts) the wildcard *.api.openai.com (review M5) instead of ten literal hosts; Decision 2 (Change summary) 'Yes, waiver covers it'.
RED/GREEN: Task 1 20 failed, 16 passed -> 36 passed (the plan's 17/15 -> 32, plus fold M2's gateway cases and a two-label test for fold M5); Task 2 3 failed, 69 deselected -> 3 passed; Task 3 12 passed (tests for existing behaviour; mutation probes: revoked key 2 failed/1 passed, allowlist on the raw name 2 failed/7 passed, the old stub 9 passed); final-review fix test_an_admitted_call_to_a_trusted_callers_off_host_base_url_counts fails with the base_url read removed (['user'] != ['user_unmetered']). Focused set 70 -> 100 passed. Wider set: SQLite 4 failed, 6862 passed, 17 skipped, 1 error -> 4 failed, 6891 passed, 17 skipped, 1 error (the same five pre-existing ids: test_document_factory_starts_without_default_executor_and_drains_cancel, test_list_evaluations, two test_media_ingest_jobs_video_dedupe tests, and the teardown error of test_extractor_never_opens_files); PostgreSQL 6 -> 6 passed. Chat streaming suites (8 files) 232 passed, 1 skipped. tests/Docs + tests/lint 478 passed. Shard coverage new_uncovered=0; test-quality ratchet OK; route-auth ratchet 208 exact; OpenAPI fingerprint and path union unchanged; create_chat_completion 2908 -> 2905 lines.
Decisions: OpenAI data-residency hosts as the wildcard *.api.openai.com, one DNS label (owner, Decision 1, review M5); a billed provider with no config section counts (user_unmetered), per the owner's brief; streams use _wrap_chat's gate, so a checked stream that records nothing also reads its items as an accounted call; chunks are parsed only when there is a plan (review M4); the ADR is amended, not superseded. Known limits: one label per * still matches the legacy S3 look-alikes and IDNA labels (fold M3; only the operator or a trusted caller sets this URL); an empty label and a trailing-dot FQDN are deferred minors (final review).
Docs: ADR-083 amendment and index row, Usage_Quotas.md (own-key bullet); Docs/Published refreshed (ADR-083, ADR README). BYOK guide unchanged (names no host). Bandit (uvx, -ll, the 5 touched source files): No issues identified, Medium 0, High 0. Reviews: plan review 0B/0I/6M; task reviews all Approved; final review (clean session) 0 Critical, 0 Important, 5 Minor, Ready to merge; Minor 1 re-graded and fixed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Closes PR #3371's deferred minors. A trusted caller's per-request base_url or api_base_url is classified like a configured base URL, so a user-key call off the vendor's hosts is user_unmetered, checked and counted. VENDOR_HOSTS lists the hosts each vendor documents (OpenAI's data-residency hosts as *.api.openai.com, api.cohere.com, api.moonshot.ai, Bedrock FIPS), and a * matches one DNS label. A billed provider without a config section counts instead of raising. A provider method nested in an own-key stream is no longer checked, and streams that record nothing no longer parse their chunks. /chat/completions asks users_own_key_serves once per request. users_own_key_serves is tested on real SQLite storage, and the allowlist stub honours its argument. ADR-083 amended; Usage_Quotas.md corrected; Docs/Published refreshed.
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
