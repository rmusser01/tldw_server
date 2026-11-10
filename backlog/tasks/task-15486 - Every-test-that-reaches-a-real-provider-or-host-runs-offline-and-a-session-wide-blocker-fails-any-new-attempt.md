---
id: TASK-15486
title: Every test that reaches a real provider or host runs offline, and a session-wide
  blocker fails any new attempt
status: To Do
labels:
- tests
- ci
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Owner request (2026-11-07), filed by TASK-15485. A test that reaches a real provider or host can spend a real key, flake with the network, and leak a request; on dev these tests mostly pass because the code under test swallows the failure. Measured on 7a020da14a by running each test inside tldw_Server_API/tests/helpers/offline_network.block_live_network() (a pytest plugin that blocks and logs every non-loopback DNS lookup and connect). 28 tests: (1) Prompt_Management_NEW/integration/test_prompt_improvement_api.py::test_llm_quota_refusal_answers_402_not_retryable_provider_unavailable: api.openai.com. (2) Media_Ingestion_Modification/test_media_processing.py, 15: TestProcessVideos::test_process_video_url_success and ::test_process_video_multi_status_mixed (www.youtube.com); TestProcessAudios::test_process_audio_url_success_no_analysis_no_chunking (cdn.pixabay.com; fails when blocked) and ::test_process_audio_multi_status_mixed (example.com); TestProcessPdfs::test_process_pdf_url_success (www.w3.org); TestProcessDocuments::test_process_doc_multiple_success and ::test_process_doc_overrides_and_options (raw.githubusercontent.com); TestProcessEbooks::test_process_ebook_url_success_no_analysis_no_chunking, ::test_process_ebook_multiple_success, ::test_process_ebook_multi_status_mixed, ::test_process_ebook_overrides, ::test_process_ebook_with_analysis_mocked and ::test_process_ebook_options_extraction_method[basic], [filtered], [markdown] (filesamples.com). (3) Media_Ingestion_Modification/test_media_contextual_integration.py, 7 (example.com): TestMediaEndpointContextualIntegration::test_add_media_contextual_defaults_from_config, ::test_add_media_with_contextual_chunking_enabled, ::test_add_media_with_contextual_chunking_disabled, ::test_contextual_chunking_preserves_other_options and ::test_contextual_chunking_with_different_media_types[document-sentences], [ebook-ebook_chapters], [pdf-sentences]. (4) Evaluations/test_security.py, 2: TestWebhookSecurityValidation::test_blocked_ports (example.com) and ::test_ssrf_protection_private_ip (getaddrinfo on 192.168.1.1, 10.0.0.1, 172.16.0.1 and 169.254.169.254: numeric hosts, so no lookup leaves the machine, but the blocker counts them and the test fails under it). (5) Evaluations/test_production_features_integration.py, 3 (example.com): TestWebhookManager::test_webhook_registration and ::test_webhook_unregistration, TestIntegration::test_full_evaluation_flow_with_features. Owned elsewhere, not in this task: RAG/test_restricted_postgres_media_retrieval.py's two stream tests (api.openai.com; the [sqlite] cases measured, the [postgresql] cases need PostgreSQL) by TASK-15482; RAG_NEW/integration/test_rag_stream_parity.py::test_rag_streaming_generation_provider_override (api.groq.com) by TASK-15427 (#3439). How many tests a session-wide blocker would fail today (a dry count; a full run of 5379 test files, about 45,000 test functions, was too slow locally): 31 tests attempted in the 137 files run under the probe (the files above, a 31-file Evaluations set and a random 100-file sample, seed 15427): the 28 above and the 3 owned elsewhere. Statically, 729 test files outside e2e/, server_e2e_tests/ and frontend_e2e/ hold a non-loopback URL literal (12,885 test functions); in the random sample 1 of 50 such files and 0 of 50 others attempted, which puts the unmeasured rest at roughly 15 files, i.e. tens of tests, not hundreds. The first CI run with the blocker gives the exact number. Probe ceilings: per test window only; a thread that outlives its test is not seen; a server subprocess is not probed.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Each listed test stubs its download, provider or DNS call and passes inside block_live_network() with no attempt
- [ ] #2 A session-wide blocker in the test config (an autouse fixture in tldw_Server_API/tests/conftest.py around tests/helpers/offline_network.block_live_network(), asserting no attempt at teardown as LLM_Adapters/conftest.py's no_live_network does) makes a new outbound DNS lookup or connect fail loudly in CI, with a test that shows the failure; getaddrinfo on a numeric non-loopback host passes through (no lookup leaves the machine) while its connect stays blocked; loopback is decided with ipaddress.ip_address(host).is_loopback (as tests/helpers/netblock_13956 does), not the four exact strings of offline_network._LOOPBACK (127.0.0.2, ::ffff:127.0.0.1, 0.0.0.0 and the machine's own hostname count as live otherwise), and the host of a developer's configured TEST_DATABASE_URL / TEST_DB_HOST is allowed; a function-scoped autouse fixture does not cover session- or module-scoped fixture setup, so the AC names that ceiling and the notes say which such fixtures were checked
- [ ] #3 Tests that legitimately need the network opt out: by a new registered live_network marker, by the existing external_api, local_llm_service and provider_inventory markers, and by directory for e2e against a live server (e2e/, server_e2e_tests/, frontend_e2e/); loopback is never blocked; the opt-out is documented in Docs/Development/Testing.md and tldw_Server_API/tests/README.md
- [ ] #4 A full-suite CI run passes with the blocker on, and the notes record how many tests it failed before the stubs
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
