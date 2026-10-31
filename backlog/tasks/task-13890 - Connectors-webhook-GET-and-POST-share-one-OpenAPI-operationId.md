---
id: TASK-13890
title: Connectors webhook GET and POST share one OpenAPI operationId
status: Done
labels:
- backend
- tests
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
connectors.py declares /providers/{provider}/webhook with api_route(methods=["GET", "POST"]), so both methods get the operationId provider_webhook_callback_api_v1_connectors_providers__provider__webhook_post. The connectors router is non-stable (default_stable=False) and is left out of the default schema, but an app that a test builds can mount it into the cached schema. On origin/dev 2e6ceab9fb, Watchlists/test_watchlists_api.py::test_output_summaries_return_a_quota_refusal_as_402 followed by Services/test_openapi_contracts.py::test_openapi_operation_ids_are_present_and_unique in one process gives 1 failed, 1 passed: Duplicate OpenAPI operationIds. A combined -n 4 run can order them that way (TASK-13864 plan review). Found while planning TASK-13864.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The connectors webhook's GET and POST have distinct operationIds (separate routes or explicit operation_id), with a test
- [x] #2 test_openapi_operation_ids_are_present_and_unique passes after test_output_summaries_return_a_quota_refusal_as_402 in one process
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
RED: new test 1 failed. GREEN: webhooks file, openapi contracts and watchlists 402 test 86 passed; route auth ratchet 13 passed after hand-editing the baseline (GET,POST entry became GET and POST entries). Drift check matches, no api-types diff. No frontend use of provider_webhook_callback. Bandit -ll clean. Docs: none needed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Webhook registered as stacked get and post routes; operationIds end _get and _post.
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
