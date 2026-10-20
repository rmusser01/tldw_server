---
id: TASK-13890
title: Connectors webhook GET and POST share one OpenAPI operationId
status: To Do
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
- [ ] #1 The connectors webhook's GET and POST have distinct operationIds (separate routes or explicit operation_id), with a test
- [ ] #2 test_openapi_operation_ids_are_present_and_unique passes after test_output_summaries_return_a_quota_refusal_as_402 in one process
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
