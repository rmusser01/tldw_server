---
id: TASK-12077
title: Fix PR 2557 grouped CI failures
status: Done
labels:
- ci
- pr-2557
- bugfix
priority: High
references:
- https://github.com/rmusser01/tldw_server/pull/2557
- https://github.com/rmusser01/tldw_server/actions/runs/28453472892
modified_files:
- Docs/Schemas/chatbooks_manifest_v1_1.json
- tldw_Server_API/app/api/v1/endpoints/explainer.py
- tldw_Server_API/app/core/Sandbox/service.py
- tldw_Server_API/tests/Admin/test_admin_smoke.py
- tldw_Server_API/tests/Chatbooks/test_chatbooks_manifest_v1_1_contract.py
- tldw_Server_API/tests/Chunking/test_chunking_templates.py
- tldw_Server_API/tests/Workflows/test_workflow_templates_api.py
- tldw_Server_API/tests/integration/test_chatbook_integration.py
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Fix the grouped failing CI checks on PR #2557 after shard coverage was repaired. Root-cause groups: Chatbooks v1.1/explainer statistics contract drift, Explainer auth dependency import boundary, workflow run visibility on Ubuntu Python 3.13, Windows sandbox queued-claim race, macOS chunking temp SQLite disk I/O, and Windows admin temp DB file handle cleanup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->

<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Fixed grouped PR #2557 CI failures by aligning Chatbooks v1.1 schema/test expectations with emitted Explainer session statistics, routing Explainer endpoint auth imports through API_Deps.auth_deps, renewing sandbox claim leases before short leases can expire, draining ChaChaNotes resources during admin smoke cleanup for Windows, using pytest tmp_path for chunking template temp DBs, and polling workflow run visibility in the async template flow test. Verification: targeted pytest set passed (9 passed); Bandit on touched runtime files passed with zero findings; pre-commit on touched files passed; git diff --check passed.
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
