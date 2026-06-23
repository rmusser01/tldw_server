---
id: TASK-9928
title: Harden External_Sources review findings
status: Done
assignee: []
created_date: '2026-06-23 18:50'
updated_date: '2026-06-23 21:40'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Track remediation for External_Sources code review findings: file-sync delta create/restore handling, atomic OAuth state consumption, fail-closed connector job creation, token encryption posture, Notion Markdown sanitization, and explicit unsupported base connector capabilities.
<!-- SECTION:DESCRIPTION:END -->

## Definition of Done
<!-- DOD:BEGIN -->
- [x] #1 Acceptance criteria completed
- [x] #2 Tests or verification recorded
- [x] #3 Documentation updated when relevant
- [x] #4 Bandit run for touched code when applicable or document non-code/environment skip
- [x] #5 Final summary added
- [x] #6 Known skips or blockers documented
<!-- DOD:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 File-sync provider deltas create first-seen files and restore archived/orphaned bindings
- [x] #2 OAuth state consumption is atomic and single-use
- [x] #3 Connector job creation failures surface instead of returning fake queued jobs
- [x] #4 Connector secrets fail closed when encryption is required
- [x] #5 Notion Markdown output escapes unsafe text and image URLs
- [x] #6 Base connector unsupported capabilities raise explicitly
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Completed; temporary implementation plan file removed after verification per repository guidance.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Touched production files:
- tldw_Server_API/app/core/External_Sources/connector_base.py
- tldw_Server_API/app/core/External_Sources/connectors_service.py
- tldw_Server_API/app/core/External_Sources/notion.py
- tldw_Server_API/app/services/connectors_worker.py

Added focused regression coverage in External_Sources tests for delta reconciliation, OAuth state replay, fail-closed job creation/token storage, Notion rendering sanitization, and base connector contract behavior.

Verification:
- python -m py_compile on touched production files
- ULTRA_MINIMAL_APP=1 python -m pytest -p no:unraisableexception tldw_Server_API/tests/External_Sources/test_connectors_worker_file_sync.py tldw_Server_API/tests/External_Sources/test_connectors_service_sanitizers.py tldw_Server_API/tests/External_Sources/test_token_refresh_envelope.py tldw_Server_API/tests/External_Sources/test_notion_connector_sanitizers.py tldw_Server_API/tests/External_Sources/test_sync_adapter_contract.py tldw_Server_API/tests/External_Sources/test_reference_manager_storage.py tldw_Server_API/tests/External_Sources/test_policy_and_connectors.py::test_notion_download_renders_nested_blocks -q --tb=short (25 passed)
- python -m bandit -r touched production files -f json -o /tmp/bandit_external_sources_9928.json (0 findings)

Known skip: full default pytest import path timed out in unrelated optional-router app import; rerun used ULTRA_MINIMAL_APP=1 to keep verification scoped to these unit tests.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Hardened External_Sources review findings: normalized Drive/OneDrive content_updated deltas into create/restore actions as appropriate, made OAuth state consumption atomic, removed fake queued-job fallback, required connector secret encryption in multi-user/production-required modes, sanitized Notion Markdown rendering, and made unsupported base connector capabilities explicit.
<!-- SECTION:FINAL_SUMMARY:END -->
