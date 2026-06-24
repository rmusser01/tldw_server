---
id: TASK-9928
title: Harden External_Sources review findings
status: Done
assignee: []
created_date: 2026-06-23 18:50
updated_date: 2026-06-24 04:06
labels: []
dependencies: []
priority: high
modified_files:
- tldw_Server_API/app/core/External_Sources/connector_base.py
- tldw_Server_API/app/core/External_Sources/connectors_service.py
- tldw_Server_API/app/core/External_Sources/notion.py
- tldw_Server_API/app/services/connectors_worker.py
- tldw_Server_API/app/core/exceptions.py
- tldw_Server_API/tests/External_Sources/test_connectors_service_sanitizers.py
- tldw_Server_API/tests/External_Sources/test_connectors_worker_file_sync.py
- tldw_Server_API/tests/External_Sources/test_notion_connector_sanitizers.py
- tldw_Server_API/tests/External_Sources/test_sync_adapter_contract.py
- tldw_Server_API/tests/External_Sources/test_token_refresh_envelope.py
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
Hardened External_Sources review findings and addressed PR #2454 comments, including PR Compliance ID 224214 docstring cleanup for the Notion sanitization helpers and connector secret helper focus area. Verification includes AST docstring checks, compile, focused tests, and Bandit with 0 findings.
<!-- SECTION:FINAL_SUMMARY:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
PR #2454 remediation started after automated review comments. Scope: rebase on latest origin/dev; handle Notion code literal rendering, safe URL parsing, SQLite commit fallback, docstrings/type hints/custom exception review items; refresh verification and push updated branch.
PR #2454 remediation completed after rebasing on latest origin/dev. Addressed automated review comments by preserving literal Notion code block text, using collision-free inline code spans, guarding malformed Markdown image URLs, avoiding the missing-commit-method await failure, adding docstrings/type hints for touched helpers, and replacing connector-domain RuntimeError raises with ConnectorServiceError.

Verification after remediation:
- python -m py_compile tldw_Server_API/app/core/External_Sources/connector_base.py tldw_Server_API/app/core/External_Sources/connectors_service.py tldw_Server_API/app/core/External_Sources/notion.py tldw_Server_API/app/services/connectors_worker.py tldw_Server_API/app/core/exceptions.py (passed)
- ULTRA_MINIMAL_APP=1 python -m pytest -p no:unraisableexception tldw_Server_API/tests/External_Sources/test_connectors_worker_file_sync.py tldw_Server_API/tests/External_Sources/test_connectors_service_sanitizers.py tldw_Server_API/tests/External_Sources/test_token_refresh_envelope.py tldw_Server_API/tests/External_Sources/test_notion_connector_sanitizers.py tldw_Server_API/tests/External_Sources/test_sync_adapter_contract.py tldw_Server_API/tests/External_Sources/test_reference_manager_storage.py tldw_Server_API/tests/External_Sources/test_policy_and_connectors.py::test_notion_download_renders_nested_blocks -q --tb=short (29 passed)
- python -m bandit -r touched production files including exceptions.py -f json -o /tmp/bandit_external_sources_pr2454_rebase.json (0 findings)
Final latest-dev rebase verification after origin/dev advanced again:
- Rebased onto origin/dev e18a86bb0186862e5ff408049d18ee882d3a8269.
- python -m py_compile touched production files including exceptions.py (passed)
- ULTRA_MINIMAL_APP=1 focused External_Sources pytest slice (29 passed)
- Bandit JSON at /tmp/bandit_external_sources_pr2454_rebase_latest.json reported 0 findings.
Second latest-dev rebase verification after origin/dev advanced again:
- Rebased onto origin/dev 5851988ca9f8f485048d84eee19396f4cc1926ec.
- python -m py_compile touched production files including exceptions.py (passed)
- ULTRA_MINIMAL_APP=1 focused External_Sources pytest slice (29 passed)
- Bandit JSON at /tmp/bandit_external_sources_pr2454_rebase_5851988.json reported 0 findings.
Third latest-dev rebase verification after origin/dev advanced again:
- Rebased onto origin/dev 54eadfcdb2a612e3b517ead4a4686fbaba5d34a1.
- python -m py_compile touched production files including exceptions.py (passed)
- ULTRA_MINIMAL_APP=1 focused External_Sources pytest slice (29 passed)
- Bandit JSON at /tmp/bandit_external_sources_pr2454_rebase_54eadfc.json reported 0 findings.
Follow-up docstring compliance cleanup: verifying PR Compliance ID 224214 against the Notion sanitization helper area and connector secret helper area. The named helpers already have docstrings; adding docstrings for nearby focus-window functions surfaced by AST scan to avoid follow-up compliance noise.
Docstring compliance follow-up completed for PR Compliance ID 224214. Added first-statement docstrings for nearby focus-window functions that the AST scan surfaced in addition to the already-documented Notion sanitization and connector secret helpers.

Verification for docstring follow-up:
- AST docstring scan over Notion lines 1-90 and connectors_service lines 120-180 found docstrings for all functions in those windows.
- python -m py_compile tldw_Server_API/app/core/External_Sources/notion.py tldw_Server_API/app/core/External_Sources/connectors_service.py (passed)
- ULTRA_MINIMAL_APP=1 focused pytest for test_connectors_service_sanitizers.py and test_notion_connector_sanitizers.py (12 passed)
- Bandit JSON at /tmp/bandit_external_sources_docstrings_224214.json reported 0 findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
