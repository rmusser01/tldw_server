---
id: TASK-482
title: Implement MCP Unified Stage 3D security test-mode seam cleanup
status: Done
labels:
- mcp
- mcp-unified
- standalone
- stage3
modified_files:
- Docs/superpowers/plans/2026-05-29-mcp-unified-stage3d-security-test-seams-plan.md
- tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py
- tldw_Server_API/app/core/MCP_unified/tests/test_http_security_guards.py
- tldw_Server_API/app/core/MCP_unified/environment.py
- tldw_Server_API/app/core/MCP_unified/config.py
- tldw_Server_API/app/core/MCP_unified/security/ip_filter.py
- tldw_Server_API/app/core/MCP_unified/security/request_guards.py
- backlog/tasks/task-482 - Implement-MCP-Unified-Stage-3D-security-test-mode-seam-cleanup.md
references:
- https://github.com/rmusser01/tldw_server/pull/2108
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remove direct host testing-helper imports from MCP Unified security/config code while preserving test-mode behavior for IP and request guards. Keep the slice limited to standalone-extraction seams and focused regression coverage.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 security/ip_filter.py and security/request_guards.py no longer import tldw_Server_API.app.core.testing directly.
- [x] #2 config.py no longer imports host testing helpers directly; test-mode/env flag behavior is provided through a host-neutral helper or runtime seam.
- [x] #3 Regression tests cover the import boundary and existing test-mode loopback/client-certificate guard behavior.
- [x] #4 Focused pytest, Ruff, and Bandit verification are recorded before PR closeout.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Docs/superpowers/plans/2026-05-29-mcp-unified-stage3d-security-test-seams-plan.md
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the Stage 3D security/config test-mode seam cleanup on branch `codex/mcp-unified-stage3d-security-test-seams`.

RED verification: the import-boundary test failed because `config.py`, `security/ip_filter.py`, and `security/request_guards.py` imported `tldw_Server_API.app.core.testing`; the two behavior tests passed through the old host helper path.

GREEN implementation: added `tldw_Server_API/app/core/MCP_unified/environment.py` with dependency-free `env_flag_enabled`, `is_truthy`, `is_test_mode`, and `is_explicit_pytest_runtime` helpers. Updated MCP config and security guard modules to use the package-local helper while preserving existing test-mode behavior.

Verification:
- `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py::test_security_and_config_use_package_local_environment_helpers tldw_Server_API/app/core/MCP_unified/tests/test_http_security_guards.py::test_ip_allowlist_normalizes_missing_client_ip_in_test_mode tldw_Server_API/app/core/MCP_unified/tests/test_http_security_guards.py::test_client_certificate_guard_allows_testclient_only_in_test_mode -q` passed with 3 passed.
- `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_extraction_contracts.py tldw_Server_API/app/core/MCP_unified/tests/test_http_security_guards.py tldw_Server_API/app/core/MCP_unified/tests/test_config_safe_defaults.py -q` passed with 43 passed.
- `python -m ruff check ...` on the touched MCP files passed.
- `python -m bandit -r ... -f json -o /tmp/bandit_mcp_stage3d_security_test_seams.json` completed with zero findings and zero errors.
- `git diff --check` passed.
Review-fix pass after rebase onto origin/dev 4be634b5f3. Confirmed open PR threads: guard test type hints, broad exception swallowing in new tests, and production spoof risk for unconditional testclient loopback normalization.
Review fixes implemented after rebase: gated `testclient` IP normalization behind explicit test runtime, added a spoofed forwarded-header regression, annotated new test functions with `MonkeyPatch`/`-> None`, and replaced broad cache-clear exception swallowing in the new tests with a direct helper. Verification: focused pytest passed with 44 passed; Ruff passed on touched MCP files; Bandit returned zero findings; git diff --check passed.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
MCP Unified config and security guard test-mode helpers now live inside the MCP package boundary instead of importing host-level testing helpers. Focused import-boundary and guard behavior coverage protects the seam and existing loopback/client-certificate test-mode behavior.
PR: https://github.com/rmusser01/tldw_server/pull/2108
Review-fix pass rebased on origin/dev 4be634b5f3: addressed Gemini spoofing concern and both Qodo findings.
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
