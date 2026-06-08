---
id: TASK-2304
title: Add MCP permission-change governance hooks
status: Done
labels:
- mcp
- policy
- governance
- admin
- followup
references:
- Docs/superpowers/specs/2026-06-07-mcp-fs-patch-write-safe-edit-tools-design.md
- Docs/superpowers/specs/2026-06-07-mcp-tool-call-hooks-design.md
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Add hooks and audit records for MCP profile/path-grant changes. Permission changes should be traceable and optionally require approval hooks, especially for granting write and future delete/share/export/admin/lock authorities.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] Add an injectable MCP profile permission-change governor with deny/ask/allow outcomes and an allow-by-default implementation.
- [x] Gate profile create, duplicate-from-preset, patch, delete, and default-profile change mutations before persistence.
- [x] Block denied permission changes with `permission_change_denied` and approval-required changes with `permission_change_requires_approval`.
- [x] Emit redacted audit metadata for allowed and blocked permission changes without raw policy documents, path prefixes, or secret-like payloads.
- [x] Allow profile semantic patches for flat/authored path-grant policy fields so path-grant changes can be governed.
- [x] Map new profile-management reason codes to stable HTTP statuses.
- [x] Add focused regression tests and run touched-scope security verification.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Plan: `Docs/superpowers/plans/2026-06-08-mcp-permission-change-governance-hooks-implementation-plan.md`

Touched files:
- `mcp_unified/gateway/profile_governance.py`
- `mcp_unified/gateway/profiles.py`
- `mcp_unified/gateway/fastapi.py`
- `mcp_unified/gateway/__init__.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py`
- `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`

Verification:
- Red check: `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py::test_create_profile_calls_permission_governor_with_redacted_summary -q` failed before implementation with missing `mcp_unified.gateway.profile_governance`.
- `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py::test_gateway_profile_management_error_status_mapping -q` passed.
- `python -m ruff check mcp_unified/gateway/profile_governance.py mcp_unified/gateway/profiles.py mcp_unified/gateway/fastapi.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py` passed.
- `python -m py_compile mcp_unified/gateway/profile_governance.py mcp_unified/gateway/profiles.py mcp_unified/gateway/fastapi.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py` passed.
- `python -m bandit -r mcp_unified/gateway/profile_governance.py mcp_unified/gateway/profiles.py mcp_unified/gateway/fastapi.py -f json -o /tmp/bandit_mcp_permission_governance.json` passed with 0 findings.

Review follow-up:
- Rebasing on `origin/dev` reported the branch was already up to date.
- Added create-time and patch-time path-grant validation regressions for malformed `path_grants`.
- Normalized default `PermissionChangeDecision.reason_code` values so blocked outcomes cannot emit `allowed`.
- Hardened policy summary helpers for Pydantic v1-style `.dict()` payloads, empty collections, and tuple/set wildcard policy values.
- Re-ran `python -m pytest tldw_Server_API/app/core/MCP_unified/tests/test_gateway_profile_management.py tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py::test_gateway_profile_management_error_status_mapping -q`; 78 passed.
- Re-ran the ruff, py_compile, and Bandit commands above; all passed and Bandit reported 0 findings.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented an allow-by-default MCP profile permission-change governance seam. Profile create, duplicate, patch, delete, and default assignment changes now call the injected governor before persistence; deny and ask decisions block mutations with stable reason codes and redacted audit events. Path-grant policy patch fields are accepted and validated for governance, and FastAPI maps the new reason codes to 403/409.
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
