---
id: TASK-368.3
title: Implement llama.cpp provider wiring and diagnostics
status: Done
assignee: []
created_date: '2026-05-15 03:43'
updated_date: '2026-05-15 06:50'
labels:
  - implementation
  - backend
  - llamacpp
dependencies:
  - TASK-368.2
documentation:
  - Docs/superpowers/specs/2026-05-15-llamacpp-server-management-webui-design.md
  - >-
    Docs/superpowers/plans/2026-05-15-llamacpp-server-management-webui-implementation-plan.md
parent_task_id: TASK-368
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Implement the provider diagnostics backend slice from the implementation plan. Add explicit use-in-chat provider wiring, bounded managed log tailing, best-effort hardware snapshot, and permission coverage. Do not change frontend behavior in this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 POST /api/v1/llamacpp/use-in-chat explicitly updates only the llama.cpp provider endpoint after a managed server is running.
- [x] #2 GET /api/v1/llamacpp/logs/tail returns bounded managed logs and cannot read arbitrary paths.
- [x] #3 GET /api/v1/llamacpp/hardware returns best-effort RAM/CPU/GPU data with structured warnings and no hard dependency on NVIDIA hardware.
- [x] #4 New endpoints retain admin-only permission coverage and focused backend tests pass.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Started subagent-driven implementation for the provider wiring, hardware snapshot, and safe log tail backend slice after TASK-368.2 passed spec and code-quality review. Scope is limited to Task 3 backend files and tests from the implementation plan.

Task 3 implemented in f2283f92d with provider wiring, managed log tail, hardware snapshot, schemas, endpoint wiring, and permission coverage. Log-tail hardening fixes followed in a44eebf4f to require a running managed server, active managed log handle evidence, and canonical configured/status log path agreement before reading. Redaction hardening followed in f85bcf74c to cover equals-style CLI secrets and quoted assignment forms after a read-only quality review found those remaining leaks.

Verification recorded: Task 3 focused bundle `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_provider_and_logs_api.py tldw_Server_API/tests/AuthNZ_Unit/test_llamacpp_permissions_claims.py -q` passed with 33 passed, 5 warnings. Backend llama.cpp regression bundle `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/LLM_Local/test_llamacpp_admin_config_api.py tldw_Server_API/tests/LLM_Local/test_llamacpp_inventory_api.py tldw_Server_API/tests/LLM_Local/test_llamacpp_management_api.py -q` passed with 52 passed, 5 warnings. Focused redaction suite passed with 9 passed, 5 warnings. `git diff --check` passed. Bandit on `tldw_Server_API/app/core/Local_LLM/llamacpp_provider_service.py` wrote `/tmp/bandit_llamacpp_task3_redaction_fix.json` with empty results/errors.

Review status: spec review approved at f2283f92d. Code-quality review requested log-tail path hardening and broader redaction coverage. Re-review at f85bcf74c approved with no blocking or important findings. Residual note: fd-identity comparison for active logs would be a future hardening step if symlink retargeting becomes in-scope.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Implemented explicit llama.cpp provider wiring and safe diagnostics. The new admin-only endpoints wire the running managed server into `Local-API.llama_api_IP`, expose a bounded managed log tail with path and secret-redaction safeguards, and return best-effort hardware data without requiring NVIDIA libraries. Tests and permission coverage are in place, with follow-up hardening for log path proof and additional secret syntaxes completed before finalizing.
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
