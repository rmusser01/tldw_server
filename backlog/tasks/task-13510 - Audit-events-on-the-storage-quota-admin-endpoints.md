---
id: TASK-13510
title: Audit events on the storage quota admin endpoints
status: Done
labels:
- quotas
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PUT /api/v1/storage/admin/quotas/user/{id} and PUT /api/v1/admin/storage-quotas/users/{id} change a user's storage quota without an audit event; PUT /admin/users/{id} and the profile path do emit one. Parent: TASK-13434.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Both endpoints emit the same audit event as PUT /admin/users/{id} when the quota changes
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Both per-user storage quota PUT routes now emit the same audit event as PUT /admin/users/{id} (USER_UPDATED, action admin.user.update, metadata storage_quota_mb) through emit_storage_quota_audit_event in services/admin_audit_service.py, called from the two endpoints. Emitting inside StorageQuotaService.set_user_quota was tried first and reverted: direct service callers with no actor (the VN generated-file harness) hung creating an audit service for a null user in a short-lived subprocess, and auditing the acting admin belongs to the admin action, not the domain service. PUT /admin/storage-quotas/users/{id} now also records the acting admin as updated_by (it passed None). Removed two unused imports in admin_storage_quotas.py.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
PUT /api/v1/storage/admin/quotas/user/{id} and PUT /api/v1/admin/storage-quotas/users/{id} record the same USER_UPDATED / admin.user.update audit event as PUT /admin/users/{id}, with the acting admin and the new storage_quota_mb (null when removed). Test: tests/Storage/test_user_quota_endpoints.py::test_user_quota_endpoints_emit_the_admin_user_update_audit_event (both routes; set, clear, and no event on a 404), red before the change. Verification: 26 test files referencing the changed symbols plus Storage, Admin and lint: 1363 passed, 1 failed (test_admin_llm_provider_test_runtime timing test under -n 4; 34 passed alone); tests/Docs passed; OpenAPI fingerprint unchanged; Bandit -ll 0 issues; ruff clean on the touched files. Docs: Storage_API_Documentation.md anchors re-verified and the audit event documented.
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
