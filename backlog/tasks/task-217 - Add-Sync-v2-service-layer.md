---
id: TASK-217
title: Add Sync v2 service layer
status: Done
assignee: []
created_date: '2026-05-10 03:59'
updated_date: '2026-05-10 04:38'
labels:
  - sync
  - service
  - api
dependencies:
  - TASK-212
references:
  - tldw_Server_API/app/core/Sync/v2/store.py
  - tldw_Server_API/app/core/DB_Management/Sync_DB.py
documentation:
  - >-
    Docs/superpowers/plans/2026-05-10-chatbook-sync-engine-implementation-plan.md
  - Docs/superpowers/specs/2026-05-10-chatbook-sync-engine-prd-design.md
  - tldw_Server_API/app/api/v1/schemas/sync_v2_models.py
priority: medium
---

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Sync v2 service, adapter registry/protocol, and security helpers are added with injected store and deterministic test hooks.
- [x] #2 Service tests cover capabilities, device registration refresh, default personal dataset enrollment, adapter registry known/unknown domains, dataset access rejection, per-envelope push outcomes, unsupported adapter versions, pull cursor paging/filtering/echo policy, and metadata-only restore manifests.
- [x] #3 Security tests cover private restore manifests and log redaction without plaintext private payload or ciphertext leakage.
- [x] #4 Focused pytest for service/security tests passes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inspect existing Sync v2 schema/store behavior and keep core internals independent from API schemas except at service boundaries.
2. Write focused service and security tests for capabilities, device/dataset lifecycle, adapter registry validation, push outcomes, pull paging/filtering/echo behavior, restore manifests, and redaction.
3. Implement adapters.py protocols/result types/registry plus security.py validation and redaction helpers.
4. Implement SyncV2Service with injected store, user/device args, deterministic clock/id hooks, settings/capabilities, and minimal store facade extensions only where needed.
5. Run focused service/security pytest, existing sync model/store tests if shared behavior changes, git diff --check, Bandit on touched production files, then update TASK-217 and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented Sync v2 service layer with adapter registry/protocol result types, private payload validation/redaction helpers, deterministic service hooks, push/pull orchestration, and restore manifest metadata inventory. Added DB/store read helpers for user datasets/devices and made Sync v2 package store export lazy to avoid an import cycle with Sync_DB.

Verification:
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_security.py -v: 15 passed, 5 warnings.
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_store.py -q: 26 passed, 5 warnings.
- git diff --check: passed.
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r touched production files -f json -o /tmp/bandit_sync_v2_service.json: 0 findings.

Docs: no user-facing docs updated; this task is internal service-layer substrate and endpoint/API docs are later tasks.
Known blockers: none.

Spec review follow-up: fixing two Task 3 blockers. Plan: add regressions for echo-filled pull windows and recursive human-readable private-field redaction; run the focused service/security tests to see them fail; patch SyncV2Service.pull bounded scan paging and security redaction policy; rerun focused tests, existing Sync model/store tests, git diff --check, and Bandit; then mark task done and commit a fix.

Spec review follow-up complete. Added regression coverage for pull paging when same-device echoes fill the first raw window before a later remote row, and for recursive redaction of label/title/body/content private text fields. Fixed pull to scan in bounded chunks while advancing the server-sequence cursor until it finds page_size + 1 visible rows or exhausts the store. Fixed private mapping redaction to conservatively redact human-readable content fields while preserving safe IDs/status/routing fields.

Follow-up verification:
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_security.py -v: 17 passed, 5 warnings.
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_store.py -q: 26 passed, 5 warnings.
- git diff --check: passed.
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Sync/v2/service.py tldw_Server_API/app/core/Sync/v2/security.py -f json -o /tmp/bandit_sync_v2_service_fix.json: 0 findings.

Spec re-review follow-up: fixing three remaining service invariants. Plan: add failing regressions for missing-domain cursor defaults, cross-dataset envelope rejection, and max-batch overflow outcomes; patch service cursor resolution and push validation; rerun focused service/security tests, existing Sync model/store tests, git diff --check, and Bandit; then mark TASK-217 done and commit.

Spec re-review follow-up complete. Added regression coverage for missing domain cursors during stored-cursor pulls, cross-dataset envelope rejection before persistence, and explicit batch_limit_exceeded outcomes for envelopes beyond max_batch_size. Fixed SyncV2Service.push to validate envelope dataset IDs and return overflow rejections for every submitted envelope. Fixed SyncV2Service.pull to resolve stored cursors after selecting dataset domains, with missing selected-domain cursors contributing 0.

Second follow-up verification:
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_security.py -v: 20 passed, 5 warnings.
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_store.py -q: 26 passed, 5 warnings.
- git diff --check: passed.
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Sync/v2/service.py -f json -o /tmp/bandit_sync_v2_service_invariants.json: 0 findings.

Spec re-review follow-up: enforce runtime adapter domain validation in core. Plan: add a regression asserting SyncAdapterRegistry.register rejects StaticSyncAdapter(domain="bogus"), verify the focused tests fail, add a core allowlist for Sync v2 domains in adapters.py, rerun focused tests, existing Sync model/store tests, git diff --check, and Bandit; then mark TASK-217 done and commit.

Spec re-review follow-up complete. Added runtime Sync adapter domain validation with a core allowlist for notes, chat, workspaces, source_cache, and media. Added regression coverage proving StaticSyncAdapter(domain="bogus") is rejected during registry registration.

Adapter-domain follow-up verification:
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_security.py -v: 20 passed, 5 warnings.
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_store.py -q: 26 passed, 5 warnings.
- git diff --check: passed.
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Sync/v2/adapters.py -f json -o /tmp/bandit_sync_v2_adapter_domains.json: 0 findings.

Code quality review follow-up: patching production blockers for ownership invariants, request-device enforcement, accepted-only pull visibility, unenrolled-domain prevalidation, payload-size enforcement, non-repeatable default IDs, and cursor persistence errors. Plan: add regression tests first across service/store, verify failures, implement minimal store/service fixes, rerun focused Sync tests plus diff/Bandit, then mark task done and commit.

Code quality review follow-up complete. Added ownership guards in SyncDatabase for device and dataset upserts so existing records cannot be reassigned across users. Hardened SyncV2Service push/pull invariants: request device_id is enforced, missing envelope device_id is filled from the request, unenrolled domains and oversized payloads are rejected per envelope before adapter/store work, normal pulls exclude non-accepted envelopes, cursor persistence errors now propagate, and default clock/ID generation uses UTC timestamps plus UUID-style IDs. Added regression coverage for all required critical and important issues. Deferred only the service file size concern, as allowed by review.

Code quality follow-up verification:
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_service.py tldw_Server_API/tests/Sync/test_sync_v2_security.py -v: 28 passed, 5 warnings.
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m pytest tldw_Server_API/tests/Sync/test_sync_v2_models.py tldw_Server_API/tests/Sync/test_sync_v2_store.py -q: 28 passed, 5 warnings.
- git diff --check: passed.
- source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate && python -m bandit -r tldw_Server_API/app/core/Sync/v2/service.py tldw_Server_API/app/core/DB_Management/Sync_DB.py -f json -o /tmp/bandit_sync_v2_ownership_invariants.json: 0 findings.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Added the Sync v2 service layer for capabilities, device registration, dataset enrollment, push, pull, and restore manifest assembly without wiring endpoints or concrete domain adapters. The implementation keeps adapters injectable, returns per-envelope push outcomes, validates adapter versions and private payload metadata, redacts ciphertext/key material from log helpers, and exposes metadata-only restore manifests for private datasets. Added focused service/security tests and small Sync DB/store read helpers needed by restore manifests.

Spec review fix: hardened SyncV2Service.pull paging so echo-filtered pulls continue scanning in bounded server-sequence chunks until a visible page or exhaustion, and expanded private log redaction to recursively redact human-readable private fields such as labels, titles, bodies, and content.

Second spec re-review fix: tightened service invariants so stored cursor resolution treats missing selected-domain cursors as 0, push rejects envelope dataset mismatches before persistence, and push returns explicit non-retryable batch_limit_exceeded outcomes instead of silently dropping overflow envelopes.

Final spec re-review fix: SyncAdapterRegistry now rejects unknown adapter domains at runtime using a core-owned Sync v2 domain allowlist, with regression coverage for bogus domains.

Code quality follow-up: closed ownership takeover vectors for datasets/devices, enforced authenticated request device identity in push, added per-envelope domain/payload validation, kept normal pull results to accepted envelopes, propagated cursor persistence errors, and replaced repeatable default IDs with UUID-style generation.
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
