---
id: TASK-13171
title: Enforce active Personal Context exchange without breaking Sync V2
status: Done
assignee:
  - '@codex'
created_date: '2026-09-04 03:34'
updated_date: '2026-09-04 23:04'
labels:
  - personal-context
  - sync
  - security
  - api
dependencies:
  - TASK-13170
references:
  - >-
    backlog/tasks/task-13161 -
    Relay-ordered-Personal-Context-authority-publications-through-Sync-V2.md
  - >-
    backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md
  - Docs/superpowers/specs/2026-09-02-personal-context-ongoing-sync-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Remediate TASK-13161 by applying one precise active-exchange proof gate to Personal Context Sync V2 operations while preserving unrelated legacy and mixed-domain behavior.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 One service gate protects Personal Context version-one push, pull, conflict listing, and conflict resolution with the exact persisted activation epoch and token plus a completed link receipt for the requesting device.
- [x] #2 A Personal Context version-zero ongoing exchange returns activation_required before mutation, delivery, or cursor advancement.
- [x] #3 Legacy first-link flows and operations containing no selected Personal Context data continue to work.
- [x] #4 A mixed Personal Context and Notes dataset can list and resolve unrelated Notes conflicts without Personal Context activation proof.
- [x] #5 Personal Context conflict listing uses the real requesting device identity and completed link receipt, and conflict resolution gates from the selected conflict identities rather than unrelated dataset contents.
- [x] #6 Only verified stored proof is echoed; exact, stale epoch, stale token, missing proof, incomplete link, tampered proof, and wrong-device cases have distinct verified outcomes.
- [x] #7 Production TestClient tests cover push, pull, conflict list, and conflict resolve across Personal Context-only, mixed-domain, legacy, missing, stale, incomplete-link, tampered, and exact-proof cases.
- [x] #8 ADR required: no new ADR; backlog/decisions/002-personal-context-profile-authority-sync-and-encryption.md governs activation proofs and compatibility.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add RED TestClient proof and mixed-conflict tests. 2. Centralize selected-operation Personal Context detection. 3. Require exact proof plus completed device link only for selected Personal Context work. 4. Preserve version-zero and unrelated Sync behavior. 5. Run targeted endpoint and security checks. 6. Self-review and close the task. ADR required: no new ADR; ADR-002 governs.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented one selected-operation Personal Context exchange gate across push, pull, bounded conflict listing, and selected-ID resolution. Exact persisted activation proof and the explicit requesting device's completed link receipt are validated only by require_active_exchange; responses echo persisted verified proof. Added backward-compatible conflict-list domain/device_id queries and parameterized domain/status/order/limit/offset DB selection. Added a 51-case TestClient and service/store matrix covering exact, missing, stale, wrong-device, incomplete-link, tampered, version-zero, mixed Notes, pagination, non-leakage, and pre-effect behavior. ADR required: no new ADR; ADR-002 governs. Targeted endpoint/transport/service/conflict tests, Ruff, Bandit, and diff-check passed. The pre-existing PostgreSQL fake receipt-lock failure was reconfirmed on the predecessor baseline and excluded; this task does not modify link receipts. Details: .superpowers/sdd/2026-09-04-personal-context-relay-remediation/task-6-report.md.

Review round 1 changes requested: require currently registered active device inside the sole exchange validator; resolve mixed Notes plus Personal Context batches per selected item; close the preflight-to-resolution TOCTOU gap with one service-owned transaction/snapshot; and fail closed for exact malformed persisted/request proof text. TDD review plan: reproduce every finding, report RED evidence and atomic coordinator design, implement minimally, rerun targeted endpoint/transport/service/conflict/concurrency/static checks, update this task and the task-6 report, then make one scoped review commit.

Review round 1 completed: `require_active_exchange` now verifies that the explicit requesting device is currently active and non-revoked, strictly revalidates both persisted and supplied proof shapes, and fails closed for malformed Unicode and non-mapping stored state. Conflict resolution now uses one service-owned dataset transaction and selected-row snapshot, gates selected Personal Context before any savepoint or mutation, and contains ordinary per-item failures with backend-portable savepoints while preserving input order, duplicate replay, and missing/foreign/already-resolved behavior. Mixed exact-proof Notes plus Personal Context batches resolve normally; bad proof mutates neither. The expanded 84-case gate, endpoint/transport, conflict service/store, Personal Context service/contract, Ruff, Bandit, and diff checks passed. ADR required: no new ADR; ADR-002 remains the governing decision.

Review round 2 changes requested: remove batch-proof-driven Personal Context shape rules from the request schema so Notes skip/overwrite/duplicate-rename retain native semantics in a mixed exact-proof batch; enforce strict Personal Context item shape only after the atomic service snapshot identifies a selected Personal Context conflict; and scope row-lock lookups by both dataset and conflict ID while locking unique selected IDs in deterministic sorted order. TDD round-2 plan: reproduce mixed-action and PostgreSQL lock-scope/order findings, report exact RED evidence and the service-validation/locking design, implement minimally, rerun the full gate and affected endpoint/transport/service/store/backend/security matrices, update this task and the task-6 report, then make one scoped review commit. ADR required: no new ADR; ADR-002 already governs these validation and isolation boundaries.

Review round 2 completed: the request schema now preserves only generic per-item structural validation, while the atomic service snapshot determines which selected conflicts require Personal Context-specific candidate shape. The central exchange gate still runs before shape rejection or mutation. Dataset-scoped parameterized row lookup prevents foreign-row locks, and unique selected IDs are locked in deterministic sorted order before results are mapped back to original input order and duplicate semantics. Mixed exact-proof Notes skip, overwrite, and duplicate-rename use native Notes shapes and materialize successfully alongside Personal Context items. Focused review, full 89-case gate, 130 endpoint/transport cases, 154 request-model cases, conflict service/store, Personal Context service/contract, Ruff, Bandit, and diff checks passed. The previously baselined PostgreSQL link-receipt fixture failure remains excluded and untouched. ADR required: no new ADR; ADR-002 remains the governing decision.
<!-- SECTION:NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Personal Context Sync V2 exchange gates now apply only to selected work, require exact persisted proof plus the real device's completed link receipt before effects, and preserve legacy and Notes-only compatibility. ongoing_sync_version remains 0.
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
