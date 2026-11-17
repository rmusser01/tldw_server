---
id: TASK-13590
title: Register existing storage routes in the full production app
status: Done
labels:
- chat
- storage
- backend
- api
dependencies:
- TASK-13589
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Canonical exporter and normal full-app source prove the existing storage router is present only in minimal-test optional registration. Restore production reach for authenticated usage, file lifecycle and admin quota handlers using the existing content router group, then qualify public routes and both shared API consumers. Continue user-authorized chat UX root-cause remediation; no new native-fork capability.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The full production router catalog reaches the existing storage router exactly once under the house storage route policy and original API prefix.
- [x] #2 Real authenticated public usage exposes legacy/native totals; existing owner/admin authorization and disabled route policy remain enforced.
- [x] #3 Regenerate/check the canonical API fingerprint and path union; both normal client type configurations remain valid.
- [x] #4 Run causal red/green, connected router/storage tests, scoped static/Bandit, two independent source reviews and normal exact-owned hooks before commit.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: existing grouped-router policy and ADR-069 govern production registration and canonical API artifacts; no new durable architecture rule or ADR is required. Scope is the existing content router catalog, registration/public contract tests, canonical API artifacts and bounded qualification records. Source probe loaded this worktree main.py, found storage_schemas absent from full OpenAPI, and found new fields only when directly importing the actual model. Current canonical fingerprint remains49734587c27f42521abec7d08fe03f0eaca689d330aac86b6277348f2c223031 (2114 paths/3276 schemas). Search found no duplicate storage-router task; scanmax13589 acrossdev and51openPRs.
Plan: Docs/Plans/IMPLEMENTATION_PLAN_chat_storage_route_13590.md. Scope checked: full content catalog registration, existing composed storage endpoints and house route gating, production-route regressions, canonical shared package fingerprint/path union, frontend and extension type consumers. Existing endpoint authorization and persistence contracts are reused.
Independent review found activation would expose existing storage mutation routes to nonadmin read-only API keys. Scope includes house write-scope dependencies on all user storage mutation routes, preserving JWT/single-user/admin house semantics; qualify real read-only refusal with unchanged metadata/legacy/native counters and authorized write success. The normal multi-user admin fixture must assign canonical RBAC admin membership using AuthnzUsersRepo, not merely users.role; scalar role correctly does not grant canonical admin claims. Initial fixture diagnostics are excluded from acceptance.
Symmetric independent review confirms nonadmin write-only keys currently reach user-facing storage GETs even though house has_scope requires explicit read. Scope includes existing read dependencies on all eight user GETs, retaining documented JWT/single-user/admin semantics. First canonical generation qualified2129 paths/3301 schemas, adding15 paths and25 schemas, fingerprint3b8ce49f0c3f97a0fc22ebe66df5599f528d431148096c500ae36a64ae4d2691; regenerate after final scope qualification.
Final qualification: genuine catalog/authenticated404 RED; read-only delete200/deletedtrue/L1-to0 RED; write-only metadata200 RED. Seven-file connected run322 passed/192 warnings in64.03s using supported3.12, no skipped tests. Both independent final source reviews CLEAR. Canonical generation and full exporter fingerprint/path checks exit0:2129 paths/3301 schemas, SHA3b8ce49f0c3f97a0fc22ebe66df5599f528d431148096c500ae36a64ae4d2691. Both normal client type configurations exit0. Eight-file Ruff baseline1/current1/introduced0 (unchanged folder exception-chain warning); six-production-file Bandit baseline0/current0/errors0; supported compileall0. New regression Black preserves exact AST. Same normal full private API19767 refreshed on3.12, usage/breakdown200 with components and combined-once proof. Original9 conversations/28 rows deep-equal before/after; immutable baseline not rewritten. No browser opened. Normal exact-owned hooks and incremental commit remain.
All exact-owned normal repository hooks passed without bypass. Remaining bounded acceptance criteria are complete; only the two owned completed plans retire to private evidence. Final metadata hooks and the incremental route-repair commit follow. Broader master acceptance remains open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Registered the existing storage router in the normal production content group and enforced house read scopes on all eight user GETs and write scopes on all seven mutations. Genuine normal-app and key regressions, 322 connected checks, canonical API generation/checks, both normal client types, scoped static/Bandit, two independent reviews and exact-owned repository hooks pass. Same private profile preserves all nine original conversations and 28 messages. Public H2, physical PostgreSQL/download, installed-extension, AT and participant qualification remain open.
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
