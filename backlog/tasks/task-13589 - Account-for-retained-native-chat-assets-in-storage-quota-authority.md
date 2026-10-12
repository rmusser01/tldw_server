---
id: TASK-13589
title: Account for retained native chat assets in storage quota authority
status: Done
labels:
- chat
- storage
- quota
- backend
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Continue approved H2 Task 2.2 in the isolated chat UX branch: add owner-bound same-key retained-byte reservations to the existing AuthNZ quota authority, include fresh native charges once in existing storage readers and admission, and preserve conservative cross-store cleanup. Public H2 capability remains disabled pending later projection and surface qualification.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Real AuthNZ reservations serialize current user and applicable shared-pool budgets; exact replay is idempotent and changed owner, scope, size or released identity cannot allocate again.
- [x] #2 Existing admission and public usage readers include legacy plus fresh native usage once; recalculation and caches retain legacy-only counters and exclude managed native bytes.
- [x] #3 Native quota intents and confirmed physical removal preserve same-key reserve/release recovery outside chat transactions; unknown outcomes never authorize uncharged live bytes or release.
- [x] #4 Run supported-runtime red/green, affected regressions, normal lint and Bandit, two independent reviews and normal hooks; record official unavailable PostgreSQL and broader H2 acceptance limits.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR assessment: existing ADR-049/050 govern native ownership and lifecycle; ADR-064 governs configured allowance and shared pool semantics. New L plus N accounting and durable native reservation rule needs ADR-070, proposed from approved H2 design section 8 and plan Task 2.2. Fresh origin/dev 19d213e936da37a908b30e79d1e4ff7a4c3839d1 is an ancestor of current HEAD 66b920dcc176a1c6be4842f097704abff4b20eda. Task/ADR collision scan includes 51 open PRs: maximum task 13588 and ADR 069. Root owns every runner and mutation; reviewers are source-only.
Real SQLite controls verified replay/terminal release, concurrent admission, identity conflicts, uncached budgets, filesystem/cache split, lost reserve/release responses, policy and cancellation. Two independent source reviews identified exact receipt-key checks, post-lock lease validation, component serialization, Billing NULL-membership policy, admin false-zero fallback and post-commit required ledger reads. Causal RED evidence under /private/tmp/chat-native-quota-13589-*.log; focused correction run: 88 passed, 13 official PostgreSQL skips. Subsequent real ordinary registration/delete/bulk/restore/voice and org/team accounting controls reproduced seven partial commits; existing transaction-bound service view now covers that tail, with physical voice cleanup deferred until commit. Qualification remains in progress; no public H2, PostgreSQL execution, AT or study claim.
Additional independent review reproduced stale batch accounting (real L/pools 1 instead of 0), conflicting file/owner lock order and stale post-lock single-delete state. Shared canonical selected-ID rereads now acquire globally sorted user/org/team locks before file mutation; bulk SQL returns exact changed IDs. Connected run:124 passed/65 official skips; its sole old exact-lock assertion was migrated to include stable org/team identity rows. Review also reproduced actual uncharged bytes after a caller-bound AuthNZ transaction rollback. Durable native facades now reject uncommitted bound views; focused durable/bootstrap run14 passed/3 official skips, including genuine AuthNZ prior100 and ChaCha78/82 preservation. Two final independent source CLEAR verdicts; full connected qualification remains in progress. Latest origin/dev19d213 still included.
Final accounting source:26-file supported Python3.12 connected444 passed/258 official/backend skips; final SQL/admin/bridge22 passed/3 official PG skips. Both normal client types exit0. Ruff21→21/Bandit11→11 introduce0 findings/errors; exact-HEAD inherited token-label warnings retained, new interpolation removed using bound boolean. Two final independent source CLEAR. Source-equivalent import/format checks pass. Canonical exporter loaded this checkout but omits existing storage routes in the full production catalog; TASK13590 tracks that discovered root. Stage3/AC2 remain open until production registration/contract qualification. Bounded review record Docs/Reviews/CHAT_NATIVE_QUOTA_13589_2026_10_11.md. Normal exact-owned hooks and incremental commit follow; no whole H2/public/physical acceptance claim.
All41 exact-owned normal repository hooks exit0 without bypass; supported3.12 compileall passes. Final SQL leaf cleared independently by both reviewers after boolean binding. Incremental accounting commit is ready while AC2/Stage3 stay open for TASK13590 production registration.
TASK-13590 now qualifies the remaining public route/contract gate: actual normal full app authenticated L1/N2/total3, admin components, all eight read/all seven write scope controls and original profile preservation pass. Canonical2129 paths/3301 schemas and both normal client configurations pass. Stage3 closure and task completion follow normal exact-owned hooks for the route repair; foundation source remains committed6efe408.
All exact-owned normal repository hooks passed without bypass. Remaining bounded acceptance criteria are complete; only the two owned completed plans retire to private evidence. Final metadata hooks and the incremental route-repair commit follow. Broader master acceptance remains open.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Committed retained native storage authority in 6efe408; TASK-13590 closes the final public-reader and canonical-contract gate. Legacy counters/caches remain separate from fresh native charges, with same-key durable reservation/release recovery and transactional ordinary lifecycle accounting. Supported accounting runs 444 passed/258 official/backend skips and 22 passed/3 official PostgreSQL skips; route qualification adds 322 passed, both normal client types, scoped static/Bandit, two independent source reviews and normal hooks. Broader H2 and physical acceptance remain open.
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
