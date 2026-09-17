# Independent UAT222 review — CLEAR, bounded source acceptance

TASK13260.160. No remaining blocking correctness finding in the frozen two-file StudyPack/provenance ownership repair. Native Alice-owned/Bob-foreign acceptance remains the parent's gate. No product/test/browser/runtime/task/tracker/git changes were made by this reviewer.

## Exact reviewed boundary

- ChaChaNotes_DB.py SHA256 `63b3ccd15b9ce19bd1d964821eb0d33b4f191ff1a329cd83356e8481a6423d97`.
- test_study_pack_owner_contract.py SHA256 `4182f5357e4e591c5176fb054043268ea8b6ac9ccc5819ce9dab26bb5837f5ff`.
- Author manifest `9f61176f2a1fec65739e84a676f14e36e2b986477a2bce719eee375ffe88e3c6`; source and snapshot equality verified before and after execution.
- Independent AST comparison verifies exactly12 existing StudyPack/provenance methods plus the existing `_require_selected_owner_row` extension changed. Every other production AST node matches baseline `0bbf4442fa4f41d53ff61c3261b480387bcbdc866e1f15785244fbdb6065513c`. Schema, operation ownership, transaction managers, Notes branches, worker accessors and parent-owned223 provenance routing are not altered by this patch.

## Correctness review

The four read methods bind the selected PostgreSQL database owner's client ID. Membership reads require owner agreement on the membership, pack and card; citation reads require the citation and its card. The first-pack lookup filters all relevant parents before ORDER BY/LIMIT, retaining a later valid owned membership. Existing deletion/status semantics remain explicit: this does not broadly add live-parent rules to historical read paths.

Writes validate ownership inside existing transactions. Destination deck uses the established PostgreSQL locked check. Pack/card parent checks use fixed allowlisted identifiers and bound values, including UUID lookup only for flashcards. Membership validates the pack and all unique cards before its first insert; deterministic card lock order does not change insertion order. Citation replacement scopes old child rows so malformed foreign history cannot be rewritten. Parent locks keep subsequent existing id-based writes in the same ownership transaction. Supersession retains sorted pair-lock order, optimistic versions and self/deleted replacement checks; soft delete retains owned stale-version and idempotent behavior.

No new commit, rollback or savepoint behavior was added. The permanent caller-transaction control proves a caught parent denial preserves earlier caller work, then an explicit outer rollback removes membership/card/citation/sync effects. A propagated injected write failure likewise rolls back with its caller on both backends. This preserves the existing nested-transaction contract; it does not assert new inner-savepoint atomicity when a caller catches an unrelated low-level write error.

SQLite's owner clauses and locked checks remain no-ops, preserving same-file sync-device access and mutation; separate account files remain separate. Administrator completed-job access already chooses the authorized job owner's database, so these predicates bind that selected owner rather than the ambient administrator. The existing endpoint detail/regenerate path now receives None for a foreign pack before serialization or enqueue; owned-card assistant reads use scoped child methods. No new recipient-sharing or source-bundle authorization system is claimed.

## Independent evidence

- Official required-PostgreSQL runner, actual isolated PostgreSQL/SQLite fixtures: **63 passed, zero skipped,4 warnings,81.90s**. All frozen tests ran, including real detail/regenerate/assistant routes, malformed historical children, owner/deleted/version/order/duplicate controls, genuine NOSUPERUSER/NOBYPASSRLS read controls, caller rollback and SQLite device semantics.
- Fresh Ruff on both owned paths: **0 findings**.
- Fresh Bandit: **0 findings /0 parse errors** on production and test (B101 excluded only for test assertions).
- Both Python sources compile in memory; independent AST13-only comparison passes.
- Author retained causal receipts inspected: corrected read8fail/9controls; corrected write10fail/10SQLite controls; transaction/order2fail/4controls; historical citation replacement replay2PostgreSQL failures/3SQLite controls. Initial fixture/import/transaction-test mistakes are documented separately and not counted as product failures.
- Author's additional11-file adjacent run is **204 passed /zero skipped /201.79s**. This is author evidence, not an independently repeated204-run. Its existing storage/generation, provenance, completed-job/admin-owner, serialization, worker lifecycle, membership count and Flashcard owner-edge coverage was reviewed for applicability.

## Reproduce

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat222-sidebar-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs tldw_Server_API/tests/StudyPacks/test_study_pack_owner_contract.py -q --tb=short
```

The existing explicit-Jobs wrapper removes inherited JOBS_DB_URL only; mandatory PostgreSQL provisioning uses the official fixture. It does not reprovision the native database. Regenerate tests enqueue isolated fixture jobs, with no worker or model call. Route dependency overrides bind fixture principals/stores; this proves actual router/storage behavior, not browser login.

## Limits

The original native source role is privileged/BYPASSRLS. Actual restricted-role tests here establish application read predicates; no table RLS is added or raw-SQL isolation asserted. Source-reference authoring and workspace-sharing policy remain the prior contract. No broad domain audit or concurrent multi-operation deadlock proof is claimed. Original native job5/pack2/deck10 were untouched. The independent source review is clear; native ownership readback remains required before task completion.
