# UAT292: billing API usage on the canonical schema

Tracking: TASK13260.229. Follow-up branch starts from PR2969's merged dev1dfdd819b6; UAT295 is already verified. This repair does not certify the pending full matrix.

## Stage 1: causal reproduction
**Goal**: Reproduce nonzero API usage incorrectly read as zero against canonical fresh schemas.
**Success Criteria**: SQLite and official-fixture PostgreSQL positive controls fail for the recorded missing-column cause; writer and organization attribution are traced.
**Tests**: Real DatabasePool initialization and AuthnzUsageRepo rollup, cross-org/day controls, explicit failure policy.
**Status**: Complete

## Stage 2: bounded repair and verification
**Goal**: Read canonical per-user/day requests through the existing AuthNZ repository.
**Success Criteria**: Aggregate each user once under the established billing primary-org rule (earliest added_at, lowest org_id tie-break); keep UTC day and configured failure policy. No schema or usage-writer changes.
**Tests**: SQLite and real PostgreSQL regressions, adjacent Billing/AuthNZ usage suites, scoped Ruff/Bandit and independent review.
**Status**: Complete

## Stage 3: targeted acceptance and records
**Goal**: Verify fresh PostgreSQL native ingestion/QA no longer emit UAT292 missing-column errors.
**Success Criteria**: Immutable-source targeted native evidence and positive API usage control retained; all findings and limitations tracked; owned apps/official fixtures cleaned up.
**Tests**: Native relevant operations, backend/cluster log audit, source hashes and cleanup receipts. No new full-matrix claim.
**Status**: In Progress

The canonical aggregate has no per-request organization or API-key attribution. This repair follows existing billing's deterministic user-primary-org convention and does not claim historical request-time organization accounting. Errors propagate from the repository to the enforcer's existing fail-open/fail-closed handling.

Causal reproduction: six positive/primary-org controls fail on both engines because nonexistent-column queries return zero; four unavailable-data controls pass. Revised canonical repository read passes all207 Billing tests (including10 real SQLite/PostgreSQL cases), no skips. Bandit0 findings/errors; Ruff11unchanged baseline diagnostics. Independent final review and adjacent usage verification pending. Fixture setup failures and invocation errors remain retained locally; they are not product regressions.

Final verification:229 unique scoped cases pass (209 Billing,20 adjacent AuthNZ),0PostgreSQL skips. Five baseline-proven PG fixture failures repaired with UsersDB and unchanged assertions. Review caught legacy nullable SQLite membership ordering; two canonical migration016 regressions failed then pass after excluding undated memberships. Independent final review clear. Bandit0findings/errors;15unchanged Ruff diagnostics/no new across7Python files. Native acceptance remains pending.
