# UAT225 Stage B — independent review

## Disposition

**CLEAR for the frozen implementation. No remaining actionable finding within the approved Stage B boundary.** The reviewer found a real SQLite prior-device keyword incompatibility in the provisional candidate; the author preserved four causal failures, corrected the local consumer, and added permanent controls. That finding is resolved in this release. Native UAT225 acceptance and the full fresh matrix remain separate parent-owned gates.

This review covers TASK13260.164: nine production files and seven tests. Author manifest SHA256: `916e17df1a977d7531265736b3b031ab6f21abb3d67e6ae9affa2f0c93af2e7b`. Baseline: `42a65b4dd14620b5653d01e25a1ff138361b65f8`. Author report SHA256: `33d9f56bb69a3c70e8351abb0fa3877797dcbfee3a12da9297bb08256e2773bc`. All 16 working files and their review snapshots matched before and after independent verification; exact hashes are in `source-before.json` and `source-after.json`.

## Independent verification

| Check | Result |
| --- | --- |
| Eight-file focused suite, official required-PG runner | **173 passed, 0 skipped**, 190.60s; five warnings |
| Nine-file adjacent suite, official required-PG runner | **200 passed, 0 skipped**, 103.74s; four warnings |
| Ruff, all 16 files | 0 findings |
| Bandit, nine production files | 0 findings, 0 parse errors |
| Bandit, seven test files, only B101 excluded | 0 findings, 0 parse errors |
| Python compile, all 16 files | PASS |
| Final current/snapshot hash equality | 16/16 stable |

Commands and sanitized logs are copied here as `uat225-b-sidebar-{focused,adjacent}-command.json` and matching `.redacted.log` files. The fixtures exercised actual PostgreSQL and SQLite, including restricted-role calls and real transaction barriers. Provider replies were deterministic test inputs; no inference was performed. The source review also inspected the owned patch, changed-method inventory, permanent controls, and retained causal receipts. It did not substitute author results for the independent runs.

## Boundary review

- **Admission and authority:** local access is restricted to the exact selected-owner `legacy:<owner>` key while no canonical owner binding exists. It does not create a fake legacy authority row. Factory ownership and canonical readiness checks remain explicit.
- **Product decisions:** the local adapter reuses existing Notes operations, identity guards and transaction finalizers. Keyword creation alone cannot finalize acceptance. Product membership and final receipt completion commit or roll back together. Completed receipt replay remains immutable, including when the underlying resource later changes.
- **Canonical transition:** the shared service seam re-reads and validates the actual owned default personal dataset. Profile/default, Personal Context supplied/new dataset, and direct link/organization bootstrap paths fence local work before snapshots. PostgreSQL authority locks precede product locks; the tests observe real blocking in both orderings. Matching existing flags are preserved; conflicting bindings fail. Retirement is an authority transition with bounded subsequent cleanup, not a cross-database migration.
- **Retired cleanup:** only exact retired local scope gets maintenance access. It cannot authorize admission, publication or acceptance. Cancellation validates owner, job/run identity, domain, queue, type and payload. Already accepted products and terminal envelopes stay unchanged.
- **Late enqueue and races:** failed unbound admissions retain bounded lookup obligations beyond the ten-minute missing-job grace. Lease ordering allows later rows to progress with budget one. Cancellation and lookup do not extend original expiry or rewrite terminal receipts. Maintenance that loses the check-to-mutation race to enrollment defers through the narrow scope-error path. The original 30-day run and 90-day receipt retention contracts remain intact.
- **Canonical compatibility:** canonical coordinator paths remain separate. Lazy Notes DB selection preserves link-only collaborators, including the previously failing adjacent privacy control. Stage B makes no keyword schema, organization store, endpoint schema, or canonical wire-head changes; committed task168 is its separately reviewed persistence dependency.

## Resolved reviewer finding

The provisional adapter resolved a same-file SQLite keyword written by an earlier device, then passed it to a canonical membership path that treated the device label as an owner. Existing, merged and normalized same-label proposals failed; publication also failed to suppress a prior-device membership. The original candidate and finding remain in `PROVISIONAL-REVIEW.md`, `provisional-candidate-manifest.json`, and the author's `provisional-before-sqlite-review/` snapshots.

Retained author causal receipts show three consumer failures and then four failures including duplicate publication. The final local SQLite adapter uses ordinary per-file list/count/link operations and the existing NFC/casefold normalizer; PostgreSQL still uses owner-qualified lookup. Two membership reads omit the keyword device predicate only for the exact local SQLite scope. The permanent tests cover existing and merged tags, same-label and Unicode reuse, already-linked publication suppression, full prior keyword row preservation, and failure after actual acceptance finalization. The rollback control proves both product membership and decision completion roll back. Those controls pass in the independently executed focused suite. The initial correction's two fixture assertion failures remain retained and are distinguished from product failures.

## Limits

This is bounded source and automated integration approval, not native browser acceptance, a full matrix run, or a whole-product authorization audit. Raw-SQL isolation is not inferred from application predicates. Broader historical prior-device **note** behavior is unchanged and is not claimed covered by the keyword correction. Canonical enrollment retires local review state; seamless cross-dataset replay, migration and rejection continuity are intentionally not promised. No production/test edits, runtime/browser actions, native database access, task/tracker changes or git mutations were made by the reviewer.

`review-inputs.json` binds the author release and copied verification/causal receipts. `reviewer-manifest.json` binds this report and the private reviewer packet.
