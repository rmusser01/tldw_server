# PR3016 Manual Review Repairs

Associated Backlog task: original TASK-13369, Address PR3016 Qodo durability
findings. The direct requester instructed "do a manual review and merge it".
This replaces the unavailable Qodo-only gate with fresh manual exact-head review;
it does not waive required CI, the human Change summary or normal merge rules.
PR3067 remains strictly read-only.

The full 99-path comparison of f1f063f against actual dev1fc353c was split into
VN, database, shared storage/AuthNZ, Jobs, frontend and controller scopes.
Three production findings were verified and repaired:

- A competing registration can return another delivery's stored image. Attach
  that winner's byte length and content type, retaining frozen dimensions and
  request identity. Completed replay must not compare against the losing render.
- The owning service's usage caches must be invalidated even when a response is
  cancelled after commit, or a source-reference replay returns inside the
  transaction. A finally block preserves cancellation and accounting semantics.
- VN deletion and user/org/team usage decrements must commit together. Reuse the
  existing per-item AuthNZ transaction, re-read the record and lock quota scopes
  in the existing order. A bound service joins that connection; a failed removal
  retains registration and charges for terminal redelivery. Non-VN removal keeps
  its previous flow.

A proposed Jobs RLS grant was withdrawn after tracing actual callers and existing
role restrictions. No permissions change was made. An inherited smoke test now
uses Recover pending request after an aborted retry, while asserting ordinary
Retry remains disabled; its original idempotency and layout assertions remain.

## Verification

The 11 new regressions failed at intended assertions on the original production
implementation. The first repair run passed ten native accounting/cache cases;
the winner-size assertion passed but a subsequent new test lookup used the wrong
repository field name. That attempt remains failed, not relabeled green.

After correcting only that lookup, the bounded registration, replay, cancellation
cleanup and service matrix passed 135 tests, zero failures/errors/skips, with 226
warnings. It includes real isolated SQLite/PostgreSQL fixtures. Selected existing
frontend recovery component tests passed 123, with 134 outside the name filter.
The two browser smoke cases failed before reaching recovery due a local React
runtime dependency error; this is neither a smoke pass nor the intended RED.
No dependency files were changed to hide it.

Four-file Bandit comparison: 540 to 549 B101 assertion findings, six inherited
B106 unchanged, no errors or new non-assert findings. Ruff retains one inherited
BLE001; this is not a blanket clean result. Exact full-PR configured hooks passed
11 checks with three no-file skips, without changes or bypass.

Evidence is retained under the SDD ledger's manual-review-merge-20261006-r1
directory. Local Python3.11/pytest8.4/asyncio1.1 remain below declared floors;
available frontend dependencies are not a lockfile certification. These scoped
results do not certify the whole repository, supported runtimes or future CI.
Fresh independent SPEC/QUALITY review and exact published-head required CI are
mandatory before normal match-head merge; prior f1f063f passes do not transfer.
