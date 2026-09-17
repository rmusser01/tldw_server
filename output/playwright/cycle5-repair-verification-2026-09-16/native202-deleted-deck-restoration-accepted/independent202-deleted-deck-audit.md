# UAT202 independent native API acceptance audit

TASK13260.140 acceptance criterion3 is satisfied for the bounded **native ordinary-user API deleted-deck name reuse** operation, paired with the already retained independent source/regression review. This is not browser Delete UI acceptance or full matrix/whole-backend isolation acceptance.

The retained script makes a fresh synthetic Bob login, uses its token only in memory, and never reads browser material. Login response/session headers are not retained. It checks actor3 before the mutations and targets only a newly created own empty private deck. The script logs out this session (`all_devices:false`) in finally. Its result contains no credential/token/header-shaped fields. I reviewed the script but did not execute it or make any native request.

Independent parsing confirms all **9 actual request receipts returned200**, from2026-09-17T06:14:47.717Z through06:14:47.876Z:

- Login200→auth/me actor3.
- Initial deck list contains only existing own deck7.
- New own deck8 begins atversion1; DELETE with expected_version1 returnsdeleted=true.
- include_deleted list shows the same deck8 tombstone atversion2.
- Same-name POST restores the same deck8 atversion3, owner3, deleted=false.
- Canonical GET exactly equals the restoration response; all pre-existing deck7 fields are unchanged.
- Current-session logout200 reports user3.

The source-before-start receipt records revisiona130b8e5507ff2f1ec6a930a7aa12183a44b5d44. Its ChaChaNotes_DB.py SHA6089c5e0cac45fd0dd2c5253ad906b20108746e35a9189bf3934fc1eca9506b7 matches both the actual git bytes at that runtime revision and the reviewed repair commit61e5af6702. This independently binds the exercised named-row restore implementation to the reviewed code. Parent reports the owned API remained on this non-hot-reload source; no additional runtime introspection was performed here.

`independent202-deleted-deck-verification.json` records exact source/evidence hashes and machine-checked assertions. Existing foreign-owner restoration rejection and SQLite compatibility remain supported by the separate required-fixture regression/review receipts; this single live sequence does not independently reprove them. It makes no raw-SQL RLS or other backend-role claim. The ordinary fresh-login harness is distinct from the rejected browser interceptor approach and uses no captured browser session.

No browser, runtime, configuration, DB, source, task, tracker or git mutation by this reviewer. Only these private audit artifacts were written.
