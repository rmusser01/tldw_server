# Independent native acceptance audit — UAT194

## Disposition

**PASS for the targeted native account-isolation gate.** Recommend closing TASK13260.132 with its already recorded independent ownership/schema/security review and zero-skip PostgreSQL/SQLite tests. No new product defect is demonstrated by these receipts.

The first native script remains explicitly **failed**: it expected 404 for a foreign restore and received 409. Its original source and 23-record result are preserved and hash-bound. The separate 15-record continuation validates the actual restore contract against a nonexistent ID, proves the foreign tombstone remained unchanged, then completes owned restore and isolation checks. This does not retroactively turn the first script into a passing run.

## Actual API controls

Both scripts used the running API at `127.0.0.1:18503`, fresh synthetic Alice/Bob login sessions and test-created characters. These are actual authenticated API calls, not browser mutation clicks or mocked responses. The auditor only read the retained script/result files and did not execute them or read their credential inputs.

All times are UTC on 2026-09-17. `/auth/me` verifies Alice ID 2 and Bob ID 3 in both runs.

| Boundary | Observation |
| --- | --- |
| Alice creates own fixture | **201 at 07:56:59.650**, character **5 / version 1**, name `UAT194 private character 20260917 0757`, Alice-only violet-compass description. |
| Bob foreign visibility | Detail **404**; list excludes ID5; name query returns total 0; search returns `[]`. |
| Bob foreign mutation | PUT description **404 at 07:56:59.742** and DELETE **404 at 07:56:59.758**. Alice’s following read remains version 1 with its original description. |
| Owned update | Alice PUT succeeds **200 at 07:56:59.791**, version 2, expected updated Alice-only description. |
| Same name, separate owner | Bob creates **ID6 / version 1**, **201 at 07:56:59.809**, using the same name and a Bob-only amber-map description. Alice cannot fetch ID6. Each account’s list/query returns its own fixture and excludes the other. |
| Owned delete | Alice DELETE succeeds **200 at 07:56:59.909**, advancing its fixture to tombstone version 3. |
| Initial harness stop | Bob restore receives **409 at 07:56:59.925**, with a “not found” detail. The script’s expected404 assertion stops execution; its result retains `passed: false`. Both finally-logouts return 200. |
| Independent absent control | Continuation repeats foreign restore at **07:58:03.732** and nonexistent ID999999999 restore at **07:58:03.749**: both **409**, same “not found” message structure apart from the submitted IDs. |
| No foreign restore effect | Alice `deleted_only=true` query at **07:58:03.768** still finds ID5 / version 3 with its expected description. Bob `include_deleted=true` query returns only its own ID6. |
| Owned restore | Alice restore **200 at 07:58:03.815** gives ID5 / version 4; following detail and query confirm it. Bob still receives 404 for ID5; Bob ID6 remains version 1 and unchanged description. |
| Session cleanup | Both continuation finally-logouts return 200. Across the two scripts there are four successful logout records and no recorded cleanup errors. |

The restore status is consistent with the existing source boundary: `characters_endpoint.py` routes `ConflictError` through `map_db_error_to_http`, whose documented mapping is 409. The absent-ID control supports the same not-found treatment rather than a foreign-object existence distinction. No product change was made to accommodate the script expectation.

## Actual browser account visibility

| Capture | Identity and real query | Settled UI |
| --- | --- | --- |
| Alice Characters | Alice ID2; query **200 at 07:58:40.385**, IDs **5v4 and 4v1** | The private same-name character has only the Alice violet-compass description. Bob’s description is absent. |
| Bob Characters | Bob ID3; query **200 at 08:01:05.174**, IDs **6v1 and 3v1** | The private same-name character has only the Bob amber-map description. Alice’s description is absent. |
| Bob subsequent reload capture | Bob ID3; query **200 at 08:02:19.682**, same IDs **6v1 and 3v1** | Settled capture at **08:02:48.242** still shows only Bob’s description. |

The query responses distinguish the private characters by IDs and descriptions; identical labels are not treated as proof of shared objects. The additional IDs are each account’s existing Helpful AI Assistant. The supplied open receipts show ordinary navigation to `/characters`. The subsequent reload-labelled receipts prove a fresh query and settled account view; a separate reload command receipt was not among the 12 allowed inputs, so this audit does not independently certify that particular browser action.

## Criteria and limits

- AC1’s targeted native read/mutation boundaries pass: foreign list/query/detail/search exclusion; rejected foreign update/delete/restore; owned create/update/delete/restore; preserved ownership and version transitions.
- AC2’s wider shared-PostgreSQL/SQLite, immutable snapshot, transaction and child/batch/setup cases remain supported by the already recorded **242 + 5** independent automated passes. They were not rerun or promoted into native claims here.
- AC3’s remaining targeted native account-isolation gate is satisfied alongside the recorded source/security review. This is not the complete product, sharing, role or tenant matrix. It also does not close separate UAT193 fresh-chat greeting-save work by implication.

The original exposure and this accepted runtime use the configured privileged/superuser/BYPASSRLS service role. This run tests the repaired application boundary under that role. It does **not** demonstrate a prior ordinary-role RLS leak or promise arbitrary raw-SQL isolation for bypass roles. Existing ordinary restricted-role policy controls are separate automated evidence.

Parent handoff attributes the running backend to `47e23` / PID56113. These API/browser receipts establish actual behavior, identity and timestamps, but are not an independent process-start source attestation. Current restore-error source inspection is recorded separately and does not substitute for that runtime attribution.

## Evidence integrity

`input-manifest.json` hashes 12 exact allowed scripts/results/UI receipts plus the official CLI task snapshot, without normalization or modifying originals. `audit_inputs.py` reads those records only, retains the initial failure, verifies identities and cleanup, compares restore error structures, checks versions and confirms the three UI account views; all audit assertions pass. `verified-facts.json` contains reduced evidence.

No browser secrets, login-private logs, credential profiles or private helper files were read during this audit. The native scripts’ credential-loading code was inspected as source only. No source, test, task, git, browser or runtime changes were made. This private audit packet is the only write.
