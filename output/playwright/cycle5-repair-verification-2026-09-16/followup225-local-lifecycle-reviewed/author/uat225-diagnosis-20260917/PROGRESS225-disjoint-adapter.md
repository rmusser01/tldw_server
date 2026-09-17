# TASK13260.164 — disjoint implementation checkpoint

This is a partial checkpoint, not UAT225 closure or an integrated source release. Source013's Stage A owns the shared store/API/service lease. The new local adapter is not imported by runtime/factory.

## Owned files

- `tldw_Server_API/app/core/Notes_Graph/suggestion_local_mutations.py`
- `tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_legacy_mutations.py`
- `tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_legacy_lifecycle.py`
- `tldw_Server_API/tests/Notes_Graph/integration/test_suggestion_legacy_transition.py`

The concrete adapter calls only existing `NotesLinkStore.upsert`, `NotesOrganizationSyncStore.apply_resource` and `apply_relationship`. Existing callbacks guard and finalize inside the product transaction; an outer guard runs before product locks. Object identities come from the future shared decision caller, and mismatches are rejected. Keyword creation is before-only; only membership finalizes tag acceptance. Existing timestamp normalization is required by the link payload contract.

## Evidence so far

All commands activate `.venv` and use the official required-PG fixture runner `.tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs`, with a unique `TLDW_UAT_EVIDENCE_LABEL`. No native profile, browser, provider or model calls occur.

| Label | Result | Meaning |
|---|---|---|
| `uat225-local-adapter-red` | 10 FAIL, 0 skip | 2 actual fresh factory results are None; 8 new adapter controls reach expected missing module. |
| `uat225-local-adapter-first-green` | 4 FAIL / 4 PASS, 2 deselected | Candidate used raw Notes millisecond-Z timestamp; actual link validator requires canonical UTC ISO. Receipt retained. |
| `uat225-local-adapter-green` | 8 PASS, 2 deselected | Existing normalizer fixes the candidate. Factory intentionally unwired. |
| `uat225-local-adapter-controls` | 16 PASS, 2 deselected | Product/finalizer rollback, outer caller rollback, keyword-only partial state, membership interruption, identity and normalized collision controls. |
| `uat225-local-adapter-owner-controls` | 2 PASS, 18 deselected | Same normalized label under two shared PG owners; foreign membership denied, owned membership accepted, foreign row unchanged. SQLite counterpart uses separate owner files. |
| `uat225-enrollment-entrypoints-red` | 6 FAIL, 0 skip | Actual profile creation, Personal Context default creation, and supplied-default binding all leave authority empty, on both backends. These are assertions on the missing canonical fence, not fixture failures. |
| `uat225-local-admission-red` | 2 FAIL / 2 PASS, 0 skip | Actual fresh local store→Jobs admission raises scope error; registered canonical controls create one content-free idempotent Job and replay. |
| `uat225-local-adapter-final-disjoint` | 18 PASS, 2 deselected, 0 skips | Exact final adapter suite including owner-label control; 23.07 seconds. The two held factory controls are explicitly selected out only for this adapter-only run. |

The final combined adapter replay has its own retained command/log; it is not a sum of earlier individual passes.

The registered dataset in adapter controls is deliberate isolation of the mutation boundary, not proof of fresh local authorization. New factory/admission/enrollment controls remain RED until the shared implementation is released. They are not disabled or converted to xfail.

Ruff passes for all four new files. Bandit production and tests (B101 excluded only for test assertions) report zero findings/errors. Final hashes and compile result are in the disjoint checkpoint manifest. No broad adjacent suite or native acceptance is claimed yet.

## Next work

After shared Stage A review/release, implement the approved exact local scope and actual factory path, then the common owner-validated retirement boundary for both profile entrypoints. Add barrier controls for late enqueue, acceptance, keyword/membership and publication before changing those shared paths. Preserve the merged-tag obligation from the full design; no keyword/schema repair is authorized implicitly.
