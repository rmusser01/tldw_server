# Independent UAT223 review — CLEAR

TASK13260.161. No remaining finding in the frozen two-file repair. Native same-original-card Deep dive acceptance remains pending after the reviewed restart. No product/test/browser/runtime/task/git changes were made by this review.

## Source and contract

The only changed production AST node is `_build_note_route` in `core/StudyPacks/provenance.py`. It now targets the existing `/notes` page and passes the authoritative normalized note ID as `source_ref_id`. The actual frontend `routes/option-notes.tsx` reads this query value and `NotesManagerPage` passes it to the existing guarded `openSourceNote` flow; the registered Next page is `pages/notes.tsx`. The change does not alter authorization or create a new route.

Missing/empty locators preserve workspace-route classification. Mapping and string locators preserve their existing query representation, with empty values filtered by the existing helper. The authoritative source query is merged last, preventing a locator's `source_ref_id` from replacing the selected note. Existing `urlencode` encodes slash, question mark, ampersand, spaces and Unicode inside the value; there is no new raw path interpolation or open redirect. Other production AST nodes, including media/message builders and citation selection/persistence, are identical to baseline. This preserves locator data, not text-anchor scrolling behavior.

The twelve new cases exercise the real public resolver for two IDs across six locator shapes. Three existing full assistant/target response expectations change only the previously unsupported URL. Existing assertion coverage remains; no assertion was removed. The original causal log records12 route failures /9 deselected against old source; the failures are missing/wrong routing, not a test harness import failure.

## Fresh independent verification

- **68 passed, zero skipped, four warnings, 33.87s**, across provenance, actual citation responses, pack responses and endpoints. Official fixtures require PostgreSQL and also retain SQLite controls. Exact command and redacted output are copied into this directory.
- The explicit-Jobs runner differs from the established runner by exactly `delete env.JOBS_DB_URL;`, preventing inherited Jobs configuration from overriding existing SQLite fixtures. Required PostgreSQL settings remain enabled. No manual DB creation, native DB query, server restart or provider call was performed.
- Independent Ruff:0 findings. Bandit production:0 findings/0 parse errors; tests:0 findings/0 parse errors with existing B101 assertion exemption. Both files compile in memory without bytecode output.
- Exact source/test hashes and author snapshots match before and after testing. The adjacent ChaCha dependency also remained unchanged at `0bbf4442fa4f41d53ff61c3261b480387bcbdc866e1f15785244fbdb6065513c`; pending UAT222 edits are outside this run and review. `ast-review.json` independently confirms the sole production function change.

Source `provenance.py`: `a7f6fbd65cecec7361f76725ac78fd27208ccc2d57f8cec1985af77651f2f745`.
Test `test_provenance.py`: `3526f229d4889378fb0bdc5c34c1b90fe64e1f626fe8756a772357e192042d6e`.

The original actual UI failure involved card `a33c73dc-1f41-472a-b037-f2d6b041b363`, source note `b83dca90-fab0-4c6f-8c0f-6f1e93dfffc8`: click navigated to unsupported `/notes/<id>` and showed404. That failure is retained in the accepted bounded Study Pack package as a separate unresolved223 outcome. This review does not claim the fixed browser click has occurred, nor that unrelated UAT222 pack ownership is repaired.

## Independent command

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat223-sidebar-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs tldw_Server_API/tests/StudyPacks/test_provenance.py tldw_Server_API/tests/StudyPacks/test_citation_response_timestamps.py tldw_Server_API/tests/StudyPacks/test_study_pack_response_timestamps.py tldw_Server_API/tests/StudyPacks/test_study_pack_endpoints_api.py -q --tb=short
```
