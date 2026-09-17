# UAT219 / TASK13260.157 — StudyPack response timestamps

## Result and scope

Frozen two-file unit: `schemas/study_packs.py` and new `tests/StudyPacks/test_study_pack_response_timestamps.py`. The only production behavior change is a datetime-only before-validator on **StudyPackSummaryResponse.created_at/last_modified**, using the established API `isoformat()` convention. A datetime import and import-block blank-line normalization accompany it. No persistence, worker, Jobs, endpoint, ownership or scheduler logic changed. All other schema classes are AST-identical to the recorded baseline.

Causal RED: **5 failures /14 controls passed /0 skipped**,11.15s. Actual PG pack detail and completed-job reads returned500, while SQLite reads passed. UTC/offset/naive datetime schema cases supplied the other3failures. Minimal GREEN: **80 passed /0 skipped**,37.66s, across the new19 and four adjacent files. Following a lint-only import-block whitespace correction, the final19 permanent cases passed again on frozen bytes: **19 passed /0 skipped**,11.21s. Four warnings were reported; the configured pytest output does not expand their details.

Final owned manifest SHA: `9a08c43e57eb59f14370a825acbaf3561e55cf3df743101397dd6301f285f56d`.

## Root cause and contract

ChaCha's real PostgreSQL study_packs table stores created_at/last_modified as TIMESTAMPTZ. Its row conversion deserializes only JSON fields, leaving Python datetime values. The completed-job path `_study_pack_from_job_result → _serialize_study_pack → StudyPackSummaryResponse.model_validate` rejects those values because the response contract is Optional[str]. Direct pack detail shares the serializer. Existing endpoint tests seeded SQLite-only content and therefore saw strings.

The validator converts only actual datetime values. Existing strings (including their exact spelling), null, timezone offsets and naive datetimes retain the established convention. Integers, objects, lists and date-only objects still fail normal Pydantic string validation. All non-time fields and row versions/source/deck/owner identities remain unchanged. The fix does not weaken or redesign the API contract.

## Tests and isolation

The new19 cases cover:

- Official disposable PG and normal SQLite content: actual persisted pack/deck plus real explicitly SQLite-backed JobManager creation/acquisition/completion, then actual FastAPI detail and completed-job responses. Assertions compare full stored pack and job before/after reads and the full serialized response.
- Both backends: foreign job remains404, missing detail remains404, completed job without a result remains200 with null study_pack. Existing behavior is preserved.
- Schema UTC, non-UTC offset, naive datetime, exact SQLite string and null; non-time fields unchanged and output revalidation idempotent.
- Four unrelated invalid input types are rejected for both timestamp fields.

The new fixture specifies `JobManager(..., backend="sqlite")` explicitly. Combined adjacent verification uses the existing `run-pg-tests-explicit-jobs.mjs` helper, which retains required official PG configuration but removes global JOBS_DB_URL so older SQLite Jobs fixtures are not silently redirected. Existing adjacent quarantined-job tests select their PostgreSQL backend explicitly and remain included. Authentication is supplied through established dependency overrides; these tests are real route/serializer/storage controls, not native authentication acceptance or provider calls.

## Commands and receipts

Original actual RED used the standard required-PG runner:

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat219-causal-red node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/StudyPacks/test_study_pack_response_timestamps.py -q --tb=short
```

Combined80 (use a unique evidence label for independent reruns):

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat219-green node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs \
 tldw_Server_API/tests/StudyPacks/test_study_pack_response_timestamps.py \
 tldw_Server_API/tests/StudyPacks/test_study_pack_schemas.py \
 tldw_Server_API/tests/StudyPacks/test_study_pack_endpoints_api.py \
 tldw_Server_API/tests/Flashcards/test_study_response_timestamp_contract.py \
 tldw_Server_API/tests/Flashcards/test_review_response_timestamp_contract.py -q --tb=short
```

Final19 repeats the first file with the explicit-Jobs helper and label `uat219-final-focused-green`. Every redacted log and exact runner command receipt is retained. `causal-red-test.py` retains original RED test bytes; formatting changed afterward without assertion changes. `baseline-study_packs.py` retains original schema bytes. `owned.patch`, `owned-manifest.json` and `review-snapshot/` bind the exact candidate.

## Static checks

Ruff:0 findings on both final files (`ruff-final.json`). Initial import-block formatting finding remains in `ruff.json`; corrected by the project's formatter. Bandit production and tests:0 findings/0 parse errors; test B101 excluded for pytest assertions. AST parse2files, stable final hashes and all-other-schema-class equality in `verification.json`. Scoped diff check is clean.

## Separately proven sibling, excluded from219 production

Parent authorized one bounded official-fixture citation probe during verification. It produced **1PG failure /1SQLite pass /0 skips**,3.14s, on the actual GET `/flashcards/{card}/assistant` route with no study-pack membership. FlashcardCitationResponse rejects datetime at citations[0].created_at/last_modified and primary_citation.created_at/last_modified (four errors). FlashcardProvenanceStore copies those timestamps from the real DB unchanged. The private probe and compact receipts/log are retained here; the parent was notified to associate a separate issue before any sibling production edit. This report does not claim citation responses are fixed by219.

## Native and ownership limits

Original native job5 remains completed and preserved by the parent; this agent did not read/mutate native DB data, regenerate a job, operate a browser or restart a service. Same-job5 native readback and independent review remain parent-owned. No task/tracker/git mutations were made. This is a response serialization repair, separate from216/218 storage initialization and trigger repairs.
