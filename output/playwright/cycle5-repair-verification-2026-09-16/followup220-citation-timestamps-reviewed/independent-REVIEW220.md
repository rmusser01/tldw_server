# Independent review: UAT220 / TASK13260.158

## Disposition

CLEAR. No blocking correctness or security findings. Independent mandatory PostgreSQL/SQLite verification: **97 passed, 0 skipped, 4 warnings in 47.05s**. No production/test changes were made by this reviewer. Native cited-card acceptance remains parent-owned.

## Frozen source and attribution

Author owned-manifest SHA256: `01d8c4ac5eb3895718064a8a6fd7c5ee78ecd3171bf066261f281a549613e03e`.

| File | SHA256 |
| --- | --- |
| `tldw_Server_API/app/api/v1/schemas/study_packs.py` | `96843579f5f1c87a855dcb7ab4220bd8114c44ab5ce6ae60188c7add0705349a` |
| `tldw_Server_API/tests/StudyPacks/test_citation_response_timestamps.py` | `d10286fb8dc07d3b41267f67bd687735cdf0c5c35b07b2a6f479609f068dd09d` |

Both paths matched the author's frozen snapshots and retained their hashes through independent verification. The baseline schema hash is the independently released UAT219 version: `26b5919233ef56eb2f84211fbff8b911b751039f2ad91876b1e51b40f7866be8`. Removing the one newly added FlashcardCitationResponse validator from the current AST makes the entire module AST identical to that baseline. UAT219 Summary behavior and every other schema class remain unchanged.

## Correctness and controls

The actual assistant endpoint validates citation-list members and primary_citation through the same FlashcardCitationResponse model. FlashcardProvenanceStore preserves database timestamp values. The narrow before-validator converts datetime objects using isoformat(); existing strings/null continue through unchanged and other values remain subject to the existing string validation. It does not parse timestamps, change public field types, rewrite rows, or alter access decisions.

The 17 new cases run the actual FastAPI assistant GET with real persisted note/deck/card/citation data on official PostgreSQL and separate SQLite content. Citation-only and StudyPack-linked cases verify both list and primary timestamps, original source/card/owner/version identity, card/citation row equality, preserved source content and preserved linked pack. The cited-pack case also verifies compatibility with UAT219 nested metadata. Empty citations retain200 with [] and null primary. A distinct actor using a real separate selected-owner DB receives404 while owner card/citations remain unchanged; PostgreSQL uses the shared fixture backend and SQLite separate files.

Schema controls preserve UTC and non-UTC offsets, naive values, exact existing string spelling and null; non-time values and repeated validation remain unchanged. Integer/object/list/date-only values are rejected for both fields. These controls support the unchanged Optional[str] contract rather than a general datetime parser.

Reviewed retained causal evidence: permanent5FAIL/12controls, comprising two actual PostgreSQL route failures and three datetime schema failures. Earlier isolated citation-only probe records four string_type errors at citations[0] and primary_citation timestamps, with actual datetime input; SQLite200. This separates UAT220 from the prior Summary failure. Baseline was inspected, not rerun independently.

## Independent verification

Project virtual environment activated first. Exact command is bound in `test-command.json`:

```sh
TLDW_UAT_EVIDENCE_LABEL=uat220-independent node .tmp/fresh-uat-recovery-20260916/run-pg-tests-explicit-jobs.mjs \
 tldw_Server_API/tests/StudyPacks/test_citation_response_timestamps.py \
 tldw_Server_API/tests/StudyPacks/test_study_pack_response_timestamps.py \
 tldw_Server_API/tests/StudyPacks/test_study_pack_schemas.py \
 tldw_Server_API/tests/StudyPacks/test_study_pack_endpoints_api.py \
 tldw_Server_API/tests/Flashcards/test_study_response_timestamp_contract.py \
 tldw_Server_API/tests/Flashcards/test_review_response_timestamp_contract.py -q --tb=short
```

- **97 PASS / 0 skipped / 4 warnings / 47.05 seconds** across six files; configured output does not expand warning details.
- Official required-PG fixture runner used with globally injected JOBS_DB_URL removed, preserving legacy SQLite Jobs fixture selection. Explicit PostgreSQL Jobs controls remain in the adjacent suite.
- Scoped Ruff on both owned paths: **0 findings**.
- Bandit production and new tests: **0 findings / 0 parse errors**; only B101 excluded for pytest assertions in the test file.
- Compile of both source texts: **PASS**, without writing bytecode.
- Final hash verification: **2/2 unchanged**.

`test-green.log` is copied sanitized output; source-before/after and static receipts are bound by reviewer-manifest.json.

## Limits

Authentication is supplied using existing dependency overrides; this verifies actual route/storage/response behavior, not native login, comprehensive RLS, or cold runtime configuration. Existing assistant-context creation behavior is not changed by the schema fix. No native job/card, provider, browser, configuration, process, task, tracker, staging or commit action occurred. This clears the bounded citation response repair; it does not claim native same-card acceptance or full Study workflow/matrix completion.
