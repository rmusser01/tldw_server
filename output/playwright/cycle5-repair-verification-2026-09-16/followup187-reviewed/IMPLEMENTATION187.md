# UAT187 implementation and frozen review handoff

TASK13260.124. Parent released production after [permanent RED](RED187.md). [Design](DESIGN187.md), [owned diff](owned.patch), [exact file/snapshot hashes](owned-manifest.json).

## Change and boundary

`FlashcardReviewResponse` now applies the existing datetime-only `.isoformat()` before-validator to `due_at`, `last_reviewed_at`, and `last_modified`. Existing strings/nulls pass through; invalid unrelated types retain validation errors. No scheduling, SQL, transaction, auth or endpoint change. PostgreSQL was committing the rating before FastAPI rejected its timestamp types; this repair serializes that already-committed state. The original native card was not rated again by this agent.

## Permanent evidence

RED:5 failed,10 passed,0 skipped. Actual PostgreSQL due/cram requests committed exactly one review, an active session count1 and version+1 before returning500; SQLite and no-write404/422 controls passed. UTC/offset/naive datetime schema inputs supplied the other3 failures. See RED187.md for the original immutable log and source hashes.

GREEN command:

```sh
source .venv/bin/activate
TLDW_UAT_EVIDENCE_LABEL=uat187-green node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/Flashcards/test_review_response_timestamp_contract.py tldw_Server_API/tests/Flashcards/test_study_response_timestamp_contract.py tldw_Server_API/tests/Flashcards/test_flashcards_timestamp_contract.py -q --tb=short
```

Result: **71 passed,0 skipped,4 warnings**,69.31s. Includes all15 new controls,28 existing Study response timestamp controls and28 existing Flashcard timestamp controls. [Redacted log](../fresh-uat-recovery-20260916/uat187-green.redacted.log); official function-scoped PostgreSQL and real SQLite fixtures only. Helper required network escalation to the existing test cluster. No native database, runtime, browser or provider requests. Route dependencies inject fixture DBs; these tests do not claim login-flow acceptance.

Static checks:

```sh
source .venv/bin/activate
python -m ruff check tldw_Server_API/app/api/v1/schemas/flashcards.py tldw_Server_API/tests/Flashcards/test_review_response_timestamp_contract.py
python -m ruff format --check tldw_Server_API/tests/Flashcards/test_review_response_timestamp_contract.py
python -m py_compile tldw_Server_API/app/api/v1/schemas/flashcards.py tldw_Server_API/tests/Flashcards/test_review_response_timestamp_contract.py
python -m bandit tldw_Server_API/app/api/v1/schemas/flashcards.py -f json -o .tmp/uat187-repair-20260917/bandit-production.json
python -m bandit tldw_Server_API/tests/Flashcards/test_review_response_timestamp_contract.py -s B101 -f json -o .tmp/uat187-repair-20260917/bandit-test.json
```

All exit0. Both Bandit reports have zero findings/errors; B101 is excluded only for pytest assertions. Existing schema formatting was preserved; no whole-file formatting change. Owned diff whitespace passes.

Independent review and native scheduled-review/session/reload acceptance remain parent-owned. Production and test files are frozen; other agents' endpoint/core files are outside this diff.
