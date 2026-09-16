# UAT179 / TASK-13260.116

## Result

Frozen repair adds the existing datetime-only, before-validation timestamp conversion to three public response models:

- FlashcardReviewSessionSummary: started_at, last_activity_at, completed_at.
- StudyAssistantThreadSummary: last_message_at, created_at, last_modified.
- StudyAssistantMessage: created_at.

Each retains Optional[str]. Datetimes use their own isoformat(), keeping UTC offsets or lack of timezone exactly; existing strings/nulls and rejection of unrelated values remain unchanged. No database timestamp changes, generic stringify helper, route changes, or unrelated model refactoring.

## Evidence

Required official PostgreSQL/SQLite regression run `uat179-red`: **14 failures /14 passes /0 skips**. Five actual PostgreSQL HTTP cases failed (active/completed history, end session, empty/populated assistant), plus nine direct datetime-contract cases. The populated assistant failure explicitly includes response.messages[0].created_at, as required before adding that model's validator. SQLite HTTP controls and string/null/invalid-type controls passed.

Same suite `uat179-green`: **28 passed /0 skipped**, 11.97 seconds. Real router/database tests exercise writes and response validation, preserving IDs, status, thread count, message content, structured payload, and context snapshot. Only downstream suggestion-job enqueue is replaced by a test dependency; no provider is contacted.

Existing assistant/session/analytics regression selection: **32 passed /155 deselected /0 skipped**, 24.42 seconds. This covers the existing study-assistant DB tests and relevant endpoint integration tests with both UAT177 and UAT179 applied.

Ruff touched source/test: 0 findings; source baseline 0. Bandit touched schema: 0 findings/0 scan errors. `git diff --check` clean.

## Commands

Activate `.venv/bin/activate` first.

- RED/GREEN: `TLDW_UAT_EVIDENCE_LABEL=uat179-red` (or green), then `node .tmp/fresh-uat-recovery-20260916/run-pg-tests.mjs tldw_Server_API/tests/Flashcards/test_study_response_timestamp_contract.py -q --tb=short`.
- Existing controls: `python -m pytest tldw_Server_API/tests/Flashcards/test_study_assistant_db.py tldw_Server_API/tests/Flashcards/test_flashcards_endpoint_integration.py -k 'assistant or review_session or analytics' -q --tb=short`.
- Ruff: `python -m ruff check tldw_Server_API/app/api/v1/schemas/flashcards.py tldw_Server_API/tests/Flashcards/test_study_response_timestamp_contract.py --output-format json`.
- Bandit: `python -m bandit tldw_Server_API/app/api/v1/schemas/flashcards.py -f json -o .tmp/uat179-repair-20260916/bandit.json`.

The PostgreSQL helper uses official per-test fixture databases and TLDW_TEST_POSTGRES_REQUIRED=1 with the existing owned cluster. No AuthNZ test_db_pool, live requests, service starts, inference, or live-data mutations.

## Freeze / limits

Exact copies, hashes, and new-file diff are in review-snapshot/, owned-manifest.json, and owned.patch. Source SHA256: 802232415b9b8314e85e2a26f59300692f151921f7b6bc40fceadb5872394315. Test SHA256: 92e886ca7cedb6d374edeccdad374fe2a9ec31bb7b0245d5383d6a461edf9b25.

Native PostgreSQL acceptance and independent review remain parent-owned/pending. These fixtures prove response contracts, not real-model quality. UAT177 separately owns the analytics SQL/read lifecycle repair; UAT180 separately owns the subsequently discovered deletion/scheduling positional-row defect. No git/commit/tracker edits performed.
