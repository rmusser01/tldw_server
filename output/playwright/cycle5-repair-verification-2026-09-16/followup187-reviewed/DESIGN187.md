# UAT187: scheduled-review response timestamp boundary

TASK13260.124, 2026-09-17 UTC. Parent released the minimal production change after the permanent RED checkpoint. Owned future production scope: `tldw_Server_API/app/api/v1/schemas/flashcards.py`, only `FlashcardReviewResponse`; owned regression: `tests/Flashcards/test_review_response_timestamp_contract.py`. Source013 owns assistant/core and Flashcards/Quizzes endpoints; no overlap is planned.

## Evidence and cause

Native POST `/api/v1/flashcards/review` at00:51:18 returns500 after scheduling is persisted. The retained redacted diagnostic identifies exactly three `string_type` response errors: PostgreSQL `datetime` values in `due_at`, `last_reviewed_at`, `last_modified`. This is a separate schema from UAT179's completed-session/thread/message models. The parent will not rate the original Citrine card again to reproduce it.

`endpoints/flashcards.py:2015` returns `db.review_flashcard` directly. That DB method returns from inside its managed transaction; context exit commits before FastAPI validates the response. `FlashcardReviewResponse` declares the three fields `Optional[str]` without the existing before-validator. Three neighboring established patterns are Deck, Flashcard and FlashcardReviewSessionSummary: only `datetime` becomes `.isoformat()`, while strings/null retain their current semantics and unrelated types still fail validation.

## Stages

1. **RED (complete):** actual FastAPI router with only DB dependency injection; real SQLite and official required-PostgreSQL fixture. Review a new disposable fixture card once in due/cram contexts. Inspect committed version, one review row, one matching active session and unchanged source/content before asserting HTTP200. Do not retry a500. Missing-card/invalid-rating no-write controls and exact UTC/offset/naive/string/null/unrelated-type schema controls are retained. Result:5 expected failures,10 controls pass,0 skips; see RED187.md.
2. **GREEN (complete):** following parent release, added one before-validator for only the three fields, matching the neighboring pattern. No scheduler, SQL, transaction, auth or endpoint change.
3. **Verify/review (author checks complete; independent review pending):** the exact route/schema suite and both adjacent timestamp suites pass71/0skips; required PG means zero skipped PG cases. Scoped Ruff, compile/static security checks and independent review. Parent owns native scheduled review/session/reload acceptance on an appropriate disposable/pending card, original-state reconciliation and commits.

Limits: route tests inject a real fixture DB and do not certify login/auth flows. No provider, inference, browser or native database mutation is needed. Existing string public schemas remain strings; this is not a timestamp-type or timezone normalization redesign. Null fields remain optional, and non-time scheduling fields must be unchanged by serialization.
