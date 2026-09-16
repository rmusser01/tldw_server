# UAT179 / TASK-13260.116

Reuse the existing Deck/Flashcard response-model datetime-only before-validator pattern. Localize it to FlashcardReviewSessionSummary (started_at,last_activity_at,completed_at), StudyAssistantThreadSummary (last_message_at,created_at,last_modified), and StudyAssistantMessage(created_at) only after a populated actual-router RED demonstrates that additional field. Keep Optional[str] public contracts, unchanged strings/nulls, offset/naive semantics, and invalid-type rejection. Do not normalize the database, alter routes, or refactor unrelated models.

Stages: (1) actual official-PG/SQLite router and schema RED; (2) minimal validators; (3) same GREEN plus existing timestamp/history/assistant regressions, scoped Ruff/Bandit, frozen snapshot and independent parent review. Native acceptance remains parent-owned.
