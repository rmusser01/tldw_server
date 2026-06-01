# StudySuggestions

StudySuggestions builds review suggestions from quiz and flashcard activity. The package adapts quiz and flashcard records into topic evidence, normalizes topic labels, builds stored snapshots, exposes actions against suggestions, and queues refresh jobs when study content changes.

## Start Here

- `snapshot_service.py` builds, stores, serializes, and refreshes suggestion snapshots.
- `topic_pipeline.py` and `topic_aliases.py` normalize topic labels and derive suggestion topics.
- `quiz_adapter.py` and `flashcard_adapter.py` extract evidence from quiz and flashcard domains.
- Related API surface: `app/api/v1/endpoints/study_suggestions.py`.
- Related tests: `tests/StudySuggestions/`.

## Responsibilities

- Extract suggestion evidence from quizzes, quiz attempts, flashcards, and flashcard review sessions.
- Normalize, alias, and deduplicate topic labels before they become suggestion keys.
- Build snapshots of study suggestions with evidence, strengths, weaknesses, and timestamps.
- Persist snapshots and expose actions such as refresh and dismissal through endpoint flows.
- Queue background refresh jobs from flashcard and quiz update paths.
- Serialize job status and snapshot responses for the study suggestions API.

## Module Map

- `snapshot_service.py` - snapshot creation, storage, refresh, serialization, and job status helpers.
- `topic_pipeline.py` - topic derivation pipeline.
- `topic_aliases.py` - topic label cleanup, aliasing, and deduplication.
- `quiz_adapter.py` - quiz evidence adapter.
- `flashcard_adapter.py` - flashcard evidence adapter.
- `actions.py` - suggestion action helpers.
- `jobs.py` - refresh job payload and enqueue helpers.
- `types.py` - shared dataclasses and types.

## How It Connects

- `app/api/v1/endpoints/study_suggestions.py` exposes snapshots, refresh, and suggestion actions.
- `app/api/v1/endpoints/flashcards.py` and `app/api/v1/endpoints/quizzes.py` enqueue suggestion refresh work after study activity changes.
- ChaChaNotes provides quiz, flashcard, review-session, and snapshot storage.
- StudyPacks can create flashcards that later feed suggestion snapshots.

## Extension Points

- For a new evidence source, add an adapter beside `quiz_adapter.py` and `flashcard_adapter.py`, then wire it into `topic_pipeline.py`.
- For topic normalization changes, update `topic_aliases.py` and add focused tests in `tests/StudySuggestions/test_topic_pipeline.py`.
- For new actions, inspect `actions.py`, `snapshot_service.py`, and the study suggestions endpoint tests.
- For refresh scheduling changes, update `jobs.py` and `tests/StudySuggestions/test_study_suggestions_jobs_worker.py`.

## Testing

- `tests/StudySuggestions/test_study_suggestion_adapters.py`
- `tests/StudySuggestions/test_topic_pipeline.py`
- `tests/StudySuggestions/test_study_suggestion_storage.py`
- `tests/StudySuggestions/test_study_suggestion_schemas.py`
- `tests/StudySuggestions/test_study_suggestions_endpoints_api.py`
- `tests/StudySuggestions/test_study_suggestions_jobs_worker.py`
- `tests/StudySuggestions/test_flashcard_review_sessions.py`

## Gotchas

- Snapshot quality depends on stable quiz and flashcard anchors; adapter changes should preserve enough evidence for the UI and tests.
- Topic aliasing happens before deduplication, so alias rules can merge suggestions that originally came from different sources.
- Refresh jobs are often triggered from adjacent study modules, so enqueue failures should be handled without breaking the original quiz or flashcard operation.
