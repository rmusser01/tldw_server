# UAT133 — Failed Retry answers an earlier turn

## Native outcome

Single ordinary Cedar conversation 1d4c6033-63eb-4cec-a406-41c579d7e791 had two successful user/assistant pairs. The third user requested exactly `CEDAR RETRY READY.` Explicit unavailable Ollama returned502. Actual Gemma Retry returned200, reused the same canonical user/correlation and retained one UI user plus grouped assistant variants. Its answer instead repeats the first question's coordinator/book-return answer. Canonical reload contains exactly system+three pairs, with the latest user text intact. Request265 correctly contains all prior turns and the current request once. No additional generation was attempted.

Evidence: single259,265,266,267,269. This is a response-content failure while Retry identity/persistence controls pass. Browser input alone does not prove the provider's final input; the source/probe below isolates that boundary.

## Root cause and private reproduction

`chat_service.build_context_and_messages` defaults to configured descending history. It recognizes the already-persisted failed user and trims overlapping client history; the current-turn list then becomes empty. It returns the descending saved history unchanged. The last non-system message supplied to downstream templating is consequently the oldest prior user question, rather than the failed turn being retried.

The private probe executes the actual function with a controlled in-memory DB implementing the existing history fixture, synthetic neutral conversation, two prior pairs and one unanswered user. No network, browser, live database or repository edits. Fresh process removes inherited credentials, uses a private empty configuration/data root and blocks outbound sockets.

- Descending full-history Retry: FAIL, final non-system `Who coordinates Cedar?`.
- Ascending full-history Retry: PASS, final non-system exact current request.
- Descending single-user-message Retry: FAIL with the same wrong final question.
- Ascending single-user-message Retry: PASS.
- All four preserve canonical failed-user identity and perform zero duplicate writes.

Command: activate project venv, then `PYTHONDONTWRITEBYTECODE=1 python /private/tmp/cycle5-controller-retry-order-probe.py`. Result2RED/2GREEN. Script, output and hashes retained under controller evidence. This proves the application context-ordering defect; it does not claim a captured native provider HTTP body.

## Minimal correction and required controls

An explicit accepted failed-turn Retry must place its exact saved user turn last in the model context exactly once, without another database write. Preserve the existing requested history ordering contract for other messages; the current test suite explicitly asserts descending historical payload order. Prefer moving the identified saved retry row out of the historical prefix and into the final current turn, preserving its literal text/images and identity. If it lies outside the selected history window, use the already-validated requested current turn once. Do not repair by changing live history configuration or duplicating a user message.

Permanent tests must cover full-client-history and user-only requests, ASC/DESC and limited/zero history windows, explicit Retry versus an ordinary identical new user message, neutral/persona/tracked contexts, image MIME/bytes, legacy error envelopes, correlation conflicts, answered-tail409 and ownership. Inspect actual final provider-bound message order, not only counts. Existing transport/persistence/image recovery suites and Bandit are required. Targeted native acceptance after freeze should ask a distinct safe new question following prior turns, trigger controlled provider failure, Retry with working model, and verify the new question is answered once with canonical identity retained.
