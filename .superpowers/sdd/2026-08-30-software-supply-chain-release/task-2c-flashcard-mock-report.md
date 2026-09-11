# Task 2C — deterministic critical-journey mock responses

Status: DONE_WITH_CONCERNS

## Scope and precedence

Task 2C changes only the committed local mock dispatch, its fixtures and focused tests, plus the Ingest → Search → Chat journey's deterministic local fixture setup. Production parsers, UI assertions, streaming/timeouts, provider code, and the generic default response remain unchanged.

The investigation studied these existing patterns before changing setup:

1. `mock_openai_server/tests/test_live_tier_analysis_responses.py` — priority-30 system-prompt cases.
2. `mock_openai_server/tests/test_server.py` — request-to-response mock API behaviour.
3. `mock_openai_server/tests/test_scenario_failures.py` — ordered scenario selection/failure behaviour.
4. `apps/tldw-frontend/e2e/workflows/media-ingest.spec.ts` — checked-in local Quick Ingest fixture.
5. `apps/tldw-frontend/e2e/workflows/media-review.spec.ts` — authenticated local text upload through `/api/v1/media/add`.
6. `apps/tldw-frontend/e2e/workflows/journeys/ingest-search-chat.spec.ts` — previous external-URL journey.

Dispatch is priority-descending: tier analysis is priority 30, the exact critical contracts are priority 20, source summary remains priority 10, and the generic default is unchanged. Tests retain source-summary and default negative controls, so no exact contract shadows an unrelated prompt.

## Separate root traces and RED/GREEN evidence

### Flashcards

The actual generation request ends with `Generate flashcards from:` followed by the note text. Claim verification is a separate strict-JSON judge call with the full evidence block. Earlier focused RED captured those two stages: the generation request selected `chat/default.json`; the verifier's full source selected `chat/source-summary.json` and surfaced `422 claim_verification_failed`.

The fixture now provides structured flashcard JSON and a separately anchored strict verifier response. The unchanged production parser is exercised by the focused adapter tests below.

### Ingest → Search → Chat

The RAG-final provider call is distinct from the user's OpenAI text-part chat message. The actual RAG format is:

```text
You are a helpful AI assistant. Use the following context to answer the user's question.
If the context doesn't contain relevant information, say so clearly.

Context:
[Source 1: <title> (<source>)]
<ingested document>

Question: Playwright

Answer:
```

The direct chat request instead carries `What is Playwright? Use the ingested content to answer.` in a `type: text` content part. The focused RAG RED first selected `chat/source-summary.json` because the existing broad source-summary matcher matched `[Source ...]`. The second RED reproduced the real local-file shape (the filename prefix before the sentence), which also fell through to source-summary. Each exact, start/end anchored RAG contract then went GREEN without broadening the fallback matcher.

The quick-ingest local file path was hypothesis H1. It reached nonempty context once, but a repeat on the same graph produced `No relevant context found`; it is therefore rejected as non-deterministic. H2 used the existing authenticated `/api/v1/media/add` text-upload pattern with the checked-in `playwright-grounded.txt` fixture. From a fresh task-local user DB it returned `Success`, `db_id: 1`, persisted the exact sentence plus one sentence chunk, and `/api/v1/rag/search` returned that document. The mock captured the corresponding nonempty RAG-final prompt. H2 is the committed setup; H3 was unnecessary.

The dispatcher now concatenates only dictionary parts whose `type` is `text` and whose `text` value is a string. A negative control supplies an object as `text` and proves it selects `chat/default.json`, rather than being coerced into the source-summary scenario.

The focused RED/GREEN loop used:

```bash
source ../../.venv/bin/activate && python -m pytest \
  mock_openai_server/tests/test_critical_e2e_response_contracts.py -q
```

Results, in order: `1 failed, 5 passed` for the concrete RAG document before its exact matcher; `6 passed` after the narrow matcher; `1 failed, 6 passed` after adding the captured filename-prefixed local-file shape; and then all seven dispatch tests passed after the second narrow matcher. Each RED failed by selecting `chat/source-summary.json`, the expected pre-fix dispatch defect.

## Focused GREEN command

```bash
source ../../.venv/bin/activate && python -m pytest \
  mock_openai_server/tests/test_critical_e2e_response_contracts.py \
  tldw_Server_API/tests/Workflows/adapters/test_content_adapters.py \
  -k 'critical_e2e_response_contracts or flashcard_generate' -q
```

Result: `36 passed, 142 deselected, 4 warnings in 5.87s`.

This covers the production flashcard parser, independently anchored flashcard generation and verifier dispatch, both nonempty Playwright RAG prompt shapes, the direct text-part response, malformed text-part handling, and the unchanged source/default fallbacks.

## Fix round 2: shared-database isolation

The earlier fresh-DB live proof is superseded for final package verification.
The final paired command ran against the existing shared Task2C user DB, which
already contained the failed run's fixed `Generated Flashcards` deck.  The
Notes → Flashcards journey now creates a unique `task2c-flashcards-*` deck through
the supported deck API, selects it through the unchanged Transfer UI, and removes
it by exact id/version in `finally`.  That prevents the pre-existing fixed deck
from being selected while retaining its unchanged generation, parser, and UI
assertions.  Post-run API inspection retained only `Generated Flashcards`.

The Ingest → Search → Chat journey's existing generated
`task2c-playwright-*` fixture/title/query is likewise isolated from a freshly
seeded `preexisting-playwright-*` document.  Its direct real `top_k: 1` RAG
request verifies the exact fixture is selected before the UI chat; both documents
are deleted in `finally`.  The captured final streamed provider body contains the
same title/query token and the concrete Playwright sentence.  No Task2C media
records remained after the run.

The exact RED/GREEN, shared-DB paired journey, request-body, gate, and cleanup
outputs are committed in
`task-2c-flashcard-mock-evidence.md` beside this report.  Isolation hypothesis
H1 succeeded; no additional hypothesis or stop condition was needed.

Fix round 3 adds an actual save-request `deck_id` assertion, because the earlier
successful save/global count could not prove destination ownership.  It also adds
a real mismatched-token fallback control.  The code/test commit is
`2517e29599a1d2d2120ff55d0b7ad76635865b1c`; before this report was written,
both `git diff --check 73c6998901..2517e29599a1d2d2120ff55d0b7ad76635865b1c`
and `git show --check 2517e29599a1d2d2120ff55d0b7ad76635865b1c` exited 0.
The evidence artifact records those immutable outputs without claiming this
later documentation commit's hash.

After that code commit, the shared-DB Notes → Flashcards live journey executed
the new captured-save-request `deck_id` assertion and passed in `11.4s`; cleanup
left only the pre-existing fixed deck.  A paired diagnostic rerun still captured
the nonempty final RAG context but failed the separate direct-chat semantic
assertion with the generic default response.  This is recorded as one diagnostic
verification failure; Task2C made no additional matcher or product change.

The final H3 diagnosis found the graph, not the committed matcher, was using the
main checkout's `mock_openai_server` through an absolute `PYTHONPATH`.  Pointing
that local graph at this worktree's mock package made the exact captured
multimodal stream select `chatcmpl-critical-playwright`.  The final shared-DB
pair then passed (`2 passed (27.4s)`) with the existing deck/media cleanup
controls intact.  H1/H2 cache/import experiments were reverted; no mock
implementation change was needed.  The evidence artifact captures module paths,
the direct HTTP proof, gates, and cleanup result.

## Live proof

The Task2A-owned local graph used Redis `62457`, mock provider `18091`, API `62458`, and WebUI `62459`.  The final proof used the existing shared `/tmp/tldw-task2c-h2-userdb` user database (not a fresh DB) and a worktree-local mock `PYTHONPATH`; no external or protected scope was contacted.

```bash
TLDW_WEB_AUTOSTART=false TLDW_WEB_URL=http://127.0.0.1:62459 \
TLDW_SERVER_URL=http://127.0.0.1:62458 TLDW_E2E_SERVER_URL=http://127.0.0.1:62458 \
TLDW_API_KEY=test-api-key-for-e2e-testing-12345 \
TLDW_E2E_API_KEY=test-api-key-for-e2e-testing-12345 \
TLDW_MOCK_OPENAI_URL=http://127.0.0.1:18091/v1 \
NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=advanced NEXT_PUBLIC_API_URL=http://127.0.0.1:62458 \
bunx playwright test e2e/workflows/journeys/notes-flashcards.spec.ts \
  e2e/workflows/journeys/ingest-search-chat.spec.ts \
  --project=journeys --workers=1 --reporter=line --trace=on
```

The final shared-DB replacement command is recorded in the evidence artifact;
its result is `2 passed (27.4s)`.

The final mock request body for the Ingest → Search → Chat RAG stream contained `[Source 1: playwright-grounded (media_db)]` and `Playwright is an open-source framework for reliable end-to-end browser testing.` The live response used that same concrete semantic. The flashcard journey also completed its structured generation and claim-verification path.

## Final gates

```bash
source ../../.venv/bin/activate && python -m bandit -r \
  mock_openai_server/mock_openai/config.py \
  mock_openai_server/tests/test_critical_e2e_response_contracts.py \
  -s B101 -f json -o /tmp/bandit_task2c_fix.json
```

Result: `results=0`.

```bash
cd apps/tldw-frontend && bun run typecheck
```

Result: Task2C has no remaining type errors. The command remains red on four pre-existing errors in `apps/packages/ui/src/components/Option/Documentation/DocumentationPage.tsx` and four pre-existing dirty `scripts/__tests__/skills-certification-*.test.ts` errors; those paths are outside Task2C.

The committed-range whitespace check is recorded after the Task2C fix commit:

```bash
git diff --check 4cdf7aa6b2..HEAD
```
