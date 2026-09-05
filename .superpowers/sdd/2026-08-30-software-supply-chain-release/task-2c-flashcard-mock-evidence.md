# Task 2C fix-round-2 captured evidence

Status: DONE_WITH_CONCERNS

This artifact records the command outputs and request bodies used to validate the
two independent Task 2C contracts.  It is intentionally kept with the package,
rather than relying on a host-local log.

## Dispatch RED then GREEN

Before the title-and-query-bound Playwright matcher, the focused RED selected the
existing fallback for the new unique fixture boundary:

```text
$ source ../../.venv/bin/activate && python -m pytest \
  mock_openai_server/tests/test_critical_e2e_response_contracts.py \
  -k unique_playwright_fixture_context -q
E       AssertionError: expected chat/playwright-grounded.json,
E       got chat/source-summary.json
1 failed, 7 deselected
```

After adding the exact media-db, context, and stream shapes, plus the same-token
backreferences and fallback controls:

```text
$ source ../../.venv/bin/activate && python -m pytest \
  mock_openai_server/tests/test_critical_e2e_response_contracts.py \
  -k 'unique_playwright_fixture_context or unique_playwright_stream_context' -q
2 passed, 7 deselected
```

The completed focused parser/dispatch run was:

```text
$ source ../../.venv/bin/activate && python -m pytest \
  mock_openai_server/tests/test_critical_e2e_response_contracts.py \
  tldw_Server_API/tests/Workflows/adapters/test_content_adapters.py \
  -k 'critical_e2e_response_contracts or flashcard_generate' -q
collected 180 items / 142 deselected / 38 selected
38 passed, 142 deselected, 6 warnings in 5.52s
```

The focused tests retain the `chat/source-summary.json` and `chat/default.json`
negative controls.  The content-list negative control also proves that malformed
or non-string `text` parts are ignored rather than coerced.

## Shared-database paired journey

The existing shared DB contained the old fixed-name deck before the run:

```text
$ curl .../api/v1/flashcards/decks?limit=1000&include_deleted=false
[{"id":1,"name":"Generated Flashcards", ...}]
```

The Notes -> Flashcards journey creates and selects a per-run
`task2c-flashcards-*` deck using the supported deck endpoint, then deletes that
exact deck in `finally`.  The Ingest -> Search -> Chat journey also uses its own
generated `task2c-playwright-*` media title/query token and deletes both its
fixture and its deliberately unrelated `preexisting-playwright-*` control in
`finally`.

```text
$ TLDW_WEB_AUTOSTART=false TLDW_WEB_URL=http://127.0.0.1:62459 \
  TLDW_SERVER_URL=http://127.0.0.1:62458 TLDW_E2E_SERVER_URL=http://127.0.0.1:62458 \
  TLDW_API_KEY=test-api-key-for-e2e-testing-12345 \
  TLDW_E2E_API_KEY=test-api-key-for-e2e-testing-12345 \
  TLDW_MOCK_OPENAI_URL=http://127.0.0.1:18091/v1 \
  NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=advanced NEXT_PUBLIC_API_URL=http://127.0.0.1:62458 \
  bunx playwright test e2e/workflows/journeys/notes-flashcards.spec.ts \
    e2e/workflows/journeys/ingest-search-chat.spec.ts \
    --project=journeys --workers=1 --reporter=list --trace=on
Running 2 tests using 1 worker
  ✓  1 ... Ingest -> Search -> Chat journey ... (15.8s)
  ✓  2 ... Notes -> Flashcards journey ... (10.4s)
2 passed (27.4s)
```

The same run's deterministic mock captured the final nonempty streamed RAG
provider request.  The title and query have the same unique token, and the
concrete document content is present:

```text
Context:
[Source 1: task2c-playwright-1788195747338-uv8wo5 (media_db)]
Playwright is an open-source framework for reliable end-to-end browser testing.

Question: Playwright task2c-playwright-1788195747338-uv8wo5

Answer:
```

It separately captured the flashcard-generation prompt beginning `Generate
flashcards from:` and the independent strict-JSON claim-verification prompt
beginning `Given the EVIDENCE snippets and a CLAIM`.

Post-run API checks returned only the pre-existing `Generated Flashcards` deck;
there were no `task2c-flashcards-*`, `task2c-playwright-*`, or
`preexisting-playwright-*` records.  Thus prior fixed-name deck state cannot
select the generated cards' destination, and prior Playwright-like media cannot
select the RAG fixture.

## Gates

## Fix round 3: immutable code commit and missing boundaries

The scoped code/test commit is immutable evidence, separate from this later
documentation commit:

```text
code commit: 2517e29599a1d2d2120ff55d0b7ad76635865b1c
$ git diff --check 73c6998901..2517e29599a1d2d2120ff55d0b7ad76635865b1c
diff_check_exit=0
$ git show --check --format=fuller 2517e29599a1d2d2120ff55d0b7ad76635865b1c
show_check_exit=0
```

That code commit makes the existing real save-request capture assert its JSON
`deck_id` is exactly the unique per-run deck id.  The prior journey could pass
with a successful save and a larger all-decks count even if the selected deck
was wrong; it did not inspect this payload.  The new assertion is the missing
behavior boundary and preserves the actual UI deck selection and cleanup.

The new real-token negative control uses media context/title token
`task2c-playwright-1735689600000-abc123` and query token
`task2c-playwright-1735689600001-def456`.  It must select the established
`chat/source-summary.json` fallback, so removing or widening the scenario's
title/query backreference makes this test fail by selecting the grounded
response.  Its focused GREEN result was:

```text
$ source ../../.venv/bin/activate && python -m pytest \
  mock_openai_server/tests/test_critical_e2e_response_contracts.py \
  -k mismatched_query_token -q
1 passed, 9 deselected, 4 warnings in 1.81s
```

Post-code-commit live execution used the same shared DB, which still contained
the prior fixed `Generated Flashcards` deck.  The required deck-destination
journey executed the new `saveRequest.postDataJSON().deck_id === deck.id`
assertion successfully:

```text
$ ... bunx playwright test e2e/workflows/journeys/notes-flashcards.spec.ts \
  --project=journeys --workers=1 --reporter=list --trace=on
Running 1 test using 1 worker
  ✓  1 ... Notes -> Flashcards journey ... (10.6s)
1 passed (11.4s)
```

Post-run list checks returned only the pre-existing `Generated Flashcards`
deck—no `task2c-flashcards-*` deck—and no Task2C or preexisting Playwright media
fixtures.  Bandit for the changed Python test scope completed with
`bandit_results=0`.

The optional paired rerun reached the final nonempty media-db RAG prompt (title,
query token, and concrete Playwright sentence all captured) but failed the
separate unchanged direct-chat `/playwright/i` assertion when its generic default
response was returned.  This did not affect the deck-destination run above; no
product or matcher change was made from this single diagnostic failure.

## Fix round 3 H3: owned mock import path

The direct-chat fallback was not a scenario mismatch.  The owned graph had set
`PYTHONPATH` to the main checkout's `mock_openai_server`, while focused tests
used this worktree.  That older checkout did not include Task2C's content-list
text normalization, so the exact captured multimodal request selected the
generic default there.

H1 (lazy package export) and H2 (removing the server-local cache) both left the
actual CLI HTTP request on `chatcmpl-onboarding-uat-default`; both changes were
reverted and no production/mock-server code is retained from either attempt.
H3 set only this graph's Python path to the worktree package:

```text
mock_openai=/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/task-13013-7-supply-chain-design/mock_openai_server/mock_openai/__init__.py
mock_openai.config=/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/task-13013-7-supply-chain-design/mock_openai_server/mock_openai/config.py
mock_openai.server=/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/task-13013-7-supply-chain-design/mock_openai_server/mock_openai/server.py
```

The exact retained `gpt-3.5-turbo`, streaming, multimodal request then streamed
`chatcmpl-critical-playwright`, proving the configured grounded fixture won.
The final shared-DB pair was:

```text
Running 2 tests using 1 worker
  ✓  1 ... Ingest -> Search -> Chat journey ... (16.0s)
  ✓  2 ... Notes -> Flashcards journey ... (10.2s)
2 passed (27.4s)
```

Post-run list checks retained only the pre-existing `Generated Flashcards` deck;
there were no `task2c-flashcards-*`, `task2c-playwright-*`, or
`preexisting-playwright-*` records.  Focused critical response contracts were
`10 passed`; H3 Bandit was `bandit_results=0`; and
`git diff --check 73c6998901..HEAD` exited 0.

### Replayable H3 launch and direct-stream proof

The H3 mock was launched from this worktree with the following command (the
three `print` lines are the ownership guard recorded at startup):

```bash
PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/task-13013-7-supply-chain-design/mock_openai_server \
  ../../.venv/bin/python -c "import uvicorn, mock_openai, mock_openai.config, mock_openai.server as server; from mock_openai.config import load_config; print('mock_openai=' + mock_openai.__file__); print('mock_openai.config=' + mock_openai.config.__file__); print('mock_openai.server=' + server.__file__); load_config('apps/tldw-frontend/e2e/onboarding-uat/mock-openai/configs/local-success.json'); server.get_config_instance.cache_clear(); uvicorn.run(server.app, host='127.0.0.1', port=18091, log_level='warning')"
```

Its module-identity output was:

```text
mock_openai=/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/task-13013-7-supply-chain-design/mock_openai_server/mock_openai/__init__.py
mock_openai.config=/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/task-13013-7-supply-chain-design/mock_openai_server/mock_openai/config.py
mock_openai.server=/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/task-13013-7-supply-chain-design/mock_openai_server/mock_openai/server.py
```

The direct HTTP reproduction used the captured live request unchanged:

```bash
curl -sN -H 'Authorization: Bearer sk-uat-mock-openai' -H 'Content-Type: application/json' \
  http://127.0.0.1:18091/v1/chat/completions \
  --data '{"model":"gpt-3.5-turbo","messages":[{"role":"system","content":"You are a helpful AI assistant."},{"role":"user","content":[{"type":"text","text":"What is Playwright? Use the ingested content to answer."}]}],"temperature":0.7,"top_p":1.0,"max_tokens":4096,"n":1,"presence_penalty":0.0,"frequency_penalty":0.0,"stream":true,"logprobs":false}'
```

Its captured response begins:

```text
data: {"id": "chatcmpl-critical-playwright", "object": "chat.completion.chunk", "created": 1770000000, "model": "gpt-3.5-turbo", "choices": [{"index": 0, "delta": {"role": "assistant"}}]}
data: {"id": "chatcmpl-critical-playwright", "object": "chat.completion.chunk", "created": 1770000000, "model": "gpt-3.5-turbo", "choices": [{"index": 0, "delta": {"content": "Playwright is an open-source framework for"}}]}
data: {"id": "chatcmpl-critical-playwright", "object": "chat.completion.chunk", "created": 1770000000, "model": "gpt-3.5-turbo", "choices": [{"index": 0, "delta": {"content": " reliable end-to-end browser testing."}}]}
```

The paired command used this same H3 mock process, shared user DB
`/tmp/tldw-task2c-h2-userdb`, Redis `62457`, API `62458`, and WebUI `62459`:

```bash
TLDW_WEB_AUTOSTART=false TLDW_WEB_URL=http://127.0.0.1:62459 \
TLDW_SERVER_URL=http://127.0.0.1:62458 TLDW_E2E_SERVER_URL=http://127.0.0.1:62458 \
TLDW_API_KEY=test-api-key-for-e2e-testing-12345 \
TLDW_E2E_API_KEY=test-api-key-for-e2e-testing-12345 \
TLDW_MOCK_OPENAI_URL=http://127.0.0.1:18091/v1 \
NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=advanced NEXT_PUBLIC_API_URL=http://127.0.0.1:62458 \
bunx playwright test e2e/workflows/journeys/notes-flashcards.spec.ts \
  e2e/workflows/journeys/ingest-search-chat.spec.ts \
  --project=journeys --workers=1 --reporter=list --trace=on
```

```text
$ source ../../.venv/bin/activate && python -m bandit -r \
  mock_openai_server/mock_openai/config.py \
  mock_openai_server/tests/test_critical_e2e_response_contracts.py \
  -s B101 -f json -o /tmp/bandit_task2c_round2.json
exit 0; results=0
```

```text
$ cd apps/tldw-frontend && bun run typecheck
exit 1; no Task2C path reported.
Known unrelated errors: four DocumentationPage.tsx errors and four dirty
scripts/__tests__/skills-certification-*.test.ts errors.
```

```text
$ git diff --check 5ab7935ad8..HEAD
exit 0; stdout empty
```
