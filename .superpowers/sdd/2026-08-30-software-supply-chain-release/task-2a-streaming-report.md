# Task 2A — deterministic completed-stream journeys

## Status

**DONE_WITH_CONCERNS.** The initial evidence did not support a streaming-product fix; the controlled reproduction instead isolated stale local-provider configuration.  The final continuation records the tracked minimal fixture, focused regression coverage, and the live streaming result.  The report-only commits remain part of the evidence trail; the fixture package is committed separately.  The remaining Phase 6 responsive UI failures are out of this streaming/configuration package's scope.

## Root-cause trace and working comparison

The supplied prompt journey failure artifact showed this valid narrow chain:

1. `POST /api/v1/chat/completions` returned HTTP 200 as `text/event-stream` with a 2,430-byte response.
2. The rendered assistant article contained the deterministic provider text: `onboarding UAT ready. The mock provider returned a deterministic success response.`
3. The assistant was initially `aria-busy=true` with the stop-streaming control. Later polling observed an empty transcript and timed out.

That trace alone does not identify the state transition which cleared the transcript. It also cannot be treated as a single-run product trace: the diagnostic command that produced it had accidentally launched two local runner processes sharing the test services. The trace therefore interleaves one page that never submitted a turn and another that did render the deterministic answer.

The immediate comparison is the normal-chat stream pipeline: `background-proxy.ts` parses SSE data frames, `TldwChat.ts` iterates the stream, and `chatModePipeline.ts` updates the assistant stub then calls `setStreaming(false)` in `finally`. The captured response is compatible with that contract. The character completion path is structurally separate, so the common browser failure cannot be assigned to normal SSE parsing without a clean, one-run reproduction.

## Hypotheses

1. **Rejected:** the model-selector helper was treating a selected model as unselected. A temporary helper/unit test was added and passed, but a fresh send still reached the backend and rendered deterministic assistant text. The helper and its test were removed completely.
2. **Not confirmed:** an asynchronous server-chat/session hydration response overwrites the new local turn. The trace included a 404 for a prior server-chat settings reference, but the trace also contained the concurrent runner contamination, so no stale response can be identified as the source of the transcript reset. No hydration change was made.

No third product hypothesis or fix was attempted.

## TDD

No regression test was retained and no production/mock behavior was changed. The discarded selector experiment had an expected red import failure followed by two green helper checks, but it did not test the actual completed-stream failure and is intentionally not presented as repair evidence.

The required narrow red/green regression test cannot be written honestly until a clean reproduction identifies which state transition fails.

## Verification commands and results

```bash
git diff -- apps/tldw-frontend/e2e/utils/page-objects/ChatPage.ts
git status --short
git diff --check
```

Result: the temporary selector diff is gone; only the pre-existing dirty files listed in the brief remain; `git diff --check` exited 0.

The first attempted isolated service command used a nonexistent `<worktree>/.venv/bin/python` path and never ran Playwright. A corrected single-run command started the deterministic mock, API, and WebUI using the repository-root virtual environment, but its orchestration session was lost while the WebUI was still starting. The exact owned PIDs were stopped; a read-only listener check confirmed ports 18091, 62458, and 62459 were clear. It did not produce valid pass/fail test evidence.

No typecheck, Bandit, or focused test result is claimed because no files were changed.

## Files and commit

- Added: this report only.
- Reverted/deleted: the temporary `ChatPage.ts` selector helper and its untracked unit test.
- Commit: `66da1bf579 docs: record streaming repair investigation (TASK-13013.7.1)` (report-only; created before this continuation).

## Self-review

- Preserved all pre-existing dirty changes.
- Removed the 1.5 GB untracked Next build directory created by the diagnostic run.
- Did not alter journey assertions, timeouts, product streaming behavior, mock behavior, Research Workspace, Literature, or Playground paths.

## Concerns / context required

The previously reported five failures must be rerun with one managed mock/API/WebUI service set and no duplicate runner before a repair is justified. The available browser trace proves provider output arrives and renders, but not the subsequent state mutation responsible for the timeout. A clean trace or the original un-contaminated managed-run artifacts are required to proceed without guessing.

## Continuation — controlled one-run reproduction (2026-08-31)

### Status

**NEEDS_CONTEXT.** The requested one-run, shell-owned Redis/mock/API/WebUI graph completed and produced five failures, but it did not enter either stream endpoint. The direct graph is blocked before the user turn is submitted by model-metadata discovery against stale configured local endpoints. No production or test change is justified from this evidence.

### Service graph and command

One `zsh` process with `set -euo pipefail` owned Redis `62457`, deterministic mock `18091`, API `62458`, and the Playwright-started WebUI `62459`. The health gates passed for Redis `PING`, mock `/health`, and authenticated API `/api/v1/health`. The API used `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python`; the mock used `PYTHONPATH=/Users/macbook-dev/Documents/GitHub/tldw_server2/mock_openai_server`; all requested mock/provider/default/auth/Redis/frontend variables were set, with no `TEST_MODE`.

The one direct invocation was:

```bash
bunx playwright test \
  e2e/workflows/journeys/prompts-chat.spec.ts \
  e2e/workflows/journeys/character-chat.spec.ts \
  e2e/workflows/journeys/character-chat-phase6.spec.ts \
  --project=journeys --workers=1 --reporter=list --trace=on
```

The transport tool detached from the parent shell after start-up, but the recorded owned listener PIDs remained the exclusively controlled processes and were terminated after log/trace capture: Redis `44720`, mock `44721`, API `44722`, WebUI `44788`. Listener checks then showed all four ports clear.

### Root-cause trace and working comparison

1. The mock log contains only its two `/health` requests. It received neither `/v1/models` nor a chat-completion request.
2. The two stream-journey traces contain no browser `POST` to `/api/v1/chats/{id}/complete-v2` or `/api/v1/chat/completions`. Both timed out waiting for an assistant article because no turn was submitted.
3. Each trace records `GET http://127.0.0.1:62458/api/v1/llm/models/metadata` failing with status `-1` before a later successful request. The successful request took `18107.279ms` (character) and `18132.772ms` (prompt).
4. The API log records the matching metadata requests as HTTP 200 only after `18149ms` / `18126ms`. During that latency, `get_configured_providers_async` calls `discover_models_from_endpoint` for local entries. It probes the checked-in `[Local-API]` values, including `ooba_api_IP = http://192.168.2.235:5000/v1/chat/completions` and `tabby_api_IP = http://127.0.0.1:5000/v1/chat/completions` from `tldw_Server_API/Config_Files/config.txt`. The API log shows their three-second `ConnectTimeout` probes followed by 403 model probes to `127.0.0.1:5000`; none is the owned mock at `18091`.
5. The browser trace reports `[tldw:request] GET /api/v1/llm/models/metadata 0 Failed to fetch` and `fetchChatModels resolved {tldwCount: 0, total: 0}`. The UI retains a catalog-only `OpenAI / gpt-4o` chip. In both stream journeys, `ChatPage.sendMessage` filled the composer and pressed Enter, then the UI opened `Current Chat Model Settings` about 150ms later rather than sending a request. There is no usable configured model at that decision point and no send request.

The working comparison remains the managed workflow's mock at port `5000`: its static local-provider probes resolve locally rather than timing out. The direct graph intentionally used the required distinct mock port; that exposes that setting only the OpenAI/custom base URLs does not override every legacy Local-API endpoint considered by metadata discovery. This is a test-environment/configuration-resolution boundary, not evidence that a completed SSE response is cleared by hydration.

### Hypothesis and minimal test

**Confirmed environmental hypothesis:** metadata discovery walks stale Local-API endpoints from `config.txt`; the resulting ~18-second response causes the WebUI to have no usable model at send time, so no streaming request exists to debug.

No RED/GREEN regression test was added. A product test for this conclusion would either encode the local diagnostic topology or enlarge this package beyond the requested completed-stream repair. The existing behavior is directly proven by the clean trace and owned logs; adding a product change before a stream request is observed would be speculative.

### Exact verification results

```text
Playwright direct run: 5 failed, 0 passed, one worker, trace=on.
  - character-chat-phase6: desktop/tablet/mobile each failed its existing 30s UI assertion
  - character-chat: no completed streamed assistant response within its existing 60s assertion
  - prompts-chat: no completed streamed assistant response within its existing 60s assertion
```

No focused RED/GREEN, typecheck, or Bandit result is claimed because no source/test file changed. At the end of this continuation, the report-only commit was the sole Task 2A commit; the later tracked-fixture continuation superseded that state without amending history. No unrelated dirty file was staged or changed in this continuation.

### Self-review and concerns

- Confirmed the rejected selector-helper experiment remains absent.
- Confirmed hydration overwrite is **not** proven and identified no stale response that overwrote assistant content.
- The direct run is a valid single invocation and its trace/logs are uncontaminated, but it cannot validate the managed port-5000 flow because the API model catalog still probes legacy endpoints that the requested isolated graph does not own.
- A further repair needs direction on whether to (a) make metadata discovery skip non-selected stale Local-API providers, which has broad provider-catalog implications, or (b) change the controlled test topology to emulate every configured legacy endpoint. Neither is a narrow streaming fix supported by the Task 2A brief.

## Continuation — tracked minimal critical E2E fixture (2026-08-31)

### Controller decision and scope

The critical workflow now selects the tracked `tldw_Server_API/Config_Files/e2e-critical-config.txt` through `TLDW_CONFIG_FILE`.  The minimal config is intentionally only an `[API]` section: it leaves the existing environment-only Custom OpenAI endpoint/key/model authoritative and omits every checked-in stale `[Local-API]` endpoint.  Keeping it beside `config.txt` preserves `resolve_config_root()` access to canonical Prompt and module YAML assets.  Global provider discovery and `ChatPage.ensureModelSelected` remain unchanged.

### TDD evidence

Before the fixture/workflow change, this focused command collected two tests and failed both:

```bash
python -m pytest \
  tldw_Server_API/tests/CI/test_frontend_e2e_critical_fixture_contract.py \
  tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_readiness.py::test_critical_e2e_fixture_discovers_only_the_env_custom_openai_provider -q
```

RED results: the workflow contract raised `KeyError: 'TLDW_CONFIG_FILE'`; the provider-readiness test failed because `e2e-critical-config.txt` did not exist.

After the fixture and the one critical-workflow environment line were added, the identical command was GREEN: **2 passed, 3 warnings in 5.61s**.  The readiness test sets the workflow Custom OpenAI environment, loads the fixture via `TLDW_CONFIG_FILE`, asserts the config root is `Config_Files`, and proves metadata discovery makes exactly one call, `custom_openai_api -> http://127.0.0.1:18091/v1`, yielding enabled runnable text model `local-uat-chat`.  Its exact-call assertion excludes stale ooba/tabby probes.  The exact captured RED boundary, the current full warning origins, and retained live-run evidence are appended in fix round 1.

### Controlled live result

The requested single shell-owned Redis/mock/API/WebUI graph was rerun once with the fixture selected and the same five Playwright cases, one worker, list reporter, and trace enabled.  Result: **2 passed, 3 failed (2.0m)**.

- `character-chat.spec.ts` passed; `prompts-chat.spec.ts` passed.
- The owned mock received exactly two `POST /v1/chat/completions` requests and no stale Local-API probe.  API telemetry recorded successful streaming responses (nine chunks) for both.
- The remaining failures are exclusively `character-chat-phase6.spec.ts`: desktop/tablet time out waiting for the `Character chat sessions` region, and mobile times out waiting for the active Character Chat mode.  They occur before any mock completion request, so they are unrelated to provider catalog latency or completed-stream handling.

This clears the previously observed stale-config/model-metadata boundary for the two streaming journeys.  The live default OpenAI model still routes to the owned mock through its workflow environment; the focused readiness test separately proves the env-only Custom OpenAI model is discoverable without legacy endpoint probes.  There is no clean evidence of a catalog-only selector failure after the fixture, so no selector change was made.

### Remaining concern

The unrelated Phase 6 responsive character-session assertions remain red and should be investigated as a separate UI task.  This package does not broaden scope to alter those tests or their implementation.

### Final scoped verification

- `python -m pytest tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_readiness.py -q`: **27 passed, 3 warnings in 19.21s**.
- `python -m bandit -r tldw_Server_API/tests/CI/test_frontend_e2e_critical_fixture_contract.py tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_readiness.py -s B101 -f json -o /tmp/bandit_task2a_streaming.json`: **0 findings**.  B101 is excluded because both scanned paths are pytest tests and its only unfiltered findings were expected test `assert` statements.
- `git diff --check`: exited 0.

## Fix round 1 — report evidence correction (2026-08-31)

### Review findings addressed

This report's terminal status is `DONE_WITH_CONCERNS`, an allowed brief value.  The prior `COMPLETE` spelling has been removed.  This round changes only this report; it does not change provider discovery, workflow behavior, product code, tests, or journey assertions.

### Exact RED/GREEN evidence and warning origins

The original RED command was:

```bash
python -m pytest \
  tldw_Server_API/tests/CI/test_frontend_e2e_critical_fixture_contract.py \
  tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_readiness.py::test_critical_e2e_fixture_discovers_only_the_env_custom_openai_provider -q
```

Its captured failure boundary was:

```text
collected 2 items
FAILED tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_readiness.py::test_critical_e2e_fixture_discovers_only_the_env_custom_openai_provider
E       assert fixture_path.is_file()
FAILED tldw_Server_API/tests/CI/test_frontend_e2e_critical_fixture_contract.py::test_critical_e2e_workflow_selects_the_tracked_minimal_config_fixture
E       KeyError: 'TLDW_CONFIG_FILE'
2 failed, 3 warnings
```

The exact pre-commit GREEN command was the same command.  It produced `2 passed, 3 warnings in 5.61s`; the repository's pytest `addopts` includes `--disable-warnings`, so that original console result did not identify the warnings.  To identify their origins without changing code, the same two tests were rerun after this review with warning suppression disabled:

```bash
python -m pytest -o addopts= -q \
  tldw_Server_API/tests/CI/test_frontend_e2e_critical_fixture_contract.py \
  tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_readiness.py::test_critical_e2e_fixture_discovers_only_the_env_custom_openai_provider \
  -W default
```

Exact result:

```text
..                                                                       [100%]
=============================== warnings summary ===============================
fastapi/testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
pydantic/_internal/_fields.py:198: UserWarning: Field name "schema" in "ResponseFormatJsonSchemaSpec" shadows an attribute in parent "BaseModel"
_pytest/config/__init__.py:1474: PytestConfigWarning: Unknown config option: plugins
2 passed, 3 warnings in 5.37s
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```

The final `swigvarlink` line is emitted after pytest's summary and is not one of the three counted warnings.  The captured log also contains two application log warnings (`SINGLE_USER_API_KEY` legacy format and the test-only `USER_DB_BASE_DIR` fallback); those are Loguru records, not pytest warning-summary entries.

### Retained owned-graph evidence for the live claim

The original live graph artifacts are still present at the following paths; they are local owned-run evidence, not new CI output:

```text
/tmp/task2a-fixture-playwright.log
  SHA256 3ac4910753b15a4d373235ed514799741c8fb929129256cc19f053acda9a37e1
/tmp/task2a-fixture-mock.log
  SHA256 6bb5d7a4cfe66ccba3a1fcf9d56ddf9672481fc273445fbfd0bb9732578cba34
/tmp/task2a-fixture-api.log
  SHA256 286bbf821058b8c7ca7e37519ea506811794e793cf9913ab6360b2edabec5aad
/tmp/task2a-fixture-redis.log
  SHA256 674015c7f1ea0439c621b91d73bd81b7a1b8101c0304ab55b1a4dabda174f5d4
```

The two retained passing-stream traces are:

```text
apps/tldw-frontend/test-results/character-chat-Create-Char-19aae-te-v2-character-stream-path-journeys/trace.zip
  9,619,456 bytes; SHA256 37564464021071ddc5a2cb037088b9fc03dc58b3226dc706bb0065ba076e5f36
apps/tldw-frontend/test-results/prompts-chat-Prompts---Cha-e516b--in-chat-verify-in-API-call-journeys/trace.zip
  6,720,199 bytes; SHA256 0d1837eb88f752e19318f1cb0e01c07a75e24e03aaafe56d960306a5b839d009
```

Salient retained output establishes one owned graph and the two-pass/three-fail result:

```text
mock: Started server process [51095]
mock: Uvicorn running on http://127.0.0.1:18091
api:  Attempting to load .../Config_Files/e2e-critical-config.txt
api:  Started server process [51096]
api:  Uvicorn running on http://127.0.0.1:62458

Running 5 tests using 1 worker
✘ character-chat-phase6 ... desktop (32.6s)
✘ character-chat-phase6 ... tablet (31.9s)
✘ character-chat-phase6 ... mobile (31.2s)
✓ character-chat ... complete-v2 character stream path (6.3s)
✓ prompts-chat ... verify in API call (6.9s)
3 failed
2 passed (2.0m)

mock: POST /v1/chat/completions ... 200 OK
mock: POST /v1/chat/completions ... 200 OK
api:  POST /api/v1/chats/.../complete-v2 -> 200 in 15ms
api:  POST /api/v1/chat/completions -> 200 in 185ms
api:  provider openai; status success; streaming_response; chunks 9
```

The Phase 6 failures are independently retained in their error contexts: desktop/tablet wait for `getByRole('region', { name: 'Character chat sessions' })`; mobile waits for `getByTestId('playground-active-chat-mode')`.  These fail before either stream assertion and do not contradict the two passing streaming specs.  Because the owned logs and trace artifacts are intact, consistent in ports/PIDs/config fixture, and hash-identified above, no browser rerun was required for this report-only correction.
