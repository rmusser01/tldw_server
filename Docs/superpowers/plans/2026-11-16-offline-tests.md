# Plan: TASK-15486 - every test runs offline, and a session-wide blocker fails any new attempt

**Goal:** The 28 tests the task lists stub their download, provider or DNS call; an autouse fixture in
`tldw_Server_API/tests/conftest.py` wraps every test in `block_live_network()` and errors it at teardown
(with thread and stack per attempt) when a non-loopback DNS lookup or connect was attempted; tests that need
the network opt out by marker, by a line in a checked-in baseline of node ids, or by directory; the blocker
lands strict with the first CI run's surfaced tests recorded in that baseline (the count is AC #4's
evidence), a lint ratchet keeps the baseline from growing, and follow-up `TASK-<NEW>` stubs the rest by seam.

**Architecture:** One helper, `tldw_Server_API/tests/helpers/offline_network.py`, decides what is local
(loopback by address, `localhost`/`testserver`, the machine's hostname, the `TEST_DATABASE_URL` /
`TEST_DB_HOST` host) and which tests opt out (`live_network_allowed(item)`: the `live_network`,
`external_api`, `local_llm_service`, `provider_inventory`, `requires_model`, `stt_golden` markers; a node id
in `tldw_Server_API/tests/live_network_baseline.txt`; the `e2e/`, `server_e2e_tests/`, `frontend_e2e/`
trees). The conftest fixture `_no_live_network` is a self-contained block right after
`_offline_provider_model_inventory`; it only calls those two helper functions.
`tests/lint/test_live_network_baseline_ratchet.py` holds the baseline's ceiling. Each listed test stubs its
own seam at a core module, never an endpoint module (the endpoint-monkeypatch ratchet): the shared
downloader's `_create_async_client` with an `httpx.MockTransport`, `Audio_Files.http_download`, yt-dlp's
`extract_metadata` / `download_video`, `socket.getaddrinfo` for the webhook validator, the egress resolver
`Utils.egress.resolve_host_ips` (`offline_network.stub_host_resolution`) for the provider listing. The media
fixtures also extend CI's `WORKFLOWS_EGRESS_ALLOWLIST` (`offline_network.allow_egress_hosts`) for the fixture
hosts, since that allowlist denies them before any stub is reached. No production code changes.

**Tech Stack:** pytest 9 (`--strict-markers`, markers in `pyproject.toml`), `unittest.mock`, stdlib
`ipaddress` / `socket`; the probe plugin `<plans dir>/netprobe_15427.py` for measurement.

**Spec:** `backlog/tasks/task-15486 - Every-test-that-reaches-a-real-provider-or-host-runs-offline-and-a-session-wide-blocker-fails-any-new-attempt.md`
(description, AC #1-#4, the 2026-11-09 correction note). The task lists 28 tests (the dispatch said 22; the
file lists 1 + 15 + 7 + 2 + 3). TASK-15426 PR D (#3470, merged) already stubbed one of them,
`TestProcessAudios::test_process_audio_url_success_no_analysis_no_chunking` (it stubs
`Audio_Files.http_download`; confirmed at `test_media_processing.py:715-722` on `origin/dev` 3883a73712 and
in the RED baseline, which records no attempt for it). This plan covers the other 27.

**Prototype:** `<plans dir>/proto_15486.patch` (17 files, +573/-233: the helper, the conftest block, the marker,
the blocker tests, the baseline file and its lint ratchet, the two docs, the five listed files, and the four
files the local slice surfaced; measured as below). Apply with `git apply <plans dir>/proto_15486.patch`;
every code block in this plan is a hunk of it. The pre-review versions are `plan_15486.prefold.md` and
`proto_15486.prefold.patch`; the review is `review_15486.md` and its rulings are folded in below.

## Global Constraints

- Tests make no real outbound network requests. Every pytest run uses the network probe plugin:
  `PYTHONPATH=<plans dir> NETPROBE_LOG=<file> python -m pytest -p netprobe_15427 ...`. It blocks and logs every
  outbound DNS lookup and connection; a run must end with 0 log lines for the tests the plan adds or changes.
- A test that drives `/chat/completions`, an LLM adapter or a judge must stub the adapter or the provider call. A
  commercial provider otherwise calls the real API.
- A failed judge or model call, or a reply with no readable score, is unscored: `score: None`, a sample error, and
  out of every average. Never 0.0, never a success. This is the owner's rule, recorded 2026-11-10 and 2026-11-12.
- Backlog edits use backlog-py only (`PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit ...`).
  Use `--append-notes`, never `--notes`. Stage with `git add -A backlog/tasks`. Never hand-edit task files. A
  new follow-up task takes its id under the session's id lock (the controller does that; write `TASK-<NEW>`).
- Branch from `origin/dev`; the PR targets `dev`.
  - Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
  - The PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- Never `--no-verify`, never force-push, never `git stash`.
- PostgreSQL only through the repo fixtures, with a session-unique `TEST_DB_NAME`. Never restart the server.
  Redis: never FLUSHALL or SCRIPT FLUSH.
- A mirrored doc (ADRs, the BYOK guide, `Env_Vars.md` and the other files under `Docs/Published`) needs
  `bash Helper_Scripts/refresh_docs_published.sh`, with the `Docs/Published` diff in the same PR.
- Shifting lines in a file listed in `Helper_Scripts/ci/scope_predicate_baseline.txt` fails backend-required.
  Regenerate it with the script's `--write-baseline`, in the same PR.
- Size-capped functions (the function-size ratchet in `tests/lint`) must not grow past their caps. Measure each
  touched function before and after.
- A plan with embedded diffs fails trailing-whitespace when it is committed: keep the plan free of trailing spaces.
- Task-specific:
  - Local pytest runs need `PYTHONPATH=<plans dir>:<plans dir>/pyext` (the shared venv lacks `pytest-mock`,
    which the root `conftest.py` loads as a plugin; `pyext` holds the real package). CI installs it from the
    `dev` extras, so nothing in the PR mentions `pyext`.
  - Once the blocker is in, the probe log only sees attempts made *outside* the function-scoped fixture
    window (session/module fixture setup, opted-out tests). Attempts inside the window surface as a teardown
    error `live network attempted: [...]`; count those with `grep -c "live network attempted"` on the run output.
  - `Helper_Scripts/checks/guard_http_client_patching.py` (pre-commit and CI) rejects a test line that pairs
    `mock|patch|monkeypatch` with `requests|httpx|aiohttp`; patch `socket.getaddrinfo`, the endpoint's download
    name or the `http_client` helpers instead.
  - `tldw_Server_API/tests/conftest.py` is also changed by TASK-15497's plan (the app-main guard around
    `_keep_app_main_reloads_from_leaking`, lines 704-726) and by open PR #3437 (lines 727-760). This plan's only
    conftest hunk is the self-contained `_no_live_network` block inserted between
    `_offline_provider_model_inventory` (ends line 344) and `auth_headers` (line 347); keep it there.
  - `Docs/Development/Testing.md` and `tldw_Server_API/tests/README.md` are not mirrored under
    `Docs/Published` (no `Development/` tree there; the refresh script stages `Docs/` only), so no refresh.
  - No touched function is listed in `tldw_Server_API/tests/lint/function_size_baseline.json`, and no touched
    file is in `Helper_Scripts/ci/scope_predicate_baseline.txt` or `shard_coverage_baseline.txt`. One new
    test file, `tests/lint/test_live_network_baseline_ratchet.py`, falls in the existing
    `tldw_Server_API/tests/lint` shard (measured: `new_uncovered=0`, 910 shards, 5,494 files); the blocker
    tests extend `tests/LLM_Adapters/unit/test_no_live_network_fixture.py` (shard
    `tldw_Server_API/tests/LLM_Adapters/unit`).
  - `git apply` is atomic per call: to apply or restore one file of the prototype use one `--include=<path>`
    per call (a call that includes an already-applied file applies nothing).
  - `tests/lint/test_endpoint_monkeypatch_ratchet.py` freezes, per endpoint module under
    `app/api/v1/endpoints/`, the number of test sites that `monkeypatch.setattr` a name on it (string or
    object form). A stub must patch the collaborator where the endpoint looks it up: a core module's
    attribute (`Utils.egress.resolve_host_ips`, `download_utils._create_async_client`,
    `tokenizer_resolver._http_post`, `Video_DL_Ingestion_Lib.*`, `Audio_Files.http_download`), never
    `prompts.*`, `llm_providers.*` or `media.process_*.*`. The first prototype grew five modules and failed
    the ratchet; the patch now grows none.
  - Environment-only here, not in CI: `bandit` is not installed in the shared venv (`python -m bandit` fails
    with "No module named bandit"); `tests/lint/test_runtime_package_discovery.py` errors at collection
    without `setuptools`, so local runs pass `--ignore` for it (a `--deselect` cannot skip a collection error).
  - CI's shard environment differs from a local run in ways the blocker exposes: `ci.yml`'s shard job sets
    `WORKFLOWS_EGRESS_ALLOWLIST=93.184.216.34,does-not-resolve.invalid,example.com` (an allowlist denies every
    other host before any stub is reached; locally the list is empty and admits all), `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
    with `-p pytest_asyncio.plugin -p pytest_timeout -o timeout_method=signal -m "not jobs and not e2e"`,
    `TEST_MODE=true`, `DISABLE_HEAVY_STARTUP=1`, `DATABASE_URL=sqlite:///./Databases/users.db`. Task 2 runs the
    five files once under that environment (the `plan15486_cienv` recipe below); a fixture that stubs a URL
    download adds its host with `allow_egress_hosts`.
  - Only the Linux / Python 3.12 shards run on a PR (`ci.yml`: the 3.13, macOS, Windows and release shards
    run on push and release only), so platform-specific attempts surface after the merge to `main`; the PR
    body says so.

## Review Focus

- The blocker's local decision: `_is_local` must accept `127.0.0.2`, `::ffff:127.0.0.1`, `0.0.0.0`, `[::1]`,
  `localhost`, `testserver`, `None` (bind), the machine's hostname and the `TEST_DATABASE_URL` /
  `TEST_DB_HOST` host, and nothing else; a numeric non-loopback host passes `getaddrinfo` but not `connect`.
- The fixture is function-scoped and must not swallow the test's own exception: it yields inside the
  `with`, then asserts; a test that raised still reports its own failure plus the teardown error.
- The teardown message keeps `live network attempted: [(kind, host), ...]` as its first line (the subprocess
  test and Task 3's CI grep match it) and adds one line per attempt with the thread name and a 10-frame
  stack tail, so an attempt made by a thread an earlier test left running names that thread instead of
  reading as a flaky failure of whichever test was running (the review's finding 4). `Attempts` subclasses
  `list`, so the existing `attempts == [...]` assertions and the probe plugin are unchanged.
- Nested blocks compose: tests that already use `no_live_network` (LLM_Adapters) or `block_live_network()`
  (Media, RAG_NEW) run inside the session blocker; the inner block records, the outer stays empty.
- Each stub replaces exactly one seam and the asserted behaviour stays real (the batch endpoint, the
  validator's port check, the quota 402); a stub that would hollow a test (patching `run_video_batch`) is not
  used, and none patches an endpoint module (the endpoint-monkeypatch ratchet).
- `stub_host_resolution` answers one public address for every name: a test that asserts a private-target
  denial (`10.0.0.5`, `169.254.169.254`) must not take it; the readiness file keeps those on the real
  resolver.
- `test_media_contextual_integration.py`: the six `if mock_process.called:` guards are now
  `assert mock_process.called, response.text`, so the chunk-option assertions run or the test fails. The
  stub writes the sample PDF / EPUB bytes by extension and the parametrized URLs carry a real extension
  (`test.txt`, `test.epub`, `test.mp4`, `test.mp3`), because the add flow validates the downloaded file's
  extension and MIME type before any processor runs (the review's finding 2: with text in `test.pdf`, 8 of
  12 never reached the mock). The upload test's `save_uploaded_files` stub returns the documented
  `({"path", "original_filename", "input_ref"} dict list, errors)` shape instead of a str list, which had
  made the endpoint raise `TypeError` behind the guard.
- The baseline: `live_network_baseline.txt` starts empty; Task 3 fills it from the first CI run and sets the
  ratchet's `CEILING` to that count. A node id in it must be exact (parametrized ids included), sorted,
  unique, and name an existing file; the lint test checks all three.
- The subprocess test (`test_the_session_blocker_errors_a_test_whose_live_lookup_was_swallowed`) runs a
  second pytest; it must work with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` (CI shards) as
  `test_offline_model_inventory.py::test_unit_inventory_stays_offline_when_network_tests_are_enabled` does.

## Measured baseline (origin/dev 3883a73712, this machine)

RED, the five files that hold the 28 listed tests, under `-p netprobe_15427`, serial:
`1 failed, 162 passed, 24 skipped, 1 xfailed in 35:26` (the failure is
`test_security.py::TestWebhookSecurityValidation::test_ssrf_protection_private_ip`, which the old blocker
breaks on its numeric hosts; the other 26 pass or skip because the code under test swallows the blocked
download); 27 probe lines, one per listed test except the pixabay one (nodeid and attempts in
`<plans dir>/red_15486_probe.log`): `www.youtube.com` x16 (the two video tests, yt-dlp retries),
`example.com` x19 (the audio 404 test, the seven contextual tests, the five webhook tests),
`filesamples.com` x8, `raw.githubusercontent.com` x2, `www.w3.org` x1, `api.openai.com` x2 (the prompt test,
from `egress.py:419`'s resolver thread), and the four numeric hosts of `test_ssrf_protection_private_ip`.

GREEN with the pre-review prototype (the post-fold counts follow the contextual paragraph below), under
the probe, in two runs (both `exit=0`, 0 probe lines, 0 `live network
attempted` teardown errors): the blocker tests, the two Evaluations files and the prompt file:
`123 passed, 3 skipped in 5:01`; the blocker tests again plus the two media files:
`96 passed, 10 skipped, 1 xfailed in 19:22`. Per file: `test_no_live_network_fixture.py` 23 passed, 1 skipped
(the subprocess target); the five listed files together `174 passed, 11 skipped, 1 xfailed` (186 items: the
188 of the baseline minus the two deleted skip-helper self-tests; the 11 skips are the files' own
`pytest.skip`/`skipif` cases, none of the 27). After the ratchet rework (the `MockTransport` download seam,
`stub_host_resolution`), re-measured with `-rs`: `test_media_processing.py` alone
`61 passed, 9 skipped, 1 xfailed in 4:42` (the nine skips: analysis-needs-LLM x5, docling x4), and the
blocker tests, the prompt quota test, the readiness, provider-config, video-ingestion and redirect files
together `204 passed, 1 skipped in 0:52`; both 0 probe lines, 0 teardown errors.

Under CI's shard environment (the Global Constraints recipe, probe on), the five files with the pre-review
prototype: `14 failed, 160 passed, 11 skipped, 1 xfailed in 7:25`, 0 probe lines, 0 teardown errors. The 14
are every URL test in `test_media_processing.py`: video x2, the pixabay audio test (#3470's), PDF x1, ebook x8,
documents x2, all `207` with `URL blocked by security policy: Host not in allowlist` (youtube.com, w3.org,
filesamples.com, raw.githubusercontent.com, cdn.pixabay.com are not in CI's allowlist; dev today skips most
of them through the skip helper). With `allow_egress_hosts` in the three fixtures and the pixabay test:
`174 passed, 11 skipped, 1 xfailed in 6:23`, exit 0, 0 probe lines, 0 teardown errors, 0 `Host not in
allowlist` (measured on the final prototype; the 11 skips are the files' own: LLM analysis x5, docling x4, the
two sandbox webhook-delivery skips). The first GREEN attempt also failed `test_process_video_multi_status_mixed`:
CI's list denies `INVALID_URL`'s host too, so the invalid URL answered `Host not in allowlist` instead of the
metadata error the test asserts; `offline_video_url` therefore adds both hosts.

Contextual tests with the guards turned into asserts and the pre-review text-only stub (I2 RED):
`8 failed, 4 passed in 1:58`: `test.pdf` holding text fails MIME validation (the four `document` tests
that post `test.pdf`, and the `[pdf]` case), `test.document` and `test.ebook` fail the extension check (not
in the document / ebook allowlists), and the upload test's stub made the endpoint raise `TypeError`. With the sample PDF / EPUB bytes by
extension, real extensions in the parametrized URLs and the dict-shaped upload stub: `12 passed in 1:12`,
0 probe lines. Getting there took two more corrections the asserts forced: the `[pdf]` and `[ebook]` cases
patched `endpoints.media.process_pdf_task` / `process_epub`, names the add flow never calls (it imports
`PDF_Processing_Lib.process_pdf_task` and `Book_Processing_Lib.process_epub` at call time), so both ran the
real processors; the upload stub must write its file inside the endpoint's temp dir (paths outside it are
rejected). One finding stays: the add flow hands the PDF processor `chunk_method` / `max_chunk_size` /
`chunk_overlap` and drops `chunk_options`, so `enable_contextual_chunking` never reaches it; the `[pdf]` case
now asserts `chunk_method == "sentences"` and the gap goes to the notes (Decision 6).

After the fold (the final prototype on `origin/dev` 61ad5280ca, the review's base), all measured, all 0 probe
lines and 0 teardown errors: the five listed files locally `174 passed, 11 skipped, 1 xfailed in 6:06` and
under CI's shard environment `174 passed, 11 skipped, 1 xfailed in 6:23` (0 `Host not in allowlist`);
`test_no_live_network_fixture.py` `26 passed, 1 skipped` (three cases more than before: the two new
markers and the baselined node id); with `test_live_network_baseline_ratchet.py` (2 tests) and the review's
lint/CI ratchets, `80 passed, 1 skipped`; stage C (dev's `tests/conftest.py`) `1 failed, 25 passed,
1 skipped`; the ratchet's RED as in Task 1 Step 4. Shard coverage (`new_uncovered=0`), the HTTP-client
patch guard, `test_quality_triage --enforce` and `check_backlog_task_ids` exit 0; `git diff --check` clean.

Representative slice under the prototype (AC #4, local part): the wider set (`git grep -l` of every touched
symbol, 34 files, minus the five above and the two conftests: 25 files, 555 tests) plus a 60-file random
sample (988 tests; seed 15486, `random.Random(15486).shuffle` over `git ls-files` test files outside `e2e/`,
`server_e2e_tests/`, `frontend_e2e/`, `performance/`, `stress/`, `property/`, `benchmarks/`; the list is
`<plans dir>/slice_15486_files.txt`), 85 files, 1,543 tests, `-n 2 --dist loadfile`, under the probe:
`9 failed, 1508 passed, 26 skipped, 12 errors in 1:40:05`; 0 probe lines (nothing outside the fixture
window). The 12 errors are `live network attempted` teardown errors in four files, none of them among the
28: `MediaIngestion_NEW/unit/test_video_ingestion.py` x5 (`example.com`: `download_video` ->
`_raise_if_url_blocked` -> `evaluate_url_policy`, which resolves every name host because
`WORKFLOWS_EGRESS_BLOCK_PRIVATE` defaults to true), `http_client/test_redirect_header_hardening.py` x4
(`example.com`: `http_client._validate_egress_or_raise` -> `evaluate_url_policy`),
`LLM_Calls/test_provider_config_resolution.py` x1 (`slot-2.example`, the provider listing's endpoint
policy) and `Chat_NEW/unit/test_llm_providers_readiness.py` x2 (`api.openai.com`, the same). Eight of the
nine failures are those tests failing in the body on the blocked lookup; the ninth,
`Services/test_lifecycle_workers.py::test_stop_registered_workers_sets_events_and_waits_concurrently`
(`assert 0.2199 < 0.2`), is a timing assertion under load, unrelated. Re-running the readiness file alone
surfaced a 13th, `test_catalog_computes_discovery_once_and_passes_same_result_to_readiness` (three
`connect` attempts to `10.0.0.5`: `get_configured_providers` -> `_resolve_model_tokenizer_support` ->
`tokenizer_resolver` runtime probe -> `http_client.fetch`), hidden in the slice by test order, so a CI shard
with a different file grouping can surface tests the slice did not.

Rate: 4 of the 988 random-sample tests (1 file in 60) is 4.0 unknown network tests per 1,000; the wider-set
part, network-adjacent by construction, 8 of 555 (14.4 per 1,000); the whole slice 12 of 1,543 (7.8 per
1,000). The task's description extrapolated "tens of tests" from 1 file in its 100-file sample; pooled with
this sample (2 files in 160) the suite-wide rest is about 1.25% of the 5,379 files, about 65 files, each
with a handful of tests, so the CI count before the stubs is more likely in the 50-300 range than in the
tens (the interval is wide: one hit in 60 files). All 13 came from three seams (`evaluate_url_policy` DNS
resolution in three callers, the tokenizer runtime probe), so the fixes batch by seam, not by test. CI
measures the rest (Task 3).

## Task 0: In Progress, branch, draft PR, baseline

**Files:** `backlog/tasks/task-15486 - ...md` (backlog-py only).

- [ ] **Step 1:** `git fetch origin dev && git switch -c codex/task-15486-offline-tests origin/dev`.
- [ ] **Step 2:** `PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit TASK-15486 --status "In Progress"`;
      `git add -A backlog/tasks && git commit -m "chore(backlog): TASK-15486 in progress"`.
- [ ] **Step 3:** Push and open a draft PR against `dev` (`gh pr create --base dev --draft ...`); the body ends
      with the generated-with line. CI's `full-suite-linux-312-shards` (219 shards, `.github/workflows/ci.yml`
      lines 510-525) runs on every PR whose changed paths classify as backend; this PR's conftest change does.
- [ ] **Step 4: Baseline (RED for AC #1).** Run the five files under the probe:
      ```
      PYTHONPATH=<plans dir>:<plans dir>/pyext NETPROBE_LOG=/tmp/red_15486.log python -m pytest -p netprobe_15427 -q -p no:cacheprovider \
        tldw_Server_API/tests/Media_Ingestion_Modification/test_media_processing.py \
        tldw_Server_API/tests/Media_Ingestion_Modification/test_media_contextual_integration.py \
        tldw_Server_API/tests/Evaluations/test_security.py \
        tldw_Server_API/tests/Evaluations/test_production_features_integration.py \
        tldw_Server_API/tests/Prompt_Management_NEW/integration/test_prompt_improvement_api.py
      ```
      Expected: `1 failed, 162 passed, 24 skipped, 1 xfailed` (the SSRF test), 27 probe lines, the hosts
      above; none for the pixabay test.
- [ ] **Step 5: RED for AC #2 and #3 (measured, three stages).** Apply only the prototype's
      `tldw_Server_API/tests/LLM_Adapters/unit/test_no_live_network_fixture.py` hunk and run the file:
      `python -m pytest -q -p no:cacheprovider tldw_Server_API/tests/LLM_Adapters/unit/test_no_live_network_fixture.py`.
      Expected (stage A, dev helper, conftest and pyproject): `ImportError: cannot import name 'LIVE_NETWORK_DIRS'
      from 'tldw_Server_API.tests.helpers.offline_network'` (AC #2: the helper has no opt-out decision and spells
      loopback as four strings). Then apply the helper and conftest hunks, keep dev's `pyproject.toml`: expected
      (stage B) `collected 0 items / 1 error`, `'live_network' not found in 'markers' configuration option`
      (AC #3, `--strict-markers`). Then apply the pyproject hunk and put dev's `tests/conftest.py` back
      (`git checkout origin/dev -- tldw_Server_API/tests/conftest.py`): expected (stage C, re-measured on the
      folded prototype) `1 failed, 25 passed, 1 skipped`, the failure being
      `test_the_session_blocker_errors_a_test_whose_live_lookup_was_swallowed` at
      `assert result.returncode != 0` with the subprocess reporting `1 passed` (AC #2: without the fixture a
      swallowed live lookup passes). Restore each file with its own `git apply --include=<path>` and
      `git reset -q -- <path>` (the checkout staged it).

## Task 1: The blocker, its opt-outs, its tests and its docs (AC #2, AC #3)

**Files:**
- Modify: `tldw_Server_API/tests/helpers/offline_network.py` (rewrite, 161 lines)
- Modify: `tldw_Server_API/tests/conftest.py` (insert 20 lines after line 344)
- Modify: `pyproject.toml` (one marker line after `provider_inventory`)
- Create: `tldw_Server_API/tests/live_network_baseline.txt` (comment header only; Task 3 fills it)
- Create: `tldw_Server_API/tests/lint/test_live_network_baseline_ratchet.py` (`CEILING = 0` until Task 3)
- Modify: `tldw_Server_API/tests/LLM_Adapters/unit/test_no_live_network_fixture.py` (+120 lines)
- Modify: `Docs/Development/Testing.md` (new "Offline By Default" section before "CI Guard"),
  `tldw_Server_API/tests/README.md` (one bullet under "Pytest Markers & CLI Switches")

**Interfaces** (`offline_network.py`):
- `block_live_network() -> ContextManager[Attempts]`: `Attempts` is a `list` of `(kind, host)` tuples (so
  every existing `attempts == [...]` assertion still holds) with `.details`, one line per attempt:
  `"<kind> <host> in thread <name>: <file:line fn> <- ..."` (a 10-frame stack tail). Loopback by address
  (`ipaddress.ip_address(host)`, IPv4-mapped unwrapped, `is_loopback or is_unspecified`), the names
  `localhost`, `testserver`, `""`/`None`, `socket.gethostname()`, `TEST_DB_HOST` and the host of
  `TEST_DATABASE_URL` (read at entry; names are allowed at DNS, a connect is judged by address: a hostname
  that resolves off loopback is a false positive, not a leak); a numeric host passes
  `getaddrinfo`/`gethostbyname` unrecorded, its `connect`/`connect_ex` is recorded and raises.
- `live_network_allowed(item) -> bool`: `item.get_closest_marker(m)` for `m` in `LIVE_NETWORK_MARKERS`
  (`live_network`, `external_api`, `local_llm_service`, `provider_inventory`, `requires_model`,
  `stt_golden`), or `item.nodeid in baseline_node_ids()` (`LIVE_NETWORK_BASELINE`, read once), or
  `item.path` under `tests/<tree>` for `tree` in `LIVE_NETWORK_DIRS` (`e2e`, `server_e2e_tests`,
  `frontend_e2e`).
- `stub_host_resolution(monkeypatch, address=PUBLIC_ADDRESS)`: `Utils.egress.resolve_host_ips` answers one
  public address; `allow_egress_hosts(monkeypatch, *hosts)`: extends `WORKFLOWS_EGRESS_ALLOWLIST` when one is
  set (an empty list already admits every public host).
- A `ponytail:` comment names the unwrapped gaps: `_socket.socket.connect` on the C class, UDP `sendto`,
  `gethostbyname_ex` / `gethostbyaddr`, libpq, c-ares, grpc (all confirmed by the reviewer's probe; widen when a
  leak through one is found).
- The conftest teardown message: `live network attempted: [...]` on the first line (what the subprocess test
  and the CI grep match), then one `details` line per attempt.

- [ ] **Step 1 (RED):** apply the prototype's test file only
      (`git apply --include='tldw_Server_API/tests/LLM_Adapters/unit/test_no_live_network_fixture.py' <plans dir>/proto_15486.patch`)
      and run it. Expected: `ImportError: cannot import name 'LIVE_NETWORK_DIRS' from 'tldw_Server_API.tests.helpers.offline_network'`.
- [ ] **Step 2 (GREEN, helper):** apply the helper hunk
      (`git apply --include='tldw_Server_API/tests/helpers/offline_network.py' <plans dir>/proto_15486.patch`).
      The recorder (a `list`, so `attempts == []` and every existing `attempts == [...]` still hold) and the
      ceiling it does not cover:
      ```python
      # ponytail: wraps the Python-level socket API only. Not seen: `_socket.socket.connect` called on the C
      # class, UDP `sendto`, `gethostbyname_ex` / `gethostbyaddr`, and resolvers outside Python (libpq, c-ares,
      # grpc). Covered: `socket.connect`/`connect_ex`, `create_connection`, asyncio `open_connection` and
      # `sock_connect`, ssl, httpx/anyio/urllib3. Widen when a leak through one of those is found.


      class Attempts(list):
          """The ``(kind, host)`` tuples a blocked run recorded, plus ``details`` with thread and stack per attempt."""

          def __init__(self) -> None:
              super().__init__()
              self.details: list[str] = []

          def record(self, kind: str, host: str) -> None:
              self.append((kind, host))
              frames = traceback.extract_stack(limit=12)[:-2]
              where = " <- ".join(f"{Path(f.filename).name}:{f.lineno} {f.name}" for f in reversed(frames))
              self.details.append(f"{kind} {host} in thread {threading.current_thread().name}: {where}")
      ```
      The decision functions:
      ```python
      def _ip(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
          try:
              ip = ipaddress.ip_address(host.split("%", 1)[0])
          except ValueError:
              return None
          return getattr(ip, "ipv4_mapped", None) or ip


      def _allowed_names() -> frozenset[str]:
          """Loopback names, the machine's own hostname and the developer's test PostgreSQL host.

          These names are compared at DNS; a connect to the address they resolve to is still judged by
          address, so a hostname that resolves off loopback is a false positive (blocked), never a leak.
          """
          names = {socket.gethostname(), os.environ.get("TEST_DB_HOST", "")}
          try:
              names.add(urlsplit(os.environ.get("TEST_DATABASE_URL", "")).hostname or "")
          except ValueError:
              pass
          return _LOCAL_NAMES | {name.lower() for name in names if name}


      def _is_local(host: str, allowed: frozenset[str]) -> bool:
          if host in allowed:
              return True
          ip = _ip(host)
          return ip is not None and (ip.is_loopback or ip.is_unspecified)
      ```
      and inside `block_live_network()` the resolver passes a numeric host through:
      `if _is_local(name, allowed) or _ip(name) is not None: return real(host, *args, **kwargs)`; both
      wrappers call `attempts.record(kind, host)`. The opt-out reads the baseline once:
      ```python
      @lru_cache(maxsize=1)
      def baseline_node_ids() -> frozenset[str]:
          """Node ids listed in ``live_network_baseline.txt`` (one per line, ``#`` comments allowed)."""
          if not LIVE_NETWORK_BASELINE.exists():
              return frozenset()
          lines = (line.strip() for line in LIVE_NETWORK_BASELINE.read_text(encoding="utf-8").splitlines())
          return frozenset(line for line in lines if line and not line.startswith("#"))
      ```
      and `allow_egress_hosts` only widens a list that is set:
      ```python
      def allow_egress_hosts(monkeypatch, *hosts: str) -> None:
          """Add hosts to the egress allowlist when one is set (CI sets one; an empty list admits every public host)."""
          current = [host.strip() for host in os.environ.get(EGRESS_ALLOWLIST_ENV, "").split(",") if host.strip()]
          if current:
              monkeypatch.setenv(EGRESS_ALLOWLIST_ENV, ",".join(dict.fromkeys(current + list(hosts))))
      ```
      Run the test file again. Expected: `collected 0 items / 1 error`, `'live_network' not found in
      'markers' configuration option` (the marked test; `--strict-markers` fails the whole module). With the
      marker line of Step 3 in but not the conftest block: `1 failed, 25 passed, 1 skipped` (the subprocess
      test: no fixture, so the swallowed lookup passes; measured as Task 0 stage C).
- [ ] **Step 3 (GREEN, marker and fixture):** apply the `pyproject.toml` and `tldw_Server_API/tests/conftest.py`
      hunks. The conftest block, inserted verbatim between `_offline_provider_model_inventory` and
      `auth_headers`:
      ```python
      @pytest.fixture(autouse=True)
      def _no_live_network(request):
          """Fail a test that reaches a non-loopback host (TASK-15486).

          Opt out with the ``live_network`` marker (``external_api``, ``local_llm_service``,
          ``provider_inventory``, ``requires_model`` and ``stt_golden`` count too), by a line in
          tests/live_network_baseline.txt, or by living under tests/e2e, server_e2e_tests or
          frontend_e2e; see tests/README.md. Function-scoped: session- and module-scoped
          fixture setup runs outside it.
          """
          from tldw_Server_API.tests.helpers.offline_network import block_live_network, live_network_allowed

          if live_network_allowed(request.node):
              yield
              return
          with block_live_network() as attempts:
              yield
          assert attempts == [], "live network attempted: " + repr(list(attempts)) + "\n" + "\n".join(attempts.details)
      ```
      The marker line: `"live_network: the test reaches a real host on purpose; the autouse offline blocker in tests/conftest.py (TASK-15486) stands down",`.
      Run the test file. Expected: `26 passed, 1 skipped` (the skip is the subprocess target; measured). The
      subprocess target's teardown error, measured, reads:
      ```
      E   AssertionError: live network attempted: [('dns', 'example.com')]
      E     dns example.com in thread MainThread: test_no_live_network_fixture.py:167 test_probe_target_swallows_a_live_lookup <- python.py:167 pytest_pyfunc_call <- ...
      ```
- [ ] **Step 4 (the baseline and its ratchet):** create `tldw_Server_API/tests/live_network_baseline.txt`
      (a three-line `#` header, no ids) and `tldw_Server_API/tests/lint/test_live_network_baseline_ratchet.py`
      (`CEILING = 0`; `test_baseline_does_not_grow`, and
      `test_baseline_entries_are_sorted_unique_node_ids_of_existing_files`, which also checks the file agrees
      with `baseline_node_ids()`), both from the prototype. RED (measured): one id appended to the baseline:
      `1 failed, 1 passed`, `1 node ids in live_network_baseline.txt, ceiling 0: stub the test's seam
      instead`; two unsorted ids with `CEILING = 2`: `1 failed, 1 passed`, `sort the baseline and drop
      duplicates`. GREEN with the empty baseline: `2 passed` (measured). The blocker test
      `test_a_baselined_node_id_opts_out` checks that an exact node id opts out and a different parametrization
      of it does not.
- [ ] **Step 5 (docs):** apply the `Docs/Development/Testing.md` and `tldw_Server_API/tests/README.md` hunks
      (the opt-outs including the baseline, the loopback rule, the numeric-host rule, that the names are
      compared at DNS (a hostname resolving off loopback is a false positive, a remote numeric
      `TEST_DATABASE_URL` is allowed by design), the thread and stack lines, and the ceiling: session/module
      fixtures, `app/core/MCP_unified/tests`, and the unwrapped calls).
      `python -m pytest -q tldw_Server_API/tests/Docs` Expected: passes (no mirror involved).
- [ ] **Step 6:** Commit: `test(conftest): block live network in every test, opt out by marker, baseline or e2e tree (TASK-15486)`.

## Task 2: The 27 listed tests stub their seam (AC #1)

**Files:**
- Modify: `tldw_Server_API/tests/Media_Ingestion_Modification/test_media_processing.py`
- Modify: `tldw_Server_API/tests/Media_Ingestion_Modification/test_media_contextual_integration.py`
- Modify: `tldw_Server_API/tests/Evaluations/test_security.py`
- Modify: `tldw_Server_API/tests/Evaluations/test_production_features_integration.py`
- Modify: `tldw_Server_API/tests/Prompt_Management_NEW/integration/test_prompt_improvement_api.py`

**Seams** (where each listed test reached the network, from the RED probe and stack traces):
- Videos (2): `Video_DL_Ingestion_Lib.extract_metadata` (yt-dlp, `www.youtube.com`); stub it, `download_video`
  and `perform_transcription` in a class fixture `offline_video_url` (`INVALID_URL` gets no metadata, so the
  mixed test keeps its "Failed to extract metadata" error). `evaluate_url_policy(url)` without a configured
  endpoint does not resolve, so no egress stub.
- Audio mixed (1): `Audio_Files.http_download` (`example.com`, the 404 URL); stub it to raise, which
  `download_audio_file` wraps as `AudioDownloadError("Download failed for ...")`.
- PDF (1), documents (2), ebooks (7): the shared downloader `download_utils.download_url_async`, which the
  three endpoints bind by name (`core_download_url_async`). Patching that endpoint name, as the file's own
  `test_process_doc_multi_status_mixed` does, grows the endpoint-monkeypatch ratchet
  (`tests/lint/test_endpoint_monkeypatch_ratchet.py`: `process_pdfs` 8 > 7, `process_ebooks` 8 > 7,
  `process_documents` 11 > 10), so the stub is the collaborator the downloader looks up at call time:
  `download_utils._create_async_client`, answered by an `httpx.MockTransport`, the seam the file's own
  `test_process_doc_url_various_formats` already uses. One module fixture serves the three fixture hosts:
  ```python
  FIXTURE_DOWNLOADS = {
      urlparse(VALID_PDF_URL).hostname: ("application/pdf", SAMPLE_PDF_PATH),
      urlparse(VALID_EPUB_URL).hostname: ("application/epub+zip", SAMPLE_EPUB_PATH),
      urlparse(VALID_TXT_URL).hostname: ("text/plain", None),
  }


  @pytest.fixture
  def fixture_downloads_served(monkeypatch):
      """The shared downloader's HTTP client answers the fixture URLs from the sample files; nothing leaves the machine.

      INVALID_URL never reaches the client: the downloader fails a ``.invalid`` host fast in tests.
      """
      from tldw_Server_API.app.core.Ingestion_Media_Processing import download_utils

      def respond(request: httpx.Request) -> httpx.Response:
          content_type, sample = FIXTURE_DOWNLOADS[request.url.host]
          body = sample.read_bytes() if sample else b"GNU GENERAL PUBLIC LICENSE\nThis license applies to the program."
          return httpx.Response(
              200, headers={"content-type": content_type, "content-length": str(len(body))}, content=body
          )

      transport = httpx.MockTransport(respond)

      def create_client(**kwargs):
          return httpx.AsyncClient(timeout=kwargs.get("timeout", 60.0), transport=transport)

      monkeypatch.setattr(download_utils, "_create_async_client", create_client)
  ```
  (`monkeypatch` and `httpx` stay on separate lines for the HTTP-client patch guard.) `TestProcessEbooks`
  gets a class-level autouse fixture that takes it (all seven URL tests; no other test in the class
  downloads); `test_process_pdf_url_success` and the two document tests take it by name (the class's
  `test_process_doc_url_various_formats` has its own transport and must not be pre-empted). The mixed tests
  keep their `Download/preparation failed` error for `INVALID_URL` because `download_url_async` fails a
  `.invalid` host before any client exists.
- The skip-on-network-failure helpers (`_is_fixture_download_failure`,
  `skip_if_external_fixture_url_unreachable`, `skip_if_valid_video_url_unreachable`, `_FakeBatchResponse` and
  their two self-tests, 120 lines) have no caller left and are deleted, as is the ebook mixed test's
  `time.sleep(5)`. Nothing else in the repo references them (`git grep`).
- Contextual (7): `/media/add` downloads through `download_utils.download_url_async` after
  `url_validation.assert_url_safe`, the two names the file's `test_batch_media_with_contextual_chunking`
  already patches; a class-level autouse fixture `_offline_url_download` patches both for every test in the
  class and writes the sample PDF or EPUB bytes when the URL ends in `.pdf` / `.epub` (text otherwise), because
  `_validate_downloaded_url_file` checks the extension against the media type's allowlist and the detected MIME
  type against it. The parametrized URLs get a real extension per media type (`_URL_EXTENSION`: document
  `.txt`, ebook `.epub`, video `.mp4`, audio `.mp3`; the node ids `[document-sentences]` etc. are unchanged),
  the six `if mock_process.called:` guards become `assert mock_process.called, response.text`, and the upload
  test's `save_uploaded_files` stub returns the documented dict list for a file it writes under `tmp_path`.
- The three media fixtures (`fixture_downloads_served`, `offline_video_url`, the pixabay test's stub) call
  `allow_egress_hosts(monkeypatch, <their hosts>)`: under CI's `WORKFLOWS_EGRESS_ALLOWLIST` the endpoint's
  egress check answers `Host not in allowlist` before the downloader runs (I1).
- Webhook validator (2 in `test_security.py` + 3 in `test_production_features_integration.py`):
  `_validate_hostname` calls `socket.getaddrinfo(hostname, None)` even with `check_connectivity=False`
  (`webhook_security.py:398`); `register_webhook` validates with `check_connectivity=True`, which also runs
  `_test_connectivity`. `test_blocked_ports` wraps its loop in
  `patch("socket.getaddrinfo", return_value=[(None, None, None, None, ("93.184.216.34", 0))])` (the file's
  `test_ssrf_dns_rebinding` precedent); the three registration tests take a module fixture
  `offline_webhook_validation` that patches the same plus
  `WebhookSecurityValidator._test_connectivity` (an `AsyncMock` answering reachable).
  `test_ssrf_protection_private_ip` needs no stub: its hosts are numeric, which the new blocker lets
  `getaddrinfo` resolve on the machine.
- Prompt improvement (1): the route lists providers through `llm_providers.get_configured_providers`, which
  runs `egress.evaluate_url_policy(..., configured_endpoint=...)` and resolves every configured provider host
  (`api.openai.com`, in a resolver thread: `egress.py:419`). `llm_providers` binds `evaluate_url_policy` by
  name and `prompts` binds `get_configured_providers` by name, and both are endpoint modules the
  endpoint-monkeypatch ratchet freezes (`prompts` 43 > 42), so the seam is the resolver every policy call
  shares: `Utils.egress.resolve_host_ips`, looked up as a module global at call time. The helper
  `offline_network.stub_host_resolution(monkeypatch)` sets it to answer one public address with no lookup:
  ```python
  PUBLIC_ADDRESS = "93.184.216.34"


  def stub_host_resolution(monkeypatch, address: str = PUBLIC_ADDRESS) -> None:
      """Every name host resolves to one public address with no lookup, so the egress policy admits it."""
      from tldw_Server_API.app.core.Utils import egress

      monkeypatch.setattr(egress, "resolve_host_ips", lambda *_args, **_kwargs: (address,))
  ```
  A module fixture `offline_provider_hosts` calls it and the test takes the fixture; held in a fixture, not
  inline, because a fourth mock call in the test body trips the triage ratchet's `mock_density` count for
  the file (baseline 7). The provider listing then runs to completion: discovery hits
  `_offline_provider_model_inventory`'s raising `_http_fetch`, not the network.

- [ ] **Step 1 (RED):** with Task 1 in, run the five files without the probe:
      `python -m pytest -q -p no:cacheprovider <the five files>`.
      Expected: the 27 tests error at teardown with `live network attempted: [...]` (the hosts of the RED
      probe log; the ebook, PDF and document ones still pass in the body because the endpoint turns the
      blocked download into a 207, the video ones skip in the body); nothing else in the five files errors.
- [ ] **Step 2 (GREEN):** apply the five test-file hunks of the prototype
      (`git apply --include='tldw_Server_API/tests/Media_Ingestion_Modification/*' --include='tldw_Server_API/tests/Evaluations/*' --include='tldw_Server_API/tests/Prompt_Management_NEW/*' <plans dir>/proto_15486.patch`).
      Run the five files under the probe with `-rs`. Expected (re-measured after the fold):
      `174 passed, 11 skipped, 1 xfailed`, 0 probe
      lines, `grep -c "live network attempted"` = 0, and no listed test among the `-rs` skips. Then the wider set, serially (no PostgreSQL file names a touched
      symbol):
      ```
      PYTHONPATH=<plans dir>:<plans dir>/pyext NETPROBE_LOG=/tmp/wider_15486.log python -m pytest -p netprobe_15427 -q -p no:cacheprovider \
        $(git grep -l "block_live_network\|offline_network\|no_live_network\|live_network_allowed\|stub_host_resolution\|get_configured_providers\|Audio_Files.http_download\|download_utils._create_async_client\|download_utils.download_url_async\|_test_connectivity\|WebhookSecurityValidator\|Video_DL_Ingestion_Lib.extract_metadata\|Video_DL_Ingestion_Lib.download_video\|tokenizer_resolver._http_post\|egress.resolve_host_ips" -- 'tldw_Server_API/tests/**/test_*.py')
      ```
      Expected: 0 probe lines; the only teardown errors are the 8 that Task 3 Step 1 stubs
      (`test_video_ingestion.py` x5, `test_provider_config_resolution.py` x1,
      `test_llm_providers_readiness.py` x2; run with Task 3's hunks in, 0 errors).
- [ ] **Step 3 (CI's environment, I1):** run the five files once under CI's shard environment:
      ```
      cd <repo>; export PYTHONPATH=.:<plans dir>:<plans dir>/pyext PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true DISABLE_HEAVY_STARTUP=1 \
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python WORKFLOWS_EGRESS_ALLOWLIST="93.184.216.34,does-not-resolve.invalid,example.com" \
        DATABASE_URL="sqlite:///./Databases/users.db" NETPROBE_LOG=/tmp/cienv_15486.log; unset TEST_DATABASE_URL POSTGRES_TEST_DB
      python -m pytest -q --disable-warnings -p pytest_asyncio.plugin -p pytest_timeout -o timeout_method=signal \
        -p netprobe_15427 -p no:cacheprovider -m "not jobs and not e2e" -rsfE <the five files>
      ```
      RED (measured, fixtures without `allow_egress_hosts`): `14 failed, 160 passed, 11 skipped, 1 xfailed`,
      every URL test of `test_media_processing.py` answering 207 `Host not in allowlist`. GREEN (measured, with
      the `allow_egress_hosts` calls): `174 passed, 11 skipped, 1 xfailed`, exit 0, 0 probe lines, 0 teardown
      errors, 0 `Host not in allowlist`; no listed test among the skips. The skip helpers
      (`skip_if_external_fixture_url_unreachable` and friends) are what kept these tests green in CI; they are
      deleted in this same hunk and only because this run is green (decision 3).
- [ ] **Step 4 (I2):** run `test_media_contextual_integration.py` under the probe. RED (measured, asserts with
      the text-only stub): `8 failed, 4 passed`, `Detected MIME type 'text/plain' for file 'test.pdf' is not
      allowed`, `Claimed extension '.ebook' ... is not allowed`, `Unexpected internal error: TypeError`. GREEN
      (measured, typed bytes, real extensions, the library-module patch targets for pdf/ebook, the upload
      stub writing inside the endpoint's temp dir): `12 passed`, 0 probe lines, 0 teardown errors.
- [ ] **Step 5:** `python Helper_Scripts/checks/guard_http_client_patching.py` Expected: exit 0.
      `python Helper_Scripts/ci/test_quality_triage.py --enforce` Expected: exit 0 (the fixtures hold the
      patches; no test function stacks four).
- [ ] **Step 6:** Commit: `test: the 27 tests that reached a real host stub their seam (TASK-15486)`.

## Task 3: Measure the blocker on the suite and fix what it surfaces (AC #4)

**Files:** the four the slice surfaced (in the prototype), then whatever CI surfaces (test files only; a
production seam is a finding, not a change here):
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_video_ingestion.py` (a module fixture
  `remote_url_allowed` that sets `video_lib.evaluate_url_policy` to admit the URL unresolved, the file's own
  `fake_evaluate_url_policy` precedent; the five remote-download tests take it)
- Modify: `tldw_Server_API/tests/http_client/test_redirect_header_hardening.py` (a class fixture
  `origins_admitted_offline` on `TestRedirectFlows` that sets `egress_mod.evaluate_url_policy` to a policy
  with `resolved_ips=()`, the class's own precedent at lines 179-202; an address would make the client pin
  the hop to it and the handler would never see `example.com`; the four cross-origin tests take it)
- Modify: `tldw_Server_API/tests/LLM_Calls/test_provider_config_resolution.py` (one
  `stub_host_resolution(monkeypatch)` in the env-only slot test; `llm_providers` is an endpoint module, so
  not `llm_providers.evaluate_url_policy`, which the ratchet counts: 199 > 196)
- Modify: `tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_readiness.py` (a module helper
  `_api_openai_com_admitted_offline(monkeypatch)` -> `stub_host_resolution`, called by the two
  `api.openai.com` tests, not by `_client_for_config`, because
  `test_llama_manual_model_on_metadata_target_is_egress_blocked` and the `10.0.0.5` tests need the real
  policy to deny their private targets; and in
  `test_catalog_computes_discovery_once_and_passes_same_result_to_readiness` the tokenizer runtime probe's
  own seam, `tokenizer_resolver._http_post`, set to raise `TokenizerUnavailable`; the file's line-589
  precedent patches `llm_providers._resolve_model_tokenizer_support` instead, which the ratchet counts)

- [ ] **Step 1 (RED, measured):** the slice under the prototype without these four hunks:
      ```
      PYTHONPATH=<plans dir>:<plans dir>/pyext NETPROBE_LOG=/tmp/slice_15486.log python -m pytest -p netprobe_15427 -q -p no:cacheprovider -n 2 --dist loadfile $(cat <plans dir>/slice_15486_files.txt)
      ```
      Expected: `9 failed, 1508 passed, 26 skipped, 12 errors`, the 12 teardown errors above, 0 probe lines.
- [ ] **Step 2 (GREEN):** apply the four hunks
      (`for f in tldw_Server_API/tests/MediaIngestion_NEW/unit/test_video_ingestion.py tldw_Server_API/tests/http_client/test_redirect_header_hardening.py tldw_Server_API/tests/LLM_Calls/test_provider_config_resolution.py tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_readiness.py; do git apply --include="$f" <plans dir>/proto_15486.patch; done`;
      `git apply` is atomic per call, so one `--include` per call). Run the four files under the probe:
      Expected: `180 passed`, 0 probe lines, 0 teardown errors (measured). The same rules apply to every test CI
      surfaces: stub at its seam (the three seams above cover most), or, only when the test's purpose is the
      live call, `@pytest.mark.live_network`. Record each in the notes.
      Commit: `test: stub the 13 tests the offline blocker surfaced locally (TASK-15486)`.
- [ ] **Step 3 (CI, the full suite, once):** push; the draft PR runs `.github/workflows/ci.yml`
      `full-suite-linux-312-shards` (219 shards over every test path, `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`,
      `-p pytest_asyncio.plugin`, the egress allowlist of the Global Constraints), `jobs-suite.yml`
      (`RUN_JOBS=1`), `backend-required.yml` (unit smoke over `tests/unit`, contracts and ratchets),
      `coverage-required.yml`, and the e2e workflows (exempt trees). Read the run:
      ```
      run=$(gh run list --workflow CI --branch codex/task-15486-offline-tests --limit 1 --json databaseId --jq '.[0].databaseId')
      gh run view "$run" --json jobs --jq '.jobs[] | select(.conclusion == "failure") | .name'
      gh run view "$run" --log-failed | grep -oE "^(ERROR|FAILED) [^ ]+ - .*live network attempted.*" | sort -u > /tmp/surfaced_15486.txt
      gh run view "$run" --log-failed | grep -B2 "live DNS lookup blocked\|live connect blocked"
      ```
      The second grep catches tests that failed in the body before teardown (their node id is the `FAILED` line
      above the match). Expected: a list of node ids with hosts (on CI pytest prints the whole message in the
      short summary, so the thread and stack lines follow each `ERROR` line; the grep keeps the first). Its
      length is the number
      AC #4 records: "how many tests the blocker failed before the stubs". Also list the shards that failed for
      another reason (compare with a recent `dev` PR run): they are not this PR's.
- [ ] **Step 4 (the baseline):** write every surfaced node id, one per line, sorted, into
      `tldw_Server_API/tests/live_network_baseline.txt` (`cut -d' ' -f2 /tmp/surfaced_15486.txt | sort -u`),
      and set `CEILING` in `tldw_Server_API/tests/lint/test_live_network_baseline_ratchet.py` to that count.
      Group the ids by seam while writing (the stack tail in the message names it: `egress.py` resolver,
      `tokenizer_resolver`, `download_utils`, `http_client`, yt-dlp, a provider adapter); stub in this PR only
      the groups that one existing helper closes (`stub_host_resolution`, `allow_egress_hosts`,
      `fixture_downloads_served`'s pattern) and delete their lines. Everything else stays in the baseline.
      Run the lint ratchet and the blocker tests locally
      (`python -m pytest -q tldw_Server_API/tests/lint/test_live_network_baseline_ratchet.py tldw_Server_API/tests/LLM_Adapters/unit/test_no_live_network_fixture.py`,
      expected all passed), commit `test: baseline the N tests the offline blocker surfaced in CI (TASK-15486)`,
      push. Expected: the shards pass (a baselined test runs outside the blocker; a stubbed one makes no
      attempt). If a shard surfaces a test the first run did not (a different file grouping, a leaked thread:
      the details line names the thread), add it to the baseline the same way; this converges in one or two
      pushes because the baseline, not a stub, closes each one.
- [ ] **Step 5 (the follow-up):** file `TASK-<NEW>` (the controller takes the id): "Stub the N tests in
      tests/live_network_baseline.txt by seam and shrink the baseline to empty", with the grouped list from
      Step 4 and the helper each group uses, and reference it from the baseline file's header and the notes.
- [ ] **Step 6:** Append to the notes (backlog-py `--append-notes`): the CI run id, the count before the
      stubs (AC #4), the seams stubbed in this PR and the count left in the baseline, the shards that stayed
      red for unrelated reasons, the session/module fixtures checked for the ceiling (below), the 157
      uncovered files of `app/core/MCP_unified/tests`, and that only the Linux 3.12 shards ran.

**AC #4 as the baseline split reads it.** Proposed wording, for the owner: "A full-suite CI run passes
with the blocker on. Every test that still reached the network is listed by node id in
tldw_Server_API/tests/live_network_baseline.txt, the notes record that count, a lint test keeps the file
from growing, and TASK-<NEW> stubs them by seam until the file is empty." The evidence is the CI run id, the
surfaced list, and the green run with the baseline in.

**The session/module ceiling (AC #2):** the fixture is function-scoped. The session- and module-scoped
fixtures checked by reading their bodies: autouse ones (`Collections/conftest.py::preserve_app_state`,
`Docs/conftest.py::ensure_docs_published_mirror`, `Jobs/conftest.py::_bootstrap_jobs_routes_env`,
`VectorStores/integration/conftest.py::vectorstores_isolated_env`,
`Watchlists/conftest.py::enable_full_app_for_watchlists`, `_plugins/chat_fixtures.py::preserve_app_state`,
`prompt_studio/conftest.py::prompt_studio_jobs_test_env`, `sandbox/conftest.py::set_ws_poll_timeout_for_tests`
and `::reduce_warnings_noise`, the root `conftest.py::_audit_inline_io`, `tests/conftest.py::_raise_fd_limit`
and `::_shutdown_executors_and_evaluations_pool`) touch env, app state or limits only; the e2e ones live in
exempt trees. Non-autouse ones that open sockets: the PostgreSQL fixtures (`_plugins/postgres.py`,
`helpers/pg.py`, `helpers/pgvector.py`, `AuthNZ/conftest.py`: docker on loopback or the `TEST_DB_HOST`
host, both allowed), `Evaluations/conftest.py::mock_openai_server` and
`_plugins/chat_fixtures.py::mock_openai_server` (a subprocess on `localhost:8080`),
`Chat/integration/conftest_isolated.py:217` (a socket on localhost, the reviewer's grep),
`Agent_Client_Protocol/...::compiled_runner` (a Go build; the Go module cache is local, but `go build` may
fetch modules if the cache is cold: the only session fixture that can reach a real host, and only on a cold
cache), `ChromaDB/conftest.py::hf_or_deterministic_embeddings` (downloads only with `RUN_MODEL_TESTS=1`).
Under the probe (which does wrap setup), the slice recorded 0 attempts outside the fixture window. Also
outside the fixture: `tldw_Server_API/app/core/MCP_unified/tests` (157 files, about 3% of 5,494; the second
`testpaths` entry, its own `conftest.py`) is not under `tests/conftest.py`, so the blocker does not cover it
(decision 4); and anything that bypasses Python's `socket.socket`: `_socket.socket.connect` on the C class,
UDP `sendto`, `gethostbyname_ex` / `gethostbyaddr`, libpq, c-ares, grpc (the reviewer's probe confirmed each
is unwrapped; the helper's `ponytail:` comment names them). Only the Linux 3.12 shards run on a PR.

## Task 4: Verification, close-out, review, PR

- [ ] **Step 1:** Touched areas and the wider set under the probe (Task 2 Step 2 commands); `tests/Docs`,
      `tests/lint`, `tests/CI`:
      `python -m pytest -q -p no:cacheprovider --ignore=tldw_Server_API/tests/lint/test_runtime_package_discovery.py tldw_Server_API/tests/Docs tldw_Server_API/tests/lint tldw_Server_API/tests/CI`
      Expected: passes, apart from the environment-only cases (the ignored file errors without
      `setuptools`; `test_jobs_postgres_exports_recorded_coverage` fails without `coverage`). Measured on the
      prototype: `tests/Docs` green (the 1,618-item combined run passed every Docs test before entering the
      lint tree); `tests/lint/test_endpoint_monkeypatch_ratchet.py` `8 passed` after the rework (the first
      prototype failed it: `llm_providers` 199 > 196, `process_documents` 11 > 10, `process_ebooks` 8 > 7,
      `process_pdfs` 8 > 7, `prompts` 43 > 42); on the folded prototype, the blocker tests with the new
      `test_live_network_baseline_ratchet.py` and the review's ratchets (`test_endpoint_monkeypatch_ratchet`,
      `test_function_size_ratchet`, `test_llm_calls_import_isolation`, `test_scope_predicate_ratchet`,
      `test_test_seam_env_reads_guarded`, `CI/test_skip_markers_have_reasons`): `80 passed, 1 skipped`, 0 probe
      lines (the contextual file's pdf/ebook targets moved off `endpoints.media`, which the endpoint ratchet
      accepts); the rest of `tests/lint` and `tests/CI` were not run to the end
      here (the lint tree took 40 minutes to reach 27% on the loaded machine), so the implementer's run is
      the first full measurement.
- [ ] **Step 2:** `python Helper_Scripts/ci/test_quality_triage.py --enforce`;
      `python Helper_Scripts/ci/check_shard_coverage.py --ci-file .github/workflows/ci.yml`;
      `python Helper_Scripts/ci/check_backlog_task_ids.py` Expected: exit 0 each (measured on the folded
      prototype: all three exit 0, shard coverage `new_uncovered=0` with the new lint file; the triage report
      lists no touched file).
- [ ] **Step 3:** `bandit -q tldw_Server_API/tests/helpers/offline_network.py` (the only non-test source
      touched; not installed here, so this runs in CI's security-required or after `pip install bandit`)
      Expected: no findings; the subprocess call in the test file carries `# nosec B603`/`B404` as the file's
      precedent does. `git diff --check` Expected: clean (measured). `python Helper_Scripts/checks/guard_http_client_patching.py`
      Expected: exit 0 (measured).
- [ ] **Step 4:** Close-out with backlog-py: tick AC #1-#4 and DoD #1-#6 (`--check-ac`, `--check-dod`), append
      the final summary (`--append-notes`, the Final Summary section via the tool's summary option), status
      `Done`; `git add -A backlog/tasks`; commit `chore(backlog): close TASK-15486`.
- [ ] **Step 5:** Clean-session final review (`/superpowers:requesting-code-review`), address findings, then
      mark the PR ready. PR body: what the blocker decides, the 27 seams, the CI count before the stubs and the
      count left in the baseline (with `TASK-<NEW>`), the opt-outs, the ceilings (function scope: session and
      module fixtures run outside it; a thread that outlives its test fails the test running when it connects,
      and the details line names that thread; server subprocesses are not probed; the unwrapped calls:
      `_socket.socket.connect` on the C class, UDP `sendto`, `gethostbyname_ex` / `gethostbyaddr`, and libpq,
      c-ares and grpc; `app/core/MCP_unified/tests`, 157 files, is outside `tests/conftest.py`), the false
      positives (`socket.gethostname()` and a `TEST_DB_HOST` name are allowed at DNS but their connect is
      judged by the resolved address, so one that resolves off loopback is blocked, not leaked; a
      `TEST_DATABASE_URL` naming a remote numeric address is allowed by design), that only the Linux / Python
      3.12 shards ran on the PR (3.13, macOS, Windows and release shards run after the merge), and the Change
      summary (ADR-004; the controller asks about the waiver). The merge is the controller's.

## Decisions for the owner

The independent review (`review_15486.md`) agreed with decisions 1-4 and the controller ruled on 5; the
prototype follows the rulings.

1. **Extra opt-out markers.** AC #3 names `live_network`, `external_api`, `local_llm_service`,
   `provider_inventory`. Tests marked `requires_model` (ChromaDB, `RUN_MODEL_TESTS=1`) and `stt_golden`
   (`TLDW_STT_GOLDEN_ENABLE=1`) download models by definition and are already env-gated. Ruled and done:
   both are in `LIVE_NETWORK_MARKERS` (reviewer: "yes, both env-gated, 9 files; it costs nothing").
2. **What a surfaced CI test gets.** Stub at its seam; `@pytest.mark.live_network` only for a test whose
   assertion is the live call itself (none in the 28). Reviewer: confirm. Done; Task 3 Step 4 applies it.
3. **The skip helpers.** `skip_if_external_fixture_url_unreachable` and friends (120 lines) are deleted,
   but only because the five files run green under CI's environment with `allow_egress_hosts` (Task 2
   Step 3); they were what kept the URL tests green in CI. Reviewer: confirm, once finding 1 is fixed. Done.
4. **The MCP_unified test tree.** `tldw_Server_API/app/core/MCP_unified/tests` (157 files) is not under
   `tests/conftest.py` and stays uncovered; the notes and the docs say so. Reviewer: accept now. Done; the
   alternative (the 20-line block in the repo-root `conftest.py`) is a one-line follow-up if those tests ever
   show an attempt under the probe.
5. **The baseline split (controller's ruling, offered to the owner as the recommendation).** The blocker
   lands strict with `tests/live_network_baseline.txt`, a checked-in list of node ids still allowed to reach
   the network (the repo's ratchet pattern); the lint test `test_live_network_baseline_ratchet.py` keeps it
   from growing; Task 3 runs CI once, records every surfaced node id (the AC #4 count), stubs only what an
   existing helper closes, and the PR merges green with the baseline; `TASK-<NEW>` stubs the rest by seam and
   shrinks the file to empty. The alternative, looping the PR until every shard is green without a baseline,
   is unbounded: the local slice puts the count at 50-300 (reviewer: plan for 300+, since PG, Redis, pgvector
   and `jobs` tests are gated off locally and CI's egress allowlist denies hosts the local run admits). The
   owner's call: accept the ruling (the plan) or the unbounded loop.
6. **A product gap the un-guarded contextual tests exposed (information, no code change here).** The
   `/media/add` PDF path builds the processor's arguments from `chunk_options` (`chunk_method`,
   `max_chunk_size`, `chunk_overlap`) and drops the rest, so `enable_contextual_chunking` and
   `contextual_llm_model` never reach PDF processing; document and ebook receive the whole `chunk_options`.
   The `[pdf]` contextual case now asserts the forwarded `chunk_method` and says so in a comment.
   Recommendation: a follow-up task to forward the contextual options to the PDF processor (or to document
   that PDFs ignore them), filed from the notes.

## Open PRs touching the same files

From `gh pr list --base dev --state open` (38 PRs) and each one's file list:
- #3480 (`codex/political-pdf-vip-fixes`) edits `test_media_processing.py` lines 715-741 (it moves the
  pixabay test to a controlled `example.com` URL); this plan's hunks in that file are elsewhere (imports,
  367-486, the video/audio-mixed/PDF/ebook/document tests). Same file, different hunks; whichever merges
  second rebases.
- #3437 (`codex/moderation-ci-triage-regex-design`) edits `tldw_Server_API/tests/conftest.py` lines 727-760
  (`_reset_main_app_lifecycle_state_between_tests`) and `pyproject.toml` (dev extras, package data). This
  plan's conftest hunk is at line 345 and its pyproject hunk in `markers`; no textual overlap.
- #2869, #2855 touch `pyproject.toml` outside the `markers` list; #2421 touches `mcp_unified/pyproject.toml`
  only.
- #3180 edits `test_llm_providers_readiness.py`; the review found it auto-merges with this branch (its own
  conflicts with `dev` are in `llm_providers.py` and `config_info.py`).
- No open PR touches the other test files, the helper, the two docs, `test_no_live_network_fixture.py`, or the
  two new files.
