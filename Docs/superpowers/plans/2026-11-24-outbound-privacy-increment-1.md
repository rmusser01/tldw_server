# Outbound Privacy Increment 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Ship a reviewed product destination and asset inventory and a required CI and local check for the covered default and opted-in flows.

**Architecture:** Validate declarations against code and fixture-backed dispatch. Exercise the default backend, opted-in dispatch, shipped wizard, production Next.js server, and packaged Chrome extension in disposable Linux containers with loopback networking and a private filesystem. Use the same local runner and pass/fail contract in the existing required CI contexts.

**Tech stack:** Python and pytest, the existing Python socket guard, HTTPX and the Hugging Face SDK's supported test seams, Bun, Next.js, Chromium and Playwright, and Docker for the test boundary.

**Spec:** [Outbound destination inventory and privacy regression checks](../../Design/2026-11-24-outbound-destination-inventory-design.md), Increment 1.

**Backlog:** TASK-15634, dependent on TASK-15633.

**ADR check:** Required. [Proposed ADR-097](../../ADR/097-reviewed-outbound-destinations-and-privacy-modes.md) covers the reviewed inventory and required checks. [ADR-026](../../ADR/026-security-outbound-egress-and-ssrf-policy.md) governs existing URL safety. Fit the gate into the existing license-admitted contexts and preserve compatibility with the CI tiers in [proposed ADR-096](../../ADR/096-tiered-development-and-release-ci.md).

## Global constraints

- Work in this scope only. The check has no runtime mode switches, mutable approvals, contact ledger, or new runtime network dependency.
- Keep CI dependencies, developer installation, image pulls, browser installation, and artifact builds outside the guarded scenarios. App-invoked setup installs are in scope. Record their tested and untested boundaries separately.
- Exercise defaults from tracked configuration, synthetic credentials, and temporary state. Clear ambient proxy, identity, database and test-mode settings from child processes. Keep guard and attempt recording active through shutdown. Avoid real external DNS, connections and downloads in the exercised tests.
- Default scenarios allow only their named fixture endpoints, by loopback address and port. Fixture names may use a test-owned static mapping. A product declaration grants no exception in a default smoke. Observe selected external destinations in opted-in tests using a fake transport response before any real connection.
- Guarded runs have no host service socket mounts, checkout bind mounts, forwarded agent sockets, inherited host socket descriptors, or published ports. Only the host-side runner talks to Docker. The isolated process sees image contents, container-local state, and Docker's generated host files.
- A required scenario that is skipped, fails to start or shut down, or produces missing, malformed or incomplete evidence fails. Report the tested boundaries and limitations, including SDK/argv-only evidence, untested native and child paths, and browser/OS coverage.
- Do not upload raw URL paths, request bodies, headers, environment, console dumps, browser traces, or raw syscall traces. Use a bounded, redacted report with scenario, component, capability or repository source hint, decision, and coverage.
- Create tests first and observe the intended failure. Run the applicable focused suites, actionlint for workflow changes, Bandit on touched Python, and `git diff --check`. Use backlog-py for records. Commit each finished stage with TASK-15634. Preserve the required checks and license admission.

## File map and interface

| Location | Responsibility |
| --- | --- |
| `Helper_Scripts/ci/outbound_inventory.json` and `outbound_inventory.py` | Versioned review data and side-effect-free declaration/dispatch validation |
| `Helper_Scripts/ci/outbound_check.py` | Public local and CI command, test dispatch, artifact validation, final exit status |
| `tldw_Server_API/tests/helpers/offline_network.py` | Opt-in exact endpoint guard for this smoke, with the current pytest defaults preserved |
| `Helper_Scripts/ci/outbound_backend_smoke.py` | Fresh isolated app launch, local setup/runtime assertions, process cleanup and report |
| `tldw_Server_API/tests/CI/test_outbound_inventory.py`, `test_outbound_backend_smoke.py`, `test_outbound_opt_in.py`, `test_outbound_check.py` | Manifest, guard and subprocess contracts, SDK/provider dispatch fixtures, and runner evidence tests |
| `Dockerfiles/Dockerfile.outbound-check` and `Helper_Scripts/ci/outbound_isolation.py` | Test-only images, source projection and shared container preparation, launch and preflight |
| `Helper_Scripts/ci/outbound_browser_smoke.py` | Browser process supervision and attempt evidence inside the prepared image |
| `apps/tldw-frontend/playwright.outbound.config.ts` and `apps/tldw-frontend/e2e/outbound-privacy.spec.ts` | Stable selection of the shipped wizard and Next.js scenarios |
| `apps/extension/playwright.outbound.config.ts` and `apps/extension/tests/e2e/outbound-privacy.spec.ts` | Pristine manifest, live packaged worker and controlled content-script scenario |
| `apps/packages/ui/src/components/Option/ResearchWorkspace/SourcesPane/AddSourceModal.tsx` and `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage2.intake.test.tsx` | Replace remote search-result favicons and keep a regression for the local icon |
| `.github/workflows/backend-required.yml`, `.github/workflows/frontend-required.yml`, `Helper_Scripts/ci/path_classifier.py`, `tldw_Server_API/tests/CI/test_path_classifier.py`, `test_license_first_workflow_contracts.py`, `Docs/Development/Local-CI.md` | Required context wiring, trigger tests and local operating instructions |

The stable entry point is `python -m Helper_Scripts.ci.outbound_check {backend,browser,all} --report-dir DIR` from the repo root with the project environment activated. Prepare the selected images first with `python -m Helper_Scripts.ci.outbound_isolation prepare {backend,browser,all} --report-dir DIR`. CI calls the same backend and browser selectors that `all` invokes. Each selected scenario emits exactly one terminal record with `scenario`, `status` (`pass` or `fail`), `coverage`, and an array of already-redacted `violations`. Each violation has a component, stable reason code and optional public declaration ID or repository source hint. Reports are created in a fresh subdirectory per invocation. A fixture transport verification is evidence of destination selection.

Backend scenario IDs are `inventory`, `backend-default`, and `backend-opt-in`. Browser scenario IDs are `shipped-setup`, `production-webui`, `extension-worker`, and `extension-content-script`. The runner maps executed test evidence to these IDs explicitly and reports container isolation and observer coverage alongside the browser results.

## Stage 1 · Inventory and drift checks

- **Goal:** Supply a small reviewed inventory with checks against the actual resource selection used in the targeted flows.
- **Success criteria:** Missing, duplicate, malformed or silently changed declarations fail. Declarations and transport-backed evidence have distinct meanings.
- **Tests:** `test_outbound_inventory.py` and the existing direct HTTP guard.
- **Status:** Complete
- **Verification:** The new suite failed at collection without the validator, then passed with the existing direct HTTP guard (73 tests). Bandit and the whitespace check passed.

- [x] Add `test_outbound_inventory.py` with a minimal schema fixture and cases for unknown keys, duplicate capability/asset keys, missing enabling condition, default-port normalization, different schemes and ports, userinfo/path/query in an origin declaration, and a changed asset under an unchanged ID and origin. The following is the core comparison contract:

  ```python
  key = ("setup.kokoro", "kokoro.onnx")
  expected = AssetIdentity(
      source="huggingface", repository="onnx-community/Kokoro-82M-v1.0-ONNX-timestamped",
      resource="onnx/model.onnx", revision=None,
  )
  assert inventory.require_asset(*key, expected).evidence == "declared"
  with pytest.raises(InventoryError, match="asset_identity_mismatch"):
      inventory.require_asset(*key, dataclasses.replace(expected, resource="onnx/other.onnx"))
  ```

- [x] Run `python -m pytest -q tldw_Server_API/tests/CI/test_outbound_inventory.py`. Expect the new API to be absent or a seeded mismatch to be accepted before implementation.
- [x] Implement `Inventory.load(path: Path) -> Inventory`, `require_asset(capability: str, asset_id: str, actual: AssetIdentity) -> AssetEntry`, `origin_only(request_url: str) -> str`, and `check_origin(capability: str, request_url: str) -> None` in `Helper_Scripts/ci/outbound_inventory.py`. Use `json`, immutable typed records, and one origin normalization rule matching ADR-026 for HTTP(S). Reject userinfo in requests and any non-root path, query or fragment in declarations. Compare scheme, IDNA host and effective port. Use stable `InventoryError` codes `undeclared_origin`, `undeclared_asset`, and `asset_identity_mismatch` without echoing raw URLs. No DNS call belongs in this module.
- [x] Seed the manifest with the targeted Kokoro ONNX file and documented source origin, installer package/Git declarations for the covered setup catalog, the OpenAI fixture's built-in provider origin, and relevant app browser capabilities. Record owning source/component, purpose, phase and enabling condition. Represent user-directed capabilities separately from fixed origin entries. Exercise pip/uv dispatch inputs without execution in Stage 3. Keep declarations for SDK-selected, runtime and conditional paths at their actual evidence level. Test reviewed exceptions as exact named call-site entries with owner and follow-up, separate from destination permission. A synthetic test redirect origin belongs to a fixture inventory and never to the shipped declarations.

  ```json
  {
    "version": 1,
    "capabilities": [{
      "id": "setup.kokoro",
      "kind": "builtin",
      "component": "setup-installer",
      "source_path": "tldw_Server_API/app/core/Setup/install_manager.py",
      "purpose": "Download the selected Kokoro ONNX model",
      "phase": "setup",
      "enabled_when": "An operator opts in to installing the Kokoro ONNX variant",
      "origins": ["https://huggingface.co"],
      "assets": [{
        "id": "kokoro.onnx",
        "identity": {
          "source": "huggingface",
          "repository": "onnx-community/Kokoro-82M-v1.0-ONNX-timestamped",
          "resource": "onnx/model.onnx",
          "revision": null
        },
        "evidence": "declared"
      }]
    }],
    "exceptions": []
  }
  ```
- [x] Use small catalog fixtures and exact source anchors for inventory-only tests. Define `AssetIdentity` once for Stage 3 to pass actual installer selections into `require_asset`. Reuse the existing HTTP guard and report source additions or exception changes. Do not make a repository-wide URL literal scan the required proof of coverage.
- [x] Verify the new suite and `tldw_Server_API/tests/lint/test_http_usage_guard.py`. Run Bandit on the new Python files. Commit as `test: validate outbound destination and asset declarations (TASK-15634)`.

## Stage 2 · Guarded fresh backend and local setup

- **Goal:** Exercise the production ASGI app and guarded WebSocket protocol with temporary state, validating startup, setup and a representative local runtime action within the guarded boundary.
- **Success criteria:** A named fixture works, accepted DNS/dial endpoints are exact, and caught or shutdown-time violations fail the run.
- **Tests:** `test_outbound_backend_smoke.py`, existing no-live-network fixture tests, and the child smoke.
- **Status:** Complete
- **Verification:** The real Linux container passed the host-socket and loopback controls and the guarded production startup, setup, local notes, and ASGI shutdown. The strict guard and isolation tests passed (39 tests), and the existing pytest guard suite passed (27 tests and one expected subprocess-only skip).

- [x] Add strict-policy tests for numeric loopback on a listed port, another loopback port, an external numeric address, a synthetic external name already declared in the product inventory, a safe static alias, and an ambient machine/test DB hostname. Use `unittest.mock` around the real resolver or dial seam where needed, with no live network. Add child tests for normal stop with a final marker, a caught synthetic DNS failure, and a violation emitted in ASGI shutdown. Require the latter two to fail even if the child exits zero. A missing marker after forced termination also fails.

  ```python
  with block_live_network(fixture_endpoints=frozenset({("127.0.0.1", port)})) as attempts:
      with pytest.raises(OSError):
          socket.create_connection(("127.0.0.1", other_port), timeout=1)
  assert attempts == [("connect", "127.0.0.1")]
  ```

- [x] Run `python -m pytest -q tldw_Server_API/tests/CI/test_outbound_backend_smoke.py`. Expect the new fixture policy or child result contract to fail first.
- [x] Extend `block_live_network` with optional keyword-only `fixture_endpoints: frozenset[tuple[str, int]] | None = None` and `fixture_names: Mapping[str, str] | None = None`. Preserve existing behaviour when endpoints are omitted. In the explicit branch, match the registered numeric endpoint before a dial, resolve any registered alias locally with numeric-only resolution, and block all other lookups before system DNS. Include explicit handling of resolver calls with no service, Unix sockets and IPv6, and reject invalid or non-loopback fixture configuration. Preserve the existing `Attempts` interface for other tests. Serialize only its sanitized summary in this smoke.
- [x] Implement `prepare_project(root: Path, temp: Path) -> Path` in `outbound_isolation.py` for the host-side build context and `run_backend_smoke(root: Path, report: Path) -> int` in `outbound_backend_smoke.py` for the container. Use the tracked, non-symlink file projection pattern in `Helper_Scripts/benchmarks/workflows_websocket_load.py`, including both local Python packages and the tests. Include the newly added harness files explicitly during development. Exclude real dotenv files, keys, runtime databases, host socket files, and inherited cache contents. Copy the projected source into the image. Start a real writable fixture copy for each backend run and verify that imports, config, keys and databases resolve within it or its temporary state tree. Set `TLDW_CONFIG_FILE`, `TLDW_ENV_FILE`, `TLDW_ENV_FILE_EXCLUSIVE`, `USER_DB_BASE_DIR`, `DATABASE_URL`, `JOBS_DB_PATH` and `TLDW_CONTENT_SQLITE_PATH` to fixture paths. Set temporary child home/cache directories and synthetic auth material. Do not inherit benchmark shortcuts such as `DISABLE_HEAVY_STARTUP`, or set `TEST_MODE`, `CI` or provider-network flags.
- [x] In the smoke child, use `uvicorn.Server` with the deployed launcher's app and guarded WebSocket protocol, explicit `lifespan="on"`, one worker and no forwarded headers. Use the benchmark's stop-file monitor with the lifespan enabled. SIGTERM is reserved for timeout cleanup because Uvicorn can re-raise the signal after shutdown and skip the outer report writer. Implement `write_redacted_child_evidence(report: Path, attempts: Attempts) -> None` locally. The child enables the guard before importing the app or WebSocket protocol, then writes evidence after a normal lifespan exit:

  ```python
  async def serve_until_stop(stop_file: Path) -> None:
      from tldw_Server_API.scripts.run_server_guarded_mcp import GuardedSlidesWebSocketProtocol

      server = uvicorn.Server(uvicorn.Config(
          application, host="127.0.0.1", port=port,
          workers=1, lifespan="on", proxy_headers=False,
          ws=GuardedSlidesWebSocketProtocol, ws_per_message_deflate=False,
      ))

      async def monitor_stop() -> None:
          while not stop_file.exists():
              await asyncio.sleep(0.05)
          server.should_exit = True

      monitor = asyncio.create_task(monitor_stop())
      try:
          await server.serve()
          if not server.started or not shutdown_completed:
              raise RuntimeError("ASGI lifespan did not finish successfully")
      finally:
          monitor.cancel()
          await asyncio.gather(monitor, return_exceptions=True)

  ```

- Observe the app's ASGI `lifespan.shutdown.complete` event to set `shutdown_completed`. Register the evidence writer before app import and keep the guard active through interpreter exit callbacks.

- [x] Poll `GET /internal/ready` with a bounded deadline and assert its readiness contract. Require `/` to redirect to `/setup`, and `GET /api/v1/setup/status` to show setup is required. Save local-only `POST /api/v1/setup/first-run/ingest-defaults` with the payload below and verify the acknowledged step with `GET /api/v1/setup/first-run/state`. Use the synthetic key to create an explicitly titled note at `POST /api/v1/notes/`, then read the same ID and content back. The explicit title avoids unrelated automatic LLM title generation.

  ```json
  {"allow_local_file_ingest": false, "chunking_profile": "balanced", "metadata_mode": "automatic", "allowed_local_roots": []}
  ```

- [x] Implement `run_isolated(image_id: str, argv: Sequence[str], report_dir: Path, *, timeout: float) -> int` in `outbound_isolation.py` and the backend target of `Dockerfiles/Dockerfile.outbound-check`. Build the prepared test image before running the checks. Start fixtures and app children in the same private container with a fixed fixture port. The host runner creates and verifies the container before starting it. Reuse the repository's `--network none` container precedent and do not mount any host directories or sockets:

  ```bash
  docker create --pull never --network none --cap-drop ALL \
    --security-opt no-new-privileges --user 10001:10001 \
    --entrypoint python "$outbound_image_id" \
    -m Helper_Scripts.ci.outbound_backend_smoke --inside-container
  ```

  Provision UID 10001 and writable fixture/report directories in the test image. The inside-container command writes its final result to `/outbound-report/result.json`. Use only internally prepared image IDs and argv. Inspect for network mode `none`, no bind/volume mounts or published ports and the expected unprivileged user. Keep the container through exit so the host wrapper can extract just the single bounded sanitized report with `docker cp`, then remove that container in cleanup. Keep raw logs and traces private. Fail on timeout, failed preflight, missing evidence or nonzero exit. Never retry outside isolation.
- [x] Add a containment preflight before any required smoke. Bind a throwaway host-only filesystem Unix socket, prove a host client can reach it, and give the container the path as a string with no mount. Require the container connect to fail and require no accepted connection from that probe on the host. Repeat with a working in-container loopback fixture. Verify that no host resolver/IPC directories, agent sockets or Docker socket are mounted. Treat namespace and mount inspection as a necessary check and the synthetic fixture as a regression test for accidental host socket access.
- [x] Make HTTP calls from the scenario parent inside the container with proxies disabled and no redirect to any other origin. Default startup awaits heavy initialization. If the app schedules deferred startup, await its completion signal before writing the stop file. Require a normal child exit, a final marker written after lifespan shutdown, and zero unexpected attempts. Escalate to terminate/kill and reap only on timeout and fail that run. Validate cleanup on failure as well as success.
- [x] Run the strict guard, subprocess tests and existing fixture/launcher tests. Give the backend module a direct smoke entry point for this stage. Set `outbound_report_dir="$(mktemp -d)"`, prepare the backend target with `python -m Helper_Scripts.ci.outbound_isolation prepare backend --report-dir "$outbound_report_dir"`, then run `python -m Helper_Scripts.ci.outbound_backend_smoke --report-dir "$outbound_report_dir"`. Run Bandit on changed Python. Commit as `test: exercise startup and setup with exact local fixture rules (TASK-15634)`.

## Stage 3 · Opted-in download and provider evidence

- **Goal:** Check the advertised choice at installer dispatch and the destination selected by real download and provider code while using no live services.
- **Success criteria:** The real SDK downloads the fixture bytes through an allowed synthetic redirect. An undeclared hop or changed identity fails even if caught. The real provider adapter receives a synthetic completion.
- **Tests:** `test_outbound_opt_in.py`, `test_install_manager_dependencies.py`, and `test_openai_native_http.py`.
- **Status:** Complete
- **Verification:** The prepared Linux backend passed 77 tests with Hugging Face Hub 1.33.0 and zero skips. All 11 opt-in subprocess cases also passed with Hugging Face Hub 0.36.0 in a Python 3.12 environment with OS-level network denial. Both the configured and default OpenAI paths exercised the real adapter.

- [x] Add process-isolated tests in `tldw_Server_API/tests/CI/test_outbound_opt_in.py` for the Kokoro ONNX download, the same asset selecting an undeclared initial origin, the allowed fixture CDN responding with a second redirect to an undeclared origin, a changed file, and a swap to another repository with the same file and revision on the same host. In the negative cases assert the recorder's rejection and the expected install failure. Run both the download and provider children inside the prepared backend container with the Python guard active before app or SDK imports. Each uses zero allowed raw dials. After cleanup, independently assert zero raw DNS/dial attempts and verify the expected fixture transport events. Add a caught-attempt canary for each child that returns an otherwise valid fixture response and makes the check fail based on the recorded raw attempt.
- [x] In the prepared backend test container, run `python -m pytest -q tldw_Server_API/tests/CI/test_outbound_opt_in.py`. Expect the transport or evidence assertions to fail before the fixture is implemented. Use `run_isolated` from Stage 2 for direct container commands during development.
- [x] In the isolated fixture, call real `Setup.install_manager.execute_install_plan({"tts": [{"engine": "kokoro", "variants": ["onnx"]}]})`. Use a temporary model path and config/installer/cache state. Set `HF_HUB_DISABLE_XET=1` before SDK import, `HF_DEFAULT_REVISION` to a synthetic 40-hex revision, and `TLDW_SETUP_SKIP_PIP=1` for this download-only scenario. Check the shipped default identity with the unmodified declaration and apply the synthetic revision only to the fixture copy for the opted-in dispatch. Match repo/file/revision at the recording seam. The synthetic revision is per-test configuration, not a new shipped artifact pin.
- [x] Use `huggingface_hub.set_client_factory` and `httpx.MockTransport` where that SDK hook exists. For supported 0.x use `configure_http_backend` with a `requests.Session` and a fixture adapter mounted for HTTP(S). Clear SDK state by ending the child process. The handler records only origin and a public resource ID and rejects any unmatched request before a real transport exists. The 1.x handler has this shape (the 0.x adapter returns the corresponding `requests.Response`):

  ```python
  def respond(request: httpx.Request) -> httpx.Response:
      inventory.check_origin("setup.kokoro", str(request.url))
      observed.append((request.method, origin_only(str(request.url))))
      if request.method == "HEAD" and request.url.host == "huggingface.co":
          expected_path = f"/{expected.repository}/resolve/{expected.revision}/{expected.resource}"
          assert request.url.path == expected_path
          return httpx.Response(302, headers={
              "Location": fixture_asset_url, "X-Repo-Commit": revision,
              "ETag": '"outbound-fixture"', "X-Linked-Size": "4",
          })
      assert request.method == "GET" and str(request.url) == fixture_asset_url
      return httpx.Response(200, content=b"onnx")

  huggingface_hub.set_client_factory(lambda: httpx.Client(
      transport=httpx.MockTransport(respond), follow_redirects=True, trust_env=False,
  ))
  ```

  Read `expected` from the reviewed asset declaration with the explicit fixture revision. Compare the real dispatcher arguments through `require_asset` before passing them to the SDK. Preserve a redacted rejection event on origin or identity mismatches. In the redirect refusal case first let the SDK accept metadata from the original host, then respond to the allowed CDN GET with a redirect to `unlisted-cdn.invalid`. Observe and reject that attempted hop at the recording transport. Verify bytes on disk and a completed installer result in the successful case. Report the fixture redirect separately from production CDN coverage.
- [x] Call `gateway.perform_chat_api_call` with `api_endpoint="openai"`, synthetic credentials, model `gpt-4o-mini`, and `app_config={"openai_api": {"api_base_url": "https://api.openai.com/v1"}}`. Seed the healthy override cache with `set_llm_provider_overrides_cache_for_tests({})` and a ready model inventory as in `tldw_Server_API/tests/LLM_Calls/test_adapter_provider_inventory_boundary.py`. Use the real registry and adapter. Inject a local recording HTTPX client at `openai_adapter.http_client_factory` and verify the exact `POST https://api.openai.com/v1/chat/completions` and returned synthetic completion. Do not replace the adapter or allow real API credentials. Drive one refused destination case at the recording seam as well.
- [x] Extend installer dispatch assertions in `tldw_Server_API/tests/Setup/test_install_manager_dependencies.py`. Feed the shipped selected entries through `_ensure_requirement`/the real install plan. Use the existing pip-version probe and `_run_subprocess` stubs to record the selected pip or uv argv and any explicitly configured test index. Assert version, CPU/GPU selection, fallback and VCS handling for covered entries. Record the OmniVoice `git clone` and sidecar pip argv through `_run_checked_command` in `Helper_Scripts/TTS_Installers/install_tts_omnivoice_sidecar.py`. Compare argv resource identities to the manifest and classify the selected registry/downstream hosts as unverified when argv supplies insufficient evidence. Do not run package or Git commands here.
- [x] In the prepared backend test container, run `python -m pytest -q tldw_Server_API/tests/CI/test_outbound_inventory.py tldw_Server_API/tests/CI/test_outbound_opt_in.py tldw_Server_API/tests/Setup/test_install_manager_dependencies.py tldw_Server_API/tests/LLM_Adapters/unit/test_openai_native_http.py`. Exercise the compatible SDK versions in isolated environments when both hooks are supported. Record which version produced each result. Selecting no supported hook fails with a setup diagnostic. Run Bandit. Commit as `test: verify opted-in outbound dispatch without live services (TASK-15634)`.

## Stage 4 · Shipped browser flows and local favicon

- **Goal:** Exercise the shipped API wizard, production WebUI, and packaged Chrome extension with the network boundary active before process launch.
- **Success criteria:** Only assigned fixtures receive requests. The wizard persists a local edit, WebUI renders search results, and the MV3 worker and content script execute against local fixtures.
- **Tests:** Dedicated Playwright configs and specs, the Add Source unit regression, and harness self-tests.
- **Status:** Complete
- **Verification:** All four browser scenarios passed together in the disconnected Linux image with zero violations. The harness suite passed 18 tests, the Add Sources regression suite passed 35, and both dedicated TypeScript checks passed. The observer distinguished Chromium's failed IPv6 UDP route lookup from actual network traffic.
- **Implementation finding:** The native trace found ONNX Runtime's default telemetry during API startup. The app sets the SDK's existing opt-out before import, with explicit operator environment values taking precedence. A test-first regression checks the early default, and the contained traced backend reproduced the original attempt and then completed with zero native violations.
- **Implementation finding:** The quickstart WebUI requests the provider catalog. The shipped sample custom OpenAI endpoints previously resolved before the server evaluated the existing missing-credentials rule. The rule now runs before DNS. The WebUI fixture supplies exact loopback responders for the shipped local catalog probes and uses a separate fresh API lifecycle from the wizard.
- **Implementation choice:** The worker scenario uses the packaged copilot's chat UI. It verifies the authenticated POST, service-worker attribution, and rendered fixture reply. Each Playwright title has its own invocation and private evidence.

- [x] Add a failing unit regression in `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage2.intake.test.tsx`. Replace its current Google URL assertion with a local bundled icon assertion and preserve its result/snippet assertions. Run `cd apps/packages/ui && bunx vitest run src/components/Option/ResearchWorkspace/__tests__/AddSourceModal.stage2.intake.test.tsx`. Expect failure while the remote image exists.
- [x] In `AddSourceModal.tsx`, remove `getFaviconUrl` and the remote `img` branch. Render an existing icon from the local icon system, marked decorative. Preserve the search text and import behaviour. In the browser, load a same-origin fixture JSON search response at `POST /api/v1/research/websearch`, render its result through the Add Sources modal, and assert the local marker and the visible snippet.
- [x] Add `playwright.outbound.config.ts` in each app. Point the configs only at their new specs; disable global setup and auto-server launch, retries, and artifact tracing. Use one worker and absolute fixture IP URLs. The extension config must load the prebuilt production artifact without WXT global setup or unrelated manifest mutation. Fail if the browser, worker, or expected test is missing. The dedicated extension fixture may grant a single local API fixture origin after the pristine manifest has been checked.
- [x] Add `apps/tldw-frontend/e2e/outbound-privacy.spec.ts` with tests named `shipped setup saves a local setting` and `production WebUI renders search results without external assets`. Start a fresh guarded backend for the wizard. Assert the HTML and both shipped script responses before interacting. Use this action for persistence:

  ```ts
  await expect(page.locator("#guidedWizard")).toBeVisible()
  await page.locator("#wizardSkip").click()
  const field = page.locator('[data-section="Files"][data-key="inline_max_bytes"]')
  await field.fill("262145")
  const saved = page.waitForResponse(response =>
    response.url() === apiOrigin + "/api/v1/setup/config" &&
    response.request().method() === "POST" && response.ok())
  await page.locator("#saveChanges").click()
  await saved
  await expect(page.getByText("Configuration saved", { exact: false })).toBeVisible()
  ```

  The parent reads the temporary configuration and confirms the new integer. Use the quickstart login flow in `e2e/login.spec.ts`. Require `GET /api/v1/auth/sessions` to return 200, assert that the session cookie is HttpOnly using the Playwright context, and confirm the synthetic API key is absent from browser-readable state before testing search. Scope the search response handler to its exact local origin, path, and method. The guarded request log must include passive loads and redirected destinations. The parent monitors Next.js separately because Playwright doesn't see the upstream leg of same-origin rewrites.
- [x] Add `apps/extension/tests/e2e/outbound-privacy.spec.ts` with tests named `packaged worker uses the configured fixture` and `content script activates on a local page`. Use `build/chrome-mv3`, a fresh persistent profile and explicit `channel: "chromium"` so the MV3 worker runs in full Chromium. Install Chromium during image preparation and set the same absolute `PLAYWRIGHT_BROWSERS_PATH=/opt/outbound-chromium` for the install and isolated runner. Verify that UID 10001 can read the cache and that the worker starts with `CI` unset. Use the worker request/response pattern in `background-proxy-api.spec.ts`. Demand a real local fixture chat POST plus a returned UI marker. On a local page with one textarea and a known request baseline, assert the readiness attribute `data-tldw-copilot-popup-ready` and visible `#tldw-copilot-popup-host` using the activation pattern in `copilot-popup.spec.ts`. Retain context-level request evidence and worker attribution for these flows. Page-only listeners cannot cover the launch sequence.
- [x] Extend the test Dockerfile with the browser target and implement `run_browser_smoke(root: Path, report_dir: Path) -> int` in `outbound_browser_smoke.py`. Use Stage 2's shared container boundary for Next, the guarded backend, test fixtures, Playwright and Chromium, from launch through shutdown. Put prepared browser artifacts and toolchains in the image. Use fixed distinct fixture ports and keep the quickstart API's baked and runtime port the same. Require a real child-process trace and a worker-start canary in the exact unprivileged image and runner profile before making the check required. If the observer can't run under the normal restricted container profile, fail preflight and resolve the observer choice during implementation. Do not grant broad privileges or switch to host networking or mounts to make the test pass.
- [x] Observe early socket attempts from the process tree, including the Next and Chromium launch paths, with scoped Linux `strace -ff -e trace=connect,sendto -s 0` plus application request events for origins and redirect attribution. Check each observed attempt against its scenario's exact fixtures. Treat host-service Unix socket attempts as violations, and allow fixture IPC only by its explicit local identity. Parse solely the destination and verdict in private traces. Discard them after summarizing. Require tracer exit and coverage markers for the fixture trees. Record the observed socket operations and scope unverified native paths explicitly. Calibrate with a good fixture connection and a child that catches an attempted connection to an unlisted loopback port. Include a passive asset/worker observer tripwire using a safe synthetic origin. Give browser-owned traffic its own attribution where verifiable; that label grants no external access.
- [x] Prepare and stage prebuilt artifacts before the run using the commands in Stage 5. In the isolated browser runner, invoke `bunx --no-install playwright test --config playwright.outbound.config.ts --workers=1 --retries=0 --reporter=json` once from each app and save results to separate private files. Verify Bun uses the preinstalled Playwright dependency. Confirm all four named tests execute, each trace is finalized, and no unexpected connection is present. Run the Add Source unit test and Bandit on the Python harness. Commit as `test: cover app browser egress in guarded smoke (TASK-15634)`.

## Stage 5 · Required CI gates and local command

- **Goal:** Make the identical documented command drive each required CI slice and fail when expected evidence is absent.
- **Success criteria:** Relevant PR paths trigger the checks in current protected contexts, deliberate violations fail, and the artifacts state the proved coverage.
- **Tests:** `test_outbound_check.py`, path classifier and workflow contract tests, and end-to-end runner verification.
- **Status:** Complete
- **Verification:** Both images built from the final checkout, and `all` passed all seven scenarios. The final independent selectors passed in 93 seconds for backend and 86 seconds for browser. Workflow contracts propagated a deliberate failure, and missing and stale preparation records failed closed. Focused test suites, actionlint, report redaction, Bandit comparison, and whitespace checks passed. GitHub-hosted timing remains to be measured on the first branch run.

- [x] Write `test_outbound_check.py` with cases for missing, duplicate, failed, skipped and malformed scenario records, runner startup failure, and one valid terminal result per selected scenario. Check that a transport exception containing a URL, query and synthetic credential cannot put those values in the report. Test backend/browser/all selection and propagation of nonzero child status. Add classifier and workflow tests before the wiring, including a change contained to each projected package:

  ```python
  flags = classify_paths(["Helper_Scripts/ci/outbound_inventory.json"])
  assert flags["backend_changed"]
  assert flags["tldw_frontend_changed"]

  for path in (
      "apps/mcp-unified/src/mcp_unified/federation/http_transport.py",
      "packages/tldw_profile_core/src/tldw_profile_core/__init__.py",
  ):
      flags = classify_paths([path])
      assert flags["backend_changed"]
      assert flags["tldw_frontend_changed"]
  ```

- [x] Run `python -m pytest -q tldw_Server_API/tests/CI/test_outbound_check.py tldw_Server_API/tests/CI/test_path_classifier.py tldw_Server_API/tests/CI/test_license_first_workflow_contracts.py`. Expect the missing runner/trigger assertions to fail first.
- [x] Implement `main(argv: list[str] | None = None) -> int` in `outbound_check.py`. The backend selector runs `test_outbound_inventory.py`, `test_http_usage_guard.py`, `test_outbound_backend_smoke.py`, `test_outbound_opt_in.py`, `test_install_manager_dependencies.py` and `test_outbound_check.py` and invokes the guarded default backend child. Execute live app and opt-in fixture work inside the prepared backend container. Test host-side runner orchestration with stubbed children. Validate pytest's expected executed results and reject required skips. The browser selector consumes the prepared browser image, invokes the browser runner and validates both Playwright JSON files against the four named tests before emitting sanitized evidence. Use the existing `apps/tldw-frontend/scripts/assert-playwright-no-skips.mjs` and require exact scenario names and positive execution counts. The public command exits nonzero for any selected failure, missing test, missing marker, or capture failure.
- [x] Add `Helper_Scripts/ci/outbound_*`, `Dockerfiles/Dockerfile.outbound-check`, app-invoked installer paths used by the checks, `apps/mcp-unified/src/**`, and `packages/tldw_profile_core/src/**` to `BACKEND_GLOBS`. Add the same harness, Dockerfile and two package source paths to `TLDW_FRONTEND_GLOBS`. The browser smoke starts the same app. Include its launcher, control plane, shipped wizard HTML and JS/CSS, setup endpoint, setup CSP, relevant default config, and build inputs. Shared UI and extension paths already match. Verify package-only, manifest-only, setup-only, installer-only, frontend-only and unrelated docs changes. Reuse these classifier outputs so the new steps interact correctly with the existing no-op and license paths.
- [x] Add backend image preparation and smoke after required-job admission and dependency setup in `.github/workflows/backend-required.yml` using the normal backend path flag. Put matching browser steps in `.github/workflows/frontend-required.yml` under the main frontend flag. Build the standalone Next and extension artifacts into the browser image using the existing build recipes. Keep preparation and guarded run as separate bounded steps. Preserve admitted head checkout and the required summary job. Contract tests must check that the runner is after license admission and an expected failure reaches the protected context.

  ```yaml
  - name: Prepare backend outbound privacy image
    if: needs.changes.outputs.backend_changed == 'true'
    run: python -m Helper_Scripts.ci.outbound_isolation prepare backend --report-dir "$RUNNER_TEMP/outbound-privacy"
  - name: Verify backend outbound privacy
    if: needs.changes.outputs.backend_changed == 'true'
    timeout-minutes: 5
    run: python -m Helper_Scripts.ci.outbound_check backend --report-dir "$RUNNER_TEMP/outbound-privacy"
  - name: Prepare browser outbound privacy image
    if: needs.changes.outputs.tldw_frontend_changed == 'true'
    run: python -m Helper_Scripts.ci.outbound_isolation prepare browser --report-dir "$RUNNER_TEMP/outbound-privacy"
  - name: Verify browser outbound privacy
    if: needs.changes.outputs.tldw_frontend_changed == 'true'
    timeout-minutes: 8
    run: python -m Helper_Scripts.ci.outbound_check browser --report-dir "$RUNNER_TEMP/outbound-privacy"
  ```

  Place only the corresponding steps in each workflow. Keep the prepared image IDs alongside the host-side run configuration. Use the runner's fresh validated report subdirectory for summary or upload so older artifacts can't be treated as current evidence. Measure preparation and smoke runtimes in CI before enforcing the gate.
- [x] Implement `prepare` in `outbound_isolation.py`. Build the selected targets from the projected current checkout. Record each resulting image ID and source fingerprint separately from test evidence, and reject a missing or stale preparation record. Reuse the Python 3.12 dev setup for the backend target and the repo's Node/Bun and lockfile setup for the browser target. Include the supported Hugging Face SDK in the backend image and `strace` in the browser target. Run the following existing build operations in the browser image preparation steps, outside the guarded run:

  ```bash
  (cd apps && bun install --frozen-lockfile)
  (cd apps/tldw-frontend &&
    PLAYWRIGHT_BROWSERS_PATH=/opt/outbound-chromium bunx --no-install playwright install --with-deps chromium)
  (cd apps/extension && bun run build:chrome:prod)
  (cd apps/tldw-frontend &&
    NEXT_TELEMETRY_DISABLED=1 NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=quickstart \
    NEXT_PUBLIC_API_URL='' TLDW_INTERNAL_API_ORIGIN=http://127.0.0.1:18081 bun run build:prod &&
    mkdir -p .next/standalone/apps/tldw-frontend/{public,.next/static} &&
    cp -R public/. .next/standalone/apps/tldw-frontend/public/ &&
    cp -R .next/static/. .next/standalone/apps/tldw-frontend/.next/static/)
  ```

  Assign `18081` to the API, `18082` to the standalone WebUI, `18083` to the extension API fixture and `18084` to the extension fixture page. Start Next as a new production process with `HOSTNAME=127.0.0.1`, `PORT=18082`, `NEXT_TELEMETRY_DISABLED=1`, `NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=quickstart`, `AUTH_MODE=single_user`, `TLDW_WEBUI_EXPOSE_RUNTIME_AUTH=1`, `TLDW_INTERNAL_API_ORIGIN=http://127.0.0.1:18081`, and the same synthetic `SINGLE_USER_API_KEY` as the isolated API. Keep a consistent session cookie name. Pass these to the Node child at runtime and keep the key out of public variables, builds and reports. Clear ambient auth and public backend settings. Use the staged standalone and packaged Chrome locations. The browser runner explicitly receives the same absolute `PLAYWRIGHT_BROWSERS_PATH` as preparation.
- [x] Update `Docs/Development/Local-CI.md` with the commands, Linux Docker prerequisites, meaning of the `backend` and `browser` slices, and how to read redacted reports. For a full local run after activating the project venv:

  ```bash
  outbound_report_dir="$(mktemp -d)"
  python -m Helper_Scripts.ci.outbound_isolation prepare all --report-dir "$outbound_report_dir"
  python -m Helper_Scripts.ci.outbound_check all --report-dir "$outbound_report_dir"
  ```

  Document the portable pure unit tests and the option to use a Linux container engine or VM on macOS. Fixture verification checks selected URLs and exercised flows. Report other app, browser, platform and native paths as outside the proved coverage.
- [x] Run both CI commands and `all` locally in the supported environment. Run the negative sentinel and a manifest-only change through the workflow contracts, verify actionlint and focused tests, and run `python -m bandit -r Helper_Scripts/ci/outbound_inventory.py Helper_Scripts/ci/outbound_backend_smoke.py Helper_Scripts/ci/outbound_browser_smoke.py Helper_Scripts/ci/outbound_isolation.py Helper_Scripts/ci/outbound_check.py tldw_Server_API/tests/helpers/offline_network.py`. Verify no report contains the raw synthetic secrets. Record evidence and any deliberate scope exclusions in TASK-15634. Commit as `ci: require the outbound privacy regression checks (TASK-15634)`.

## Pre-push review follow-up

ADR check — proposed ADR-097 still governs the inventory and required checks. These fixes implement that decision and require no new ADR.

| Stage | Success criteria | Verification | Status |
| --- | --- | --- | --- |
| 1. CI selection and runner capacity | Auth and other covered API changes select browser privacy. Both smokes use fresh admitted runners and their existing required checks report failures. CI records cold build time and available disk. | 255 classifier and workflow contract tests passed, including failure propagation. Actionlint and Ruff passed. Hosted cold-build measurements are pending. | Complete |
| 2. Worker WebSockets | The packaged worker's hostname-based WebSocket attempt is observed before resolution. A positive fixture run stays green and its synthetic denied control is observed. | 45 focused tests passed. Offline Chromium passed the content script and three clean worker runs, and rejected an injected worker startup WebSocket. | Complete |
| 3. Diagnostics and exception anchors | Reports identify the failing check with stable redacted data. Exception anchors resolve in the shipped source. | Negative evidence and inventory tests, both local selectors, and Bandit | In Progress |

## Handoff

Before marking TASK-15634 complete, confirm the four Backlog acceptance criteria against executed evidence. Stage 1 validates declarations, Stages 2 and 3 provide default and opted-in backend evidence, Stage 4 covers all three browser surfaces, and Stage 5 enforces the same runner in the required CI contexts. Capture transport or platform expansions as separate follow-up tasks. The implementation plan deliberately carries no strict-mode release claim.
