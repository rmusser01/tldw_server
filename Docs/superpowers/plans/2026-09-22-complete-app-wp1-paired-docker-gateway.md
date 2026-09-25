# Paired Docker distribution and runtime gateway implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a locally qualified, versioned Docker bundle that starts a matched backend and full WebUI through one secure gateway using only Docker/Compose on the host, while defining the release contract later native packages consume.

**Architecture:** Keep Next standalone and the FastAPI server as private services behind a separate Node gateway. A signed manifest fixes the paired image digests and compatibility metadata. A small control entry point in a pinned image verifies the bundle and initializes persistent configuration before host scripts ask Compose to start it. The same gateway and managed WebUI artifact are the native distribution inputs for WP3; update and restore machinery belongs to WP4.

**Tech Stack:** Next.js pages router, shared TypeScript UI, Node 24 LTS, `http-proxy-middleware` pinned in `apps/bun.lock`, FastAPI/Python 3.12, Docker Compose, Vitest/Node test runner, pytest, Playwright, `cryptography` Ed25519.

**Spec:** `Docs/Design/2026-09-20-complete-app-distribution-design.md` (TASK-13265). **Task:** TASK-13343. **Review:** `Docs/Design/2026-09-20-complete-app-distribution-review.md`.

## Global constraints

- Preserve the public `tldw-server` API-only distribution, developer quickstart, and hosted/advanced WebUI modes.
- The managed product uses one stable loopback browser origin, private backend/Next listeners, runtime-selected upstreams, and no browser-visible master API key.
- The first qualified Docker targets are `linux/amd64` and `linux/arm64`; no host Python, Node, Git, Make, or compiler may be needed.
- Use supported Node 24 LTS and Python 3.12 families; freeze exact tested patches and image digests in the manifest.
- Core Docker operation must work without external Redis/PostgreSQL; avoid silently inheriting the current broad Compose file.
- The first Docker WebUI Update action describes the authenticated host helper; it cannot operate the Docker socket from the browser.
- The frontend publication freeze in `Docs/Development/Packaging_and_Distribution_Strategy.md` remains in force. Use local images and an ephemeral registry inside a CI job for tests; do not push protected images or bundles to an external registry or release service before the separate gate.
- Run `source .venv/bin/activate` before Python, pip, pytest, or Bandit. Keep unrelated workspace edits out of commits.
- Each code task uses a failing behavioral test, a passing focused test, relevant regression checks, scoped lint/Bandit, self-review, and a commit linked to TASK-13343.

## File and interface map

| Unit | Files and responsibility |
| --- | --- |
| Release contract | `tldw_Server_API/app/core/Release/manifest.py` parses/verifies exact manifest bytes, signature, platform, compatibility, and artifact digests using only the standard library and `cryptography`; `tldw_Server_API/tests/Release/test_manifest.py` covers tampering/keys/platforms. `tldw_Server_API/scripts/app_bundle_control.py` is its narrow container CLI. `Dockerfiles/app-bundle/manifest.schema.json` documents canonical fields. WP3's launcher build must package this same source under its own namespace and prove its hash matches; the launcher must not import `tldw_Server_API`. |
| Managed WebUI mode | `apps/tldw-frontend/scripts/validate-networking-config.mjs`, `next.config.mjs`, `pages/api/_tldw-webui/runtime-auth-policy.ts`, `runtime-config.ts`, and shared `apps/packages/ui/src/services/tldw/browser-networking.ts` plus their existing tests. Managed browser requests stay same origin; the private backend origin is read by server code at runtime. |
| Instance cookie names | `tldw_Server_API/app/core/AuthNZ/settings.py` and `csrf_protection.py` supply a configurable CSRF cookie name; Next `session.ts` filters the configured names; shared/browser request call sites read one runtime cookie-name accessor. Tests cover two instances on one host. |
| Gateway | `apps/tldw-frontend/gateway/routes.mjs` classifies destinations; `gateway/server.mjs` validates Host/Origin, strips untrusted forwarding/control headers, injects one private gateway hop token to Next, proxies HTTP/upgrade, and serves a bounded status/maintenance response; `gateway/__tests__/*.test.mjs` exercises real local upstreams. |
| Docker release | `Dockerfiles/Dockerfile.webui` supports the managed build with Node 24; new `Dockerfiles/Dockerfile.gateway` and a one-shot control target/image package the gateway/control code; `Dockerfiles/app-bundle/compose.yaml`, `start.sh`, `start.ps1`, `stop.sh`, `stop.ps1`, `status.sh`, `status.ps1`, and `README.md` are the extractable bundle. |
| Qualification | `Helper_Scripts/verify_app_bundle.py` builds/checks a local candidate from a clean tree; `tldw_Server_API/tests/Release/test_app_bundle_control.py`, gateway tests, WebUI Vitest/Playwright, and a Docker smoke script exercise the exact extracted bundle. CI runs the same commands using an ephemeral job-local registry, then records G2/G4/G10 evidence without publishing protected artifacts. |

The code paths above are the planned ownership boundaries. If an existing helper already supplies a behavior, extend it and remove the redundant proposed file in the plan before implementation. Do not introduce a second manifest parser or browser networking policy.

## Stage 1: Release contract and managed WebUI

**Goal:** Define verified pairing and build a standalone WebUI whose browser API URLs do not depend on private build-time ports.
**Success Criteria:** A signed test manifest rejects tampering and wrong platform; the managed Next build contains no private backend rewrite or public key; one build works against two private backend origins.
**Tests:** Manifest pytest; networking and Next config tests; runtime-config/session Vitest; standalone asset inspection.
**Status:** Complete

### Task 1: Define and verify the paired release manifest

**Files:** Create `tldw_Server_API/app/core/Release/__init__.py`, `manifest.py`, `tldw_Server_API/tests/Release/test_manifest.py`, and `Dockerfiles/app-bundle/manifest.schema.json`. Keep signing in the CI/release helper, never in the runtime verifier.

**Interfaces:** `verify_manifest(manifest_bytes: bytes, signature_bytes: bytes, trusted_keys: Mapping[str, bytes], *, platform: str, current_version: str | None = None) -> ReleaseManifest`; `verify_artifact(path: Path, artifact: Artifact) -> None`. `ReleaseManifest` exposes typed `version`, `source_commit`, `platforms`, `artifacts`, `compatibility`, and `qualifications` fields. Later control/helper tasks consume these exact interfaces.

- [x] **Step 1: Write failing tests** for a valid Ed25519 fixture and for one-byte tampering, unknown signer, duplicate artifact IDs, wrong `linux/arm64` tuple, missing role/digest/size, and an escaping artifact path. Example:

  ```python
  def test_wrong_platform_is_rejected(signed_manifest, trusted_keys):
      with pytest.raises(ManifestError, match="platform"):
          verify_manifest(*signed_manifest, trusted_keys, platform="linux/arm64")
  ```

- [x] **Step 2: Run** `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Release/test_manifest.py -q`; expect the import/test to fail before adding implementation.
- [x] **Step 3: Implement** strict JSON field/type validation, exact-byte signature verification via `cryptography.hazmat.primitives.asymmetric.ed25519`, trusted key lookup by signer ID, platform/upgrade checks, SHA-256 streaming verification, and rejection of non-canonical paths. Schema requires all identity, compatibility, artifact, dependency/data, and qualification field groups from spec section 11. It must reject unknown schema versions and duplicate JSON keys.

  ```python
  public_key = Ed25519PublicKey.from_public_bytes(trusted_keys[signer_id])
  public_key.verify(signature_bytes, manifest_bytes)
  return _manifest(parsed, platform=platform, current_version=current_version)
  ```

- [x] **Step 4: Run** the same pytest file; expect pass. Then run `python -m bandit -r tldw_Server_API/app/core/Release -f json -o /tmp/bandit_task13343_manifest.json` in the activated venv and inspect findings.
- [x] **Step 5: Commit** only these files and the current TASK-13343/plan progress with `feat: verify paired application release manifests (TASK-13343)`.

### Task 2: Add a managed same-origin WebUI build mode

**Files:** Modify `apps/tldw-frontend/scripts/validate-networking-config.mjs`, `next.config.mjs`, `apps/packages/ui/src/services/tldw/browser-networking.ts`, `apps/packages/ui/src/services/tldw/direct-browser-config.ts`, and targeted tests in `apps/tldw-frontend/__tests__/next-config-quickstart-health.test.ts` and `apps/packages/ui/src/services/tldw/__tests__/browser-networking.test.ts`. Inspect all callers of `resolveBrowserTransportMode` and the quickstart cookie setup before editing.

**Interfaces:** Build marker `NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=managed`; runtime-only `TLDW_INTERNAL_API_ORIGIN`; browser transport remains the existing `BrowserTransport` shape with `mode: "quickstart"` for same-origin semantics. The explicit managed build marker distinguishes packaging and Next rewrites from developer quickstart. `validateNetworkingConfig(env)` returns `deploymentMode`, `internalApiOrigin`, `publicApiUrl` unchanged in shape.

- [x] **Step 1: Write failing tests** that managed mode accepts an empty `NEXT_PUBLIC_API_URL` without a build-time backend URL, `next.config.mjs` returns `rewrites() === []`, and browser HTTP/WebSocket requests resolve to the page origin. Preserve quickstart rewrite and advanced absolute-origin tests. Example:

  ```ts
  expect(resolveBrowserTransport({surface: "webui-page", deploymentMode: "managed", pageOrigin: "http://127.0.0.1:8080"})).toMatchObject({mode: "quickstart", apiOrigin: ""})
  ```

- [x] **Step 2: Run** `bun run --cwd apps/tldw-frontend test:run -- __tests__/next-config-quickstart-health.test.ts` and the shared browser-networking test through the workspace Vitest command; expect the new managed assertions to fail.
- [x] **Step 3: Implement** managed build validation and no rewrites, map managed browser transport to same origin, and keep the private origin out of `NEXT_PUBLIC_*`. Audit `connection.tsx`, `TldwApiClient.ts`, `browser-websocket.ts`, and `direct-browser-config.ts` for literal `quickstart` comparisons; cover each behavior in a focused regression test. Do not change extension or hosted mode behavior.

  ```js
  if (deploymentMode === 'managed') return [];
  // Runtime-only TLDW_INTERNAL_API_ORIGIN remains read by Next API routes.
  ```

- [x] **Step 4: Run** both targeted suites, existing quickstart runtime-config/session suites, and `bun run --cwd apps/tldw-frontend typecheck`. Build once with `NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=managed` and inspect generated client files for a fixture private origin/master key; expect neither to appear. The managed build and 64 focused tests passed; full typecheck reports 93 existing diagnostics in untouched files, recorded as a WP1 qualification issue.
- [x] **Step 5: Commit** the targeted WebUI/shared-code changes with `feat: add managed same-origin WebUI mode (TASK-13343)`.

## Stage 2: Browser/session boundary and gateway

**Goal:** Make the single public origin secure across multiple local installations and functional for HTTP and WebSockets.
**Success Criteria:** Runtime session exchange works through the gateway, instance cookies do not collide on one host, and the route/auth matrix passes against fake and real upstreams.
**Tests:** Backend CSRF tests; Next API Vitest; gateway Node tests; Playwright cookie lifecycle, uploads, streaming, WebSockets.
**Status:** In Progress

### Task 3: Scope session and CSRF cookies to one managed instance

**Files:** Modify `tldw_Server_API/app/core/AuthNZ/settings.py`, `csrf_protection.py`, and `tldw_Server_API/app/api/v1/endpoints/persona.py` where it bridges cookie-based WebSockets; Next `pages/api/_tldw-webui/runtime-auth-policy.ts`, `runtime-config.ts`, `session.ts`; shared `apps/packages/ui/src/services/tldw/request-core.ts`, `runtime-auth-override.ts`, `apps/packages/ui/src/services/acp/client.ts`, `recipe-persistence-uncertainty.ts`, `background-proxy.ts`; and `apps/tldw-frontend/lib/api.ts`. Test in `tldw_Server_API/tests/AuthNZ/unit/test_csrf_binding.py`, `integration/test_single_user_cookie_session.py`, `apps/tldw-frontend/__tests__/pages/api/runtime-session.test.ts`, `runtime-config.test.ts`, and focused shared UI tests.

**Interfaces:** Backend setting `CSRF_COOKIE_NAME` defaults to `csrf_token`; `SINGLE_USER_SESSION_COOKIE_NAME` retains its existing setting. Managed configuration generates distinct validated names once and supplies both to backend and Next. `runtime-config` returns a public `csrfCookieName` only when runtime auth is available. Shared `setRuntimeCsrfCookieName(value: string | null): void` and `getRuntimeCsrfCookieName(): string` hold a validated name for WebUI requests and default to `csrf_token` until set. The extension has a separate bundle/runtime and keeps its legacy default. No API key appears in the response.

- [x] **Step 1: Write failing tests** for backend CSRF mint/validate/clear with two custom cookie names, Next session forwarding only the configured pair, public runtime config excluding the master key, and browser request helpers reading the configured CSRF name. Example:

  ```ts
  setRuntimeCsrfCookieName("tldw_csrf_a1")
  expect(getRuntimeCsrfCookieName()).toBe("tldw_csrf_a1")
  setRuntimeCsrfCookieName(null)
  expect(getRuntimeCsrfCookieName()).toBe("csrf_token")
  ```

- [x] **Step 2: Run** the targeted backend pytest and WebUI Vitest files; expect failures at the new configuration seam.
- [x] **Step 3: Add** validated backend setting and replace fixed cookie identifiers in the CSRF manager/getters/WebSocket auth where applicable. Add the browser accessor and replace the fixed cookie reads. Have `runtime-config` expose only the public cookie identifier and initialize the accessor before authenticated API traffic; `session.ts` allows exactly the configured cookie names. Keep the legacy default in developer/extension/hosted modes.

  ```python
  token_cookie_name = get_settings().CSRF_COOKIE_NAME
  # Validate with the existing HTTP cookie-token pattern and reject session-name equality.
  ```

- [x] **Step 4: Run** focused suites plus existing auth/session regression tests. Exercise two app origins on `127.0.0.1` with different ports and cookie names; assert neither app accepts the other's session/CSRF pair. Run scoped Python Bandit and frontend lint on touched files. Backend unit (32), custom HTTP integration and logout regression (6), Next runtime/session (179), and affected service tests (219) pass; gateway-level two-instance browser exercise remains in Task 4 after the gateway exists. Typecheck has 93 existing unrelated diagnostics; touched Python files were already outside Black format at HEAD. Bandit finds the same 11 existing B106 findings in `auth.py` as HEAD and no new findings.
- [x] **Step 5: Commit** the backend/Next/shared UI cookie changes with `fix: isolate managed application session cookies (TASK-13343)`.

### Task 4: Implement the runtime gateway and trust boundary

**Files:** Create `apps/tldw-frontend/gateway/routes.mjs`, `server.mjs`, `gateway/__tests__/routes.test.mjs`, `gateway/__tests__/server.test.mjs`. Add pinned `http-proxy-middleware` as a runtime dependency in `apps/tldw-frontend/package.json` and update `apps/bun.lock`. Modify Next `runtime-auth-policy.ts` to accept only an authenticated gateway hop in managed mode while preserving current quickstart checks.

**Interfaces:** `routeForPath(pathname: string): "managed" | "next" | "backend"`; `authorizeRequest(req: http.IncomingMessage): boolean`; `createGateway({backendOrigin, nextOrigin, publicHost, publicPort, gatewayHopSecret, phase}): http.Server`. Only the gateway supplies `X-Tldw-Gateway-Hop` to private Next; the browser cannot set it through. `/_tldw/*` serves read-only status/maintenance and the exact Docker host-helper instruction. Mutating control requests remain unavailable until WP3/WP4.

- [x] **Step 1: Write failing Node tests** for the complete spec section 5 route table, path/query preservation, forbidden unknown Host/Origin/forwarded headers, stripped browser-supplied hop headers, stable Set-Cookie/redirect handling, multipart streaming, request cancellation, and WebSocket upgrade. Example:

  ```js
  assert.equal(routeForPath('/api/_tldw-webui/session'), 'next');
  assert.equal(routeForPath('/api/v1/chat/completions'), 'backend');
  assert.equal(routeForPath('/_next/static/app.js'), 'next');
  ```

- [x] **Step 2: Run** `node --test apps/tldw-frontend/gateway/__tests__/*.test.mjs`; expect missing-module failures.
- [x] **Step 3: Implement** explicit routing and an `http-proxy-middleware` proxy for each private upstream. Reject host/Origin mismatches before HTTP proxying or WebSocket upgrade, remove client `Forwarded`, `X-Forwarded-*`, and hop headers, and synthesize only the canonical headers Next/backend need. Use the library's documented `upgrade` path for WebSockets and do not buffer request bodies. The Next managed auth policy checks the hop secret in constant time and a non-public listener/network; no browser request may carry master credentials into arbitrary backend routes.

  ```js
  server.on('upgrade', (req, socket, head) => {
    if (!authorizeRequest(req)) return socket.destroy();
    const target = routeForPath(new URL(req.url, 'http://gateway').pathname);
    proxyFor(target).upgrade(req, socket, head);
  });
  ```

- [x] **Step 4: Run** Node tests and Next runtime-auth tests. Start dummy upstreams twice on different private ports with one built WebUI/gateway; verify each public request reaches the currently configured port. Run Playwright/HTTP probes for auth, docs API, uploads, streaming cancellation, WebSockets, redirects, and maintenance status. Capture failures as tests before changing routing. Twelve real-socket Node checks, 145 Next runtime-auth/session tests, and a rebuilt standalone Next through-gateway cookie exchange pass. Exact extracted-bundle plus real FastAPI and two-instance browser smoke remain Task 6/7 qualification gates. Frontend typecheck retains 93 existing unrelated diagnostics.
- [x] **Step 5: Commit** gateway/lockfile/Next-policy changes with `feat: route managed WebUI through runtime gateway (TASK-13343)`.

## Stage 3: Extractable Docker release bundle

**Goal:** Boot paired, digest-pinned images and persistent credentials/data from an extracted bundle with Docker as the only host runtime.
**Success Criteria:** First start initializes once; repeat start reuses credentials/data/project identity; stop/status work from another directory; no backend/Next port is published.
**Tests:** Control pytest, Compose config validation, shell/PowerShell helper checks, fresh Docker account smoke.
**Status:** Not Started

### Task 5: Build managed Docker images and an idempotent control entry point

**Files:** Modify `Dockerfiles/Dockerfile.webui` for the managed build and supported Node 24 runtime; create `Dockerfiles/Dockerfile.gateway`, `tldw_Server_API/scripts/app_bundle_control.py`, `tldw_Server_API/tests/Release/test_app_bundle_control.py`, and a control image target with the verifier and its trusted key set. Reuse `Dockerfiles/Dockerfile.prod` as the initial backend build, adapting only managed profile behavior needed for this bundle.

**Interfaces:** `app_bundle_control init --state /state --manifest /bundle/manifest.json --signature /bundle/manifest.sig --platform linux/amd64|linux/arm64` verifies first, then creates private config if absent; `verify` checks exact image/artifact digests and compatibility without changing state. Config records stable Compose project ID, unique session/CSRF names, API key, gateway hop secret, and pinned image references. Control exits nonzero if existing state conflicts with the signed release.

- [ ] **Step 1: Write failing pytest** using a temporary state directory: first init creates mode-0600 files, second init leaves keys byte-identical, malformed signature leaves state untouched, existing data with mismatched identity is rejected, and no secret appears in stdout. The later image smoke must execute Node 24 and request a copied `public`/`static` asset from the running WebUI; inspecting Dockerfile text is insufficient.

  ```python
  first = initialize_bundle(state_dir, verified_release)
  second = initialize_bundle(state_dir, verified_release)
  assert first.api_key == second.api_key
  ```

- [ ] **Step 2: Run** `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Release/test_app_bundle_control.py -q`; expect failure.
- [ ] **Step 3: Implement** transactional initialization through a temporary directory + atomic replace, strict ownership/permissions, secret-safe messages, and refusal to overwrite valid credentials. Build WebUI/gateway/control images from the same clean source commit; include declared proxy dependency closure and Next traced assets. Use a private backend origin at runtime in Next, never a build-time Docker service URL. Do not put the control secret or API key in image layers.
- [ ] **Step 4: Run** control tests, Docker builds for both Linux architectures in CI, image-content guards, `docker image inspect`, and scoped Bandit. Verify no development database, `.env`, `node_modules` source tree, or build cache leaks into runtime images.
- [ ] **Step 5: Commit** image/control changes with `feat: build paired managed application images (TASK-13343)`.

### Task 6: Add Docker-only host helpers and Compose bundle

**Files:** Create `Dockerfiles/app-bundle/compose.yaml`, `start.sh`, `start.ps1`, `stop.sh`, `stop.ps1`, `status.sh`, `status.ps1`, and `README.md`. Add focused helper tests under `tldw_Server_API/tests/Release/` and a local extracted-bundle smoke script under `Helper_Scripts/`.

**Interfaces:** Host helpers call only Docker/Compose. The official bundle embeds a fixed control-image digest and trusted key ID; a local CI test bundle uses its own explicitly trusted test key/registry. `start` verifies the signed manifest through a one-shot pinned control image before `compose up`; configuration and project identity live in an OS user data directory outside the extracted bundle. `stop` uses the persisted project ID and retains data. No script accepts an arbitrary image URL or shell fragment from the WebUI.

- [ ] **Step 1: Write failing helper tests** with a fake `docker` executable recording arguments. Check first-start order (`control verify/init` before `compose up`), repeat start reuse, stop from a different working directory, unavailable Docker/Compose, occupied public port, bad signature, and private-only backend/Next ports. Example:

  ```python
  assert calls.index("control:verify") < calls.index("compose:up")
  assert "ports" not in compose["services"]["app"]
  ```

- [ ] **Step 2: Run** the helper pytest file; expect missing-helper failures.
- [ ] **Step 3: Implement** the digest-pinned Compose services for gateway, WebUI, backend, and one-shot control without Postgres/Redis defaults. `start` finds its bundle directory independent of the shell CWD, selects a documented state root, calls control verification/initialization, then uses `docker compose --project-name <persisted-id> --env-file <state>/config.env -f <bundle>/compose.yaml up -d`. Bind only `127.0.0.1:8080` (or the persisted first-install alternative) on the host. `stop` does not remove volumes. PowerShell follows the same contract without requiring bash.

  ```sh
  docker compose --project-name "$project_id" --env-file "$state_dir/config.env" -f "$bundle_dir/compose.yaml" up -d
  ```

- [ ] **Step 4: Run** helper tests; run `docker compose config` for both target architectures and actual first/repeat/stop/restart flows from an extracted directory outside the checkout. Assert server data and credentials persist, browser setup works, and the backend/Next are unreachable directly from the host. Add a negative test for an unrelated process already bound to 8080.
- [ ] **Step 5: Commit** bundle/helpers/docs with `feat: start paired Docker bundle from release archive (TASK-13343)`.

## Stage 4: Qualification and release boundary

**Goal:** Make WP1 evidence reproducible without treating an incomplete or frozen publication as available to users.
**Success Criteria:** CI tests the exact extractable candidate using an ephemeral job-local registry; manifest qualification refuses missing artifacts/gates; documentation states current availability accurately.
**Tests:** Fresh extracted-bundle G2; routing/auth G4; tampered/missing artifact G10; supported runtime and policy G12 checks.
**Status:** Not Started

### Task 7: Build candidate qualification and promotion gate

**Files:** Create `Helper_Scripts/build_app_bundle.py`, `Helper_Scripts/verify_app_bundle.py`, `Helper_Scripts/test_app_bundle_docker.sh`, and a focused `.github/workflows/verify-app-bundle.yml`; modify existing `publish-docker.yml` only when the separate frontend publication gate is authorized. Add `tldw_Server_API/tests/Release/test_candidate_gate.py`.

**Interfaces:** `build_app_bundle.py --artifacts <inventory.json> --evidence <evidence.json> --signing-key <path> --output <directory>` produces the exact manifest bytes and detached Ed25519 signature using an ephemeral CI test key. The private key is never packaged, logged, or checked in. `verify_app_bundle.py --manifest <path> --signature <path> --evidence <path> --platform <tuple>` exits zero only when required roles/digests and G2/G4/G10 evidence for that tuple are present and verified. Candidate registry/image refs are recorded in signed metadata; protected publishing requires a later separate authorization/gate record.

- [ ] **Step 1: Write failing tests** for absent WebUI image, mixed commits, unsupported Node patch, missing arm64 evidence, corrupted digest, and a complete local fixture. Example:

  ```python
  assert candidate_is_promotable(complete_fixture, required={"linux/amd64", "linux/arm64"})
  assert not candidate_is_promotable(missing_webui_fixture, required={"linux/amd64"})
  ```

- [ ] **Step 2: Run** the focused candidate-gate pytest; expect missing-function failure.
- [ ] **Step 3: Implement** clean-checkout builds and artifact inventory from allowlisted source paths, ephemeral job-local registry digest capture, exact-byte Ed25519 signing, per-platform signature/hash verification, and evidence aggregation. CI pulls and runs the candidate outside its checkout using Docker/Compose only for the host flow. Keep `publish-docker.yml` and external registry publication behavior unchanged until the existing release gate is satisfied.
- [ ] **Step 4: Run** focused tests plus the extracted-bundle smoke on amd64 and arm64 CI runners/emulation where qualified; record exact runtime patches and limitations. Run frontend lint/typecheck, focused backend tests, gateway tests, `git diff --check`, and Bandit on touched Python source. Failure in either required tuple blocks promotion.
- [ ] **Step 5: Commit** qualification workflow and evidence tooling with `test: qualify paired Docker candidate before release (TASK-13343)`.

### Task 8: Review WP1 against the product contract

**Files:** Update `Dockerfiles/app-bundle/README.md`, `Docs/Development/Packaging_and_Distribution_Strategy.md` only if current availability changed, and TASK-13343 notes/plan status. No product claim goes into `README.md` installation directions before release approval.

**Interfaces:** Acceptance record links the exact candidate manifest and G2/G4/G10/G12 results. WP2-WP5 consume the signed manifest schema and the gateway/managed WebUI artifact without copying implementations.

- [ ] **Step 1: Check** every WP1 requirement in spec sections 5, 10, 11, and the applicable section 12 gates against a test or a clearly recorded deferred WP4/WP5 boundary. Treat missing G4 networking scenarios as failing acceptance, not a documentation caveat.
- [ ] **Step 2: Run** full targeted frontend/backend/gateway suites and the candidate smoke once more only if Task 7 changed artifacts; otherwise cite its immutable evidence. Run `git diff --check`, scoped lint, and Bandit, and inspect the exact staged diff for secrets or unrelated files.
- [ ] **Step 3: Record** measured download/installed sizes, startup times, tested platforms, and any unsupported behavior in the candidate evidence. Mark TASK-13343 complete only when its criteria and local/CI gates pass; note that public publication remains separately gated.
- [ ] **Step 4: Commit** final task/doc/evidence updates with `docs: record paired Docker candidate qualification (TASK-13343)`.

## Plan self-review checklist

- Sections 5/10/11 and gates G2/G4/G10/G12 map to Tasks 1–8. Backend dependency slimming and complete storage inventory are WP2; native lifecycle and the single-source verifier packaging check are WP3; automatic update/backup/restore is WP4; guided heavy components and final public promotion are WP5.
- Verify exact code symbols in the interface blocks when implementing each task; preserve existing quickstart/hosted paths and test both managed and legacy modes.
- Do not claim G2/G4/G10/G12 complete until candidate artifacts pass the corresponding tests. This plan is executable work, not evidence that they already pass.
