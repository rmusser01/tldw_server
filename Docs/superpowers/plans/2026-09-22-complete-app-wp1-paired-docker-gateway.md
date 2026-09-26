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
**Status:** Complete

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
**Status:** Complete

### Task 5: Build managed Docker images and an idempotent control entry point

**Files:** Modify `Dockerfiles/Dockerfile.webui` for the managed build and supported Node 24 runtime; create `Dockerfiles/Dockerfile.gateway`, `tldw_Server_API/scripts/app_bundle_control.py`, `tldw_Server_API/tests/Release/test_app_bundle_control.py`, and a control image target with the verifier and its trusted key set. Reuse `Dockerfiles/Dockerfile.prod` as the initial backend build, adapting only managed profile behavior needed for this bundle.

**Interfaces:** `app_bundle_control init --state /state --manifest /bundle/manifest.json --signature /bundle/manifest.sig --platform linux/amd64|linux/arm64` verifies first, then creates private config if absent; `verify` checks exact image/artifact digests and compatibility without changing state. Config records stable Compose project ID, unique session/CSRF names, API key, gateway hop secret, and pinned image references. Control exits nonzero if existing state conflicts with the signed release.

- [x] **Step 1: Write failing pytest** using a temporary state directory: first init creates mode-0600 files, second init leaves keys byte-identical, malformed signature leaves state untouched, existing data with mismatched identity is rejected, and no secret appears in stdout. The later image smoke must execute Node 24 and request a copied `public`/`static` asset from the running WebUI; inspecting Dockerfile text is insufficient. Ten control tests now cover these paths plus read-only verify, required images, symlink traversal, and signed env-injection input.

  ```python
  first = initialize_bundle(state_dir, verified_release)
  second = initialize_bundle(state_dir, verified_release)
  assert first.api_key == second.api_key
  ```

- [x] **Step 2: Run** `source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Release/test_app_bundle_control.py -q`; expected collection failure observed before implementation, and env-injection regression was red before its guard.
- [x] **Step 3: Implement** transactional initialization through a temporary directory + atomic replace, strict ownership/permissions, secret-safe messages, and refusal to overwrite valid credentials. WebUI/gateway/control Dockerfiles and a frozen gateway-only dependency lock are present; the managed WebUI stage omits build-time private origins/keys and copies traced plus public/static assets. Actual image builds remain in Step 4.
- [x] **Step 4: Run** control tests, Docker builds for both Linux architectures in CI, image-content guards, `docker image inspect`, and scoped Bandit. Verify no development database, `.env`, `node_modules` source tree, or build cache leaks into runtime images. Corrected run 36210634045 at e759322854 builds all four images on each native architecture, passes the content/runtime guards and extracted first/repeat startup, and records image metadata in the acceptance review. The local missing `tldw_profile_core` regression is corrected; both platforms execute its import guard. Production Bandit has zero findings. Broader browser and joined-release qualification remain in Tasks 6–8.
- [x] **Step 5: Commit** image/control changes at `80ba4ec172` with `feat: build paired managed application images (TASK-13343)`; corrected image execution is recorded in Step 4.

### Task 6: Add Docker-only host helpers and Compose bundle

**Files:** Create `Dockerfiles/app-bundle/compose.yaml`, `start.sh`, `start.ps1`, `stop.sh`, `stop.ps1`, `status.sh`, `status.ps1`, and `README.md`. Add focused helper tests under `tldw_Server_API/tests/Release/` and a local extracted-bundle smoke script under `Helper_Scripts/`.

**Interfaces:** Host helpers call only Docker/Compose. The official bundle embeds a fixed control-image digest and trusted key ID; a local CI test bundle uses its own explicitly trusted test key/registry. `start` verifies the signed manifest through a one-shot pinned control image before `compose up`; configuration and project identity live in an OS user data directory outside the extracted bundle. `stop` uses the persisted project ID and retains data. No script accepts an arbitrary image URL or shell fragment from the WebUI.

- [x] **Step 1: Write failing helper tests** with a fake `docker` executable recording arguments. Check first-start order (`control verify/init` before `compose up`), repeat start reuse, stop from a different working directory, unavailable Docker/Compose, occupied public port, bad signature, and private-only backend/Next ports. The extracted-directory fixture runs away from the checkout and records every Docker argument.

  ```python
  assert calls.index("control:verify") < calls.index("compose:up")
  assert "ports" not in compose["services"]["app"]
  ```

- [x] **Step 2: Run** the helper pytest file; expected missing-helper failures observed before implementation.
- [x] **Step 3: Implement** the digest-pinned Compose services for gateway, WebUI, backend, and one-shot control without Postgres/Redis defaults. Source helper templates require candidate packaging to embed a fixed control-image digest and trusted key ID; `start` runs the pinned one-shot control image to verify/init, pulls signed image refs, then starts Compose with a saved project ID. Only the gateway binds host loopback. Shell and PowerShell stop/status keep volumes. The backend image now exposes a writable managed-config mount point for persistent generated keys.

  ```sh
  docker compose --project-name "$project_id" --env-file "$state_dir/config.env" -f "$bundle_dir/compose.yaml" up -d
  ```

- [x] **Step 4: Run** helper tests; run `docker compose config` for both target architectures and actual first/repeat/stop/restart flows from an extracted directory outside the checkout. Assert server data and credentials persist, browser setup works, and the backend/Next are unreachable directly from the host. Add a negative test for an unrelated process already bound to 8080. Shell/fake-Docker control-helper suite and Compose config pass locally; Current exact Linux local/native13lifecycle38browser proof passes. PowerShell parsing passes, while actual Windows runtime remains explicitly unqualified and separate from this Linux slice.
- [x] **Step 5: Commit** bundle/helpers/docs with `feat: start paired Docker bundle from release archive (TASK-13343)`; live Docker and Windows execution remain open in Step 4 until qualification CI runs.

## Stage 4: Qualification and release boundary

**Goal:** Make WP1 evidence reproducible without treating an incomplete or frozen publication as available to users.
**Success Criteria:** CI tests the exact extractable candidate using an ephemeral job-local registry; manifest qualification refuses missing artifacts/gates; documentation states current availability accurately.
**Tests:** Fresh extracted-bundle G2; routing/auth G4; tampered/missing artifact G10; supported runtime and policy G12 checks.
**Status:** Complete

### Task 7: Build candidate qualification and promotion gate

**Files:** Create `Helper_Scripts/build_app_bundle.py`, `Helper_Scripts/verify_app_bundle.py`, `Helper_Scripts/test_app_bundle_docker.sh`, and a focused `.github/workflows/verify-app-bundle.yml`; modify existing `publish-docker.yml` only when the separate frontend publication gate is authorized. Add `tldw_Server_API/tests/Release/test_candidate_gate.py`.

**Interfaces:** `build_app_bundle.py --artifacts <inventory.json> --evidence <evidence.json> --signing-key <path> --output <directory>` produces the exact manifest bytes and detached Ed25519 signature using an ephemeral CI test key. The private key is never packaged, logged, or checked in. `verify_app_bundle.py --manifest <path> --signature <path> --evidence <path> --platform <tuple>` exits zero only when required roles/digests and G2/G4/G10 evidence for that tuple are present and verified. Candidate registry/image refs are recorded in signed metadata; protected publishing requires a later separate authorization/gate record.

- [x] **Step 1: Write failing tests** for absent WebUI image, mixed commits, unsupported Node patch, missing arm64 evidence, corrupted digest, and a complete local fixture. The gate also rejects tampered helper bytes, and the manifest tests cover identical shared helper paths across platforms.

  ```python
  assert candidate_is_promotable(complete_fixture, required={"linux/amd64", "linux/arm64"})
  assert not candidate_is_promotable(missing_webui_fixture, required={"linux/amd64"})
  ```

- [x] **Step 2: Run** the focused candidate-gate pytest; expected missing-module failure observed before implementation, then unsupported-runtime regression failed before its guard.
- [x] **Step 3: Implement** clean-checkout builds and artifact inventory from allowlisted bundle files, ephemeral job-local registry digest capture, exact-byte Ed25519 signing, per-platform signature/hash verification, and a required-both-platform CI status job. Each native runner builds and smokes its own local single-platform candidate; these are provisional artifacts, not a published multi-platform manifest. The evidence keeps G2/G4/G12 false until browser setup, two-instance networking, and runtime-support checks exist; the promotion verifier must refuse that candidate. The existing `publish-docker.yml` remains unchanged.
- [x] **Step 4: Run** focused tests plus the extracted-bundle smoke on amd64 and arm64 CI runners; record exact runtime patches and limitations. Run frontend lint/typecheck, focused backend tests, gateway tests, `git diff --check`, and Bandit on touched Python source. Run 36210634045 at e759322854 passes both native extracted smokes, Windows parsing, and the required-both status. Each native job passes 49 Release, 207 frontend, and 13 gateway tests, scoped lint/format checks, shell syntax, and production Bandit (0 findings). Both run Python 3.12.14 and Node 24.21.0; downloaded signatures and helper hashes independently verify. The acceptance review records immutable manifest hashes, Compose first/repeat health timings, and Docker image metadata; actual download/installed footprints remain outstanding. The whole-frontend typecheck baseline remains 93 untouched-file diagnostics and is recorded without counting it as success. The lane emits provisional evidence and refuses promotion while G2/G4/G12 remain false.
- [x] **Step 5: Commit** qualification workflow and evidence tooling with `test: qualify paired Docker candidate before release (TASK-13343)`; corrected provisional CI execution is recorded in Step 4.

### Task 8: Review WP1 against the product contract

**Files:** Update `Dockerfiles/app-bundle/README.md`, `Docs/Development/Packaging_and_Distribution_Strategy.md` only if current availability changed, and TASK-13343 notes/plan status. No product claim goes into `README.md` installation directions before release approval.

**Interfaces:** Acceptance record links the exact candidate manifest and G2/G4/G10/G12 results. WP2-WP5 consume the signed manifest schema and the gateway/managed WebUI artifact without copying implementations.

- [x] **Step 1: Check** every WP1 requirement in spec sections 5, 10, 11, and the applicable section 12 gates against a test or a clearly recorded deferred WP4/WP5 boundary. The review record at `Docs/superpowers/reviews/2026-09-25-complete-app-wp1-acceptance.md` records the initial live gaps and final exact local/nativeG2/G4/G10 proof; G12 remains false and unsupported release behavior is explicitly deferred.
- [x] **Step 2: Run** full targeted frontend/backend/gateway suites and the candidate smoke once more only if Task 7 changed artifacts; otherwise cite its immutable evidence. Run `git diff --check`, scoped lint, and Bandit, and inspect the exact staged diff for secrets or unrelated files.
- [x] **Step 3: Record** tested Linux platforms, signed identities, exact runtime patches, Docker image Size metadata, CI build/qualification durations, and unsupported behavior. Actual download/installed footprints and startup benchmarks remain explicitly deferred to G12 release work; these metadata/durations do not substitute for them. TASK-13343 closes after its fresh extracted setup/networking/security criteria and local/native mapped checks pass; publication remains separately gated.
- [x] **Step 4: Commit** final task/doc/evidence updates with `docs: record provisional Docker candidate review (TASK-13343)`; the record explicitly states that qualification remains open.

### Task 9: Qualify live browser setup and two isolated instances

**Goal:** Close the current live-browser gap in Tasks 6/7 using the exact signed
extracted candidate. Keep the existing transport/unit evidence and broader G4
transport qualification distinct from these browser results.

**Files:** Create `apps/tldw-frontend/scripts/qualify-app-bundle-browser.mjs`
and focused behavioral tests under `apps/tldw-frontend/scripts/__tests__/`;
create `Helper_Scripts/test_app_bundle_browser.sh`; wire the probe into
`Helper_Scripts/qualify_app_bundle_candidate.sh` and
`.github/workflows/verify-app-bundle.yml`. Reuse existing Playwright dependencies.

**Interfaces:** The shell probe takes the extracted bundle path, resolves its
own repository tooling before changing directories, creates two disposable
instance state directories, and starts both with the signed host helper. It
passes a whitelist-only JSON file of public URLs and instance cookie names to
the browser probe. No API key, session token, cookie value, signing private key,
browser storage state, or trace is printed or uploaded. Output is a bounded
public JSON checklist with timings and booleans. Cleanup stops only these two
owned Compose projects and removes only disposable test state.

Reuse the same WebUI image digest for both instances. After the second signed
first start, a qualification-only Compose override changes its private backend
and Next hostnames/ports, including matching healthchecks. This fixture proves
runtime routing without editing the signed bundle or advertising a new user
configuration interface. Check actual container environment/image identity
without exposing credentials.

- [x] **Step 1: Write the behavioral checks first.** Require two distinct public
  loopback origins and cookie-name pairs; fail invalid/remote/duplicate inputs
  before browser launch. In a fresh real Chromium context, render managed setup
  without manually entering the master API key or seeding browser storage;
  inspect the actual visible setup flow and interact through accessible controls.
  Do not mock application API responses in live qualification.
- [x] **Step 2: Verify red evidence** before implementing the new probe. Preserve
  meaningful failure output in the task report. Existing Playwright tooling may
  be used for inspection; new qualification uses the repository's script style.
- [x] **Step 3: Implement the smallest probe.** Check anonymous profile refusal,
  browser-managed session bootstrap, HttpOnly session/readable CSRF attributes,
  cookie-only profile access, both instances in the same browser context,
  missing and foreign CSRF rejection, foreign-session rejection, logout of one
  instance without invalidating the other, and re-bootstrap. Fail on manual
  master-key requirements instead of automating credential entry. Record live
  readiness gaps as regression tests before changing product code.
- [x] **Step 4: Verify locally and on native CI.** Run focused probe tests, lint,
  shell syntax, release/gateway regressions, and the actual local/native paired
  candidates. Install only the pinned existing Playwright browser needed by CI;
  failure blocks its candidate job. Retain only the public checklist. Keep G4
  and G12 false until all their remaining transport/policy scenarios pass.
- [x] **Step 5: Review and commit** this scoped qualification slice with
  TASK-13343. Update the acceptance record and Backlog with actual evidence,
  discovered fixes, unsupported paths, and exact candidate identity.

### Task 10: Preserve local setup through the private Docker gateway

**Problem:** A real loopback request through the Docker gateway reaches the
backend with bridge-network peer and forwarded addresses. The existing setup
guard rejects this scope with 403, causing managed onboarding to fall back to
manual master-key entry.

**Files:** Scoped changes to `Dockerfiles/app-bundle/compose.yaml`,
`apps/tldw-frontend/gateway/server.mjs`,
`tldw_Server_API/app/api/v1/API_Deps/setup_deps.py`, and first-run metadata in
`tldw_Server_API/app/api/v1/endpoints/setup.py`; focused gateway/setup/Release
tests. Extend existing secret redaction only if the new private header needs it.

**Interface:** Extend the explicit authenticated gateway-hop contract to the
private backend. Configure its strong hop secret and exact persisted public
loopback origin only in managed Compose. One shared predicate validates the
single authenticated hop, private peer, bounded forwarding envelope and exact
Host/Origin for both setup access and first-run locality metadata. Ordinary
developer/hosted setup keeps its existing policy. No blanket proxy trust,
remote-setup override, API-key injection or client-visible secret is allowed.

Also configure existing backend `ALLOWED_ORIGINS` to that exact public origin.
Real fresh cookie WebSocket inspection fails its upgrade while the origin is
absent from `trusted_webui_origins`; ordinary notification SSE succeeds. This
uses the existing origin policy and adds no wildcard or global auth relaxation.

- [x] **Step 1:** Capture actual backend HTTP failure and write red behavioral
  tests for the measured Docker scope, metadata and Compose contract.
- [x] **Step 2:** Implement the bounded managed-hop predicate and gateway/header
  configuration. Reject missing/wrong/duplicate hop, public peer, mismatched
  origin/Host, malformed forwarding and disabled/unconfigured managed mode.
  Verify the managed persisted origin is the sole cookie WebSocket origin.
- [x] **Step 3:** Run focused positive/negative setup and gateway tests, scoped
  formatting/lint and Bandit. Review before live qualification.
- [x] **Step 4:** Rebuild from a clean committed source and run the actual fresh
  browser/two-instance candidate. Keep broader gates false until their complete
  requirements pass. Record exact evidence and TASK-13343 notes.

### Task 11: Ship the documentation read by the WebUI API

**Problem:** Actual extracted-image inspection returns 500 for the Next
documentation manifest and 404 for published content. The image omits the
`Docs/Published` files that the existing documentation reader requires.
Mounting only that directory read-only makes both API requests return 200.

**Files:** `Dockerfiles/Dockerfile.webui`,
`Helper_Scripts/test_app_bundle_docker.sh`, and candidate image-content checks
in `Helper_Scripts/qualify_app_bundle_candidate.sh`. Reuse the existing reader;
ship only its existing published source directory. No dependency or API change.

- [x] **Step 1:** Add real image/API checks that fail on the inspection image;
  record the existing 500/404 versus read-only-source-mount 200/200 evidence.
- [x] **Step 2:** Include the published documentation in the runtime image with
  correct ownership and its expected directory. Preserve managed/quickstart
  packaging and avoid shipping unrelated design/private content.
- [x] **Step 3:** Check manifest entries and actual published content from the
  signed extracted bundle outside a checkout; verify unsafe path refusal.
  Review and commit with TASK-13343.

### Task 12: Preserve MCP local policy through managed ingress

**Problem:** With cookie authentication and exact public origin working, actual
MCP WebSocket upgrade still returns 403. The backend logs its IP-policy rejection
of the Docker bridge peer. Audio WebSockets work through the same gateway.
MCP also has a separate exact-origin configuration with port-8000 defaults.

**Files:** Shared bounded hop validation under `app/core/Security/` if needed;
reuse it from setup access/metadata and MCP HTTP/WebSocket IP resolution. Scoped
MCP guards/server, gateway and managed Compose, with focused adversarial tests.

**Contract:** Recognize authenticated managed ingress as the persisted loopback
client before applying the existing MCP allow/block policy. Keep explicit block
rules and normal authentication/permissions active. Configure MCP origins to the
exact persisted gateway origin. Missing/forged hop, direct private peers,
unconfigured mode and hostile origins retain their current refusals. Extend hop
injection only to the bounded MCP surface; never expose it in browser responses,
stored transport metadata or unrelated backend endpoints. Reuse the existing
predicate instead of inventing a second forwarding policy or disabling IP rules.

- [x] **Step 1:** Write red HTTP/WebSocket behavior for measured bridge peers,
  existing block rules and hostile direct/forwarded inputs.
- [x] **Step 2:** Implement the smallest shared contract and exact MCP origin
  configuration. Preserve ordinary hosted/developer policies.
- [x] **Step 3:** Run focused positive/negative tests, scoped lint/format/Bandit,
  independent security review and actual cookie-only MCP WebSocket roundtrip.

### Task 15: Preserve writable MCP audit logging in managed Docker

**Problem:** Exact clean arm64 candidate087e77ce4a builds all four roles, then
38 MCP security tests fail with PermissionError at /app/audit.log. The runtime
uses non-root appuser with /app cwd, audit logging is enabled by default, and
managed Compose supplies no audit path. The default also affects real MCP use.

**Files:** Managed Compose, built-backend qualification environment, and focused
existing Release/configuration tests. Keep product code and dependencies intact.
Execute after Task14 review and before resuming Task13 actual qualification.

**Contract:** Keep MCP audit logging enabled. Set MCP_AUDIT_LOG_FILE to
/app/Databases/mcp-audit.log in managed Compose and the built-backend security
test container. Reuse the existing writable persisted database volume and image
user; do not run as root, disable audit, broaden filesystem permissions, or
alter ordinary developer/hosted logging defaults. Qualification logs and test
containers stay disposable and excluded from public evidence.

- [x] **Step 1:** Preserve actual clean-image permission failure and add a red
  regression for the managed persisted audit path and matching test environment.
- [x] **Step 2:** Configure the existing path override minimally, with no new
  dependency or production source behavior. Validate Compose and focused tests.
- [x] **Step 3:** Run the exact lean MCP/setup commands on the already-built
  backend as its default non-root user with only the path override and narrow
  read-only test mounts. Keep bounded results, remove owned test resources,
  independently review, commit, then retry the clean full candidate in Task13.

### Task 16: Make embedded public trust readable to the control caller

**Problem:** Exact d081a4dd43 candidate passes all image builds and built MCP/setup
security tests, then its signed helper verification raises PermissionError at
/opt/tldw/trusted-keys/ci-test.pub. Candidate umask077 creates the public source
key as0600, COPY preserves root ownership/mode, and helpers intentionally run
control under the host caller UID. Private signing key restriction is correct.

**Files:** Dockerfiles/Dockerfile.control; existing candidate image-content guard
and focused Release packaging/tool tests. No verifier/key/trust-policy changes.
Execute before resuming Task13 actual local/native qualification.

**Contract:** Precreate the root-owned public trust directory as0755 so caller
UIDs can traverse it; explicitly copy embedded public keys read-only as0444
independently of source umask. Actual BuildKit COPY --chmod also applies the
mode to a newly created destination directory, so implicit directory creation
is insufficient. Keep private signing keys0600/outsideimage, trust fixed in the
control digest, and helper caller UID/no-new-privileges/read-only isolation.
Check actual public-key readability under the helper caller UID in candidate
content guards rather than only checking size as image-default root.

- [x] **Step 1:** Preserve actual PermissionError and write red packaging/guard
  behavior for restrictive public source key mode and non-root caller access.
- [x] **Step 2:** Precreate the public trust directory as0755, add COPY
  --chmod=0444 for the existing public-key copy and strengthen only the bounded
  control-content guard; preserve trust boundaries.
- [x] **Step 3:** Build only a scoped control image from a restrictive public-key
  source and verify non-root key read/signature behavior with bounded output.
  Run focused tests/lint/syntax/Bandit where applicable, remove owned test
  containers/state, independently review and commit. Fullcandidate retry belongs
  to Task13 and remains separate from this scoped proof.

### Task 13: Qualify remaining real-container transport paths

**Files:** Extend the existing bounded browser probe/tests and paired shell
fixture; wire any exact required evidence into the candidate workflow.

Qualification tooling may select a validated local registry port through
`TLDW_CANDIDATE_REGISTRY_PORT` (default `5000` in CI; `15000` on this host, where
macOS ControlCenter owns port 5000). Keep the registry host loopback-only and
remove only its own created container/storage. All fixture cleanup, including
browser shutdown and the older lifecycle smoke, must fail closed and preserve
recovery state on failure.

- [x] **Step 1:** Add real published documentation, redirect, multipart document
  processing, cancellable notification SSE, cookie WebSocket roundtrip and
  hostile forwarding/Origin checks to both varying private-target instances.
  Use harmless public fixtures and no paid provider/model download.
- [x] **Step 2:** Keep raw response bodies, cookies, tokens, traces and error
  messages out of outputs. Prove every required result fails closed and owned
  cleanup completes before reporting success.
- [x] **Step 3:** Run the exact clean signed local candidate, then both native CI
  candidates. Record immutable identities and actual bounded evidence. G2/G4
  become true only when their complete mapped checks pass; G12 publication
  policy remains separately gated. Review the whole WP1 branch and update final
  acceptance/Backlog records without publishing protected artifacts.

### Task 14: Keep packaging regression guards aligned with paired targets

**Problem:** Five existing Dockerfile regression tests fail at the Task 11 base.
Four assert the single-builder spelling replaced earlier in WP1; the API
allowlist omits the intentionally added local profile package. These are not
Task 11 regressions, but must be resolved before whole-WP1 acceptance.

**Files:** `tldw_Server_API/tests/Utils/test_docker_quickstart_hardening.py`,
`test_makefile_quickstart_same_origin.py`; lean candidate workflow test wiring.
Execute this bounded task after Task 12 review and before Task 13 qualification.

- [x] **Step 1:** Preserve the current red evidence and replace obsolete exact
  builder/argument spellings with checks of both targets' actual inherited
  configuration, scoped source copies and ownership. Keep default quickstart
  same-origin behavior and managed artifact secret/origin independence guarded.
- [x] **Step 2:** Extend the API COPY allowlist only for the reviewed local
  profile package; retain protected-frontend exclusion and legal-file checks.
  Add meaningful negative fixtures proving the guards still reject unsafe
  packaging. Do not remove or disable tests or broaden the allowlist.
- [x] **Step 3:** Run the focused suite, wire it into lean CI, lint/format and
  review. Keep real artifact qualification in Task 13.

## Plan self-review checklist

CI follow-up (September 25): user-authorized branch push bootstrapped the new
workflow with a branch/file-restricted push trigger. Run 36206663211 at
754c9dd521 built all four images and reached healthy extracted backend/WebUI/
gateway services on native amd64 and arm64; Windows helper parsing passed.
Both smoke checks then failed on an HTTP request. A red Compose regression and
real standalone Next/gateway exchange identify missing WebUI AUTH_MODE:
omitted returns 503, single_user returns 204 with two cookies. The corrected
Compose setting, HTTP failure-line diagnostics, and failure-evidence retention
are implemented; 47 lean release tests, shell/Compose validation, and production
Bandit (zero findings) pass. Corrected both-platform smoke remains required.
Further G4 review found `/setup` incorrectly routed to legacy FastAPI setup and
backend docs assets routed to Next. The real-socket regression was red; routing
is corrected, 13 gateway tests/lint pass, and standalone Next `/setup` returns
200 through the gateway. Added exact container smoke checks for that page and
docs assets; this source revision must be exercised after the auth-mode run.
Cookie review then found the backend lacked the shared instance session-cookie
name and retained its HTTPS-only default on the loopback HTTP gateway. Two
regressions failed before the Compose correction; 49 release tests and three
focused existing AuthNZ integration tests now pass. The smoke requires both
configured cookie names and authenticates the profile using only the cookie
jar. Both-platform run 36208320371 passed its older smoke, whose generic cookie
assertion did not establish authenticated browser behavior. Run 36209873850
was cancelled before exercising the obsolete configuration. Corrected native
smoke passed on both platforms in run 36210634045 at e759322854. Each job passes
49 Release, 207 frontend, and 13 gateway tests, production Bandit, and the
strengthened cookie-only profile/setup/docs checks. Downloaded signatures and
all eight helper hashes independently verify for each provisional manifest;
promotion is correctly refused. Focused independent review found no actionable
issues. G2 interactive browser setup, G4 broader browser/two-instance networking,
and full G12 policy/footprint qualification remain open; a qualified combined
multi-platform candidate is still required.

- Sections 5/10/11 and gates G2/G4/G10/G12 map to Tasks 1–8. Backend dependency slimming and complete storage inventory are WP2; native lifecycle and the single-source verifier packaging check are WP3; automatic update/backup/restore is WP4; guided heavy components and final public promotion are WP5.
- Verify exact code symbols in the interface blocks when implementing each task; preserve existing quickstart/hosted paths and test both managed and legacy modes.
- Do not claim G2/G4/G10/G12 complete until candidate artifacts pass the corresponding tests. This plan is executable work, not evidence that they already pass.

Qualification tooling review (September 25): Tasks9-12 scoped implementation
and Task14 packaging guards are approved. Task13 commits9dfce026da and558fdb814a
pass focused review after adding the exact excluded setup test as a read-only
qualification mount. Browser50, candidate38/helper14, MCP39 and Setup60 checks
pass; production security Bandit reports zero findings. These are tooling/unit
results, not real-image acceptance. Exact clean local arm64 and both native CI
candidates, independent artifact verification and whole-WP1 review remain the
next gates. G2/G4 and AC4 remain open until their complete mapped evidence passes;
G12 publication stays separately blocked. Existing dependency warnings and
whole-frontend typecheck baseline remain disclosed.

Managed audit-path correction b0e847166e is independently approved. Two existing
environment overrides retain enabled MCP auditing under the default non-root
user in /app/Databases/mcp-audit.log. Cached exact087e77 backend audit init/write,
MCP39 and Setup60 pass; Release81 passes. Owned diagnostic/test resources removed.
At this historical checkpoint full candidate proof was pending; cached-image checks do not close
G2/G4. Container property tests warn that their example database falls back to
memory under the read-only /app directory; dependency warnings remain visible.

Embedded public-trust correction537581410c/db9157a342 is independently approved.
Root-owned directory0755 and public files0444 remain readable under callerUID501
while private signing files stay0600/outsideimage. Scoped exactcontrolimage
validsignature passes and tampering refuses;87Release tests pass. Owned proof
containers/tags/signing state removed. Exactfulllocal/nativecandidate remains
Task13, and G2/G4 are not qualified by these scoped checks.

### Task 17: Finish bootstrap responses and qualify the existing API cookie scope

**Files:** `apps/tldw-frontend/extension/shims/runtime-bootstrap.ts`, its focused
Vitest tests, and the existing paired browser probe/tests. TASK-13343 covers this
actual-browser follow-up; production auth/cookie settings stay unchanged.

- [x] **Step 1:** Regress actual session `Path=/api` discovery and validation at
  every probe lookup, including hostile-input checks and rebootstrap. Require
  exact existing session `/api` and readable CSRF `/` scopes; reject broader or
  mismatched scopes. Preserve all other cookie/isolation policy checks.
- [x] **Step 2:** Add behavioral response-body lifecycle regressions before
  draining successful bootstrap and profile responses inside their existing
  request-timeout callbacks. Preserve failure/manual-config behavior, deadlines
  and secret-free runtime metadata. Keep the strict network-error latch.
- [x] **Step 3:** Run focused suites, scoped lint/security and a real Chromium
  response-consumption proof, independently review, then return exact clean
  full local/native artifact qualification to Task13. No public publication.

Task17 e59e926194 is independently approved: runtime34/probe53 pass, strict
Chromium source comparison preserves authentication/cookies and removes the
abandoned-body cancellation. Both body reads remain inside eight-second
deadlines; existing session /api scope stays unchanged. Owned proof resources
removed. This scoped result does not qualify the rebuilt candidate; Task13
At that checkpoint exact local/native checks and whole-WP1 review were pending; their completed final evidence is recorded below.

### Task 18: One combined whole-branch review fix wave

**Spec clarification:**
`Docs/Design/2026-09-26-complete-app-wp1-installer-trust-readiness-follow-up.md`.
**Review:** plan workspace `final-review.md`, findings I1-I4 and M1.

**Files:** All six host helpers, `tldw_Server_API/scripts/app_bundle_control.py`,
`Helper_Scripts/verify_app_bundle.py`, paired-inventory definitions as needed
within existing Release source, existing Release control/helper/candidate tests,
`Dockerfiles/app-bundle/README.md`; narrow new focused readiness module/tests or
Dockerfile.control copy adjustment only if required by the maintained layout.
No generic parser restriction for native formats, host dependencies or socket
mounts. Existing workflow/qualification fixtures may receive narrow adaptation
required by the new helper contract; no gates/tests are weakened.

#### Stage 1: Trusted configuration and complete inventory
**Goal:** Fix I1/I2.
**Success Criteria:** Conflicting inherited image/origin/credential/cookie input
cannot override verified state, and both consumers refuse incomplete bundles.
**Tests:** Red/green helper-child and actual Compose config regressions; omitted,
missing, wrong-platform and tampered required files in control and promotion.
**Status:** Complete
- [x] Add failing tests, implement minimal paired-boundary fixes, run covering
  Release/helper suites and preserve generic native manifest compatibility.

#### Stage 2: Authenticated readiness and honest cleanup
**Goal:** Fix I3/M1.
**Success Criteria:** Helpers check actual runtime identity and gateway paths,
cookie-only auth and exact temporary-session revocation before URL announcement;
failed readiness/cleanup cannot report success or erase persistent config.
**Tests:** Real HTTP upstream fixtures including healthy containers/broken auth,
wrong role/source, failed/stalled body, failed revoke and cleanup failure.
**Status:** Complete
- [x] Add failing behavior tests, implement bounded existing-control readiness
  and mirror host invocation on Unix/PowerShell; run targeted proof and Bandit.

#### Stage 3: Safe first-origin choice
**Goal:** Fix I4.
**Success Criteria:** Occupied default yields an available choice before init,
explicit occupied choice permits alternate retry, established origin never moves.
**Tests:** Docker-owned port preflight/failure cleanup, unrelated listener
preserved, explicit retry and existing-state refusal without credential rotation.
**Status:** Complete
- [x] Add failing first-install regressions, implement owned preflight and
  truthful failure messages, run helper tests; label Windows runtime unqualified.

#### Stage 4: Final fix-wave review and actual acceptance
**Goal:** Complete the single final fix wave without claiming scoped tests are
artifact qualification.
**Success Criteria:** Covering tests/lint/security pass, one scoped rereview
resolves I1-I4/M1 or records residual rulings; exact candidate local/native proof
and final acceptance record remain controller-owned Task13.
**Tests:** Focused amended scopes only, immutable candidate identities and
full mapped lifecycle/browser checklists after clean source is committed.
**Status:** Complete
- [x] Commit working slices, append fix report with exact tests/results and
  limitations; controller dispatches one scoped rereview and fresh qualification.

Scoped rereview `19f2174383..1d0bcb9b28` approves all I1-I4/M1 and finds no new code defects. Release158 passed, followed by amended control15 after one final regression addition; Black/Ruff/shell pass and all changed production Python Bandit0. Real owned Docker fixtures pass port binding, actual inspection/network, gateway auth/revoke and broken-auth refusal. At that checkpoint rebuilt local/native artifacts were pending. Final exactba124 local/native proof below completes Stage4/Task13; Windows runtime remains unqualified.

### Task 19: Correct the actual installer readiness route

**Authority:** Approved design section5; actual bd0de933 first-start failure and
read-only cached-artifact diagnosis. This repairs Task13 qualification rather
than dispatching another whole-branch review fix wave. The sole final fix-wave
rereview found no residual code findings; this integration defect was discovered
by the mandatory actual artifact run afterward.

**Files:** `tldw_Server_API/scripts/app_bundle_readiness.py`, its focused Release
tests, and existing control tests only if their fixture contract needs alignment.
Do not change backend readiness/permission guards, gateway routes, host helpers,
image policy, signed inventory, public evidence or qualification gates.

#### Stage 1: Reproduce the actual route contract
**Goal:** Capture the private-bridge internal404 and authenticated operator route.
**Success Criteria:** A real HTTP fixture rejects /internal/ready, requires the
probe session for /api/v1/health/ready, and reproduces current installer failure.
**Tests:** Expected red; absent/foreign/operator-forbidden cookie,503/not_ready,
malformed/truncated/oversized or stalled readiness and exact cleanup on failure.
**Status:** Complete
- [x] Add focused failing behavior cases matching the existing backend policy.

#### Stage 2: Use the existing authenticated readiness projection
**Goal:** Complete the readiness components before opening the browser.
**Success Criteria:** After bootstrap/profile, cookie-only GET
/api/v1/health/ready returns200 plus JSON status ready; finally revoke the exact
session and refuse replay. Preserve request/body/overall/cleanup bounds.
**Tests:** Focused Release/readiness and affected control coverage; Black/Ruff,
Bandit on changed production Python, and bounded actual cached gateway proof.
**Status:** Complete
- [x] Implement the minimal probe ordering/path/result change; do not substitute
  public liveness, inject a master key, or exempt internal loopback restrictions.

#### Stage 3: Review and retry actual qualification
**Goal:** Verify the correction and exact final artifact, with no public release.
**Success Criteria:** Scoped task review passes; fresh clean local and both native
Linux container architecture bundles pass13lifecycle/38browser checks and signatures.
Windows runtime remains explicitly unqualified and G12 false.
**Tests:** Parent-owned full candidate and independent artifact verification.
**Status:** Complete
- [x] Commit scoped code/tests/report; review amended code, then retry Task13.

Task19 implementation9e4f8fc620/reportc63cb74bf4 scoped review approves spec and quality with no Critical/Important findings. Readiness/control52 pass with4disclosed baseline warnings; Black/Ruff pass and production Bandit0. Actual maintained source-mounted cached probe proves operator200/ready, DELETE200 and revoked profile/operator401; ownedcleanup passes. Actual not_ready injection was not exercised; realHTTPfixture covers its cleanup. Fresh signedlocal/native qualification was pending at this checkpoint and is completed in the final exactba124 record below.


### Task 20: Preserve CSRF protection for managed noAuth setup mutations

**Authority:** Approved credential-free managed setup contract and exact2840100
candidate browser failure. Unchanged cached images show setup GETs200 and cookie
profile200 after bootstrap, then first-run state POST403 without X-CSRF-Token;
the same observed body with the browser instance token returns200. Task13 stays
unqualified until fresh full artifact checks pass.

**Files:** `apps/packages/ui/src/services/tldw/request-core.ts` and focused
request-core cookie-transport tests. A narrow setup-onboarding integration test
may use existing harnesses if required; no setup domain API rewrites, backend or
gateway guard changes, cookie scope changes, dependencies or qualification latch
exemptions. Read package and frontend instructions before edits.

#### Stage 1: Reproduce noAuth cookie mutation and boundaries
**Goal:** Capture the first-run request contract in behavioral regression tests.
**Success Criteria:** The current client omits the required runtime CSRF token
for noAuth unsafe requests in cookie transport; tests distinguish key/bearer
omission from cookie-bound CSRF protection and prevent cross-origin leakage.
**Tests:** Red on first-run POST noAuth with configured instance cookie;
case-insensitive stale auth/CSRF headers replaced; safe requests omit CSRF;
relative and same-page/configured-origin absolute requests receive only their
instance token; allowlisted external and configured-but-not-page-origin requests
receive no page CSRF; missing/foreign instance cookie supplies no fallback token.
**Status:** Complete
- [x] Add failing behavior tests using the existing request-core harness.

#### Stage 2: Separate CSRF from key/bearer omission
**Goal:** Allow the real managed wizard mutation without weakening server policy.
**Success Criteria:** Unsafe same-origin cookie transport attaches only the
runtime-configured CSRF token even when noAuth skips explicit authentication.
Cookie transport continues stripping stale API key, bearer and CSRF headers;
external destinations, safe requests, manual/hosted transports retain contracts.
**Tests:** Focused request-core and affected setup/background-proxy suites;
existing package lint/format checks and diff review. Bandit has no changed Python
production scope; record the non-Python limitation explicitly.
**Status:** Complete
- [x] Make the minimal proven request-core correction, self-review and commit.

#### Stage 3: Scoped review and actual qualification
**Goal:** Resolve Task13 browser setup failure on fresh exact artifacts.
**Success Criteria:** Independent task review approves spec and quality; actual
changed-source browser reaches provider setup with setup requests200 and strict
network/API latches preserved; temporary session revoke and owned cleanup pass.
Full parent clean signed local and native13lifecycle/38browser checks remain
mandatory, with G12 false and Windows runtime explicitly unqualified.
**Tests:** Bounded changed-source managed production WebUI proof when feasible,
then controller-owned exact full candidate and independent signatures.
**Status:** Complete
- [x] Report focused evidence/limitations; controller reviews and retries Task13.

Task20 codee34d3151da scoped spec/quality approved, no Critical/Important findings. One production condition plus18behavioral tests, including actualsetupmethod→bgRequest→realrequestcore. Expectedred6fail8pass→focused14pass; finalaffected291pass10baselinepersistencefail across11suites (10pass), baseline10fail6pass unchangedproduction. Overallrunnotpassing. ESLint0errors86identicalbaselinewarnings; baseline/current wholefileformat2warnings+printercrash, addedblocksformatpass. BanditN/AallTS. Actualcompiled qualification was pending at this checkpoint; final exactba124local/native13/38/signature proof below completes Task13.


### Task 21: Exercise the required privacy acknowledgement in browser qualification

**Authority:** Actual69e605 retry1 signedlifecycle13 passes and setup API access
passes after Task20, but the maintained browser probe omits the product's required
privacy checkbox and times out on disabled Continue. One unchanged cached compiled
stack proves unchecked/disabled state, then normal check/click reaches provider
setup with setup_path POST200, privacy POST200 and catalog200. Strict API/network
trackers stay clean. This is a qualification fixture defect, not a product defect.

**Files:** `apps/tldw-frontend/scripts/qualify-app-bundle-browser.mjs` and
`scripts/__tests__/qualify-app-bundle-browser.test.mjs` only. Do not change product
UI/backend/gateway, cookie/CSRF/remote policy, qualification scope or refusal latches,
image/dependency policy, helpers, public evidence schema, or unrelated test harnesses.

#### Stage 1: Model and reproduce the real required action
**Goal:** Prevent synthetic success that skips the real wizard acknowledgement.
**Success Criteria:** The paired fixture renders Privacy and security and the
checkbox labelled exactly "I understand local or remote setup access and provider
secret storage." after normal Solo Docker selection. Continue starts disabled and
becomes enabled only through normal acknowledgement. Model the two actual setup
saves as asynchronous HTTP POSTs, with the fixture's existing cookie/CSRF contract;
provider view appears only after successful privacy save. Existing GET access
cannot substitute for successful mutation.
**Tests:** Expected red against unchanged probe; behavior verifies acknowledgement
before Continue and setup_path then privacy writes. Include missing/disabled
acknowledgement or refused save as a failed qualification with sticky API/network
latches, rather than skipping or accepting later success.
**Status:** Complete
- [x] Strengthen the actual browser fixture and add focused failing behavior test.

#### Stage 2: Follow normal user wizard progression
**Goal:** Align the maintained probe with the current real UI flow.
**Success Criteria:** After Solo Docker, wait for the Privacy and security heading,
check the exact acknowledgement using the normal locator action, then normally
click Continue and require the existing provider heading. No force option,
JavaScript state injection, UI modification, acknowledgement bypass or weakened
tracker/timeout success condition. Partial setup scope remains unchanged.
**Tests:** Focused behavior red/green, then complete maintained browser-probe test
file once and scoped ESLint/diff checks. Bandit is inapplicable to MJS-only edits.
**Status:** Complete
- [x] Make the minimal probe correction, self-review and commit scoped files.

#### Stage 3: Review and exact artifact proof
**Goal:** Qualify the corrected probe and preserve the product security contract.
**Success Criteria:** Scoped spec/quality review approves; controller rebuilds a
clean exact local and both native Linux candidates requiring all13lifecycle and
all38browser checks, independent signatures and owned cleanup. Windows runtime
remains unqualified, G12 false, publication frozen. A cleared-cookie profile401
from diagnosis is not claimed as stale-token replay; parent checklist must prove
its own actual revocation contract.
**Tests:** Parent-owned full exact candidate and independent evidence verifier.
**Status:** Complete
- [x] Report scoped tests/limits; controller reviews and resumes Task13 acceptance.

Task21 `a7e246c38c` scoped review approves spec/quality with no findings. Expected red reproduces disabled Continue; focused6/6 and complete maintained browser-probe file58/58 pass, scoped ESLint/diff clean. Only two MJS files changed; Bandit inapplicable. Stage3 proof was pending at this checkpoint; final exactba124 local/native proof below completes it.

### Task 22: Return a bounded HTTP refusal for unauthorized WebSocket upgrades

**Authority:** Exact13d9 native amd64/arm64 both pass all13 signed lifecycle checks
and normal setup on both instances, then fail only hostile_inputs_1. One actual
cached69 runtime reproduction (unchanged product source against13d9) isolates the
fifth request: hostile Origin with a valid cookie produces ECONNRESET/no HTTP
because gateway upgrade authorization calls socket.destroy(). First four hostile
cases pass. This is refusal transport inconsistency, not observed auth acceptance.

**Files:** `apps/tldw-frontend/gateway/server.mjs` and
`apps/tldw-frontend/gateway/__tests__/server.test.mjs` only. Preserve Host/Origin
allowlists, authentication/private-hop normalization, ready-phase and invalid-path
behavior, HTTP routes, successful upgrades and strict maintained probe. No backend,
UI, cookie/CSRF, dependency/image, helper, schema or tracker changes. This is a
bounded repair inside the already approved gateway refusal design and ongoing
user authorization to complete WP1.

#### Stage 1: Reproduce real refusal transport
**Goal:** Test the actual gateway rather than a synthetic403 fixture.
**Success Criteria:** Native HTTP upgrade with hostile Origin plus a cookie, and
hostile Host, receive403 with no101, Set-Cookie, auth/hop headers or reflected input;
neither backend nor Next receives the upgrade. Same legitimate cookie/origin upgrade
continues to proxy. Baseline reset must fail the new behavioral regression.
**Tests:** Focused real-socket expected red, no broad test or policy bypass.
**Status:** Complete
- [x] Add real-gateway forbidden-upgrade regression and witness red.

#### Stage 2: Provide a bounded protocol refusal
**Goal:** Preserve rejection while supplying a deterministic HTTP response.
**Success Criteria:** Split authorization failure from existing other upgrade
refusals. Send only a small constant HTTP403 handshake response then end the socket;
use an absolute1000ms destruction backstop for a non-cooperative peer, cleared on
close and unreferenced so it cannot retain process shutdown. Install a pre-end
socket error handler that destroys a refused socket after peer reset; actual
gateway regression must prove the process survives and socket cleanup completes.
Never proxy or reflect
request material. Existing invalid-path, not-ready and managed-route behavior stays
unchanged. Do not broaden helpers or weaken hostile-request failure checks.
**Tests:** Focused green, complete gateway real-socket test file once and route tests,
non-cooperative socket/cleanup behavior, scoped ESLint/diff. Bandit inapplicable to
MJS-only edits; disclose any baseline warnings precisely rather than suppressing them.
**Status:** Complete
- [x] Make minimal gateway correction, self-review and scoped TASK-13343 commits9722148d6f/d0afab1daf.

#### Stage 3: Independent review and exact proof
**Goal:** Resolve the newly reproduced actual-candidate gate before acceptance.
**Success Criteria:** Scoped independent spec/quality review approves; parent rebuilds
fresh exact local and native Linux candidates requiring all13lifecycle/all38browser
checks, independent signature/eight-file verification and owned cleanup. Windows
runtime remains unqualified, G12 false, frontend publication frozen. No second broad
whole-branch review or release/push of protected images.
**Tests:** Parent-owned exact candidate/checklist/signature verification. Previous
failed candidates and diagnostic images cannot supply successful qualification.
**Status:** Complete
- [x] Report tests/limits; controller reviews and completes Task13 acceptance.

Task22 scoped review found a detached-upgrade peer-reset crash risk. Actual
createGateway regression witnessed unhandled ECONNRESET/process exit1; round1
adds the single refusal-socket error handler and proves actual ECONNRESET,
destroyed/error-close plus subsequent status200/process exit0. Final gateway and
route files20/20 pass once, scoped ESLint/diff clean, no test/lint warnings;
Bandit inapplicable to MJS-only scope. Independent scoped fix review approves: the reset finding is addressed with
no new breakage or out-of-scope findings. Fresh exact local/native qualification was pending at this checkpoint and is
completed in the final exactba124 record below. Existing Git housekeeping warnings
are disclosed without manual GC/prune.

### Task 23: Keep forged foreign-session probes outside browser cookie storage

**Authority:** Exactff3 local candidate built all four roles, passed builtMCPSetup
and all13 lifecycle checks, then passed35 browser checks with onlylogout_isolated
false. Both hostile-upgrade checks now pass. One actual exactff two-instance
runtime diagnosis proves shared Playwright context.request installs a replacement
CSRF cookie from each foreign-session401, making the previously captured logout
token stale. Current-token logout200 and profiles401/200 prove product isolation.
This is a bounded probe repair within the approved strict qualification design.

**Files:** Only `apps/tldw-frontend/scripts/qualify-app-bundle-browser.mjs` and
`scripts/__tests__/qualify-app-bundle-browser.test.mjs`. Use the existing native
hostileRequest for the deliberately forged foreign-session GET, with the same
explicit foreign Cookie and strict401/403 assertion. Native response Set-Cookie
must not enter the browser jar. Keep the captured-CSRF logout assertion, same
browser context/two instances, rebootstrap, security latches, bounds, evidence
schema and product policies unchanged. Do not refresh tokens to hide contamination,
preseed/mutate storage, alter backend CSRF/logout/auth, or broaden the helper API.

#### Stage 1: Reproduce cookie-jar contamination
**Goal:** Model the actual backend response in the existing paired fixture.
**Success Criteria:** A foreign-session profile401 emits a replacement target CSRF
Set-Cookie when its explicit Cookie omits CSRF. Regression proves shared request
context poisons the live token and fails the strict logout sequence before repair.
The fixture confirms each hostile request reached the target and the real browser
session/CSRF state survives after the isolated native request. Keep401/403 strict.
**Tests:** Focused behavioral expected red against unchanged probe; no product bypass.
**Status:** Complete
- [x] Add the real-shaped response fixture and witness strict logout failure.

#### Stage 2: Isolate the hostile request
**Goal:** Forge only the request rather than browser storage.
**Success Criteria:** Minimal native-request replacement preserves authentication
refusal, original legitimate browser state, captured-token logout200, revoked
profile refusal and other instanceprofile200, rebootstrap and all latches.
**Tests:** Focused green, full maintained probe test file once after final code,
scoped ESLint/diff, Bandit inapplicable to MJS-only edits; disclose all warnings.
**Status:** Complete
- [x] Implement, self-review and commit ae980cead7 for TASK-13343.

#### Stage 3: Independent review and fresh exact proof
**Goal:** Complete the actual matrix without weakening its requirements.
**Success Criteria:** Scoped independent spec/quality approval; fresh exact local
and native Linux candidates passall13lifecycle/all38browser checks, signatures,
eight signed helper files and owned cleanup. Failedff3 artifacts remain invalid;
diagnostic unsigned setup cannot qualify them. Windowsruntime unqualified/G12false,
frontend publication frozen; no second broad branch review or public release.
**Tests:** Parent-owned exact candidate and independent artifact verification.
**Status:** Complete
- [x] Report results and limits; controller reviews and completes Task13 acceptance.

Task23 ae980cead7 changes only the native forged-session GET and its existing
paired fixture/regression. Expectedredlogout_isolated, focusedgreen1/1 and final
maintainedfile58/58 pass once; scoped ESLint/diff clean, no test/lint warnings,
Bandit inapplicable MJS-only. Scoped Sol-high review approves spec/quality with
no findings; final exactba124 local/native13/38/signatures/eightfiles proof below completes qualification.
Existing Git housekeeping warnings remain disclosed without manualGC/prune.

## Final WP1 qualification (2026-09-26)

Exact built source `ba12489e0d7b7219140394784d1556cd0bfc0fe5` passes fresh local
arm64 and native amd64/arm64 CI 36244097597. Each candidate passes all 13 lifecycle
and all 38 browser checks; independent raw Ed25519, production verification, eight
helper hashes and owned cleanup pass. G2/G4/G10 true apply only to managed
connection and initial wizard progression; `planned_setup_complete=false` in browser evidence, signed G12 false,
and promotion refusal remain required. Final branch review and one combined fix
wave were completed earlier; Tasks19–23 actual integration repairs each have
scoped independent approval. No second broad branch review or public release.
Acceptance record contains all three immutable manifest hashes and limitations.
Task8 footprint/startup measurements are deferred release work per recorded
ruling; Windows helper parsing supplies no Windows runtime qualification.
The full frontend baseline of 93 diagnostics and Task20 persistence baseline of
10 failures remain
reported as failures, not passing tests. Final documentation-only commit is
not the built source and requires no artifact requalification claim.
