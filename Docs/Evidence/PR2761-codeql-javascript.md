# PR #2761 JavaScript CodeQL remediation and review

Follow-up CI validation: the new redirect test originally forced a Node
environment, but the owning UI package's setup requires jsdom. Removing only
that override preserves native HTTP fixtures and all network guards. Both the
UI and frontend configurations pass all 13 cases. The exact previously failing
UI direct-changed shard 6/8 passes **545 tests across 42 files** with the workflow's
deterministic configuration and timeout settings. Logs:
`/tmp/pr2761-ui-redirect-jsdom.log`,
`/tmp/pr2761-frontend-redirect-jsdom.log`, and
`/tmp/pr2761-ui-shard6-jsdom.log`.

Date: 2026-09-10. Tracking: TASK-13013.3.1 (parent TASK-13013.3).
Reviewed candidate: `28797892b1`. Inventory: `/tmp/pr2761-codeql-current-alerts.json`;
full JavaScript SARIF: `/tmp/pr2761-js-current.sarif.json`. The table covers all
**32 open JavaScript alert instances** in that inventory, not every result in the
full SARIF. Locations below identify the analyzed candidate before these edits.
No alert was dismissed, rule disabled, threshold changed, or test weakened by
this work. Existing request-core `lgtm` comments were removed when fixing the
request boundary; no replacement suppression was added.

## Repaired boundaries

**Request destination (2355).** Two real loopback HTTP fixtures reproduced the
redirect behavior using native fetch: a configured server returned HTTP 307 to
another origin, which received `X-API-KEY`; a 401/refresh/retry sequence also
followed an unvalidated redirect. Native fetch stripped Authorization on that
cross-origin retry, but still sent the request. Both initial and retry fetches
now use `redirect: "error"`. Two additional regressions demonstrated that
quickstart and hosted transports accepted network-path references such as
`//attacker.example/collect` and `/\attacker.example/collect`. Relative
same-origin transports now compare the browser-resolved destination origin
before constructing authentication headers. Explicit allowlisted absolute URLs
remain supported; a native-fetch test verifies success without attaching the
configured server's API key. Existing configured-server, hosted, cookie/CSRF,
timeout, binary-body, refresh, and URL-allowlist tests remain.

Compatibility: HTTP redirects now fail, including same-origin redirects. Clients
must use canonical API endpoints. This avoids forwarding custom API-key headers
and request bodies across a redirect boundary; it does not remove support for
user-configured remote/private servers. Custom injected fetch implementations
must honor standard Fetch redirect semantics. No live external requests were
used: network regressions bind disposable loopback fixtures only.

**Operator-key UAT seeding (2646–2653).** The five scripts previously installed
the operator's environment-provided key through an unrestricted context init
script. Playwright executes context init scripts in new frames and after
navigation, so an unrelated origin could receive that key. The shared,
self-contained `seedManualUatBrowser` callback now returns before any storage
write unless `location.origin` exactly equals the configured WebUI URL's origin.
The callback preserves the previous minimal configuration for oversight/round3
and legacy bootstrap mirrors/flags for stats/admin. The tests reject foreign
origins, lookalike subdomains, different ports, HTTP downgrade, and opaque
origins; the allowed-origin control verifies the actual stored configuration.
All five callers create disposable `browser.newContext()` contexts; these are
manual BYOK workflow probes, not production credential storage code. The key
still exists in that selected page's localStorage for the authorized UAT
session. This change closes unintended origin seeding; it does not claim that
client-side storage is encrypted or make those intentional writes disappear.

**CORS fixtures (2620–2622).** Both loopback protocol servers reflected any
Origin, and JSON replies enabled credentials. The fixtures now authorize only
the selected WebUI origin and/or exact launched extension ID, using the same
allowlist on preflight and actual responses. The corrupt-state test starts with
an empty set and adds the launched extension's origin before navigation to the
editor. No-origin requests still work for Node health probes but receive no CORS
authorization. Seven tests preserve selected WebUI/extension access while
rejecting unrelated origins, ports, extensions, and opaque origins. These
servers return synthetic prompt/health/error data, not production user data;
the permissive fixture behavior was hardened without changing its revision or
reset assertions.

## Exact disposition

| Alert | Rule / analyzed location | Disposition and evidence |
|---|---|---|
| 2355 | request-forgery; `services/tldw/request-core.ts:433` | Repaired initial and refresh redirect boundary and same-origin network-path bypass. See request regressions above. The configured-server/allowlist contract remains intentional. |
| 2251 | xss; `components/Common/AssistantSelect.tsx:774` | Guarded false positive: `normalizeCharacterSelection`/`normalizePersonaSelection` call shared selection normalization before rendering. `types/assistant-selection.ts:133–139` spreads `rest` and then overwrites `avatar_url` with `resolveAvatarUrl(candidate)`. That calls `safeImageUrl` and raster-only `createImageDataUrl` (lines 94–101). SARIF follows `rest` through the spread while omitting the overwrite. Existing component tests reject JavaScript and SVG-data URLs and preserve HTTPS avatars; image/selection tests also pass. No extra sink-only sanitization was added. |
| 2630 | insecure-randomness; `db/dexie/helpers.ts:1015` | Nonsecurity identifier: source is `generateID():37–39`, which returns a `pa_` chat-history ID. SARIF flows through `saveHistory` and `hooks/chat-helper/index.ts:550,610` into `addFileToSession(sessionId, file)`. `db/dexie/chat.ts:61–79` uses it as an IndexedDB session-files key; `schema.ts` declares `sessionId` as the table key. It is neither an authentication session nor an authorization bearer. No cryptographic claim is made about these IDs. |
| 2623 | insecure-randomness; `scripts/skills-certification/run.mjs:359` | Nonsecret path: `evidence.mjs:38` run-directory suffix → `webuiDir` → `TLDW_SKILLS_CERT_WEB_OUTPUT`. |
| 2624 | insecure-randomness; `scripts/skills-certification/run.mjs:360` | Same nonsecret directory source → `webReportPath` → `TLDW_SKILLS_CERT_WEB_REPORT`. |
| 2625 | insecure-randomness; `scripts/skills-certification/run.mjs:361` | Same nonsecret directory source → `webResultPath` → `TLDW_SKILLS_CERT_WEB_RESULT`. |
| 2626 | insecure-randomness; `scripts/skills-certification/run.mjs:443` | Same nonsecret directory source → `relayLedgerPath` → `TLDW_SKILLS_CERT_EXTENSION_LEDGER`. |
| 2627 | insecure-randomness; `scripts/skills-certification/run.mjs:444` | Same nonsecret directory source → `extensionDir/output` → `TLDW_SKILLS_CERT_EXTENSION_OUTPUT`. |
| 2628 | insecure-randomness; `scripts/skills-certification/run.mjs:445` | Same nonsecret directory source → `reportPath` → `TLDW_SKILLS_CERT_EXTENSION_REPORT`. |
| 2629 | insecure-randomness; `scripts/skills-certification/run.mjs:446` | Same nonsecret directory source → `resultPath` → `TLDW_SKILLS_CERT_EXTENSION_RESULT`. |
| 2620 | cors-misconfiguration-for-credentials; `extension/tests/e2e/service-prompts.spec.ts:352` | Replaced unrestricted JSON-response reflection with exact fixture-origin allowlisting. |
| 2621 | cors-misconfiguration-for-credentials; same file `:366` | Unresolved-scope preflight now shares the selected WebUI/extension allowlist. |
| 2622 | cors-misconfiguration-for-credentials; same file `:419` | Corrupt-state preflight now shares the selected extension allowlist. |
| 2653 | clear-text-storage-of-sensitive-data; `scripts/stats-probe.mjs:21` | Real operator key; exact-origin UAT seeding fix. Legacy `apiKey` mirror retained in the disposable selected context. |
| 2652 | clear-text-storage-of-sensitive-data; same file `:13` | Same fix and contract for `tldwConfig`. |
| 2651 | clear-text-storage-of-sensitive-data; `scripts/round3-verify.mjs:31` | Real operator key; exact-origin UAT seeding fix, minimal prior configuration retained. |
| 2650 | clear-text-storage-of-sensitive-data; `scripts/oversight-live-probe.mjs:23` | Real admin key; exact-origin UAT seeding fix, minimal prior configuration retained. |
| 2649 | clear-text-storage-of-sensitive-data; `scripts/admin-uat-interactions.mjs:25` | Real operator key; exact-origin UAT seeding fix. Legacy key mirror retained. |
| 2648 | clear-text-storage-of-sensitive-data; same file `:16` | Same fix and contract for `tldwConfig`. |
| 2647 | clear-text-storage-of-sensitive-data; `scripts/admin-uat-driver.mjs:109` | Real operator key; both authenticated contexts now use exact-origin seeding. Legacy key mirror retained. |
| 2646 | clear-text-storage-of-sensitive-data; same file `:100` | Same fix and contract for `tldwConfig`. |
| 2654 | clear-text-storage-of-sensitive-data; `scripts/shared-research-workspace-cdp-uat.mjs:2068` | False key-propagation path: `prepareContext` spreads existing config, then explicitly overwrites `apiKey` with `""` before storage. An executed initializer probe with a distinct key sentinel verifies empty replacement, metadata preservation, multi-user mode, and no persisted sentinel (4 assertions). New isolated contexts are created at lines 2694–2696; retained storage proof hashes the config rather than exporting its plaintext. |
| 2642 | clear-text-storage-of-sensitive-data; `e2e/workflows/presentation-studio-standalone-html.security.spec.ts:1661` | Synthetic fixture configuration. Callback receives constant `TEST_API_KEY = "TASK17-NONSECRET-TEST-KEY"` (line 14), including invocation at line 1762. No operator credential is loaded for this seed. Storage instrumentation and negative exfiltration assertions remain intact. |
| 2643 | clear-text-storage-of-sensitive-data; same file `:1670` | Same synthetic key written into the compatibility mirror; intentional fixture state. |
| 2644 | clear-text-storage-of-sensitive-data; same file `:1911` | Synthetic fixture principal transition: config copied from the above seed; access/refresh tokens removed on simulated logout. No real API-key source is introduced. |
| 2645 | clear-text-storage-of-sensitive-data; same file `:1937` | Simulated login writes deterministic `task17-owner-${nextPrincipal}-token` and the same synthetic config. Principal-change tests require this transition; assertions are unchanged. |
| 2616 | clear-text-storage-of-sensitive-data; `extension/tests/e2e/service-prompts.spec.ts:251` | Synthetic release-gate config. `requireGateConfig` rejects every key except the fixed fake key (lines 110–115) and requires loopback server/WebUI URLs. `seedWebUi` receives that gate config. |
| 2617 | clear-text-storage-of-sensitive-data; same file `:256` | Same guarded fake key, legacy storage mirror. |
| 2618 | clear-text-storage-of-sensitive-data; `e2e/workflows/service-prompts-runtime.spec.ts:485` | Synthetic release-gate config. `getRuntimeConfig` rejects an environment key unequal to `DISPOSABLE_API_KEY` (lines 89–94), whose value is the documented fake key at line 41. |
| 2619 | clear-text-storage-of-sensitive-data; same file `:490` | Same guarded fake key, legacy storage mirror. |
| 2357 | clear-text-storage-of-sensitive-data; `e2e/manual-api-key-persistence.spec.ts:184` | Intentional persistence test with `MANUAL_API_KEY = "manual-persistence-e2e-key"` from the local HTTP fixture. The test verifies legacy device-key migration and authenticated media after hard reload; neighboring tests distinguish device persistence from session-only storage. Removing the seed would remove the tested contract. |
| 2358 | clear-text-storage-of-sensitive-data; `e2e/single-user-cookie-lifecycle.spec.ts:471` | Intentional preservation test for a separate remote manual connection. `PRESERVED_REMOTE_API_KEY` is the same synthetic `MANUAL_API_KEY`; the fixture URL is loopback. This is distinct from environment-configurable initial/rotated cookie-session keys, which the surrounding test asserts are absent from browser storage. |

Paths in the table without an explicit app prefix are under
`apps/packages/ui/src`, `apps/tldw-frontend`, or the explicitly named extension
folder as appropriate to their first segment.

The seven certification paths are intentionally visible in retained evidence,
not secret tokens. `createSkillsCertificationEvidence` validates a single path
component, creates the run directory without recursive reuse (collision fails),
and marks the created directory; cleanup requires exact roots and markers.
The authentication fixture is a separate explicit constant in `profile.mjs`.
Changing the directory suffix to cryptographic randomness would not establish
an authentication security improvement. Existing evidence and runner tests
cover the path, marker, command, cleanup, and redaction contracts.

## Verification and limits

Independent review found two additional runtime edge cases. Non-string IPC
paths (including an array containing an external URL) bypassed the string-only
classification and were later coerced into a request URL. The entry point now
rejects non-string paths before configuration/credential lookup; all five new
cases failed against the prior source. A non-HTTP configured WebUI could also
share the opaque `null` origin with unrelated documents. The UAT helper now
requires HTTP(S); three file/data/FTP cases failed before that restriction.
The final combined request/seed run passes **65 tests** (55 request/allowlist,
10 seed), recorded in `/tmp/pr2761-js-review-fixes-final.log`. Red evidence is
`/tmp/pr2761-js-path-type-red.log` and `/tmp/pr2761-js-seed-protocol-red.log`.

A further review reproduced refreshed credentials being sent to the original
server after the active server changed. The wrapper now snapshots the original
configuration and uses the existing target matcher plus derived JWT user scope
to reject changed server/auth-mode/auth-source/organization/principal before
retrying. Five in-place mutation cases failed before repair. After this change,
the final combined run (including API-client scope tests) passes **77 tests**;
`/tmp/pr2761-js-review-fixes-final.log` now records that latest run. Normal token
rotation for the same principal remains supported. The red transcript is
`/tmp/pr2761-js-refresh-scope-red.log`. A changed scope produces the existing
structured 412 rejection so callers restart the operation in the new scope.

- Native request regressions: **50 passed** across six request/allowlist suites,
  including direct redirect, refresh redirect, two same-origin modes, and allowed
  cross-origin initial request. Red transcripts demonstrate both repaired gaps.
- CORS helper: **7 passed**; before the allowlist, five rejected-origin cases
  failed and two allowed-origin controls passed.
- UAT seed: initial copied seed behavior failed all five foreign-origin cases
  while the selected-origin control passed. **7 final tests pass**, preserving the
  minimal setup state used by oversight/round3.
- Existing avatar/image/selection and certification evidence/runner suites plus
  the seed helper: **144 passed**. These are behavioral/unit checks, not a
  complete UAT run.
- Frontend `bun run typecheck`: exit 0. Extension `bun run compile`: exit 0.
- All six changed `.mjs` modules pass `node --check`.
- Bandit was invoked with the project virtual environment on the touched JS/TS
  implementation scope. It reports Python AST syntax errors for all three
  files and cannot assess JavaScript; its exit 0 is **not** a security pass.
  JavaScript security evidence here comes from the reviewed dataflows and
  regression tests; a fresh CodeQL run remains necessary.
- Scoped ESLint reports no errors. Six existing `no-explicit-any` warnings remain
  in request-core; root invocation also emits the existing Next pages-directory
  lookup diagnostic. The newly introduced test cast was changed to `PathOrUrl`.
- Full WebUI/built-extension browser workflows were not executed; they require
  the disposable application/extension fixture stack. Their credential
  persistence, principal transition, reset, and exfiltration assertions were
  preserved. No live external credential or production service was exercised.

Local transcripts: `/tmp/pr2761-js-redirect-red.log`,
`/tmp/pr2761-js-origin-red.log`, `/tmp/pr2761-js-request-final.log`,
`/tmp/pr2761-js-seed-red.log`, `/tmp/pr2761-js-seed-green.log`,
`/tmp/pr2761-js-cors-red.log`, `/tmp/pr2761-js-cors-green.log`,
`/tmp/pr2761-js-guard-evidence-tests.log`, `/tmp/pr2761-js-storage-proof.log`,
`/tmp/pr2761-js-typecheck.log`, `/tmp/pr2761-js-extension-compile.log`,
`/tmp/pr2761-js-shared-eslint.log`, `/tmp/pr2761-js-bandit.json`. These are local evidence references, not
committed artifacts. Analyzer closure and any per-alert state decision require
separate current-candidate analysis/review; passing runtime tests alone does not
establish CodeQL closure.

## A440 rescan: four newly identified alerts

These four alerts are separate from the prior approved disposition inventory.
This assessment changes no alert state. Source references below match the A440
SARIF and the subsequent F7 candidate before the UAT memory-storage repair
described below. The test environment override was removed after A440 so the real HTTP
regressions also run with the owning UI package's jsdom setup.

| Alert | Exact source and sink | Classification and boundary evidence |
| --- | --- | --- |
| 2675 | Four SARIF flows: `RunDetailDrawer.tsx:531` WebSocket `event.data` → `parseWatchlistsRunStreamPayload` → snapshot/run-update `run.job_id` → `handleRetryRun` at 1019 → `watchlists.ts:785` fixed `/api/v1/watchlists/jobs/${jobId}/run` → `request-core.ts:457` initial fetch. | False positive for the reported request-forgery flow. `watchlists-stream.ts:61–79` converts the value to an integer, rejects a failed conversion, and returns a fresh object containing the numeric `job_id`. Its string representation cannot introduce URL separators or change the destination. The API path prefix and suffix are fixed. Independently, request-core rejects non-string runtime paths before configuration access, enforces exact configured/allowlisted absolute origins, prevents same-origin network-reference escapes, suppresses configured credentials for an explicitly selected external origin, and uses `redirect: "error"`. |
| 2676 | The same four source flows and integer conversion, ending at `request-core.ts:525` refresh retry. | False positive for the reported request-forgery flow. The numeric/fixed-path boundary above still applies. The retry reuses the validated URL, forbids redirects, and compares the original configuration snapshot with refreshed server/auth mode/auth source/org and derived JWT principal before attaching a refreshed credential (`request-core.ts:499–505`). A changed scope throws 412 before retry dispatch. |
| 2673 | Four SARIF flows from the real `apiKey` parameter at `browser-uat-seed.mjs:4`, through the config object at 10 and JSON serialization, into `localStorage.setItem("tldwConfig", …)` at 11. | Confirmed plaintext-storage finding, now repaired in source; awaiting a fresh scan. After the HTTP(S) and exact-origin guard, the UAT helper replaces both native storage properties with document-memory facades and seeds the config directly into `memoryLocal` (current helper line 46). All subsequent application storage calls use the facades. Native storage is never the fallback. |
| 2674 | One SARIF flow from the same real `apiKey` parameter at 4 into the legacy `localStorage.setItem("apiKey", apiKey)` at 18. | Confirmed plaintext-storage finding, now repaired in source; awaiting a fresh scan. The legacy key is seeded directly into `memoryLocal` (current helper line 53), with the same origin/facade boundary and an additional `legacyBootstrap` requirement. Admin driver/interactions and stats retain their legacy fields; oversight/round3 retain their minimal seed. |

The request classifications concern the exact reported flows. The integer parser
accepts numeric prefixes (for example, `42?next=…` becomes the number 42); it is not
a strict identifier syntax validator, but that behavior does not permit URL
injection. These browser request helpers deliberately support the operator's
configured server, including local/private addresses. They are not a general
server-side network allowlist, nor do they replace backend job authorization.

For the two storage alerts, origin confinement is not encryption. A compromised
selected WebUI origin, an operator-selected untrusted origin, or access to the
active browser context can expose the key. The helper deliberately permits HTTP
for local self-hosted UAT; it does not enforce HTTPS for remote deployments. The
regressions demonstrate origin confinement and absence from native Web Storage,
not protection against renderer compromise or those transport risks. Removing the key or substituting a fake
one without a compatible authentication bridge would break these real-server UAT scenarios. No synthetic-key rationale is
claimed for either new storage alert.

Alternatives assessed before the memory-storage repair:

- Keeping the key in Node and using `route.fetch({ maxRedirects: 0, headers })`
  followed by `route.fulfill({ response })` can authenticate ordinary HTTP
  calls without putting the real key in renderer requests. Such a handler must
  require both the selected server destination and an initiating frame from the
  exact selected HTTP(S) WebUI origin; requests from foreign frames or without a
  frame must not receive authentication. `route.continue` header replacement is
  unsuitable because browser redirects can forward the replacement credential.
- This is not currently a drop-in replacement for all UAT traffic. Installed
  Playwright 1.58 buffers response chunks until completion in
  `playwright-core/lib/server/fetch.js:343`; the `route.fetch` result therefore
  cannot preserve the active notification SSE stream at
  `services/notifications.ts:187–188`. These scripts explicitly account for
  persistent streams during page settling and collect resulting errors.
- `RunDetailDrawer.tsx:516–517` opens authenticated Watchlists WebSockets whose
  `api_key` query comes from config. Installed `WebSocketRoute.connectToServer()`
  has no URL/header override, so replacing the stored key with a marker also
  requires a separate authenticated WebSocket relay to preserve this behavior.
  Quickstart additionally sends API requests through the WebUI origin rather
  than directly to the configured API origin; a server-only route misses them.
- Existing `saveManualSingleUserCredential({ persistence: "session" })` is not a
  no-storage alternative: `TldwApiClient.ts:2164` writes the real key to its
  session storage adapter. Its memory fallback occurs only after storage fails,
  and is not an explicit supported memory-only mode. The lower-level runtime
  override setter is module-scoped, lacks an installed browser UAT bootstrap
  bridge, and does not replace all config-based WebSocket consumers. A window
  property alone is therefore not a verified replacement either.

The implemented repair is confined to the shared manual UAT helper. A small
Map-backed Storage facade replaces both global storage properties before seeding
or app boot. Both property descriptors are checked first; installation failure
throws before any credential is seeded, with no native fallback. Direct seeding
uses `memoryLocal.setItem`, whose implementation calls `Map.set`. The facade
supports get/set/remove/clear, null for missing keys, string coercion, key/length,
named properties and key enumeration. Wrong/opaque origins return before changing
either storage property. Genuine device/session-persistence E2E tests are untouched.

This is deliberately document-scoped UAT state: reload and hard navigation discard
app changes and rerun the initializer; same-origin frames have independently
seeded Maps. It does not emulate cross-tab storage events or durable storage, and
it is not a complete branded browser Storage implementation. The five scripts
inspect admin pages, server statistics, selected-user oversight, role/forms and
layout/first-steps behavior. They do not assert that browser settings, logout, or
credentials persist across reload. Backend changes such as role creation are
unaffected. The real credential remains in renderer memory for normal HTTP/SSE/
WebSocket authentication; it is not stored in native local/session storage.

Compatibility with the frontend Plasmo and wxt storage adapters was reviewed:
they capture/read the global storage properties and use the supported methods,
so installing before app boot covers their later writes. The browser fixture
uses application-shaped storage calls, not a full app boot or the real
`saveManualSingleUserCredential` method; SSE/WS compatibility is a source-based
assessment, not a live-server UAT claim. No request interception or app/network
behavior was changed. No alert was dismissed; analyzer closure needs a fresh scan.

Verification on the current candidate:

- Owning UI config: six suites, **59 tests passed** (`request-core` redirect,
  quickstart, hosted, refresh/timeout; absolute URL guard; Watchlists stream).
  The 13 redirect/security cases include native disposable loopback HTTP initial
  and post-refresh 307 proofs, supported explicit external requests without the
  server credential, five runtime path-type cases, and five scope-change cases.
- Frontend seed suite: **12 tests passed**, including captured-native-store
  checks after seed and application-shaped writes, locked-descriptor fail-closed
  behavior, three unsupported protocols, five wrong/opaque origins, selected
  origin config reads, and minimal versus legacy bootstrap. Together with the
  existing dev-runtime/live-tier UAT suites: **61 tests passed**.
- Standalone real Chromium regression: **1 passed**. Before the repair it failed
  with nine native local-storage entries. Afterward both captured native stores
  remain empty after seeding and application-shaped writes. It checks Chromium's
  own configurable accessor descriptors, storage methods/coercion/enumeration/
  named properties, reload reseeding, same/foreign-origin frames and navigation,
  and absence of credentials from `context.storageState()` export. Run with
  `node --test scripts/__tests__/browser-uat-seed.browser.test.mjs` from the
  frontend package. The `.mjs` test is outside default Vitest `.ts/.tsx` globs,
  so the frontend unit CI lane does not acquire an undeclared Chromium dependency.
- Frontend typecheck and changed-module syntax checks pass.
- A read-only execution of the actual exported Watchlists parser checked **32
  cases** across both snapshot and run-update events: URL/network references,
  traversal strings, numeric prefixes, arrays/objects/null/nonfinite values, and
  valid numbers. Every accepted value was an integer whose fixed API path kept
  the configured origin. This is additional local evidence, not a new committed
  regression suite.
- The earlier exact owning UI CI shard 6/8 replay passed **42 files / 545 tests**
  after removing only the incompatible Node environment directive. Network
  guards, package setup, and production source were unchanged.

Local evidence: `/tmp/pr2761-js-four-traces.json`,
`/tmp/pr2761-js-four-watchlists-proof.json`,
`/tmp/pr2761-js-four-transport-tests.log`, `/tmp/pr2761-js-four-seed-tests.log`,
and `/tmp/pr2761-ui-shard6-jsdom.log`. No live credential or external server was
used. Additional repair transcripts: `/tmp/pr2761-uat-memory-browser-red.log`,
`/tmp/pr2761-uat-memory-browser-green.log`, `/tmp/pr2761-uat-memory-focused.log`,
`/tmp/pr2761-uat-memory-typecheck.log`, and `/tmp/pr2761-uat-memory-eslint.log`.
The first non-escalated related-UAT run hit a sandbox loopback-bind restriction;
the unchanged suites passed with their existing local fixture permissions.
