# PR #2761 JavaScript CodeQL remediation and review

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
