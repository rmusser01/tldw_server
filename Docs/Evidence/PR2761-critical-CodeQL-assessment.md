# PR #2761 critical CodeQL assessment

**Date:** 2026-09-10. **Tracking:** TASK-13013.3; [release plan](../superpowers/plans/2026-09-10-pr2761-release-refresh-plan.md).

This assesses the seven critical alerts in [aggregate check 102920723916](https://github.com/rmusser01/tldw_server/pull/2761/checks?check_run_id=102920723916), which failed with 379 reported new alerts (7 critical, 367 high, 5 medium). The large-diff warning means “new” is not proof that the release introduced the underlying behavior.

The completed [default CodeQL run](https://github.com/rmusser01/tldw_server/actions/runs/34491672201) analyzed `7d7a2e708dd2e2621199bfbcaa709f9e44e76026`. Its Python analysis `1755691860` contains 566 results, with no ingestion error. All seven alert instances below identify that commit. The three affected source files are unchanged in pushed candidate `7c79e085dfcc19b1ce40cf0783681100a815ced9`. Main comparison uses `d9c245ac14c40df855d1ab6cd19b3c137b16b47b`.

## Exact alerts and baseline

| Alert | Rule and candidate sink | Main comparison |
|---|---|---|
| [2604](https://github.com/rmusser01/tldw_server/security/code-scanning/2604) | `py/full-ssrf`; `tldw_Server_API/app/core/http_client.py:3000`, aiohttp request | Main has the same request operation at line 2282 under open alert 2338. Candidate changes response handling; alert-ID correspondence alone is not asserted. |
| [2603](https://github.com/rmusser01/tldw_server/security/code-scanning/2603) | `py/full-ssrf`; `http_client.py:2862`, bounded httpx stream | New bounded-response branch of the existing validated transport. Main has no corresponding stream branch. |
| [2337](https://github.com/rmusser01/tldw_server/security/code-scanning/2337) | `py/full-ssrf`; `http_client.py:2893`, pinned httpx request | Same alert ID is open on main at line 2182; pinned transport operation already exists there. |
| [2602](https://github.com/rmusser01/tldw_server/security/code-scanning/2602) | `py/xpath-injection`; `tldw_Server_API/app/core/Web_Scraping/selectors/engine.py:294`, bounded XPath wrapper | New result-limiting wrapper around a user-authored selector. Main already evaluates user-authored XPath, but lacks this wrapper. |
| [2601](https://github.com/rmusser01/tldw_server/security/code-scanning/2601) | `py/xpath-injection`; `selectors/engine.py:252`, uncached compilation | Extracted compilation behavior: main `Watchlists/fetchers.py:977` calls `XPath(stripped)`. No same-ID main alert. |
| [2600](https://github.com/rmusser01/tldw_server/security/code-scanning/2600) | `py/xpath-injection`; `selectors/engine.py:235`, cached compilation | Extracted compilation behavior: main `Watchlists/fetchers.py:371` calls `XPath(expr)`. No same-ID main alert. |
| [2355](https://github.com/rmusser01/tldw_server/security/code-scanning/2355) | `js/request-forgery`; `apps/packages/ui/src/services/tldw/request-core.ts:487`, fetch | Same alert ID is open on main at line 475. |

Alerts 2600–2604 were first created on 2026-08-23; 2337 and 2355 on 2026-07-16. Creation dates distinguish alert history, not exploitability.

## Source evidence and limits

**HTTP alerts 2604, 2603, 2337:** Public async wrappers validate the destination through `_avalidate_egress_or_raise` before transport (`http_client.py:3301`, `3717`). `_validate_egress_or_raise` delegates to central policy, retains accepted DNS answers and raises on denial (`1681–1762`). `_prepare_pinned_transport_target` constructs transport URLs using an accepted IP while preserving HTTP Host and TLS identity (`1576–1636`). Calls into the flagged transports pass those accepted addresses (`3360–3378`, `3778–3796`); httpx automatic redirects are disabled, and aiohttp sets `allow_redirects=False`. The bounded response branch preserves this path. This supports a guarded-sink assessment; it is not proof that every caller, configuration or redirect combination is safe.

**XPath alerts 2600–2602:** These APIs intentionally accept complete selectors for extraction from supplied/fetched HTML, rather than inserting a value into an authorization query or querying a secret database. Runtime selection checks `_selector_safety_error` before compilation (`engine.py:266`); validation checks it through `_selector_validation_error` (`167–176`). Limits include expression length, descendant steps, predicates and function count; axes, unions, variables and wildcard descendants are rejected. The bounded wrapper evaluates on the provided HTML node and caps delivered matches. No custom XPath extension registration was found in the selector package. This is consistent with intentional expression evaluation inherited from Watchlists, with new resource limits, but is not an exhaustive proof against resource exhaustion or all extraction-context boundaries. Related implementation record: TASK-12989.1.

**Browser alert 2355:** `request-core.ts:343–353` rejects cross-origin absolute URLs unless explicitly allowlisted, before the fetch. `absolute-url-guard.ts` parses URLs, allows HTTP(S) only and compares normalized origins against the configured server and explicit allowlist; `request-core.ts:363` prevents automatic server authentication for cross-origin absolute requests. This supports the existing intended configured-server/browser transport boundary. Helper tests do not prove the safety of every injected fetch implementation, caller-provided header or redirect behavior. Related guard extraction record: TASK-12095.

## Focused verification

Commands used normal repository fixtures without network-guard bypasses, after activating the project virtual environment. All exited 0.

| Command / scope | Result and established behavior |
|---|---|
| `python -m pytest tldw_Server_API/tests/http_client/test_http_client.py tldw_Server_API/tests/http_client/test_http_client_stream_timeouts.py tldw_Server_API/tests/http_client/test_http_client_pinning.py -q -k 'egress or pin or private'` | **28 passed**, 114 deselected, 4 warnings. Includes private-IP denial, accepted-IP transport/Host/SNI behavior and propagation of pin denials. Controlled test transports; no live network attack simulation. |
| `python -m pytest tldw_Server_API/tests/Web_Scraping/test_phase4_selectors.py -q -k 'bounded_xpath or sanitizes_xpath or fixture_matches_predecessor'` | **21 passed**, 138 deselected, 4 warnings. Establishes selected predecessor parity, bounded node-set/attribute/scalar behavior, and sanitized compile/evaluation errors. |
| From `apps/tldw-frontend`: `bunx vitest run ../packages/ui/src/utils/__tests__/absolute-url-guard.test.ts --reporter=dot` | **9 passed**. Includes rejection of an attacker origin, configured-origin acceptance, and cross-origin allowlist/auth decisions. |

Local transcripts: `/tmp/pr2761-critical-http-review.log`, `/tmp/pr2761-critical-xpath-review.log`, `/tmp/pr2761-critical-url-guard-review.log`.

**Unresolved:** No additional exploitable bypass among these seven alerts was demonstrated by this bounded review. They remain open; this document does not classify all seven as false positives or satisfy the failing aggregate gate. Current-candidate analysis and individual security review remain necessary. No thresholds, suppressions, alert states or source behavior were changed for this assessment. The separately demonstrated slash-parser denial of service (high alert 2599) was fixed and tested in the pushed batch; it is not one of these seven critical alerts.

## Supplement: partial scan of 7c79e085df

The [candidate default run](https://github.com/rmusser01/tldw_server/actions/runs/34495652240) has completed JavaScript analysis `1755853801` (85 results, no ingestion error) and Actions analysis `1755785764` (239 results, no ingestion error). Its [Python job](https://github.com/rmusser01/tldw_server/actions/runs/34495652240/job/102933313666) was still running at this review. Consequently, [aggregate check 102933918456](https://github.com/rmusser01/tldw_server/runs/102933918456) reports **272** changed-code alerts (1 critical, 268 high, 3 medium) and explicitly says Python is missing. This is not a verified reduction from the earlier complete 379-alert aggregate. Alert 2599 still references the pre-fix Python analysis; its analyzer closure remains unverified.

### Speech persistence: replacement alert 2668

[Old alert 2655](https://github.com/rmusser01/tldw_server/security/code-scanning/2655) became fixed for this PR, but [replacement alert 2668](https://github.com/rmusser01/tldw_server/security/code-scanning/2668), `js/clear-text-storage-of-sensitive-data`, flags `SpeechPlaygroundPage.tsx:1061–1070`. JavaScript therefore still has 37 open alerts, not 36.

The downloaded SARIF contains four paths for this result. Each path passes from the OpenAI API-key getter into `getTTSSettings` through the same imprecise tuple transition:

1. `tts.ts:553`, `getOpenAITTSApiKey()`, is **zero-based array position 13** in the `Promise.all` input.
2. At `tts.ts:538`, SARIF changes `[13, PromiseValue]` to `[PromiseValue, ArrayElement]`; the destructuring at line 499 likewise carries only `[ArrayElement]`.
3. The path then selects `tldwTtsBackend` at `tts.ts:523`, which is actually **position 20**, populated by `getTldwTTSBackend()` at line 562. Position 13 populates `openAITTSApiKey` at line 514.
4. That backend field flows through `configuredTldwBackend` and `selectedTldwBackend` (`SpeechPlaygroundPage.tsx:889, 911–915`) into the persisted `backend` field at line 1067.

The persistence fix explicitly projects provider, voice, model, backend and fallback fields instead of spreading the selection object. Existing regression tests establish that extra credential fields in a selection are excluded. For the new trace, a bounded local check executed the actual parsed `getTTSSettings` initializer with distinct deterministic getter results: the API-key sentinel remained in `openAITTSApiKey`, and `tldwTtsBackend` contained only the backend getter result. All three mapping assertions passed. This verifies tuple pairing with stubbed getters; it is not a live credential-storage test. The trace and source provide specific evidence of array-index imprecision for this replacement finding. The alert remains open and requires review; no source restructuring or suppression was used to force analyzer closure.

Local evidence: `/tmp/pr2761-codeql-7c79-js.sarif.json` and `/tmp/pr2761-speech-sarif-tuple-proof.log`.

### Added Actions alert and count reconciliation

[Alert 2667](https://github.com/rmusser01/tldw_server/security/code-scanning/2667), `actions/cache-poisoning/poisonable-step`, flags the added WebUI typecheck step at `.github/workflows/frontend-required.yml:640–645`. Its message explicitly combines the checkout expression at line 596 with `workflow_dispatch`. On that event, admission is skipped and the PR/workflow-run payload properties are absent, so the expression falls through to the selected ref's `github.sha`. The untrusted PR-head alternatives belong to different events. This is the same cross-event source combination identified in the earlier cache-alert review; adding a command adds another flagged sink. It does not demonstrate a new workflow-run cache-write exploit. The inspected CodeQL Actions model associates jobs with workflow trigger events without fully evaluating these event-specific alternatives. Review remains open, particularly for any separate cache-write trust issue; the typecheck gate and analyzer thresholds are unchanged.

The branch-open snapshot changed from **448 to 449**: old 2655 disappeared, replacement 2668 appeared, and Actions 2667 appeared. JavaScript remains 37 open; Actions increased from 238 to 239; 173 Python instances still came from the old `7d7a2e7` analysis. Branch-open totals and changed-code aggregate totals are different measures. A complete current-head Python analysis is required before reconciling the final aggregate against 379.

## Completed scan follow-up: 30338ef7de

All three current-source analyses completed without ingestion errors: Python `1756055450` (564 results), JavaScript `1755993012` (85), Actions `1755924298` (239). The prior 7c79 Python analysis also completed with 564 results. The partial-scan caveats above describe that earlier observation only.

The branch alert API now reports **448 open instances**: 172 Python, 37 JavaScript and 239 Actions; severity distribution is 7 critical, 435 high and 6 medium. **Slash-regex alert 2599 is fixed** for this PR. Old Speech 2655 is fixed and replacement 2668 remains open as assessed above. The aggregate [check 102941684745](https://github.com/rmusser01/tldw_server/runs/102941684745) still reports failure and 379 changed-code alerts (7 critical, 367 high, 5 medium). Its output timestamp predates the completed Python scan; do not equate its total with the branch-open inventory or manufacture a green result. No alerts were dismissed. Snapshot: `/tmp/pr2761-30338-all-alerts.json`.

## Follow-up: verified source fixes after decdf9db77

The fresh `decdf9db7756031c379d5b019b0436711b9068b1` inventory still has **448 open alerts** (7 critical, 435 high, 6 medium), all with instances on that exact commit. Python analysis `1756430231` (564 results), JavaScript `1756419893` (85), and Actions `1756356894` (239) completed without ingestion errors. [Aggregate check 102966050262](https://github.com/rmusser01/tldw_server/runs/102966050262) still fails with 379 changed-code alerts. This is a source/security gate failure, not a missing-language or upload failure. Snapshots: `/tmp/pr2761-fresh-codeql-{head,alerts,analyses,check}.json`.

### Exception details exposed in responses

Three additional high-severity alerts have concrete, locally reproduced disclosure paths:

- [2120](https://github.com/rmusser01/tldw_server/security/code-scanning/2120) and [2121](https://github.com/rmusser01/tldw_server/security/code-scanning/2121): single and bulk DLQ requeue handlers interpolated the complete schema-validation exception into response warnings. A real `jsonschema` rejection included the invalid input value, schema, and instance details. The regression supplies a synthetic secret/path as an invalid media ID and reproduces its inclusion in both successful admin responses.
- [2119](https://github.com/rmusser01/tldw_server/security/code-scanning/2119): `_check_runner_binary` returned the configuration-loader exception verbatim, which reached the authenticated ACP health response. The regression supplies a configuration error with a synthetic secret and local path and reproduces their disclosure.

The handlers now return fixed messages and log only the exception class at these sites. DLQ requeue still returns HTTP 200, moves the item, deletes the requested DLQ entry, closes the Redis client, and includes a warning. ACP health still returns HTTP 200 with an error runner status and an unavailable overall status. These are response-disclosure fixes; they do not change the existing authorization requirements or classify all diagnostic endpoints as safe.

All three regression cases failed for the expected sensitive-response mismatch before the source changes. After the fixes and scoped Ruff cleanup, **31 tests pass** across the DLQ admin, DLQ quarantine, and ACP health suites. Two existing warnings remain: the Starlette/httpx deprecation and unknown pytest `plugins` configuration. Touched Python source and tests pass Ruff; Bandit reports zero findings with test assertions (`B101`) excluded. Runtime-only baseline Bandit also reported zero findings. Logs: `/tmp/pr2761-codeql-disclosure-{red,final}.log`, `/tmp/bandit_pr2761_disclosure_final.json`.

### Speech credential/preference separation

`getTTSSettings` now reads noncredential preferences in a private helper and keeps each credential result tied to its own named promise. All reads start concurrently; one `Promise.all` barrier attaches rejection handling without destructuring its mixed results. The public settings shape is unchanged. This removes the specific credential-to-preference tuple path described for alert 2668 without adding a suppression.

New service tests verify distinct credential/preference sentinels, concurrent read initiation, and failure propagation while preferences remain pending. These tests also passed before the refactor: the runtime pairing was already correct, and this change makes the source boundary explicit. The Speech UI still legitimately reads credentials for provider setup; this is not a claim that credential access or credential persistence elsewhere was eliminated. After the aligned dependency installation, **42 tests pass** across TTS defaults and both Speech suites; the final defaults-only run passes **7 tests**. Logs: `/tmp/pr2761-tts-aligned-after.log`, `/tmp/pr2761-tts-defaults-final.log`.

### Remaining work and disposition boundary

None of the four targeted alerts is claimed fixed by GitHub until a new analysis includes these source changes. No alert has been dismissed, and no query, threshold, workflow trigger, or suppression was changed.

The remaining inventory is not wholly an external/manual blocker. In particular, **151 path-injection alerts** still require source/sink-specific boundary investigation, and **238 cache-poisoning alerts** span mixed-event workflows and composite actions. Separating privileged/manual execution from PR execution is a possible source-level remediation, but is a substantial workflow change that must preserve admission, exact commit selection, cache ownership, and required gate behavior. The observed cross-event modeling limitation does not justify blanket dismissal or imply that all cache-writing paths are safe. The separate trusted-license checkout alert also remains unresolved.

The bounded MIME validation behind alert 2598 already rejects inputs longer than 255 characters; the remaining two exponential-regex alerts occur in deliberate adversarial regex tests. The seven critical transport/XPath/browser findings retain the guarded/intentional-expression assessments above. Those observations support focused review, not a clean security verdict. Current-head analyzer execution is external evidence; security disposition requires individual review. Further source remediation remains agent work where a bypass or inadequate boundary is demonstrated.
