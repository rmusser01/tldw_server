# UAT135 — outage diagnostics incorrectly blame credentials and CORS

## Confirmed observation

`/private/tmp/cycle5-single-native-287-offline-qa.txt` captures Knowledge QA during the deliberate API outage. Main copy correctly says the configured server is unreachable, but its checks simultaneously say Credentials Missing, Browser access Blocked, and backend Waiting for credentials; the primary action is Update credentials. Last error asserts an origin is not allowed and suggests ALLOWED_ORIGINS or disabling CORS. Root reports Settings had restored the key and correctly reported Core unreachable. This is a bounded diagnostic/presentation defect (P3 appropriate), not new evidence of missing credentials or a real CORS denial.

## Exact causes

1. `apps/packages/ui/src/store/connection.tsx:443–500`, `maybeAnnotateCorsMismatchError`: generic `Failed to fetch`, NetworkError, Load failed and even aborted-operation text qualifies as a network block. Different browser/backend origins plus status0 then produce “Likely CORS mismatch: [origin] is not allowed...” and disable-CORS advice. A stopped server produces the same browser-visible fetch failure; different development ports alone cannot establish the cause.
2. `apps/packages/ui/src/components/Option/KnowledgeQA/SetupDiagnostics.tsx:46–51,94–98,138–142`: the renderer regex matches the store-generated `cors` / `not allowed` text and converts that unproven suggestion into a definite Browser access Blocked state.
3. Same component line127–132: `authMissing` includes `connection.configStep === 'auth'` independent of current failure/UX classification. The store sets that onboarding step when credentials change (`connection.tsx:1187–1218`) or are missing (line827), preserves an existing auth step even after valid credentials resolve (`deriveOnboardingConfigStep`, line615), and does not update it in successful/failed health-result assignment (lines1036–1057). Thus a restored valid key can coexist with `configStep:auth`, `errorKind:unreachable`, `uxState:error_unreachable`.
4. That stale-step predicate drives Credentials Missing, backend Waiting, and Update credentials, overriding the actionable network failure. It is not inference from a status0 response itself. Native internal state was not captured, but this is the component branch consistent with the captured title/check combination and is independently reproduced below.

The Request host access button is rendered when the host-permission API appears available; that button alone does not prove a permission denial. This report does not expand into notification or browser-shim behavior seen elsewhere in the screenshot.

## Private interacting regression

`/private/tmp/cycle5-uat135-offline-diagnostics-probe.config.ts` adds two cases by a read-only Vite transform to the existing connection suite. It executes actual `checkOnce` with a configured synthetic restored API key and normalized transport result, then mounts actual `KnowledgeQASetupDiagnostics` using the resulting state and real `deriveConnectionUxState`.

Command from `apps/packages/ui`: `bun run test --config /private/tmp/cycle5-uat135-offline-diagnostics-probe.config.ts`.

`/private/tmp/cycle5-uat135-offline-diagnostics-probe.log`: **1 RED / 1 GREEN**, no unhandled errors:

- status0 Failed to fetch: real state remains configStep auth / errorKind unreachable; manufactured CORS assertion, Credentials Missing, Browser Blocked, backend Waiting and absent Retry connection all reproduce.
- actual401 control: auth classification and Update credentials remain correct under the existing presentation contract.

No HTTP/browser/inference occurred. Transport/config are controlled test seams; the private component test is not native acceptance. Existing connection tests explicitly expect the misleading CORS hints for generic network/abort failures, so those expectations must change with the correction rather than be cited as correct behavior.

## Smallest post-freeze correction

- Keep generic network/timeout/abort failures cause-neutral. Preserve the original failure and offer server URL/reachability/retry guidance. Do not invent an origin-denial fact or disable-CORS remedy from an opaque browser fetch error. Reserve blocked browser/allowlist status for affirmative existing denial evidence; explicit extension allowlist failures must remain actionable.
- In SetupDiagnostics, current authoritative auth/UX failure must outrank historical onboarding-step metadata. An unreachable result after credentials were configured should say configured (not successfully verified), show backend unreachable, and offer Retry connection. `configStep:auth` alone must not assert missing credentials during an unrelated current error. Narrowly normalize the step after a completed credential-presence check if needed, but avoid new auth abstractions or duplicating credential storage reads.
- Likely bounded production files: connection.tsx and SetupDiagnostics.tsx plus their existing tests. Coordinate connection.tsx with UAT134 / existing TASK13260.24; no concurrent overlapping edits.

Permanent controls: restored-key→status0 outage through store and real diagnostics; missing-key early stop; real401/403 auth contract; explicit extension allowlist denial; generic cross-origin/same-origin fetch failure, abort/timeout; reconnect200 clears outdated outage guidance; no misleading host-access action as the primary recovery for an unproven network cause. Native repeat of the existing deliberate outage after freeze is sufficient—no new inference needed.

## Retention / limits

No repository/source/test/task/browser/runtime changes. SHA256 records for five production/test/probe paths are in `/private/tmp/cycle5-uat135-diagnosis-hashes.txt`.

Requested UAT133 controller copies are byte-identical (`cmp` verified):

- `/private/tmp/cycle5-controller-uat133-final-payload-probe-source.txt`
- `/private/tmp/cycle5-controller-uat133-final-payload-probe.log`

They copy the independent final adapter-payload probe source/log; root's original probe remains unchanged. No source implementation started; still waiting for explicit release.
