# UAT248 — captured Character request lifetime

Associated task: TASK13260.190. This is separate from the integrated UAT231/232/236/243 unit and from UAT246 timeout diagnosis. Root approved three production paths and owns native acceptance, task records, and integration.

## Problem and fix

The independent real-auth probe showed that WebUI logout aborted both captured Service Prompt lease signals while the original caller signal remained active. Late Character success then called persistence for the old conversation without request scope. The replacement draft stayed visible. This proved stale client dispatch, not a successful native cross-account write.

Character execution now uses the existing captured scope signal, while retaining the original caller signal for controller ownership and deliberate cancellation semantics. Existing captured request options accompany create, greeting/user writes, streaming, normal completion persistence and fallback. Guards following awaited prerequisites/results prevent subsequent old-owner writes and publication. The existing success-save boundary receives scope signals/requestScope, and queued streaming updates are suppressed after invalidation. Partial recovery continues to distinguish owner invalidation from ordinary same-owner failures.

The domain stream method forwards the existing requestScopeFields metadata outside the inference body. The existing request-path policy now permits only POST to the exact complete-v2 path. The real scoped stream otherwise rejected the request before fetch; this prerequisite was separately approved after its causal RED. No new authority framework, timeout adjustment, cache, backend, or scheduler change was added.

## Exact owned files

Three production files under apps/packages/ui/src:

- hooks/chat/useChatActions.ts — Character function only.
- services/tldw/domains/chat-rag.ts — streamCharacterChatCompletion only.
- services/tldw/service-prompt-scope-error.ts — exact POST complete-v2 allowlist alternative.

Three test files:

- hooks/chat/__tests__/useChatActions.character.integration.test.tsx — opt-in actual lease/auth logout controls, existing lifecycle controls, and scoped-option expectation alignment.
- services/tldw/__tests__/TldwApiClient.request-scope.test.ts — actual domain adapter forwarding, unchanged inference body/idle budget.
- services/tldw/__tests__/service-prompt-scope-error.test.ts — exact route/method positives and wrong-method/path/traversal negatives.

owned-manifest.json SHA256 **6dd7a5e19cdd9f5a2bf91192ad0dc3dbb417891561957f3c335686846a1247a4** binds all six current files and review-snapshot copies. Baseline f1cefecb22ae2447a53e2ffdb5cda67866ccf4a4; parent integrated disjoint work before freeze HEAD e82c0ba845e038a44d9aab4f90d04fab3511ed6d. owned.patch is limited to these six files.

## Causal proof and positive controls

- reviewer-* retains the exact independent actual-auth failure, probe, command, report and hash map. author-actual-auth-causal-red.log independently repeats it:1 failed /33 filtered.
- final-baseline-red.log nonmutating loader replays final permanent tests against the three retained pre-fix production files: **14 expected failures /9 passing controls /116 filtered**. Creation ACK/greeting ACK also have separately retained2RED receipt. No production rollback or test weakening.
- scoped-stream-allowlist-red.log: actual same-owner captured stream fails before fetch with the existing policy rejection, prior to the exact allowlist correction.
- Actual-auth cases cover late SSE success/failure, partial output then logout, pre-dispatch prerequisite, acknowledged user, visual metadata await, persistence success/failure, new-chat ACK and greeting ACK. Owner changes suppress fallback/local success/error writes and preserve replacement draft.
- Same-owner success, same-owner network-error partial recovery, deliberate caller stop, original acknowledged IDs, one user write/no replay, existing unavailable-model Retry, successful branching, workspace, greeting and emote controls remain covered. Caller stop intentionally does not remotely persist partial text; non-abort same-owner failure can recover partial text.

The actual transport controls use real useChatActions → streamCharacterChatCompletion → bgStream → controlled fetch/ReadableStream, and real loadServicePromptSnapshot plus authService.logout. Persistence endpoints and local storage writers are controlled mocks at the side-effect boundary. The separate domain test checks expected-user header/config forwarding. This is not native server authorization or provider testing.

## Final verification

Run exact argv arrays from commands.json at repository root. The private Vitest config reuses the frontend config and the existing installed Bun pa-tesseract.js path; no install or product configuration change.

| Check | Result |
| --- | --- |
| Focused final, default configuration | **139 passed,0 skipped,3 files**; focused-default-green.log |
| Relevant adjacent, default configuration | **456 passed,7 failed,8 files**,37.16s; adjacent-default-config.log |
| Adjacent fixture baseline | Same **7 failed /118 passed** in unchanged background-proxy.test.ts with original248 production files; background-proxy-baseline.log |
| TypeScript full frontend differential | **90 baseline /90 current,0 added/removed**; tsc-comparison.json |
| Scoped ESLint | **0 errors;274 baseline →273 current warnings**, one no-explicit-any warning removed; zero added; all six logical paths parsed |
| Bandit | **0 findings,6 TypeScript/TSX parse errors**; unsupported-language limitation, not security proof |
| Diff whitespace | Pass; diff-check.txt |

The seven unrelated adjacent failures assert no console warnings while their shared spy retains four/five/nine genuine warnings from prior tests. All seven reproduce on the original248 source. Parent explicitly retained the baseline and assigned a separate fixture repair; this unit does not edit that file. Initial adjacent run forced Quickstart globally and additionally affected URL/coalescing expectations (11 failures); final commands use the default configuration, with only actual-auth fixtures setting their own mode. Both receipts remain.

## Preserved intermediate evidence

permanent-red.log includes an initial test syntax error and the real adapter/policy REDs. permanent-causal-red.log then exposed an unresolved fixture prerequisite; permanent-auth-causal-red.log identifies missing mocked ensureConfigForRequest. That collaborator was aligned with the already-working reviewer fixture before classifying permanent product failures. permanent-actual-auth-red.log and permanent-confirmed/lifetime-red.log also contain partial-preview readiness expectations and a cached visual resolver fixture; final tests clear the existing resolver cache and await actual generator delivery rather than assume visible-preview timing. full-focused-initial.log records the old synthetic snapshot missing the now-required captured credential scope; actual transport tests now obtain the real lease. full-focused-scoped-options.log records the five old exact options expectations before their narrow alignment. Original body/identity assertions remain.

All intermediate outputs are retained rather than described as passing proof. Only final receipts above support the result.

## Limits and disposition

Source/tests are frozen and independent review is pending. No browser, native profile, held database, runtime, provider inference, or backend test was touched. Native cross-account persistence is not claimed. A request legitimately dispatched before invalidation can already be accepted by the server; frontend cancellation cannot roll back that prior server action. These guards cancel the client lifetime and suppress subsequent stale dispatch/fallback/local publication.

UAT246's original silent native provider response remains a separate diagnosis. Existing45-second configured idle and60-second Character watchdog policies are unchanged. Parent owns final fixture reconciliation, native acceptance and matrix gate decisions.
