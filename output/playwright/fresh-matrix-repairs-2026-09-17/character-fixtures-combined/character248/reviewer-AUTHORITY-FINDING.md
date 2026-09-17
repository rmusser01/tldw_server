# Separate Character authority-lifetime finding

## Disposition

Confirmed at the frontend hook/service boundary; separate from the approved UAT231/232/236/243 changes. Do not infer a native cross-account write or the cause of UAT246's original 45-second failure.

The first private probe stopped at module import because its WebUI networking environment was incomplete. `actual-auth-lease-setup-failure.log` preserves that harness error. With `NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE=quickstart`, the same probe reaches the intended boundary and fails the no-persistence assertion: **1 causal failure, 33 filtered tests**, 1.67 seconds.

## Actual boundary and observation

The private test uses the frozen actual `useChatActions`, actual `loadServicePromptSnapshot`/lease (wrapped only to observe the returned snapshot), actual Character stream method, actual `bgStream`, and injected fetch/SSE. It calls the actual WebUI `authService.logout()` during an active acknowledged Character turn, rather than directly aborting a fake invalidation controller. Authentication network traffic, canonical persistence, mood and identity collaborators remain controlled; this is not a browser/server test.

Assertions passed before the causal failure:

- The real logout credential event aborts both `snapshot.scopeInvalidatedSignal` and `snapshot.scopeSignal`.
- The caller controller remains un-aborted and the fetch transport receives no abort.
- A late successful stream ends; the replacement Bob draft remains unchanged.

The no-persistence assertion fails because `persistCharacterCompletion` receives exactly one call for `tracked-chat-1`, `assistant_content: "Late Alice answer"`, the acknowledged `user-server-1`, and **undefined request options**. Thus real lease invalidation does not prevent the old Character success branch from dispatching persistence. Backend owner predicates may still reject an eventual HTTP write; this test deliberately stops at dispatch and makes no claim about that result.

## Source explanation

- `services/service-prompts.ts:604–652`: the lease has child abort controllers. The `tldw:auth-credentials-changed` listener invalidates both child signals; parent-to-child cancellation is one-way.
- `lib/auth.ts:206–222`: actual WebUI logout emits principal and credentials events.
- `hooks/chat/useChatActions.ts:3432` creates the lease from the caller signal; `:3796` passes the original caller signal separately into Character mode.
- Character streaming at `:2309` passes that original signal, not the lease signal/request scope. The success branch's `:2567` persistence call has neither an invalidation check immediately before dispatch nor the snapshot signal/request scope. The catch/recovery checks at `:2752` and later are bypassed on this successful late finish.
- `components/Common/PageAssistProvider.tsx` stores the caller controller but contains no credentials listener or unmount abort. By contrast, `hooks/chat/useServerChatLoader.ts:781` explicitly bridges lease invalidation to controller abort and pins requests.

This is not introduced by the four-repair patch: its only `useChatActions` production edit is the ordinary-chat duplicate ID setter guard near line 1310. The six production baseline snapshots were independently matched to the recorded Git commit.

## Bounded next action

Track this separately before modifying the frozen Chat unit. At the actual Character lifetime boundary, reuse the existing lease signal/request-scope contract and reject invalidated success-side effects. Preserve intentional user cancellation/partial recovery as a distinct case. A repair should cover invalidation while streaming, between completion and persistence, and while persistence is pending; assert replacement state and IDs remain unchanged. No global auth redesign is required by this evidence.

## Reproduction and provenance

`actual-auth-lease-command.json` contains the exact command. `auth-probe.config.ts` substitutes private test text at the original logical test path; it never overwrites source. `actual-auth-lease-probe.tsx.txt` and the failing log are retained. `authority-source-manifest.json` binds the relevant source. No browser, real provider, native runtime, credentials, database, source or task mutation occurred.
