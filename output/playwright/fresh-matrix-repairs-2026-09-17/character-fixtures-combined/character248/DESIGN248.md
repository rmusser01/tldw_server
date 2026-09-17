# UAT248: Character turn authority lifetime

Task TASK13260.190 exists before these evidence edits. Root approved this bounded design; runtime, native data, tracking, and integration remain parent-owned. Production edits are held until the preceding Chat unit integration is confirmed.

## Proven defect and limits

The retained reviewer actual-auth probe uses real service-prompts.loadServicePromptSnapshot and real WebUI authService.logout. Both captured lease signals abort, while the original caller signal remains live. Controlled late SSE success reaches persistCharacterCompletion for the old conversation without scope options. Replacement draft UI is preserved. This proves stale client persistence dispatch, not a successful native cross-account database write.

runCharacterChatTurn receives a captured snapshot but streamCharacterChatCompletion uses the original caller signal (useChatActions.ts around2309). Normal completion persist and fallback use only workspace scope (around2567/2636); success save omits captured signals/requestScope. Catch/recovery already checks owner invalidation and supplies captured scope. Creation/greeting/user pre-dispatch requests have the same missing captured-scope options and need boundary guards to prevent a delayed prerequisite from dispatching after logout.

## Existing patterns (no new authority mechanism)

1. normalChatMode.ts:609–652 selects snapshot.scopeSignal as execution signal and retains original caller signal for controller ownership.
2. ensurePersonaServerChatWithState in useChatActions.ts passes requestScope/scopeSignal/scopeInvalidatedSignal to existing helpers; checks follow awaited creation.
3. Character partial recovery already forwards snapshot requestScope/scopeSignal and rechecks owner invalidation before fallback/publication.
4. chat-rag.ts persistCharacterCompletion uses requestScopeFields to bind headers and servicePromptConfig; its adjacent streaming method currently lacks that option.

## Bounded production design

- In existing Character function, choose snapshot.scopeSignal when present for transport/normal request execution, preserving the original signal for caller controller ownership and explicit discard semantics.
- Build existing request options from workspace scope, captured requestScope, and execution signal. Pass through create, greeting/user POST, normal completion persistence, and fallback. No new singleton/cache/lease type.
- Guard before dispatch and after asynchronous prerequisites/results. Owner invalidation suppresses success/error publication and all subsequent writes. Keep explicit owner checks distinct from ordinary caller cancellation so current partial-response recovery stays eligible for its current owner.
- Forward scopeSignal, scopeInvalidatedSignal and requestScope to the existing success-save boundary so local writes/mirroring use its existing guard.
- Extend only streamCharacterChatCompletion options in domains/chat-rag.ts with existing ServicePromptRequestScope; use requestScopeFields exactly as sibling persist method does. No transport timeout changes.

Owned production: hooks/chat/useChatActions.ts Character function only; services/tldw/domains/chat-rag.ts streamCharacterChatCompletion only. Tests: existing Character integration fixture with opt-in actual lease controls and narrowly adjusted scoped-option expectations, plus focused domain forwarding control if needed. Prior Chat unit bytes stay frozen until parent release.

## Permanent causal controls

Use real loadServicePromptSnapshot/authService.logout and real streamCharacterChatCompletion/bgStream with controlled fetch/ReadableStream. Keep synthetic credentials and no provider/browser/native actions.

- Logout before dispatch after a delayed prerequisite: no create/user/complete request, no old UI/local writes.
- In-flight logout: captured execution signal cancels fetch while original caller remains live; no success/partial/error persistence or replacement UI damage, acknowledged user identity unchanged and no replay.
- Delayed success or failure at normal persist/visual metadata awaits: no fallback or local publication under replacement owner.
- Same-owner success: captured target forwarded through complete/persist/local save; canonical acknowledged IDs preserved.
- Same-owner failure/partial recovery and ordinary caller stop retain existing behavior; owner-invalidated partial recovery cannot persist.
- Existing unavailable-model Retry, branching, workspace, greeting, emote, and no-duplicate recovery controls remain in suite.

## Verification and scope limits

Retain reviewer RED and author RED before production. Run focused Character/domain tests and pertinent service-prompt/persona/saved-normal/coordinator regressions. Scoped lint and full compiler baseline comparison; Bandit unsupported TypeScript parse errors disclosed. Freeze exact source snapshots/patch/evidence manifest for independent review. No claim that this explains UAT246 initial native timeout. Native acceptance remains parent-owned.

## Approved integration prerequisite and outcome

Parent approved one exact third production hunk: POST /api/v1/chats/{id}/complete-v2 in existing service-prompt-scope-error.ts allowlist. scoped-stream-allowlist-red.log proves the real scoped stream was rejected before fetch without this entry; method/path/traversal negatives remain. Three production and three test paths are final.

The permanent partial-response fixture waits for the actual generator to resume after delivering a chunk. A visible-preview wait was an invalid readiness assumption in this mocked action fixture; its failed attempts remain. Caller cancellation intentionally prevents remote partial recovery; a same-owner non-abort network failure still permits it. Existing normal Character tests now expect captured scope options, with their original body/identity assertions retained.
