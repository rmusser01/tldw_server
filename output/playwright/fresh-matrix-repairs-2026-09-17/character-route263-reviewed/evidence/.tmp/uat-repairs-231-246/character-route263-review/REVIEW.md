# UAT263 / TASK13260.205 independent review

## Verdict

**APPROVE.** The fix closes the stale Character URL/store race at the existing owned-route replacement boundary. It does not introduce a new routing mechanism or alter server branching, ordinary chat flow, error retry, cancellation, or route ownership policy.

## Route and authority analysis

After the branch server chat and copied prefix are accepted, `createBranchMessage` invokes its new optional acceptance callback before it publishes the child as active store state. `useChatActions` supplies that callback only for the existing Character branch actions. It captures the parent server chat ID, history ID, current restore revision, character ID, and accepted child chat ID in the existing route-replacement event.

Playground continues to reject a replacement event unless the captured URL, current parent chat, history, restore revision, and Character route intent match. A stale route/account/navigation transition clears or changes one of those ownership conditions; it cannot rewrite the route. Once accepted, the existing handler retires the old route before its asynchronous restore can reassert it, then replaces `chatId` directly with the accepted child ID. Route loading still validates the eventual child conversation, so this event does not add access to another account’s chat.

The replacement helper preserves its prior clear-action behavior: `nextChatId` is optional and existing callers omit it. The Character callback is after successful child creation and copied-prefix persistence, so cancellation and branch creation failures do not publish a child route. Generic `createRegenerateLastMessage` behavior remains unchanged.

## Causal coverage review

The added coordinator regression begins with a saved Character parent URL, sends the accepted child replacement while a stale restore cycle is allowed to run, then verifies all of:

- the parent `chatId` cannot return to the URL;
- the active store and persisted session remain on the accepted child;
- normal reload resolves the child transcript;
- no expected response text is hardcoded into production behavior.

The Character integration covers complete, partial, and malformed assistant branch paths, asserting the event’s parent and child identity. Existing coordinator cases cover cancellation before owner retirement, route navigation, stale/replaced saved identities, and account transition. Existing branch tests retain parent-character precedence over stale local state.

## Independent focused validation

```sh
cd apps/tldw-frontend && node node_modules/vitest/vitest.mjs run --config ../../.tmp/uat-repairs-231-246/character248/vitest.config.ts ../packages/ui/src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx ../packages/ui/src/hooks/handlers/__tests__/messageHandlers.branch.test.ts --maxWorkers=1 --no-file-parallelism --silent
```

Result: exit 0, **109 passed**. This includes the causal stale-reassertion coordinator test, 47 Character action cases, 61 route/account/cancel coordinator cases, and the Character branch owner-precedence test.

```sh
cd apps/tldw-frontend && node node_modules/vitest/vitest.mjs run --config ../../.tmp/uat-repairs-231-246/character248/vitest.config.ts ../packages/ui/src/hooks/handlers/__tests__/messageHandlers.regenerate.test.ts --maxWorkers=1 --no-file-parallelism --silent
```

Result: exit 0, **8 passed**. This retains exact Retry input, local diagnostic Retry, image handling, and pre-submit override behavior.

`git diff --check` on the six frozen files passed.

## Frozen source hashes

```text
ee1349b4059581d49667e725767db308fe40ecac0f87c32002f82fa6cf60c3e9  Playground.tsx
6fa7aa29e70104aa241113beb3a01a84f664ee3675e719ae5287e7ec547069db  Playground.coordinator.integration.test.tsx
0b0524c04945282aba4f9add789fbba1d5f735a235fb92221a6d9fc0ae85db1b  useChatActions.ts
9ab6e6f727f3ce573edf25496a3421db597fe99e9f31089502556d29ecb91a98  useChatActions.character.integration.test.tsx
ec6ddb62f75bf7c12eceae7d8db1d2db7c1cb79e3d0941662cb2a2d5d012d3d0  messageHandlers.ts
bc4983feb9e494c000832a2f63bf84930ee5c99718eb70f9dbf2979f17b4dbbe  character-chat-mode-intent.ts
```

Author static evidence remains accurate: ESLint has zero errors with 77 warnings outside changed lines; the TypeScript comparison has zero added diagnostics; Bandit has zero findings but cannot parse the six TypeScript/TSX inputs.
