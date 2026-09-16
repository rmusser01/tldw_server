# UAT-031 failed character Retry repair

Author subtask: `retry031_repair`; tracking: TASK-13260.7. Initial source revision: `2d5ad06c86cf279fe0fcc6445009813d97ab1c1b`. Report capture HEAD: `05e0c5593ef79e1b9cacc4e741f7a4ee559a6f87` (parent may have committed unrelated work). No staging, commits, runtime, browser or inference actions by this author. The accompanying manifest records exact current source/test/task and evidence hashes.

## Root cause and scope

Playground composer Retry invokes the real `regenerateLastMessage` handler. Its `useChatActions` pre-submit callback unconditionally branched saved character conversations, including pure provider-failure bubbles. The branch copied greeting/user rows to a new conversation while the existing visible rows retained their original server message IDs. This made original greeting derived-save requests combine a new conversation with old message IDs.

Production change: import `decodeChatErrorPayload`, take `lastAssistant` in the existing callback, and skip branch creation for a valid structured error bubble. Completed prose, interrupted partial prose, and malformed error-marker prose keep existing Regenerate branching. No broad Retry overhaul or mirror change.

Owned paths:

- `apps/packages/ui/src/hooks/chat/useChatActions.ts`
- `apps/packages/ui/src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx`
- `backlog/tasks/task-13260.7 - Repair-tracked-Chat-provider-routing-and-visible-conversation-state.md` (official CLI notes only)

## RED/GREEN

Permanent regression uses the real hook, regenerate and branch factories, character SSE transport/parser, error-saving factory, and failed-history helper. Remote endpoint operations and IndexedDB writes/transaction are substituted. It starts with a local greeting, sends one question, receives SSE `provider_unavailable`, then retries twice.

Before production edits: expected `conversation-1`, got `conversation-3`. Three controls already passed for complete prose, interrupted partial prose and malformed marker prose. Earlier fixture failures (uninitialized i18next, omitted global scope query, remote fake reset) are not counted as product RED.

After repair: one conversation, one canonical greeting/user pair, exact original greeting/user receipts, same completion path three times, three visible rows and coherent history. The three branch controls still pass.

Exact commands run from `apps/tldw-frontend`:

```sh
bunx vitest run ../packages/ui/src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx -t 'retries failed complete-v2|retains server branching'
bunx vitest run ../packages/ui/src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx ../packages/ui/src/hooks/handlers/__tests__/messageHandlers.regenerate.test.ts ../packages/ui/src/hooks/handlers/__tests__/messageHandlers.branch.test.ts ../packages/ui/src/hooks/chat/__tests__/useCharacterChatMode.contract.test.ts ../packages/ui/src/hooks/chat-helper/__tests__/saveMessageOnError.test.ts
NODE_OPTIONS=--max-old-space-size=8192 bunx tsc --noEmit --pretty false
```

Final full focused run: **53 tests / 5 files PASS, zero skipped**. Logs: `/private/tmp/uat031-red.log`, `/private/tmp/uat031-green.log`. Existing fallback-persistence tests deliberately emit error logs; Node emits its existing localStorage experimental warning.

## Static/security checks

From repository root:

```sh
apps/tldw-frontend/node_modules/.bin/eslint -c apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/hooks/chat/useChatActions.ts apps/packages/ui/src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx
git show HEAD:apps/packages/ui/src/hooks/chat/useChatActions.ts | apps/tldw-frontend/node_modules/.bin/eslint -c apps/tldw-frontend/eslint.config.mjs --stdin --stdin-filename apps/packages/ui/src/hooks/chat/useChatActions.ts -f json
git show HEAD:apps/packages/ui/src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx | apps/tldw-frontend/node_modules/.bin/eslint -c apps/tldw-frontend/eslint.config.mjs --stdin --stdin-filename apps/packages/ui/src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx -f json
source .venv/bin/activate && python -m bandit apps/packages/ui/src/hooks/chat/useChatActions.ts apps/packages/ui/src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx -f json -o /private/tmp/uat031-bandit.json
git diff --check
```

ESLint: **0 errors / 39 warnings**, matching HEAD source17 + test22 baseline counts. Initial root-level `bunx eslint` failed resolving a writable Bun tempdir; installed binary command above succeeded. Full compiler exits2 with **90 diagnostics**, none naming the touched hook/test. Parent owns normalized full-baseline comparison; do not call this a clean compiler run. Diff check passed.

Bandit: zero findings, **both TS/TSX files fail Python AST parsing**. This does not establish TypeScript security coverage.

## Remaining acceptance / evidence limits

Independent review and native saved-character Retry → greeting Save Notes/Flashcards → canonical reload are parent-owned and pending at author handoff. No native acceptance claim. Latest native d91a… → 68cf… observation and original greeting9383… action mismatch are transcript-only because temporary artifacts were lost; retained older evidence must not be relabeled as that latest observation. Successful Regenerate receipt behavior remains outside this narrowly authorized pure failed-Retry repair.
