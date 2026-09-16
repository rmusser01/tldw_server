# UAT156 / TASK13260.94 chronology repair

Ready for independent review; native acceptance remains pending. Base HEAD2d5ad06c86. Root created and owns Backlog/task/tracker updates. No runtime, browser, inference, staging or commit actions.

## Change and why

`apps/packages/ui/src/db/dexie/server-chat-mirror.ts` now assigns a valid canonical creation timestamp after preserving protected local content in both memory and persisted-mirror reconciliation. Finite numeric timestamps (including epoch0) are accepted; absent/malformed canonical timestamps retain the local value, with the existing current-time fallback for a new persisted row. The raw source question, unknown/equal/newer local content, images, IDs, parent links, owner guards and unsent drafts retain their existing behavior. This fixes content preservation accidentally preserving a post-stream local timestamp and moving the user below its answer.

Only production file above and new `apps/packages/ui/src/db/dexie/__tests__/server-chat-chronology.test.ts` changed. Production diff7insertions/1deletion. No new dependencies or schema changes.

## Test-first evidence

Permanent regression runs actual `saveMessage`, `reconcileServerChatMessages`, `reconcileServerChatMirror`, and `formatToMessage`, replacing only IndexedDB I/O and unrelated imports. Local saves use real post-stream Date.now()+1/+2 semantics and canonical messages use earlier user/answer timestamps. Before production editing:13expected behavioral failures /4passing controls in `red-behavior.txt`. The actual source question rendered after the answer; malformed remote timestamps also replaced usable local values. Initial command was run from repo root and found no tests; retained separately in `red.txt`, not counted as product RED.

After correction: `green-final.txt`128passed/7suites, no failures/skips/unhandled errors. This includes17new chronology cases and existing mirror, transaction, loader, owner-scope, mounted mirror integration, and image-loading controls. Earlier60/2and128/7runs overlap and are not additive. Final test fixture review added the required empty `sources` field before the final rerun.

New cases exercise repeated reload idempotence; ordinary equal-content control; unknown/equal/newer-version protected edits and images; typing during snapshot await; six absent/invalid timestamp cases; epoch0; distinct same-text drafts and older plain local retrieval refusal; and owner/conversation/history changes. Tests assert content/identities as well as order. Existing guarded correlation and anchored-recovery cases remain green.

## Validation

Final command (cwd `apps/packages/ui`):

```
./node_modules/.bin/vitest run src/db/dexie/__tests__/server-chat-chronology.test.ts src/db/dexie/__tests__/server-chat-mirror.test.ts src/db/dexie/__tests__/chat-persistence-transaction.test.ts src/hooks/__tests__/useServerChatLoader.test.ts src/hooks/__tests__/useServerChatLoader.scope.test.tsx src/hooks/__tests__/useServerChatLoader.mirror.integration.test.tsx src/hooks/__tests__/useServerChatLoader.images.test.ts --reporter verbose
```

Root-config ESLint both owned files:0errors/0warnings before and final. `eslint-before.json` and `eslint-final.json` retain receipts. CLI emits the existing project-root Pages-directory configuration notice in both runs; not suppressed. Vitest emits the Node localStorage experimental environment warning in both RED/GREEN; test setup provides its normal in-memory storage. Owned diff-check clean. No whole TypeScript claim; parent owns integrated compiler verification. Bandit not applicable: TypeScript-only production/test scope, no Python touched.

## Scope limits

- UAT156 chronology only. UAT103 older-history promotion receipts and conditional missing-current-ACK/raw-wrapper recovery remain separate.
- UAT013 wrong source answer remains unresolved; this change does not alter provider messages, source facts, prompt escaping or model behavior.
- The diagnostic scenario has deterministic evidence; full native source send/reload is root-owned and still required. Do not close from tests alone.

`manifest.json` contains SHA256 for the exact production/test freeze and reports/results. Original production snapshot is `server-chat-mirror.before.ts`.
