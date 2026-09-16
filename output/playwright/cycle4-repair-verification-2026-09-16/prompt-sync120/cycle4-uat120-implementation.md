# UAT120 / TASK-13260.60 — Prompt normalized-failure sync status

## Outcome and scope

Frozen code at **2026-09-16T05:54:22.171Z**, current HEAD **a8e5430d7692bf44669971a0187056b1fe7fb442**. No staging/commit, browser, runtime, inference, tracker, or shared-plan edits. Native acceptance and independent review remain pending.

Production adds four lines in `apps/packages/ui/src/services/prompt-sync.ts:649`: unsuccessful **standard** Prompt writes with a non-validation failure persist `syncStatus: "pending"`. Recipe uncertainty/error writes and validation branches are unchanged. This fixes both auto-sync and direct push consumers at their shared persistence owner.

The transport resolves fetch rejection as `{ok:false,status:0,...}`; the sync service classifies it `invalid_server_payload` and already returns Pending. Previously only thrown/transient failures reached auto-sync's Pending write, so a new local revision could retain the previous Synced status in storage. The list correctly reread that incorrect persisted status. The new write changes only status, retaining local content and last acknowledged server linkage/version/time. A later acknowledged update adopts the returned versioned ID.

Exact code ownership:

1. `apps/packages/ui/src/services/prompt-sync.ts`
2. `apps/packages/ui/src/services/__tests__/prompt-sync.auto-sync.test.ts`
3. `apps/packages/ui/src/components/Option/Prompt/__tests__/usePromptEditor.transport-sync.test.tsx`

Tracking only: `backlog/tasks/task-13260.60 - Keep-standard-Prompt-sync-state-pending-after-transport-failure.md` (official MCP update). Code byte hashes: `/private/tmp/cycle4-uat120-code-freeze.json`. Full owned manifest including task: `/private/tmp/cycle4-uat120-owned-manifest.json`.

## RED → GREEN evidence

- `/private/tmp/cycle4-uat120-mounted-red.log`: **1 failed** at the intended assertion: actual saved record retained `synced` instead of `pending` after new local text, failed fetch, and the genuine saved-locally warning. The first fixture-development run had been stopped by the canonical API-key guard; adding the actual device-key metadata made it reach the network boundary before recording this RED. No production changes preceded the intended RED.
- `/private/tmp/cycle4-uat120-service-red.log`: **2 failed / 47 passed**. Standard create and update returned Pending but kept Local/Synced in the table. These controls cover normalized status0, HTTP503, and empty acknowledged-response shapes via auto-sync and direct push.
- First post-fix run (`/private/tmp/cycle4-uat120-focused-green.log`) passed the mounted regression; one old empty-create assertion expected no status write despite returned Pending. Updated it to assert durable Pending plus retained local content and no invented server linkage. Also removed an internal one-write-count assertion from the existing thrown-failure test; its result/status/project/timestamp assertions remain.
- Final `/private/tmp/cycle4-uat120-related-green.log`: **1,197 passed / 9 suites**, exit0.

The new mounted regression uses real `usePromptEditor`, `usePromptSync`, React Query invalidation, local `updatePrompt` and `PageAssistDatabase` methods, Prompt Studio service, `apiSend`, request-core, canonical stored device-key projection, and `SyncStatusBadge`. Only the unavailable IndexedDB table and browser storage/network boundaries are controlled. The test unmounts and uses a fresh QueryClient over retained rows, verifies Pending without another network request, then explicitly saves and holds the response unresolved to prove it remains Pending until acknowledgement. Its successful update/1 response returns id2/version2/parent1 and then renders Synced#2. Exactly two explicit PUTs occur and no create is dispatched. The badge number is a server ID, not the displayed version.

Additional new controls preserve exact known 401/403/409/422 mutation-rejection handling. Existing recipe uncertainty (62), actual owner/transport contracts (33), structured-prompt property/compatibility tests (1,011), API-send policy (13), caller hooks (10), badge (12), save-state owner (6), auto-sync (49), and mounted regression (1) all passed.

Final test command, from `apps/tldw-frontend`:

```sh
./node_modules/.bin/vitest run \
  ../packages/ui/src/components/Option/Prompt/__tests__/usePromptEditor.transport-sync.test.tsx \
  ../packages/ui/src/components/Option/Prompt/__tests__/usePromptEditor.save-state.test.tsx \
  ../packages/ui/src/components/Option/Prompt/__tests__/prompt-sync.owner-callers.test.tsx \
  ../packages/ui/src/components/Option/Prompt/__tests__/SyncStatusBadge.test.tsx \
  ../packages/ui/src/services/__tests__/prompt-sync.auto-sync.test.ts \
  ../packages/ui/src/services/__tests__/prompt-sync.structured-prompts.test.ts \
  ../packages/ui/src/services/__tests__/prompt-sync.uncertainty.test.ts \
  ../packages/ui/src/services/__tests__/recipe-persistence-owner.contract.test.ts \
  ../packages/ui/src/services/__tests__/api-send.test.ts
```

## Static validation

- Scoped ESLint, root config: **0 errors / 1 unchanged pre-existing no-explicit-any warning** at auto-sync test line11. Production/new test have zero warnings. Compared original HEAD bytes using `--stdin --stdin-filename`, not source swaps. Artifacts: `/private/tmp/cycle4-uat120-{production-lint-before,test-lint-before,lint-after,lint-comparison}.json`.
- Whole frontend `./node_modules/.bin/tsc --noEmit --pretty false`: exit2, **exact90 baseline diagnostics**, zero added/removed after normalizing line/column but preserving path, TS code, complete message, and multiplicity. Baseline `/private/tmp/uat032-merged-typecheck.log`; current `/private/tmp/cycle4-uat120-typecheck.log`; comparison `/private/tmp/cycle4-uat120-typecheck-comparison.json`.
- Scoped `git diff --check`: clean.
- Bandit is not applicable to this TS/TSX-only scope; no Python changed. No new dependency, backend endpoint, global transport, automatic retry, or recipe-authority behavior added.

## Limits and next check

This proves production logic through controlled storage and transport boundaries, not real IndexedDB durability, an actual browser hard reload, or an extension background lifecycle. Root's native outage evidence remains the original failure; the corrected native outage/save/reload/explicit-recovery check is still required. No claim that failed standard writes are automatically replayed. UAT119's four editor background-token changes belong to root and are excluded from this manifest.
