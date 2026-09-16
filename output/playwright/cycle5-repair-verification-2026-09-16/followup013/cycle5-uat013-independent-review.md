# Independent review: UAT013

**Verdict: one actionable P2; not yet approved for all accepted handoff orderings.**

## Finding

### P2 — Same-value accepted handoff is overwritten by initial session restore

`apps/packages/ui/src/hooks/usePlaygroundSessionPersistence.tsx:676–686`

The guard compares only serialized mode/source values against its mount baseline. If the live option store already holds `chatMode: rag, ragMediaIds: [42]` before mounting, then a valid new Home handoff selecting Rowan42 leaves those values unchanged. An old persisted session with source7 still passes the equality check and overwrites the just-accepted handoff. A read-only private regression using the author's actual Form→useChatActions→RAG pipeline/request serializer reproduces an outbound `include_media_ids: [7]` when `[42]` was explicitly accepted immediately before restore. Retrieval is already enabled in this case, so adding that boolean to the equality tuple alone would not resolve it.

Record/observe accepted source intent across initial restore even when the selected value is identical, while retaining later intentional restore semantics, old transcript restoration and existing scope/owner cancellation. Add this permanent actual-transport control. This is an automated boundary finding, not a claimed additional native run.

## Verification and positive assessment

- All four hashes match `/private/tmp/cycle5-repair-chat-013-manifest.json`.
- Independent installed-local run: **156 passed / 8 suites**, exit0, no skips. `/private/tmp/cycle5-uat013-independent-tests.log`.
- Read-only actual-boundary private probe: **1 failed / 24 intentionally unselected**. `/private/tmp/cycle5-uat013-independent-private-probe.log`.
- Retained baseline RED and restore-only RED inspected: failures match missing retrieval activation and overwritten source orderings, respectively.
- Actual RAG API domain serialization is exercised; the request source filter, expected-user header and returned source text in model prompt/source metadata are asserted. Empty/error selected-source retrieval never falls through to ordinary generation. Full-content handoff still uses normal Chat.
- Permanent tests meaningfully cover initial restore before/during/after a new changed-value source, later explicit restore, changed conversation/account targets, synchronous real localStorage session hydration, stale owned storage A→B→A, and late invalidated request output. Transcript restoration remains separate from source selection protection. The same-value accepted source above is the missing case.

## Exact commands

From `apps/packages/ui`:

```sh
node_modules/.bin/vitest run src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx src/hooks/__tests__/usePlaygroundSessionPersistence.test.tsx src/hooks/chat/__tests__/useChatActions.service-prompts.test.tsx src/hooks/chat-modes/__tests__/ragMode.sanitization.test.ts src/hooks/chat/__tests__/chat-action-utils.rag-overrides.test.ts src/services/__tests__/media-chat-handoff.test.ts src/routes/__tests__/option-index.setup-flow.test.tsx src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx --maxWorkers=2 > /private/tmp/cycle5-uat013-independent-tests.log 2>&1
node_modules/.bin/vitest run --config /private/tmp/cycle5-uat013-independent-private.config.mts src/components/Option/Playground/__tests__/PlaygroundForm.openui-mode.test.tsx -t 'independent:' --maxWorkers=1 > /private/tmp/cycle5-uat013-independent-private-probe.log 2>&1
```

Private fixture: `/private/tmp/cycle5-uat013-independent-private-test.txt`. Its Vite load override inserts that test into the existing actual-boundary describe without writing repository files. Source was not swapped or edited.

## Limits

No source/test/task edits, browser/runtime/API/inference actions, staging or commits. Network/model/Dexie results and unrelated UI remain controlled as documented by the author; this is not native proof or full auth UI validation. The synchronous hydration claim is limited to today's synchronous localStorage adapter/migration. Root owns integrated compiler and native acceptance. Review excludes notification139/141/142 and other unrelated working-tree repairs. Existing test-environment warnings retained.
