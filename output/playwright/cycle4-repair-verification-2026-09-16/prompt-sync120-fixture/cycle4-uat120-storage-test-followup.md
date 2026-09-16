# UAT120 test storage compatibility follow-up

Frozen: 2026-09-16T06:37:04.147Z. Existing task TASK13260.60 updated through the official Backlog CLI. No production changes.

## Root cause and minimal correction

The shared-UI config uses native @plasmohq/storage, whose default area is sync. The test supplied only a local Firefox prompt-array stub; setPromptStudioDefaults therefore failed in beforeEach with an undefined storage backend. Its get method was also callback-only and its set method discarded arbitrary keys. The WebUI config aliases Plasmo to its localStorage shim, so it had hidden the incomplete extension fixture.

The single test now supplies separate local/sync/session key/value areas. get supports Promise consumers and the legacy Firefox callback consumer; set merges arbitrary stored keys. Production Plasmo, safe storage, canonical direct credentials, local-save helpers, sync owner, request-core normalization, React Query refresh and SyncStatusBadge remain real. All original assertions are unchanged: one actual failed PUT, persisted content/Pending with last acknowledged identity retained, remount/New QueryClient still Pending, no automatic replay, and explicit acknowledged PUT recovery adopting version2/id2.

## Evidence

- RED: /private/tmp/cycle4-uat120-native-storage-red.log — 1 failure in beforeEach at native Plasmo rawSetMany from setPromptStudioDefaults.
- Focused GREEN native shared UI: /private/tmp/cycle4-uat120-native-storage-green.log — 1/1 passed.
- Focused GREEN WebUI shim: /private/tmp/cycle4-uat120-web-storage-green.log — 1/1 passed.
- Related native shared UI: /private/tmp/cycle4-uat120-native-storage-related.log — 130 tests /5 suites passed.
- Related WebUI: /private/tmp/cycle4-uat120-web-storage-related.log — 130 tests /5 suites passed.
- ESLint HEAD baseline and final both 0 errors /0 warnings: /private/tmp/cycle4-uat120-storage-test-lint-baseline.log and /private/tmp/cycle4-uat120-storage-test-lint.log.

Related suites: usePromptEditor.transport-sync, usePromptEditor.save-state, SyncStatusBadge, prompt-sync.auto-sync and prompt-sync.uncertainty. Native command runs ./node_modules/.bin/vitest run followed by those paths under apps/packages/ui; WebUI command uses the corresponding ../packages/ui/src paths under apps/tldw-frontend. Both use their ordinary unmodified vitest.config.ts.

## Scope and limits

Owned paths and exact hashes: /private/tmp/cycle4-uat120-storage-test-freeze.json (test + official task record). No production, UAT118, shared config, tracker, plan, browser, runtime, inference, staging or commit changes. No new full TypeScript run for this bounded fixture-only follow-up; prior full baseline is separate evidence. Bandit does not analyze this TSX-only change. These are jsdom controls with a controlled IndexedDB table and browser storage fixture, not native browser acceptance or the complete 93-suite shared run. Root retains native outage/reload/recovery and final integration.
