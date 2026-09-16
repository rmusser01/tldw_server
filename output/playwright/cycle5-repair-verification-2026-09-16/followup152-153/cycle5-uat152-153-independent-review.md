# Independent review — UAT152 / UAT153

## Verdict

**Clear for the bounded source change; no actionable finding.** All three frozen paths match the author's 2026-09-16T17:49:38.290Z freeze. Independently repeated the exact four-suite scope: **25 passed / 4 files, exit0, 20.01s**. Native acceptance of the positive Home→source RAG answer remains pending.

## Reviewed scope and source

Baseline `1ea5402c83`; only two production files:

- `apps/packages/ui/src/components/Option/Settings/TldwTimeoutSettings.tsx`: Balanced Chat request/startup/RAG values120seconds, Extended240seconds. Generic, idle, Media and upload budgets remain unchanged. The Segmented value uses the actual `custom` state, so neither preset is falsely selected and clicking Balanced invokes its normal handler.
- `apps/packages/ui/src/components/Option/Settings/tldw.tsx`: initial timeout state comes from the same preset map; save fallback for the generation fields uses generation defaults. Stored numeric overlay, explicit bounds, save/persistence flow and credential logic are unchanged. No migration rewrites existing10second/custom records.

The existing `loadConfig` overlay, `determinePreset`, `applyTimeoutPreset`, and request-core precedence were inspected. This is the smallest direct correction of the observed ordinary-save boundary and the actual Custom→Balanced click failure. It does not rewrite request transport, widen unrelated budgets or introduce a timeout abstraction.

## Tests and honesty of the evidence

New `tldw.timeouts.form.test.tsx` uses actual Settings and AntD Form, real `TldwApiClient.initialize/updateConfig`, and actual Plasmo serialization/localStorage fallback. The spy calls through to the save owner. Its storage fallback supplies missing jsdom extension infrastructure rather than replacing config saving. Server health, unrelated UI/routing/i18n and transport fetch are appropriately controlled.

Permanent controls independently passed:

- Ordinary save without opening Advanced, including the no-config first connection, persists120000ms generation budgets.
- Save/remount, Extended240000ms, explicit Reset and Custom→Balanced selection.
- Deliberate10000ms and237000ms custom values stay unchanged on ordinary save/reload; invalidzero generation fallback is120000ms.
- Actual saved config enters real `tldwRequest` at `/api/v1/rag/search`. Abort-aware simulated fetch completes at19689ms with the default, while deliberate10000ms returns `REQUEST_TIMEOUT`. Both assert one dispatch; this does not test or authorize retrying a mutation.
- Existing settings-tabs, request-core refresh/timeouts and storage controls pass.

Inspected valid historical RED evidence: `/private/tmp/cycle5-uat152-red-storage.log` (4 failures/2 passes) and `/private/tmp/cycle5-uat153-selector-red.log` (actual Balanced click still saved10000ms). The implementation report correctly distinguishes earlier harness corrections and the unrelated broader baseline failures from valid RED/GREEN.

## Independent command

Working directory `/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui`:

```sh
./node_modules/.bin/vitest run src/components/Option/Settings/__tests__/tldw.timeouts.form.test.tsx src/components/Option/Settings/__tests__/tldw-settings-tabs.test.tsx src/services/tldw/__tests__/request-core.refresh-timeout.test.ts src/services/__tests__/tldw-settings-storage.test.ts --maxWorkers=1 > /private/tmp/cycle5-uat152-153-independent-green.log 2>&1
```

Result:25/25 tests,4/4 files,exit0,20.01s. Existing jsdom CSS parsing/localStorage notices and the first-connection private-transfer cleanup warning remain visible in the log; they were not suppressed or represented as clean native console evidence.

## Integrity and static validation

Independent hashes: `/private/tmp/cycle5-uat152-153-independent-hashes.json`.

- `TldwTimeoutSettings.tsx`: `7123b21be2545f87fc0428f9466167c6bc022ad4ab99a63fe729fe78565a5051`
- `tldw.tsx`: `931e92bf7d74580194531b6a47c905d6a296bbfa0e1023a22257da7a81093a69`
- `tldw.timeouts.form.test.tsx`: `5af76fc32b3e93ce3e35e386a8b89d6d4fa986df8e35e66253653fb5840884e3`

Inspected author ESLint comparison:0errors,33unchanged warnings,0added/removed (`/private/tmp/cycle5-uat152-153-eslint-comparison.json`). No independent full compiler run; controller owns the combined baseline. Bandit is not applicable to these TypeScript files.

## Limits / next acceptance boundary

- The19.689second success is an automated fake-clock response through real request code, not a live model/browser claim.
- The old native10second configuration deliberately remains unchanged. Explicitly select Balanced or Reset and save, record that action, then repeat the original source RAG scenario for UAT013/152 acceptance.
- Independent unset `TldwChat` startup fallback, full Settings auth/lifecycle coverage and native preset interaction are not certified by this four-file run. The28 matched baseline failures in three older fixtures are separately assigned UAT154; this reviewer did not edit or run those in-flight fixtures.
- No repository, source, task, runtime, browser, inference or staging changes were made during this review. Only private review evidence was written.
