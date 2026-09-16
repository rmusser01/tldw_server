# UAT152 / UAT153 — Settings generation timeouts and actionable presets

Task TASK-13260.91. Approved design: `/private/tmp/cycle5-uat152-readonly-design.md`. Base source1ea5402c83. Two production paths only: `components/Option/Settings/{tldw.tsx,TldwTimeoutSettings.tsx}` under `apps/packages/ui/src`.

## Behavior

Ordinary connection save previously persisted10second Chat request/startup and RAG limits even when Advanced settings was never opened. These explicit values overrode request-core's120second generation fallback. Balanced now uses120seconds and Extended240seconds for those three generation fields. Initial state and invalid generation fallback use the preset. Generic requests, idle, Media and upload values retain their existing defaults.

Stored valid custom values, including10seconds, remain unchanged until an explicit preset/reset choice. No migration or credential/storage/request-layer change.

Actual component testing also confirmed UAT153: Custom was rendered with Balanced selected, so clicking Balanced did nothing. `value={timeoutPreset}` now keeps Custom controlled with neither available preset selected. Clicking Balanced applies and persists its values; Reset still works. This discovery is component-test-confirmed, not a native browser claim.

## Actual boundaries covered

New file: `components/Option/Settings/__tests__/tldw.timeouts.form.test.tsx` (9tests).

- Actual Settings, AntD Form, `TldwApiClient.initialize/updateConfig`, and Plasmo serialization/localStorage persistence. Plasmo's built-in `allCopied` localStorage fallback is enabled in jsdom because extension storage is absent. No replacement config-save implementation.
- First-ever connection with no stored config; ordinary save with existing server but no timeout values; remount persistence.
- Extended/Reset save and reload; preserved10s/237s overrides; invalidzero generation fallback.
- Custom neither preset checked; actual Balanced radio click then save/reload reflects Balanced120s.
- Saved output enters real `tldwRequest` with an abort-aware delayed fetch and fakeclock: answer at19.689seconds succeeds with defaults; explicit10seconds still returns REQUEST_TIMEOUT, with one dispatch. This is automated transport behavior with a simulated response, not live model evidence.

## RED, GREEN and harness corrections

- Valid UAT152 RED:4failed/2passed in `/private/tmp/cycle5-uat152-red-storage.log`: ordinary10s versus120s, Extended20s versus240s, old10s not distinguished from Balanced, actual request abort before19.689s. Deliberate custom237s and10s abort already passed.
- UAT153 RED: selected case1failed (8unselected), `/private/tmp/cycle5-uat153-selector-red.log`: actual Balanced click still saved10s.
- Final **25passed/4files**,20.99s,exit0: `/private/tmp/cycle5-uat152-153-confirmed-green.log`.

Working directory `apps/packages/ui`:

```sh
./node_modules/.bin/vitest run src/components/Option/Settings/__tests__/tldw.timeouts.form.test.tsx src/components/Option/Settings/__tests__/tldw-settings-tabs.test.tsx src/services/tldw/__tests__/request-core.refresh-timeout.test.ts src/services/__tests__/tldw-settings-storage.test.ts --maxWorkers=1 > /private/tmp/cycle5-uat152-153-confirmed-green.log 2>&1
```

Preliminary harness corrections, not product regressions:

1. Initial jsdom run used default Plasmo storage without extension storage and loaded no config; fixed via the library's real localStorage fallback before valid RED.
2. Simulated successful Response lacked JSON content-type; added the header so real transport parses the intended JSON. Deadline behavior was already correct.
3. Save helper now awaits the newly observed actual updateConfig call/promise, avoiding an earlier spy call satisfying a later save.
4. Initial selector truthfulness assertion counted Auth Mode radios too; narrowed to Balanced/Extended specifically. Product code did not change for this correction.

## Existing broader baseline failures

A broader7-file run had28failed/32passed: `/private/tmp/cycle5-uat152-final-green.log` (historical filename; **not green**). All28 failures were in existing auth-mode/form-lifecycle/cookie-logout files. The same3files run against unchanged1ea5402c83 production via private Vite loader had28failed/8passed; exact28 failure headings match.

- Baseline log: `/private/tmp/cycle5-uat152-existing-baseline.log`.
- Comparison: `/private/tmp/cycle5-uat152-baseline-comparison.json`.
- Reproducible config: `/private/tmp/cycle5-uat152-baseline.config.mts`.
- Baseline source files: `/private/tmp/cycle5-uat152-baseline-{tldw.tsx,TldwTimeoutSettings.tsx}` (git show1ea5402c83).

Baseline command from the same working directory:

```sh
./node_modules/.bin/vitest run --config /private/tmp/cycle5-uat152-baseline.config.mts src/components/Option/Settings/__tests__/tldw.form-lifecycle.test.tsx src/components/Option/Settings/__tests__/tldw-auth-mode.form.test.tsx src/components/Option/Settings/__tests__/tldw.cookie-logout.test.tsx --maxWorkers=1 > /private/tmp/cycle5-uat152-existing-baseline.log 2>&1
```

These existing fixture failures remain a validation limit; no claim that all Settings tests pass. Logs also retain jsdom CSS parsing/localStorage warnings and a first-connection Flashcards-transfer cleanup warning; no unrelated repairs made.

## Static checks / remaining acceptance

Scoped ESLint:0errors,33unchanged warnings;0added/removed compared with baseline. Command from repository root:

```sh
apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Option/Settings/tldw.tsx apps/packages/ui/src/components/Option/Settings/TldwTimeoutSettings.tsx apps/packages/ui/src/components/Option/Settings/__tests__/tldw.timeouts.form.test.tsx -f json
```

Bandit is not applicable to this TypeScript-only change. Root owns combined compiler baseline comparison, independent review, and original native source RAG acceptance. No browser/runtime actions, source staging, or commits performed. Existing old10second native profile must explicitly choose Balanced (now actionable) or Reset and save; it is intentionally not silently migrated. Independent unset TldwChat startup fallback remains outside this Settings unit.
