# Independent review: UAT140

## Findings

**P2 — Give the actual six-control persistence test adequate local time.**
`apps/packages/ui/src/components/Common/Settings/tabs/__tests__/ConversationTab.input-number.test.tsx:131`

The exact author four-suite command independently produced 19 passed / 1 failed because this new test exceeded the default 5000ms (5163ms elapsed). An unchanged single-suite rerun passed 5/5, with the same case taking 4315ms; the author run recorded 4267ms. This leaves little timing margin for six real AntD edits, asynchronous settings persistence and remount. Use a bounded per-test timeout (e.g. 10000ms) or reduce query cost without removing the persistence assertions. This is a test stability issue, not evidence of a product persistence defect. Both observed outcomes are retained; the first run is not reported as green.

## Product assessment

No production findings. All six numeric controls use explicit label/htmlFor and input/id pairs with a React.useId prefix, so accessible names do not include AntD spinner icon names and separate mounted instances avoid ID collisions. Existing translated labels, values, disabled conditions, min/max/step/precision, onChange and onBlur behavior remain unchanged. Only labels/wrappers, IDs and cell width changed.

The new regression genuinely mounts AntD and the actual settings hook/normalizer/persistence layer. Controlled storage/server boundaries are accurately documented. Checks cover precise accessible names, bounds/initial values, no write before blur, six saved values after remount, unaffected stop/enabled settings, disabled modes and threshold/window clamping. The retained RED shows the actual addonBefore deprecation; final isolated suite emits no matching warning.

## Verification

All three frozen SHA256 hashes match /private/tmp/cycle5-repair-140-manifest.json. Inspected author scoped ESLint JSON: 0 errors / 0 warnings.

From apps/packages/ui, installed local binary:

```sh
node_modules/.bin/vitest run src/components/Common/Settings/tabs/__tests__/ConversationTab.input-number.test.tsx src/components/Common/__tests__/ConversationTab.generationOverride.test.tsx src/components/Common/Settings/tabs/__tests__/ConversationTab.persona-memory-mode.test.tsx src/services/__tests__/chat-settings.sync.test.ts --maxWorkers=1 --no-file-parallelism > /private/tmp/cycle5-uat140-independent-tests.log 2>&1
```

Result: **19 passed / 1 timeout**, 4 suites; see finding.

```sh
node_modules/.bin/vitest run src/components/Common/Settings/tabs/__tests__/ConversationTab.input-number.test.tsx --maxWorkers=1 --no-file-parallelism > /private/tmp/cycle5-uat140-independent-focused-rerun.log 2>&1
```

Result: **5 passed / 1 suite**, exit 0; no assertions changed. No skipped tests.

No repository edits, browser/runtime/inference actions or commits. Native acceptance and root combined compiler remain pending. Existing Node localStorage experimental warning retained. Review excludes UAT139/013 and unrelated working tree changes.
