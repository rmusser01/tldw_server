# UAT140 / TASK-13260.79 — Conversation numeric labels

Status: implementation frozen for independent review; native AC2 remains pending.
Base: 3c30685611a26d0495dafabd38f594fee93a1655. Frozen: 2026-09-16T15:19:41.349Z.

## Cause and correction

ConversationTab passed deprecated addonBefore to six real AntD InputNumber controls: author-note Depth; generation Temp, Top-p, Rep pen; summary Threshold, Recent window. Existing neighboring tests mocked AntD and therefore could not detect its runtime warning.

Replace only those six adornments with the existing translated label text in ordinary label elements. One React.useId prefix associates each label with its numeric input; the existing grid structure remains and each control fills its grid cell. All value, disabled, min/max/step/precision, onChange and onBlur handlers are preserved. No new abstraction, dependency, locale, service, or global setting change.

The first wrapping-label iteration passed the warning check but exposed AntD up/down icon names in the accessible field name. Separate label/input siblings fix that; exact accessible names are asserted. This intermediate result is retained at /private/tmp/cycle5-repair-140-label-accessibility-red.log.

## Scope

- /Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/Settings/tabs/ConversationTab.tsx
- /Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/Settings/tabs/__tests__/ConversationTab.input-number.test.tsx
- /Users/macbook-dev/Documents/GitHub/tldw_server2/backlog/tasks/task-13260.79 - Remove-deprecated-InputNumber-adornment-in-Chat-Settings.md

No other product/test files owned or modified in this unit. Backlog updates used the official installed CLI. Task remains In Progress, AC1 checked and native AC2 unchecked.

## RED → GREEN

- /private/tmp/cycle5-repair-140-red.log: before production edits, the real mounted ConversationTab with real AntD fails the no-deprecation assertion, recording 11 exact InputNumber addonBefore warnings; 1 failed / 4 deselected.
- /private/tmp/cycle5-repair-140-green.log: 20 tests / 4 suites PASS. New actual-AntD suite: 5 passed. Adjacent generation override 5, persona-memory 2, settings sync 8.
- The new tests exercise real useChatSettingsRecord and chat-settings patch/normalization/persistence. The extension storage bridge is a controlled in-memory adapter; server calls return controlled responses. AntD, query provider, component, hook, and settings service are real. These are DOM/integration controls, not native-browser or actual extension-backend evidence.
- Controls verify six translated accessible labels and values/bounds; no save before blur; all six edited values survive unmount/reopen; existing generation stop/enabled and summary enabled settings survive edits; disabled modes keep controls disabled; lowering threshold clamps the summary window and updates its maximum.

From apps/packages/ui, installed-local Vitest command:

```sh
./node_modules/.bin/vitest run src/components/Common/Settings/tabs/__tests__/ConversationTab.input-number.test.tsx src/components/Common/__tests__/ConversationTab.generationOverride.test.tsx src/components/Common/Settings/tabs/__tests__/ConversationTab.persona-memory-mode.test.tsx src/services/__tests__/chat-settings.sync.test.ts --maxWorkers=1 --no-file-parallelism
```

RED used the new test file alone with -t 'opens all six' and the same worker flags. No package download or pnpm exec was used.

## Static validation

- /private/tmp/cycle5-repair-140-eslint.json and -eslint-baseline.json: current owned code/test 0 errors, 0 warnings; original production source via git show HEAD and ESLint stdin also 0/0.
- /private/tmp/cycle5-repair-140-eslint-comparison.json: no new diagnostics.
- Explicit repo-root command uses apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs with the two owned code paths. Both runs emit the same repository-root Next pages-directory notice outside JSON diagnostics.
- git diff --check on owned code paths: clean.
- Vitest logs retain the environment's Node localStorage experimental warning; the targeted AntD deprecation no longer occurs.
- No full TypeScript run in this bounded unit; no assertion of a new whole-project compiler result. Bandit is not applicable to this TypeScript-only markup/test repair.

## Acceptance / limits

Controller must reopen CurrentChatSettings → Conversation in the same native scenario and confirm no addonBefore warning, readable labels and usable controls. No runtime, browser, inference, staging, or commit actions occurred here. Global tracker/plan and other agents' paths were untouched.

Exact bytes: /private/tmp/cycle5-repair-140-manifest.json.

## Independent-review timeout follow-up — 2026-09-16T15:25:05.577Z

The independent run recorded 19 PASS / 1 timeout at 5163ms, followed by unchanged focused 5/5 PASS with the roundtrip at 4315ms. Applied the requested local 10_000ms timeout to only that test. A reverse-change SHA256 check proves every other test byte/assertion is unchanged, and production SHA256 remains a40a206fab81a30a8bf92f67f30ed6c7e060762a12c0aa47e6f9e582bf7fa671. No global timeout changed.

Reran the exact four suites: **20/20 PASS**, roundtrip 4019ms; /private/tmp/cycle5-repair-140-timeout-green.log. Test-only ESLint again 0 errors / 0 warnings; /private/tmp/cycle5-repair-140-timeout-eslint.json. Current manifest refrozen at /private/tmp/cycle5-repair-140-manifest.json; original retained at -first-freeze-manifest.json. Native AC2 remains pending. Official task notes updated. UAT141/142 private fixtures/report retained unchanged and their paths sent to account_access.
