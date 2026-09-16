# UAT068 / TASK13260.15 — frozen repair

## Result and cause

Ready for independent source review; native verification pending. A saved Character URL kept republishing its old conversation during explicit picker replacement and delayed New saved navigation. Header also cleared identity before a dirty-navigation cancellation was accepted.

The accepted action now synchronously retires only its captured route/target/history/restore generation. Existing router navigation consumes the URL; profile continuations from the retired route cannot republish it. Search/hash controls are preserved. A later deliberate visit to the old saved route can load it normally. Header updates selection and mode only after clear accepts. No saved records, equal-text rows, loaders or authentication policies are discarded.

## Frozen files

- apps/packages/ui/src/utils/character-chat-mode-intent.ts
- apps/packages/ui/src/components/Common/AssistantSelect.tsx
- apps/packages/ui/src/hooks/chat/useClearChat.ts
- apps/packages/ui/src/components/Layouts/Header.tsx
- apps/packages/ui/src/components/Option/Playground/Playground.tsx
- apps/packages/ui/src/components/Option/Playground/__tests__/Playground.coordinator.integration.test.tsx
- apps/packages/ui/src/components/Layouts/__tests__/Header.character-mode.test.tsx
- apps/packages/ui/src/hooks/chat/__tests__/useClearChat.settings-navigation.test.tsx
- apps/packages/ui/src/components/Common/__tests__/AssistantSelect.behavior.test.tsx

Exact source/test/task hashes: cycle5-repair-chat-068-manifest.json.

## Validation

- Original route regression: cycle5-repair-chat-068-red.log, 3 RED / 1 GREEN. Separate actual Header Cancel RED: header-cancel-red.log.
- Final real Header fixture replay against unchanged ba5233a5a9 modules: actual-header-baseline-red.log, 4 expected RED / 1 GREEN (54 unselected). This includes rapid replacement; no checkout/source swapping. Config: baseline.config.ts.
- Final focused suite: focused-final.log, **200 tests / 13 suites PASS**. Actual Playground/Header/clear/selected storage and loader boundaries run with controlled transport, DB adapter and route clock. No native Next timing claim. Related mirror, selected-assistant, Notes backlink and session controls included.
- Additional deliberate old-route revisit and rapid A-B-A regressions were first RED, then GREEN (return-red.log, roundtrip-red.log, roundtrip-green.log).
- Fixture-only maintenance: AssistantSelect's mocked store now exposes real getState API; all 31 picker behaviors pass. The intermediate broader run failed only this missing mock interface, retained focused-green.log.
- After final 200-pass run, removed only an unused local promise binding from the clear fixture; no behavior change. ESLint rerun uses repository-root explicit apps/tldw-frontend/eslint.config.mjs over all 9 code/test files: **0 errors / 22 unchanged baseline warnings**, 0 added/removed. Comparison normalizes embedded moved line references (eslint-comparison.json; lint-compare.mjs).
- Scoped git diff --check clean. Bandit N/A: TypeScript-only unit. Parent owns whole compiler checkpoint.

## Limits

No browser, API, inference, runtime, staging or commit action. This unit does not claim UAT131 promotion ACK repair or native acceptance. Existing baseline lint debt remains. No account or saved-history data was rewritten.
