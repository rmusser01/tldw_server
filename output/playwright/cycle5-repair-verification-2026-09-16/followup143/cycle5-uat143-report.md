# UAT143 / TASK-13260.82 — stale Form fixtures and guards

## Result

Six assigned test files corrected; no production changes. Final current-source run: **35 passed / 6 suites**, exit 0, no unhandled errors. Final corrected tests against **3c30685611** production: **35 passed / 6 suites**. These runs overlap; they are not 70 distinct tests. Independent review and root integrated verification remain pending.

## Baseline and causes

Unchanged tests against read-only 3c30685611 production replay reproduced **27 failed / 6 passed**, plus **3 unhandled rejections**, exactly matching the newly included six suites in the 150-suite combined log. The private config substitutes changed UI production modules in memory and redirects source-guard fs reads to Git baseline bytes; it does not swap repository source files. Original test copies/hashes are retained in `/private/tmp/cycle5-uat143-before/` and `-before.json`.

| Suite | Cause | Final correction |
| --- | --- | --- |
| document-processing | Dependency fixture omitted `beginPromptAssistReset` and `markPromptAssistAttemptSaved`; submit failed before send. | Supply current callback contract. All six document behavior cases unchanged. |
| follow-up-research | Generic query mock returned an array where unrelated Prompt Assist expected a capability object; then missing Drawer. Home milestone hook also reached missing mocked client getConfig. | Isolate unrelated Prompt Assist and Home scope using existing Form-test boundaries. Five research cases/assertions unchanged. |
| image-refine.integration | Same unrelated Prompt Assist/Home fixture gaps. | Same boundaries; retain actual image refinement, real composer/submit, preview and dictation paths. Fifteen cases/assertions unchanged. |
| voice-visibility.integration | Partial composer fixture lacked Prompt Assist revision contract; then unrelated Drawer missing. | Isolate Prompt Assist. Both voice/error-diagnostics cases/assertions unchanged. |
| composer-options.guard | Exact contiguous class substring rejected added items-end/gap-2 classes. | Check all original required mobile send-row class tokens on the identified mobile element; retain two-column and forbidden three-column checks. Four cases retained. |
| llamacpp-controls.guard | Serialization moved from Form into usePlaygroundRawPreview; old source-location assertions searched the wrong module. | Exercise real extracted hook and assert all five exact payload fields for library/inline/none. Replaces one static case with three behavioral cases. |

Initial partial capability/revision fixture expansion was abandoned after exposing the unrelated Drawer boundary. The next run isolated Prompt Assist and exposed Home getConfig. Before the third correction, both remaining fixture graphs were audited against working pinned-fallback/openui fixtures and current hooks; no change to behavior under test was needed. All intermediate logs remain retained. No product defect established by these six failures.

## Files

Under `apps/packages/ui/src/components/Option/Playground/__tests__/`:

- `PlaygroundForm.composer-options.guard.test.ts`
- `PlaygroundForm.llamacpp-controls.guard.test.ts`
- `PlaygroundForm.document-processing.test.tsx`
- `PlaygroundForm.follow-up-research.test.tsx`
- `PlaygroundForm.image-refine.integration.test.tsx`
- `PlaygroundForm.voice-visibility.integration.test.tsx`

Also the official TASK-13260.82 record, updated through Backlog CLI only. SHA-256 hashes and timestamp: `/private/tmp/cycle5-uat143-manifest.json`.

## Reproduce

Working directory: `/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui`.

```sh
./node_modules/.bin/vitest run src/components/Option/Playground/__tests__/PlaygroundForm.{composer-options.guard.test.ts,llamacpp-controls.guard.test.ts,document-processing.test.tsx,follow-up-research.test.tsx,image-refine.integration.test.tsx,voice-visibility.integration.test.tsx}
```

Baseline production replay uses the same command with `--config /private/tmp/cycle5-uat143-baseline.config.mts`.

Evidence:

- `/private/tmp/cycle5-uat143-baseline-red.log`: unchanged baseline 27 failed / 6 passed / 3 unhandled.
- `/private/tmp/cycle5-uat143-corrected-first.log`: 23 failed / 10 passed; pending stale llama guard and unrelated Drawer.
- `/private/tmp/cycle5-uat143-corrected-second.log`: 20 failed / 15 passed; unrelated Home scope mock.
- `/private/tmp/cycle5-uat143-corrected-third.log`: final current-source 35/6 GREEN.
- `/private/tmp/cycle5-uat143-baseline-corrected-green.log`: same corrected tests on baseline 35/6 GREEN.

Static command from repository root:

```sh
apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundForm.{composer-options.guard.test.ts,llamacpp-controls.guard.test.ts,document-processing.test.tsx,follow-up-research.test.tsx,image-refine.integration.test.tsx,voice-visibility.integration.test.tsx} --format json
```

`cycle5-uat143-eslint-{final,baseline,comparison}.json`: all six files covered; **0 errors, 73 unchanged baseline warnings, 0 added/removed signatures**. Comparison ignores line movement and preserves multiplicity. Root config emits its existing pages-directory advisory. Node test runner emits its existing localStorage experimental warning; no unhandled errors. Scoped `git diff --check` clean. Bandit N/A (TS/TSX tests only); whole compiler parent-owned.

## Limits

No browser, runtime, provider inference, production code, dependency, skipped test, or assertion suppression change. Prompt Assist/Home ownership is deliberately outside these suites and remains covered in its own suites and the accepted UAT013 controls. The mobile check is an existing source guard, not native geometry proof. The llama cases test the actual extracted preview builder, not a live provider. Root owns independent review, combined 150-suite run, retention and commit. UAT013 production/test files remained untouched.
