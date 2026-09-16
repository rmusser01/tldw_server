# UAT106 / TASK13260.47: finish setup before browser authentication

Status: frozen for independent review at 2026-09-16T03:46:24.383Z. Base commit: bc15c22657106da118c8e38547dafd8a2800675c. No browser, runtime, provider inference, backend, whole TypeScript, staging or commit actions were performed.

## Root cause and bounded repair

The real TldwModelsService intentionally returns an empty catalog when the browser's single-user config has no usable key (TldwModels.ts isConfiguredForModels/getModels). Native first-run setup legitimately verifies the provider/model and completes before the separate manual-key Settings flow. UnifiedSetupWizard previously required exactly one protected catalog match after successful first-chat and completion, so it stayed on First chat despite both server operations succeeding.

Only UnifiedSetupWizard production code changed. Its already matching, same-authority `ready` response now supplies the provider/model directly. The existing selection parser and provider-availability aliases canonicalize that pair, with two bounded local setup aliases for koboldcpp and the second Custom OpenAI slot. The exact model string (including slash/colon identifiers) must survive parsing; unknown providers remain actionable errors. There is no model catalog request, credential injection or authentication change.

Existing captured target/config generation, selection-operation revision, eligible empty selection, returned-pair comparison, awaited durable preference publication, completion retry cache and final refresh guard remain. The obsolete catalog-await guard was removed with its await; no artificial delay was added. A newer choice during real server completion is the replacement competing-selection regression.

The earlier missing/ambiguous catalog rejection contract was explicitly replaced by the parent-approved authoritative verified-pair contract. Those tests now prove that absent/duplicate catalog fixtures cannot override a successful matching first-run verification. Unknown provider, different provider, different custom slot and different model remain negative controls.

## Exact changed files

- apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx
- apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.model-handoff.test.tsx
- Official TASK13260.47 notes, updated using Backlog CLI; status remains In Progress with native acceptance pending.

## Reproduction and verification

All Vitest commands run from apps/packages/ui, using `./node_modules/.bin/vitest run ... --maxWorkers=1 --no-file-parallelism`.

### RED, before production correction

Command: run only `src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.model-handoff.test.tsx`.

Result: **11 failed /25 passed**, retained `/private/tmp/cycle4-setup-preauth-red.log`. Failures are the actual missing-catalog completion alert: real blank-key boundary, missing/duplicate catalog, and eight canonical setup/slot cases. No import or harness failures.

The blank-key case uses real TldwModelsService, actual TldwApiClient `getConfig`/initialize, real WebUI Plasmo storage adapter, real shared selected-model owner, Wizard and FirstChatStep. Setup network responses are deterministic mocks; the singleton config seam forwards to that actual client. It first asserts the blank key and empty catalog, then attempts the actual Send test chat UI. Final assertions also prove no protected model request or network call and no Wizard catalog invocation. Native inference is not claimed by this test.

### GREEN

Initial four-suite result: **89 passed /4 suites**, `/private/tmp/cycle4-setup-preauth-green.log`.

Final strengthened guard and adjacent controls: **114 passed /6 suites**, `/private/tmp/cycle4-setup-preauth-final-green.log`:

```sh
./node_modules/.bin/vitest run \
  src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.model-handoff.test.tsx \
  src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.extension-authority.test.tsx \
  src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx \
  src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx \
  src/utils/__tests__/resolve-api-provider.test.ts \
  src/hooks/__tests__/useSetupOnboarding.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

These counts overlap; do not sum. Final coverage includes existing/newer/cleared selection, hydration, key/target/A-to-B-to-A/unmount, actual installed extension local-storage authority boundary, failed/mismatched verification, completion retry, storage rejection/Finish setup retry without repeated inference, final-refresh invalidation, aliases and unknown provider. Node experimental localStorage warnings remain in logs; they are not test failures.

### Static checks

Root command: `apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs` followed by both exact changed code/test paths and `--format json`.

Current and exact HEAD-byte baseline: **0 errors /0 warnings, zero added/removed**. Raw results and comparison: `/private/tmp/cycle4-setup-preauth-lint-{baseline,current,comparison}.json`; reproducible runner `/private/tmp/cycle4-setup-preauth-static-check.mjs`, output `/private/tmp/cycle4-setup-preauth-static.log`. Root pages-directory advisory retained in the runner log. Owned `git diff --check` was clean. Both changed files formatted with the installed frontend Prettier.

No Python changed, so Bandit is not applicable. No whole TypeScript/build ran; parent owns combined compiler/native checks.

## Frozen evidence and limits

- `/private/tmp/cycle4-setup-preauth-owned-manifest.json`: both code/test hashes plus official task-record hash.
- `/private/tmp/cycle4-setup-preauth-code-freeze.json`: two code/test hashes and exact before hashes.
- `/private/tmp/cycle4-setup-preauth-production-freeze.json`: production-only subset.
- `/private/tmp/cycle4-setup-preauth-final.diff`: narrow final patch.
- `/private/tmp/cycle4-setup-preauth-paths.json`: exact code/test review scope.

Native failure is parent-owned evidence at `/private/tmp/uat-cycle4-targeted-single-first-chat.txt` and `/private/tmp/uat-cycle4-targeted-single-handoff-failure.png`. This implementation did not touch the waiting native browser. Native recovery, normal Chat outgoing selection and independent review remain pending. Successful completion caching remains mounted-component lifetime only; no reload retry guarantee is introduced.
