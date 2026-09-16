# Independent review — UAT106 / UAT115 model ownership

## Result

No actionable correctness or security findings in the frozen ten-file patch. The reviewed change is ready for the coordinator's targeted live acceptance and combined verification.

## Scope and evidence

- Reviewed implementation notes at `/private/tmp/cycle4-model-selection-implementation.md`, Task 2 of `IMPLEMENTATION_PLAN_uat_cycle_4.md`, and the model handoff requirement in `Docs/Design/2026-09-16-uat-cycle-4-repairs.md`.
- Reviewed the production diff for `UnifiedSetupWizard.tsx`, `steps/FirstChatStep.tsx`, `useSelectedModel.ts`, and `AnalysisModal.tsx`, and the six frozen test files.
- Inspected adjacent real boundaries: `useSetupOnboarding` verification/completion refresh behavior, WebUI storage and useStorage shims, installed extension Storage/useStorage implementation, canonical TldwApiClient storage areas and manual credential writes, provider-qualified model parsing, and target/authorization comparison helpers.
- SHA256 checks before and after review matched all 10 entries in `/private/tmp/cycle4-model-selection-frozen-manifest.json` (HEAD recorded there: `6d2f3abc559a2ff229d50664857bf22384a7d86b`). No repository edits, browser/runtime operations, inference, or commits performed.

## Reviewed guarantees

- Successful setup verification is tied to captured target, credentials, and mounted generation; delayed completion and final state publication re-check that authority. The extension watcher uses local storage, matching canonical configuration persistence, and its actual-extension regression exercises an A-to-B-to-A transition without window events.
- Setup publication preserves hydrated, legacy-serialized, existing, and newer choices. A store-operation revision also protects a choice followed by an explicit clear. Catalog resolution requires a unique provider-and-model match and produces a provider-qualified identifier.
- The successful backend completion response is retained during the mounted handoff. Rejected local persistence leaves a Finish setup retry that does not repeat successful inference or acknowledged backend completion. State publication follows the awaited explicit storage write.
- Media explicit selection uses the shared model owner. The real WebUI-storage regression checks both durable and in-memory selection and the outgoing request's exact model/provider rather than only Select rendering.

## Independent verification

From `apps/packages/ui`, ran:

```sh
./node_modules/.bin/vitest run \
  src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.extension-authority.test.tsx \
  src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.model-handoff.test.tsx \
  src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx \
  src/components/Media/__tests__/AnalysisModal.model-owner.test.tsx \
  src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx \
  src/components/Media/__tests__/AnalysisModal.stage3.regression.test.tsx \
  src/components/Media/__tests__/AnalysisModal.stage1.cancel.test.tsx \
  src/utils/__tests__/model-startup-selection.test.ts \
  src/hooks/__tests__/useSetupOnboarding.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

Result: **103 tests passed in 9 files**, exit 0, 10.44 seconds. An earlier focused four-suite run also passed 45 tests. Node emitted its existing experimental localStorage warning; no test failures or unhandled errors were reported.

## Limits

This is a bounded code/test review, not live UAT, a compiler run, or full application verification. Setup integration tests mock the service hook/API boundary; the production hook's refresh behavior was reviewed separately. Persistence retry guarantees are for the current mounted wizard, as documented. The review does not cover unrelated QA, modal error handling, or UAT107 inventory repair. Author-reported baseline lint comparison was not independently rerun.
