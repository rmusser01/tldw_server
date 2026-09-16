# Cycle4 Task2: model selection handoff (UAT106 / UAT115)

Status: frozen and ready for independent review. No staging, commit, browser, runtime, inference, or external-provider action was performed. Parent owns combined compiler and native acceptance.

## Scope and root cause

Approved design: Docs/Design/2026-09-16-uat-cycle-4-repairs.md; IMPLEMENTATION_PLAN_uat_cycle_4.md Task2. Official tasks13260.47 and13260.55 were marked In Progress before repository edits and remain In Progress.

Setup verified its own provider/model but never published that choice to regular Chat. With no selectedModel, regular Chat chose the first catalog item (the unrelated Ollama model). Media AnalysisModal independently wrote selectedModel storage while the consolidated hook treated its existing store value as authoritative and reverted that write.

### Final implementation

- UnifiedSetupWizard captures canonical single-user authority, lifetime generation, and selection-operation revision before verification; verifies the returned provider/model, and checks ownership across completion, catalog resolution, preference write, and final parent refresh. Only an initially empty, still-eligible selection is seeded. Existing and newer deliberate choices, including choose-then-clear, remain authoritative.
- Catalog resolution requires exactly one matching provider/model and writes the canonical provider-qualified choice. Existing provider normalization handles custom_openai/custom_openai_api spelling. Missing/ambiguous choices stay actionable.
- Successful server completion is cached in the mounted handoff. Explicit preference persistence is awaited before parent completed-state publication can unmount setup. Failed local publication keeps verified response and exposes Finish setup. Retry does not repeat inference or a previously successful server-completion request. The parent approved this bounded UX decision.
- useSelectedModel waits for storage hydration, ignores a stale render synchronization, normalizes serialized legacy strings, and returns the explicit setter persistence promise. Existing consolidated store ownership is retained.
- AnalysisModal calls that consolidated setter; a rejected device write shows a local retry message. Existing analysis request error/cancellation/logging logic was not broadened (later QA unit owns it).
- Final extension correction uses explicit local config storage for canonical authority watch/unwatch, matching TldwApiClient on both platforms. Selected-model default storage semantics remain unchanged. This was required because installed extension Plasmo defaults to sync while the WebUI shim defaults to local.

## Owned files

Four production and six focused test files (three new tests, three existing fixture adjustments):

- apps/packages/ui/src/components/Media/AnalysisModal.tsx
- apps/packages/ui/src/components/Media/__tests__/AnalysisModal.model-owner.test.tsx
- apps/packages/ui/src/components/Media/__tests__/AnalysisModal.stage1.cancel.test.tsx
- apps/packages/ui/src/components/Media/__tests__/AnalysisModal.stage3.regression.test.tsx
- apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx
- apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.extension-authority.test.tsx
- apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.model-handoff.test.tsx
- apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx
- apps/packages/ui/src/components/Option/Onboarding/steps/FirstChatStep.tsx
- apps/packages/ui/src/hooks/chat/useSelectedModel.ts

Additional official task records are13260.47 and13260.55. No PlaygroundForm, global design/plan, dependencies, auth service, or other producer file was changed by this unit. Concurrent owners have unrelated working-tree changes.

## Evidence and commands

### RED before corresponding implementation

- /private/tmp/cycle4-model-selection-setup-red.log: 8 failed /5 passed, actual mounted Wizard + FirstChatStep + real WebUI storage/selection owner. Parent unmounts on completed-state notification.
- /private/tmp/cycle4-model-selection-media-red.log: 1 failed, actual AnalysisModal plus concurrent consolidated model owner, proving selected choice reversion.
- /private/tmp/cycle4-model-selection-extra-red.log: 1 failed /22 passed, catalog using setup provider spelling.
- /private/tmp/cycle4-model-selection-extension-red.log: 1 failed, actual installed extension Storage and useStorage with an in-memory browser.storage API. Pending completion survived local config A→B→A without window events and incorrectly published the choice/workspace. Same permanent assertion passes after explicit-local watch correction.

### GREEN

Before the final explicit-local storage correction, the complete focused unit passed102 tests /8 suites: /private/tmp/cycle4-model-selection-final-tests.log. From apps/packages/ui:

```sh
./node_modules/.bin/vitest run \
  src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.model-handoff.test.tsx \
  src/components/Media/__tests__/AnalysisModal.model-owner.test.tsx \
  src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx \
  src/components/Option/Onboarding/__tests__/FirstChatStep.test.tsx \
  src/components/Media/__tests__/AnalysisModal.stage3.regression.test.tsx \
  src/components/Media/__tests__/AnalysisModal.stage1.cancel.test.tsx \
  src/utils/__tests__/model-startup-selection.test.ts \
  src/hooks/__tests__/useSetupOnboarding.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

After the final correction and formatting, the covering setup suites passed56 tests /3 suites: /private/tmp/cycle4-model-selection-extension-green.log. From apps/packages/ui:

```sh
./node_modules/.bin/vitest run \
  src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.extension-authority.test.tsx \
  src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.model-handoff.test.tsx \
  src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

Together these cover103 distinct tests across9 suites; this is not a claim that one final all-nine-suite command ran. The final correction changes only config watcher storage area and adds its extension regression. Per parent request only covering tests were rerun.

Behavior controls include delayed target/key/A→B→A/unmount, failed/unknown verification, delayed hydration, stored legacy values, newer choice during verification/catalog, choose-then-clear, no same-name foreign-provider choice, missing/ambiguous catalog, local write rejection/retry, server failure/retry, and authority replacement during final state refresh. Media asserts both durable selection and exact outgoing model/provider.

### Lint, formatting, diff, compiler

- Root ESLint command: apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs <all10 paths from owned-paths.json> --format json.
- /private/tmp/cycle4-model-selection-eslint-final.json and -comparison.json confirm actual coverage of10 files, current1 error/18 warnings, exact HEAD baseline1 error/18 warnings, zero added. Existing require-yield error in AnalysisModal.stage1.cancel.test.tsx and existing any warnings remain unchanged; lint is not claimed clean. HEAD baseline was obtained by exact HEAD bytes piped with each real source filename. Missing root pages directory advisory retained in -eslint-stderr.log.
- Targeted Prettier ran on changed setup/hook/new tests; no whole-file AnalysisModal formatting.
- /private/tmp/cycle4-model-selection-diff-check.log: all owned paths have no whitespace diagnostics (new-file no-index exit1 means differences, not a whitespace failure).
- No individual TypeScript invocation, by parent instruction. Combined compiler is pending; known prior baseline90 is not a clean-typecheck claim.
- Bandit not applicable: no Python changed in this unit.

## Freeze and limits

- /private/tmp/cycle4-model-selection-frozen-manifest.json: exact10 code/test SHA256 values and timestamp.
- /private/tmp/cycle4-model-selection-production-freeze.json: exact4 production SHA256 values.
- /private/tmp/cycle4-model-selection-owned-paths.json: machine-readable review scope.
- Real WebUI shim and actual installed extension storage/hook are exercised with deterministic mocked network/server APIs; these tests do not certify native browser/extension or real provider behavior.
- The successful completion retry cache is intentionally component-lifetime only. No durable pending-completion state or full-page reload retry claim is introduced.
- Canonical authority changes conservatively invalidate setup handoff. No new same-account token-rotation continuity promise is introduced.
- Independent review, final combined compiler, fresh setup→regular Chat native request and native explicit Media selection/generation remain pending. No full UAT or acceptance completion claim.
