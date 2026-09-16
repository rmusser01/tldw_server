# Independent review — UAT106 pre-auth setup handoff correction

## Verdict

**Clear for parent integration and targeted native verification. No actionable findings in the frozen two-file correction.**

Reviewed against base `bc15c22657106da118c8e38547dafd8a2800675c`, frozen at `2026-09-16T03:46:24.383Z` in `/private/tmp/cycle4-setup-preauth-code-freeze.json`. Both product/test SHA-256 hashes matched after independent tests. Scope:

- `apps/packages/ui/src/components/Option/Onboarding/UnifiedSetupWizard.tsx`
- `apps/packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.model-handoff.test.tsx`

No repository edits, runtime/browser/inference calls, whole TypeScript run, commits or subagents by this reviewer.

## Contract review

The previous catalog round-trip was incompatible with the permitted local first-run flow: the actual model service returns an empty catalog before a usable browser API key is configured. The correction leaves that protected-service guard intact. It publishes the already exact-matched successful verification response, through the existing shared selected-model owner, before publishing completion to the parent.

Canonicalization preserves provider slots and the exact verified model string, including slash/colon IDs. The existing parser handles recognized provider names; availability normalization and the two local aliases cover the real setup spellings (including koboldcpp and the second custom slot). Unrecognized providers fail visibly instead of selecting a guessed route. Different provider/model/slot verification responses still fail before completion.

The removed await and its now-redundant guards belong solely to catalog resolution. The existing checks after completion and before publication still validate captured authority/generation; selection revision checks still protect existing/newer/cleared choices. Awaited device persistence, cached successful server completion, retry without duplicate verification, unmount, account/target A-to-B-to-A, and final-refresh invalidation remain covered.

## Independent execution

**85 permanent tests passed across 4 suites**, exit 0, with the WebUI Vitest config:

```sh
cd apps/tldw-frontend
./node_modules/.bin/vitest run \
  ../packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.model-handoff.test.tsx \
  ../packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.extension-authority.test.tsx \
  ../packages/ui/src/services/tldw/__tests__/TldwModels.test.ts \
  ../packages/ui/src/utils/__tests__/resolve-api-provider.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

Log: `/private/tmp/cycle4-uat106-preauth-independent-green.log`.

This independently exercised the permanent actual no-key boundary: real TldwApiClient configuration reader, real TldwModelsService, real WebUI storage and selected-model owner, actual Wizard/FirstChatStep. The service returned an empty protected catalog, setup published the verified canonical choice, and no protected network/catalog call occurred. Setup verification/completion responses are controlled fixtures, not real inference.

**Two additional independent probes passed**, injected in memory without editing repository files:

1. With a blank API key, reject the first verified-model storage write, choose a newer durable model, then use Finish setup. The newer store/storage choice survives; server completion and verification each occur once; no catalog lookup occurs.
2. With a blank API key, hold verification while real WebUI local Storage writes target A→B→A. The old response is rejected as connection-changed, no preference is published, and completion is never called.

Reproduce:

```sh
cd apps/tldw-frontend
./node_modules/.bin/vitest run \
  --config /private/tmp/cycle4-uat106-preauth-review.config.mjs \
  ../packages/ui/src/components/Option/Onboarding/__tests__/UnifiedSetupWizard.model-handoff.test.tsx \
  -t 'independent:' --maxWorkers=1 --no-file-parallelism
```

Probe source `/private/tmp/cycle4-uat106-preauth-review-probes.txt`; result `/private/tmp/cycle4-uat106-preauth-private-probes.log`. Two pass, 36 existing tests filtered out in this probe run. Counts overlap the author's 114-test run and should not be added to it as separate coverage.

## Evidence limits

No new native acceptance is claimed. Parent owns the retained failed first-chat page, API/browser lifecycle and exact normal-Chat outgoing provider/model check. The automated tests control setup responses and exercise browser storage in jsdom; the actual installed extension authority test covers storage events, not native extension navigation. Completion caching remains a mounted-owner lifetime behavior; reload/HMR recovery is not a new guarantee.

Author/root static evidence reports ESLint 0 errors/0 warnings and an exact 90-diagnostic TypeScript baseline with zero added/removed; the reviewer did not rerun whole TypeScript. No Python files changed, so Bandit is not applicable. The protected model authentication guard and shared registry were not edited.
