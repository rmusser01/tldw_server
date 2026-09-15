# TASK13260.36 / UAT095: failed Chat completion error handling

Durable automated evidence for the bounded failed-Chat transport repair. Production and tests froze at **2026-09-15T21:59:38.149Z**; final independent review is clear after correction of its cancellation finding. The seven entries in owned-paths.json match at bundling: one production module, five tests, and the associated task. Only the task-record hash was refreshed for later review/evidence notes; original production/test hashes and freeze time remain unchanged.

Failed canonical POST /api/v1/chat/completions responses reuse the existing server-error sanitizer before request warnings, stored diagnostics, rejected errors and returnResponse. Nested structures, actionable validation text, status, existing codes and retry metadata remain available. Successful assistant content and non-Chat routes retain their existing behavior. The existing raw cancellation classification is preserved before shortening display text. No client/domain/auth production changes belong to this repair.

## RED, correction and overlapping GREEN counts

| Evidence | Result | Meaning |
| --- | --- | --- |
| red-confirmed.log | 5 failed, 7 passed, 109 filtered | Original real parsed direct/extension failed-response sanitation regressions. |
| quick-test-red.log | 1 failed, 1 passed | Original proxy replay exposes a raw-path notification through the actual Quick Test hook; successful output control already passes. |
| quick-test-green.log | 2 passed | Corrected Quick Test notification and exact successful output. Both cases also occur in the full run. |
| independent-cancellation-red.log | 2 failed, 121 filtered | First sanitation implementation shortened a multiline text-only cancellation and lost AbortError/REQUEST_ABORTED. |
| independent-cancellation-baseline.log | 2 passed, 121 filtered | The same unchanged independent probe passes against the exact pre-repair proxy. |
| cancellation-permanent-red.log | 2 failed, 121 filtered | Permanent direct/extension regression reproduces that review finding before correction. |
| independent-cancellation-final.log | 2 passed, 123 filtered | Unchanged original independent probe after correction. |
| final-corrected-tests.log | 217 passed, 7 suites | Implementer final relevant regression run. |
| independent-final-tests.log | 217 passed, 7 suites | Independent rerun of the same relevant cases. |

Counts overlap and must not be added as unique coverage. Filtered/skipped tests are not passes. The final runs include permanent rejection and returnResponse cancellation checks, actual HTTP500/422 parsing, exact successful payload preservation, nested details, retry-after values, non-Chat route negatives, current scope policy and Quick Test behavior.

## Baseline assertions and original probes

initial-sanitization-investigation.md and its probe/log retain the original diagnosis. That probe intentionally expects the old raw diagnostic leak and is historical characterization, not a final passing expectation. initial-sanitization-existing-tests.log records the stale successful-response sanitization assertion. The corrected test now requires complete successful content preservation; actual failed-transport coverage is separate.

web-refresh-baseline.log records one stale exact-GET negative failure with seven negative controls passing and 52 filtered. The committed policy permits GET /api/v1/chats/{id}/messages; its negative fixture now uses the unsupported /messages/extra child and retains no-dispatch checks. The utility test's obsolete literal Bearer marker-format assertion was removed while all checks that synthetic secret values are absent remain active. The sanitizer utility itself was not changed.

before-proxy.ts and baseline-background-proxy.ts are byte-identical exact original production source snapshots, retained under both names because the original configs reference different paths. These are source files, not private runtime configuration. Original TypeScript probe/source files and JSON retain their bytes. Logs and Markdown normalize only trailing horizontal whitespace and final newlines. Historical reports retain their original status wording; independent-review.md contains the final clear disposition.

## Static checks and limits

The retained ESLint current/baseline/comparison JSON covers all six changed source/test paths: **0 errors, 100 unchanged warnings, 0 added**. The root pages-directory advisory is retained in eslint-baseline-stderr.log. Scoped diff-check passed during review. Combined TypeScript is **pending parent verification** after adjacent work freezes; the known 90-diagnostic baseline is not a clean-compiler claim. Bandit does not apply to this TypeScript-only repair.

No live backend disclosure was observed. All transport responses use synthetic fetch data; no real model inference or server is contacted. Extension tests forward the runtime payload at a mocked messaging boundary to the real request core; they do not certify a launched extension worker or popup lifetime. The actual Quick Test hook and notification sink are exercised, with unrelated startup/catalog dependencies stubbed. Native error-path acceptance and the complete repaired single/multi workflow acceptance remain pending in this bundle. No browser/runtime or application changes were made to prepare evidence.

## Retention and replay

Configs preserve the original absolute checkout and /private/tmp references. To replay elsewhere, provide the referenced pre-repair source snapshots at their original paths or use copies of the configs with adjusted paths; the retained originals must remain untouched. The unchanged independent probe uses UAT036_BASELINE=1 to load baseline-background-proxy.ts. The focused final command is in repair-report.md; the independent probe command and evidence are in independent-review.md.

All 26 copied artifacts and this README are scanned against **14 known isolated runtime credential values** and JWT/private-key patterns before writing; no values or private runtime manifests are copied. SHA256SUMS covers every retained artifact except itself. The builder is /private/tmp/uat036-build-evidence.mjs, following chat-titles034 conventions and the scanner from /private/tmp/uat093-build-evidence.mjs. Verification summary is /private/tmp/uat036-evidence-build-result.json.
