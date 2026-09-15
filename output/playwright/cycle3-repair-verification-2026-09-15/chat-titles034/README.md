# TASK13260.34: saved Chat browser and header titles

Reviewed automated evidence for the bounded title repair, frozen at **2026-09-15T21:42:49.051Z** after correcting the overlapping-rename review finding. All six production hashes match production-freeze.json at bundling. owned-paths.json records six production files, four tests and the associated task; it is a path inventory, not a hash guarantee for the mutable task record.

The Chat page owns its title with Next Head. Header uses canonical saved-conversation metadata and the current reactive local-title adapter. Saved rename carries captured scope/version; local rename stays local. Failed rename keeps the saved title and offers Retry rename. A second edit submitted during a pending save remains visible, with a saving status, for explicit commit after the first settles. Old account/selection completions cannot publish into the replacement selection.

## Automated results and overlapping counts

| Retained run | Result | Interpretation |
| --- | --- | --- |
| title-final.log | 88 passed, 10 suites | Final implementation run, including 15 Header/Head title cases and 1 Node SSR case. |
| rereview-title.log | 16 passed, 2 suites | Independent rerun of those same 15 title cases and 1 SSR case. |
| independent-transport.log | 32 passed, 2 suites | Independent scope/policy checkpoint; these 32 also appear in the 88-test run. Their production files did not change in the overlap correction. |
| independent-overlap.log | 1 failed, 14 filtered/skipped | Original review probe against the first freeze. |
| overlap-reproduced-red.log | 1 failed, 14 passed | Implementer reran the unchanged original probe before correction. |
| overlap-permanent-red.log | 1 failed, 14 filtered/skipped | Permanent overlapping-edit regression before correction. |
| overlap-independent-green.log | 16 passed | Corrected permanent title suite plus the unchanged original review probe; overlaps the final suite. |
| rereview-overlap.log | 1 passed, 15 filtered/skipped | Independent final run of the original probe alone. |
| independent-overlap-boundaries.log | 2 passed, 15 filtered/skipped | Additional independent first-save-failure and principal-invalidation controls. |

These counts must not be added as unique coverage. Filtered/skipped cases are not passes. title-red.log retains the initial 4 title failures; title-red-expanded.log retains 9 failures/21 passes before implementation. Additional edit-draft first-render and missing-retry REDs are also retained. The final independent review is clear; its historical initial-review section describes the superseded first freeze.

The integration fixture mounts actual Header/ChatHeader and Next Head/head manager, with a controlled reactive local adapter and controlled authority/transport fixtures. It is **not real browser IndexedDB acceptance**. title-ssr.log is a separate passing Node import/render check of the real shared hook with no browser globals; its case is also in the final 88. It does not certify a full Next build. No dependency/lock changes were made; the installed-only exploratory fake-indexeddb run mentioned in the historical report is intentionally not retained as acceptance evidence here.

## Static checks and baseline failure

ESLint covers all ten owned source/test files: zero errors and no added normalized warnings. The existing 532 client and 235 domain warnings are retained in title-lint-final.json and matched to baseline in lint-baseline-comparison.json; other owned files have zero warnings. Full TypeScript exits 2 with the same **90 existing diagnostic signatures**, no additions/removals (comparison preserves multiplicities while ignoring locations). This is not a clean typecheck. Bandit is inapplicable to this TypeScript-only scope.

An unrelated pre-existing Chat-completion sanitization assertion fails in title-broader.log and in api-baseline.log (baseline 4 passed/1 failed). The exact baseline client/domain sources are retained in api-baseline.json, with their private replay config. This failure was not fixed, skipped into a pass, or counted in the green final focused run. The existing version-conflict rename retry passes. api-baseline.json contains source text, not runtime configuration or user data.

## Native status and retention

**Native repaired title/rename/reload acceptance remains pending in this bundle**, owned by the parent UAT runner. The report's new-module HMR overlay was an environment observation while files were changing, not a passing or failed product acceptance result. No new browser, runtime or application actions were performed to prepare this bundle. The full fresh single/multi matrices also remain pending.

Original JSON and TypeScript/MTS probe bytes are preserved. Logs and Markdown normalize only trailing horizontal whitespace and final newlines. The reports/configs retain their original absolute checkout and /private/tmp references; replaying a probe elsewhere requires making its referenced source fixtures available. All 28 copied artifacts plus this README were scanned against 14 known isolated runtime credential values and JWT/private-key patterns, then indexed in SHA256SUMS. The original source/review freeze timestamps are retained; independent re-review is now clear.
