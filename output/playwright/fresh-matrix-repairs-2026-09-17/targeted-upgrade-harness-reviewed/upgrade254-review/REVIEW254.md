# UAT254 independent review — TASK13260.196

**Verdict: no actionable findings. The corrected helper is clear for the parent's separately controlled retry with a new copy-run identity. Native startup/readback acceptance remains pending.**

## Scope and exact bytes

Product candidate: `a7d3155a567afb25982eb360ea24b973cc3249c9`.

| File | SHA256 |
| --- | --- |
| Current matrix-upgrade.mjs | `5e0cc74c2c66a5c01eb7fc2f0d1680cffe018c072a892dd566041cdfac73567f` |
| Current matrix-upgrade.test.mjs | `f6ab8c0e5f03139fd0360272c47453ed48c798130fd30d5884ae188c8c1f950f` |
| Actual next.config.mjs | `55f31677133e8fc03759d961e6bca7534c1a6d007fd0554723ee54d5bd5e3188` |
| Retained pre-fix helper | `1d3254bc0c7403770348132974b42ff28568051fc5e3a06a829f920b410c01e6` |

All current hashes match the author's source-freeze.json. The actual Next config is byte-identical to the fixed product candidate. The helper differs from the retained, previously reviewed helper by exactly `.next-upgrade-` to `.next-live-tier-upgrade-`. Existing test changes are the two corresponding directory expectations; the added regression imports the actual repository Next configuration with the synthetic frontend environment captured from the helper's spawn boundary.

## Independent verification

- **Causal RED:** the added regression, selected by test name with `UPGRADE_TEST_SOURCE` pointing to the retained old helper, failed at actual next.config.mjs:28 with `TLDW_NEXT_DIST_DIR must be a direct .next-live-tier-* child directory`. One test, one expected failure, zero skips, exit 1. See red-independent.log.
- **Combined GREEN:** `node --experimental-vm-modules --test .tmp/uat-repairs-231-246/native-upgrade-preparation/matrix-upgrade.test.mjs .tmp/uat-next-matrix-20260916/matrix-launcher.test.mjs .tmp/uat-matrix-browser-20260917/browser-wrapper.test.mjs` passed **125/125**, zero failures/cancellations/skips/todos, exit 0. See combined-independent.log and verification.json.
- Independent Node syntax checks passed for both touched files. Installed ESLint with the existing project configuration parsed both files with zero errors/warnings. The launcher still differs from its retained original baseline only by the three previously reviewed exports. See static-independent.json.
- Inspected the author's scoped Bandit result: both touched JavaScript files produce parse limitations. That is not JavaScript security certification. Static inspection found no new security issue in the prefix/test change.

## Preservation and retry conditions

The prefix satisfies the existing direct-child naming guard for the helper's validated run/cell characters. Build-directory ownership checks remain intact. The correction changes no initialization, profile, holder, archive, source-proof, binding, process-receipt, signal, port, runtime-role, or browser-origin logic. The prior 124 checks remain present and pass.

The helper hash is part of an immutable binding. Preserve the failed run's binding, build directory and attempt receipts; use a distinct new copy-run and released source gate for the corrected helper. Keep the original profile identity `repairs231-250-targeted-20260917` and product candidate unchanged. The parent owns any process stop, copy, launcher, or acceptance action.

This reviewer performed no real launcher/runtime/browser/database/network/copy actions and no source, task, tracker, Git, profile, archive, initialization, holder, binding, or receipt edits. Live APIs 55121/55385 and PostgreSQL holders 18859/18878 were not touched. Writes were limited to this review packet and automatically cleaned synthetic test fixtures. Tests imported the actual Next configuration but did not start Next or certify native startup.
