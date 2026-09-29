# UAT371: use the existing external sidepanel theme bootstrap

Task: TASK-13260.277.21. Own scope: extension sidepanel index.html and the existing options-theme-bootstrap test. Root owns fresh packaging, native console acceptance, tracker and git integration.

## Approved design

The native sidepanel inline script hash exactly matches the blocked CSP console entry. It only applies early theme preference; application startup continues and React later applies theme. Options already loads the shared public `/theme-bootstrap.js` synchronously. Reuse that same script tag in sidepanel and keep the current CSP. No new bootstrap or build configuration is required.

## Stage 1: Causal coverage
**Goal**: Cover both actual extension entrypoint HTML files with the existing bootstrap tests.
**Success Criteria**: Sidepanel fails external-script/no-inline checks; existing theme/storage controls pass.
**Tests**: Same-origin synchronous head script, no executable inline body, public artifact exists, stored dark/system/light preferences and storage failures.
**Status**: Complete

## Stage 2: Minimal HTML repair
**Goal**: Replace only the sidepanel inline script with the shared external script tag.
**Success Criteria**: Both entrypoint contracts and existing theme behavior pass without changing CSP or shared bootstrap.
**Tests**: Focused installed Vitest command; read current WXT publicDir/entrypoint configuration and existing packaged options bootstrap.
**Status**: Complete

## Stage 3: Verification and native handoff
**Goal**: Freeze reviewed source for root packaging/native console acceptance.
**Success Criteria**: Tests/lint and security applicability recorded; root independently reviews; fresh built sidepanel/options contain the shipped script and native console has no bootstrap CSP error.
**Tests**: Source-level bootstrap and VM controls; scoped lint/format checks; root candidate2 build/native console.
**Status**: In Progress

## Evidence

Initial native observation: `.tmp/uat-frontend-repair1-20260920/sqlite-multi-ext-039-native-console.txt`. Exact inline SHA256 base64: `KBN6R1KcvjRVpG1rNylmQazReH5ay4n5vCmBrCLTJSE=`. Existing packaged options uses the external script; sidepanel uses the blocked inline body. No frozen artifacts will be edited.

Causal RED: `/tmp/uat371-causal-red.log`, 2 failed / 16 passed. Only the new sidepanel external-script and no-inline-script checks failed; options and theme/storage controls passed. Test command from `apps/extension`: `node ../packages/ui/node_modules/vitest/vitest.mjs run tests/unit/options-theme-bootstrap.test.ts --maxWorkers=1 --no-file-parallelism`.

GREEN: `/tmp/uat371-green.log`, 18 tests. Final bootstrap plus post-build Vitest suites: 22 passed (`/tmp/uat371-final-vitest.log`). Existing WXT publicDir tests use Bun and passed 3/3 with installed `bun test tests/unit/wxt-config-public-dir.test.ts` (`/tmp/uat371-wxt-config-bun.log`). The initial combined Vitest command retained 22 passing tests but could not import `bun:test` for that separate suite (`/tmp/uat371-final-tests.log`); no test changes or skips were used to resolve the runner mismatch.

WXT configuration points `entrypointsDir` to extension entrypoints and `publicDir` to `apps/packages/ui/src/public`; post-build tasks do not rewrite this tag. The shared bootstrap is byte-identical to candidate1's already-shipped `extension-native/theme-bootstrap.js`, SHA256 `691ddeffd19a00c6e3661dec46aa2624455219dfbeda8248d95f230ad9c3f479`. Root will create the new built candidate; no existing build was touched.

Scoped test TypeScript passes (`/tmp/uat371-test-typecheck-final.log`) using a temporary config mapping Vitest and Node types to their installed workspace locations. Initial standalone module/type resolution attempts are retained in `/tmp/uat371-test-typecheck.log` and `/tmp/uat371-test-typecheck-mapped.log`; no dependencies added. Production change is HTML only. Matched test lint: 0 errors/warnings before and after (`/tmp/uat371-lint-comparison.json`); Prettier passes (`/tmp/uat371-format.log`). Bandit ran over the HTML/TypeScript entrypoint directory with the project virtualenv and reports 0 Python LOC (`/tmp/bandit_uat371.json`), not JS/HTML coverage. Manual review confirms no CSP relaxation or script behavior change.

Source frozen, four-file manifest `/tmp/uat371-changed-files.txt`. Independent root review and fresh packaged options/sidepanel native console acceptance remain pending. Task stays In Progress; a visible theme flash was never claimed as natively reproduced.
