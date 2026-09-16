# UAT158 / TASK-13260.96 — llama.cpp admin List rendering repair

## Result

Replaced deprecated AntD List/List.Item in the approved four panels only: Assets (download queue and grouped assets), Profiles, Runtime, and Inventory. The replacement uses semantic ul/li with existing border/divider styles and responsive flex action rows. Existing callbacks, forms, profile/runtime logic, warning text, and empty/Card loading branches are retained. Download refresh retains the real AntD Spin wrapper and exposes aria-busy while retaining existing rows.

The unrelated grammar modal is untouched. No shared component abstraction was added. No runtime, model, browser, staging, or commit action was performed. Parent owns task/tracker updates and fresh native acceptance.

## Evidence

- RED before production edits: 5 failures and 2 passing controls. Each failure contains the exact real AntD List deprecation. The mounted regression spies console.error before the existing global test filter, so the warning cannot be hidden by vitest.setup.ts. No AntD module mock.
- Final GREEN: 30/30 tests across 5 suites, including all four existing panel suites plus the new 8-case mounted regression suite. Tests cover real profile create/edit/duplicate/delete dialogs, model/projector rendering and grouping, runtime action states/capabilities, inventory selection and failed registration, local asset registration/import/download cancellation, empty/card-loading controls, and download refresh state.
- ESLint: 5 actual files processed, 0 errors, 0 warnings. Initial frontend-working-directory command ignored files outside its base path; it was discarded and rerun from the repository root with the existing frontend config. The corrected invocation emits an existing next/pages configuration notice on stderr, while the file-level JSON contains no diagnostics.
- git diff --check: exit 0.
- New test was formatted with the existing frontend Prettier configuration.
- Bandit not applicable: all changed production/test files are TSX; no Python, shell execution, configuration, dependency, or security-sensitive logic changed. No claim of a Python security scan.

## Reproduction commands

From apps/packages/ui:

    ./node_modules/.bin/vitest run src/components/Option/Admin/__tests__/LlamacppPanels.list-rendering.test.tsx --reporter verbose

Final scoped run (same directory):

    ./node_modules/.bin/vitest run src/components/Option/Admin/__tests__/LlamacppPanels.list-rendering.test.tsx src/components/Option/Admin/__tests__/LlamacppAssetsPanel.test.tsx src/components/Option/Admin/__tests__/LlamacppInventoryPanel.test.tsx src/components/Option/Admin/__tests__/LlamacppProfilesPanel.test.tsx src/components/Option/Admin/__tests__/LlamacppRuntimePanel.test.tsx --reporter verbose

From repo root:

    apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Option/Admin/LlamacppAssetsPanel.tsx apps/packages/ui/src/components/Option/Admin/LlamacppProfilesPanel.tsx apps/packages/ui/src/components/Option/Admin/LlamacppRuntimePanel.tsx apps/packages/ui/src/components/Option/Admin/LlamacppInventoryPanel.tsx apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppPanels.list-rendering.test.tsx --format json --output-file .tmp/uat158-llamacpp-lists-20260916/eslint.json

## Frozen file hashes

- apps/packages/ui/src/components/Option/Admin/LlamacppAssetsPanel.tsx — 1c93e2b8b006da97690721c40232ae1f5defdeb19b05213d00aa9cdf07507560
- apps/packages/ui/src/components/Option/Admin/LlamacppProfilesPanel.tsx — b16a3370c0b46ddd134e2eaa981bfe09a2a2f9366f52652bbe034d461aced9ea
- apps/packages/ui/src/components/Option/Admin/LlamacppRuntimePanel.tsx — f979b5de65faa20bb5418869ebe8e88091be8d1c8969d2d56b1998d04ba23808
- apps/packages/ui/src/components/Option/Admin/LlamacppInventoryPanel.tsx — 1fb6dc59517463a4d9e3bdf0ce6b18d26f135d639d889adf6b126a4fa27a95f2
- apps/packages/ui/src/components/Option/Admin/__tests__/LlamacppPanels.list-rendering.test.tsx — 27e2d72fa9d5dd055598060986200a8dcdbf2e364fed7340427f182eeef4149f

Base HEAD at test preparation: 82d61258f3a026c24418f3616a059b28d6d7d28e. Final HEAD observed: 38fd4948216d36043ad9ea1191cbb8fff614642f. Concurrent unrelated work exists; no claims about its contents. See manifest.json for original source snapshot and artifact hashes.

## Limits and handoff

No full application build or global TypeScript check was run for this rendering-only scope. JSDOM proves mounted component behavior and console output, not pixel layout. Fresh /admin/llamacpp native acceptance must confirm desktop/narrow action layout and no List console error while visiting populated Assets/downloads, Profiles, Runtime, and Inventory paths. Root is responsible for that acceptance. This repair does not alter or certify UAT118 image-chat persistence or UAT103 history behavior.
