# UAT170 / TASK-13260.107 — narrow default selectors

## Diagnosis and approved bounded design

The native390px screenshot `.tmp/uat165-repair-20260916/native-narrow.png` shows both AntD default selectors extending beyond their card, with arrows clipped. Their shared grid has no base column template (`grid gap-3 sm:grid-cols-2`), leaving the narrow track's automatic minimum influenced by the long model label. Both Selects already use w-full. Desktop uses explicit two-column fractional tracks.

Add only `grid-cols-1` to the defaults grid, establishing a minmax(0,1fr) track below sm and preserving existing two columns at sm. Do not alter labels, selection state, callbacks, options, AntD dropdowns, readiness tiles or accepted165 break-all. Existing explicit single-column responsive grids in AgentRegistry, Skills and Watchlists provide the pattern.

## Verification stages

1. Native failure: retained pre-edit screenshot and source hash; no new browser/runtime work by author.
2. Minimal source change: one class in Models/index.tsx. Existing Models regressions and scoped lint/diff checks.
3. Freeze exact source/report for independent review and root-owned real390px/desktop bounds plus actual selection. No implementation-mirroring class test is added; JSDOM does not prove grid geometry.

Task already exists In Progress. Root owns task/tracker/git and native acceptance. Only Models/index.tsx and this private packet are owned by this subtask.

## Frozen author result

One production class added; accepted165 wrapping unchanged. Existing Models21 tests/3 files PASS; scoped ESLint0errors/0warnings; git diff --check exit0. Bandit exits0 with0findings and1unsupported TSX parse error, which is not TSX security assurance. No native/browser action performed. Actual layout/selection acceptance remains pending with root, as does independent source review.

Source SHA256: `21142d835bbc9cd687ce7fe6b70cf68f3debf337581052866ef3d0013f36fcdc`. Baseline matches HEAD `15af5a6693b3c2585c1ed49e160b0bdff3109af0`.

Commands:

```sh
# apps/tldw-frontend
bunx vitest run ../packages/ui/src/components/Option/Models/__tests__
# repository root
node apps/tldw-frontend/node_modules/eslint/bin/eslint.js --config apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Option/Models/index.tsx -f json
source .venv/bin/activate && python -m bandit apps/packages/ui/src/components/Option/Models/index.tsx -f json -o .tmp/uat170-repair-20260916/bandit.json
git diff --check -- apps/packages/ui/src/components/Option/Models/index.tsx
```

Native acceptance should measure both visible Select and arrow bounds inside the defaults card at390px and desktop, open each control using ordinary input, verify option selection works, and restore/preserve the original provider/model defaults. The screenshot proves baseline clipping; passing component tests alone does not prove CSS containment.
