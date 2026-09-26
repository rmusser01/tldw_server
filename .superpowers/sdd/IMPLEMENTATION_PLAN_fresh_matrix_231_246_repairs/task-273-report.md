# TASK13260.214 / UAT273 report

## Result

The Visual Identity panel now reads capabilities before deciding whether active-pack metadata is available. When the backend explicitly reports `metadata_supported: false`, it no longer requests active packs, shows clear authoring availability guidance, and disables ZIP import. Expression slots and ordinary binding resolution still load.

## Cause and scope

The safe native PostgreSQL observation showed `/capabilities` returning `metadata_supported: false` while the panel issued its active-packs request in the same `Promise.all`, producing the known unsupported-authoring `501`. The fix changes only the panel and its focused test. It does not alter backend support, storage, shared client code, or resolver behavior.

The panel treats only an explicit `false` as unsupported. A legacy response with no `metadata_supported` field remains authoring-capable after the capabilities request completes; the existing import and draft tests cover that behavior.

## Verification

- Causal red: an isolated HEAD source overlay called active packs once with `status: active` despite `metadata_supported: false`; the retained receipt proves the pre-fix request.
- Green: `VisualIdentityPackPanel.test.tsx` passed all 4 tests. The new tests verify no packs call, continued slots/resolver calls, availability guidance, disabled ZIP controls, and the explicit supported capability path.
- `git diff --check` passed.
- Scoped ESLint passed for both changed files; it emitted the existing frontend pages-directory configuration notice but no scoped lint violation.
- Frontend `bun run typecheck` exits with the known 90 diagnostics and has no changed-panel/test path reference.

Bandit was run for the touched source as required, but it cannot parse TypeScript and is not a TypeScript security result. Prettier reported existing whole-file warnings; files were not rewritten to avoid unrelated formatting churn. Full evidence and hashes are in [verification.json](../../../.tmp/uat-repairs-231-246/visualidentity273/verification.json).

## Tests-only type correction

Independent review found that the capability mock inferred a return shape without the optional metadata fields. The two new `mockResolvedValue` controls therefore produced `TS2353` under an exact AST projection, even though the app typecheck does not include this package test. The test now annotates the async mock return as `VisualIdentityCapabilitiesResponse`. The original reviewer diagnostic remains retained; the copied projection with the maintained type import reports zero diagnostics. No production source changed during this correction.

## Changed files

- `apps/packages/ui/src/components/Common/VisualIdentity/VisualIdentityPackPanel.tsx`
- `apps/packages/ui/src/components/Common/VisualIdentity/__tests__/VisualIdentityPackPanel.test.tsx`
