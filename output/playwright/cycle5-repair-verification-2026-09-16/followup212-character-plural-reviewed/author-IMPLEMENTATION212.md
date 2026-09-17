# UAT212 — Character result announcement

## Result and scope

Ready for independent review. Two production strings now use the existing ICU pattern: `{count, plural, one {# character found} other {# characters found}}`. The active English settings resource and the component fallback agree. The count, success condition, polite/atomic status semantics, filters, requests and table/gallery rendering are unchanged. No browser/runtime/config/task/git actions were performed.

Owned paths are CharacterListContent.tsx, assets/locale/en/settings.json and the existing CharacterListContent.design-system.test.tsx; exact SHA256, baseline and byte-identical review snapshots are in owned-manifest.json. owned.patch includes all and only this unit's changes against captured baseline bytes.

## Causal verification

The mounted component uses real AntD, real i18next and production ICUWithInterpolation, with the actual English resource. Tests cover zero/one/two results with both the resource and missing-key fallback, a two→one→zero count transition on the same translator, and silence during pending/error. The existing real Retry action test remains unchanged.

- **RED:** 3 failed, 7 passed. Actual received output was `1 characters found`, where resource, fallback and count-transition assertions required `1 character found`. Test source was complete before either production edit. Retained red.log.
- **GREEN:** 18 passed across 3 files, zero skips (2.00s): mounted Characters10, ICU formatting5, locale-source3. Retained green.log. Node emits its existing experimental localStorage warning; no assertion was silenced.
- **Scoped ESLint:** baseline/current zero errors and identical 44 warnings (41 existing any, 2 unused vars, 1 existing img warning). The test file has no new warning. Initial frontend-working-directory lint attempts ignored shared files; those non-evidence receipts remain, and corrected `*-scoped.json` outputs are the actual validation.
- **Compiler:** full frontend `tsc --noEmit --incremental false --pretty false` has 90 baseline and 90 current diagnostics, byte-identical output; none in CharacterListContent or its test. This does not claim a clean repository build.
- **Bandit limitation:** attempted touched TSX/JSON scope using the venv. It cannot parse the two TSX files and reports the same two B105 hits on existing JSON labels “Password” in baseline/current. These are unchanged translation-label findings, not introduced credentials. Bandit provides no meaningful TSX security analysis; this repair changes only two presentation strings and adds tests.

## Reproduce focused checks

From apps/tldw-frontend:

```sh
bunx vitest run ../packages/ui/src/components/Option/Characters/__tests__/CharacterListContent.design-system.test.tsx ../packages/ui/src/i18n/__tests__/icu-format.test.ts ../packages/ui/src/i18n/__tests__/sources-locale.test.ts
bunx tsc --noEmit --incremental false --pretty false
```

Actual scoped lint from repository root (explicit config establishes the correct base):

```sh
apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Option/Characters/CharacterListContent.tsx apps/packages/ui/src/components/Option/Characters/__tests__/CharacterListContent.design-system.test.tsx -f json
```

## Locale pipeline and native boundary

WebUI lib/i18n-web.ts imports assets/locale/en/settings.json; shared i18n/index.ts dynamically loads the same assets tree. public/_locales/en/settings.json is generated Chrome metadata, produced by extension/scripts/sync-public-locales.js. Its CLI rewrites all locale settings files and it is not the active i18next namespace source. Neither that generator nor generated duplicates were edited; a future normal regeneration derives the updated key from the asset.

The original one-result native snapshot is alice-reviewed-characters.txt in .tmp/uat198-181-native-20260917. Parent owns independent review, native one-result reacceptance, task finalization and commit. This packet does not claim native acceptance after the repair.
