# UAT212 / TASK13260.150 — bounded approved design

Use the production ICU plural message `{count, plural, one {# character found} other {# characters found}}` in the active English settings resource and matching CharacterListContent fallback. Preserve the supplied totalCharacters, success-only live-region condition, filters, requests and table/gallery behavior.

Mounted tests use real CharacterListContent, AntD, i18next and the production ICUWithInterpolation plugin with actual English settings. Check zero/one/two with resource present and missing-key fallback, count transition on one translator instance, and pending/error silence. The regression catches the actual plural-only announcement, not source text.

The WebUI i18n-web.ts imports assets/locale/en/settings.json; shared i18n/index.ts also loads assets/locale. public/_locales is a generated Chrome-message catalogue produced by extension/scripts/sync-public-locales.js, whose CLI rewrites all locale settings files. It is not the mounted i18next source. Do not run that broad generator or hand-edit generated duplicates for this bounded WebUI fix.

Validation: causal RED before production, GREEN plus ICU and Character list adjacent tests; baseline/current scoped lint and compiler diagnostic comparison; Bandit attempted with its TSX/JSON parser limitation explicit. Parent owns task/native acceptance/git. No runtime/browser action.
