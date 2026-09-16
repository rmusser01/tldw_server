# Existing Settings fixture failures — read-only evidence for UAT154

No edits to these files:

1. `apps/packages/ui/src/components/Option/Settings/__tests__/tldw-auth-mode.form.test.tsx`.
2. `apps/packages/ui/src/components/Option/Settings/__tests__/tldw.form-lifecycle.test.tsx`.
3. `apps/packages/ui/src/components/Option/Settings/__tests__/tldw.cookie-logout.test.tsx`.

Current expanded run28fail/32pass (includes4other passing suites); baseline-only3files28fail/8pass. All28 failure headings exactly match unchanged1ea5402c83 production loaded through `/private/tmp/cycle5-uat152-baseline.config.mts`. `/private/tmp/cycle5-uat152-baseline-comparison.json` retains the exact headings, `/private/tmp/cycle5-uat152-existing-baseline.log` the details.

## Concrete storage mismatch

- Real `useSettingsLoginStatus.ts:20` creates `createSafeStorage({area:'local'})` and reads the effective session from this storage, independently of mocked `tldwClient.getConfig`.
- Auth-mode/form-lifecycle fixtures create default `new Storage({area:'local'})` and write synthetic auth data; cookie fixtures also write directly to localStorage or create the same defaultStorage.
- Installed Plasmo1.15 BaseStorage constructor creates its extension primaryClient only if browser/chrome.storage exists. It creates localStorage secondaryClient only when `allCopied` is true or a copiedKeyList is provided. Default options have neither.
- Shared Vitest setup supplies localStorage but not an extension storage adapter. Therefore real hook reads do not observe fixture auth writes: initial Logged In is absent, billing remains gated and later auth transition tests fail.

This exact no-storage behavior was independently encountered by the new152 actual-client test; enabling Plasmo's existing allCopied fallback made the real config save/load tests operate. That is evidence of the initial-read mismatch, not proof every one of28 tests is fixed by the same one-line fallback.

## Preserve scenario semantics in a bounded fixture fix

The existing tests cover same-tab/cross-tab/config-event updates, owner/server A→B→A races, invalidation/rotation, offline credentials, StrictMode cleanup and extension-style boolean watch results. A bare allCopied fallback may restore reads but **does not implement extension onChanged subscriptions**, so it is insufficient to claim these scenarios repaired. Supply a coherent test-only browser.storage/localStorage adapter that actual Plasmo get/set/watch uses, or an existing shared test helper if available. Keep every assertion and actual owner-transition timing; do not substitute a constant logged-in hook or bypass authentication derivation.

Source/test files remain unchanged pending a separate tracked fixture unit. No rerun requested here; root owns task creation/disposition.
