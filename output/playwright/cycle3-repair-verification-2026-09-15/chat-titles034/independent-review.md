# TASK13260.34 final independent re-review — clear

**No remaining confirmed finding in the bounded title repair.** Reviewed corrected production freeze 2026-09-15T21:42:49.051Z; all six hashes match the updated `/private/tmp/uat034-production-freeze.json`. The correction changes Header and the active-title hook only.

Header now retains the newer editor/text when the prior rename is busy, displays its saving status, and permits an explicit commit after settlement. The hook refreshes the saving state at start and completion while keeping owner/lifetime guards. The permanent actual Header control confirms that the retained second draft uses version 7 returned by the first successful save. The busy state does not silently treat the second edit as saved.

## Fresh verification

- Original unchanged `/private/tmp/uat034-independent-overlap.config.ts`: **1 passed**, log `/private/tmp/uat034-rereview-overlap.log`.
- Actual Header/Next Head integration and Node SSR: **16 passed /2 suites**, log `/private/tmp/uat034-rereview-title.log`.
- Supplemental independent first-save failure and principal-invalidation controls: **2 passed**, config `/private/tmp/uat034-independent-overlap-boundaries.config.ts`, log `/private/tmp/uat034-independent-overlap-boundaries.log`. The failure control retains the newer draft and commits it with unchanged version 1; principal invalidation masks the draft and discards the old acknowledgement/local write.
- All runs exited 0 from `apps/tldw-frontend`, using its existing Vitest binary and `--maxWorkers=1 --no-file-parallelism`.

The prior unchanged scoped transport/policy independent checkpoint remains **32 passed /2 suites**; those production files did not change in the correction. No unrelated suite rerun was needed. Production/tests, browser/runtime, commits and global docs were untouched in this review. Native title/browser acceptance remains parent-owned.

---

## Historical initial review

# TASK13260.34 independent review — changes requested

Reviewed the six production files frozen at 2026-09-15T21:37:39.260Z. All six match `/private/tmp/uat034-production-freeze.json`. No production/test files, runtime, browser, commits or global documents were changed. The independent probe/report are in `/private/tmp`.

## P2 — Do not silently discard an overlapping rename

Location: `apps/packages/ui/src/hooks/useActiveChatTitle.ts:94`, together with `apps/packages/ui/src/components/Layouts/Header.tsx:123–128`.

While a saved rename is awaiting its response, the actual Header still offers the title button. The user can reopen the editor, enter a second title and press Enter. Header immediately closes the editor and clears failure state. The hook then returns early because `owner.saving` is true, with no accepted/busy result for Header. Once the first request finishes, only its title is shown and persisted; the second entered text is neither retained nor saved.

The independent actual Header interaction holds the first PUT, enters and commits “My second title”, then resolves the first request. It proves the second title is absent from both the persistent local mirror and the edit input. The test permits a bounded fix that prevents opening an editor while busy, retains the second draft, or saves the latest accepted intent. No broader rename queue/framework is required.

Frozen config: `/private/tmp/uat034-independent-overlap.config.ts`. Fresh log: `/private/tmp/uat034-independent-overlap.log` — **1 failed /14 original tests filtered**. Preserve the config unchanged for re-review.

## Fresh independent passing evidence

- Actual Header/title hook/Next Head integration plus Node SSR: **15 passed /2 suites**, exit 0. Log `/private/tmp/uat034-independent-title.log`.
- Shared scoped update transport and exact path policy: **32 passed /2 suites**, exit 0. Log `/private/tmp/uat034-independent-transport.log`.

Commands use the existing workspace Vitest binaries, `--maxWorkers=1 --no-file-parallelism`. Web tests run from `apps/tldw-frontend`; shared transport/policy tests run from `apps/packages/ui`.

## Remaining reviewed scope

No other confirmed issue in the bounded diff:

- Both browser and header titles follow canonical server metadata or the reactive local-title adapter.
- The real Next Head manager updates on hydration/rerender/remount, and the page wrapper imports/renders under Node SSR without browser globals.
- Scoped rename forwards captured target/expected owner, cancellation signal and version through metadata/version reads and PUT, including the existing conflict retry path; exact PUT route checks reject expanded/malformed paths.
- Replacement selection, explicit principal changes and unmount retire the owner. Title drafts/failure UI are owner-associated; same-owner rotation remains permitted through the canonical authority contract.
- Failed server save leaves the previous saved title and offers retry. Local-only read/rename performs no server rename or snapshot request, retaining the existing configured/offline behavior.

The reactive Dexie adapter and network/storage boundaries are controlled in the integration fixture. Native IndexedDB/browser acceptance remains parent-owned. This read-only review did not rerun the full unrelated client suite, lint or TypeScript checks, and makes no clean-typecheck claim.
