# Independent review — Cycle4 Task9 / UAT058 and UAT067

## Result

No actionable findings in the frozen eight-path patch. The bounded changes are ready for parent-owned combined verification and targeted native acceptance.

## Scope

Reviewed `/private/tmp/cycle4-task9-implementation.md`, the eight-path manifest, and diff against its base `15c727aa1d605361653b0b4eca1314531bbf1ac2`. Production scope was exactly Character Manager, shared update-page-title helper, and the Next Prompts/Characters wrappers. Inspected the real CharacterDialogs Drawer/reset paths, lazy import/preload/outer Suspense boundary, existing Chat Head ownership, and all title-helper callers to assess compatibility. No overlapping Chat Task5 production edits were reviewed or modified.

All eight SHA256 hashes matched before and after review. Scoped `git diff --check` passed. No repository edits, browser/runtime operations, inference, subagents, or commits occurred.

## Assessment

- Next route titles remain owned by Head. The imperative helper's Next-document guard matches an existing shared-layout platform distinction. Chat already supplies its title through useActiveChatTitle and Head; the change therefore stops delayed callbacks from replacing another route's title without suppressing the reactive Chat title path. Extension documents retain imperative updates, and missing-document execution remains safe.
- Prompts/Characters wrappers retain browser-only dynamic route loading while adding page-owned metadata. Tests use their actual wrappers and Next's real Head manager.
- Removing the nested Form Suspense allows the existing outer dialog boundary to withhold the interactive Drawer until the lazy Form renders. The reset handlers, form instances, submit mutations, initial values, and cancellation behavior are unchanged.
- The cold-form test gates the actual module import and confirms the load was attempted before checking that neither Form nor Close is available. It releases the actual editor, cancels, reopens, and checks the AntD warning. The warm control submits through the real form to the mocked create API and confirms the Drawer closes.

## Independent verification

From `apps/tldw-frontend`:

```sh
./node_modules/.bin/vitest run \
  __tests__/pages/core-route-titles.test.tsx \
  __tests__/pages/chat-title-late-persistence.test.tsx \
  __tests__/pages/chat-title.integration.test.tsx \
  __tests__/pages/chat-title.ssr.test.tsx \
  ../packages/ui/src/utils/__tests__/update-page-title.test.ts \
  --maxWorkers=1 --no-file-parallelism
```

**33 tests passed in 5 files**, exit 0, 3.71 seconds. Covers actual route wrappers/Head, delayed real saveMessageOnSuccess completion after Settings navigation, stale Chat title callback, active Chat history/account title behavior, extension behavior, and SSR.

```sh
./node_modules/.bin/vitest run \
  ../packages/ui/src/components/Option/Characters/__tests__/Manager.first-use.test.tsx \
  -t 'waits for the real form|opens Create without writing|keeps real create, cancel' \
  --maxWorkers=1 --no-file-parallelism
```

**3 tests passed, 96 intentionally skipped**, exit 0, 12.06 seconds. Cold lazy boundary, disconnected Edit-form control, and warm cancel/reopen/create are all green. The cold test executes first in this fresh process.

## Limits

Did not duplicate the author's full 99-test Manager run or the combined eight-suite run, and did not run a compiler or lint again. Existing Node experimental localStorage warnings occurred; focused tests reported no failures. This verifies the reachable early-close warning mechanism with real AntD/form components and mocked services. It does not establish native warning absence under every navigation/network timing; parent retains that live acceptance. Cold dialog loading still uses the pre-existing null fallback, so the review makes no new loading-feedback claim.
