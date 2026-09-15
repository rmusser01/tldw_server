# TASK13260.38 / UAT097: Chat sidebar and composer layout

## Frozen scope

Production frozen 2026-09-15T22:40:20.266Z. No browser, runtime, dependency, handler, identity, global-document or commit changes. Independent review and native acceptance are pending.

Production paths:

- `apps/packages/ui/src/components/Common/ChatSidebar.tsx`
- `apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx`

Existing tests changed:

- `apps/packages/ui/src/components/Common/ChatSidebar/__tests__/ChatSidebar.tools-first.test.tsx`
- `apps/packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.test.tsx`

Associated task: `backlog/tasks/task-13260.38 - Keep-expanded-Chat-navigation-and-composer-usable-at-desktop-size.md`.

## Cause and repair

The retained native screenshot `/private/tmp/uat097-sidebar-chat-obscured.png` and geometry `/private/tmp/uat093-native-sidebar-robot-trace.json` show the recent-chat panel and its scroller at height0. The expanded shortcut block consumes the fixed-height sidebar; search/tabs/results overflow the collapsed panel and extend behind the footer. The parent observed mouse interception; the actual keyboard handler works. These are layout findings, separate from the concurrent Chat restoration repair.

The sidebar now has one `min-h-0 flex-1 overflow-y-auto` middle region containing every shortcut, recent toggle, search, tabs and results. Header/footer remain outside it with `shrink-0`; the root clips overflowing content. The competing nested recent flex/scroll sizing is removed. This deliberately scrolls the middle region as one unit instead of adding fixed shortcut heights or hiding navigation. Most of the apparent sidebar diff is indentation inside the new wrapper.

The casual composer forced `lg:flex-nowrap` using viewport width even when sidebar and Runtime rail reduced its available center width. The action row and its two labeled groups now wrap at all viewport widths, and direct group children do not shrink. Existing control ordering/actions, pro layout and mobile overflow remain intact. No media-agent-owned handlers or inline RolePlaySetup behavior were touched.

## RED and GREEN evidence

The initial focused RED was2failed/43passed (`uat097-layout-red.log`). One intermediate GREEN attempt found a test-fixture mismatch: enumerating all configurable shortcuts included the hosted-only account shortcut. The final fixture uses the13 actual shortcuts in the native reproduction; no product behavior was changed to satisfy that fixture. The existing dense-row expectation was replaced with the approved wrapping contract, preserving desktop/mobile action tests.

The final permanent tests were replayed unchanged against exact original production sources via `/private/tmp/uat097-baseline.config.ts`: **2failed/43passed**, precisely the absent shared scroller and desktop nowrap. Sources `/private/tmp/uat097-before-ChatSidebar.tsx` and `/private/tmp/uat097-before-ComposerToolbar.tsx` are loaded by a Vitest plugin without altering the checkout. Log: `/private/tmp/uat097-final-original-replay-red.log`.

Final focused/adjacent run: **56 passed in6 suites**. The45 tests in the two changed suites are included in56; counts are not additive. Log: `/private/tmp/uat097-layout-final-tests.log`.

From `apps/tldw-frontend`:

```sh
./node_modules/.bin/vitest run ../packages/ui/src/components/Common/ChatSidebar/__tests__/ChatSidebar.tools-first.test.tsx ../packages/ui/src/components/Common/ChatSidebar/__tests__/ChatSidebar.lazy-history.test.tsx ../packages/ui/src/components/Common/__tests__/ChatSidebar.coordinator.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.role-play-mobile.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.layout.guard.test.ts --maxWorkers=1 --no-file-parallelism
```

Original replay (expected RED):

```sh
./node_modules/.bin/vitest run --config /private/tmp/uat097-baseline.config.ts --maxWorkers=1 --no-file-parallelism
```

## Static checks

Root-scoped ESLint on all4 code/test paths: **0 errors,4 unchanged warning signatures,0 added**. Sidebar and its test have0warnings; ComposerToolbar has3 and its existing test1. Evidence: `/private/tmp/uat097-eslint-{baseline,current,comparison}.json`.

```sh
apps/tldw-frontend/node_modules/.bin/eslint --config apps/tldw-frontend/eslint.config.mjs apps/packages/ui/src/components/Common/ChatSidebar.tsx apps/packages/ui/src/components/Option/Playground/ComposerToolbar.tsx apps/packages/ui/src/components/Common/ChatSidebar/__tests__/ChatSidebar.tools-first.test.tsx apps/packages/ui/src/components/Option/Playground/__tests__/ComposerToolbar.test.tsx -f json
```

Scoped `git diff --check` passed. Prettier check reports existing formatting drift in both original and current production files; no whole-file reformat was applied. No new functions or Python changes; Bandit is not applicable. No compiler run was performed during concurrent .33 source changes; parent owns the combined compiler check. The Vitest runner emits the existing Node localStorage experimental warning.

## Review and native limits

The jsdom regressions assert actual rendered layout containers/classes and retain the existing action/collapse/mobile controls; jsdom does not compute flex geometry. They cannot certify nonzero rendered height, overlap, mouse hit testing or resize behavior.

Parent must verify the frozen production hash set at1280x720 with all13 shortcuts and Recent expanded, many saved chats, and Runtime rail open: scroll the middle area, click the actual saved row with mouse, confirm footer does not intercept it, and inspect non-overlapping toolbar bounds. Repeat at a smaller supported viewport/mobile fallback and check keyboard navigation. Ordinary Chat identity/restoration acceptance belongs to the separate .33 repair. Native acceptance and independent review remain pending in task criteria.

Hashes: `/private/tmp/uat097-production-freeze.json` binds the two production files; `/private/tmp/uat097-frozen-manifest.json` binds all5 owned paths and the original-source probe inputs. `/private/tmp/uat097-evidence-SHA256SUMS` binds retained report/static/test evidence.
