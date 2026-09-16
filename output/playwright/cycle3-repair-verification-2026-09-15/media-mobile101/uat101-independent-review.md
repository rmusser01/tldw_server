# TASK13260.42 / UAT101 independent bounded review

## Disposition

**CLEAR for the frozen three-file CSS-only scope; no actionable finding confirmed.** Reviewed freeze2026-09-16T00:15:52.146Z. All three production SHA256 values matched independently before/after. Both .40 production reading-progress hooks and its existing hook test remain byte-identical to the .40 seven-file freeze.

No source/test/browser/runtime/global-document or commit changes were made. Private verification records and this report are the only writes by the reviewer. Native mobile/desktop acceptance and combined compiler remain parent-owned.

## Scope and reasoning

- `ViewMediaPage.tsx`: the mobile library uses the existing open/collapsed preference and covers the reader within the bounded relative page. Its width reserves the24px toggle strip; the toggle moves to that edge when open and remains at the left when closed. Its z20 is above the library z10, and the reader reserves the24px strip. The library still scrolls independently. At the existing md breakpoint, relative positioning, original sidebar widths and automatic toggle position restore the side-by-side arrangement. No selection, preference, navigation or close behavior was changed.
- The main owner, nested navigation/content row and viewer wrapper now permit width shrink. This addresses the intrinsic width propagation seen in the original739px/390px evidence without adding another scroll listener or changing the .40 vertical scroll owner.
- `MediaSectionNavigator.tsx`: mobile width is bounded while original desktop width/minimum remains. Existing tree buttons, full title strings, quick-jump Enter, selected-node reveal and expanded-child controls are untouched.
- `ContentViewer.tsx`: the root and title can shrink; title retains the full Tooltip/text. Header actions, Content/Analysis controls and Find controls wrap. Find input can shrink and take a new row while its existing match/count/previous/next/close controls remain. Existing Content/Analysis toggle, text-size, copy/edit and action handlers are preserved.
- Independent TypeScript AST comparison after removing className attributes yields identical syntax in all three files versus HEAD. This verifies that the apparent CSS-only diff has no concealed handler/state/prop changes. It does not prove browser geometry.

## Fresh verification

- **56 tests /7 suites PASS**, one focused run. `/private/tmp/uat101-independent-focused-tests.log`. Includes real navigator mouse/Enter/expanded/reselected-tree behavior; Find search/next/previous/escape; Content/Analysis rendering, copy/edit and keyboard/accessibility controls; actual page-owner passive versus explicit chapter behavior; and .40 save/restore/readiness/cleanup regressions. Coverage overlaps the implementer's85/16 and must not be added to it.
- Three frozen source hashes match before and after. Two .40 hooks plus its hook test match the original .40 frozen hashes. `/private/tmp/uat101-independent-pre-hashes.json`, `/private/tmp/uat101-independent-post-hashes.json`.
- Independently inspected the actual scoped root ESLint JSON and recomputed diagnostic signature comparison: all3 files covered,0errors,23unchanged warnings,0added/removed. `/private/tmp/uat101-independent-static-review.json`; original inputs `/private/tmp/uat101-eslint.json` and `uat101-eslint-baseline.json`. No second lint run was needed. Root pages-directory advisory remains.
- Scoped `git diff --check` is clean. No duplicate compiler or Bandit run; TSX class strings do not have a Python Bandit scope.

Command, working directory `apps/packages/ui`:

```sh
./node_modules/.bin/vitest run src/hooks/__tests__/useMediaReadingProgress.test.tsx src/components/Review/__tests__/ViewMediaPage.permalink.test.tsx src/components/Media/__tests__/MediaSectionNavigator.test.tsx src/components/Media/__tests__/ContentViewer.stage1.test.tsx src/components/Media/__tests__/ContentViewer.stage4.accessibility.test.tsx src/components/Media/__tests__/ContentViewer.stage10.findBar.test.tsx src/components/Media/__tests__/ContentViewer.analysis-markdown.test.tsx --maxWorkers=1 --no-file-parallelism
```

## Native acceptance boundary

The original actual390px screenshot/bounds prove clipping. The new CSS arrangement is consistent with the approved design and existing layout patterns, but jsdom does not establish hit boxes or computed flex layout. Parent must verify actual mobile library open/close through its existing toggle; chapter navigation and full source width; Actions, Find and text-size controls; Content/Analysis wrapping; and desktop scrolling/save/reload. The existing navigator test that checks responsive class names is not counted as geometry evidence. No new class-mirroring test was added, and no full-fresh UAT claim is made.
