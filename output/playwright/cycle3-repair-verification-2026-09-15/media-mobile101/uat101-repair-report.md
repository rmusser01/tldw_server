# UAT101 / TASK13260.42 — Mobile Media width repair

Status: implementation frozen and regression checks passed; independent review and native mobile/desktop acceptance pending. No browser/runtime work by this implementer. No claim that the CSS geometry is proven by jsdom.

## Native RED and cause

Root's actual single-user Media1 at390x844: content x18/width739, Actions x431, Find x704, document width390. `/private/tmp/uat099-final-native-mobile-bounds.txt`, `-snapshot.txt`, and `.png` retain this failure; PNG visually inspected. Separately, .40 desktop actual inner wheel/save/reload passed before this change.

Intrinsic width propagates through ViewMediaPage main, navigation/content row and viewer wrapper without min-w-0; navigator full-width mobile layout and nowrap labels cannot constrain those ancestors. Content/Analysis actions are unwrapped rows (including132px display selector, S/M/L, Edit and Find). The root overflow-hidden hides the oversized content. The library is also declared full-width on mobile while sitting beside the viewer in the same horizontal flex row; in RED it is effectively squeezed away despite its expanded preference.

Compared established patterns: DocumentWorkspacePage main at1179 uses min-w-0; SharedWorkspaceHeader at43/53 bounds shrinkable text; ResearchWorkspace WorkspaceHeader at2316/2322 wraps bounded controls. Parent approved a mobile overlay using the existing sidebar state/toggle, without a new navigation or selection model.

## Exact three-file change

- `apps/packages/ui/src/components/Review/ViewMediaPage.tsx`: mobile library positioned over the viewer, current toggle kept at its edge; collapsed toggle on left; desktop relative side-by-side retained. Main and nested pane widths can shrink. Same preference, handlers and selection behavior.
- `apps/packages/ui/src/components/Media/MediaSectionNavigator.tsx`: bound mobile width while retaining desktop width/min-width and current tree/keyboard behavior.
- `apps/packages/ui/src/components/Media/ContentViewer.tsx`: bound viewer/title width; allow action groups and Find input/controls to wrap.

No progress hooks, request payloads, data or tests changed. Both .40 production hook hashes and its hook test are unchanged against the .40 frozen manifest. No source rendering mode, inference, new sources or deletion.

## Verification

- Existing85tests/16suites PASS, `/private/tmp/uat101-media-regressions.log` (24.98s).
- Scoped root ESLint0errors/23pre-existing warnings,0added/removed: `/private/tmp/uat101-eslint-comparison.json`. Repro script `/private/tmp/uat101-lint.cjs`; current and HEAD-baseline JSON retained.
- `git diff --check` scoped to the three files: clean.
- Bandit not applicable: TSX class strings only, no Python changes.
- No compiler run by this agent for this CSS-only change; parent owns combined baseline comparison.
- Native geometry still requires parent check at390x844 and1280x720: library open/close reachable, source/Actions/Find/text-size controls in view, chapters usable, and actual desktop save/reload unchanged.

Frozen at2026-09-16T00:15:52.146Z: `/private/tmp/uat101-production-freeze.json` has exact3files/SHA256/bytes. `/private/tmp/uat101-owned-files.json` lists scope; `/private/tmp/uat101-source.diff` retains diff. Original native RED stands; no class-mirroring tests added.

Command (cwd apps/packages/ui):

```sh
./node_modules/.bin/vitest run \
 src/hooks/__tests__/useMediaReadingProgress.test.tsx \
 src/components/Review/__tests__/useMediaSelection.reading-progress.test.tsx \
 src/components/Media/__tests__/ContentViewer.stage1.test.tsx \
 src/components/Media/__tests__/ContentViewer.stage2.test.tsx \
 src/components/Media/__tests__/ContentViewer.stage3.test.tsx \
 src/components/Media/__tests__/ContentViewer.stage4.accessibility.test.tsx \
 src/components/Media/__tests__/ContentViewer.stage10.findBar.test.tsx \
 src/components/Media/__tests__/ContentViewer.stage12.performance.test.tsx \
 src/components/Media/__tests__/ContentViewer.analysis-markdown.test.tsx \
 src/components/Media/__tests__/ContentViewer.stage14.reprocess.test.tsx \
 src/components/Media/__tests__/MediaSectionNavigator.test.tsx \
 src/components/Review/__tests__/ViewMediaPage.permalink.test.tsx \
 src/components/Review/__tests__/ViewMediaPage.stage13.error-handling.test.tsx \
 src/components/Review/__tests__/ViewMediaPage.stage14.bulk-actions.test.tsx \
 src/routes/__tests__/route-paths.viewport.test.ts \
 src/routes/__tests__/option-media-multi.connection-state.test.tsx \
 --maxWorkers=1 --no-file-parallelism
```
