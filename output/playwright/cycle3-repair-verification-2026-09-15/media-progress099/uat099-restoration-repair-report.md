# TASK13260.40 / UAT099 — preserve reading progress through reload

## Frozen status

Frozen2026-09-15T23:56:55.774Z: **five production paths, two test paths**, `/private/tmp/uat099-restoration-final-manifest.json`; production subset `uat099-restoration-production-freeze.json`. Final **75 tests/15 suites pass**, scoped **0lint errors/38 unchanged warnings/0added or removed**. No full clean-TypeScript claim; parent owns combined compiler with concurrent .41. Independent final review and corrected native reload/mobile checks remain pending. Browser remains paused on single `/media?id=1`; no runtime/restart/commit/global-document edits.

## Native evidence and causes

The original unbounded layout produced no inner scroll or progress mutation. Agent round2_chat's three-path layout correction is preserved. First reload served old classes, explicitly excluded from corrected-source acceptance. After parent's single-frontend rebuild, actual inner geometry was540high/835scrollHeight; document720high, no document overflow. Actual wheel160 moved inner160/document0 and PUT911 returned200 with `zoom_level:100`, `percentage:54.24`, `cfi:scroll:54.24`. This is a live percentage-unit save pass.

Normal reload then failed: GET1085 returned54.2/cfi54.24; without a new user scroll, PUT1103 sent0/cfi0 and returned200. The inner view and read badge became0. Safe request/response bodies and geometry are retained under `/private/tmp/uat099-single-*`. The source is original synthetic Aster Media1; source content remains unchanged in the preceding before/after checks.

Actual owner trace: ViewMediaPage asynchronously restores/highlights a remembered or default chapter and treated this passive selection as an explicit seek. It incremented navigationSelectionNonce and passed navigationTarget to the actual reading hook, which sought the first chapter and suppressed precise server reading restoration. A real page-owner/reading-hook regression holds that passive selection until after saved54.2 restoration: position160 becomes0 before the correction.

Separate permanent hook controls established additional data-loss boundaries before changes: detail hydration changing total pages while the progress GET is pending causes unconditional cleanup writes of0; ready content arriving after the initial empty geometry does not retry restoration; a late GET can replace a newer user scroll; resampling a pending scroll after loading resize saves0, and ref detachment can drop it entirely. These are reproduced behaviors, not speculative changes to the12-frame retry loop.

## Bounded correction

- Preserve the original layout3: exact `/media` viewport policy, bounded Media root, shrinkable ContentViewer with fixed toolbar.
- Reuse navigationSelectionNonce as explicit current-view chapter intent. Passive remembered/default selection remains highlighted and persisted but does not seek. An actual chapter action still increments it and supplies the original navigation target. Direct supplied ContentViewer targets and canonical Media/source route consumption remain unchanged.
- Forward existing `!isDetailLoading` as content readiness. Do not restore or listen against the temporary loading geometry. The existing12RAF policy is unchanged; no new observer or polling loop.
- Capture the existing scroll revision when starting restoration; a newer scroll prevents late GET/frame completion from replacing it.
- Cleanup flushes only a real pending scroll. Capture its payload at the scroll event, preserving the correct percentage through loading resize or element detachment. Explicit save remains available. Percentage zoom100 and current media-ID addressing remain unchanged.
- No auth, remote provider, source mutation, route/deep-link policy, global ownership framework or unrelated cleanup changes.

## Exact files

1. `apps/packages/ui/src/routes/route-paths.ts` — original layout registration, unchanged by restoration follow-up.
2. `apps/packages/ui/src/components/Review/ViewMediaPage.tsx` — original bounded root plus passive/explicit chapter distinction.
3. `apps/packages/ui/src/components/Media/ContentViewer.tsx` — original layout only, unchanged by follow-up.
4. `apps/packages/ui/src/components/Media/hooks/useReadingProgress.tsx` — forwards existing detail readiness.
5. `apps/packages/ui/src/hooks/useMediaReadingProgress.ts` — ready restore, newer-scroll guard, pending payload.
6. `apps/packages/ui/src/components/Review/__tests__/ViewMediaPage.permalink.test.tsx` — actual owner plus actual navigation/reading hooks at controlled DOM/storage boundary.
7. `apps/packages/ui/src/hooks/__tests__/useMediaReadingProgress.test.tsx` — deferred hydration/resize/user/lifetime controls.

Task record is updated only through CLI. Parent owns global design/tracker. No changes to route-paths/WebLayout/ChatHeader in this follow-up.

## RED → GREEN

- `uat099-passive-navigation-red.log`: actual owner/default chapter1failed, genuine scroll/explicit chapter1passed. The final regression additionally covers a remembered selection.
- `uat099-hydration-red.log`:2failures: no-input cleanup wrote0 twice; delayed hydration/size change stayed0 rather than restored500.
- `uat099-late-user-scroll-red.log`:1failure: held GET moved user400 to server700.
- `uat099-pending-resize-red.log`:1failure: pending40% was resampled as0 during loading resize.
- `uat099-pending-unmount-red.log`: unmount1failure/loading1pass at the intermediate snapshot correction; captured progress was lost after ref detachment.
- `uat099-restore-final-focused.log`:31/2 at intermediate snapshot; superseded by final broader run below (final adds ref-detach control).
- `uat099-restoration-broader-green.log`: final75/15. This includes existing restore/CFI, debounce/dedup, old/new ID flush, content/analysis/edit/copy, navigation/permalink/Flashcard source consumer, empty recovery, keyboard, large content and MediaMulti controls.
- `uat099-restoration-eslint{,-baseline,-comparison}.json`:7paths,0errors,38warnings,0added/removed vsHEAD; `uat099-restoration-lint.cjs` reproduces from repo root. Bandit is inapplicable to TypeScript-only changes. Scoped diff-check is clean.

Run final suite from `apps/packages/ui`:

```sh
./node_modules/.bin/vitest run src/hooks/__tests__/useMediaReadingProgress.test.tsx src/components/Review/__tests__/useMediaSelection.reading-progress.test.tsx src/components/Media/__tests__/ContentViewer.stage1.test.tsx src/components/Media/__tests__/ContentViewer.stage2.test.tsx src/components/Media/__tests__/ContentViewer.stage3.test.tsx src/components/Media/__tests__/ContentViewer.stage4.accessibility.test.tsx src/components/Media/__tests__/ContentViewer.stage10.findBar.test.tsx src/components/Media/__tests__/ContentViewer.stage12.performance.test.tsx src/components/Media/__tests__/ContentViewer.analysis-markdown.test.tsx src/components/Media/__tests__/ContentViewer.stage14.reprocess.test.tsx src/components/Review/__tests__/ViewMediaPage.permalink.test.tsx src/components/Review/__tests__/ViewMediaPage.stage13.error-handling.test.tsx src/components/Review/__tests__/ViewMediaPage.stage14.bulk-actions.test.tsx src/routes/__tests__/route-paths.viewport.test.ts src/routes/__tests__/option-media-multi.connection-state.test.tsx --maxWorkers=1 --no-file-parallelism
```

The owner test renders actual ViewMediaPage and actual reading/navigation hooks. It controls external storage, browser dimensions and scroll-event delivery through a small ContentViewer seam; it does not prove CSS geometry. Native corrected-layout wheel/PUT is separately recorded. No claim of native restored-position or390px success is made before the post-freeze check.
