# TASK-12984.3 independent re-review 2

Reviewed fix-round range: `ec4cb0da44..344b6cefed`.

Spec-compliance verdict: **NEEDS FIXES**.

Code-quality verdict: **NEEDS FIXES**.

## Prior Important findings

1. **ADDRESSED for the original ancestor-clipping defect.** The upward menu is now a fixed body portal. Real narrow V3/V5 measurements show all three fresh-menu action centers hit their buttons, and a real Improve now mouse activation succeeds. The default downward menu remains inline. A new stacking regression when feedback is present is described below; this prevents overall menu acceptance.
2. **NOT ADDRESSED completely.** Both feedback variants now share a body portal and the original negative-x/ancestor-clipping defect is corrected in the normal 390 x 844 state. However, the new stacking order obscures menu/drawer controls, and vertical collision handling still fails on viewport shrink and container scroll. See both Important findings below.
3. **ADDRESSED.** The mount restores `narrow={isMobileViewport}`. Existing `<768px` mobile selection again passes `100vw` to both review and recipe drawers, while desktop still passes 480. The fix adds real 640px WebUI and packaged-options tests for both drawer modes. Their reported passing runs were inspected; the re-review independently verified the prop/data path rather than rerunning those complete browser suites.

## Critical

None found.

## Important

### 1. Feedback is stacked above the reopened menu and active drawer

Location: `apps/packages/ui/src/components/Chat/composer/PromptAssistComposerAction.tsx:132`, together with the unconditional applied/recipe feedback mounts at lines 500-530. The menu remains `z-50` at `PromptAssistMenu.tsx:325`.

Raising feedback to `z-[1100]` places it above both the action menu and the default Ant Design drawer. Feedback remains mounted while the menu reopens and while View changes opens inspection. This introduces real pointer blocking, not just a cosmetic overlap.

Real packaged Chromium reproduction in pro V5 at 390 x 844:

1. Enter a draft, open Improve, and mouse-click Improve now.
2. Leave Applied feedback present and reopen Improve.
3. The menu is `(x=8, y=239.78, w=288, h=244, z=50)`. Feedback is `(x=8, y=350.77, w=373.5, h=46, z=1100)`.
4. The Review changes action is `(x=17, y=334.78, w=270, h=62)`. Its center hits the feedback's **View changes** button. A real Playwright trial click on Review changes times out. This reproduced in successive diagnostic runs.

The same ordering breaks inspection in V3. After clicking feedback View changes and waiting until the full-width drawer has slid into place, feedback at y=323.47-369.47 covers the drawer's Edit/Changes buttons at y=302-346. Their center hit-tests resolve to the feedback panel, not the buttons. The underlying feedback remains clickable above a modal drawer.

Coordinate these overlay layers/states so feedback cannot cover an active menu or drawer, without clearing exact Undo state or changing lifecycle timing. Do not solve close-animation sampling by globally raising persistent feedback over active modal content. Add real mouse tests for Apply -> reopen menu -> each action and Apply -> View changes -> Edit/Changes, with feedback retained.

### 2. The feedback's vertical placement is not viewport- or collision-safe

Location: `apps/packages/ui/src/components/Chat/composer/PromptAssistComposerAction.tsx:95-102`.

`top: Math.max(8, composerTop - 8 - overlayHeight)` only enforces a top inset. It neither bounds the overlay bottom to the viewport nor chooses another collision-free region when there is insufficient space above the draft. Resize/scroll listeners run, but recomputing this expression cannot satisfy either missing constraint. Both improvement and recipe feedback share this calculation.

Real packaged-browser evidence after settled layout:

- Shrinking pro V5 from 390 x 844 to 390 x 360 leaves feedback at y=350.77 with height 46; its bottom is 396.77. Both action centers are below the viewport and fail hit-testing. At height 240 it remains entirely below the viewport. V3 similarly retains y=323.47, height 46 on shrink.
- After scrolling the actual scrollable composer container in a 390 x 180 viewport, V5's draft occupies `(x=42, y=19.77, w=306, h=60)`, while feedback clamps to `(x=8, y=8, w=373.5, h=46)`. It covers the draft. V3 reproduces the same intersection with draft y=20.47. Two animation frames were awaited after resize/scroll, and the measured positions match the current calculation.

Use a vertical placement policy that accounts for viewport height, the visible draft, and required controls, with a usable fallback when the preferred region does not fit. Preserve toolbar geometry and Undo lifetime. Add real container-scroll and substantial height-reduction tests for both feedback modes. The current 844px-only feedback cases and synthetic scroll event do not cover these states.

## Test-quality and implementation observations

- The new `clickThroughPaintedCenter` helper is condition-based, waits for a real hit-test match, and issues `page.mouse.click` without force. A permanently covered target still fails; this is a meaningful improvement over bounding-box-only checks.
- The new tests activate each menu action in a fresh state and Undo before the next menu flow. They do not exercise the overlapping states in Important finding 1.
- The replacement feedback locator, exact status + visible + `.first()`, is ambiguous during drawer transitions. An independent probe using that pattern first selected the drawer's status parent at `(x=24, y=81, w=342, h=137, z=auto)`, not the fixed feedback overlay. The probe was corrected to select the actual body-level fixed feedback. Geometry and hit-testing assertions should target that specific overlay and wait for the intended drawer state; otherwise different checks can observe different matching elements.
- The resize test grows width from 390 to 400 and reduces height only to 800, then dispatches a scroll event without changing scroll position. It does not prove repositioning under a materially different boundary or actual capture-phase scrolling.
- Portal targets are read in guarded layout effects, not during server render. Resize/capture-scroll listeners are removed and pending animation frames are cancelled. Trigger/disclosure focus is treated as one logical region through document focus tracking and React portal event bubbling. The default inline/downward branch retains the prior positioning and blur behavior.
- Tab entry, forward exit, outside click, and Escape have component coverage. Shift+Tab and more complex focus order are not newly tested; no separately reproduced Important keyboard defect is asserted here.
- The restored mobile prop is a single-line change. No backend, capability, persistence, dependency, Quick Chat, or unrelated production changes occur in this fix range. The 897-added-line diff is largely targeted browser coverage and portal JSX nesting, not broad formatter churn. Whitespace-ignored stats are 832 insertions/41 deletions versus 897/106 normally, concentrated in the menu's new portal wrapper. `git diff --check` passes.

## Independent verification and safety

Read prior review, full implementation report including appended fix evidence, ledger, and the complete fix diff.

Re-ran the four focused Vitest files from `apps/packages/ui` with `--maxWorkers=1 --no-file-parallelism`: **121 tests passed in 4 files**, exit 0.

The temporary `/tmp/tldw-placement-rereview.cjs` probe loaded unchanged existing E2E helper code in memory, launched the current packaged production extension with its local mock server, and inspected real narrow V3/V5 initial menus, Apply feedback, reopened menus, settled inspection drawers, viewport shrink, and actual container scroll. It used real hit-testing and trial mouse clicks, without changing production/test code or geometry. An initial diagnostic run exposed the ambiguous status locator described above; the corrected diagnostic completed and reproduced the two findings. This re-review did not rerun the full WebUI/extension suites, build, lint, or axe runs recorded by the implementer.

All probe browser contexts and mock servers closed in `finally` blocks; final process inspection found no re-review probe, task browser, or port-18091 dev server. The unrelated existing Next dev server on port 8093 was left untouched. Main checkout still has only the two pre-existing untracked TTS spec/task files and no tracked changes. The assigned worktree was clean before adding this artifact. Bandit is not applicable to the review-only Markdown change. Only this review artifact is committed; no subagents were used.
