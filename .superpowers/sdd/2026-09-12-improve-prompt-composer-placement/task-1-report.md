# TASK-12984.3 — compact composer Improve action

## Scope and diagnosis

The existing `PromptAssistComposerAction` was mounted as a standalone WebUI toolbar item and around the extension sidepanel's entire control area. `PromptAssistMenu` rendered a wide text trigger with a downward disclosure, while applied and recipe Undo feedback participated in normal layout. The V5 composer also owned a separate send slot, so moving only the legacy wrapper could not provide placement parity.

The implementation keeps all existing prompt-improvement and recipe lifecycle state in `PromptAssistComposerAction`; only its presentation and mounting slots changed. WebUI now mounts it in `composer-inline-send-control` immediately before the existing external Send control. Extension legacy/v1/v3 share the precise send cluster, and v5 receives the same cluster in `sendSlot`. Quick Chat remains untouched.

## RED evidence

Production code was not edited until these behavior/geometry tests failed against `ed682abb47`:

1. Focused component RED:

   ```text
   bunx vitest run src/components/Common/PromptAssist/__tests__/PromptAssistMenu.test.tsx src/components/Chat/composer/__tests__/PromptAssistComposerAction.test.tsx src/components/Sidepanel/Chat/__tests__/SidepanelComposerControlArea.prompt-assist.test.tsx --maxWorkers=1 --no-file-parallelism
   ```

   Result: **8 failed, 75 passed**. Failures proved the trigger was wide and lacked a title, feedback remained in normal flow, and legacy/v1/v3/v5 did not expose a send-action cluster.

2. WebUI `/chat` desktop/mobile RED:

   ```text
   bunx playwright test e2e/workflows/prompt-improvement.spec.ts --config=e2e/prompt-improvement.playwright.config.ts --reporter=line --workers=1 --grep "keeps one compact Improve action beside Send"
   ```

   Result: **2 failed** because the Improve trigger was outside the real inline Send cluster at both viewports.

3. Packaged-extension RED:

   ```text
   bunx playwright test tests/e2e/prompt-improvement.spec.ts --reporter=line --workers=1 --grep "render one compact upward action immediately before Send"
   ```

   Result: **1 failed** because the real sidepanel send cluster contained no Improve action.

## GREEN implementation

- Added the compact, icon-only 44 px presentation and upward placement to `PromptAssistMenu`, retaining the existing accessible name and adding a native title.
- Measured the open disclosure in a layout effect to clamp it inside the horizontal viewport and cap its height to the available space above/below the trigger. Escape still closes the same disclosure and focuses the real trigger ref.
- Anchored applied/View changes/Undo and recipe Undo feedback absolutely above the action, preserving the lifecycle handlers without changing toolbar height.
- Removed the standalone WebUI toolbar-owned action, mounted the same action before external Send, and kept the existing current draft/model/backend/focus props.
- Narrowed `SidepanelComposerControlArea` to the actual Send clusters for casual/pro legacy, v1, v3, and v5. No Quick Chat component was changed.
- Updated prompt-improvement browser expectations to preserve all three actions, including the local recipe action when remote Improve capability fails closed.

## GREEN verification

- Focused Vitest: **4 files, 119 tests passed**.
- Complete WebUI prompt-improvement Playwright: **12 passed**, including desktop/mobile placement, 44 px geometry, upward viewport bounds, stable toolbar height after Apply, system-prompt behavior, focus restoration, Undo, and axe/overflow coverage.
- Self-contained packaged-extension prompt-improvement Playwright: **13 passed**, including narrow `/options.html#/chat`, casual/pro legacy/v1/v3/v5, Quick Chat exclusion, focus restoration, Undo, and viewport assertions.
- Full packaged-extension file: **13 passed; 1 environment-gated real-local-server smoke could not run** because `TLDW_PROMPT_IMPROVEMENT_E2E_SERVER_URL` / API-key configuration was absent. The dedicated fail-closed harness test passed.
- Extension TypeScript compile: `bun run compile` — **passed**.
- Scoped ESLint using the WebUI flat config: **exit 0**; repository-baseline warnings remain in legacy files, with no errors.
- Scoped Prettier check on changed non-legacy components/tests: **passed**. The legacy `PlaygroundForm.tsx`, sidepanel form, toolbar, and Web E2E file were kept to hand-applied semantic hunks because whole-file formatting would rewrite unrelated baseline content.
- `git diff --check`: **passed**.
- Bandit: **N/A** — the touched implementation and test scope is TypeScript/TSX only.

## Worktree safety and minimal-diff recovery

An early relative `apply_patch` mistake temporarily replaced three tracked files in the main checkout with the assigned worktree's starting blobs:

- `/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Sidepanel/Chat/form.tsx`
- `/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/e2e/workflows/prompt-improvement.spec.ts`

Work stopped immediately. The root coordinator restored and verified all three tracked main-checkout paths byte-for-byte against main HEAD; only the pre-existing untracked TTS spec/task remained there. The same three isolated-worktree files were restored to `ed682abb47`, then only semantic task hunks were reapplied with absolute paths. Broad Prettier churn was removed, and normal versus whitespace-ignored diff stats were audited before verification.

## Fix round 1 — responsive overlay and drawer corrections

Independent review at `5ae32a3049` found that the nominally upward disclosure and absolutely positioned feedback could still be clipped by narrow V3/V5 composer ancestors, and that the WebUI/options mount no longer passed the existing mobile drawer signal at 640 px. The correction keeps all Prompt Assist state and actions unchanged while moving only transient presentation into viewport-aware body portals.

### Fix-round RED evidence

1. Packaged narrow V3/V5 action hit-testing:

   ```text
   bunx playwright test tests/e2e/prompt-improvement.spec.ts --reporter=line --workers=1 --grep "narrow V3 and V5 activate every Prompt Assist menu action"
   ```

   Result: **failed** because the real menu extended beyond an `overflow-hidden` composer ancestor. The test uses `elementFromPoint` plus real `page.mouse.click` activation for Improve now, Review changes, and Build from recipe rather than visibility alone.

2. Portaled disclosure component contract:

   ```text
   bunx vitest run src/components/Common/PromptAssist/__tests__/PromptAssistMenu.test.tsx --maxWorkers=1 --no-file-parallelism
   ```

   Result: **1 failed, 12 passed** because the upward disclosure remained `absolute bottom-full` inside its composer ancestor instead of a fixed body-level overlay.

3. Packaged narrow V3/V5 feedback geometry and pointer access:

   ```text
   bunx playwright test tests/e2e/prompt-improvement.spec.ts --reporter=line --workers=1 --grep "narrow V3 and V5 keep improvement and recipe feedback"
   ```

   Result: **failed** with feedback `x = -164`. The test exercises Apply, View changes, improvement Undo, recipe Apply, and recipe Undo with painted-center pointer hit-testing; it also checks full viewport containment, no overlap with the draft or required send controls, and unchanged send-cluster dimensions.

4. WebUI 640 px drawer presentation:

   ```text
   bunx playwright test e2e/workflows/prompt-improvement.spec.ts --config=e2e/prompt-improvement.playwright.config.ts --reporter=line --workers=1 --grep "640px review and recipe drawers"
   ```

   Result: **failed** because both drawers were 480 px rather than the 640 px viewport width.

### Fix-round implementation

- Upward composer disclosures now render in a fixed body portal with 8 px horizontal/vertical clamping and available-height limits. Resize and capture-phase scroll events schedule a single animation-frame reposition; cleanup removes listeners and cancels the frame.
- The portal preserves trigger/disclosure focus as one logical interaction: focus may move into the portal, Tab and Shift+Tab traverse across the portal boundary, outside focus closes it, and Escape closes it and returns focus to the real trigger. The default non-composer/system-prompt downward presentation remains inline and unchanged.
- Improvement and recipe feedback now share a fixed, collision-aware body overlay anchored above the visible draft rather than participating in composer layout. Semantic color tokens, action handlers, lifecycle timing, and exact Undo state are unchanged; the overlay z-index keeps its real buttons above drawer close layers.
- Restored `narrow={isMobileViewport}` at the WebUI/options composer mount, preserving the established `<767 px` full-width drawer behavior.
- Test hit-testing waits only for the painted center to become the target before issuing the real mouse click, avoiding animation sampling flakes without weakening clipping or geometry assertions.

### Fix-round GREEN verification

- Focused Vitest: **4 files, 121 tests passed**.
- Complete WebUI prompt-improvement Playwright: **13 passed**, including desktop/mobile placement, body-portaled status geometry, and 640 px improvement/recipe drawers.
- Self-contained packaged-extension prompt-improvement Playwright: **16 passed** against a freshly built production Chrome artifact, including real narrow V3/V5 activation for every menu action, resize/scroll containment, improvement Apply/View/Undo, recipe Apply/Undo, stable toolbar geometry, 640 px options drawers, system-prompt behavior, and Quick Chat exclusion. The separately environment-gated real-local-server smoke remains excluded because server/API-key configuration was not supplied; its fail-closed harness coverage passed.
- Extension TypeScript compile: `bun run compile` — **passed**.
- Scoped ESLint via the WebUI flat config: **exit 0**, with only out-of-base warnings for shared-package paths and no errors. Extension has no local ESLint flat config.
- Explicit extension Prettier config check on changed non-legacy components/tests: **passed** after hand-formatting only the new snippets. No whole-file formatter was run on legacy sources during final cleanup.
- `git diff --check`: **passed**.
- Bandit: **N/A** — fix-round implementation and tests are TypeScript/TSX only.

During fix-round verification, an extension test still scoped the applied status to its former `chat-messages` ancestor and a Web test scoped it to the former send-cluster ancestor. Those stale locators failed after the intentional body portal; they were changed to select the visible exact status globally while retaining layout and pointer assertions. One painted-center check sampled during drawer motion once; the same production artifact passed unchanged after the helper waited for stable hit-testing. No third production layout approach was attempted.

The main checkout was not touched during fix round 1. An accidental formatter invocation with the wrong discovered config caused semicolon-only churn in four task-owned isolated-worktree files; those exact paths were restored and the semantic changes reapplied with absolute `apply_patch` paths. A subsequent check used the extension's explicit `.prettierrc.cjs`; remaining normal-versus-whitespace-ignored diff differences are localized to the body-portal JSX nesting rather than repository-wide formatting.

## Fix round 2 — overlay ownership and viewport mutation safety

Independent re-review at `182d3d323e` verified the original clipping and 640 px drawer fixes but reproduced two follow-on defects: persistent feedback at `z-[1100]` intercepted reopened menu and inspection-drawer controls, and the feedback position formula did not bound its bottom edge or avoid a draft moved near the visual viewport top.

### Fix-round 2 RED evidence

Production was unchanged while each real packaged-browser behavior failed:

1. V5 feedback versus reopened menu:

   ```text
   TLDW_E2E_SKIP_EXTENSION_BUILD=1 bunx playwright test tests/e2e/prompt-improvement.spec.ts --reporter=line --workers=1 --grep "V5 applied feedback yields"
   ```

   Result: **1 failed**. After Improve now left feedback active, the painted center of Review changes resolved to feedback rather than the menu button; `clickThroughPaintedCenter` timed out with `matches: false`.

2. V3 feedback versus inspection drawer:

   ```text
   TLDW_E2E_SKIP_EXTENSION_BUILD=1 bunx playwright test tests/e2e/prompt-improvement.spec.ts --reporter=line --workers=1 --grep "V3 applied feedback yields"
   ```

   Result: **1 failed**. After View changes opened and the drawer settled, the Changes control's center remained covered by feedback and failed real pointer hit-testing.

3. Improvement feedback after viewport shrink and real composer scroll:

   ```text
   TLDW_E2E_SKIP_EXTENSION_BUILD=1 bunx playwright test tests/e2e/prompt-improvement.spec.ts --reporter=line --workers=1 --grep "improvement feedback stays usable after viewport shrink"
   ```

   Result: **1 failed** at 360 x 240 because the fixed feedback intersected the real draft after `scrollIntoView` moved the composer near the viewport top.

4. Recipe feedback after the same mutation:

   ```text
   TLDW_E2E_SKIP_EXTENSION_BUILD=1 bunx playwright test tests/e2e/prompt-improvement.spec.ts --reporter=line --workers=1 --grep "recipe feedback stays usable after viewport shrink"
   ```

   After correcting a test-only missing `scrollIntoViewIfNeeded` for the drawer Apply button, the production RED was **1 failed** because recipe feedback also intersected the real draft.

### Fix-round 2 implementation

- `PromptAssistMenu` reports its actual open state to its composer owner. The owner suppresses persistent feedback while the menu owns pointer/focus interaction, without clearing improvement or recipe Undo state.
- Feedback is also suppressed from drawer open intent through Ant Drawer `afterOpenChange(false)`, so it neither covers opening/active drawers nor remounts during their closing animation. Same-operation inspection feedback returns after the drawer fully closes; beginning a new Review changes operation retains the existing new-operation lifecycle.
- Removed the global `z-[1100]` escalation; feedback uses the normal transient-overlay layer and active menu/drawer visibility is coordinated explicitly.
- Feedback geometry now measures the real textarea/contenteditable draft and the owning send-action cluster. It tries deterministic above/below candidates with an 8 px gap, requires full visual-viewport containment, and hides rather than painting over required content if no collision-free rectangle exists.
- Horizontal and vertical bounds use `visualViewport` offsets/dimensions when available. The overlay width is constrained before measurement, and window resize, capture-phase scroll, plus visual-viewport resize/scroll all share the existing single-RAF scheduler and cleanup.

### Fix-round 2 GREEN verification

- Exact new packaged-browser cases: **4 passed** together. V5 Review changes, V3 Edit/Changes, improvement Undo, and recipe Undo all center-hit-test to themselves and activate through real mouse input after the required transitions and 360 x 240 scroll/resize mutations.
- Focused Vitest: **4 files, 121 tests passed**.
- Complete self-contained packaged-extension prompt-improvement Playwright: **20 passed**; the final production refinement was rebuilt and the exact four geometry/ownership cases passed again. The external real-local-server smoke remains excluded because its server/API-key configuration was not supplied; fail-closed harness coverage passed.
- Complete WebUI prompt-improvement Playwright: **13 passed**. After the final shared-component refinement, the desktop/mobile feedback geometry subset also passed **2/2**.
- Extension TypeScript compile and production Chrome build: **passed**. Build emitted only existing duplicate-import and stale Browserslist warnings.
- Explicit extension Prettier config check for all three changed source/test files: **passed**. Normal and whitespace-ignored diff stats are identical; no legacy file or main checkout path was formatted or changed.
- Scoped ESLint through the available WebUI flat config: **exit 0**, with two expected outside-base warnings and no errors; extension/shared UI has no local ESLint config.
- `git diff --check`: **passed**.
- Bandit: **N/A** — fix-round files are TypeScript/TSX only.

### Fix-round 2 self-review

- Menu visibility is reported in a layout effect, so feedback is removed before the open menu is painted. Drawer visibility covers both logical open intent and the animation-backed presented state.
- Same-operation inspection close returns feedback and exact Undo; the V3 test proves both. A reopened V5 Review changes starts a new operation, so the prior feedback correctly does not reappear after cancel.
- Draft discovery is limited to actual editor surfaces rather than generic auxiliary inputs. Collision candidates are conservative across the draft and required send cluster, remain layout-neutral, and use a non-painting fallback if the viewport cannot physically fit the feedback.
- Window and visual-viewport listeners are paired in cleanup, and pending animation frames remain cancelled on teardown. No prompt action, backend/capability contract, dependency, placement, persistence, system-prompt behavior, or Quick Chat scope changed.

## Fix round 3 — complete V3 composer collision ownership

Independent re-review at `0a9546c1b4` reproduced one remaining short-viewport obstruction: at 360 x 240 after a real V3 composer scroll, feedback fit between the draft and final send cluster but covered the intermediate Save chat to history, Select a Prompt, and scroll-to-latest controls. The root cause was the placement policy's two-rectangle ownership model, not overlay stacking or viewport clamping.

### Fix-round 3 RED evidence

Before production changes, a packaged V3 regression was added that repeats the review's 390 x 844 apply flow, shrinks to 360 x 240, scrolls the real draft to the viewport top, derives the composer surface from the real input/send ancestry, and enumerates every visible enabled interactive control. It requires every control rectangle to avoid painted feedback and every center `elementFromPoint` result to belong to that control for both improvement and recipe feedback. It also shrinks to 360 x 50, requires the feedback to become unpainted while state persists, restores the viewport, and activates the real Undo button.

The packaged test could not reach application behavior in this host session. Exactly three unchanged attempts were made, then retries stopped per the three-attempt cap. Each failed in `probePackagedRuntime` while starting `launchPersistentContext`: Chromium exited with `SIGABRT` and `Target page, context or browser has been closed` (PIDs 11335, 11355, and 11450; cleanup also reported `kill EPERM`). No extension page or mock server was reached. The independent review already supplies the corresponding real-browser production RED; the new packaged test remains committed for independent re-review.

To retain strict local TDD, a focused real-component geometry test was then added before production edits. It renders `PromptAssistComposerAction` in a marked composer surface and supplies literal rectangles for the 360 x 240 viewport, draft, intermediate control, send cluster, trigger, and 344 x 82 feedback. RED was **1 failed**: the feedback remained painted at `top: 70px` and `visibility: ""` where the intermediate control occupied the only apparent gap; the test expected the safe hidden fallback.

### Fix-round 3 implementation

- The sidepanel form now marks its exact interactive composer ownership boundary with `data-prompt-assist-collision-surface`.
- Feedback placement queries only that marked surface for standard interactive elements. Zero-sized, `display:none`, `visibility:hidden`, `aria-hidden`, `aria-disabled`, and disabled controls are excluded; no document-wide or label-specific production selector is used.
- Every visible control rectangle joins the existing draft/send blocked rectangles, and placement candidates include both edges of every owned control. Existing visual-viewport bounds, horizontal clamping, menu/drawer suppression, capture-scroll/resize scheduling, and deterministic hidden fallback remain unchanged.
- Both improvement and recipe feedback use the same corrected overlay. Hidden fallback changes only paint visibility; the existing operation and exact Undo state are retained and return when a later resize/scroll creates space.

### Fix-round 3 GREEN verification

- Exact component geometry/Undo case: **1 passed**. It proves hidden fallback at 360 x 240, retained Undo state in the DOM, resize-driven return at 360 x 500, and exact draft restoration through the real Undo handler.
- Focused Vitest: **4 files, 122 tests passed**.
- Extension TypeScript compile and fresh production Chrome build: **passed**. Build output contained only the established duplicate-import and stale Browserslist warnings.
- Complete WebUI prompt-improvement Playwright: **13 passed** against a temporary webpack dev server, including desktop/mobile placement, feedback geometry, focus, accessibility, and 640 px improvement/recipe drawers. The server was stopped afterward and port 18091 was closed.
- Packaged extension browser verification, including the new V3 case and four prior regressions: **environment-blocked after the required three-launch cap**, as detailed above. No post-fix packaged retry was made; independent re-review must execute the committed proof.
- Explicit extension Prettier check for the three changed non-legacy component/test files: **passed**. The one-line legacy sidepanel marker was hand-applied; no whole-file legacy formatter ran.
- Scoped ESLint with the repository's installed frontend version: **exit 0**, with existing warnings only and no errors. `git diff --check` passed; normal and whitespace-ignored stats differ only in the focused test-harness JSX wrapping, not legacy formatting churn.
- Bandit: **N/A** — fix-round implementation and tests are TypeScript/TSX only.

### Fix-round 3 self-review

Collision discovery is bounded to the owning composer form, so unrelated page controls neither affect placement nor incur query work. The query runs only while transient feedback is mounted and only on the existing initial/resize/capture-scroll/visual-viewport schedule. Whole control rectangles are protected, so nested icon/text descendants do not weaken the hit target; duplicate semantic matches are harmless conservative rectangles. Hidden and zero-sized inputs do not consume space. No lifecycle, request, capability, persistence, menu/drawer coordination, V5 placement, system-prompt presentation, dependency, backend, or Quick Chat code changed.
