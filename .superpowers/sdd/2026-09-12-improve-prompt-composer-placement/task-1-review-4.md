# TASK-12984.3 independent re-review 4

Reviewed range: `08ef0dbe37..5d4a4c76964cd2031e68c0b1bfb7325442a61234`.

Overall verdict: **NEEDS FIXES**, for the browser verification contract only.

Production verdict for the sole remaining Important collision finding: **ADDRESSED / APPROVED**. No new Critical or Important production defect was demonstrated in this diff. The required committed browser proof is not passing and needs a focused test-only correction before overall approval.

## Critical

None found.

## Important, test-only

### 1. Browser assertions do not consistently wait for drawer closure or accept the approved hidden fallback

Location: `apps/extension/tests/e2e/prompt-improvement.spec.ts:639-759` and `1955-2033`; related existing `expectLayoutNeutralFeedback` at lines 593-637 and shrink cases at 1827-1952.

The committed new regression launches successfully in independent review, but fails before reaching recipe verification. After Improve now, it waits for feedback visibility and immediately shrinks/scrolls. A drawer can still be completing its close transition. The resulting failure has `feedbackPainted: false` and `intersectsFeedback: false` for every control, while hit targets are the closing Prompt improvement drawer's title/status/View changes/Undo content. This is test synchronization, not feedback obstruction.

Waiting for the drawer to actually become non-presented separates two additional incorrect test assumptions:

- The helper equates CSS visibility plus an in-viewport center with actual painted visibility, and requires every such center to match its control. At a scroll-to-top position, pre-existing fixed UI can legitimately occupy those centers independently of Prompt Assist. The settled improvement set is exactly: **Message input, Save chat to history, Select a Prompt, Select a Model, Conversation context, Open knowledge search, Attach image**. The last five hit themselves. Message input instead hits the existing `tldw Assistant` header span (`ml-2 text-sm font-medium`); Save chat to history hits an existing `fixed z-20 left-0 right-0 flex justify-center` layer. Feedback is hidden and cannot be the interceptor. These unchanged fixed-header/fixed-overlay collisions are outside this Prompt Assist regression's responsibility; do not treat them as renewed feedback failures.
- The settled recipe set is exactly **Message input, Save chat to history, Select a Prompt**. All three center hit-tests match their own controls, and hidden feedback intersects none. The helper nonetheless fails `controls.length > 3`. A fixed lower-bound count is not a valid cross-operation visibility invariant; compiled recipe text changes the visible control set in the short viewport.

There is also an existing-suite mismatch after expanding the collision surface. Re-running the four prior cases yields **2 passed, 1 failed, 1 not run**: the two menu/drawer ownership cases pass, but the improvement shrink case asserts non-intersection against the layout bounds of an intentionally `visibility:hidden` feedback panel. Its bounds are `(8,8,344,82)` while the draft starts at y=0.47. Hidden geometry can intersect without painting or intercepting any pointer. The recipe case is then skipped by serial execution.

Required focused test correction:

1. Wait for all relevant drawer presentation to end before inspecting composer hit-tests. For recipe Apply, waiting by the old `Build from recipe` accessible name alone is insufficient: the shared drawer title changes during close. Wait for the actual wrapper to be unpainted/non-presented.
2. Preserve strict feedback non-intersection/non-interception assertions for genuinely painted, enabled composer controls. Establish a baseline-aware visible set or scroll below fixed chrome before requiring every center to hit itself; do not force clicks or silently ignore a feedback-owned hit.
3. Assert meaningful named controls for the intended state instead of an arbitrary `>3` count.
4. Update prior shrink cases to accept the explicitly approved deterministic hidden fallback: when hidden, require no paint/pointer ownership and retained state; restore a fitting viewport, require feedback to return, then real-click Undo and verify the exact draft. When painted, retain full viewport, collision, and real pointer assertions.

## Production evidence

- The collision boundary is the exact sidepanel composer form, selected through the trigger's closest `data-prompt-assist-collision-surface`; enumeration is not document-global. Other composers/page controls cannot enter this new query. All owned standard interactive element rectangles supplement draft/send rectangles, and their edges contribute placement candidates.
- Zero-size, `display:none`, `visibility:hidden`, own `aria-hidden`, own `aria-disabled`, and native disabled controls are excluded. Duplicate nested semantic rectangles are conservative, not unsafe. The boundary does not include body-portaled feedback or drawer contents.
- The independent settled V3 probe at 360 x 240 reports no painted feedback/control intersection for either operation. Improvement feedback is hidden with bounds `(8,8,344,82)`; recipe feedback is hidden with bounds `(8,8,197,46)`. Previously obscured Select a Prompt now center-hit-tests to itself. Recipe's three in-view controls all hit themselves.
- Both modes remain safely hidden at 360 x 50, return after restoring 390 x 844, and real mouse Undo restores `Retain exact {{topic}} draft.` exactly. Hidden fallback changes presentation only, not operation/Undo state.
- The original menu/drawer stacking regression cases independently pass after this fix. No menu, drawer, action-slot, mobile-width, system-prompt, Quick Chat, lifecycle, capability, dependency, or backend code changed in this range.
- Existing window/visual-viewport resize and capture-scroll scheduling and cleanup are unchanged. Queries run only while feedback is mounted and on the existing coalesced frame schedule, over the bounded composer rather than the page. No listener leak, server-render DOM access, or material performance issue was found in the scoped change.

## WebUI/options assessment

The new marker is sidepanel-only, so WebUI/options still use their existing editor/send rectangles. Their inline geometry differs from V3's intervening vertical control rows: the options toggle shares the editor row, followed by the external Send row. The marker omission does not introduce a new WebUI behavior change.

I additionally probed the real packaged options chat using the shared Playground implementation, with pro mode and expanded composer options. For both improvement and recipe at 360 x 240 after native editor scroll, feedback takes the hidden fallback and has no painted intersection with form controls. At 50px it remains hidden, and returning to 390 x 844 restores feedback and real exact Undo. No production reason to expand the marker to WebUI was demonstrated in these checks. This is evidence for the tested layouts, not a claim that every future WebUI layout is protected automatically; new intermediate control geometry should declare its ownership explicitly. The implementer's complete 13-test WebUI run was inspected rather than repeated.

## Independent verification

- Read the prior review, updated report/progress, and full implementation diff.
- Focused Vitest: **122 passed across 4 files**, exit 0.
- Committed new packaged V3 regression: **1 failed**, in application assertions rather than Chromium preflight. Chromium launched successfully with sandbox escalation.
- Prior four packaged regressions: **2 passed, 1 failed, 1 not run**, as detailed above.
- Temporary `/tmp/tldw-placement-review4.cjs` probes use the unchanged packaged artifact and E2E helpers, with explicit settled-wrapper checks, native scroll/resize, real center hit-testing, and real Undo mouse clicks. Diagnostic assertion failures were retained/reported while collecting the remaining operation evidence; no production/test source was edited and the committed tests were not weakened.
- `git diff --check` passes. Scope is seven task-owned files with 386 insertions/20 deletions; whitespace differences are focused component-test wrapper indentation, not formatter churn. Bandit is not applicable to the review-only Markdown artifact.

## Safety and handoff

All diagnostic contexts/mock servers use `finally` cleanup; no source files were changed. Final process inspection found no review probe, prompt-improvement test process, task browser, or port-18091 server. The unrelated existing Next dev server on port 8093 was untouched. The main checkout remains free of tracked changes, with only the two pre-existing untracked TTS spec/task files. The assigned worktree was clean before this report. Only this review artifact is committed. Complete the test-only correction, rerun the new V3 and prior shrink cases, and preserve the approved production behavior.
