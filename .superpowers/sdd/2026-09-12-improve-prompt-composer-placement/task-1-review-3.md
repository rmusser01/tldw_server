# TASK-12984.3 independent re-review 3

Reviewed fix-round-2 range: `1ceaefd080..8b9b87cb98`.

Spec-compliance verdict: **NEEDS FIXES**.

Code-quality verdict: **NEEDS FIXES**.

## Prior findings

- Original feedback-safety finding: **NOT ADDRESSED completely**. Viewport containment, draft/send avoidance, safe hiding, and Undo restoration are improved and verified, but other required composer controls remain unprotected by collision detection.
- Prior Important regression 1, feedback above reopened menu/inspection drawer: **ADDRESSED**. The exact V5 and V3 real-mouse regression cases pass independently. Menu open state and drawer open/presented state suppress feedback without clearing its backing Undo state. The arbitrary `z-[1100]` escalation is removed. Same-operation inspection closes back to feedback and exact Undo; new Review changes follows the unchanged new-operation lifecycle rather than reviving prior applied state.
- Prior Important regression 2, unsafe vertical placement: **NOT ADDRESSED completely**. Full visual-viewport bounds and draft/send collision checks now work in the tested mutations, and deterministic hiding avoids impossible placements. However, a candidate between the editor and send cluster can still cover required intermediate controls. The single residual Important finding below completes this prior feedback requirement rather than expanding scope.
- Earlier menu ancestor clipping and 640px mobile drawer corrections remain intact; this diff does not alter placement slots, the body-portaled menu, mobile width prop, or default system-prompt branch.

## Critical

None found.

## Important

### 1. Collision detection omits required controls between the editor and Send

Location: `apps/packages/ui/src/components/Chat/composer/PromptAssistComposerAction.tsx:104-136`, especially `blockedRects = [draftRect, controlsRect]` at line 116.

The new collision calculation treats only the draft and the narrow send-action cluster as blocked. The V3 composer has required controls between those rectangles, including Save chat to history and Select a Prompt. A candidate below the draft therefore passes `fitsAt` even when it covers that control row.

Independent real packaged Chromium reproduction:

1. Open pro sidepanel V3 at 390 x 844, fill `Retain exact {{topic}} draft.`, and activate Improve now through the painted center.
2. Wait for the prompt-improvement drawer to be hidden and feedback to remain visible.
3. Resize to 360 x 240 and scroll the actual draft into view with `block: "start"`; await two animation frames.
4. Feedback is `(x=8, y=68.47, w=344, h=82)`, ending at y=150.47. It is inside the viewport and avoids the draft and send cluster.
5. Save chat to history is `(x=117, y=89.48, w=28, h=16)`, entirely covered by feedback. Select a Prompt is `(x=117, y=113.48, w=44, h=44)`, with most of its target covered. Both center hit-tests resolve to the feedback portal rather than their intended controls. Scroll to latest messages is covered as well.

The body portal is painted at the normal overlay level, so these are real pointer obstructions, not merely intersecting off-screen rectangles. The approved requirement includes required composer controls, not only the final Send cluster.

Include the required control area in the collision policy, or explicitly provide its occupied rectangles from the owning composer. Retain the deterministic hidden fallback when no safe candidate exists and preserve Undo state. Add regression assertions for Save chat to history and Select a Prompt hit-testing at the short V3 viewport for both improvement and recipe feedback, since both share this overlay. The existing geometry helper checks only the draft and send cluster and consequently passes this occupied placement.

## Positive verification and scope observations

- The four exact new packaged regressions independently pass together: V5 reopened Review changes; V3 drawer Changes/Edit with feedback returning and exact Undo; improvement feedback after 360 x 240 shrink plus real scroll in V3/V5; recipe feedback through the same mutations and real Undo.
- The tests use painted-center `elementFromPoint` checks and real `page.mouse.click`, not forced clicks. Drawer settling waits for the actual transform condition. The new body-level feedback locator removes the previous ambiguity with the drawer's duplicate status. Permanent hit-test failure is not swallowed.
- A separate real-browser probe verified the hidden fallback in both V3 and V5 by shrinking the viewport to a deliberately non-fitting height of 50px. Feedback becomes `visibility: hidden`; returning to 390 x 844 restores it, and a real Undo click restores the exact original draft. This is a stress probe, not a claimed supported-device size.
- Geometry uses visual-viewport offsets/dimensions, constrains width before measurement, and requires full vertical containment. Window resize/capture-scroll and visual-viewport resize/scroll listeners are paired with cleanup; pending animation frames are cancelled. Portal DOM access remains inside guarded layout effects, not server rendering. No new Strict Mode-specific side-effect leak was found in inspection.
- Feedback visibility is derived from menu and drawer states. Existing focus/Escape/outside handling and prompt operation lifecycle handlers are otherwise unchanged. New-operation state reset is not replaced by an ad hoc feedback reset.
- The exact four browser cases cover the previously reproduced failures but do not cover all required intermediate controls. Hidden-fallback restoration was independently probed; native mobile pinch-zoom/keyboard visual-viewport offsets were inspected in code, not exercised on a physical device.
- No backend, capability, persistence, dependency, Quick Chat, standalone-row, trigger-size, or system-prompt drift appears in this fix diff. Normal and whitespace-ignored numstats are identical: 427 insertions/19 deletions over six task-scoped files. `git diff --check` passes.

## Independent commands and safety

Read prior review, appended fix-round-2 report, ledger, and full diff.

Focused Vitest, four files with `--maxWorkers=1 --no-file-parallelism`: **121 passed**, exit 0.

```text
TLDW_E2E_SKIP_EXTENSION_BUILD=1 bunx playwright test tests/e2e/prompt-improvement.spec.ts --reporter=line --workers=1 --grep "V5 applied feedback yields|V3 applied feedback yields|improvement feedback stays usable after viewport shrink|recipe feedback stays usable after viewport shrink"
4 passed (20.4s), exit 0.
```

The temporary `/tmp/tldw-placement-review3.cjs` probe loaded unchanged E2E helpers in memory, used the existing packaged artifact/local mock server, measured actual control intersections/hit-testing after settled drawer state, and verified no-space hiding/restoration with exact Undo. Its initial run used a helper whose <=40px scroll assumption did not fit the shorter diagnostic V5 content; the final probe used the actual reachable native scroll position and completed with exit 0. No production or test source was edited. Full-suite/build/lint/axe evidence was inspected in the implementation report rather than blindly rerun.

Browser contexts and mock servers closed in `finally` blocks. Final process inspection found no review probe, prompt-improvement test process, task browser, or port-18091 server. The unrelated pre-existing Next dev server on port 8093 was untouched. Main checkout still has only the two pre-existing untracked TTS spec/task files and no tracked edits. Assigned worktree was clean before this report. Only this artifact is committed; no subagents were used. Bandit is not applicable to the review-only Markdown change.
