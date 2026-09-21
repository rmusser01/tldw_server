# TASK-12984.3 independent review 1

Reviewed range: `ed682abb47..ec6a83cb115ea20f3ad9a998dac39cdf3fc4cb9e`.

Spec-compliance verdict: **NEEDS FIXES**.

Code-quality verdict: **NEEDS FIXES**.

## Critical

None found.

## Important

### 1. The upward menu remains clipped by composer ancestors

Location: `apps/packages/ui/src/components/Common/PromptAssist/PromptAssistMenu.tsx:206-208`, integrated through `apps/packages/ui/src/components/Sidepanel/Chat/form.tsx:4306-4327`.

The popup is still an absolutely positioned descendant of the composer. Horizontal viewport translation and `z-50` cannot escape `overflow-hidden` ancestors in V3 and mobile V5. In real packaged Chromium at 390 x 844, V5's menu was `(x=8, y=73.78, w=288, h=260)`, but its clipping composer started at `(x=40, y=221.47, w=310, h=171.31)`. The Improve now and Review changes button centers hit unrelated underlying elements. A real Playwright trial click of Improve now timed out because the message-log subtree intercepted pointer events. V3 also clips the left edge: menu starts at x=8 while the clipping ancestor starts at x=77.

This violates the usable upward menu contract across composer variants. Render the overlay outside clipping ancestors, or otherwise remove the clipping relationship safely, while retaining positioning and outside/blur/Escape/focus semantics. Add packaged narrow V3/V5 checks that actually activate every action and inspect clipping/hit-testing, not only bounding boxes. Exercise open/resize/scroll and the default system-prompt presentation after the overlay correction.

### 2. Applied feedback places Undo off-screen and overlaps the draft

Location: `apps/packages/ui/src/components/Chat/composer/PromptAssistComposerAction.tsx:399-429`.

`right-0 w-max max-w-[calc(100vw-1rem)]` caps width but does not clamp the feedback position. The trigger is before Send, so anchoring a nearly viewport-wide panel to its right edge pushes it left of the viewport. The feedback also inherits composer clipping and is placed directly over the input in mobile V5.

After real mocked Improve now application via keyboard at 390 x 844:

- V3 feedback: `(x=-118, y=470.48, w=374, h=82)`; Undo improvement: `(x=-109, y=515.48, w=126.05, h=28)`. Its center is off-screen, and the remaining sliver is outside the V3 clipping ancestor.
- V5 feedback: `(x=-98.19, y=251.78, w=374, h=82)`; Undo improvement: `(x=-89.19, y=296.78, w=126.05, h=28)`. Its right edge is x=36.86, still left of the composer's x=40 clipping edge, so the button is entirely unreachable by pointer.
- V5 input: `(x=50, y=272.77, w=290, h=44)`, intersecting the feedback rectangle.

Preserve layout neutrality but position feedback within a usable overlay boundary, outside clipping and without covering the draft/required controls. Apply the correction to both feedback blocks, including recipe Undo. Add real browser Apply/View changes/Undo and recipe Undo checks at narrow sizes, with hit-testing and stable toolbar dimensions. The existing class assertions and toolbar-height assertion pass while Undo is unavailable.

### 3. Moving the WebUI/options action drops its mobile drawer width

Location: `apps/packages/ui/src/components/Option/Playground/PlaygroundForm.tsx:4939-4946`.

The previous toolbar owner passed `narrow={isMobile}`. The new mounting site does not pass `narrow`, which defaults to false, so both review and recipe drawers now request 480px instead of `100vw`. The shared mobile breakpoint is 767px. A real packaged options-chat check at 640 x 844 reported `.ant-drawer-content-wrapper` style `width: 480px` and measured width 480, rather than the previous full-width 640px sheet. The existing 390px test misses this because Ant Design caps the 480px drawer to the smaller viewport.

Restore the existing mobile prop using `isMobileViewport`, and cover a mobile viewport between 480px and 767px for both drawer modes. This is a drawer-presentation regression outside the approved placement-only change.

## Minor

No additional independent minor findings. The browser coverage weaknesses are incorporated into the fixes above.

## Compliance and quality observations

- Placement ownership is otherwise coherent: WebUI uses the external Send cluster independently of collapsed options; extension pro/casual branches and V5's send slot each mount the shared cluster. React element reuse is through mutually exclusive branches, not simultaneous duplicate mounts.
- The trigger retains its accessible name, adds a native title, uses semantic color tokens, and has an explicit 44px target. The non-composer menu defaults remain non-compact and bottom-opening.
- The existing draft/model/context/backend/authorization props and lifecycle handlers are preserved, except for the omitted mobile width prop. No backend, capability implementation, persistence implementation, or dependency changes occur in this range. Quick Chat source is untouched.
- Full diff inspection and normal versus whitespace-ignored numstats show bounded semantic hunks, not the reported broad formatter rewrite. `git diff --check` passes.
- Existing outside/blur/Escape handling remains in place. Its correctness must be retained if a portal is introduced to solve finding 1.
- The new geometry assertions inspect real DOM bounds, but bounds and `toBeVisible()` alone do not establish that a popup is painted or pointer-accessible through clipping ancestors. The 119 passing component tests do not refute the reproduced defects.

## Independent verification

Read `backlog task view TASK-12984.3 --plain`, `progress.md`, `task-1-report.md`, and the full implementation diff.

Ran from `apps/packages/ui`:

```text
bunx vitest run src/components/Common/PromptAssist/__tests__/PromptAssistMenu.test.tsx src/components/Chat/composer/__tests__/PromptAssistComposerAction.test.tsx src/components/Sidepanel/Chat/__tests__/SidepanelComposerControlArea.prompt-assist.test.tsx src/components/Option/Playground/__tests__/ComposerToolbar.test.tsx --maxWorkers=1 --no-file-parallelism
4 files passed; 119 tests passed.
```

Ran a temporary `/tmp/tldw-placement-review.cjs` diagnostic against the existing packaged extension. It loads the unchanged E2E helper prelude in memory, uses the existing local mock server and extension launcher, and inspects V3/V5 at 390 x 844 plus options chat at 640 x 844. No production or test source was edited. It measures actual DOM rectangles and `elementFromPoint`, attempts a real pointer trial click, applies through keyboard, and measures resulting feedback/drawer geometry. Findings above reproduce on the current package. The first launch required sandbox escalation for a local listening port; an intermediate diagnostic needed a scoped status locator because the drawer and inline feedback share status text. Final diagnostic exited 0 after recording the expected failing pointer trial.

The implementation report's full WebUI/extension suites, compile, lint, formatting, and axe results were inspected, not independently rerun. The targeted browser diagnostic did not start a WebUI dev server. Bandit is not applicable to this review-only Markdown artifact or the TypeScript-only implementation scope.

## Safety and handoff

All diagnostic browser contexts and local mock servers closed in `finally` blocks. Process inspection after completion found no review diagnostic, task-port 18091 server, or matching task browser process. An unrelated existing Next dev server on port 8093 was left untouched. Main checkout status contains only the two previously reported untracked TTS spec/task files; no tracked changes. The assigned worktree was clean before adding this report. Only this report is committed by the reviewer. TASK-12984.3 should be reopened by the coordinator until the three findings are fixed and reviewed.
