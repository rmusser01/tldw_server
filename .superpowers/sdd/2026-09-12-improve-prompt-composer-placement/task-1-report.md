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
