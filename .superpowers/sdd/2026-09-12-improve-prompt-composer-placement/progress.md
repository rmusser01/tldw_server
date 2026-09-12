# SDD ledger — Backlog task: TASK-12984.3

Bounded design approved in chat: remove the standalone/wide Prompt Assist row; render one compact sparkle action immediately beside Send in WebUI `/chat` and recipe-capable extension sidepanel/options chat; open the menu upward; anchor Applied/View changes/Undo feedback without permanent toolbar reflow; preserve all existing Prompt Assist behavior, focus restoration, and Quick Chat exclusion.

Ruling: use `TASK-12984.3` as the requirements and three-stage execution record instead of creating a separate plan document, because the brainstorming workflow classified this as a bounded modification to existing composer flows and explicitly forbids a plan document for bounded work. — Cost if wrong: a cross-surface layout change may prove larger than bounded scope and need an architectural plan before further implementation.

Preflight interaction map:

| Producer | Consumer | Contract | Finding |
|---|---|---|---|
| `PromptAssistComposerAction` | `PromptAssistMenu` | One trigger, drawer lifecycle, applied/recipe Undo feedback | Preserve state and exact Undo; add compact/upward presentation only. |
| WebUI `PlaygroundForm` inline composer controls | `ComposerToolbar` | Send is external to toolbar while Prompt Assist currently occupies a standalone toolbar row | Move Prompt Assist beside external Send and remove toolbar-owned standalone row. |
| Extension `SidepanelForm` action cluster | composer variants via `SidepanelComposerControlArea` | One shared bottom-bar node across legacy/v1/v3/v5 | Place Prompt Assist in the shared send-action cluster so every recipe-capable variant gets exactly one action. |
| Prompt Assist popup/feedback | viewport and composer layout | Menu/feedback must remain visible without permanent reflow | Open upward and anchor transient feedback; test desktop/mobile geometry and focus. |
| Quick Chat pop-out | `QuickChatInput` | No recipe-capable adapter | Remains excluded and must not be relabeled. |

Task 1: complete. TASK-12984.3 moves the single compact 44 px Improve action immediately before Send in WebUI `/chat`, extension `/options.html#/chat`, and sidepanel legacy/v1/v3/v5; the upward menu is viewport-clamped, feedback is layout-neutral above the action, and Quick Chat remains excluded. Evidence: 119 focused Vitest tests, 12 WebUI Playwright tests, 13 self-contained packaged-extension Playwright tests, extension compile, scoped ESLint/Prettier, and `git diff --check` passed. Full details: `task-1-report.md`.
