# Workspace Playground UX Remediation Design

Backlog task: TASK-407

## Problem

`/workspace-playground` is the shared WebUI and extension research workspace surface. The current implementation exposes several usability failures in the core workspace loop:

- Sources and Studio can be collapsed, but the visible restore controls are easy to lose because they only live in the top workspace header.
- The chat composer can sit below the initial viewport, forcing the user to scroll before they can ask a question.
- The left pane's source intake action is labeled "Add" even though the user-facing workflow is "Add Sources".
- The Add Sources modal's My Media tab can show no media even when the user has many existing media records.
- The chat Model select can appear present but be unselectable or ineffective.

The remediation should stay scoped to `/workspace-playground` and the shared WorkspacePlayground component stack used by both the WebUI and extension options route.

## Goals

- Keep the workspace as a bounded app surface where the chat composer is immediately reachable.
- Make collapsed side panes recoverable from persistent controls inside the workspace, not only from the header.
- Make source intake obvious in the left pane and reliable in the Add Sources modal.
- Make the chat model picker use the established chat-model data path and remain interactive.
- Preserve the existing three-pane structure, source list capabilities, chat behavior, Studio behavior, and modal tabs outside the broken My Media path.

## Non-Goals

- Do not redesign the full Workspace Playground product surface.
- Do not change backend media, RAG, or model APIs unless implementation proves the frontend client is calling the wrong documented contract.
- Do not migrate Studio runtime settings, generated output workflows, source organization, or global navigation as part of this slice.
- Do not alter `/chat`, `/knowledge`, or other workspace-adjacent pages except for tests or shared utilities that are directly required by this route.

## Current Code Evidence

- `apps/packages/ui/src/components/Option/WorkspacePlayground/index.tsx` owns the desktop three-pane shell, mobile tabs, pane collapse state, route-level content composition, and focus helpers.
- `apps/packages/ui/src/routes/option-workspace-playground.tsx` wraps WorkspacePlayground for the shared options route with `OptionLayout` and a plain `w-full` div.
- `apps/tldw-frontend/extension/routes/option-workspace-playground.tsx` wraps the same component in `PageShell`.
- `apps/packages/ui/src/components/Option/WorkspacePlayground/SourcesPane/index.tsx` owns the left pane header, source filters, source folders, source list, and Add Source modal launcher.
- `apps/packages/ui/src/components/Option/WorkspacePlayground/SourcesPane/AddSourceModal.tsx` owns Upload, My Media, URL, Paste, and Search Server intake tabs. The My Media tab currently accepts only `media` and `results` response fields from media list/search responses.
- `apps/packages/ui/src/components/Option/WorkspacePlayground/ChatPane/index.tsx` owns the chat transcript, sticky composer, chat controls, and current raw `tldwClient.getModels()` model-picker loading path.
- `apps/tldw-frontend/e2e/utils/page-objects/WorkspacePlaygroundPage.ts` and `apps/tldw-frontend/e2e/workflows/workspace-playground.spec.ts` already cover the route, pane collapse/restore, Add Sources modal, and broad workflow behavior.

## Design

### 1. Layout Shell Hardening

The workspace should behave like a bounded application surface. The route wrappers and component root should form a complete height chain:

- route wrapper: `flex min-h-0 flex-1` or equivalent
- WorkspacePlayground root: `h-full min-h-0 overflow-hidden`
- ChatPane: transcript owns internal scroll, composer remains visible

For desktop and extension options, the user should not need to scroll the page itself to reach the composer. Page scrolling can still exist where the outer application shell requires it, but the workspace should not depend on page scroll for its primary input.

Collapsed Sources and Studio panes should leave persistent restore controls in the workspace body. Header restore shortcuts can remain, but the body-level controls are the primary recovery affordance:

- collapsed Sources: slim left-edge rail button labeled "Show sources"
- collapsed Studio: slim right-edge rail button labeled "Show studio"
- controls should be keyboard reachable and have clear `aria-label`s
- keyboard shortcuts and `focusWorkspacePane()` behavior should continue to restore the same panes

The rail buttons should not overlap the chat composer or transcript content. They may sit at the workspace edge inside the main shell and use compact icon-plus-label treatment on desktop, with responsive hiding or drawer behavior on mobile.

### 2. Sources Pane Intake

The left pane header primary action should say "Add Sources" and keep the existing plus icon. It should continue to open the existing Add Sources modal.

The My Media tab in Add Sources must normalize the same media-library response shapes used elsewhere in the app:

- `media`
- `results`
- `items`
- `data`

It should also preserve pagination totals from common server fields:

- `total_count`
- `total`
- `count`
- `results_count`
- `pagination.total`

Already-added media should still be filtered out after response normalization. Empty and error states should be distinguishable:

- "No media found" when the user truly has no matching media.
- "All visible media are already in this workspace" when response items exist but filtering removes them.
- A load error message when the media-library request fails.

Large libraries should continue to use page-based loading and the existing Load More control.

Upload, URL, Paste, and Search Server tabs remain functionally unchanged.

### 3. Chat Composer Visibility

The chat input should be visible when `/workspace-playground` first loads on desktop-sized WebUI and extension options viewports. The composer should not rely on page-level scroll. The transcript should scroll independently, and the empty state should not push the composer below the viewport.

The implementation should avoid introducing fixed-position global UI that can cover modals, browser extension chrome, or the app header. The preferred contract is layout containment rather than a viewport-fixed composer.

### 4. Model Picker

The workspace chat model picker should use the shared chat-model service rather than raw `tldwClient.getModels()` parsing in ChatPane. It should load chat-capable models only and keep a stable "Auto" option.

Expected behavior:

- selecting a model updates the shared selected-model state
- "Auto" remains selectable even when model loading fails
- while loading, the UI communicates loading without making the control look permanently broken
- when no chat models are configured, the control degrades to Auto plus a clear empty state or hint
- the control should be usable in both WebUI and extension options route

The visual placement stays in the chat controls row for this slice. Broader runtime-control redesign is out of scope.

## Data Flow

### Pane Collapse

1. User clicks a pane's hide control.
2. Workspace store updates `leftPaneCollapsed` or `rightPaneCollapsed`.
3. The pane unmounts or hides as it does today.
4. Workspace body renders a persistent restore rail for the collapsed pane.
5. User clicks restore rail, header shortcut, or keyboard shortcut.
6. Workspace store resets the collapsed state and focus moves into the restored pane.

### My Media Intake

1. User opens Add Sources and selects My Media.
2. My Media requests either `listMedia` or `searchMedia`.
3. Response is normalized through a shared helper that extracts items and total count from supported shapes.
4. Existing workspace source media IDs are filtered out.
5. The tab renders items, load-more state, all-added state, true-empty state, or error state.
6. User selects media and clicks Add Selected.
7. Existing workspace source creation behavior runs unchanged.

### Model Selection

1. ChatPane requests chat model options through the shared model service.
2. The service returns chat-capable model options only.
3. ChatPane renders Auto plus model options.
4. User selects an option.
5. ChatPane calls the existing shared selected-model setter.
6. Existing chat submission and Studio generation paths continue reading the selected model from shared state.

## Error Handling

- Pane restore rails should render from local collapse state and not depend on network or store hydration beyond the current workspace store.
- My Media request failures should set the modal error state or local tab error state and should not render the same copy as a true empty library.
- My Media normalization should tolerate unknown response shapes by treating them as empty and preserving a diagnostic-friendly error path in tests.
- Model loading failures should not disable chat input. They should leave Auto available and optionally show a small inline hint.
- Existing duplicate-source warnings and workspace-tag keyword updates should remain best-effort and unchanged.

## Accessibility

- Restore rails need descriptive `aria-label`s, visible focus rings, and keyboard activation.
- "Add Sources" must be discoverable as a button label, not only as an icon tooltip.
- My Media list rows should preserve checkbox semantics and avoid double toggles when clicking the checkbox inside a clickable row.
- The model picker should have a stable accessible name and should not be nested in a way that interferes with native select activation.
- The chat transcript should remain a `role="log"` and the composer should remain in the normal tab order.

## Testing And Verification

Component and unit coverage:

- WorkspacePlayground renders restore controls when Sources or Studio is collapsed.
- Restore controls reopen panes and return focus to a useful target.
- Workspace route wrappers provide the expected bounded-height classes.
- SourcesPane header uses the "Add Sources" label.
- AddSourceModal My Media normalizes `media`, `results`, `items`, and `data` response shapes.
- My Media distinguishes true empty, all-added, and load-error states.
- ChatPane model picker uses chat model options, keeps Auto available, and updates selected-model state.

Browser or E2E coverage:

- Desktop WebUI `/workspace-playground`: composer visible without page scrolling.
- Extension options `#/workspace-playground`: composer visible without page scrolling.
- Sources can be collapsed and restored from a visible body-level restore control.
- Studio can be collapsed and restored from a visible body-level restore control.
- Add Sources opens My Media and displays existing seeded media.
- Model picker can be opened and changed.

Manual verification:

- Run the relevant unit/component tests in `apps/packages/ui`.
- Run targeted workspace-playground Playwright coverage for WebUI and extension options where practical.
- Use browser observation for the desktop route after implementation because visual containment and composer visibility are central to the bug.

Security:

- This design changes frontend layout and client-side normalization only. Bandit is not applicable unless implementation later touches Python code.

## Open Risks

- Existing app shell scroll behavior differs between the WebUI and extension options route. The implementation should verify both route wrappers before assuming one height fix covers both.
- If `listMedia` and `searchMedia` return different shapes by backend version, tests should include both.
- If the current native select is blocked by a parent label or overlay in the browser, the implementation may need to replace the label wrapper with a plain field container.
- Large My Media libraries should avoid rendering hundreds of rows at once beyond the existing paginated Load More behavior.

## Approval Status

Approved design direction:

- Use a constrained shell hardening pass rather than a minimal patch or broad redesign.
- Include small related polish when it prevents recurrence.
- Keep visual mockups optional and use them only if they materially help.
- Include the My Media source-intake bug in the Sources Pane remediation.
