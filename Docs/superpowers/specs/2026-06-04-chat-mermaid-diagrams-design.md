# Chat Mermaid Diagrams PRD

**Date:** 2026-06-04
**Surface:** Assistant-facing chat markdown surfaces in the WebUI and extension UI package
**Status:** Approved in-session for PRD drafting
**Backlog:** TASK-510
**Upstream reference:** [ggml-org/llama.cpp#24032](https://github.com/ggml-org/llama.cpp/pull/24032)

---

## Goal

Render Mermaid diagrams in assistant chat responses automatically when a model emits fenced Mermaid markdown, while keeping user messages plain text and preserving a raw-code fallback.

The first release should make diagrams readable inline in chat, provide a larger viewer for inspection, and avoid surprising behavior in non-chat markdown consumers.

## Product Decision

Add Mermaid rendering as an opt-in capability of the shared markdown renderer, then enable that capability only from assistant-facing markdown surfaces.

This is preferable to wiring Mermaid into every chat surface separately because the WebUI already centralizes assistant markdown through [Markdown.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/Markdown.tsx:66). It is also preferable to putting the behavior only in [CodeBlock.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/CodeBlock.tsx:52), because user messages and unrelated markdown previews should not start rendering diagrams unless their caller opts in.

The first release should:

- render fenced `mermaid` blocks inline by default in assistant messages;
- provide copy source, open larger, and download SVG actions;
- support a zoomable and pannable expanded viewer;
- default the feature on, with a user setting to render Mermaid fences as code blocks instead;
- preserve raw source on failures;
- avoid rendering incomplete Mermaid fences while an assistant response is still streaming.

## Upstream PR Lessons

The llama.cpp PR merged on 2026-06-03 and added Mermaid rendering to its chat UI. The useful implementation ideas are:

- detect Mermaid code fences before generic code-block enhancement;
- render Mermaid client-side with a lazy `import("mermaid")`;
- wrap Mermaid blocks with the same header/action model as code blocks;
- copy Mermaid source, not generated SVG;
- open rendered SVG in a larger preview dialog with zoom, pan, reset, and SVG download;
- mark Mermaid nodes immediately before async rendering to avoid duplicate renders during streaming;
- treat incomplete Mermaid fences as a special streaming state instead of repeatedly trying to render invalid syntax.

The PR comments also surfaced two follow-up risks that should shape this project:

- raw SVG block rendering should stay out of the first release and require separate security review;
- a 2026-06-04 upstream comment reported a build-resolution failure for `mermaid`, so this project needs explicit build verification even though [package.json](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/package.json:111) already includes `mermaid`.

## Current Local Context

The local codebase already has most building blocks:

- [Markdown.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/Markdown.tsx:251) uses `react-markdown` and owns fenced code block rendering.
- [Markdown.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/Markdown.tsx:270) extracts the code fence language and source text before delegating to compact, GitHub, or default code block renderers.
- [Mermaid.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/Mermaid.tsx:41) already renders Mermaid diagrams.
- [Mermaid.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/Mermaid.tsx:107) already lazy-loads `mermaid`.
- [Mermaid.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/Mermaid.tsx:111) initializes Mermaid with `securityLevel: "strict"`.
- [CodeBlock.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/CodeBlock.tsx:73) already recognizes Mermaid-like diagram languages for artifact handling.
- [QuickChatMessage.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/QuickChatHelper/QuickChatMessage.tsx:44) keeps user messages plain and renders assistant output through `Markdown`.
- [MessageContent.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/Playground/MessageContent.tsx:182) already renders active main-chat streams as plain text, which naturally prevents partial Mermaid rendering in the main playground.
- [ArtifactModalContent.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/ArtifactModalContent.tsx:31) has an existing Mermaid artifact viewer with zoom and SVG export behavior.
- [ArtifactsPanel.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Sidepanel/Chat/ArtifactsPanel.tsx:303) already renders diagram artifacts through `Mermaid`.

## Scope

### In Scope

- Assistant responses rendered through shared Markdown in:
  - main `/chat` playground messages;
  - model playground messages;
  - chat workspace messages;
  - research workspace chat messages;
  - document/workspace chat messages that use `PlaygroundMessage`;
  - quick chat and quick chat popout assistant messages;
  - extension sidepanel assistant messages that use the same shared components.
- Fenced code blocks with language exactly normalized to `mermaid`.
- Inline rendered diagram with actions.
- Expanded diagram dialog with zoom, pan, reset, copy source, and download SVG.
- User setting: `Render Mermaid diagrams`, default enabled.
- Raw code fallback when disabled, invalid, unavailable, or unsupported.

### Out Of Scope

- User-message Mermaid rendering.
- Standalone tool-result Mermaid rendering.
- Raw SVG block rendering.
- PNG export.
- Live token-by-token diagram rendering.
- Server-side Mermaid rendering.
- Backend API changes.
- Mermaid syntax generation prompts.
- Broad rendering changes for documentation, flashcards, notes, review, or other non-chat markdown consumers unless those consumers explicitly opt in later.
- Graphviz/DOT rendering.

## Functional Requirements

### FR1: Opt-In Shared Markdown API

Add a prop to `Markdown`, such as:

```ts
enableMermaidDiagrams?: boolean
```

Default it to `false` so existing non-chat consumers are unchanged.

Assistant-facing call sites pass `true` when:

- the message role is assistant output intended to be rendered as assistant markdown; and
- the user setting is enabled.

For v1, assistant output includes assistant-role messages and assistant greeting messages that already use the assistant markdown path. Standalone tool-result blocks and system messages do not opt in unless their content is already embedded into an assistant message string rendered by that assistant path.

User-message paths continue to render as plain text and do not pass this prop.

### FR2: Mermaid Fence Detection

When `enableMermaidDiagrams` is true, `Markdown` detects fenced code blocks whose normalized language is `mermaid`.

For those blocks:

- render `MermaidDiagramBlock`;
- pass the original source text without the closing newline stripped beyond the current code block behavior;
- preserve the block index or stable key so actions and rendered state do not shift between renders.

When disabled:

- route the block through existing code block behavior.

### FR3: Inline Diagram Block

`MermaidDiagramBlock` renders:

- a header label: `mermaid`;
- a rendered diagram area;
- copy Mermaid source action;
- open expanded viewer action;
- download SVG action after SVG exists;
- raw source fallback when render fails.

The block should visually align with existing code block and artifact chrome. It should not use large marketing-style cards or introduce a separate visual language.

### FR4: Expanded Viewer

`MermaidPreviewDialog` renders the generated SVG in a larger dialog.

Controls:

- zoom in;
- zoom out;
- reset;
- pan/drag;
- copy source;
- download SVG;
- close.

The viewer receives source and/or generated SVG from the inline block. It should not need to reparse the whole assistant message.

### FR5: Renderer Hardening

Reuse `Common/Mermaid.tsx` as the rendering core, with hardening as needed:

- continue dynamic import of `mermaid`;
- keep `securityLevel: "strict"`;
- expose render success, render error, and generated SVG to parent blocks;
- avoid stale async writes after unmount or source changes;
- re-render on light/dark theme changes;
- avoid repeated concurrent renders for the same source/theme pair;
- provide a raw-source fallback if Mermaid fails to load or render.

### FR6: Settings

Add a user-facing setting:

```ts
renderMermaidDiagrams: boolean
```

Default: `true`.

Recommended placement: chat or markdown display settings near existing rich text, code theme, and external image settings.

Behavior:

- `true`: assistant Mermaid fences render as diagrams.
- `false`: assistant Mermaid fences render through existing code block behavior.

The setting must be respected in WebUI and extension surfaces that share the UI package.

### FR7: Rich Text Compatibility

[Markdown.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/Markdown.tsx:234) has an ST-compatible rich-text path that bypasses `ReactMarkdown` and writes an HTML string.

When Mermaid rendering is enabled and the message contains a Mermaid fence, the renderer must use the React component path for that message or otherwise ensure the Mermaid block is rendered by the safe component pipeline. Mermaid diagrams must not be implemented by injecting raw Mermaid output through the ST-compatible HTML string path.

### FR8: Streaming Behavior

Main playground messages already render active assistant streams as plain text through [MessageContent.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/Playground/MessageContent.tsx:182). Keep that behavior.

For assistant surfaces that render Markdown during streaming:

- do not render Mermaid for an unclosed fenced block;
- show raw fenced text or a small disabled placeholder until the fence closes;
- avoid repeated render attempts on partial syntax.

The first release does not need live diagram rendering while tokens are arriving.

## Data Flow

1. Assistant response text reaches an assistant message component.
2. The component decides whether the message is assistant-role markdown, including existing assistant greeting paths.
3. The component reads the `renderMermaidDiagrams` setting.
4. The component passes `enableMermaidDiagrams={true}` to `Markdown` only when both conditions are true.
5. `Markdown` parses markdown with the existing `ReactMarkdown` pipeline.
6. The `pre` renderer detects `language-mermaid`.
7. If enabled, `Markdown` renders `MermaidDiagramBlock`.
8. `MermaidDiagramBlock` calls `Mermaid` to render SVG.
9. The rendered SVG supports inline display, expanded preview, and SVG download.
10. If any step fails, the user sees raw Mermaid source and can copy it.

## Component Boundaries

### Markdown

Purpose: markdown parsing and routing from Markdown syntax to React components.

It should know:

- whether Mermaid diagrams are enabled;
- how to detect Mermaid code fences;
- which component renders Mermaid blocks.

It should not own:

- zoom/pan state;
- Mermaid library initialization details beyond delegating to `Mermaid`;
- SVG download logic.

### MermaidDiagramBlock

Purpose: one complete inline diagram block.

It should own:

- block chrome;
- action buttons;
- rendered SVG state from the renderer;
- raw-source fallback;
- opening the preview dialog.

It should depend on:

- `Mermaid`;
- shared button/icon patterns;
- clipboard/download helpers.

### Mermaid

Purpose: safely convert Mermaid source into SVG.

It should own:

- dynamic import;
- Mermaid initialization;
- theme resolution;
- strict security;
- render cancellation and error reporting.

It should not own:

- chat-specific UI;
- copy/download controls;
- settings.

### MermaidPreviewDialog

Purpose: inspect a rendered diagram outside the chat bubble constraints.

It should own:

- zoom state;
- pan state;
- reset;
- dialog layout;
- SVG download/copy actions exposed from the block.

## Error Handling

Rendering failure should be local to the diagram block. A bad Mermaid diagram must not break the whole assistant message.

The block should display:

- concise error text, for example `Unable to render Mermaid diagram.`;
- an expandable or visible raw source block;
- copy-source action if clipboard is available.

Errors that should fall back:

- invalid Mermaid syntax;
- Mermaid module import failure;
- render timeout or cancellation;
- empty source;
- browser APIs unavailable.

Do not silently hide invalid diagrams.

## Security Requirements

- Mermaid must use `securityLevel: "strict"`.
- Do not support raw SVG blocks in this release.
- Do not use `dangerouslySetInnerHTML` for user/model-provided source except where the existing Mermaid renderer must insert Mermaid-generated SVG after strict-mode rendering.
- Preserve existing URL/image safeguards in Markdown.
- SVG download should serialize only Mermaid-generated SVG, not arbitrary raw SVG supplied by the model.
- Any future support for raw SVG/XML preview must go through a separate security review.

## Accessibility Requirements

- Inline diagram container uses `role="img"` and an accessible label such as `Mermaid diagram`.
- Action buttons have accessible names.
- The expanded dialog has a title and focus management.
- Keyboard users can open, close, zoom, reset, copy, and download.
- Raw source remains available for screen reader users and for diagrams that are visually dense.
- Color/theme handling must keep text and edges legible in light and dark modes.

## Performance Requirements

- Mermaid is dynamically imported only when a rendered Mermaid block exists.
- Non-Mermaid markdown should not load Mermaid.
- Rendering should be local to each diagram block and cancel stale async results.
- Avoid repeated concurrent renders for the same source/theme.
- Build verification must confirm `mermaid` resolves in the frontend package.
- Bundle analysis or build output should be checked to ensure Mermaid is not statically pulled into the initial chat route chunk where tooling makes that visible.

## Acceptance Criteria

- Assistant Mermaid fences render inline automatically when the setting is enabled.
- Assistant Mermaid fences render as existing code blocks when the setting is disabled.
- User messages remain plain text even if they contain Mermaid fences.
- Invalid Mermaid syntax shows a local error and raw source fallback.
- Copy-source copies Mermaid syntax, not generated SVG.
- Open-expanded displays the rendered diagram with zoom, pan, reset, and download SVG.
- Active streaming in main chat does not attempt to render incomplete Mermaid fences.
- Quick chat and other assistant markdown surfaces do not render unclosed Mermaid fences while streaming.
- Light/dark theme changes re-render or restyle diagrams appropriately.
- Frontend build/compile verifies `mermaid` resolution.

## Test Plan

### Unit And Component Tests

- `Markdown` routes fenced `mermaid` code blocks to `MermaidDiagramBlock` when `enableMermaidDiagrams` is true.
- `Markdown` routes the same fence to existing code block rendering when false.
- `Markdown` does not render Mermaid from ST-compatible HTML mode when Mermaid component rendering is required.
- `MermaidDiagramBlock` shows rendered content on success.
- `MermaidDiagramBlock` shows raw-source fallback on renderer error.
- `MermaidDiagramBlock` copy action uses source text.
- `MermaidPreviewDialog` exposes zoom, reset, pan affordance, close, and download SVG action wiring.
- Assistant message tests pass `enableMermaidDiagrams`; user message tests do not.
- Tool-result and system-message tests do not pass `enableMermaidDiagrams` in v1 unless the content is embedded in an assistant message.
- Quick chat streaming test covers an incomplete Mermaid fence.

### Integration And E2E Tests

- Main chat assistant response with a Mermaid diagram renders inline after generation completes.
- User message containing a Mermaid fence remains plain text.
- Setting off renders Mermaid as code.
- Expanded viewer opens from the inline diagram and downloads an SVG.
- Build or compile command resolves Mermaid and completes.

### Security Verification

Because this is a frontend docs/PRD task, Bandit is not applicable to this PRD itself. During implementation, run the frontend tests and build checks above. If backend code is touched later, run Bandit on the touched backend scope.

## Implementation Notes For Follow-Up Planning

Suggested implementation stages:

1. Add settings contract and Markdown prop.
2. Harden `Common/Mermaid.tsx` to expose generated SVG and render status.
3. Add `MermaidDiagramBlock` and `MermaidPreviewDialog`.
4. Wire assistant-facing Markdown call sites.
5. Add tests and build verification.

Keep the implementation PR small enough to review by avoiding unrelated markdown refactors.

## Open Questions

- Should diagram artifacts and inline Mermaid blocks share one viewer component immediately, or should artifact viewer cleanup be a follow-up?
- Should the setting live in chat settings only, or also in a general markdown display settings section if such a section becomes canonical?

## Non-Goals For Follow-Up Tracking

These are useful but should not block the first PR:

- raw SVG block rendering;
- PNG export;
- real-time diagram rendering while a code fence is still streaming;
- graphviz/DOT rendering;
- Mermaid prompt templates;
- server-side diagram thumbnails.
