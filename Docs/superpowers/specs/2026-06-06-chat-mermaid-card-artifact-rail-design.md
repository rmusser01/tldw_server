# Chat Mermaid Card Artifact Rail Design

## Metadata

- Task: TASK-2264
- Status: Draft for review
- Target area: `apps/packages/ui/src/components/Common`, `apps/packages/ui/src/components/Option/Playground`, `apps/packages/ui/src/components/Sidepanel/Chat`
- Related work:
  - PR #2268: assistant-only Mermaid rendering in chat markdown
  - TASK-495: OpenUI dynamic chat rendering
  - `Docs/superpowers/specs/2026-06-01-openui-dynamic-chat-rendering-design.md`

## Problem

Assistant chat messages can now render fenced `mermaid` diagrams inline, but diagrams remain trapped inside the transcript. The user can preview, download SVG, or copy source from the inline block, but cannot open a diagram into the existing chat artifact/card rail or pin it while reading the surrounding conversation.

The repository already has a lightweight chat artifact panel used by code blocks and tables. It supports `kind: "diagram"` and renders diagram artifacts with the shared Mermaid renderer. Mermaid chat blocks should use this existing rail instead of introducing a separate Mermaid-specific card surface.

## Goals

- Add an explicit "open as card" affordance to assistant Mermaid diagram blocks in `/chat`.
- Reuse the existing chat artifact rail and `ArtifactItem.kind === "diagram"` path.
- Keep Mermaid source as the canonical artifact payload.
- Keep inline assistant rendering unchanged.
- Keep user messages unchanged.
- Keep assistant-only gating from PR #2268 intact across the markdown surfaces that already opt in.
- Preserve the current Mermaid preview, copy, download, and error fallback behavior.
- Make the implementation compatible with later shared card support for Dynamic UI/OpenUI without expanding the artifact model prematurely.

## Non-Goals

- No backend persistence for Mermaid chat cards in the first version.
- No server-side Mermaid rendering.
- No generated SVG persistence as canonical data.
- No Mermaid source editor in the artifact panel.
- No card affordance for user-authored messages.
- No automatic conversion of every diagram into a persistent workspace artifact.
- No broad redesign of the chat artifact rail.

## Existing Constraints

- `Markdown` only renders `MermaidDiagramBlock` when `enableMermaidDiagrams` is true and the code fence is a closed `mermaid` fence.
- `Message` and `MessageContent` only enable Mermaid rendering for assistant-facing paths.
- `Markdown` is shared by multiple assistant-facing surfaces, including main chat, QuickChat, and reasoning blocks. Not every caller mounts `ArtifactsPanel`.
- `MermaidDiagramBlock` currently owns preview, copy-source, download-SVG, and render-error UI.
- `CodeBlock` and `TableBlock` directly use `useArtifactsStore` to open chat artifacts.
- `useArtifactsStore` is a transient client-side rail store with `active`, `history`, `isOpen`, `isPinned`, and `openArtifact`.
- `ArtifactsPanel` already renders `active.kind === "diagram"` with `Mermaid` and already supports pin, close, copy, download, and jump-to-source.

## Product Requirements

### Inline Block Actions

Assistant Mermaid blocks should expose a card action beside the existing preview, download, and copy actions.

- Label: `View`
- Tooltip: `View diagram`
- Icon: use a lucide icon already consistent with artifact opening, preferably `ExpandIcon` or `PanelRightOpenIcon`.
- The action opens the existing artifacts panel with a diagram artifact.
- The action is available only when the caller explicitly opts the markdown surface into Mermaid artifact actions.
- The default for shared markdown surfaces is no artifact action.
- Main `/chat` assistant response content should opt in because it mounts the artifact rail.
- QuickChat and other assistant-facing surfaces should keep Mermaid rendering without a card action unless they also mount the artifact rail.
- The action should not appear in user messages because user messages continue to render as plain text.

### Artifact Shape

The card should use the existing `ArtifactItem` shape:

```ts
{
  id: string
  title: string
  content: string
  language: "mermaid"
  kind: "diagram"
  lineCount: number
}
```

Recommended fields:

- `id`: stable deterministic id based on `mermaid`, source surface or message context, block index, and source hash.
- `title`: `Mermaid diagram` for a single diagram; `Mermaid diagram N` when a block index is available.
- `content`: raw Mermaid source.
- `language`: `mermaid`.
- `kind`: `diagram`.
- `lineCount`: source line count.

The implementation should not extend `ArtifactItem` until a concrete need appears. Source message lineage is desirable later, but the current rail only needs source jump anchoring by `artifact.id`.

The id must not be based on source hash and block index alone. Two messages can contain the same diagram in the same code-block position, and duplicate DOM ids would cause `Jump to source` to scroll to the wrong message.

### Source Anchoring

The inline Mermaid block should set:

- `id="artifact-origin-${artifactId}"`
- `data-artifact-origin={artifactId}`

This keeps `ArtifactsPanel.handleJumpToSource` working the same way it does for code and table artifacts.

The `artifactId` must be shared by:

- the inline origin element
- the artifact item opened by the `View` action
- any auto-open behavior, if added later

When the caller has a saved message id, the artifact id should include it. When no saved message id exists, use a per-render context id from the markdown/component tree so that jump-to-source remains correct for the current session.

### Artifact Panel Rendering

The first implementation should reuse the existing `ArtifactsPanel` diagram branch:

```tsx
active.kind === "diagram" ? <Mermaid code={active.content} ... /> : ...
```

No panel renderer registry is needed for Mermaid v1. The existing branch is already the minimal shared renderer boundary because both inline and artifact views use `Mermaid`.

### Dynamic UI Compatibility

Dynamic UI/OpenUI should not be bundled into this Mermaid implementation. The only design alignment needed now is to avoid coupling `MermaidDiagramBlock` too tightly to chat if that would block later cards.

Preferred implementation:

- Add `enableMermaidArtifactActions?: boolean` to `Markdown`, defaulting to `false`.
- Add `artifactContextId?: string` to `Markdown` so main chat can pass the saved message id when available.
- Pass the artifact-action gate and context id to `MermaidDiagramBlock`.
- Add a small artifact-opening helper local to `MermaidDiagramBlock`, mirroring `CodeBlock`, but render the action only when the gate is true.
- Keep the action disabled by default for shared markdown callers.
- Do not add Dynamic UI concepts to Mermaid types.

Later, if OpenUI artifacts are added, they can use the same rail and a separate `kind` or renderer-specific payload after the artifact model has a real persistence requirement.

## UX Requirements

- The inline block header remains visually similar to the merged Mermaid implementation.
- Card action should not crowd mobile layouts; icon-only with tooltip is acceptable in narrow headers.
- The artifact panel title should be readable and short.
- Clicking `View` should open the panel immediately.
- Pinning remains the panel's responsibility.
- Jump-to-source should scroll back to the diagram block.
- Invalid Mermaid source should behave consistently:
  - inline block shows the existing raw-source fallback
  - artifact panel attempts the same shared `Mermaid` render path
  - errors should not break the transcript or panel shell

## Accessibility Requirements

- The card action must have an aria-label, for example `View Mermaid diagram`.
- The artifact origin wrapper should keep the existing `aria-labelledby` relationship in `MermaidDiagramBlock`.
- The action should be keyboard reachable.
- Existing preview, copy, and download labels should remain unchanged.

## Security Requirements

- Raw Mermaid source remains the canonical payload.
- Do not store or execute arbitrary HTML.
- Do not add backend calls for this feature.
- Continue to rely on the shared `Mermaid` renderer's current safe rendering behavior.
- Do not render Mermaid for user messages.

## Implementation Plan

### Stage 1: Artifact Helper And Inline Action

Add deterministic artifact metadata construction to `MermaidDiagramBlock`.

Acceptance criteria:

- `MermaidDiagramBlock` renders a `View diagram` action only when artifact actions are explicitly enabled.
- Clicking it opens `useArtifactsStore.openArtifact` with `kind: "diagram"` and `language: "mermaid"`.
- The artifact source equals the original Mermaid source.
- The artifact id includes context plus source hash, and the block has an artifact origin id matching the opened artifact id.

### Stage 2: Artifact Panel Fit And Error Path

Verify the existing diagram branch in `ArtifactsPanel` handles chat Mermaid cards well.

Acceptance criteria:

- A Mermaid artifact renders in the panel.
- Copy/download use raw Mermaid source.
- Jump-to-source returns to the inline Mermaid block.
- Invalid Mermaid source does not crash the panel.

### Stage 3: Assistant-Only Surface Guardrails

Preserve the user-message boundary from PR #2268.

Acceptance criteria:

- Main `/chat` assistant messages with both Mermaid rendering and artifact actions enabled show the card action.
- User messages do not render Mermaid blocks and do not show card actions.
- QuickChat, reasoning blocks, sidepanel fallback, and workspace fallback surfaces remain unchanged unless they explicitly opt into artifact actions and mount the artifact rail.

### Stage 4: Tests And Verification

Add focused frontend tests.

Acceptance criteria:

- `MermaidDiagramBlock` test covers artifact action payload and origin id.
- `Markdown.mermaid` test covers continued source/block index propagation and default-disabled artifact actions.
- `Message.mermaid-rendering` test covers assistant-only behavior and main-chat artifact-action opt-in.
- `QuickChatMessage.mermaid` test covers Mermaid rendering without artifact-action opt-in.
- `ArtifactsPanel` test covers `kind: "diagram"` rendering and jump-to-source if not already covered.
- Run the targeted Vitest files touched by the implementation.

## Recommended Test Targets

- `apps/packages/ui/src/components/Common/__tests__/MermaidDiagramBlock.test.tsx`
- `apps/packages/ui/src/components/Common/__tests__/Markdown.mermaid.test.tsx`
- `apps/packages/ui/src/components/Common/Playground/__tests__/Message.mermaid-rendering.test.tsx`
- `apps/packages/ui/src/components/Common/QuickChatHelper/__tests__/QuickChatMessage.mermaid.test.tsx`
- `apps/packages/ui/src/components/Sidepanel/Chat/__tests__/ArtifactsPanel.*.test.tsx` or a new focused panel test if none exists

## Risks

- `blockIndex` and source hash alone are not stable enough for source jump ids if two messages contain identical diagram positions. Include a caller-provided message or context id in the artifact id.
- The current artifact store is global and transient. This is acceptable for `/chat` cards, but should not be described as durable persistence.
- Because `Markdown` is shared, adding the store action directly to every Mermaid block would expose dead artifact actions in surfaces without `ArtifactsPanel`. Keep the action opt-in.
- Auto-open behavior could be surprising for diagrams generated in normal assistant responses. Do not add Mermaid auto-open in v1.
- The artifact panel currently renders diagrams with the base `Mermaid` component, not `MermaidDiagramBlock`, so preview/download behavior differs between inline and panel. This is acceptable because the panel already has copy/download controls for source; SVG preview/download can remain an inline-only enhancement for v1.

## Open Questions

- Should the button text be icon-only or icon plus `View` on desktop? Recommendation: match `TableBlock` and use icon plus `View` where space allows.
- Should the panel title include the source message position? Recommendation: use `Mermaid diagram N` for now and avoid message metadata until the artifact model is expanded.

## Definition Of Done

- Mermaid cards open in the existing artifact rail from assistant chat blocks.
- Inline Mermaid rendering behavior is otherwise unchanged.
- User messages remain unchanged.
- Tests cover artifact payload, source anchoring, and assistant-only gating.
- Targeted frontend tests pass.
- Bandit is documented as not applicable if only frontend files are touched.
