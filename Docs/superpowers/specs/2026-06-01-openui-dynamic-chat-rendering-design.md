# OpenUI Dynamic Chat Rendering Design

**Date:** 2026-06-01
**Surface:** WebUI `/chat`, extension sidepanel chat, shared chat workspace surfaces
**Status:** Approved in-session, pending spec review
**Backlog:** TASK-491
**Reference:** https://github.com/pewdiepie-archdaemon/odysseus/pull/151

---

## Goal

Support literal OpenUI rendering in chat while shaping the implementation as the first step toward a broader dynamic UI/artifact system.

The immediate outcome is that assistant responses can be marked as OpenUI, rendered by a shared renderer across chat surfaces, and submit structured user actions back into the normal chat flow. The long-term outcome is a renderer-neutral dynamic content contract that can later support charts, dashboards, forms, timelines, maps, richer tables, and other typed interactive artifacts without reworking each chat route.

## PR #151 Review Summary

The Odysseus PR demonstrates the right product pattern:

- an explicit OpenUI chat mode;
- OpenUI-aware prompt injection;
- assistant response metadata that marks messages as renderable OpenUI;
- inline rendering during chat, including a streaming preview path;
- generated form/action events that can become follow-up chat turns;
- theme variables that bridge generated UI into the host app.

The same idea is applicable to this repository, but the implementation should not be copied directly.

Avoid these PR-specific choices in this codebase:

- committing large generated vendor bundles instead of using the frontend package build;
- making OpenUI only a route-specific toggle;
- hardcoding an OpenAI-only generation endpoint instead of using existing provider/chat abstractions;
- relying on message text pattern detection as the durable contract;
- using a permanent global window event as the action bridge;
- designing only for WebUI without extension CSP and sidepanel constraints.

## Product Decision

Use OpenUI as the first registered renderer in a shared **Dynamic UI** layer.

This means OpenUI support is literal, but the persistence and rendering contract should not be named as if OpenUI is the only possible dynamic format. Chat messages should carry a typed dynamic UI metadata envelope. The renderer registry should route that envelope to the OpenUI adapter in v1, and later to other adapters as needed.

Recommended initial mode:

1. final-render OpenUI after the assistant response completes;
2. persist explicit dynamic metadata with the assistant message;
3. support a source fallback;
4. support validated OpenUI action submission back into chat;
5. add streaming preview only after the renderer and persistence path are stable.

## Non-Goals

- No arbitrary HTML, JavaScript, or iframe execution from model output.
- No vendored generated OpenUI runtime bundles committed into source.
- No OpenAI-only backend endpoint that bypasses existing provider abstractions.
- No broad redesign of `/chat`, the sidepanel, or chat workspace.
- No requirement to support every future dynamic artifact type in the first implementation.
- No replacement of Markdown, code blocks, tables, Mermaid, or existing artifact behavior.
- No hidden conversion of ordinary assistant messages based only on regex-like source detection.

## Implementation Anchors

The design should stay grounded in the current shared chat architecture:

- shared chat route:
  - [option-chat.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/routes/option-chat.tsx)
  - [Playground.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/Playground/Playground.tsx)
- shared message model:
  - [types.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/store/option/types.ts)
- shared message rendering:
  - [Message.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/Playground/Message.tsx)
  - [MessageContent.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Common/Playground/MessageContent.tsx)
- streaming pipeline:
  - [chatModePipeline.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/chat-modes/chatModePipeline.ts)
  - [streaming-chunks.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/utils/streaming-chunks.ts)
- saved message hydration:
  - [useServerChatLoader.ts](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/hooks/chat/useServerChatLoader.ts)
- extension sidepanel chat:
  - [sidepanel-chat.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/extension/routes/sidepanel-chat.tsx)
  - [body.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Sidepanel/Chat/body.tsx)
- chat workspace surfaces:
  - [WorkspaceChatPanel.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceChatPanel.tsx)
- artifact rail:
  - [artifacts.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/store/artifacts.tsx)
  - [ArtifactsPanel.tsx](/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/packages/ui/src/components/Sidepanel/Chat/ArtifactsPanel.tsx)

## Data Contract

Add a dynamic UI metadata envelope to assistant messages. The exact field location should follow the existing `metadataExtra` mapping so saved messages can round-trip through current chat persistence.

Conceptual shape:

```ts
metadataExtra.dynamic_ui = {
  renderer: "openui",
  version: "v1",
  source: "...",
  state: {},
  capabilities: []
}
```

Field meanings:

| Field | Meaning |
| --- | --- |
| `renderer` | Registered renderer ID. V1 supports `openui`. |
| `version` | Contract version for renderer-specific migration and compatibility checks. |
| `source` | Renderer source payload. For OpenUI, this is the OpenUI document/source. |
| `state` | Optional renderer state snapshot for form values or local UI state that should survive rerenders. |
| `capabilities` | Optional allowlist for renderer features such as forms or charting when a renderer supports sub-capabilities. |

Rendering must key from metadata first, not text sniffing. Text/source detection can be an import helper or dev convenience, but it must not be the durable persisted contract.

## Architecture

Add a shared `DynamicMessageRenderer` inside `apps/packages/ui`. This component receives a chat message, inspects `metadataExtra.dynamic_ui`, and renders through a registry.

The registry maps renderer IDs to lazy renderer adapters:

- `openui` -> OpenUI React adapter;
- future IDs -> chart/table/form/dashboard adapters.

The OpenUI adapter should wrap the OpenUI React renderer, translate host theme tokens into OpenUI theme variables, and expose a narrow action callback. It should not know about `/chat`, sidepanel routing, or workspace routing.

All chat surfaces should continue to render through the shared `PlaygroundMessage` boundary where possible. This keeps `/chat`, extension sidepanel chat, and workspace chat behavior aligned without route-specific forks.

## Components

### `DynamicMessageRenderer`

Responsibilities:

- validate the dynamic UI envelope enough to choose a renderer;
- lazy-load the requested renderer adapter;
- pass source, state, theme, and action callback to the adapter;
- show source fallback when disabled, unknown, or failed;
- avoid changing message layout for ordinary Markdown messages.

### Renderer Registry

Responsibilities:

- define supported renderer IDs;
- associate each ID with a lazy import;
- expose feature availability to UI controls;
- provide a single place to disable a renderer globally or per surface.

### OpenUI Adapter

Responsibilities:

- load OpenUI runtime through the normal frontend build;
- render OpenUI source into a bounded message surface;
- translate app theme tokens into OpenUI-compatible theme variables;
- report user actions through the shared dynamic action callback;
- degrade cleanly in extension contexts if a capability is unavailable.

### OpenUI Mode Control

Responsibilities:

- let the user opt into OpenUI output for an outgoing chat request;
- expose the mode first on `/chat`;
- make the setting portable to other shared chat surfaces when the renderer is stable;
- avoid silently forcing every chat into OpenUI mode.

The control can start as a compact toggle or segmented mode in the existing chat controls. Implementation planning should pick the exact placement based on the current `PlaygroundForm` and settings UI.

### Dynamic Action Bridge

Responsibilities:

- receive renderer action payloads;
- validate renderer ID, action ID/type, and submitted values;
- convert valid actions into a normal user chat turn;
- preserve enough provenance for debugging;
- avoid a permanent global browser event contract.

Conceptual action payload:

```ts
{
  renderer: "openui",
  messageId: "...",
  actionId: "...",
  actionType: "submit",
  values: {}
}
```

## Data Flow

1. User enables OpenUI mode for a chat request.
2. The chat send path adds OpenUI instructions through the existing chat/provider pipeline.
3. The model returns OpenUI source as the assistant response.
4. During v1 streaming, the transcript can continue showing source/plain text.
5. When the assistant response completes, the message is saved with `metadataExtra.dynamic_ui`.
6. Saved message hydration maps backend `metadata_extra` back into the frontend `Message.metadataExtra`.
7. `PlaygroundMessage` sees `metadataExtra.dynamic_ui` and delegates to `DynamicMessageRenderer`.
8. The registry loads the `openui` adapter.
9. The user interacts with the generated UI.
10. The dynamic action bridge validates the action and sends a structured follow-up as a normal user turn.

## Prompting And Provider Path

OpenUI prompting should use the existing provider/chat paths wherever possible.

The prompt layer needs:

- a clear OpenUI system or developer instruction;
- a renderer contract that tells the model to emit only OpenUI source when the mode is active;
- compatibility with current chat context, attachments, RAG/context injection, character/persona overlays, and future workspace context.

The backend should not add an OpenAI-only `/api/openui/generate` equivalent as the main path. If a future document-generation endpoint is needed, it should still use the project provider abstraction and the same dynamic UI metadata contract.

## Rendering Policy

Use metadata-driven rendering:

- `metadataExtra.dynamic_ui.renderer === "openui"` renders OpenUI;
- unknown renderers show source fallback;
- missing metadata renders normal Markdown/message content;
- disabled feature flag shows source fallback;
- render failures show a compact error plus source/details disclosure.

Streaming preview is intentionally a later enhancement. Once final rendering works, the stream pipeline can add throttled preview updates using the existing 80 ms streaming cadence rather than a route-specific character-count throttle.

## Persistence Policy

Dynamic UI metadata must survive:

- initial assistant message creation;
- optimistic frontend message updates;
- successful save;
- chat reload from server;
- sidepanel snapshots if they carry messages across surfaces;
- export/import paths when those paths include message metadata.

The renderer source should remain inspectable. A saved dynamic message should never become opaque UI with no readable source.

## Artifact Rail Relationship

Inline rendering is the primary v1 product experience because it preserves the strongest idea from PR #151: the assistant can answer with a live interface directly in conversation.

The artifacts rail should be a secondary target:

- allow an OpenUI message to be opened or pinned as an artifact;
- reuse the same renderer registry;
- keep artifact state separate from chat message source;
- avoid blocking v1 inline rendering on a full artifact model expansion.

Longer term, the artifact rail may become the home for larger dynamic dashboards while inline chat uses compact previews.

## Security And CSP

Dynamic UI rendering is a security-sensitive boundary because model output becomes interactive UI.

Required boundaries:

- no arbitrary HTML/script execution from model output;
- only registered renderer IDs are executable;
- renderer output must be bounded to the message/artifact container;
- renderer actions are untrusted input and must be validated;
- source fallback remains available for audit;
- feature flag or capability checks can disable the renderer globally, per route, or in extension contexts;
- extension CSP behavior must be tested before enabling OpenUI in the sidepanel by default.

If OpenUI requires capabilities that conflict with extension CSP, WebUI can support OpenUI first while the sidepanel shows source fallback until the extension path is safe.

## Error Handling

Fail closed and keep chat usable:

- unknown renderer -> source fallback;
- disabled feature flag -> source fallback;
- failed lazy import -> source fallback with load error;
- invalid OpenUI source -> render error plus source disclosure;
- malformed action payload -> ignore action and optionally show a non-blocking error;
- action submission failure -> keep the generated UI visible and report the send error through existing chat error paths.

OpenUI rendering failures must not corrupt the chat transcript, reset the conversation, or prevent the user from continuing with plain chat.

## Testing Strategy

Automated coverage should include:

1. dynamic metadata maps into and out of the frontend `Message` model;
2. `PlaygroundMessage` delegates only when dynamic metadata is present and enabled;
3. unknown/disabled renderers fall back to source;
4. OpenUI render failure does not break normal message rendering;
5. dynamic action payload validation accepts expected OpenUI form submissions and rejects malformed payloads;
6. action submission produces the expected normal user message shape;
7. saved/reloaded dynamic messages render consistently in `/chat`, sidepanel chat, and workspace chat;
8. extension build does not violate import/CSP assumptions;
9. existing Markdown, code, reasoning, and artifact rendering behavior remains unchanged.

Manual QA should cover:

- plain chat with the feature disabled;
- OpenUI response in `/chat`;
- reload of a saved OpenUI message;
- generated form submission into a follow-up chat turn;
- safe fallback in extension sidepanel if OpenUI is not enabled there;
- theme compatibility in light and dark modes;
- long OpenUI output that exceeds normal message height.

## Rollout Plan

### Stage 1: Contract And Final Render

- Add dynamic UI metadata types.
- Add renderer registry and `DynamicMessageRenderer`.
- Add OpenUI adapter behind a feature flag.
- Render completed OpenUI messages from metadata.
- Preserve source fallback.

### Stage 2: OpenUI Chat Mode

- Add user-facing OpenUI mode control.
- Inject OpenUI instructions through the existing chat/provider path.
- Save completed responses with OpenUI metadata.
- Rehydrate saved messages with dynamic UI metadata.

### Stage 3: Action Round Trip

- Add the dynamic action bridge.
- Validate OpenUI action payloads.
- Convert supported actions into normal user chat turns.
- Add regression coverage for form submission.

### Stage 4: Shared Surface Expansion

- Enable the renderer in sidepanel/workspace surfaces after CSP and layout checks.
- Add artifact rail open/pin support for dynamic UI messages.
- Add streaming preview if final rendering proves stable.

## Open Questions For Implementation Planning

1. Where should the first OpenUI mode control live in the current `/chat` controls?
2. Should OpenUI mode be per-message, per-chat-session, or a temporary composer mode in the first slice?
3. Should dynamic UI metadata be written by the frontend after a successful OpenUI-mode completion, or returned by the backend as part of message persistence?
4. Which provider abstraction layer should own the reusable OpenUI system/developer prompt?
5. Should sidepanel OpenUI rendering be enabled in v1 or explicitly source-fallback until CSP testing passes?

## Acceptance Criteria

- The approved design documents literal OpenUI support as the first renderer.
- The design also establishes a renderer-neutral dynamic UI path for future content types.
- `/chat`, extension sidepanel chat, and workspace chat are addressed as shared surfaces.
- Persistence, action round-trip, source fallback, feature gating, and extension safety are covered.
- The design avoids PR #151 implementation choices that do not fit this repository.

