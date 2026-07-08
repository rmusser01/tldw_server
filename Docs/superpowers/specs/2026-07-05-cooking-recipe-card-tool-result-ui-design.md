# Cooking Recipe Card Tool-Result UI Design

Date: 2026-07-05
Status: Human-reviewed after spec reviewer timeout
Backlog: TASK-12150

## Summary

Add a small, typed UI affordance for cooking recipes by letting the model call a read-only backend MCP tool named `cooking.recipe_card.render`. The tool validates model-provided recipe card data and returns a namespaced `tldw_ui` envelope. WebUI and the browser extension render that envelope with a shared recipe card component when it appears in a tool result.

This is not a frontend recipe detector and not a generic UI rendering protocol. The model decides when a recipe card is useful by calling the tool. The frontend decides only whether a known, valid `tldw_ui` payload can be rendered specially. Unknown or invalid tool results continue through the existing generic tool-call display.

## Goals

- Give recipe answers a richer inline UI without requiring users to switch modes.
- Let the model opt in through normal tool calling instead of frontend text heuristics.
- Keep the payload typed, bounded, namespaced, and safe to replay from saved chat history.
- Share the renderer between WebUI chat and extension sidepanel chat.
- Reuse the existing MCP tool and tool-call UI paths.
- Keep the first slice useful without durable timers, grocery export, or recipe storage.

## Non-Goals

- Do not add a generic `ui.render` tool in v1.
- Do not enable or extend the existing OpenUI dynamic UI path for this feature.
- Do not infer recipes from assistant prose in the frontend.
- Do not persist recipes in a dedicated recipe database.
- Do not add notification permissions, background timers, or OS-level alarms.
- Do not add grocery list export, nutrition analysis, or source crawling.
- Do not change existing dynamic UI surface gating.

## Current Context

The repository already has three relevant paths:

- MCP Unified modules define tools with `create_tool_definition(...)` and metadata such as `readOnlyHint`.
- `/api/v1/tools` delegates list and execute operations through `ToolExecutor`, which wraps MCP `tools/list` and `tools/call`.
- Chat message rendering already shows assistant tool calls through the shared `ToolCallBlock` used by WebUI and extension chat surfaces.

The frontend also has an OpenUI dynamic UI feature, but it is intentionally surface-gated. Web chat can render OpenUI while the extension sidepanel and workspace surfaces fall back. Cooking cards should not use that path. A typed renderer inside the existing tool-call display path is smaller, safer, and easier to share.

## Recommended Approach

Add one domain-specific MCP tool:

```text
cooking.recipe_card.render
```

The tool is read-only and side-effect free. It accepts structured recipe data from the model, validates it, normalizes it into a stable envelope, and returns JSON content containing:

```json
{
  "tldw_ui": {
    "kind": "recipe_card",
    "version": 1,
    "recipe": {}
  }
}
```

The frontend adds a tiny typed UI result parser. When a tool result parses as a supported `tldw_ui` envelope, it renders the matching shared component. When parsing fails, the existing generic tool result UI remains the fallback.

## Alternatives Considered

### Generic `ui.render` Tool

Pros:

- One tool could eventually cover recipes, checklists, itineraries, tables, and other structured UI.
- The frontend could grow a general typed UI renderer registry.

Cons:

- It creates a UI protocol before there is enough evidence for one.
- Permission and safety semantics are vaguer than a domain-specific read-only cooking tool.
- The model would have a broader tool that invites unrelated UI payloads.

Decision: defer. A future `ui.render` can absorb proven envelopes after at least two or three specific renderers exist.

### Reuse OpenUI Dynamic UI

Pros:

- Existing dynamic UI envelope, renderer, and action bridge already exist.
- Could produce more flexible visual layouts.

Cons:

- OpenUI is intentionally disabled on extension and workspace surfaces.
- Model-generated UI source is harder to validate than a recipe schema.
- The feature needs one known card, not arbitrary component source.

Decision: do not use OpenUI for cooking cards.

### Frontend Recipe Detector

Pros:

- No backend tool needed.
- Could display cards when the assistant forgets to call a tool.

Cons:

- Brittle prose parsing.
- Surprising UI triggers.
- Harder to explain and test than explicit tool use.

Decision: do not detect recipes from prose in v1.

## Backend Design

### Tool Module

Add a small MCP module or register the tool in the nearest existing UI/helper module if implementation planning finds one. The tool definition should include:

- name: `cooking.recipe_card.render`
- description: explains that it validates and returns a recipe card UI payload
- metadata:
  - `readOnlyHint: true`
  - category such as `ui` or `cooking`
  - no write/destructive/network capabilities

The tool does not call external services, read local files, write databases, or modify user state.

Implementation must also wire the module into normal MCP discovery, not only unit-test it directly. The default path is `tldw_Server_API/Config_Files/mcp_modules.yaml`, which is loaded by the MCP server module autoloader. If implementation planning chooses a new `CookingModule`, the plan must add the module config entry and a test proving `tools/list` exposes `cooking.recipe_card.render` through the normal server configuration path.

### Input Contract

Expected input:

```json
{
  "title": "Cajun Alfredo Sauce",
  "servings": {
    "value": 2,
    "label": "2 servings"
  },
  "ingredients": [
    {
      "display": "3 tbsp butter",
      "name": "butter",
      "quantity": 3,
      "unit": "tbsp",
      "note": null,
      "scalable": true
    }
  ],
  "steps": [
    {
      "display": "Melt butter in a pan over medium heat.",
      "timer_seconds": null
    }
  ],
  "summary": "Rich Cajun-style Alfredo sauce for pasta.",
  "notes": [
    "Add pasta water slowly until the sauce loosens."
  ]
}
```

Only `title`, `servings`, `ingredients`, and `steps` are required. Each ingredient keeps `display` as the authoritative text. Numeric scaling is optional and used only when `quantity`, `unit`, and `scalable` are present and safe.

### Output Contract

The tool returns one canonical envelope:

```json
{
  "tldw_ui": {
    "kind": "recipe_card",
    "version": 1,
    "recipe": {
      "title": "Cajun Alfredo Sauce",
      "servings": {
        "value": 2,
        "label": "2 servings"
      },
      "ingredients": [],
      "steps": [],
      "summary": null,
      "notes": []
    }
  }
}
```

The namespaced `tldw_ui` wrapper avoids collisions with ordinary tool JSON. The `version` field lets the frontend reject or fall back on unsupported future shapes.

### Validation Limits

Implementation should set boring, explicit limits:

- title: 1 to 120 characters
- summary: 0 to 300 characters
- notes: at most 8 items, 300 characters each
- ingredients: 1 to 60 items
- ingredient display: 1 to 180 characters
- steps: 1 to 40 items
- step display: 1 to 600 characters
- servings value: integer or decimal from 1 to 50
- timer seconds: optional, 1 to 86400

If the payload exceeds limits or fails schema validation, the tool returns a normal tool error. The frontend then shows the generic tool error/result path rather than a partial recipe card.

## Frontend Design

### Typed Parser

Add a shared parser for tool result content:

- parse string content as JSON when possible
- require a top-level `tldw_ui`
- require `kind === "recipe_card"`
- require `version === 1`
- validate core recipe fields and caps before rendering
- return `null` on mismatch

This parser should never throw to React render paths. Bad payloads fall back to the current generic tool-call UI.

### Renderer Placement

Keep the existing `ToolCallBlock` behavior as the default. Add one check before the generic formatted-result body:

1. Parse the tool result with the typed UI parser.
2. If the result is not an error and it returns a recipe card payload, render `RecipeCard`.
3. Otherwise render the existing generic result block.

This keeps the feature local to tool-result rendering and avoids changing message metadata, dynamic UI envelopes, or chat completion transport.

Also add a cooking label and icon mapping for `cooking.recipe_card.render` so collapsed, loading, error, and fallback states read as intentional even when the special renderer is not active.

### Shared Component

Add a shared `RecipeCard` component under the existing shared UI package so WebUI and extension sidepanel can use the same rendering. It should be compact enough for chat:

- title
- ingredient and step counts
- servings stepper
- ingredients list
- steps list
- `Cooking mode` button

The first `Cooking mode` behavior is an inline expanded step view. It can focus one step at a time and show any provided duration as text. It should not start reliable alarm timers in v1.

### Ingredient Scaling

The serving stepper changes local display state only. It should:

- keep original `display` text available
- scale only ingredients with numeric `quantity`, known `unit`, and `scalable: true`
- leave non-scalable ingredients unchanged
- avoid trying to parse freeform strings on the frontend

Examples that should remain unchanged unless explicitly structured:

- `1 can evaporated milk`
- `salt to taste`
- `half an onion`
- `reserved pasta water as needed`

### Visual Treatment

The card should match the existing restrained product UI:

- no nested cards
- no decorative blobs or hero treatment
- compact header, subdued metadata, clear affordances
- accessible buttons for servings and cooking mode
- responsive layout that fits the extension sidepanel width

The screenshot is a reference for the information hierarchy, not a directive to copy Claude's visual styling.

## Data Flow

1. User asks for a recipe or cooking idea.
2. Model chooses to call `cooking.recipe_card.render` with structured recipe data.
3. Backend MCP tool validates and returns the `tldw_ui` recipe envelope.
4. Chat stores the assistant tool call and tool result durably enough for later message replay.
5. Frontend receives the message history.
6. `ToolCallBlock` sees the tool result, parses the envelope, and renders `RecipeCard`.
7. Saved chats replay the same card from the persisted tool result.

No extra frontend detector or post-processing pass is needed.

Implementation planning must verify this persistence path before building the renderer. If a target chat surface does not currently persist `toolResults`, the implementation must add the smallest adapter needed for that surface or exclude that surface from the first slice explicitly. The intended MVP includes WebUI chat and extension sidepanel replay.

## Error Handling

- Invalid tool input returns a tool error with a concise validation message.
- Tool results marked as errors never render recipe UI.
- Unsupported `tldw_ui.version` falls back to generic JSON display.
- Unknown `tldw_ui.kind` falls back to generic JSON display.
- Renderer parse failures fall back without throwing.
- Missing optional fields render as absent UI, not as placeholder copy.

## Security And Privacy

- Treat tool output as untrusted because it originates from model-provided data.
- Render all strings as text, never as raw HTML.
- Do not persist recipe data outside the normal chat/tool-result storage.
- Do not request browser notification permissions in v1.
- Do not let the cooking tool access files, network, databases, or credentials.
- Keep explicit schema and display caps on both backend and frontend boundaries.

## Testing Plan

Backend:

- tool appears in MCP tool list with read-only metadata
- tool appears through normal MCP module configuration, not only direct module construction
- valid minimal recipe returns a `tldw_ui.kind = "recipe_card"` envelope
- valid full recipe preserves display strings and structured quantities
- too many ingredients or steps returns a validation error
- invalid servings and overlong strings return validation errors

Frontend:

- valid recipe tool result renders `RecipeCard`
- failed recipe tool result renders the generic error/result path, not `RecipeCard`
- unknown tool result still renders the generic tool-call block
- malformed JSON falls back to generic display
- unsupported `tldw_ui.version` falls back
- serving stepper scales only structured scalable ingredients
- non-scalable ingredients keep original display text
- `cooking.recipe_card.render` has an intentional label/icon in collapsed and fallback states
- extension-width render does not overflow core controls

## Implementation Boundaries

Implementation planning should start from existing code paths:

- MCP tool registration under `tldw_Server_API/app/core/MCP_unified/modules/implementations/`
- MCP module config wiring under `tldw_Server_API/Config_Files/mcp_modules.yaml`
- shared tool-call rendering under `apps/packages/ui/src/components/Sidepanel/Chat/ToolCallBlock.tsx`
- shared UI types/utilities under `apps/packages/ui/src/types` or `apps/packages/ui/src/utils`

The final file placement can follow nearby patterns after reading the exact module registry and frontend export structure.

## Rollout

Ship behind normal tool availability. No separate feature flag is required unless implementation planning finds that MCP tool exposure is globally on by default in places where this would be noisy.

The feature is safe to merge in slices:

1. Backend MCP tool and validation tests.
2. Frontend parser and renderer tests.
3. Shared card UI and chat integration.
4. Visual/browser verification for WebUI and extension widths.

## Decisions For Implementation Planning

- Prefer a new `cooking` MCP module unless implementation planning finds an existing UI/helper module that already owns typed presentation payloads.
- Verify chat tool-result persistence before frontend integration. Replay is part of the MVP for WebUI chat and extension sidepanel.
- Verify both target surfaces expose tool results to `ToolCallBlock`; add the smallest adapter needed if one does not.
- Start with the tool description as the model-facing hint. Add prompt guidance only if manual or automated testing shows models do not discover the tool reliably.
