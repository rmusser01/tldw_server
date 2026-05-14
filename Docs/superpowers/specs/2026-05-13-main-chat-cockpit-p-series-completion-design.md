# Main /chat Cockpit P-Series Completion Design

## Scope

This document defines the staged completion plan for GitHub issue #1646 and draft PR #1582. The work is limited to the main WebUI `/chat` page cockpit rails inside the chat window.

This plan does not cover the browser-extension sidepanel, app sidebar, character-library page, settings page, onboarding, backend architecture, MCP Hub lifecycle or policy management, media ingestion, evaluations, or unrelated cleanup.

PR #1582 must remain draft and must not be merged until P0, P1, and P2 are fully complete and the human maintainer explicitly approves each tier as complete.

## Current Problem

The current PR has visible cockpit rail controls, but several flows are still reachability-only. A control that merely opens a dialog is not complete if the user cannot finish the workflow, verify the state changed, recover from empty/degraded states, and use the configured state in a real chat turn.

The most important gap is the `Character / Persona` rail. It shows selection state and opens the shared assistant selector, but it does not yet provide a complete cockpit workflow for clearing, changing, inspecting, default bootstrap behavior, persona memory behavior, or real-server proof.

## Completion Strategy

Use workflow-first stages. Each stage must finish one user workflow end to end, including shared-state wiring, empty/degraded states, component coverage, and real-server proof where the issue requires it.

Do not mark a stage complete because controls are visible. A stage is complete only when the workflow can be completed from the main `/chat` cockpit rails and verified.

## Stage 0: Reopen Honest Tracking

Goal: make the tracked state match reality before implementation continues.

Work:

- Keep `TASK-295` as the umbrella Backlog task for PR #1582 completion.
- Link issue #1646 and this spec from `TASK-295`.
- Treat earlier "complete cockpit" notes as historical, not as evidence that the merge bar is met.
- PR #1582 remains draft.

Exit criteria:

- `TASK-295` references issue #1646 and this staged completion spec.
- The tracking language states that P0, P1, and P2 require explicit human approval before merge readiness.

## Stage 1: Character / Persona Rail Completion

Goal: make the right rail a real assistant control surface.

Work:

- Add a direct `Clear assistant` or `No character` action in the `Character / Persona` rail.
- Make the selector open on the correct tab:
  - current character or no assistant: Characters
  - current persona: Personas
- Support complete change flows:
  - character to another character
  - character to persona
  - persona to character
  - persona to none
  - none to character or persona
- Add an inspect/details action for the selected character/persona from the rail flow.
- Keep Scene Director separate and character-only.
- Define and cover behavior for:
  - fresh chat
  - existing server chat
  - temporary chat
  - default-character bootstrap
  - persona memory mode
- Preserve the same selected assistant state used by the composer and send pipeline.

Verification:

- Component/integration tests cover none, character, persona, clear, tab-targeting, change flows, and Scene Director separation.
- Real-server Playwright selects, changes, and clears a real character/persona when real server data exists.
- If real server data is absent, Playwright asserts the real recoverable empty state, and populated behavior is covered in component/integration tests.

P0 approval dependency:

- This stage is the first P0 gate. Do not proceed to P0 approval without it.

## Stage 2: Prompt Rail Completion

Goal: make prompts fully selectable and clearable from the left rail.

Work:

- Select prompt from the rail using the same prompt state used by the composer.
- Clear selected template, quick prompt, and inline/custom prompt safely.
- Show inline/custom system prompt contribution when active.
- Ensure clearing prompt context does not clear files, knowledge, media, research, web search, model, MCP, or assistant state.
- Add explicit loading, empty, disabled, and error states for prompt data.

Verification:

- Component/integration tests cover selected template, quick prompt, inline custom prompt, no prompt, unavailable prompt data, and clear isolation.
- Real-server Playwright selects and clears a real prompt when prompt data exists.
- If no real prompts exist, Playwright asserts the real recoverable empty state.

## Stage 3: Model & Chat Rail Completion

Goal: prove active model/chat settings are scoped, editable, and reflected in the chat request path.

Work:

- Persist and restore a harmless model/chat setting through the rail-opened `CurrentChatModelSettings` flow.
- Show active provider:model scope in the rail.
- Show inherited/default versus explicitly overridden values.
- Verify duplicate model IDs across providers route and scope correctly from actual selector behavior.
- Keep default model selector behavior limited to configured usable choices, with broader search as discovery only.

Verification:

- Component/store tests prove provider:model settings isolation.
- Browser tests prove selector choice updates route/scope summaries.
- Real-server Playwright persists/restores a harmless setting or uses disposable local-only state.
- A real chat send verifies request payload uses the expected model/provider route after configuration.

## Stage 4: MCP Rail Completion

Goal: make MCP configuration real from the runtime rail.

Work:

- Ensure rail tool choice updates the same state used by request construction.
- Make `Configure MCP` complete a real settings workflow when MCP is available.
- Show real unavailable/degraded reasons when MCP is not available.
- Disable impossible actions rather than leaving enabled controls that dead-end.
- Distinguish tool states where available:
  - discovered
  - executable
  - chat-enabled
  - user-disabled
  - unavailable
- Keep MCP Hub lifecycle, credentials, server policy, and catalog governance out of `/chat`.

Verification:

- Component/integration tests cover available, unavailable, degraded, and disabled MCP states.
- Real-server Playwright completes an MCP settings workflow when available, or asserts the real unavailable/degraded state.
- Real chat request construction reflects rail-selected tool choice.

P0 approval gate:

- After Stages 1 through 4 are complete, run the full P0 verification set.
- Ask the human maintainer to approve P0 completion.
- Do not move to merge readiness. P1 and P2 still block merge.

## Stage 5: Context And Session Completion

Goal: make the left rail answer what will affect the next reply and what session the user is controlling.

Work:

- Ensure the context rail represents:
  - files
  - knowledge
  - media
  - research context
  - web search
  - prompt/system prompt
  - character/persona/worldbook context where applicable
- Preserve isolated clear/remove actions for each supported context class.
- Ensure empty, loading, disabled, degraded, and error states are explicit and recoverable.
- Verify session state:
  - temporary chat
  - saved/server/local status
  - history-linked state
  - session switching without stale rail state

Verification:

- Component/integration tests cover all supported context classes and clear isolation.
- Session-switch tests prove rail state updates and stale context is not retained.
- Real-server proof covers at least web search and any available real context class without mocked data.

## Stage 6: Run Controls And Recovery

Goal: prove the cockpit can control and recover from generation states.

Work:

- Verify stop streaming from the rail/status surface.
- Verify regenerate last response from the rail/status surface.
- Add disabled states when stop/regenerate are unavailable.
- Add recoverable provider/server error state behavior.
- Ensure rail controls do not bypass the existing request state machine.

Verification:

- Component/integration tests cover streaming, ready, disabled, regenerate, and error states.
- Real-server or focused integration coverage proves a real stop/regenerate path where feasible.

## Stage 7: Keyboard, Focus, And Mobile Completion

Goal: make completed workflows accessible and responsive.

Work:

- Focus enters and returns from:
  - assistant selector
  - prompt selector
  - MCP settings
  - Model & Chat settings
- Keyboard operation works for all major rail controls.
- Mobile cockpit tabs support completed workflows, not just visibility checks.
- Mobile focus mode remains a clean chat-first layout.

Verification:

- Focus restoration tests cover each rail-launched surface.
- Keyboard tests cover opening, closing, selecting, clearing, and disabled states.
- Mobile Playwright covers assistant selection/clear, prompt selection/clear, model settings, MCP settings, web search, and focus-mode return.

P1 approval gate:

- After Stages 5 through 7 are complete, run the full P1 verification set.
- Ask the human maintainer to approve P1 completion.
- Do not move to merge readiness. P2 still blocks merge.

## Stage 8: P2 Polish And Merge Readiness

Goal: finish the cockpit as a coherent product surface after workflows are real.

Work:

- Recheck rail information architecture and ordering:
  - Runtime
  - Model & Chat
  - MCP Tools
  - Character / Persona
  - Scoped Settings
  - Run Controls
- Tighten copy:
  - selector actions
  - settings actions
  - clear actions
  - disabled and degraded explanations
- Finalize degraded-health behavior:
  - chat-blocking degradation is distinct from unrelated subsystem degradation
  - unrelated degradation permits chat with warnings
- Resolve composer/rail duplication intentionally:
  - do not remove composer controls until rail equivalents are proven
  - de-emphasize duplicates only where it improves clarity without reducing power-user speed
- Perform final responsive visual QA:
  - desktop cockpit
  - desktop focus mode
  - mobile cockpit tabs
  - mobile focus mode
  - long labels
  - empty states
  - degraded states
  - selected character/persona/prompt/model states
- Refresh real-server screenshot proof:
  - actual conversation
  - visible prompt state
  - visible context state
  - visible model/chat state
  - visible MCP state
  - visible character/persona state

Verification:

- Real-server Playwright passes without mocked payloads or `page.route`.
- Focused component/integration suites pass.
- Visual QA screenshots show no overlap or incoherent density at desktop and mobile sizes.
- PR closeout comment links issue #1646 and summarizes P0, P1, and P2 evidence.
- Known non-chat CI baseline issues are documented separately from cockpit status.

P2 approval gate:

- Ask the human maintainer to approve P2 completion.
- Only after P0, P1, and P2 are explicitly approved should PR #1582 be considered for ready-for-review or merge discussion.

## Approval And Merge Rules

- P0, P1, and P2 are sequential approval gates.
- The PR remains draft through all implementation stages.
- Passing tests alone is not sufficient. The human maintainer must explicitly approve each tier.
- No implementation stage may be marked complete if it only proves reachability rather than completed state-changing workflow behavior.
- Real-server proof must use the running server. Do not use mocked payloads or `page.route` for merge-critical proof.
