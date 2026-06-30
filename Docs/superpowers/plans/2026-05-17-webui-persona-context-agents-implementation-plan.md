# WebUI Persona Context Agents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clarify persona, character, companion, context asset, and agent relationships across the WebUI/extension while preserving fast launch and expert workflows.

**Architecture:** Add a small route-job policy layer and focused route tests before changing route copy or actions. Reuse the existing Persona Garden, Companion, character quick-chat, dictionary, world-book, ACP, and agent components instead of introducing a parallel runtime or backend contract.

**Tech Stack:** React, TypeScript, React Router, Ant Design, shared `@tldw/ui` state primitives, Vitest, Testing Library, Playwright, Backlog.md task tracking.

---

## Source Documents

- Source spec: `Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md`
- Parent implementation plan: `Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md`
- Planning Backlog task: `TASK-418.4`
- Parent planning Backlog task: `TASK-418`
- Dependency plan: `Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md`
- Dependency plan: `Docs/superpowers/plans/2026-05-17-webui-chat-global-chrome-implementation-plan.md`

## Findings Closed Or Supported

- F1 support: route labels, headings, and primary actions must make each top-level job understandable.
- F9 support: persona, companion, context asset, and agent unavailable states must use consistent capability language.
- F15 support: advanced persona, agent, and context controls must remain discoverable without overwhelming first-time users.
- F18 support: experimental, hosted, local-only, and degraded agent or companion states must be visible in user language.

## Route Scope

Primary implementation routes:

- `/persona`
- `/characters`
- `/companion`
- `/agents`
- `/agent-tasks`
- `/acp-playground`
- `/chat-workflows`
- `/dictionaries`
- `/world-books`

Cross-route launch targets:

- `/chat`
- `/companion/conversation`
- `/mcp-hub`
- `/settings/model`
- `/settings/tldw`

## Out Of Scope

- No backend API changes unless browser verification proves a route cannot expose state responsibly from existing data.
- No new persona runtime, visual-pack generation, companion renderer, ACP transport, or agent execution system.
- No route renaming. Existing URLs and aliases stay valid.
- No replacement of the shared state primitives from WP2.
- No unrelated media, knowledge, settings, or chat redesign work.

## Current Code Evidence

- `apps/packages/ui/src/routes/route-registry.tsx` registers `/characters`, `/companion`, `/world-books`, `/dictionaries`, `/chat-workflows`, `/acp-playground`, `/agents`, and `/agent-tasks`.
- `apps/packages/ui/src/routes/sidepanel-route-registry.tsx` registers sidepanel `/persona`, `/agent`, and `/companion` routes.
- `apps/packages/ui/src/routes/route-paths.ts` defines `CHAT_PATH = "/chat"` and `buildChatThreadPath`.
- `apps/packages/ui/src/routes/sidepanel-persona.tsx` owns the real Persona Garden, Companion sidepanel, setup wizard, live voice controller, buddy render context, and persona capability gates.
- `apps/packages/ui/src/routes/option-characters.tsx` renders `CharactersWorkspace` inside `RouteErrorBoundary`.
- `apps/packages/ui/src/components/Option/Characters/CharactersWorkspace.tsx` already frames characters as reusable chat personas and has unsupported-state copy for missing `/api/v1/characters`.
- `apps/packages/ui/src/components/Option/Characters/CharacterPreviewPopup.tsx` exposes Chat, Chat in new tab, and Test in popup actions.
- `apps/packages/ui/src/components/Option/Characters/useCharacterQuickChat.tsx` owns character quick-chat launch and server chat handoff.
- `apps/packages/ui/src/routes/option-companion.tsx` renders `CompanionHomeShell` inside `RouteErrorBoundary`.
- `apps/packages/ui/src/components/Option/Companion/CompanionHomeShell.tsx` links Companion to `/chat`, `/knowledge`, and `/media-multi`.
- `apps/packages/ui/src/components/Option/Companion/CompanionHomePage.tsx` owns the Companion dashboard heading and `Open conversation` action.
- `apps/packages/ui/src/routes/option-chat-workflows.tsx` renders `ChatWorkflowsPage`.
- `apps/packages/ui/src/components/Option/ChatWorkflows/ChatWorkflowsPage.tsx` creates the structured QA handoff to `/chat`.
- `apps/packages/ui/src/routes/option-dictionaries.tsx` renders `DictionariesWorkspace`.
- `apps/packages/ui/src/components/Option/Dictionaries/useDictionaryChatContextNavigation.ts` currently falls back to `#/` rather than canonical `/chat`.
- `apps/packages/ui/src/routes/option-world-books.tsx` renders `WorldBooksWorkspace`.
- `apps/packages/ui/src/components/Option/WorldBooks/WorldBookDetailPanel.tsx` owns attached-character context and links back to `/characters`.
- `apps/packages/ui/src/routes/option-agents.tsx`, `apps/packages/ui/src/routes/option-agent-tasks.tsx`, and `apps/packages/ui/src/routes/option-acp-playground.tsx` render agent surfaces without the same route-boundary wrapper pattern used by other option routes.
- `apps/packages/ui/src/components/Option/AgentRegistry/index.tsx` shows ACP health, registered agents, status labels, and launch to `/acp-playground`.
- `apps/packages/ui/src/components/Option/AgentTasks/index.tsx` owns orchestration unsupported/setup states and links to `/agents` and `/acp-playground`.
- `apps/packages/ui/src/components/Option/ACPPlayground` owns the interactive ACP session surface.

## Concept And Route Job Taxonomy

Use this route-job policy as the implementation source of truth. The copy does not need to appear verbatim in the UI, but the product meaning and primary action must be reflected in headings, labels, empty states, and tests.

| Route | Concept | Primary job | Primary action | Power-user job |
| --- | --- | --- | --- | --- |
| `/persona` | Persona Garden | Configure and live-test assistant persona behavior, commands, voice, visual buddy state, and policy scope. | Use persona in live assistant | Inspect setup, policy, voice, command, and buddy state quickly. |
| `/characters` | Character | Create reusable chat-facing characters with role, prompt, cards, and linked context. | Start in chat | Test, edit, import, filter, and launch without losing the current library position. |
| `/companion` | Companion | Show the personal companion home state and route to the companion conversation. | Open conversation | Jump to work surfaces and see provider/setup state. |
| `/chat-workflows` | Workflow | Run structured chat workflows and hand off the result to free chat. | Continue to chat | Re-run structured answers and compare prompts quickly. |
| `/dictionaries` | Context asset | Manage dictionary entries that can be attached to chat context. | Use in chat context | Edit entries, inspect scope, and attach to active conversations. |
| `/world-books` | Context asset | Manage world-book lore/context and attach it to characters. | Attach to character | Inspect attached characters, entries, stats, and activation scope. |
| `/agents` | Agent registry | Inspect configured ACP agents, readiness, and launch targets. | Launch agent | See setup, health, and unavailable/degraded diagnostics. |
| `/agent-tasks` | Agent orchestration | Manage task queues and projects for configured agents. | Create or inspect task | Diagnose unsupported orchestration and recover to setup routes. |
| `/acp-playground` | Agent session | Create and inspect direct ACP sessions. | Create session | Test agent connections and understand protocol failures. |

## Launch Path Policy

| Source route | Action | Target policy |
| --- | --- | --- |
| `/persona` | Use persona in live assistant | Stay on `/persona` or open the sidepanel live assistant flow; do not fake a chat thread if the runtime is persona-stream based. |
| `/characters` | Start in chat | Use `useCharacterQuickChat` and canonical `CHAT_PATH` or `buildChatThreadPath` after a server chat exists. |
| `/companion` | Open conversation | Navigate to `/companion/conversation`. |
| `/companion` | Open main chat | Navigate to `/chat`, with label text that distinguishes main chat from companion conversation. |
| `/chat-workflows` | Continue to chat | Use the existing server chat handoff and canonical `/chat` path. |
| `/dictionaries` | Use in chat context | Use `buildChatThreadPath` or `/chat`; do not navigate to the root hash path. |
| `/world-books` | Open attached character | Navigate to `/characters` with focus query parameters already supported by the route. |
| `/world-books` | Start attached character chat | Use the character quick-chat launch path when a focused attached character is available. |
| `/agents` | Launch agent | Navigate to `/acp-playground?agent=<id>` after preserving current registry health state. |
| `/agent-tasks` | Open setup | Navigate to `/agents`, `/acp-playground`, or `/mcp-hub` depending on the missing capability. |
| `/acp-playground` | Create session | Stay on `/acp-playground` and make the ACP state visible in the session panel. |

## File Ownership Map

### Route Job Policy

- Create: `apps/packages/ui/src/routes/persona-context-route-jobs.ts`
  - Own route job taxonomy, primary actions, and launch target labels for this route family.
  - Keep it pure TypeScript. No React hooks, network calls, or component imports.

- Test: `apps/packages/ui/src/routes/__tests__/persona-context-route-jobs.test.ts`
  - Assert every WP7 route has one concept, one primary job, and one primary action.
  - Assert launch targets point to canonical route paths.

### Route Registry And Boundaries

- Modify: `apps/packages/ui/src/routes/route-registry.tsx`
  - Reuse route job labels where it improves route metadata.
  - Keep existing route paths and aliases.

- Modify: `apps/packages/ui/src/routes/sidepanel-route-registry.tsx`
  - Preserve `/persona`, `/agent`, and `/companion` separation.
  - Add route metadata only if WP1 route taxonomy exposes a shared hook.

- Modify: `apps/packages/ui/src/routes/option-agents.tsx`
  - Wrap route content in `RouteErrorBoundary` with route id `agents`.

- Modify: `apps/packages/ui/src/routes/option-agent-tasks.tsx`
  - Wrap route content in `RouteErrorBoundary` with route id `agent-tasks`.

- Modify: `apps/packages/ui/src/routes/option-acp-playground.tsx`
  - Wrap route content in `RouteErrorBoundary` with route id `acp-playground`.

- Test: `apps/packages/ui/src/routes/__tests__/route-registry.persona.test.ts`
  - Preserve existing persona/agent route separation.
  - Assert all WP7 routes have user-facing labels and route jobs.

- Test: `apps/packages/ui/src/routes/__tests__/chat-workflows-route.test.tsx`
  - Preserve `/chat-workflows` registration and handoff expectations.

- Create or modify: `apps/packages/ui/src/routes/__tests__/persona-context-route-boundaries.test.tsx`
  - Assert route wrappers use `RouteErrorBoundary` for character, companion, context asset, and agent routes where route ownership allows it.

### Persona And Companion

- Modify: `apps/packages/ui/src/routes/sidepanel-persona.tsx`
  - Clarify Persona Garden versus Companion mode headings and primary actions.
  - Preserve setup wizard, voice controls, buddy render context, persona policy summary, and live session behavior.
  - Use WP2 capability language for setup, unavailable, unreachable, and degraded states.

- Modify: `apps/packages/ui/src/routes/option-companion.tsx`
  - Preserve `RouteErrorBoundary`.
  - Pass only route-level metadata if needed.

- Modify: `apps/packages/ui/src/components/Option/Companion/CompanionHomeShell.tsx`
  - Keep quick links, but separate main chat, companion conversation, knowledge, and media actions.
  - Use provider/setup state language aligned with WP2.

- Modify: `apps/packages/ui/src/components/Option/Companion/CompanionHomePage.tsx`
  - Keep `Open conversation` distinct from main chat.
  - Keep dashboard state visible without implying Companion and Persona Garden are the same feature.

- Test: `apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx`
  - Assert Persona Garden and Companion modes expose distinct jobs and state language.

- Test: `apps/packages/ui/src/routes/__tests__/option-companion.test.tsx`
  - Assert Companion route heading, primary action, setup state, and conversation target.

### Characters And Character Chat Launch

- Modify: `apps/packages/ui/src/routes/option-characters.tsx`
  - Preserve `RouteErrorBoundary`.
  - Pass route job metadata only if needed.

- Modify: `apps/packages/ui/src/components/Option/Characters/CharactersWorkspace.tsx`
  - Keep character library density.
  - Make first-run and unsupported states distinguish "character" from "persona" and "companion".
  - Surface the primary "Start in chat" job without hiding import/edit flows.

- Modify: `apps/packages/ui/src/components/Option/Characters/CharacterPreviewPopup.tsx`
  - Rename primary launch actions to "Start in Chat", "Open in New Chat Tab", and "Test in Popup" or equivalent labels with the same intent.
  - Keep existing quick-chat and popup behavior.

- Modify: `apps/packages/ui/src/components/Option/Characters/useCharacterQuickChat.tsx`
  - Keep existing server-chat handoff.
  - Use canonical chat path helpers where a route string is needed.

- Test: `apps/packages/ui/src/components/Option/Characters/__tests__/CharacterPreviewPopup.test.tsx`
  - Assert label clarity and launch callbacks.

- Test: `apps/packages/ui/src/components/Option/Characters/__tests__/useCharacterQuickChat.test.tsx`
  - Assert canonical `/chat` navigation and server chat handoff.

- Test: `apps/tldw-frontend/e2e/workflows/journeys/character-chat.spec.ts`
  - Preserve create character, start chat, select character, and system prompt API coverage.

### Context Assets

- Modify: `apps/packages/ui/src/routes/option-chat-workflows.tsx`
  - Preserve existing `ChatWorkflowsPage` ownership and route boundary.

- Modify: `apps/packages/ui/src/components/Option/ChatWorkflows/ChatWorkflowsPage.tsx`
  - Keep structured QA flow.
  - Make handoff wording explicit enough that users understand the result opens `/chat`.

- Modify: `apps/packages/ui/src/routes/option-dictionaries.tsx`
  - Preserve route boundary and current workspace ownership.

- Modify: `apps/packages/ui/src/components/Option/Dictionaries/DictionariesWorkspace.tsx`
  - Clarify dictionary activation scope in headings, empty states, and action labels.
  - Preserve expert editing, import, and filter workflows.

- Modify: `apps/packages/ui/src/components/Option/Dictionaries/useDictionaryChatContextNavigation.ts`
  - Replace root hash fallback with `CHAT_PATH` or `buildChatThreadPath`.
  - Keep existing state transfer and chat context hydration behavior.

- Modify: `apps/packages/ui/src/routes/option-world-books.tsx`
  - Preserve route boundary and current workspace ownership.

- Modify: `apps/packages/ui/src/components/Option/WorldBooks/WorldBooksWorkspace.tsx`
  - Clarify world-book relationship to characters and chat context.
  - Preserve entries, attachments, stats, and settings workflows.

- Modify: `apps/packages/ui/src/components/Option/WorldBooks/WorldBookDetailPanel.tsx`
  - Keep attached-character list and focus links.
  - Add or clarify attached-character chat launch only when an existing character launch path is available.

- Test: `apps/packages/ui/src/routes/__tests__/chat-workflows-route.test.tsx`
  - Assert route heading and explicit chat handoff.

- Test: `apps/packages/ui/src/components/Option/Dictionaries/__tests__/useDictionaryChatContextNavigation.test.ts`
  - Assert dictionary chat context navigates to `/chat` or `buildChatThreadPath`.

- Test: `apps/packages/ui/src/components/Option/WorldBooks/__tests__/WorldBookDetailPanel.test.tsx`
  - Assert attached-character actions distinguish open character, attach character, and start chat.

### Agents And ACP Capability States

- Modify: `apps/packages/ui/src/components/Agent`
  - Preserve existing agent components and imports.
  - Use WP2 shared state language where agent cards or panels show unavailable, setup-required, permission-denied, or degraded states.

- Modify: `apps/packages/ui/src/components/Option/AgentRegistry/index.tsx`
  - Keep ACP system health and agent card density.
  - Make setup required, ready, unavailable, and degraded labels consistent with WP2.
  - Keep launch target `/acp-playground?agent=<id>`.

- Modify: `apps/packages/ui/src/components/Option/AgentTasks/index.tsx`
  - Keep orchestration concepts distinct from direct ACP sessions.
  - Make unsupported orchestration state route users to setup or registry surfaces in user language.

- Modify: `apps/packages/ui/src/components/Option/ACPPlayground/index.tsx`
  - Keep direct ACP session creation.
  - Align no-session, unhealthy, and setup states with WP2 vocabulary.

- Modify: `apps/packages/ui/src/components/Option/ACPPlayground/ACPPlaygroundHeader.tsx`
  - Keep health indicator but avoid protocol-first language as the only explanation.

- Modify: `apps/packages/ui/src/components/Option/ACPPlayground/ACPSessionPanel.tsx`
  - Keep direct session controls.
  - Make no-session versus ACP-unhealthy states visually and textually distinct.

- Test: `apps/packages/ui/src/components/Option/AgentRegistry/__tests__/AgentRegistryPage.connection.test.tsx`
  - Assert canonical connection config, health labels, and launch target.

- Test: `apps/packages/ui/src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx`
  - Assert unsupported orchestration uses shared capability state language.

- Test: `apps/packages/ui/src/components/Option/ACPPlayground/__tests__/ACPPlayground.connection.test.tsx`
  - Assert setup, unavailable, and degraded states use shared language.

- Test: `apps/packages/ui/src/components/Option/ACPPlayground/__tests__/ACPSessionPanel.test.tsx`
  - Assert no-session and unhealthy-session states are separate.

## Implementation Tasks

### Task 0: Implementation Setup And Evidence Refresh

**Files:**
- Reference: `Docs/superpowers/plans/2026-05-17-webui-persona-context-agents-implementation-plan.md`
- Reference: `Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md`
- Reference: `Docs/superpowers/plans/2026-05-17-webui-chat-global-chrome-implementation-plan.md`
- Backlog: create or update the implementation Backlog task before product code edits.

- [ ] **Step 1: Verify branch and dirty worktree**

Run:

```bash
git branch --show-current
git status --short
```

Expected:
- The implementation branch is known.
- Existing unrelated dirty files are left untouched.

- [ ] **Step 2: Create or update implementation Backlog task**

Expected:
- The task links this plan, the parent plan, the source spec, and the audit report.
- The task lists findings F1 support, F9 support, F15 support, and F18 support.
- The task states that product code edits are limited to the WP7 files in this plan.

- [ ] **Step 3: Capture current browser baseline**

Use Playwright or the in-app browser for:
- `/persona`
- `/characters`
- `/companion`
- `/agents`
- `/agent-tasks`
- `/acp-playground`
- `/chat-workflows`
- `/dictionaries`
- `/world-books`

Expected:
- Each route has a screenshot or DOM observation covering heading, primary action, empty or setup state, and first visible recovery path.
- Observations are linked from the Backlog task.

### Task 1: Lock The Route Job Contract

**Files:**
- Create: `apps/packages/ui/src/routes/persona-context-route-jobs.ts`
- Create: `apps/packages/ui/src/routes/__tests__/persona-context-route-jobs.test.ts`
- Modify: `apps/packages/ui/src/routes/__tests__/route-registry.persona.test.ts`
- Modify only if needed: `apps/packages/ui/src/routes/route-registry.tsx`
- Modify only if needed: `apps/packages/ui/src/routes/sidepanel-route-registry.tsx`

- [ ] **Step 1: Write failing route-job tests**

Create `persona-context-route-jobs.test.ts`:

```ts
import { describe, expect, it } from "vitest"

import {
  PERSONA_CONTEXT_ROUTE_JOBS,
  getPersonaContextRouteJob,
} from "../persona-context-route-jobs"

const expectedRoutes = [
  "/persona",
  "/characters",
  "/companion",
  "/agents",
  "/agent-tasks",
  "/acp-playground",
  "/chat-workflows",
  "/dictionaries",
  "/world-books",
]

describe("persona context route jobs", () => {
  it("defines one user job for every WP7 route", () => {
    expect(PERSONA_CONTEXT_ROUTE_JOBS.map((job) => job.route).sort()).toEqual(
      expectedRoutes.sort(),
    )

    for (const route of expectedRoutes) {
      const job = getPersonaContextRouteJob(route)
      expect(job).toBeDefined()
      expect(job?.primaryJob).toMatch(/\w/)
      expect(job?.primaryActionLabel).toMatch(/\w/)
      expect(job?.concept).toMatch(
        /persona|character|companion|context_asset|agent|workflow/,
      )
    }
  })
})
```

- [ ] **Step 2: Run test to verify failure**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/persona-context-route-jobs.test.ts
```

Expected:
- FAIL because `persona-context-route-jobs.ts` does not exist.

- [ ] **Step 3: Implement route-job policy**

Create `persona-context-route-jobs.ts`:

```ts
import { CHAT_PATH } from "./route-paths"

export type PersonaContextConcept =
  | "persona"
  | "character"
  | "companion"
  | "context_asset"
  | "agent"
  | "workflow"

export type PersonaContextRouteJob = {
  route: string
  concept: PersonaContextConcept
  label: string
  primaryJob: string
  primaryActionLabel: string
  launchTarget?: string
}

export const PERSONA_CONTEXT_ROUTE_JOBS: PersonaContextRouteJob[] = [
  {
    route: "/persona",
    concept: "persona",
    label: "Persona Garden",
    primaryJob: "Configure and live-test assistant persona behavior, commands, voice, visual buddy state, and policy scope.",
    primaryActionLabel: "Use Persona",
  },
  {
    route: "/characters",
    concept: "character",
    label: "Characters",
    primaryJob: "Create reusable chat-facing characters with role, prompt, cards, and linked context.",
    primaryActionLabel: "Start in Chat",
    launchTarget: CHAT_PATH,
  },
  {
    route: "/companion",
    concept: "companion",
    label: "Companion",
    primaryJob: "Open the companion home state and companion conversation.",
    primaryActionLabel: "Open Conversation",
    launchTarget: "/companion/conversation",
  },
  {
    route: "/agents",
    concept: "agent",
    label: "Agents",
    primaryJob: "Inspect configured ACP agents, readiness, and launch targets.",
    primaryActionLabel: "Launch Agent",
    launchTarget: "/acp-playground",
  },
  {
    route: "/agent-tasks",
    concept: "agent",
    label: "Agent Tasks",
    primaryJob: "Manage task queues and projects for configured agents.",
    primaryActionLabel: "Create Task",
    launchTarget: "/agents",
  },
  {
    route: "/acp-playground",
    concept: "agent",
    label: "ACP Playground",
    primaryJob: "Create and inspect direct ACP sessions.",
    primaryActionLabel: "Create Session",
  },
  {
    route: "/chat-workflows",
    concept: "workflow",
    label: "Chat Workflows",
    primaryJob: "Run structured chat workflows and hand off the result to free chat.",
    primaryActionLabel: "Continue to Chat",
    launchTarget: CHAT_PATH,
  },
  {
    route: "/dictionaries",
    concept: "context_asset",
    label: "Dictionaries",
    primaryJob: "Manage dictionary entries that can be attached to chat context.",
    primaryActionLabel: "Use in Chat Context",
    launchTarget: CHAT_PATH,
  },
  {
    route: "/world-books",
    concept: "context_asset",
    label: "World Books",
    primaryJob: "Manage world-book lore and attach it to characters.",
    primaryActionLabel: "Attach to Character",
    launchTarget: "/characters",
  },
]

export const getPersonaContextRouteJob = (
  route: string,
): PersonaContextRouteJob | undefined =>
  PERSONA_CONTEXT_ROUTE_JOBS.find((job) => job.route === route)
```

- [ ] **Step 4: Run route-job tests**

Run:

```bash
bunx vitest run src/routes/__tests__/persona-context-route-jobs.test.ts src/routes/__tests__/route-registry.persona.test.ts
```

Expected:
- PASS.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/routes/persona-context-route-jobs.ts apps/packages/ui/src/routes/__tests__/persona-context-route-jobs.test.ts apps/packages/ui/src/routes/__tests__/route-registry.persona.test.ts
git commit -m "test: lock persona context route jobs"
```

### Task 2: Separate Persona Garden From Companion

**Files:**
- Modify: `apps/packages/ui/src/routes/sidepanel-persona.tsx`
- Modify: `apps/packages/ui/src/routes/option-companion.tsx`
- Modify: `apps/packages/ui/src/components/Option/Companion/CompanionHomeShell.tsx`
- Modify: `apps/packages/ui/src/components/Option/Companion/CompanionHomePage.tsx`
- Modify: `apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx`
- Modify: `apps/packages/ui/src/routes/__tests__/option-companion.test.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/persona.spec.ts`

- [ ] **Step 1: Write failing persona and companion distinction tests**

Add assertions that:
- `/persona` exposes Persona Garden as a configuration/live assistant surface.
- `/companion` exposes Companion as a home or conversation surface.
- `Open conversation` targets `/companion/conversation`.
- Main chat links use `/chat` and are labelled separately from companion conversation.
- setup, unavailable, and degraded states use WP2 language.

- [ ] **Step 2: Run tests to verify failure or current gaps**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/sidepanel-persona.test.tsx src/routes/__tests__/option-companion.test.tsx
```

Expected:
- FAIL on missing distinction assertions, or PASS only if current UI already meets the route-job contract.

- [ ] **Step 3: Update route headings and actions**

Implementation rules:
- Preserve `PersonaGardenTabs`, `AssistantSetupWizard`, `BuddyShellRenderContext`, `usePersonaLiveVoiceController`, and `usePersonaLiveSession`.
- Do not replace Persona Visual or Buddy shell behavior.
- Do not make Companion a duplicate route to Persona Garden.
- Use existing setup and provider state data. Only change user-facing labels, hierarchy, and route handoff wiring.

Suggested action labels:
- Persona Garden: `Use Persona`, `Configure Persona`, `Test Live Assistant`.
- Companion: `Open Conversation`, `Open Main Chat`, `Review Knowledge`, `Add Media`.

- [ ] **Step 4: Run focused tests**

Run:

```bash
bunx vitest run src/routes/__tests__/sidepanel-persona.test.tsx src/routes/__tests__/option-companion.test.tsx
```

Expected:
- PASS.

- [ ] **Step 5: Run Persona browser workflow**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/persona.spec.ts --reporter=line
```

Expected:
- PASS, or any failure is documented with exact failing step and whether it is pre-existing.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/routes/sidepanel-persona.tsx apps/packages/ui/src/routes/option-companion.tsx apps/packages/ui/src/components/Option/Companion/CompanionHomeShell.tsx apps/packages/ui/src/components/Option/Companion/CompanionHomePage.tsx apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx apps/packages/ui/src/routes/__tests__/option-companion.test.tsx
git commit -m "fix: clarify persona and companion jobs"
```

### Task 3: Make Character Launch Semantics Explicit

**Files:**
- Modify: `apps/packages/ui/src/routes/option-characters.tsx`
- Modify: `apps/packages/ui/src/components/Option/Characters/CharactersWorkspace.tsx`
- Modify: `apps/packages/ui/src/components/Option/Characters/CharacterPreviewPopup.tsx`
- Modify: `apps/packages/ui/src/components/Option/Characters/useCharacterQuickChat.tsx`
- Modify or create: `apps/packages/ui/src/components/Option/Characters/__tests__/CharacterPreviewPopup.test.tsx`
- Modify or create: `apps/packages/ui/src/components/Option/Characters/__tests__/useCharacterQuickChat.test.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/journeys/character-chat.spec.ts`

- [ ] **Step 1: Write failing character launch tests**

Test expectations:
- primary launch label includes `Start in Chat` or `Use in Chat`
- popup test label includes `Test in Popup`
- new-tab label includes `Open in New Chat Tab`
- launch uses canonical `/chat` route helpers after server-chat creation
- unsupported character endpoint state uses user language first and endpoint details only as diagnostics

- [ ] **Step 2: Run tests to verify failure**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Characters/__tests__/CharacterPreviewPopup.test.tsx src/components/Option/Characters/__tests__/useCharacterQuickChat.test.tsx
```

Expected:
- FAIL if labels or route assertions are not yet implemented.

- [ ] **Step 3: Update labels and handoff routing**

Implementation rules:
- Keep `useCharacterQuickChat` as the launch path owner.
- Keep existing popup test behavior.
- Keep import, filter, edit, and gallery density.
- Use `CHAT_PATH` or `buildChatThreadPath` from `apps/packages/ui/src/routes/route-paths.ts` instead of hard-coded root routes.

- [ ] **Step 4: Run focused tests**

Run:

```bash
bunx vitest run src/components/Option/Characters/__tests__/CharacterPreviewPopup.test.tsx src/components/Option/Characters/__tests__/useCharacterQuickChat.test.tsx
```

Expected:
- PASS.

- [ ] **Step 5: Run character journey**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/journeys/character-chat.spec.ts --reporter=line
```

Expected:
- PASS and verifies character creation, start in chat, selected character, and chat API prompt payload.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/routes/option-characters.tsx apps/packages/ui/src/components/Option/Characters/CharactersWorkspace.tsx apps/packages/ui/src/components/Option/Characters/CharacterPreviewPopup.tsx apps/packages/ui/src/components/Option/Characters/useCharacterQuickChat.tsx apps/packages/ui/src/components/Option/Characters/__tests__/CharacterPreviewPopup.test.tsx apps/packages/ui/src/components/Option/Characters/__tests__/useCharacterQuickChat.test.tsx
git commit -m "fix: clarify character chat launch"
```

### Task 4: Clarify Context Asset Activation

**Files:**
- Modify: `apps/packages/ui/src/routes/option-chat-workflows.tsx`
- Modify: `apps/packages/ui/src/components/Option/ChatWorkflows/ChatWorkflowsPage.tsx`
- Modify: `apps/packages/ui/src/routes/option-dictionaries.tsx`
- Modify: `apps/packages/ui/src/components/Option/Dictionaries/DictionariesWorkspace.tsx`
- Modify: `apps/packages/ui/src/components/Option/Dictionaries/useDictionaryChatContextNavigation.ts`
- Modify: `apps/packages/ui/src/routes/option-world-books.tsx`
- Modify: `apps/packages/ui/src/components/Option/WorldBooks/WorldBooksWorkspace.tsx`
- Modify: `apps/packages/ui/src/components/Option/WorldBooks/WorldBookDetailPanel.tsx`
- Modify: `apps/packages/ui/src/routes/__tests__/chat-workflows-route.test.tsx`
- Modify or create: `apps/packages/ui/src/components/Option/Dictionaries/__tests__/useDictionaryChatContextNavigation.test.ts`
- Modify or create: `apps/packages/ui/src/components/Option/WorldBooks/__tests__/WorldBookDetailPanel.test.tsx`

- [ ] **Step 1: Write failing context activation tests**

Test expectations:
- Chat Workflows handoff labels make `/chat` continuation explicit.
- Dictionary chat context navigation does not fall back to `#/`.
- Dictionary context scope is visible in the route heading or first actionable state.
- World Book attachment actions distinguish `Open Character`, `Attach Character`, and `Start Character Chat` when a launch path exists.

- [ ] **Step 2: Run tests to verify failure**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/chat-workflows-route.test.tsx src/components/Option/Dictionaries/__tests__/useDictionaryChatContextNavigation.test.ts src/components/Option/WorldBooks/__tests__/WorldBookDetailPanel.test.tsx
```

Expected:
- FAIL if navigation or labels still use ambiguous route targets.

- [ ] **Step 3: Replace root hash fallback**

In `useDictionaryChatContextNavigation.ts`, use canonical chat route helpers:

```ts
import { CHAT_PATH, buildChatThreadPath } from "@/routes/route-paths"

export const buildDictionaryChatTarget = (serverChatId?: string): string => {
  if (serverChatId) {
    return buildChatThreadPath({ serverChatId })
  }

  return CHAT_PATH
}
```

Keep the existing state transfer before navigation. If the current file cannot import through `@/routes/route-paths`, use the repo-local relative import that matches nearby tests.

- [ ] **Step 4: Clarify context labels without reducing density**

Implementation rules:
- Preserve dictionary entry editing and filters.
- Preserve world-book entries, attachments, stats, and settings.
- Do not remove existing focus query parameters from `/characters`.
- Do not add a character chat launch from world books unless the existing character launch hook can be reused without a new backend call.

- [ ] **Step 5: Run focused tests**

Run:

```bash
bunx vitest run src/routes/__tests__/chat-workflows-route.test.tsx src/components/Option/Dictionaries/__tests__/useDictionaryChatContextNavigation.test.ts src/components/Option/WorldBooks/__tests__/WorldBookDetailPanel.test.tsx
```

Expected:
- PASS.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/routes/option-chat-workflows.tsx apps/packages/ui/src/components/Option/ChatWorkflows/ChatWorkflowsPage.tsx apps/packages/ui/src/routes/option-dictionaries.tsx apps/packages/ui/src/components/Option/Dictionaries/DictionariesWorkspace.tsx apps/packages/ui/src/components/Option/Dictionaries/useDictionaryChatContextNavigation.ts apps/packages/ui/src/routes/option-world-books.tsx apps/packages/ui/src/components/Option/WorldBooks/WorldBooksWorkspace.tsx apps/packages/ui/src/components/Option/WorldBooks/WorldBookDetailPanel.tsx apps/packages/ui/src/routes/__tests__/chat-workflows-route.test.tsx apps/packages/ui/src/components/Option/Dictionaries/__tests__/useDictionaryChatContextNavigation.test.ts apps/packages/ui/src/components/Option/WorldBooks/__tests__/WorldBookDetailPanel.test.tsx
git commit -m "fix: clarify context asset activation"
```

### Task 5: Align Agent And ACP Capability States

**Files:**
- Modify: `apps/packages/ui/src/routes/option-agents.tsx`
- Modify: `apps/packages/ui/src/routes/option-agent-tasks.tsx`
- Modify: `apps/packages/ui/src/routes/option-acp-playground.tsx`
- Modify: `apps/packages/ui/src/components/Agent`
- Modify: `apps/packages/ui/src/components/Option/AgentRegistry/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/AgentTasks/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/ACPPlayground/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/ACPPlayground/ACPPlaygroundHeader.tsx`
- Modify: `apps/packages/ui/src/components/Option/ACPPlayground/ACPSessionPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/AgentRegistry/__tests__/AgentRegistryPage.connection.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/ACPPlayground/__tests__/ACPPlayground.connection.test.tsx`
- Modify or create: `apps/packages/ui/src/components/Option/ACPPlayground/__tests__/ACPSessionPanel.test.tsx`
- Modify or create: `apps/packages/ui/src/routes/__tests__/persona-context-route-boundaries.test.tsx`

- [ ] **Step 1: Write failing route-boundary tests**

Test expectations:
- `/agents`, `/agent-tasks`, and `/acp-playground` route files use `RouteErrorBoundary`.
- Existing route labels remain unchanged.
- ACP Playground remains the launch target for agents.

- [ ] **Step 2: Write failing capability-state tests**

Test expectations:
- Agent Registry has ready, setup required, unavailable, and degraded labels.
- Agent Tasks unsupported state uses shared WP2 vocabulary and preserves recovery links.
- ACP Playground no-session state differs from ACP-unhealthy state.
- Raw endpoint/protocol details are in diagnostics or secondary detail, not the only explanation.

- [ ] **Step 3: Run tests to verify failure**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/persona-context-route-boundaries.test.tsx src/components/Option/AgentRegistry/__tests__/AgentRegistryPage.connection.test.tsx src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx src/components/Option/ACPPlayground/__tests__/ACPPlayground.connection.test.tsx src/components/Option/ACPPlayground/__tests__/ACPSessionPanel.test.tsx
```

Expected:
- FAIL if wrappers or state labels are missing.

- [ ] **Step 4: Add route boundaries**

Wrap agent option routes in `RouteErrorBoundary` following the pattern used by character and companion routes. Keep each existing `PageShell` and component tree intact.

- [ ] **Step 5: Adopt shared capability language**

Use the WP2 vocabulary:
- `empty`
- `unavailable`
- `setup_required`
- `auth_required`
- `permission_denied`
- `degraded`

Implementation rules:
- Preserve ACP health checks and launch logic.
- Preserve Agent Tasks setup links to `/agents` and `/acp-playground`.
- Keep protocol details available for operators.
- Avoid new global state stores.

- [ ] **Step 6: Run focused tests**

Run:

```bash
bunx vitest run src/routes/__tests__/persona-context-route-boundaries.test.tsx src/components/Option/AgentRegistry/__tests__/AgentRegistryPage.connection.test.tsx src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx src/components/Option/ACPPlayground/__tests__/ACPPlayground.connection.test.tsx src/components/Option/ACPPlayground/__tests__/ACPSessionPanel.test.tsx
```

Expected:
- PASS.

- [ ] **Step 7: Commit**

```bash
git add apps/packages/ui/src/routes/option-agents.tsx apps/packages/ui/src/routes/option-agent-tasks.tsx apps/packages/ui/src/routes/option-acp-playground.tsx apps/packages/ui/src/components/Agent apps/packages/ui/src/components/Option/AgentRegistry/index.tsx apps/packages/ui/src/components/Option/AgentTasks/index.tsx apps/packages/ui/src/components/Option/ACPPlayground/index.tsx apps/packages/ui/src/components/Option/ACPPlayground/ACPPlaygroundHeader.tsx apps/packages/ui/src/components/Option/ACPPlayground/ACPSessionPanel.tsx apps/packages/ui/src/routes/__tests__/persona-context-route-boundaries.test.tsx apps/packages/ui/src/components/Option/AgentRegistry/__tests__/AgentRegistryPage.connection.test.tsx apps/packages/ui/src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx apps/packages/ui/src/components/Option/ACPPlayground/__tests__/ACPPlayground.connection.test.tsx apps/packages/ui/src/components/Option/ACPPlayground/__tests__/ACPSessionPanel.test.tsx
git commit -m "fix: align agent capability states"
```

### Task 6: Route Family Browser QA

**Files:**
- Modify if needed: `apps/tldw-frontend/e2e/workflows/persona.spec.ts`
- Modify if needed: `apps/tldw-frontend/e2e/workflows/journeys/character-chat.spec.ts`
- Create if needed: `apps/tldw-frontend/e2e/workflows/persona-context-route-family.spec.ts`
- Backlog: update implementation task with before and after observations.

- [ ] **Step 1: Write or extend route-family browser checks**

Cover:
- `/persona` heading, setup/live assistant state, and primary action.
- `/characters` heading, empty/library state, and start-in-chat action.
- `/companion` heading and open-conversation action.
- `/chat-workflows` structured workflow handoff.
- `/dictionaries` chat context activation target.
- `/world-books` attached-character actions.
- `/agents` health and launch target.
- `/agent-tasks` unsupported/setup state.
- `/acp-playground` no-session and unhealthy state.

- [ ] **Step 2: Run focused E2E checks**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/persona.spec.ts e2e/workflows/journeys/character-chat.spec.ts --reporter=line
```

Expected:
- PASS for existing persona and character journeys.

- [ ] **Step 3: Run route-family E2E if added**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/persona-context-route-family.spec.ts --reporter=line
```

Expected:
- PASS, or the test is not created because focused route/component tests and browser snapshots already cover the changed pages.

- [ ] **Step 4: Capture browser observations**

Expected:
- Before and after observations record heading, primary action, capability state, and launch result for every route in scope.
- Any unverified route names the blocker, server state, and exact command attempted.

- [ ] **Step 5: Commit verification updates**

If `persona-context-route-family.spec.ts` was not created, omit that path from
the `git add` command.

```bash
git add apps/tldw-frontend/e2e/workflows/persona.spec.ts apps/tldw-frontend/e2e/workflows/journeys/character-chat.spec.ts apps/tldw-frontend/e2e/workflows/persona-context-route-family.spec.ts
git commit -m "test: verify persona context route family"
```

## Full Verification

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/route-registry.persona.test.ts src/routes/__tests__/chat-workflows-route.test.tsx src/routes/__tests__/sidepanel-persona.test.tsx
```

Run additional focused tests when touched:

```bash
bunx vitest run src/routes/__tests__/persona-context-route-jobs.test.ts src/routes/__tests__/persona-context-route-boundaries.test.tsx src/routes/__tests__/option-companion.test.tsx src/components/Option/Characters/__tests__/CharacterPreviewPopup.test.tsx src/components/Option/Characters/__tests__/useCharacterQuickChat.test.tsx src/components/Option/Dictionaries/__tests__/useDictionaryChatContextNavigation.test.ts src/components/Option/WorldBooks/__tests__/WorldBookDetailPanel.test.tsx src/components/Option/AgentRegistry/__tests__/AgentRegistryPage.connection.test.tsx src/components/Option/AgentTasks/__tests__/AgentTasksPage.connection.test.tsx src/components/Option/ACPPlayground/__tests__/ACPPlayground.connection.test.tsx src/components/Option/ACPPlayground/__tests__/ACPSessionPanel.test.tsx
```

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/persona.spec.ts e2e/workflows/journeys/character-chat.spec.ts --reporter=line
```

If route-family E2E is added, run:

```bash
bunx playwright test e2e/workflows/persona-context-route-family.spec.ts --reporter=line
```

Run from the repo root:

```bash
git diff --check
```

Expected final state:
- Users can distinguish persona, character, companion, context assets, and agents from headings and primary actions.
- Character and persona pages expose use/start flows without hiding import, edit, or test actions.
- Dictionary, world-book, and workflow routes expose activation or handoff scope.
- Agent unavailable, setup-required, and degraded states use WP2 shared capability language.
- Existing URLs, aliases, and expert workflows remain intact.

## Acceptance Criteria

- Every route in `/persona`, `/characters`, `/companion`, `/agents`, `/agent-tasks`, `/acp-playground`, `/chat-workflows`, `/dictionaries`, and `/world-books` has a distinct route job documented in code and verified by tests.
- Persona Garden, Characters, Companion, Context Assets, and Agents each have a distinct first-screen purpose.
- Character and persona routes expose a clear use/start path into the appropriate chat or live assistant surface.
- Context asset routes explain activation or attachment scope without requiring users to infer backend implementation details.
- Agent and ACP unavailable/degraded states use WP2 capability language and keep diagnostics available.
- Power-user flows remain efficient: imports, filters, edits, attachments, launch, testing, and session creation stay reachable within the same route family.
- Browser QA covers every changed route or records the exact blocker.

## Rollback Plan

- Revert route-job policy and tests if route metadata integration creates unexpected coupling.
- Revert copy/action label changes route by route if a test or browser observation shows a workflow regression.
- Revert agent route-boundary wrappers separately if they interfere with ACP session rendering.
- Preserve any test additions that accurately document existing behavior unless the behavior itself is intentionally reverted.

## Handoff Notes

- Start with route-job tests before changing visible UI; this prevents another shallow copy-only pass.
- Keep Persona Garden tied to the existing `sidepanel-persona.tsx`, Persona Visual, live voice, Buddy shell, and policy stack.
- Keep Characters tied to the existing quick-chat hook rather than building a second character launch path.
- Keep Dictionaries and World Books as context assets with explicit activation or attachment scope.
- Treat Agents, Agent Tasks, and ACP Playground as one capability family, but keep their primary jobs separate.
- Do not close this WP7 slice without browser-observed evidence for all changed pages.
