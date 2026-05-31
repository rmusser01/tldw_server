# WebUI Knowledge Workspace Transform Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Ask, Research, Workspace, and Transform surfaces understandable from their root routes while preserving the existing route paths and product intent.

**Architecture:** Add a small route-job contract for this slice, then apply that contract to the affected route shells and route-owned page components. Keep `/knowledge` as direct cited Q&A, keep `/research` as a research-run console, and reuse the existing workspace and transform tool runtimes rather than introducing a new artifact system.

**Tech Stack:** React, Next.js pages, shared `apps/packages/ui` route shells, route registry metadata, Vitest, React Testing Library, Playwright.

---

## Source Documents

- Backlog task: `TASK-418.6`
- Parent implementation plan: `Docs/superpowers/plans/2026-05-17-webui-extension-ux-remediation-implementation-plan.md`
- UX remediation spec: `Docs/superpowers/specs/2026-05-17-webui-extension-ux-remediation-program-design.md`
- Dependency plans:
  - `Docs/superpowers/plans/2026-05-17-webui-route-contract-visibility-implementation-plan.md`
  - `Docs/superpowers/plans/2026-05-17-webui-capability-error-state-implementation-plan.md`
  - `Docs/superpowers/plans/2026-05-17-webui-responsive-landmarks-implementation-plan.md`
  - `Docs/superpowers/plans/2026-05-17-webui-chat-global-chrome-implementation-plan.md`

## Audit Findings Addressed

- `F14`: Users cannot predict whether to use Ask, Research, Workspace, or a tool-specific transform page.
- `F1 support`: Several route labels, subtitles, and aliases expose product history instead of current user jobs.
- `F2 support`: Empty states and setup states do not always describe the next usable action.
- `F15 support`: Advanced controls exist, but the main path and expert path are not consistently separated.

## Route Inventory And Product Ladder

| Route | Ladder concept | Primary user goal | Existing ownership | UX contract |
| --- | --- | --- | --- | --- |
| `/knowledge` | Ask | Ask a direct question and inspect cited answers from selected sources | `apps/packages/ui/src/routes/option-knowledge.tsx`, `KnowledgeQA` | Keep it direct cited Q&A. Do not turn it into a generic knowledge-management hub. |
| `/search` | Alias | Reach the Ask surface from a search mental model | `apps/tldw-frontend/pages/search.tsx` | Preserve redirect or alias behavior and make canonical route ownership explicit in tests. |
| `/research` | Research | Start, resume, inspect, checkpoint, and export long-running research runs | `apps/tldw-frontend/pages/research.tsx` | Use user-language research-run framing and keep live run, trust, artifact, and bundle controls visible. |
| `/workspace-playground` | Workspace | Work from sources into chat and generated workspace artifacts | `apps/packages/ui/src/routes/option-workspace-playground.tsx`, `WorkspacePlayground` | Keep the route path. Use one canonical label, with Studio reserved for the artifact pane. |
| `/chat-workspace` | Workspace | Chat against a bounded workspace with staged sources and context | `apps/packages/ui/src/routes/option-chat-workspace.tsx`, `ChatWorkspacePage` | Show workspace readiness, source scope, and recovery actions before advanced rails. |
| `/document-workspace` | Workspace | Read, inspect, and work with a document-centered workspace | `apps/packages/ui/src/routes/option-document-workspace.tsx`, `DocumentWorkspacePage` | Keep document identity, sync state, and panel purpose clear. |
| `/repo2txt` | Transform | Convert a repository or local files into prompt-ready text | `apps/packages/ui/src/routes/option-repo2txt.tsx`, `Repo2TxtPage` | Make input, selection, generated output, copy, and download states explicit. |
| `/model-playground` | Transform | Test model prompts, compare outputs, and tune parameters | `apps/packages/ui/src/routes/option-model-playground.tsx`, `ModelPlayground` | Separate primary run path from parameter and debug tools. |
| `/writing-playground` | Transform | Draft, transform, inspect, and continue writing sessions | `apps/packages/ui/src/routes/option-writing-playground.tsx`, `WritingPlayground` | Preserve dense writer controls while clarifying session, model, generation, save, and recovery state. |
| `/presentation-studio` | Transform | Create and edit generated presentation projects | `apps/packages/ui/src/routes/option-presentation-studio.tsx`, `PresentationStudio` | Keep start, project detail, export, asset, and error states consistent across child routes. |

Related presentation routes stay under the same product contract:

- `/presentation-studio/new`
- `/presentation-studio/start`
- `/presentation-studio/:projectId`

## Non-Goals

- Do not rename route paths.
- Do not create `/workspace-studio`.
- Do not redesign the research engine.
- Do not create a new artifact model or persistence system.
- Do not change backend APIs for this slice.
- Do not move `/audiobook-studio`; WP11 owns that route.
- Do not add a broad visual redesign or new design system.
- Do not replace working tool runtimes with a new generic tool shell.

## File Structure

### New Files

- `apps/packages/ui/src/routes/knowledge-workspace-transform-route-jobs.ts`
  - Owns the route-job metadata for WP9 routes.
  - Exposes canonical labels, ladder concept, primary action label, output kind, and alias targets.
- `apps/packages/ui/src/routes/__tests__/knowledge-workspace-transform-route-jobs.test.ts`
  - Verifies every WP9 route has an owned route-job entry.
  - Verifies `/search` aliases to `/knowledge`.
  - Verifies `/workspace-playground` keeps its path while using the canonical workspace label.
- `apps/packages/ui/src/routes/__tests__/knowledge-workspace-transform-route-boundaries.test.tsx`
  - Verifies route wrappers expose error boundaries and page titles consistently where the local route shell owns the wrapper.

### Modified Files

- `apps/packages/ui/src/routes/option-knowledge.tsx`
  - Apply Ask route metadata without changing the `KnowledgeQA` purpose.
- `apps/packages/ui/src/routes/option-workspace-playground.tsx`
  - Add route error boundary if missing.
  - Apply canonical route label and workspace no-state framing.
- `apps/packages/ui/src/routes/option-chat-workspace.tsx`
  - Add route error boundary if missing.
  - Apply workspace-scoped heading and readiness copy.
- `apps/packages/ui/src/routes/option-document-workspace.tsx`
  - Keep existing route error boundary.
  - Align page label and document-workspace no-state wording.
- `apps/packages/ui/src/routes/option-repo2txt.tsx`
  - Add route shell consistency and route error boundary if missing.
  - Keep `Repo2TxtPage` as the runtime owner.
- `apps/packages/ui/src/routes/option-model-playground.tsx`
  - Apply Transform route metadata and route error boundary if missing.
- `apps/packages/ui/src/routes/option-writing-playground.tsx`
  - Apply Transform route metadata and route error boundary if missing.
- `apps/packages/ui/src/routes/option-presentation-studio.tsx`
  - Align root route metadata with presentation project workflow.
- `apps/packages/ui/src/routes/option-presentation-studio-new.tsx`
  - Align child route metadata with new-project workflow.
- `apps/packages/ui/src/routes/option-presentation-studio-start.tsx`
  - Align child route metadata with extension start workflow.
- `apps/packages/ui/src/routes/option-presentation-studio-detail.tsx`
  - Align child route metadata with project detail workflow.
- `apps/packages/ui/src/components/Option/KnowledgeQA/index.tsx`
  - Keep Ask path and cited-answer intent visible in the first viewport.
- `apps/packages/ui/src/components/Option/KnowledgeQA/SettingsPanel.tsx`
  - Keep source and citation controls discoverable after the main question path.
- `apps/packages/ui/src/components/Option/WorkspacePlayground/WorkspaceHeader.tsx`
  - Own canonical workspace label and output/export affordances.
- `apps/packages/ui/src/components/Option/WorkspacePlayground/SourcesPane.tsx`
  - Clarify source no-state and source-add recovery.
- `apps/packages/ui/src/components/Option/WorkspacePlayground/ChatPane.tsx`
  - Keep source scope and cited response state visible.
- `apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane.tsx`
  - Preserve artifact actions and make output persistence visible.
- `apps/packages/ui/src/components/Option/ChatWorkspace/ChatWorkspacePage.tsx`
  - Clarify workspace readiness and selected-context state.
- `apps/packages/ui/src/components/DocumentWorkspace/DocumentWorkspacePage.tsx`
  - Clarify document-open, sync, and side-panel states.
- `apps/packages/ui/src/components/Option/Repo2Txt/Repo2TxtPage.tsx`
  - Clarify input source, selected files, generation readiness, and output persistence.
- `apps/packages/ui/src/components/Option/Repo2Txt/components/OutputPanel.tsx`
  - Keep Generate, Copy, Download, and output preview states explicit.
- `apps/packages/ui/src/components/Option/ModelPlayground/index.tsx`
  - Clarify primary Run path, compare mode, parameter drawer, and debug panel.
- `apps/packages/ui/src/components/Option/ModelPlayground/ModelPlaygroundChat.tsx`
  - Clarify empty state and selected-model readiness.
- `apps/packages/ui/src/components/Option/WritingPlayground/index.tsx`
  - Clarify session selection, model readiness, generation status, and save state.
- `apps/packages/ui/src/components/Option/PresentationStudio/PresentationStudioPage.tsx`
  - Clarify project list, generation status, and project recovery.
- `apps/packages/ui/src/components/Option/PresentationStudio/ExtensionStartPanel.tsx`
  - Clarify seed input, server readiness, and project creation errors.
- `apps/tldw-frontend/pages/search.tsx`
  - Keep `/search` as an explicit alias to `/knowledge` and document that in tests.
- `apps/tldw-frontend/pages/research.tsx`
  - Preserve existing run console logic while making Start, Resume, Checkpoint, Trust, Artifact, Bundle, and Back to Chat labels consistent.

### Existing Tests To Extend

- `apps/tldw-frontend/e2e/workflows/knowledge-qa.spec.ts`
- `apps/tldw-frontend/e2e/workflows/workspace-playground.spec.ts`
- `apps/tldw-frontend/e2e/workflows/tier-2-features/document-workspace.spec.ts`
- `apps/tldw-frontend/e2e/workflows/tier-5-specialized/model-playground.spec.ts`
- `apps/tldw-frontend/e2e/workflows/search.spec.ts`
- `apps/tldw-frontend/e2e/workflows/tier-5-specialized/repo2txt.spec.ts`
- `apps/tldw-frontend/e2e/workflows/tier-2-features/writing-playground.spec.ts`
- `apps/tldw-frontend/e2e/ux-audit/presentation-studio.spec.ts`

### New Tests If Needed

- `apps/tldw-frontend/e2e/workflows/research.spec.ts`
  - Add only if the existing E2E suite does not already cover research-run root behavior.

## Implementation Strategy

Use a route-job metadata contract as the smallest shared layer. The metadata helps tests assert route intent, but each route still owns the actual UI wording and state because these surfaces have different workflows.

The route-job contract must not become a design system. It is a small inventory object for this remediation slice.

Suggested type shape:

```ts
export type KnowledgeWorkspaceTransformConcept =
  | "ask"
  | "alias"
  | "research"
  | "workspace"
  | "transform"

export type KnowledgeWorkspaceTransformRouteJob = {
  route: string
  concept: KnowledgeWorkspaceTransformConcept
  label: string
  primaryJob: string
  primaryActionLabel: string
  canonicalRoute?: string
  outputKind?: string
  relatedRoutes?: string[]
}
```

The object must include all root routes in this plan:

```ts
export const KNOWLEDGE_WORKSPACE_TRANSFORM_ROUTE_JOBS: KnowledgeWorkspaceTransformRouteJob[] = [
  {
    route: "/knowledge",
    concept: "ask",
    label: "Knowledge Q&A",
    primaryJob: "Ask a direct question with cited answers",
    primaryActionLabel: "Ask question",
    outputKind: "cited answer"
  },
  {
    route: "/search",
    concept: "alias",
    label: "Search",
    primaryJob: "Reach Knowledge Q&A from search",
    primaryActionLabel: "Search knowledge",
    canonicalRoute: "/knowledge",
    outputKind: "cited answer"
  },
  {
    route: "/research",
    concept: "research",
    label: "Research Runs",
    primaryJob: "Run and review long-running research",
    primaryActionLabel: "Start run",
    outputKind: "research bundle"
  },
  {
    route: "/workspace-playground",
    concept: "workspace",
    label: "Workspace Playground",
    primaryJob: "Work from sources into chat and artifacts",
    primaryActionLabel: "Add sources",
    outputKind: "workspace artifact"
  },
  {
    route: "/chat-workspace",
    concept: "workspace",
    label: "Chat Workspace",
    primaryJob: "Chat with scoped workspace context",
    primaryActionLabel: "Ask workspace",
    outputKind: "workspace answer"
  },
  {
    route: "/document-workspace",
    concept: "workspace",
    label: "Document Workspace",
    primaryJob: "Open and work with a document",
    primaryActionLabel: "Open document",
    outputKind: "document workspace"
  },
  {
    route: "/repo2txt",
    concept: "transform",
    label: "Repo2Txt",
    primaryJob: "Convert repository files into prompt-ready text",
    primaryActionLabel: "Generate output",
    outputKind: "text export"
  },
  {
    route: "/model-playground",
    concept: "transform",
    label: "Model Playground",
    primaryJob: "Test prompts and compare model behavior",
    primaryActionLabel: "Run prompt",
    outputKind: "model response"
  },
  {
    route: "/writing-playground",
    concept: "transform",
    label: "Writing Playground",
    primaryJob: "Draft and transform writing sessions",
    primaryActionLabel: "Generate",
    outputKind: "writing continuation"
  },
  {
    route: "/presentation-studio",
    concept: "transform",
    label: "Presentation Studio",
    primaryJob: "Create and edit generated presentation projects",
    primaryActionLabel: "Start presentation",
    outputKind: "presentation project",
    relatedRoutes: [
      "/presentation-studio/new",
      "/presentation-studio/start",
      "/presentation-studio/:projectId"
    ]
  }
]
```

If any label conflicts with already approved WP1 route labels, use the WP1 label and update this metadata test to match that source of truth.

## Task 1: Add Route-Job Contract Tests

**Files:**
- Create: `apps/packages/ui/src/routes/knowledge-workspace-transform-route-jobs.ts`
- Create: `apps/packages/ui/src/routes/__tests__/knowledge-workspace-transform-route-jobs.test.ts`
- Modify: `apps/packages/ui/src/routes/route-registry.tsx`
- Modify: `apps/packages/ui/src/routes/route-paths.ts`

- [ ] **Step 1: Write failing coverage test for all WP9 routes**

Test that every scoped route has an entry:

```ts
import { KNOWLEDGE_WORKSPACE_TRANSFORM_ROUTE_JOBS } from "../knowledge-workspace-transform-route-jobs"

const requiredRoutes = [
  "/knowledge",
  "/search",
  "/research",
  "/workspace-playground",
  "/chat-workspace",
  "/document-workspace",
  "/repo2txt",
  "/model-playground",
  "/writing-playground",
  "/presentation-studio"
]

it("defines the WP9 route-job inventory", () => {
  const routes = new Set(KNOWLEDGE_WORKSPACE_TRANSFORM_ROUTE_JOBS.map((job) => job.route))

  for (const route of requiredRoutes) {
    expect(routes.has(route)).toBe(true)
  }
})
```

- [ ] **Step 2: Write failing alias test for `/search`**

```ts
it("keeps search as an alias to Knowledge Q&A", () => {
  const searchJob = KNOWLEDGE_WORKSPACE_TRANSFORM_ROUTE_JOBS.find((job) => job.route === "/search")

  expect(searchJob).toMatchObject({
    concept: "alias",
    canonicalRoute: "/knowledge",
    outputKind: "cited answer"
  })
})
```

- [ ] **Step 3: Write failing canonical workspace label test**

```ts
it("keeps workspace playground path while owning a canonical label", () => {
  const workspaceJob = KNOWLEDGE_WORKSPACE_TRANSFORM_ROUTE_JOBS.find(
    (job) => job.route === "/workspace-playground"
  )

  expect(workspaceJob).toMatchObject({
    concept: "workspace",
    label: "Workspace Playground"
  })
})
```

- [ ] **Step 4: Run route-job tests and verify failure**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/knowledge-workspace-transform-route-jobs.test.ts
```

Expected: FAIL because the route-job module does not exist.

- [ ] **Step 5: Add the route-job module**

Create `apps/packages/ui/src/routes/knowledge-workspace-transform-route-jobs.ts` with the type shape and inventory from the Implementation Strategy section.

- [ ] **Step 6: Run route-job tests and verify pass**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/knowledge-workspace-transform-route-jobs.test.ts
```

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

```bash
git add apps/packages/ui/src/routes/knowledge-workspace-transform-route-jobs.ts apps/packages/ui/src/routes/__tests__/knowledge-workspace-transform-route-jobs.test.ts
git commit -m "test: add webui route job contract"
```

## Task 2: Preserve Knowledge Ask And Search Alias Behavior

**Files:**
- Modify: `apps/packages/ui/src/routes/option-knowledge.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/KnowledgeQA/SettingsPanel.tsx`
- Modify: `apps/tldw-frontend/pages/search.tsx`
- Test: `apps/packages/ui/src/routes/__tests__/knowledge-workspace-transform-route-boundaries.test.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/knowledge-qa.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/search.spec.ts`

- [ ] **Step 1: Write failing route boundary test for Knowledge Q&A**

Assert that the `/knowledge` wrapper still presents the route as Q&A and uses the existing `RouteErrorBoundary`.

Expected text assertions:

- `Knowledge QA` or the WP1-approved canonical equivalent.
- `Ask question` or the route-owned primary action from the route-job contract.
- Citation or source scope text owned by `KnowledgeQA`.

- [ ] **Step 2: Write failing Playwright assertion for first-time Ask path**

Extend `knowledge-qa.spec.ts` to assert:

- The first viewport exposes a question input.
- The primary action is visible without opening settings.
- Source scope is visible before citation settings.
- Empty or no-source state gives a next action that does not require knowing backend implementation details.

- [ ] **Step 3: Write failing Playwright assertion for `/search`**

Extend `search.spec.ts` to assert:

- Visiting `/search` lands on the canonical Ask surface.
- Browser URL behavior matches the current redirect or alias implementation.
- The user sees the same primary Ask action as `/knowledge`.

- [ ] **Step 4: Run the failing tests**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/knowledge-qa.spec.ts e2e/workflows/search.spec.ts --reporter=line
```

Expected: FAIL on the new visibility and alias assertions.

- [ ] **Step 5: Apply minimal UI copy and structure changes**

Keep `/knowledge` direct Q&A:

- Do not add generic library-management framing.
- Keep settings and citation controls available, but behind the primary Ask path.
- Make source scope visible near the question composer.
- Keep error and no-source states focused on what the user can do next.
- Keep diagnostics available through existing controls, not as the first-time path.

- [ ] **Step 6: Re-run tests**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/knowledge-qa.spec.ts e2e/workflows/search.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

```bash
git add apps/packages/ui/src/routes/option-knowledge.tsx apps/packages/ui/src/components/Option/KnowledgeQA apps/tldw-frontend/pages/search.tsx apps/tldw-frontend/e2e/workflows/knowledge-qa.spec.ts apps/tldw-frontend/e2e/workflows/search.spec.ts
git commit -m "fix: clarify knowledge ask route"
```

## Task 3: Reframe Research As A Research-Run Console

**Files:**
- Modify: `apps/tldw-frontend/pages/research.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/research.spec.ts`

- [ ] **Step 1: Write failing research root E2E test**

Create the E2E file only if no existing workflow test already covers `/research`.

Required assertions:

- The root page identifies the surface as research runs, not generic search.
- The primary action starts a run from a research question.
- Existing runs are presented as recoverable work, not only newly created items.
- Selected run state shows phase, progress, checkpoint, trust, artifacts, and bundle sections.
- Pause, Resume, Cancel, Refresh, and Back to Chat controls are visible when the selected run state supports them.

- [ ] **Step 2: Run the failing research test**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/research.spec.ts --reporter=line
```

Expected: FAIL on the new route framing assertions.

- [ ] **Step 3: Update the research page framing**

In `apps/tldw-frontend/pages/research.tsx`:

- Rename "Deep Research" framing only if WP1 does not already own that label.
- Replace "Newly created runs" with language that covers resumed and historical runs.
- Keep `Start run` as the primary creation action.
- Keep run control labels user-facing: Refresh, Pause, Resume, Cancel, Back to Chat.
- Keep raw checkpoint JSON only as fallback when the typed editor cannot render the checkpoint.
- Keep trust details, artifacts, and bundle sections visible under selected run state.
- Preserve all API calls and query keys.

- [ ] **Step 4: Re-run the research test**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/research.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 5: Commit Task 3**

```bash
git add apps/tldw-frontend/pages/research.tsx apps/tldw-frontend/e2e/workflows/research.spec.ts
git commit -m "fix: clarify research run console"
```

## Task 4: Clarify Workspace Playground As The Canonical Workspace Shell

**Files:**
- Modify: `apps/packages/ui/src/routes/option-workspace-playground.tsx`
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/WorkspaceHeader.tsx`
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/SourcesPane.tsx`
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/ChatPane.tsx`
- Modify: `apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/workspace-playground.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/workspace-playground.output-matrix.probe.spec.ts`

- [ ] **Step 1: Write failing workspace E2E assertions**

Extend `workspace-playground.spec.ts` to assert:

- The route label is consistent with the route-job contract.
- The first empty state offers source addition and existing workspace recovery where current data supports it.
- Source scope is visible before asking workspace questions.
- Studio or output actions describe persistence and export without inventing a new artifact model.
- Mobile viewport keeps source, chat, and studio navigation reachable without overlapping controls.

- [ ] **Step 2: Run the failing workspace tests**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/workspace-playground.spec.ts --reporter=line
```

Expected: FAIL on the new route label or empty-state assertions.

- [ ] **Step 3: Update route wrapper and workspace components**

Apply minimal route-shell changes:

- Add `RouteErrorBoundary` if the wrapper does not already have one.
- Keep `/workspace-playground` as the route path.
- Use `Workspace Playground` as the route label unless WP1 selected a different canonical label.
- Use Studio only for the artifact/output pane, not as a new route identity.
- Preserve existing source, chat, artifact, citation, and export behavior.
- Do not introduce another workspace store.

- [ ] **Step 4: Re-run workspace tests**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/workspace-playground.spec.ts e2e/workflows/workspace-playground.output-matrix.probe.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

```bash
git add apps/packages/ui/src/routes/option-workspace-playground.tsx apps/packages/ui/src/components/Option/WorkspacePlayground apps/tldw-frontend/e2e/workflows/workspace-playground.spec.ts apps/tldw-frontend/e2e/workflows/workspace-playground.output-matrix.probe.spec.ts
git commit -m "fix: clarify workspace playground shell"
```

## Task 5: Align Chat Workspace And Document Workspace States

**Files:**
- Modify: `apps/packages/ui/src/routes/option-chat-workspace.tsx`
- Modify: `apps/packages/ui/src/routes/option-document-workspace.tsx`
- Modify: `apps/packages/ui/src/components/Option/ChatWorkspace/ChatWorkspacePage.tsx`
- Modify: `apps/packages/ui/src/components/DocumentWorkspace/DocumentWorkspacePage.tsx`
- Test: `apps/packages/ui/src/components/Option/ChatWorkspace/ChatWorkspacePage.test.tsx`
- Test: `apps/packages/ui/src/components/Option/ChatWorkspace/WorkspaceChatPanel.test.tsx`
- Test: `apps/packages/ui/src/components/DocumentWorkspace/__tests__/DocumentWorkspacePage.test.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/tier-2-features/document-workspace.spec.ts`

- [ ] **Step 1: Write failing chat workspace tests**

Assert:

- The page communicates whether the workspace is ready.
- A user can see selected sources or context before sending a message.
- Recovery action is visible when no workspace is selected.
- Expert rails are discoverable without taking over the main path.

- [ ] **Step 2: Write failing document workspace tests**

Assert:

- A user can tell whether a document is open.
- Sync status and recovery are visible.
- Left and right panels have route-specific purpose labels.
- Empty document state directs the user to open or select a document.

- [ ] **Step 3: Run component tests and verify failure**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/ChatWorkspace/ChatWorkspacePage.test.tsx src/components/Option/ChatWorkspace/WorkspaceChatPanel.test.tsx src/components/DocumentWorkspace/__tests__/DocumentWorkspacePage.test.tsx
```

Expected: FAIL on the new user-state assertions.

- [ ] **Step 4: Update route wrappers and states**

- Add `RouteErrorBoundary` to `option-chat-workspace.tsx` if missing.
- Keep `option-document-workspace.tsx` boundary intact.
- Make workspace readiness visible in the header or status strip.
- Keep advanced rails and inspectors accessible after the main selected-context path.
- Preserve current stores, query keys, and keyboard behavior.

- [ ] **Step 5: Re-run component tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/ChatWorkspace/ChatWorkspacePage.test.tsx src/components/Option/ChatWorkspace/WorkspaceChatPanel.test.tsx src/components/DocumentWorkspace/__tests__/DocumentWorkspacePage.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Re-run document E2E**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/tier-2-features/document-workspace.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 7: Commit Task 5**

```bash
git add apps/packages/ui/src/routes/option-chat-workspace.tsx apps/packages/ui/src/routes/option-document-workspace.tsx apps/packages/ui/src/components/Option/ChatWorkspace apps/packages/ui/src/components/DocumentWorkspace apps/tldw-frontend/e2e/workflows/tier-2-features/document-workspace.spec.ts
git commit -m "fix: clarify workspace readiness states"
```

## Task 6: Clarify Transform Tool Inputs, Outputs, Persistence, And Export

**Files:**
- Modify: `apps/packages/ui/src/routes/option-repo2txt.tsx`
- Modify: `apps/packages/ui/src/routes/option-model-playground.tsx`
- Modify: `apps/packages/ui/src/routes/option-writing-playground.tsx`
- Modify: `apps/packages/ui/src/routes/option-presentation-studio.tsx`
- Modify: `apps/packages/ui/src/routes/option-presentation-studio-new.tsx`
- Modify: `apps/packages/ui/src/routes/option-presentation-studio-start.tsx`
- Modify: `apps/packages/ui/src/routes/option-presentation-studio-detail.tsx`
- Modify: `apps/packages/ui/src/components/Option/Repo2Txt/Repo2TxtPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/Repo2Txt/components/OutputPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModelPlayground/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/ModelPlayground/ModelPlaygroundChat.tsx`
- Modify: `apps/packages/ui/src/components/Option/WritingPlayground/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/PresentationStudio/PresentationStudioPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/PresentationStudio/ExtensionStartPanel.tsx`
- Test: `apps/packages/ui/src/components/Option/Repo2Txt/__tests__/Repo2TxtPage.flow.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Repo2Txt/__tests__/Repo2TxtPage.smoke.test.tsx`
- Test: `apps/tldw-frontend/e2e/workflows/tier-5-specialized/repo2txt.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/tier-5-specialized/model-playground.spec.ts`
- Test: `apps/tldw-frontend/e2e/workflows/tier-2-features/writing-playground.spec.ts`
- Test: `apps/tldw-frontend/e2e/ux-audit/presentation-studio.spec.ts`

- [ ] **Step 1: Write failing Repo2Txt transform tests**

Assert:

- Source input is visible.
- Generate is disabled until a source is loaded.
- Selected file count is visible.
- Output panel exposes Generate, Copy, Download, and output preview state.
- Error state says what failed and what the next recovery action is.

- [ ] **Step 2: Write failing Model Playground transform tests**

Assert:

- Primary prompt run path is visible without opening debug.
- Compare mode remains discoverable.
- Parameter sidebar has a named toggle and persists state.
- Debug panel stays available to experts.
- Empty state says model selection or prompt input is required.

- [ ] **Step 3: Write failing Writing Playground transform tests**

Assert:

- Session state is visible.
- Model readiness is visible.
- Generate and Stop controls reflect current generation state.
- Save state and recovery state remain visible.
- Advanced settings and analysis tools are discoverable behind existing toggles or menus.

- [ ] **Step 4: Write failing Presentation Studio transform tests**

Assert:

- Root page identifies project creation and project recovery.
- New/start routes identify seed inputs and server readiness.
- Detail route identifies generation status, assets, export, and recoverable errors.
- Child routes share the same presentation product identity.

- [ ] **Step 5: Run transform tests and verify failure**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Repo2Txt/__tests__/Repo2TxtPage.flow.test.tsx src/components/Option/Repo2Txt/__tests__/Repo2TxtPage.smoke.test.tsx
```

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/tier-5-specialized/repo2txt.spec.ts e2e/workflows/tier-5-specialized/model-playground.spec.ts e2e/workflows/tier-2-features/writing-playground.spec.ts e2e/ux-audit/presentation-studio.spec.ts --reporter=line
```

Expected: FAIL on new transform framing assertions.

- [ ] **Step 6: Update route wrappers**

- Add `RouteErrorBoundary` to `option-repo2txt.tsx` if missing.
- Add `RouteErrorBoundary` to `option-model-playground.tsx` if missing.
- Add `RouteErrorBoundary` to `option-writing-playground.tsx` if missing.
- Keep existing presentation studio route boundaries.
- Keep each tool runtime in its existing component tree.

- [ ] **Step 7: Update transform component framing**

Use the same user-language pattern across transform tools:

- Input: what the tool needs now.
- Output: what the tool will produce.
- Persistence: where the output, session, or project lives.
- Export: copy, download, bundle, or presentation export controls.
- Recovery: what to do when the server, source, model, or project is unavailable.

- [ ] **Step 8: Re-run transform tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/Repo2Txt/__tests__/Repo2TxtPage.flow.test.tsx src/components/Option/Repo2Txt/__tests__/Repo2TxtPage.smoke.test.tsx
```

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/tier-5-specialized/repo2txt.spec.ts e2e/workflows/tier-5-specialized/model-playground.spec.ts e2e/workflows/tier-2-features/writing-playground.spec.ts e2e/ux-audit/presentation-studio.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 9: Commit Task 6**

```bash
git add apps/packages/ui/src/routes/option-repo2txt.tsx apps/packages/ui/src/routes/option-model-playground.tsx apps/packages/ui/src/routes/option-writing-playground.tsx apps/packages/ui/src/routes/option-presentation-studio.tsx apps/packages/ui/src/routes/option-presentation-studio-new.tsx apps/packages/ui/src/routes/option-presentation-studio-start.tsx apps/packages/ui/src/routes/option-presentation-studio-detail.tsx apps/packages/ui/src/components/Option/Repo2Txt apps/packages/ui/src/components/Option/ModelPlayground apps/packages/ui/src/components/Option/WritingPlayground apps/packages/ui/src/components/Option/PresentationStudio apps/tldw-frontend/e2e/workflows/tier-5-specialized/repo2txt.spec.ts apps/tldw-frontend/e2e/workflows/tier-5-specialized/model-playground.spec.ts apps/tldw-frontend/e2e/workflows/tier-2-features/writing-playground.spec.ts apps/tldw-frontend/e2e/ux-audit/presentation-studio.spec.ts
git commit -m "fix: clarify transform tool workflows"
```

## Task 7: Browser QA And Final Verification

**Files:**
- Verify only unless the browser run exposes a defect in scoped files.

- [ ] **Step 1: Run route contract tests**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/routes/__tests__/knowledge-workspace-transform-route-jobs.test.ts src/routes/__tests__/knowledge-workspace-transform-route-boundaries.test.tsx
```

Expected: PASS.

- [ ] **Step 2: Run parent-required E2E command**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/knowledge-qa.spec.ts e2e/workflows/workspace-playground.spec.ts e2e/workflows/tier-2-features/document-workspace.spec.ts e2e/workflows/tier-5-specialized/model-playground.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 3: Run expanded route coverage**

Run from `apps/tldw-frontend`:

```bash
bunx playwright test e2e/workflows/search.spec.ts e2e/workflows/research.spec.ts e2e/workflows/tier-5-specialized/repo2txt.spec.ts e2e/workflows/tier-2-features/writing-playground.spec.ts e2e/ux-audit/presentation-studio.spec.ts --reporter=line
```

Expected: PASS.

- [ ] **Step 4: Browser QA first-time path**

Use a local dev server and inspect:

- `/knowledge`: question input, source scope, ask action, empty state.
- `/research`: run creation, recoverable run list, selected run status, checkpoint, trust, artifact, bundle.
- `/workspace-playground`: source no-state, workspace header, chat source scope, studio artifact actions.
- `/chat-workspace`: readiness state, staged context, expert rails.
- `/document-workspace`: document open state, sync state, panels.
- `/repo2txt`: provider input, selected file count, generate state, output copy/download.
- `/model-playground`: prompt path, compare mode, parameters, debug.
- `/writing-playground`: session state, generation state, save state.
- `/presentation-studio`: project start, project recovery, detail status, export.

Expected: Each route exposes the main action before expert controls.

- [ ] **Step 5: Browser QA power-user path**

At desktop and mobile widths, inspect:

- Settings and expert controls remain reachable.
- Repeated actions do not require returning to route navigation.
- Recovery paths are next to the state that failed.
- Controls do not overlap or hide primary actions.
- Status labels update without requiring page reload.

Expected: Returning users can continue work quickly.

- [ ] **Step 6: Accessibility checks**

Use Playwright accessibility snapshots or Testing Library assertions for:

- Page-level heading.
- Named primary action.
- Landmark or route root.
- Visible focus state for route-critical controls.
- Status regions for long-running runs, generation, save, and output states.

Expected: No new unlabeled primary controls in the touched routes.

- [ ] **Step 7: Final commit**

```bash
git status --short
git add apps/packages/ui/src/routes apps/packages/ui/src/components/Option/KnowledgeQA apps/packages/ui/src/components/Option/WorkspacePlayground apps/packages/ui/src/components/Option/ChatWorkspace apps/packages/ui/src/components/DocumentWorkspace apps/packages/ui/src/components/Option/Repo2Txt apps/packages/ui/src/components/Option/ModelPlayground apps/packages/ui/src/components/Option/WritingPlayground apps/packages/ui/src/components/Option/PresentationStudio apps/tldw-frontend/pages/search.tsx apps/tldw-frontend/pages/research.tsx apps/tldw-frontend/e2e
git commit -m "fix: align ask research workspace transform routes"
```

Expected: Commit contains only WP9 scoped files.

## Acceptance Criteria

- `/knowledge` remains direct cited Q&A.
- `/search` is explicitly tested as an alias or redirect to the Ask surface.
- `/research` uses research-run language and keeps live run, checkpoint, trust, artifact, bundle, and Back to Chat controls visible.
- `/workspace-playground` keeps its route path and has one canonical label policy.
- `/chat-workspace` and `/document-workspace` communicate workspace readiness and recovery paths.
- `/repo2txt`, `/model-playground`, `/writing-playground`, and `/presentation-studio` each explain input, output, persistence, export, and recovery through route-owned states.
- Advanced controls stay available without dominating the first-time path.
- All changed routes pass desktop and mobile browser QA.
- No backend API changes are introduced.

## Verification Commands

Run route unit tests:

```bash
cd apps/packages/ui
bunx vitest run src/routes/__tests__/knowledge-workspace-transform-route-jobs.test.ts src/routes/__tests__/knowledge-workspace-transform-route-boundaries.test.tsx
```

Run parent-required E2E:

```bash
cd apps/tldw-frontend
bunx playwright test e2e/workflows/knowledge-qa.spec.ts e2e/workflows/workspace-playground.spec.ts e2e/workflows/tier-2-features/document-workspace.spec.ts e2e/workflows/tier-5-specialized/model-playground.spec.ts --reporter=line
```

Run expanded WP9 E2E:

```bash
cd apps/tldw-frontend
bunx playwright test e2e/workflows/search.spec.ts e2e/workflows/research.spec.ts e2e/workflows/tier-5-specialized/repo2txt.spec.ts e2e/workflows/tier-2-features/writing-playground.spec.ts e2e/ux-audit/presentation-studio.spec.ts --reporter=line
```

## Rollback Plan

- Revert route-job metadata first if it causes route registry conflicts.
- Revert route-wrapper boundary changes route by route.
- Keep component-level wording changes independent so one tool can be rolled back without affecting the others.
- Do not roll back unrelated WP1, WP2, WP4, or WP6 changes while isolating WP9 defects.

## Handoff Notes

- Start with tests that lock product identity. The wording changes are small, but the risk is route drift.
- Treat `/knowledge` as Ask, `/research` as Research, workspace routes as Workspace, and playground/studio tools as Transform.
- Reuse existing stores, API clients, query keys, route paths, and component boundaries.
- Keep browser-observed evidence in the implementation PR or task notes, especially for mobile and empty states.
