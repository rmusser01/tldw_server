# Research Workspace NotebookLM Agent Tasks WP4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add NotebookLM Ultra-style Research Workspace agent task entrypoints from Chat and Studio while routing through the existing governed ACP/agent-task handoff.

**Architecture:** Reuse the existing `WorkspaceAgentTaskHandoffModal` and page-level handoff signal instead of creating a second task system. Chat and Studio only build bounded task context; the modal creates the canonical bridge/project/task and stores context under a namespaced metadata key. Existing Agent Tasks and ACP run history remain the governed execution surface; completed run results can be saved back as traceable Studio artifacts with ACP provenance and version metadata.

**Tech Stack:** React, TypeScript, Zustand workspace store, Ant Design, lucide-react, Vitest, Testing Library.

---

## File Structure

- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceAgentTaskHandoffModal.tsx`
  - Extend `WorkspaceAgentTaskPrefill` with optional `metadata`.
  - Include a short governed-task notice in the modal.
  - Attach prefill metadata to project/task metadata under `research_workspace_task_context`.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceACPHistoryModal.tsx`
  - Expose observable activity as labeled counts and links without hidden reasoning.
  - Add an optional save callback for completed run result previews.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx`
  - Add one page-level `openWorkspaceAgentTask(prefill)` callback.
  - Pass it to Chat and Studio render sites.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx`
  - Save ACP run result previews back into Studio as traceable `report` artifacts.
  - Preserve ACP run/session/task provenance and artifact version metadata.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/ChatPane/index.tsx`
  - Add optional `onStartWorkspaceTask` prop.
  - Add a compact toolbar button that builds a task prefill from selected sources and the latest chat context.
- Modify `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/index.tsx`
  - Add optional `onStartWorkspaceTask` prop.
  - Add a compact Studio header button that builds a task prefill from selected sources and generated artifacts.
- Modify tests:
  - `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ChatPane.stage1.test.tsx`
  - `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage1.test.tsx`
  - `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx`
  - `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage2.responsive.test.tsx`
- Update Backlog task `TASK-12170` with plan, touched files, and verification notes.

---

### Task 1: Preserve Handoff Context in Agent Task Metadata

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceAgentTaskHandoffModal.tsx`
- Test: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx`

- [ ] **Step 1: Write the failing metadata test**

Add a test near the existing agent-task modal tests:

```tsx
it("stores workspace task context metadata on created agent tasks", async () => {
  render(
    <WorkspaceHeader
      leftPaneOpen={true}
      rightPaneOpen={true}
      onToggleLeftPane={vi.fn()}
      onToggleRightPane={vi.fn()}
      agentTaskHandoffOpenSignal={1}
      agentTaskPrefill={{
        title: "Investigate chat thread",
        description: "Use selected sources.",
        metadata: {
          entrypoint: "chat",
          selectedSourceIds: ["source-1"]
        }
      }}
    />
  )

  // Fill the execution root, submit, and assert the project/task POST bodies
  // include metadata.research_workspace_task_context with entrypoint/source ids.
})
```

- [ ] **Step 2: Write the failing governance-copy test**

Add a small assertion near the same tests:

```tsx
it("explains agent tasks are governed by ACP sandbox and approvals", async () => {
  render(
    <WorkspaceHeader
      leftPaneOpen={true}
      rightPaneOpen={true}
      onToggleLeftPane={vi.fn()}
      onToggleRightPane={vi.fn()}
      agentTaskHandoffOpenSignal={1}
    />
  )

  expect(
    await screen.findByText(/ACP capabilities, sandbox checks, and approvals/i)
  ).toBeInTheDocument()
})
```

- [ ] **Step 3: Run the new tests and confirm they fail**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx -t "stores workspace task context metadata on created agent tasks"
bunx vitest run apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx -t "explains agent tasks are governed by ACP sandbox and approvals"
```

Expected: FAIL because `WorkspaceAgentTaskPrefill` has no `metadata`, created task metadata does not include the context, and the modal does not yet show explicit governance copy.

- [ ] **Step 4: Implement the minimal metadata extension**

Update `WorkspaceAgentTaskPrefill`:

```ts
export type WorkspaceAgentTaskPrefill = {
  title?: string | null
  description?: string | null
  metadata?: Record<string, unknown> | null
}
```

Build namespaced metadata only when it is a non-null object:

```ts
const prefillMetadata =
  prefill?.metadata && typeof prefill.metadata === "object"
    ? prefill.metadata
    : null

const metadata = {
  ...baseMetadata,
  acp_workspace_id: acpWorkspaceId,
  ...(prefillMetadata
    ? { research_workspace_task_context: prefillMetadata }
    : {})
}
```

Add a short modal notice after success/error alerts: task runs are governed by ACP capabilities, sandbox checks, and approvals; run history shows observable events, artifacts, diagnostics, and results.

- [ ] **Step 5: Re-run the focused tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx -t "stores workspace task context metadata on created agent tasks"
bunx vitest run apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx -t "explains agent tasks are governed by ACP sandbox and approvals"
```

Expected: PASS.

- [ ] **Step 6: Commit this slice**

```bash
git add apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceAgentTaskHandoffModal.tsx apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx
git commit -m "feat: preserve research workspace agent task context"
```

---

### Task 2: Add Chat Agent Task Entrypoint

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/ChatPane/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx`
- Test: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ChatPane.stage1.test.tsx`
- Test: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage2.responsive.test.tsx`

- [ ] **Step 1: Write the failing ChatPane test**

Add a test that renders `ChatPane` with `onStartWorkspaceTask={vi.fn()}`, selected ready source state, and at least one chat message. Click the new button and assert:

```ts
expect(onStartWorkspaceTask).toHaveBeenCalledWith(
  expect.objectContaining({
    title: expect.stringContaining("chat"),
    description: expect.stringContaining("DSPy Prompting Talk"),
    metadata: expect.objectContaining({
      entrypoint: "chat",
      workspaceId: "workspace-a",
      selectedSourceIds: ["source-1"]
    })
  })
)
```

- [ ] **Step 2: Run the ChatPane test and confirm it fails**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ChatPane.stage1.test.tsx -t "starts a workspace agent task from chat context"
```

Expected: FAIL because there is no ChatPane prop or button.

- [ ] **Step 3: Add the ChatPane prop and toolbar button**

Add `onStartWorkspaceTask?: (prefill: WorkspaceAgentTaskPrefill) => void`, import the type, and add a compact toolbar button next to clear chat using `Bot` or existing `Briefcase` icon with tooltip text `Start workspace task`.

Build bounded context:

- title: `Investigate chat thread`
- description sections:
  - `Workspace task from chat.`
  - selected source titles, limited to a small count
  - latest user message or draft, truncated
  - note that tool/code actions should use existing approvals
- metadata:
  - `entrypoint: "chat"`
  - `workspaceId`
  - `selectedSourceIds`
  - `selectedSourceTitles`
  - `messageCount`
  - `latestUserMessagePreview`

Disable the button only when `onStartWorkspaceTask` is missing. Do not invent an `agent_task` capability state in Chat; the button opens the governed handoff modal, and readiness/setup remains enforced by the modal creation flow plus Agent Tasks ACP readiness.

- [ ] **Step 4: Wire ResearchWorkspace to the ChatPane**

In `index.tsx`, add:

```ts
const openWorkspaceAgentTask = React.useCallback(
  (prefill: WorkspaceAgentTaskPrefill) => {
    setAgentTaskPrefill(prefill)
    setAgentTaskHandoffOpenSignal((current) => current + 1)
  },
  []
)
```

Pass `onStartWorkspaceTask={openWorkspaceAgentTask}` to both ChatPane render sites.

- [ ] **Step 5: Add the ResearchWorkspace wiring assertion**

Update the responsive/page test mock props to capture `onStartWorkspaceTask`; call it with a sample prefill and assert the latest `WorkspaceHeader` props contain incremented `agentTaskHandoffOpenSignal` and the same prefill.

- [ ] **Step 6: Re-run focused tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ChatPane.stage1.test.tsx -t "starts a workspace agent task from chat context"
bunx vitest run apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage2.responsive.test.tsx -t "opens the agent task handoff from pane entrypoints"
```

Expected: PASS.

- [ ] **Step 7: Commit this slice**

```bash
git add apps/packages/ui/src/components/Option/ResearchWorkspace/ChatPane/index.tsx apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ChatPane.stage1.test.tsx apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage2.responsive.test.tsx
git commit -m "feat: start research workspace tasks from chat"
```

---

### Task 3: Add Studio Agent Task Entrypoint

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/index.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx`
- Test: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage1.test.tsx`

- [ ] **Step 1: Write the failing StudioPane test**

Add a test that renders `StudioPane` with `onStartWorkspaceTask={vi.fn()}`, selected ready source state, and generated artifacts. Click `Start workspace task` and assert:

```ts
expect(onStartWorkspaceTask).toHaveBeenCalledWith(
  expect.objectContaining({
    title: expect.stringContaining("Studio"),
    description: expect.stringContaining("DSPy Prompting Talk"),
    metadata: expect.objectContaining({
      entrypoint: "studio",
      selectedSourceIds: ["source-1"]
    })
  })
)
```

- [ ] **Step 2: Run the StudioPane test and confirm it fails**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage1.test.tsx -t "starts a workspace agent task from studio context"
```

Expected: FAIL because there is no StudioPane prop or button.

- [ ] **Step 3: Add the StudioPane prop and header button**

Add `onStartWorkspaceTask?: (prefill: WorkspaceAgentTaskPrefill) => void`, import the type, and add a small icon button in the Studio header with tooltip/label `Start workspace task`.

Build bounded context:

- title: `Continue Studio work`
- description sections:
  - `Workspace task from Studio.`
  - selected source titles, limited
  - generated artifact titles/statuses, limited
  - note that outputs can be saved back through existing artifact or notes actions
- metadata:
  - `entrypoint: "studio"`
  - `workspaceId`
  - `workspaceTag`
  - `selectedSourceIds`
  - `selectedSourceTitles`
  - `artifactIds`
  - `artifactTitles`

- [ ] **Step 4: Pass the callback from ResearchWorkspace**

Pass `onStartWorkspaceTask={openWorkspaceAgentTask}` to `StudioPane` in `renderStudioPane`.

- [ ] **Step 5: Re-run the focused test**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage1.test.tsx -t "starts a workspace agent task from studio context"
```

Expected: PASS.

- [ ] **Step 6: Commit this slice**

```bash
git add apps/packages/ui/src/components/Option/ResearchWorkspace/StudioPane/index.tsx apps/packages/ui/src/components/Option/ResearchWorkspace/index.tsx apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage1.test.tsx
git commit -m "feat: start research workspace tasks from studio"
```

---

### Task 4: Save ACP Run Results Back to Studio Artifacts

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceACPHistoryModal.tsx`
- Modify: `apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx`
- Test: `apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx`

- [ ] **Step 1: Write the failing save-back test**

Add a test near the ACP run-history tests that stubs:

- canonical project lookup for the active workspace
- task detail with a completed run containing `result_summary`, `history.artifact_count`, `history.diagnostic_count`, `history.audit_event_count`, `history.event_count`, and `session.links`
- workspace store `addArtifact`

Open ACP run history, click `Save to Studio`, and assert:

```ts
expect(mockAddArtifact).toHaveBeenCalledWith(
  expect.objectContaining({
    type: "report",
    status: "completed",
    title: expect.stringContaining("Agent result"),
    content: expect.stringContaining("Completed synthesis"),
    version: 1,
    artifactVersionId: "acp-run-99-v1",
    rootArtifactId: "acp-run-99",
    producerMetadata: expect.objectContaining({
      producerType: "acp_agent_task",
      runId: "99",
      sessionId: "sess-99",
      taskId: "77"
    }),
    versionMetadata: expect.objectContaining({
      revisionReason: "Saved from ACP run history"
    })
  })
)
```

Also assert the modal renders an observable activity strip with labels for artifacts/files, diagnostics/warnings, audit/approvals, and events/tool activity. These labels must describe counts/links only; do not expose or imply hidden chain-of-thought.

- [ ] **Step 2: Run the save-back test and confirm it fails**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx -t "saves ACP run results as traceable Studio artifacts"
```

Expected: FAIL because the history modal has no save action and `WorkspaceHeader` does not convert runs into Studio artifacts.

- [ ] **Step 3: Add the history modal save contract**

In `WorkspaceACPHistoryModal.tsx`, export a small flattened input type:

```ts
export type WorkspaceACPRunArtifactSaveInput = {
  projectId: number
  projectName: string
  taskId: number
  taskTitle: string
  runId: number
  status: string
  resultPreview: string
  sessionId: string | null
  agentType?: string | null
  startedAt?: string | null
  completedAt?: string | null
  links?: Record<string, string>
  history?: RunItem["history"]
}
```

Add optional prop:

```ts
onSaveRunArtifact?: (input: WorkspaceACPRunArtifactSaveInput) => void
```

Render `Save to Studio` only when `onSaveRunArtifact` is provided, the run has a non-empty `resultPreview`, and the run status is completed/complete. Use existing run links for artifacts/diagnostics/audit buttons.

- [ ] **Step 4: Convert saved runs to traceable artifacts in WorkspaceHeader**

In `WorkspaceHeader.tsx`, read:

```ts
const addArtifact = useWorkspaceStore((s) => s.addArtifact)
const generatedArtifacts = useWorkspaceStore((s) => s.generatedArtifacts)
```

Add `handleSaveAcpRunArtifact`. It should:

- derive `rootArtifactId = acp-run-${runId}`
- find the latest existing artifact with that root and increment version, or use version 1
- set `previousVersionId` when a prior version exists
- call `addArtifact` with `type: "report"`, `status: "completed"`, `content`, `previewText`, `summary`, `version`, `artifactVersionId`, `rootArtifactId`, `projectId`, `taskId`, `ownerScope: "research_workspace"`, `ownerId: workspaceId || undefined`
- set `producerMetadata` with `producerType: "acp_agent_task"`, run/session/task/project ids, agent type, and ACP links
- set `versionMetadata.revisionReason = "Saved from ACP run history"`
- set `data.acpRun` with observable counts and timestamps
- show a success toast and save the current workspace

This exposes version history only where local artifact storage supports it. If the same run is saved again, the saved artifact should become the next version with `previousVersionId` pointing at the prior `artifactVersionId`.

- [ ] **Step 5: Re-run the focused save-back test**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx -t "saves ACP run results as traceable Studio artifacts"
```

Expected: PASS.

- [ ] **Step 6: Commit this slice**

```bash
git add apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceACPHistoryModal.tsx apps/packages/ui/src/components/Option/ResearchWorkspace/WorkspaceHeader.tsx apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx
git commit -m "feat: save ACP run results to workspace artifacts"
```

---

### Task 5: Regression Verification and Backlog Finalization

**Files:**
- Modify: `backlog/tasks/task-12170 - Implement-Research-Workspace-NotebookLM-Ultra-agent-task-WP4.md`

- [ ] **Step 1: Run focused UI tests**

Run:

```bash
bunx vitest run \
  apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/WorkspaceHeader.test.tsx \
  apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ChatPane.stage1.test.tsx \
  apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/StudioPane.stage1.test.tsx \
  apps/packages/ui/src/components/Option/ResearchWorkspace/__tests__/ResearchWorkspace.stage2.responsive.test.tsx
```

Expected: PASS.

- [ ] **Step 2: Run type/lint verification if available**

Check package scripts:

```bash
bun run --cwd apps/packages/ui
```

Run the smallest applicable existing check, such as:

```bash
bun run --cwd apps/packages/ui typecheck
```

Expected: PASS, or record the exact missing script.

- [ ] **Step 3: Run Bandit on touched backend scope**

This WP touches frontend only. Record that Bandit is not applicable because no Python/backend files changed.

- [ ] **Step 4: Update Backlog task**

Use MCP/CLI to add:

- plan path
- touched files
- test commands and results
- final summary

- [ ] **Step 5: Final self-review**

Check:

```bash
git status --short
git diff --stat
git diff
```

Confirm:

- no hidden chain-of-thought text is shown
- no sandbox or capability checks are bypassed
- metadata is namespaced
- buttons are accessible and do not crowd mobile layouts
- save-back remains via existing Studio artifact/note actions and ACP artifact links

- [ ] **Step 6: Commit final task update if needed**

```bash
git add "backlog/tasks/task-12170 - Implement-Research-Workspace-NotebookLM-Ultra-agent-task-WP4.md"
git commit -m "chore: record WP4 agent task implementation"
```
