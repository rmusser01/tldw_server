# Scheduled Tasks Phase 4A API-First Planned Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the existing `/scheduled-tasks` Recurring Question and Agent Task planned templates so they explain the API-first Phase 4 contract, capability fallback, result destinations, and safety model without enabling fake creation or drafts.

**Architecture:** Keep Slice 4A frontend-only and additive. Add a small pure helper that builds planned-family panel copy for `recurring_question` and `agent_task`, render that model from the existing `ScheduledTaskCreatePanel`, and update Results/Home empty copy only where it can stay API-honest. Do not alter backend schemas, create adapters, Watch/Ingest gates, or existing task list projections.

**Tech Stack:** React, TypeScript, Ant Design, React Router links, Vitest, Testing Library, existing ScheduledTasks and CompanionHome components.

---

## Source Inputs

- Approved spec: `Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase4-recurring-question-agent-task-api-contract-design.md`
- Backlog: `TASK-2343`
- Related design task: `TASK-2342`
- Existing Create panel: `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskCreatePanel.tsx`
- Existing template registry: `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-templates.ts`
- Existing Watch/Ingest capability helper: `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-template-capabilities.ts`
- Existing Results panel: `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskResultsPanel.tsx`
- Existing Home automation inbox: `apps/packages/ui/src/components/Option/CompanionHome/cards/AutomationInboxCard.tsx`

## Scope Check

The approved Phase 4 spec covers API contract, future backend foundations, Recurring Question execution, and Agent Task execution. This implementation plan is intentionally only **Slice 4A**:

- planned-template UX shell;
- API-first requirements and fallback copy;
- result destination copy;
- Agent Task safety copy;
- no executable backend changes.

Do not implement preview endpoints, draft persistence, server task creation, ACP schedule normalization, RAG schedule execution, approvals, run history, or normalized results in this slice.

## File Structure

| File | Responsibility |
| --- | --- |
| `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-planned-template-copy.ts` | New pure helper for planned Recurring Question and Agent Task panel models. Keeps planned-family copy separate from Watch/Ingest capability gates. |
| `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-planned-template-copy.test.ts` | Unit tests for planned-family helper, fallback order, links, requirements, and no-create contract. |
| `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskCreatePanel.tsx` | Render the richer planned panel for Recurring Question and Agent Task. Keep Reminder and Watch/Ingest behavior unchanged. |
| `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx` | Component coverage for Recurring Question and Agent Task planned panels, extension-width copy, deep links, and no create controls. |
| `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskResultsPanel.tsx` | Optional copy-only update explaining future Recurring Question and Agent Task results through visibility policy. No fake items or filters. |
| `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultsPanel.test.tsx` | Copy regression if Results panel text changes. |
| `apps/packages/ui/src/components/Option/CompanionHome/cards/AutomationInboxCard.tsx` | Optional empty-state copy update for future routed outputs. No fake items. |
| `apps/packages/ui/src/components/Option/CompanionHome/__tests__/AutomationInboxCard.test.tsx` | Copy regression if Home empty copy changes. |
| `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx` | Route integration for `?tab=create&template=recurring_question` and `?tab=create&template=agent_task`. |
| `backlog/tasks/task-2343 - Plan-Scheduled-Tasks-Phase-4A-API-first-planned-shell-implementation.md` | Planning task notes and verification record. |

## Guardrails

- Keep Recurring Question and Agent Task in `planned` state unless a real future API advertises support.
- Do not extend `resolveTemplateCapabilityState()` to mark these templates `available`.
- Do not reuse `REQUIRED_WATCH_AVAILABILITY_GATES` or `REQUIRED_INGEST_AVAILABILITY_GATES` for these templates.
- Do not add local storage, server drafts, fake task rows, fake run rows, fake results, or sample Home inbox items.
- Do not remove or simplify Watchlists copy.
- Do not use GitHub or YouTube as primary IA labels.
- Use text labels for status and requirements; do not rely on color.

## Task 1: Planned Family Helper

**Files:**
- Create: `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-planned-template-copy.ts`
- Create: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-planned-template-copy.test.ts`

- [ ] **Step 1: Write failing helper tests**

Add tests for the pure helper:

```ts
import { describe, expect, it } from "vitest"

import {
  buildPlannedScheduledTaskPanelModel,
  isPlannedAutomationTemplate
} from "../scheduled-task-planned-template-copy"

describe("scheduled task planned template copy", () => {
  it("builds API-first Recurring Question copy without executable support", () => {
    const model = buildPlannedScheduledTaskPanelModel("recurring_question")

    expect(model.statusLabel).toBe("Planned automation type")
    expect(model.jobStatement).toBe(
      "Run this question on a schedule across selected searchable content."
    )
    expect(model.requirements).toContainEqual(
      expect.objectContaining({ label: "Scheduled RAG query support" })
    )
    expect(model.resultDestinations).toContain(
      "Every run is recorded in task history."
    )
    expect(model.resultDestinations).toContain(
      "Home and Results receive summaries only when selected by the task visibility policy."
    )
    expect(model.links).toContainEqual({ label: "Open Research", href: "/research" })
    expect(model.createEnabled).toBe(false)
  })

  it("builds API-first Agent Task copy with preview and approval expectations", () => {
    const model = buildPlannedScheduledTaskPanelModel("agent_task")

    expect(model.statusLabel).toBe("Planned automation type")
    expect(model.jobStatement).toBe(
      "Send this message to the selected agent at the scheduled time."
    )
    expect(model.requirements).toContainEqual(
      expect.objectContaining({ label: "Schedulable ACP/API agents" })
    )
    expect(model.safetyLines).toContain(
      "Preview is required before scheduling an agent task."
    )
    expect(model.safetyLines).toContain(
      "Some permission classes may require approval before each run."
    )
    expect(model.links).toContainEqual({ label: "Open Agent Tasks", href: "/agent-tasks" })
    expect(model.links).toContainEqual({ label: "Open ACP Playground", href: "/acp-playground" })
    expect(model.createEnabled).toBe(false)
  })

  it("treats non-planned families as unsupported by this helper", () => {
    expect(isPlannedAutomationTemplate("watch")).toBe(false)
    expect(buildPlannedScheduledTaskPanelModel("watch")).toBeNull()
  })
})
```

- [ ] **Step 2: Run helper tests to verify they fail**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-planned-template-copy.test.ts
```

Expected: FAIL because `scheduled-task-planned-template-copy.ts` does not exist.

- [ ] **Step 3: Implement the pure helper**

Create `scheduled-task-planned-template-copy.ts` with a small model:

```ts
import type { ScheduledTaskTemplateId } from "./scheduled-task-templates"

export type PlannedScheduledTaskTemplateId = Extract<
  ScheduledTaskTemplateId,
  "recurring_question" | "agent_task"
>

export type PlannedRequirementStatus = "planned" | "related_available" | "missing"

export interface PlannedScheduledTaskRequirement {
  label: string
  detail: string
  status: PlannedRequirementStatus
}

export interface PlannedScheduledTaskLink {
  label: string
  href: string
}

export interface PlannedScheduledTaskPanelModel {
  templateId: PlannedScheduledTaskTemplateId
  statusLabel: string
  jobStatement: string
  availabilityReason: string
  requirements: PlannedScheduledTaskRequirement[]
  resultDestinations: string[]
  safetyLines: string[]
  links: PlannedScheduledTaskLink[]
  createEnabled: false
}
```

Implementation rules:

- `isPlannedAutomationTemplate()` returns true only for `recurring_question` and `agent_task`.
- `buildPlannedScheduledTaskPanelModel()` returns `null` for all other IDs.
- Both models use `statusLabel: "Planned automation type"` and `createEnabled: false`.
- Recurring Question requirements:
  - `Scheduled RAG query support`
  - `Searchable scope selection`
  - `Normalized run history`
  - `Task visibility policy`
- Agent Task requirements:
  - `Schedulable ACP/API agents`
  - `Preview and risk classification`
  - `Approval policy`
  - `Normalized agent run outputs`
- Links:
  - Recurring Question: `/research`, `/scheduled-tasks/results`
  - Agent Task: `/agent-tasks`, `/acp-playground`, `/scheduled-tasks/results`

- [ ] **Step 4: Run helper tests to verify they pass**

Run:

```bash
bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-planned-template-copy.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit helper**

```bash
git add apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-planned-template-copy.ts apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-planned-template-copy.test.ts
git commit -m "test: add scheduled task planned template contract"
```

## Task 2: Render API-First Planned Panels

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskCreatePanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx`

- [ ] **Step 1: Write failing component tests for Recurring Question**

Extend the existing planned Recurring Question test:

```ts
it("renders API-first Recurring Question planned requirements without create controls", () => {
  render(
    <ScheduledTaskCreatePanel
      selectedTemplateId="recurring_question"
      onSelectTemplate={vi.fn()}
      onCreateReminder={vi.fn()}
    />
  )

  expect(screen.getByText("Planned automation type")).toBeInTheDocument()
  expect(
    screen.getByText("Run this question on a schedule across selected searchable content.")
  ).toBeInTheDocument()
  expect(screen.getByText("Scheduled RAG query support")).toBeInTheDocument()
  expect(screen.getByText("Task visibility policy")).toBeInTheDocument()
  expect(screen.getByText("Every run is recorded in task history.")).toBeInTheDocument()
  expect(screen.getByRole("link", { name: "Open Research" })).toHaveAttribute(
    "href",
    "/research"
  )
  expect(screen.queryByRole("button", { name: /Create/i })).not.toBeInTheDocument()
})
```

- [ ] **Step 2: Write failing component tests for Agent Task**

Add:

```ts
it("renders API-first Agent Task planned safety and approval copy without create controls", () => {
  render(
    <ScheduledTaskCreatePanel
      selectedTemplateId="agent_task"
      onSelectTemplate={vi.fn()}
      onCreateReminder={vi.fn()}
    />
  )

  expect(screen.getByText("Planned automation type")).toBeInTheDocument()
  expect(
    screen.getByText("Send this message to the selected agent at the scheduled time.")
  ).toBeInTheDocument()
  expect(screen.getByText("Schedulable ACP/API agents")).toBeInTheDocument()
  expect(screen.getByText("Preview and risk classification")).toBeInTheDocument()
  expect(screen.getByText("Preview is required before scheduling an agent task.")).toBeInTheDocument()
  expect(
    screen.getByText("Some permission classes may require approval before each run.")
  ).toBeInTheDocument()
  expect(screen.getByRole("link", { name: "Open Agent Tasks" })).toHaveAttribute(
    "href",
    "/agent-tasks"
  )
  expect(screen.getByRole("link", { name: "Open ACP Playground" })).toHaveAttribute(
    "href",
    "/acp-playground"
  )
  expect(screen.queryByRole("button", { name: /Create/i })).not.toBeInTheDocument()
})
```

- [ ] **Step 3: Run create-panel tests to verify they fail**

Run from `apps/packages/ui`:

```bash
bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx
```

Expected: FAIL because the current `PlannedPanel` is too thin.

- [ ] **Step 4: Implement `PlannedPanel` rendering**

Update `ScheduledTaskCreatePanel.tsx`:

- Import `buildPlannedScheduledTaskPanelModel`.
- Change `PlannedPanel` to build a model from `template.id`.
- If the model exists, render:
  - status tag;
  - template intent;
  - `model.jobStatement`;
  - `Availability` copy;
  - `Requires` list;
  - `Results appear in` list;
  - `Safety` group only when `model.safetyLines.length > 0`;
  - `Related workspaces` links;
  - `No scheduled task has been created yet.`
- If `buildPlannedScheduledTaskPanelModel()` returns `null`, keep the old minimal planned panel fallback.

Use existing `CapabilityCopyGroup` for text groups where practical. Do not add create/preview buttons.

- [ ] **Step 5: Run create-panel tests to verify they pass**

Run:

```bash
bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Commit create panel**

```bash
git add apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskCreatePanel.tsx apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx
git commit -m "feat: explain planned scheduled task families"
```

## Task 3: Preserve Capability Fallback And Template State

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-template-capabilities.test.ts`
- Modify only if needed: `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-template-capabilities.ts`
- Modify only if needed: `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-templates.ts`

- [ ] **Step 1: Write fallback regression tests**

Add tests proving Recurring Question and Agent Task do not become available from the current Watch/Ingest capability helper:

```ts
it("does not resolve Recurring Question availability from Watch/Ingest capability gates", () => {
  const capability = buildScheduledTaskTemplateCapability("recurring_question", {
    creationAdapterSupported: true,
    passedGates: REQUIRED_WATCH_AVAILABILITY_GATES
  })

  expect(resolveTemplateCapabilityState("recurring_question", capability)).toBeNull()
})

it("does not resolve Agent Task availability from Watch/Ingest capability gates", () => {
  const capability = buildScheduledTaskTemplateCapability("agent_task", {
    creationAdapterSupported: true,
    passedGates: REQUIRED_INGEST_AVAILABILITY_GATES
  })

  expect(resolveTemplateCapabilityState("agent_task", capability)).toBeNull()
})
```

- [ ] **Step 2: Run capability tests**

Run:

```bash
bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-template-capabilities.test.ts
```

Expected: PASS if current helper already follows the spec. If it fails, restore behavior so only Watch/Ingest use those gates.

- [ ] **Step 3: Optionally tune static template copy**

If implementation review finds the cards too vague before opening a planned panel, update only static descriptions in `scheduled-task-templates.ts`:

- Recurring Question description: `Run a question repeatedly across selected searchable content when scheduled RAG support is available.`
- Agent Task description: `Send a message to a selected agent later when schedulable agent support is available.`

Do not change state from `planned`.

- [ ] **Step 4: Run template tests**

Run:

```bash
bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts src/components/Option/ScheduledTasks/__tests__/scheduled-task-template-capabilities.test.ts
```

Expected: PASS.

- [ ] **Step 5: Commit fallback guards**

```bash
git add apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-template-capabilities.test.ts apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-template-capabilities.ts apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-templates.ts
git commit -m "test: guard planned automation availability"
```

If only tests changed, stage only the changed test file.

## Task 4: Results And Home Copy For Future Destinations

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskResultsPanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultsPanel.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/CompanionHome/cards/AutomationInboxCard.tsx`
- Modify: `apps/packages/ui/src/components/Option/CompanionHome/__tests__/AutomationInboxCard.test.tsx`

- [ ] **Step 1: Write failing Results copy test**

In `ScheduledTaskResultsPanel.test.tsx`, update or add a projected-mode assertion:

```ts
expect(
  screen.getByText(
    "Future scheduled questions and agent outputs appear here only when the results API and each task visibility policy route them here."
  )
).toBeInTheDocument()
```

- [ ] **Step 2: Write failing Home empty-state copy test**

In `AutomationInboxCard.test.tsx`, update the empty-state assertion:

```ts
expect(
  screen.getByText(
    "Results and failures from scheduled tasks appear here after a run. Future scheduled questions and agent outputs appear here only when routed by task visibility policy."
  )
).toBeInTheDocument()
```

- [ ] **Step 3: Run copy tests to verify they fail**

Run:

```bash
bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultsPanel.test.tsx src/components/Option/CompanionHome/__tests__/AutomationInboxCard.test.tsx
```

Expected: FAIL because copy has not been updated.

- [ ] **Step 4: Update copy only**

In `ScheduledTaskResultsPanel.tsx`, keep the existing projected-mode warning and add one concise second sentence to the projected-mode `description`. Do not add result items, filters, actions, or fake counts.

In `AutomationInboxCard.tsx`, update only the empty description default. Do not change item rendering or data loading.

- [ ] **Step 5: Run copy tests to verify they pass**

Run:

```bash
bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultsPanel.test.tsx src/components/Option/CompanionHome/__tests__/AutomationInboxCard.test.tsx
```

Expected: PASS.

- [ ] **Step 6: Commit copy changes**

```bash
git add apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskResultsPanel.tsx apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultsPanel.test.tsx apps/packages/ui/src/components/Option/CompanionHome/cards/AutomationInboxCard.tsx apps/packages/ui/src/components/Option/CompanionHome/__tests__/AutomationInboxCard.test.tsx
git commit -m "copy: clarify future automation result routing"
```

## Task 5: Route-Level Create Shell Coverage

**Files:**
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx`

- [ ] **Step 1: Add URL route tests for both planned templates**

Add tests next to the existing "opens the Create tab from the URL" test:

```ts
it("opens the Recurring Question planned shell from the URL", async () => {
  mocks.listScheduledTasks.mockResolvedValue({
    items: [],
    total: 0,
    partial: false,
    errors: []
  })

  renderWithQueryClient(
    <ScheduledTasksPage />,
    "/scheduled-tasks?tab=create&template=recurring_question"
  )

  expect(await screen.findByText("Planned automation type")).toBeInTheDocument()
  expect(screen.getByText("Scheduled RAG query support")).toBeInTheDocument()
  expect(screen.queryByRole("button", { name: /Create recurring/i })).not.toBeInTheDocument()
})

it("opens the Agent Task planned shell from the URL", async () => {
  mocks.listScheduledTasks.mockResolvedValue({
    items: [],
    total: 0,
    partial: false,
    errors: []
  })

  renderWithQueryClient(
    <ScheduledTasksPage />,
    "/scheduled-tasks?tab=create&template=agent_task"
  )

  expect(await screen.findByText("Planned automation type")).toBeInTheDocument()
  expect(screen.getByText("Schedulable ACP/API agents")).toBeInTheDocument()
  expect(screen.getByText("Preview is required before scheduling an agent task.")).toBeInTheDocument()
  expect(screen.queryByRole("button", { name: /Create agent/i })).not.toBeInTheDocument()
})
```

- [ ] **Step 2: Run page tests**

Run:

```bash
bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
```

Expected: PASS after Task 2 is implemented.

- [ ] **Step 3: Commit route coverage**

```bash
git add apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx
git commit -m "test: cover planned automation deep links"
```

## Task 6: Final Verification, Backlog, And Branch Hygiene

**Files:**
- Modify: `backlog/tasks/task-2343 - Plan-Scheduled-Tasks-Phase-4A-API-first-planned-shell-implementation.md`
- No backend/Python files expected.

- [ ] **Step 1: Run focused Scheduled Tasks tests**

From `apps/packages/ui`:

```bash
bunx vitest run \
  src/components/Option/ScheduledTasks/__tests__/scheduled-task-planned-template-copy.test.ts \
  src/components/Option/ScheduledTasks/__tests__/scheduled-task-template-capabilities.test.ts \
  src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx \
  src/components/Option/ScheduledTasks/__tests__/ScheduledTaskResultsPanel.test.tsx \
  src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx \
  src/components/Option/CompanionHome/__tests__/AutomationInboxCard.test.tsx
```

Expected: PASS.

- [ ] **Step 2: Run diff checks**

From repo root:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 3: Browser verification if dev server is practical**

If dependencies are installed and a dev server is practical, start the frontend and use Browser or Playwright to inspect:

- `/scheduled-tasks?tab=create&template=recurring_question`
- `/scheduled-tasks?tab=create&template=agent_task`
- `/scheduled-tasks/results`
- Home route that renders `Automation Inbox`

Verify:

- planned panels show requirements and links;
- no create/preview/draft controls appear;
- no fake task/result rows appear;
- 390px width has no obvious text overlap or horizontal overflow outside expected tables.

If a browser check is not practical, document why and rely on component tests.

- [ ] **Step 4: Bandit**

No Bandit run is required if only frontend, docs, and Backlog files changed. Record:

```text
Bandit skipped because no Python/backend files were changed.
```

If backend files are unexpectedly changed, run:

```bash
source .venv/bin/activate
python -m bandit -r <backend_touched_paths> -f json -o /tmp/bandit_scheduled_tasks_phase4a.json
```

- [ ] **Step 5: Update Backlog task**

Update `TASK-2343` with:

- changed files;
- test commands and results;
- browser verification or skip reason;
- Bandit skip reason;
- final summary.

- [ ] **Step 6: Final commit**

```bash
git status --short
git add backlog/tasks/task-2343\ -\ Plan-Scheduled-Tasks-Phase-4A-API-first-planned-shell-implementation.md
git commit -m "chore: close scheduled tasks phase 4a shell plan"
```

If there are no Backlog-only changes after previous commits, skip this commit and record the task update in the PR summary.

## Plan Review

Reviewed locally against the `plan-document-reviewer` rubric on 2026-06-09.

Status: Approved.

Blocking issues: None.

Subagent note:

- The writing-plans skill normally dispatches a plan-document-reviewer subagent. In this environment, the available subagent tool is restricted to cases where the user explicitly requests delegation, so the same review rubric was applied locally instead.
