# Scheduled Tasks Phase 2B Capability-Aware Frontend Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the frontend-only Phase 2B.2 capability-aware shell for Watch and Ingest templates so `/scheduled-tasks` can display runtime capability, Limited availability, generated source/destination/notification copy, and safety messaging without promoting Watch/Ingest to Available before all gates and an explicit creation-adapter guard pass.

**Architecture:** Keep the implementation inside the existing ScheduledTasks feature folder. Add a pure capability-contract helper module that overlays capability metadata onto the existing template registry, then update the Create panel to render effective capability states and generated copy. Do not add backend calls, Watchlists creation adapters, source preview APIs, duplicate APIs, notification APIs, or Home result surfacing in this slice.

**Tech Stack:** React, TypeScript, Ant Design, React Router search params, TanStack Query, Vitest, Testing Library, existing ScheduledTasks components, existing scheduled-tasks control-plane service.

---

## Source Spec

- `Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase2b-watch-ingest-product-contract-design.md`
- Backlog design tasks: `TASK-2324`, `TASK-2325`
- Planning task: `TASK-2326`
- Current branch: `codex/scheduled-tasks-phase2b-contract`

## Scope

In scope:

- Add a frontend capability model for Watch/Ingest template gates.
- Add the `limited_availability` template state.
- Keep Reminder as the only default Available template.
- Keep Watch/Ingest default behavior non-creating. Tests may prove the pure resolver can model future all-gates availability, but only when an explicit creation adapter flag is true. The page default must not pass that adapter guard or create Watch/Ingest tasks in this slice.
- Generate source-intent, destination, and notification copy from capability metadata.
- Render Limited availability, missing gate reasons, source support, and result-only notification copy.
- Add redaction-safe copy helpers for capability and preview-adjacent text.
- Update tests so Watch/Ingest cannot appear in Available now unless all gates and the explicit creation-adapter guard pass.
- Preserve current Watchlists handoff behavior and exact copy that says no scheduled task was created.
- Preserve extension route parity because the extension reuses the same component.

Out of scope:

- Backend endpoints for capability health, preview, duplicate detection, creation, notification policy, or result destinations.
- Watchlists source setup or prefilled deep-link adapter.
- Creating Watch/Ingest tasks from `/scheduled-tasks`.
- Home automation cards.
- Source preview tables.
- Duplicate warning UI beyond copy/model placeholders.
- Bulk actions, run now, dry run, saved views, export.
- Recurring RAG or Agent Task scheduling.

## Pre-Implementation Requirement

Before product-code edits, create a new implementation Backlog.md task such as `Implement Scheduled Tasks Phase 2B capability-aware frontend shell`.

Link that task to:

- `TASK-2326`
- `TASK-2325`
- `TASK-2324`
- `Docs/superpowers/plans/2026-06-09-scheduled-tasks-phase2b-capability-aware-frontend-shell-implementation-plan.md`
- `Docs/superpowers/specs/2026-06-09-scheduled-tasks-phase2b-watch-ingest-product-contract-design.md`

Record modified files, verification commands, known skips, and final summary on the implementation task. `TASK-2326` tracks this implementation plan only.

## Existing Code Context

Current Phase 2A files already exist:

- `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-templates.ts`
  - Static template registry.
  - Current states: `available`, `handoff_only`, `needs_setup`, `managed_in_watchlists`, `planned`, `unavailable`.
  - Watch/Ingest/Advanced are currently `handoff_only`.
  - Reminder is the only `available` template.
- `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskCreatePanel.tsx`
  - Renders finder, filters, template cards, Reminder editor, planned panels, handoff panels.
  - Handoff panels already avoid task-created language.
  - Source/setup notes already reject private-looking URL params and prose secrets.
- `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx`
  - Owns Overview/Tasks/Create tabs and passes selected template into Create panel.
  - Does not currently fetch or pass template capability data.
- Tests already cover template registry, matcher, handoff copy, source-note sanitization, route state, and page behavior.

## File Structure

### New Files

- `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-template-capabilities.ts`
  - Pure Phase 2B frontend capability model.
  - Availability gate definitions.
  - Source-intent, result, and notification copy generation.
  - Capability overlay for existing templates.
  - Redaction helpers for capability-sourced text.
- `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-template-capabilities.test.ts`
  - Unit tests for gate enforcement, Limited availability, generated copy, source-intent metadata, and redaction.

### Modified Files

- `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-templates.ts`
  - Add `limited_availability` to `ScheduledTaskTemplateState`.
  - Update Watch and Ingest base descriptions to avoid overpromising notifications/search/RAG.
  - Export helpers that allow filtering an effective template list, not only the static registry.
- `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskCreatePanel.tsx`
  - Accept optional `templateCapabilities` prop.
  - Resolve effective template states through the new capability helper.
  - Render Limited availability and missing gate copy.
  - Render generated source/destination/notification copy in selected Watch/Ingest panels.
  - Keep non-Available states out of creation paths.
- `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx`
  - Pass the default no-contract capability map into `ScheduledTaskCreatePanel`.
  - Update empty-state copy to avoid source-vendor examples and overpromising.
- `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts`
  - Add label/filter coverage for `limited_availability`.
  - Add regression coverage that Available now excludes Limited availability.
- `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx`
  - Add coverage for Limited availability UI, generated source/destination/notification copy, no create CTA, and extension-width essentials.
- `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx`
  - Update source-vendor empty-state copy assertions if needed.
  - Add route-level smoke that Watch/Ingest still do not create tasks.

### Files To Avoid Changing

- `apps/packages/ui/src/components/Option/Watchlists/**`
- `apps/packages/ui/src/services/watchlists*.ts`
- `apps/packages/ui/src/services/scheduled-tasks-control-plane.ts`
- `tldw_Server_API/**`
- Home, RAG, ACP, Jobs, Scheduler, Notifications.

If implementation discovers that one of these files must change, stop and update the plan or create a smaller follow-up task. Do not silently expand Phase 2B.2 into a backend or Watchlists adapter.

## Capability Model Rules

The shell must enforce these rules:

1. `reminder` remains `available`.
2. `watch` and `ingest` default to `handoff_only` when no capability contract exists.
3. `watch` and `ingest` become `limited_availability` when capability metadata exists but at least one required gate is missing.
4. `watch` and `ingest` remain `limited_availability` when every required gate passes but `creationAdapterSupported !== true`.
5. `watch` and `ingest` become `available` only when every required gate passes and `creationAdapterSupported === true`.
6. `available_now` includes only effective state `available`.
7. Limited availability must never render a create CTA or "scheduled" success copy.
8. Generated source/destination/notification copy must be based on capability metadata, not hardcoded promises.
9. Phase 2B.2 product defaults must not provide `creationAdapterSupported: true` for Watch/Ingest because no Watchlists creation adapter exists in this slice.
10. `sourceIntent.can_create` describes source-level capability only. It must not be treated as a substitute for `creationAdapterSupported`.

Required gates for Watch:

- `capability_health`
- `source_preview`
- `duplicate_detection`
- `created_entity_response`
- `task_visibility`
- `run_result_links`
- `failure_contract`
- `result_destination`
- `notification_contract`
- `safe_source_handling`
- `watchlists_preservation`

Required gates for Ingest:

- `capability_health`
- `source_preview`
- `duplicate_detection`
- `created_entity_response`
- `task_visibility`
- `run_result_links`
- `failure_contract`
- `result_destination`
- `safe_source_handling`
- `watchlists_preservation`

Do not require `notification_contract` for Ingest unless a future task explicitly adds alerting as part of the first-time Ingest path.

## Task 1: Add Capability Helper Module

**Files:**

- Create: `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-template-capabilities.ts`
- Test: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-template-capabilities.test.ts`

- [ ] **Step 1: Write failing tests for gate enforcement**

Add tests:

```ts
import { describe, expect, it } from "vitest"
import {
  REQUIRED_INGEST_AVAILABILITY_GATES,
  REQUIRED_WATCH_AVAILABILITY_GATES,
  buildScheduledTaskTemplateCapability,
  getMissingAvailabilityGates,
  resolveTemplateCapabilityState
} from "../scheduled-task-template-capabilities"

describe("scheduled task template capabilities", () => {
  it("requires preview before Watch can be available", () => {
    const capability = buildScheduledTaskTemplateCapability("watch", {
      passedGates: REQUIRED_WATCH_AVAILABILITY_GATES.filter(
        (gate) => gate !== "source_preview"
      )
    })

    expect(resolveTemplateCapabilityState("watch", capability)).toBe("limited_availability")
    expect(getMissingAvailabilityGates("watch", capability)).toContain("source_preview")
  })

  it("keeps Watch limited when gates pass but no creation adapter is supported", () => {
    const capability = buildScheduledTaskTemplateCapability("watch", {
      passedGates: REQUIRED_WATCH_AVAILABILITY_GATES
    })

    expect(resolveTemplateCapabilityState("watch", capability)).toBe("limited_availability")
  })

  it("allows Watch only when every Watch gate and the creation adapter guard pass", () => {
    const capability = buildScheduledTaskTemplateCapability("watch", {
      creationAdapterSupported: true,
      passedGates: REQUIRED_WATCH_AVAILABILITY_GATES
    })

    expect(resolveTemplateCapabilityState("watch", capability)).toBe("available")
  })

  it("does not require notification gate for Ingest availability once creation is supported", () => {
    const capability = buildScheduledTaskTemplateCapability("ingest", {
      creationAdapterSupported: true,
      passedGates: REQUIRED_INGEST_AVAILABILITY_GATES
    })

    expect(resolveTemplateCapabilityState("ingest", capability)).toBe("available")
  })
})
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-template-capabilities.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: fail because the module does not exist.

- [ ] **Step 3: Implement the capability model**

Create a small pure module. Use explicit string unions so future API data must map through a typed boundary.

```ts
import type {
  ScheduledTaskTemplate,
  ScheduledTaskTemplateId,
  ScheduledTaskTemplateState
} from "./scheduled-task-templates"

export type ScheduledTaskAvailabilityGate =
  | "capability_health"
  | "source_preview"
  | "duplicate_detection"
  | "created_entity_response"
  | "task_visibility"
  | "run_result_links"
  | "failure_contract"
  | "result_destination"
  | "notification_contract"
  | "safe_source_handling"
  | "watchlists_preservation"

export type ScheduledTaskSourceFamily =
  | "unknown"
  | "feed"
  | "website"
  | "repository_issues"
  | "video_channel"
  | "publication"
  | "advisory"

export interface ScheduledTaskSourceIntentCapability {
  sourceFamily: ScheduledTaskSourceFamily
  can_watch: boolean
  can_ingest: boolean
  can_preview: boolean
  can_notify: boolean
  can_index_search: boolean
  can_index_rag: boolean
  can_create: boolean
  reason?: string | null
}

export interface ScheduledTaskResultDestinationMetadata {
  home_supported: boolean
  notifications_supported: boolean
  search_indexed: boolean
  rag_scope_included: boolean
}

export interface ScheduledTaskTemplateCapability {
  templateId: ScheduledTaskTemplateId
  passedGates: readonly ScheduledTaskAvailabilityGate[]
  creationAdapterSupported?: boolean
  sourceIntent?: ScheduledTaskSourceIntentCapability | null
  resultDestinations?: ScheduledTaskResultDestinationMetadata | null
  reason?: string | null
}

export type ScheduledTaskTemplateCapabilityMap = Partial<
  Record<ScheduledTaskTemplateId, ScheduledTaskTemplateCapability>
>

export const REQUIRED_WATCH_AVAILABILITY_GATES = [
  "capability_health",
  "source_preview",
  "duplicate_detection",
  "created_entity_response",
  "task_visibility",
  "run_result_links",
  "failure_contract",
  "result_destination",
  "notification_contract",
  "safe_source_handling",
  "watchlists_preservation"
] as const satisfies readonly ScheduledTaskAvailabilityGate[]

export const REQUIRED_INGEST_AVAILABILITY_GATES = [
  "capability_health",
  "source_preview",
  "duplicate_detection",
  "created_entity_response",
  "task_visibility",
  "run_result_links",
  "failure_contract",
  "result_destination",
  "safe_source_handling",
  "watchlists_preservation"
] as const satisfies readonly ScheduledTaskAvailabilityGate[]

export const getRequiredAvailabilityGates = (
  templateId: ScheduledTaskTemplateId
): readonly ScheduledTaskAvailabilityGate[] =>
  templateId === "watch"
    ? REQUIRED_WATCH_AVAILABILITY_GATES
    : templateId === "ingest"
      ? REQUIRED_INGEST_AVAILABILITY_GATES
      : []

export const getMissingAvailabilityGates = (
  templateId: ScheduledTaskTemplateId,
  capability: ScheduledTaskTemplateCapability | null | undefined
): ScheduledTaskAvailabilityGate[] => {
  const required = getRequiredAvailabilityGates(templateId)
  const passed = new Set(capability?.passedGates ?? [])
  return required.filter((gate) => !passed.has(gate))
}

export const resolveTemplateCapabilityState = (
  templateId: ScheduledTaskTemplateId,
  capability: ScheduledTaskTemplateCapability | null | undefined
): ScheduledTaskTemplateState | null => {
  if (templateId !== "watch" && templateId !== "ingest") {
    return null
  }

  if (!capability) {
    return null
  }

  if (getMissingAvailabilityGates(templateId, capability).length > 0) {
    return "limited_availability"
  }

  // Keep this separate from sourceIntent.can_create. That flag describes source-level
  // support, not whether this /scheduled-tasks shell has a real creation adapter.
  return capability.creationAdapterSupported === true
    ? "available"
    : "limited_availability"
}

export const applyScheduledTaskTemplateCapabilities = (
  templates: readonly ScheduledTaskTemplate[],
  capabilities: ScheduledTaskTemplateCapabilityMap | null | undefined
): ScheduledTaskTemplate[] =>
  templates.map((template) => {
    const resolvedState = resolveTemplateCapabilityState(
      template.id,
      capabilities?.[template.id]
    )
    return resolvedState ? { ...template, state: resolvedState } : template
  })

export const buildScheduledTaskTemplateCapability = (
  templateId: ScheduledTaskTemplateId,
  overrides: Partial<ScheduledTaskTemplateCapability> = {}
): ScheduledTaskTemplateCapability => ({
  templateId,
  passedGates: [],
  creationAdapterSupported: false,
  sourceIntent: null,
  resultDestinations: null,
  reason: null,
  ...overrides
})
```

- [ ] **Step 4: Run tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-template-capabilities.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-template-capabilities.ts \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-template-capabilities.test.ts
git commit -m "feat: add scheduled task capability model"
```

## Task 2: Add Limited Availability To Template Registry

**Files:**

- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-templates.ts`
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts`

- [ ] **Step 1: Write failing tests for Limited availability**

Add assertions:

```ts
it("labels Limited availability", () => {
  expect(getScheduledTaskTemplateStateLabel("limited_availability")).toBe(
    "Limited availability"
  )
})

it("does not include Limited availability in Available now", () => {
  const templates = [
    { ...getScheduledTaskTemplate("watch")!, state: "limited_availability" as const }
  ]
  expect(filterScheduledTaskTemplates("available_now", templates)).toEqual([])
})
```

Expected TypeScript failure before implementation because `limited_availability` is not in the union and `filterScheduledTaskTemplates` does not accept a custom list.

- [ ] **Step 2: Run the failing tests**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: fail.

- [ ] **Step 3: Update template state type and labels**

Modify `ScheduledTaskTemplateState`:

```ts
export type ScheduledTaskTemplateState =
  | "available"
  | "limited_availability"
  | "handoff_only"
  | "needs_setup"
  | "managed_in_watchlists"
  | "planned"
  | "unavailable"
```

Modify descriptions:

```ts
description: "Surface new matching items and notify when supported."
description: "Add new source content to supported library, search, or knowledge destinations."
```

Modify `filterScheduledTaskTemplates` to accept an optional template list:

```ts
export const filterScheduledTaskTemplates = (
  filterId: ScheduledTaskTemplateFilterId,
  templates: readonly ScheduledTaskTemplate[] = SCHEDULED_TASK_TEMPLATES
): readonly ScheduledTaskTemplate[] => {
  if (filterId === "all") return templates
  if (filterId === "available_now") {
    return templates.filter((template) => template.state === "available")
  }
  return templates.filter((template) => template.category === filterId)
}
```

Modify `getScheduledTaskTemplate` to accept an optional template list:

```ts
export const getScheduledTaskTemplate = (
  id: ScheduledTaskTemplateId | string | null | undefined,
  templates: readonly ScheduledTaskTemplate[] = SCHEDULED_TASK_TEMPLATES
): ScheduledTaskTemplate | null =>
  templates.find((template) => template.id === id) ?? null
```

Add state label:

```ts
case "limited_availability":
  return "Limited availability"
```

- [ ] **Step 4: Run tests**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-templates.ts \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts
git commit -m "feat: add limited scheduled task template state"
```

## Task 3: Generate Source, Result, And Notification Copy From Metadata

**Files:**

- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-template-capabilities.ts`
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-template-capabilities.test.ts`

- [ ] **Step 1: Write failing tests for copy generation**

Add tests:

```ts
import {
  buildSourceIntentCopy,
  buildResultDestinationCopy,
  buildNotificationPolicyCopy,
  redactCapabilityPreviewText
} from "../scheduled-task-template-capabilities"

it("generates source-intent copy from source support metadata", () => {
  expect(
    buildSourceIntentCopy({
      sourceFamily: "feed",
      can_watch: true,
      can_ingest: false,
      can_preview: true,
      can_notify: false,
      can_index_search: false,
      can_index_rag: false,
      can_create: false,
      reason: "Ingest setup continues in Watchlists."
    })
  ).toEqual([
    "Detected source: feed.",
    "Watch: supported.",
    "Ingest: not supported for this source yet.",
    "Ingest setup continues in Watchlists."
  ])
})

it("generates destination copy from metadata", () => {
  expect(
    buildResultDestinationCopy({
      home_supported: false,
      notifications_supported: false,
      search_indexed: false,
      rag_scope_included: false
    })
  ).toEqual([
    "Home: not yet shown.",
    "Notifications: not available for this source yet.",
    "Search: content may be saved but not searchable.",
    "RAG: not included in the selected knowledge scope."
  ])
})

it("generates notification copy from support state", () => {
  expect(buildNotificationPolicyCopy({ notifications_supported: false })).toBe(
    "Notifications are not available for this source yet."
  )
  expect(buildNotificationPolicyCopy({ notifications_supported: true })).toBe(
    "Notifications can open exact task, run, or result detail when supported."
  )
})

it("redacts private-looking preview text", () => {
  expect(redactCapabilityPreviewText("https://example.com/feed?token=secret")).toBe(
    "[redacted private source]"
  )
  expect(redactCapabilityPreviewText("https://example.com/feed#private")).toBe(
    "[redacted private source]"
  )
  expect(redactCapabilityPreviewText("https://example.com/feed?api_key=secret")).toBe(
    "[redacted private source]"
  )
  expect(redactCapabilityPreviewText("https://example.com/feed?access_token=secret")).toBe(
    "[redacted private source]"
  )
  expect(redactCapabilityPreviewText("https://example.com/feed?client_secret=secret")).toBe(
    "[redacted private source]"
  )
  expect(redactCapabilityPreviewText("Authorization: Bearer abc123")).toBe(
    "[redacted private source]"
  )
  expect(redactCapabilityPreviewText("api key: sk-test-secret")).toBe(
    "[redacted private source]"
  )
  expect(redactCapabilityPreviewText("Provider response: token=private-value")).toBe(
    "[redacted private source]"
  )
  expect(redactCapabilityPreviewText("Public release feed")).toBe("Public release feed")
})
```

- [ ] **Step 2: Run failing tests**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-template-capabilities.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: fail because helpers are missing.

- [ ] **Step 3: Implement copy helpers**

Use deterministic copy and avoid raw implementation details:

```ts
export const buildSourceIntentCopy = (
  intent: ScheduledTaskSourceIntentCapability | null | undefined
): string[] => {
  if (!intent) {
    return ["Source support: configured in Watchlists."]
  }

  return [
    `Detected source: ${intent.sourceFamily.replace(/_/g, " ")}.`,
    intent.can_watch ? "Watch: supported." : "Watch: not supported for this source yet.",
    intent.can_ingest ? "Ingest: supported." : "Ingest: not supported for this source yet.",
    ...(intent.reason ? [redactCapabilityPreviewText(intent.reason)] : [])
  ]
}

export const buildResultDestinationCopy = (
  metadata: ScheduledTaskResultDestinationMetadata | null | undefined
): string[] => {
  if (!metadata) {
    return ["Results destination: configured in Watchlists."]
  }

  return [
    metadata.home_supported ? "Home: latest results will appear." : "Home: not yet shown.",
    metadata.notifications_supported
      ? "Notifications: available when the task policy triggers."
      : "Notifications: not available for this source yet.",
    metadata.search_indexed
      ? "Search: indexed when ingest completes."
      : "Search: content may be saved but not searchable.",
    metadata.rag_scope_included
      ? "RAG: included in the selected knowledge scope."
      : "RAG: not included in the selected knowledge scope."
  ]
}

export const buildNotificationPolicyCopy = (
  metadata: Pick<ScheduledTaskResultDestinationMetadata, "notifications_supported"> | null | undefined
): string =>
  metadata?.notifications_supported
    ? "Notifications can open exact task, run, or result detail when supported."
    : "Notifications are not available for this source yet."
```

For redaction, reuse the same sensitive URL/prose logic if it can be exported cleanly from `scheduled-task-templates.ts`. If exporting creates circular responsibility, duplicate a small private-looking pattern in this module and add tests. The redaction check must catch URL fragments, query params such as `token`, `api_key`, `access_token`, `client_secret`, `key`, `secret`, `password`, and `auth`, bearer tokens, prose such as `api key:` or `client secret:`, and provider snippets such as `Provider response: token=...`. Return one generic replacement string instead of partially masking secrets in this UI.

- [ ] **Step 4: Run tests**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-template-capabilities.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-template-capabilities.ts \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/scheduled-task-template-capabilities.test.ts
git commit -m "feat: add scheduled task capability copy helpers"
```

## Task 4: Integrate Capability States Into Create Panel

**Files:**

- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskCreatePanel.tsx`
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx`

- [ ] **Step 1: Write failing component tests**

Add tests:

```ts
import {
  REQUIRED_WATCH_AVAILABILITY_GATES,
  buildScheduledTaskTemplateCapability
} from "../scheduled-task-template-capabilities"

it("shows Limited availability without create language when a Watch gate is missing", () => {
  const capability = buildScheduledTaskTemplateCapability("watch", {
    passedGates: REQUIRED_WATCH_AVAILABILITY_GATES.filter(
      (gate) => gate !== "source_preview"
    ),
    sourceIntent: {
      sourceFamily: "feed",
      can_watch: true,
      can_ingest: false,
      can_preview: false,
      can_notify: false,
      can_index_search: false,
      can_index_rag: false,
      can_create: false,
      reason: "Ingest setup continues in Watchlists."
    },
    resultDestinations: {
      home_supported: false,
      notifications_supported: false,
      search_indexed: false,
      rag_scope_included: false
    }
  })

  render(
    <ScheduledTaskCreatePanel
      selectedTemplateId="watch"
      onSelectTemplate={vi.fn()}
      onCreateReminder={vi.fn()}
      templateCapabilities={{ watch: capability }}
    />
  )

  expect(screen.getByText("Limited availability")).toBeInTheDocument()
  expect(screen.getByText(/source preview/i)).toBeInTheDocument()
  expect(screen.getByText("Detected source: feed.")).toBeInTheDocument()
  expect(screen.getByText("Watch: supported.")).toBeInTheDocument()
  expect(screen.getByText("Ingest: not supported for this source yet.")).toBeInTheDocument()
  expect(screen.getByText("Ingest setup continues in Watchlists.")).toBeInTheDocument()
  expect(screen.getByText("Home: not yet shown.")).toBeInTheDocument()
  expect(screen.getByText("Notifications: not available for this source yet.")).toBeInTheDocument()
  expect(screen.getByText("No scheduled task has been created yet.")).toBeInTheDocument()
  expect(screen.queryByRole("button", { name: /Create watch/i })).not.toBeInTheDocument()
})

it("keeps Available now from showing Limited availability templates", () => {
  render(
    <ScheduledTaskCreatePanel
      selectedTemplateId={null}
      onSelectTemplate={vi.fn()}
      onCreateReminder={vi.fn()}
      templateCapabilities={{
        watch: buildScheduledTaskTemplateCapability("watch", {
          passedGates: REQUIRED_WATCH_AVAILABILITY_GATES.filter(
            (gate) => gate !== "source_preview"
          )
        })
      }}
    />
  )

  // Click Available now and assert Reminder remains while Watch is absent.
})

it("keeps capability essentials visible in an extension-width container", () => {
  const capability = buildScheduledTaskTemplateCapability("watch", {
    passedGates: REQUIRED_WATCH_AVAILABILITY_GATES.filter(
      (gate) => gate !== "source_preview"
    ),
    sourceIntent: {
      sourceFamily: "feed",
      can_watch: true,
      can_ingest: false,
      can_preview: false,
      can_notify: false,
      can_index_search: false,
      can_index_rag: false,
      can_create: false,
      reason: "Preview is not available for this source yet."
    }
  })

  render(
    <div style={{ width: 360 }}>
      <ScheduledTaskCreatePanel
        selectedTemplateId="watch"
        onSelectTemplate={vi.fn()}
        onCreateReminder={vi.fn()}
        templateCapabilities={{ watch: capability }}
      />
    </div>
  )

  expect(screen.getByText("Limited availability")).toBeInTheDocument()
  expect(screen.getByText("Detected source: feed.")).toBeInTheDocument()
  expect(screen.getByText("Preview is not available for this source yet.")).toBeInTheDocument()
  expect(screen.queryByRole("button", { name: /Create watch/i })).not.toBeInTheDocument()
})
```

Complete the second test using the existing `Segmented` labels. If Ant Design renders segmented options as labels instead of buttons, use `getByText("Available now")` and click its closest interactive parent, matching current test patterns.

The extension-width test is a jsdom presence test, not a pixel-perfect layout proof. It exists to prevent hiding the essential status, detected source, reason, and action-suppression copy behind desktop-only assumptions. If the implementation later adds browser e2e for the extension route, add a visual/narrow-viewport smoke there too.

- [ ] **Step 2: Run failing component tests**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: fail because `templateCapabilities` is not accepted and Limited availability is not rendered.

- [ ] **Step 3: Add capability props and effective templates**

Update props:

```ts
import type { ScheduledTaskTemplateCapabilityMap } from "./scheduled-task-template-capabilities"
import {
  applyScheduledTaskTemplateCapabilities,
  buildNotificationPolicyCopy,
  buildResultDestinationCopy,
  buildSourceIntentCopy,
  getMissingAvailabilityGates
} from "./scheduled-task-template-capabilities"

export interface ScheduledTaskCreatePanelProps {
  selectedTemplateId: ScheduledTaskTemplateId | null
  onSelectTemplate: (templateId: ScheduledTaskTemplateId | null) => void
  onCreateReminder: (payload: CreateScheduledTaskReminderPayload) => Promise<void> | void
  savingReminder?: boolean
  templateCapabilities?: ScheduledTaskTemplateCapabilityMap
}
```

Resolve templates:

```ts
const effectiveTemplates = useMemo(
  () => applyScheduledTaskTemplateCapabilities(SCHEDULED_TASK_TEMPLATES, templateCapabilities),
  [templateCapabilities]
)
const selectedTemplate = getScheduledTaskTemplate(selectedTemplateId, effectiveTemplates)
const templates = useMemo(
  () => filterScheduledTaskTemplates(filterId, effectiveTemplates),
  [effectiveTemplates, filterId]
)
```

If `getScheduledTaskTemplate` currently accepts only the static registry, update it in Task 2 or here to accept an optional template list.

- [ ] **Step 4: Render capability copy in non-Reminder panels**

Update `HandoffPanel` to accept capability:

```ts
const missingGates = getMissingAvailabilityGates(template.id, capability)
const sourceIntentCopy = buildSourceIntentCopy(capability?.sourceIntent)
const resultDestinationCopy = buildResultDestinationCopy(capability?.resultDestinations)
const notificationCopy = buildNotificationPolicyCopy(capability?.resultDestinations)
```

Render:

- `Limited availability` tag via existing state label.
- Missing gates as human-readable lines such as `Missing: source preview`.
- Source support copy under a heading such as `Source support`, including detected source family, Watch/Ingest support, and capability reason text after redaction.
- Destination copy under a heading such as `Result destinations`.
- Notification copy under a heading such as `Notifications`.
- Existing Watchlists setup and "No scheduled task has been created yet."

Do not add a Watch/Ingest create API, mutation, or success path. Keep all-gates plus adapter-guard availability coverage in pure helper tests only unless a later Watchlists creation adapter is implemented. Component/page defaults should exercise Handoff only and Limited availability states, not an artificial all-gates Available product path.

- [ ] **Step 5: Run component tests**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTaskCreatePanel.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx
git commit -m "feat: render scheduled task capability states"
```

## Task 5: Wire Default Capability Shell Into Page

**Files:**

- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx`
- Modify: `apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx`

- [ ] **Step 1: Write failing page regression tests**

Add or update tests:

```ts
it("does not describe planned automation through GitHub or YouTube as primary IA", async () => {
  renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=tasks")

  expect(await screen.findByText("No scheduled tasks yet.")).toBeInTheDocument()
  expect(screen.queryByText(/GitHub, YouTube/i)).not.toBeInTheDocument()
  expect(screen.getByText(/Watch and Ingest setup continue in their owner workspaces/i)).toBeInTheDocument()
})

it("keeps Watch template non-creating from the page route", async () => {
  renderWithQueryClient(<ScheduledTasksPage />, "/scheduled-tasks?tab=create&template=watch")

  expect(await screen.findByText("No scheduled task has been created yet.")).toBeInTheDocument()
  expect(screen.queryByRole("button", { name: /Create watch/i })).not.toBeInTheDocument()
})
```

Adjust exact copy to match the final UI text. The intent is to prevent source-vendor IA and accidental create CTA.

- [ ] **Step 2: Run failing page tests**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: fail if old empty-state copy still mentions GitHub/YouTube or page route lacks the new copy.

- [ ] **Step 3: Pass default capabilities into Create panel**

In the capability module, add:

```ts
export const DEFAULT_SCHEDULED_TASK_TEMPLATE_CAPABILITIES: ScheduledTaskTemplateCapabilityMap = {}
```

Pass it from `ScheduledTasksPage`:

```tsx
<ScheduledTaskCreatePanel
  selectedTemplateId={selectedTemplateId}
  onSelectTemplate={handleSelectTemplate}
  onCreateReminder={handleCreateReminderFromPanel}
  savingReminder={saving}
  templateCapabilities={DEFAULT_SCHEDULED_TASK_TEMPLATE_CAPABILITIES}
/>
```

Use an empty default map so Watch/Ingest remain `handoff_only` until runtime capability contracts and a real creation adapter exist. Do not set `creationAdapterSupported: true` in page defaults in Phase 2B.2.

- [ ] **Step 4: Update empty-state copy**

Replace source-vendor examples:

```tsx
<Typography.Text type="secondary">
  Create a reminder now. Watch and Ingest setup continue in their owner workspaces
  until capability, preview, duplicate, creation, and result contracts are available.
</Typography.Text>
```

- [ ] **Step 5: Run page tests**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx --maxWorkers=1 --no-file-parallelism
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add apps/packages/ui/src/components/Option/ScheduledTasks/ScheduledTasksPage.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx \
  apps/packages/ui/src/components/Option/ScheduledTasks/scheduled-task-template-capabilities.ts
git commit -m "feat: wire scheduled task capability shell"
```

## Task 6: Focused Full Verification

**Files:**

- Modify: implementation Backlog task final notes.

- [ ] **Step 1: Run focused unit/component tests**

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/ScheduledTasks/__tests__/scheduled-task-template-capabilities.test.ts \
  src/components/Option/ScheduledTasks/__tests__/scheduled-task-templates.test.ts \
  src/components/Option/ScheduledTasks/__tests__/ScheduledTaskCreatePanel.test.tsx \
  src/components/Option/ScheduledTasks/__tests__/ScheduledTasksPage.test.tsx \
  --maxWorkers=1 --no-file-parallelism
```

Expected: all tests pass.

- [ ] **Step 2: Run route-state regression tests**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/ScheduledTasks/__tests__/scheduled-task-route-state.test.ts --maxWorkers=1 --no-file-parallelism
```

Expected: pass.

- [ ] **Step 3: Run formatting/whitespace check**

```bash
git diff --check
```

Expected: no output, exit 0.

- [ ] **Step 4: Bandit scope decision**

No Python code should be touched in this implementation. Record in the implementation Backlog task:

```text
Bandit skipped: Phase 2B.2 capability-aware frontend shell touched TypeScript/frontend files only and no Python backend code.
```

If any Python file is touched unexpectedly, run Bandit on the touched Python scope before final commit:

```bash
source .venv/bin/activate
python -m bandit -r <touched_python_paths> -f json -o /tmp/bandit_scheduled_tasks_phase2b_capability_shell.json
```

- [ ] **Step 5: Self-review the diff**

Check:

- Watch/Ingest are not created from `/scheduled-tasks`.
- Reminder remains the only default Available template.
- Limited availability is not included in Available now.
- Watch/Ingest can resolve to Available only when all required gates and `creationAdapterSupported === true` pass.
- Source-intent copy is visible in Limited availability UI when metadata is present.
- Home/search/RAG/notification copy is generated from metadata.
- Capability text redacts URL fragments, secret query params, bearer/prose secrets, and provider snippets.
- Extension-width component coverage keeps status, source, reason, and action-suppression copy present.
- No hardcoded GitHub/YouTube primary IA copy was introduced.
- Existing Watchlists handoff copy still says no task was created.
- No backend/service API calls were added.

- [ ] **Step 6: Update Backlog implementation task**

Record:

- Files changed.
- Tests run and results.
- Bandit skip rationale or Bandit output path.
- Known skips or blockers.
- Final summary.

- [ ] **Step 7: Final commit**

```bash
git status --short
git add apps/packages/ui/src/components/Option/ScheduledTasks \
  backlog/tasks/<implementation-task-file>.md
git commit -m "feat: add scheduled task capability-aware create shell"
```

Expected: one final implementation commit if prior task commits were not already made, otherwise no extra commit needed beyond Backlog task finalization.

## Implementation Notes

- Keep all capability logic pure and testable. Do not hide behavior inside React state if it can be a helper.
- Prefer passing capability data into `ScheduledTaskCreatePanel` as props so future runtime API integration can replace the default empty map without rewiring the component.
- Avoid adding a fake backend or mock fetch in product code. Tests may provide capability fixtures directly.
- Do not expose raw gate IDs in user-facing copy. Convert `source_preview` to `source preview`.
- Do not use "Task created", "Watch scheduled", or "Ingest scheduled" unless the implementation actually calls a create API and gets a scheduled task ID.
- Keep Ant Design components consistent with existing ScheduledTasks code.

## Expected End State

After this plan is implemented:

- `/scheduled-tasks?tab=create&template=watch` can explain Watch capability status without creating a task.
- `/scheduled-tasks?tab=create&template=ingest` can explain Ingest capability status without promising search/RAG readiness.
- Tests prove Watch/Ingest cannot appear in Available now unless all gates and the explicit creation-adapter guard pass.
- Tests prove missing preview produces Limited availability.
- Source-intent, result, and notification copy is generated from metadata.
- Redaction tests cover URL fragments, secret query params, bearer/prose secrets, and provider snippets.
- Existing Reminder creation remains unchanged.
- Existing Watchlists UX remains unchanged.
