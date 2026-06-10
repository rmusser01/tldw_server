import type {
  ScheduledTask,
  ScheduledTaskAutomationFamily
} from "@/services/scheduled-tasks-control-plane"

import type {
  ScheduledTaskProductStatus,
  ScheduledTaskStatusKey
} from "./scheduled-task-status"

type AutomationDefinitionTaskInput =
  | ScheduledTask
  | {
      primitive?: unknown
      status?: string
      enabled?: boolean
      source_ref?: Record<string, unknown>
    }

type AutomationDefinitionTask = AutomationDefinitionTaskInput & {
  primitive: "automation_definition"
}

const AUTOMATION_FAMILY_LABELS: Record<ScheduledTaskAutomationFamily, string> = {
  recurring_question: "Recurring question",
  agent_task: "Agent task"
}

const isKnownAutomationFamily = (
  value: unknown
): value is ScheduledTaskAutomationFamily =>
  value === "recurring_question" || value === "agent_task"

const KNOWN_AUTOMATION_LIFECYCLES = new Set([
  "configured",
  "paused",
  "archived",
  "disabled"
])

const KNOWN_AUTOMATION_HEALTH = new Set([
  "ready",
  "execution_unavailable",
  "capability_unavailable",
  "needs_attention",
  "permission_required"
])

const KNOWN_AUTOMATION_STATUSES = new Set([
  "configured",
  "configured_execution_unavailable",
  "paused",
  "archived",
  "disabled"
])

const makeAutomationStatus = (
  key: ScheduledTaskStatusKey,
  label: string,
  tone: ScheduledTaskProductStatus["tone"],
  description: string
): ScheduledTaskProductStatus => ({
  key,
  label,
  tone,
  description
})

const makeUnknownAutomationStatus = (): ScheduledTaskProductStatus =>
  makeAutomationStatus(
    "needs_attention",
    "Needs attention",
    "error",
    "This automation definition has unrecognized status metadata and should be reviewed before it runs."
  )

export const isAutomationDefinitionTask = (
  task: AutomationDefinitionTaskInput
): task is AutomationDefinitionTask => task.primitive === "automation_definition"

export const getAutomationDefinitionFamilyLabel = (
  task: Pick<AutomationDefinitionTaskInput, "source_ref">
): string => {
  const family = task.source_ref?.family
  return isKnownAutomationFamily(family)
    ? AUTOMATION_FAMILY_LABELS[family]
    : "Automation"
}

export const getAutomationDefinitionProductStatus = (
  task: AutomationDefinitionTask
): ScheduledTaskProductStatus => {
  const status = task.status || ""
  const sourceRef = task.source_ref || {}
  const lifecycle = typeof sourceRef.lifecycle === "string" ? sourceRef.lifecycle : ""
  const health = typeof sourceRef.health === "string" ? sourceRef.health : ""

  if (lifecycle === "paused" || status === "paused") {
    return makeAutomationStatus(
      "paused",
      "Paused",
      "warning",
      "This automation definition is paused and will not run until resumed."
    )
  }

  if (lifecycle === "archived" || status === "archived") {
    return makeAutomationStatus(
      "archived",
      "Archived",
      "default",
      "This automation definition is archived and cannot run."
    )
  }

  if (lifecycle === "disabled" || status === "disabled") {
    return makeAutomationStatus(
      "disabled",
      "Disabled",
      "default",
      "This automation definition is disabled and cannot run."
    )
  }

  const hasUnknownLifecycle =
    Boolean(lifecycle) && !KNOWN_AUTOMATION_LIFECYCLES.has(lifecycle)
  const hasUnknownHealth = Boolean(health) && !KNOWN_AUTOMATION_HEALTH.has(health)
  const hasUnknownStatus =
    Boolean(status) && !KNOWN_AUTOMATION_STATUSES.has(status)

  if (hasUnknownLifecycle || hasUnknownHealth || hasUnknownStatus) {
    return makeUnknownAutomationStatus()
  }

  if (
    status === "configured_execution_unavailable" ||
    (lifecycle === "configured" && health === "execution_unavailable")
  ) {
    return makeAutomationStatus(
      "blocked",
      "Configured, execution unavailable",
      "warning",
      "This automation definition is configured, but execution is unavailable."
    )
  }

  if (health === "permission_required" || health === "capability_unavailable") {
    return makeAutomationStatus(
      "blocked",
      "Execution unavailable",
      "warning",
      "This automation definition cannot run until the required capability is available."
    )
  }

  if (health === "needs_attention") {
    return makeAutomationStatus(
      "needs_attention",
      "Needs attention",
      "error",
      "This automation definition needs attention before it can run reliably."
    )
  }

  if (lifecycle === "configured" && health === "ready") {
    return makeAutomationStatus(
      "waiting",
      "Configured",
      "processing",
      "This automation definition is configured and ready for execution."
    )
  }

  if (task.enabled === false) {
    return makeAutomationStatus(
      "disabled",
      "Disabled",
      "default",
      "This automation definition is disabled and cannot run."
    )
  }

  return makeUnknownAutomationStatus()
}
