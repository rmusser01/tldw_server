import type { ScheduledTaskTemplateId } from "./scheduled-task-templates"

export type PlannedScheduledTaskTemplateId = Extract<
  ScheduledTaskTemplateId,
  "recurring_question" | "agent_task"
>

export type PlannedRequirementStatus =
  | "planned"
  | "related_available"
  | "missing"

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

const PLANNED_TEMPLATE_IDS = new Set<ScheduledTaskTemplateId>([
  "recurring_question",
  "agent_task"
])

const PLANNED_TEMPLATE_PANEL_MODELS: Record<
  PlannedScheduledTaskTemplateId,
  PlannedScheduledTaskPanelModel
> = {
  recurring_question: {
    templateId: "recurring_question",
    statusLabel: "Planned automation type",
    jobStatement:
      "Run this question on a schedule across selected searchable content.",
    availabilityReason:
      "Recurring Question scheduling is planned for the API contract and is not executable in this client yet.",
    requirements: [
      {
        label: "Scheduled RAG query support",
        detail: "The backend needs an execution path for recurring RAG queries.",
        status: "planned"
      },
      {
        label: "Searchable scope selection",
        detail: "Users need to choose which searchable content each run can query.",
        status: "planned"
      },
      {
        label: "Normalized run history",
        detail: "Each scheduled answer needs a durable task history entry.",
        status: "planned"
      },
      {
        label: "Task visibility policy",
        detail: "Tasks need explicit rules for surfacing results outside history.",
        status: "planned"
      }
    ],
    resultDestinations: [
      "Every run is recorded in task history.",
      "Home and Results receive summaries only when selected by the task visibility policy."
    ],
    safetyLines: [],
    links: [
      { label: "Open Research", href: "/research" },
      { label: "Open Results", href: "/scheduled-tasks/results" }
    ],
    createEnabled: false
  },
  agent_task: {
    templateId: "agent_task",
    statusLabel: "Planned automation type",
    jobStatement: "Send this message to the selected agent at the scheduled time.",
    availabilityReason:
      "Agent Task scheduling is planned for the API contract and is not executable in this client yet.",
    requirements: [
      {
        label: "Schedulable ACP/API agents",
        detail: "The platform needs agents that can accept scheduled work safely.",
        status: "planned"
      },
      {
        label: "Preview and risk classification",
        detail: "Each task needs a preview and risk class before it can be scheduled.",
        status: "planned"
      },
      {
        label: "Approval policy",
        detail: "Permission-sensitive runs need configurable approval handling.",
        status: "planned"
      },
      {
        label: "Normalized agent run outputs",
        detail: "Agent outputs need a consistent task result record.",
        status: "planned"
      }
    ],
    resultDestinations: [
      "Every run is recorded in task history.",
      "Home and Results receive summaries only when selected by the task visibility policy."
    ],
    safetyLines: [
      "Preview is required before scheduling an agent task.",
      "Some permission classes may require approval before each run."
    ],
    links: [
      { label: "Open Agent Tasks", href: "/agent-tasks" },
      { label: "Open ACP Playground", href: "/acp-playground" },
      { label: "Open Results", href: "/scheduled-tasks/results" }
    ],
    createEnabled: false
  }
}

const clonePlannedScheduledTaskPanelModel = (
  model: PlannedScheduledTaskPanelModel
): PlannedScheduledTaskPanelModel => ({
  ...model,
  requirements: model.requirements.map((requirement) => ({ ...requirement })),
  resultDestinations: [...model.resultDestinations],
  safetyLines: [...model.safetyLines],
  links: model.links.map((link) => ({ ...link }))
})

export const isPlannedAutomationTemplate = (
  templateId: ScheduledTaskTemplateId
): templateId is PlannedScheduledTaskTemplateId =>
  PLANNED_TEMPLATE_IDS.has(templateId)

export const buildPlannedScheduledTaskPanelModel = (
  templateId: ScheduledTaskTemplateId
): PlannedScheduledTaskPanelModel | null =>
  isPlannedAutomationTemplate(templateId)
    ? clonePlannedScheduledTaskPanelModel(PLANNED_TEMPLATE_PANEL_MODELS[templateId])
    : null
