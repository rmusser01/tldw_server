export type ScheduledTaskTabId = "overview" | "tasks" | "create"

export interface ScheduledTaskTabDefinition {
  id: ScheduledTaskTabId
  label: string
}

export interface ScheduledTaskRouteState {
  tab: ScheduledTaskTabId
  invalidTab: string | null
  templateId: string | null
  taskId: string | null
}

export const SCHEDULED_TASK_TABS: readonly ScheduledTaskTabDefinition[] = [
  { id: "overview", label: "Overview" },
  { id: "tasks", label: "Tasks" },
  { id: "create", label: "Create" }
] as const

const SCHEDULED_TASK_TAB_IDS = new Set<ScheduledTaskTabId>(
  SCHEDULED_TASK_TABS.map((tab) => tab.id)
)

const normalizeNullableParam = (value: string | null): string | null => {
  const trimmed = String(value ?? "").trim()
  return trimmed || null
}

export const parseScheduledTaskRouteState = (
  params: URLSearchParams
): ScheduledTaskRouteState => {
  const rawTab = normalizeNullableParam(params.get("tab"))
  const tab =
    rawTab && SCHEDULED_TASK_TAB_IDS.has(rawTab as ScheduledTaskTabId)
      ? (rawTab as ScheduledTaskTabId)
      : "overview"

  return {
    tab,
    invalidTab: rawTab && tab === "overview" && rawTab !== "overview" ? rawTab : null,
    templateId: normalizeNullableParam(params.get("template")),
    taskId: normalizeNullableParam(params.get("task_id"))
  }
}

export const buildScheduledTaskSearch = ({
  tab,
  templateId,
  taskId
}: {
  tab: ScheduledTaskTabId
  templateId?: string | null
  taskId?: string | null
}): string => {
  const params = new URLSearchParams()
  const normalizedTemplateId = normalizeNullableParam(templateId ?? null)
  const normalizedTaskId = normalizeNullableParam(taskId ?? null)
  if (tab !== "overview") params.set("tab", tab)
  if (normalizedTemplateId && tab === "create") params.set("template", normalizedTemplateId)
  if (normalizedTaskId && tab === "tasks") params.set("task_id", normalizedTaskId)
  const serialized = params.toString()
  return serialized ? `?${serialized}` : ""
}
