export type ScheduledTaskTabId = "overview" | "results" | "tasks" | "create"

export interface ScheduledTaskTabDefinition {
  id: ScheduledTaskTabId
  label: string
}

export interface ScheduledTaskRouteState {
  tab: ScheduledTaskTabId
  invalidTab: string | null
  templateId: string | null
  taskId: string | null
  runId: string | null
  resultId: string | null
}

export interface ParseScheduledTaskRouteStateOptions {
  defaultTab?: ScheduledTaskTabId
}

export const SCHEDULED_TASK_TABS: readonly ScheduledTaskTabDefinition[] = [
  { id: "overview", label: "Overview" },
  { id: "results", label: "Results" },
  { id: "tasks", label: "Tasks" },
  { id: "create", label: "Create" }
] as const

const SCHEDULED_TASK_TAB_IDS = new Set<ScheduledTaskTabId>(
  SCHEDULED_TASK_TABS.map((tab) => tab.id)
)

const normalizeNullableParam = (value: string | null): string | null => {
  const trimmed = String(value ?? "").trim()
  if (/[\r\n]/.test(trimmed)) return null
  return trimmed || null
}

export const parseScheduledTaskRouteState = (
  params: URLSearchParams,
  options: ParseScheduledTaskRouteStateOptions = {}
): ScheduledTaskRouteState => {
  const rawTab = normalizeNullableParam(params.get("tab"))
  const defaultTab =
    options.defaultTab && SCHEDULED_TASK_TAB_IDS.has(options.defaultTab)
      ? options.defaultTab
      : "overview"
  const isRawTabValid = rawTab
    ? SCHEDULED_TASK_TAB_IDS.has(rawTab as ScheduledTaskTabId)
    : false
  const tab =
    rawTab && isRawTabValid
      ? (rawTab as ScheduledTaskTabId)
      : rawTab
        ? "overview"
        : defaultTab

  const invalidTab =
    rawTab && !isRawTabValid ? rawTab : null

  const templateId = normalizeNullableParam(params.get("template"))
  const taskId = normalizeNullableParam(params.get("task_id"))
  const runId = normalizeNullableParam(params.get("run_id"))
  const resultId = normalizeNullableParam(params.get("result_id"))

  return {
    tab,
    invalidTab,
    templateId,
    taskId,
    runId,
    resultId
  }
}

export const buildScheduledTaskSearch = ({
  tab,
  templateId,
  taskId,
  runId,
  resultId
}: {
  tab: ScheduledTaskTabId
  templateId?: string | null
  taskId?: string | null
  runId?: string | null
  resultId?: string | null
}): string => {
  const params = new URLSearchParams()
  const normalizedTemplateId = normalizeNullableParam(templateId ?? null)
  const normalizedTaskId = normalizeNullableParam(taskId ?? null)
  const normalizedRunId = normalizeNullableParam(runId ?? null)
  const normalizedResultId = normalizeNullableParam(resultId ?? null)
  if (tab !== "overview") params.set("tab", tab)
  if (normalizedTemplateId && tab === "create") params.set("template", normalizedTemplateId)
  if (normalizedResultId && tab === "results") params.set("result_id", normalizedResultId)
  if (normalizedRunId && tab === "results") params.set("run_id", normalizedRunId)
  if (normalizedTaskId && (tab === "tasks" || tab === "results")) {
    params.set("task_id", normalizedTaskId)
  }
  const serialized = params.toString()
  return serialized ? `?${serialized}` : ""
}
