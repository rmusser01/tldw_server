import { bgRequest } from "@/services/background-proxy"
import type { AllowedPath } from "@/services/tldw/openapi-guard"
import { appendPathQuery, toAllowedPath } from "@/services/tldw/path-utils"

export type NoteTaskStatus = "open" | "done"
export type NoteTaskProjectionStatus = "live" | "unlinked" | "ambiguous" | "deleted"
export type NoteTaskPriority = "high" | "medium" | "low"

export type NoteTaskMetadata = {
  due_date?: string | null
  priority?: NoteTaskPriority | null
  estimate?: string | null
}

export type NoteTaskProjection = {
  note_id: string
  note_version: number
  line_number: number
  start_offset: number
  end_offset: number
  raw_line: string
  has_child_content: boolean
  projection_status: NoteTaskProjectionStatus
}

export type NoteTaskNoteSummary = {
  id: string
  title: string
  version: number
}

export type NoteTask = {
  id: string
  note_id: string
  text: string
  status: NoteTaskStatus
  metadata: NoteTaskMetadata
  projection_status: NoteTaskProjectionStatus
  version: number
  created_at?: string | null
  updated_at?: string | null
  completed_at?: string | null
  note?: NoteTaskNoteSummary | null
  projection?: NoteTaskProjection | null
}

export type NoteTaskReconciliationSummary = {
  status: "clean" | "warnings" | "incomplete"
  note_id?: string | null
  note_version?: number | null
  parsed_count?: number | null
  created_count?: number
  updated_count?: number
  unlinked_count?: number
  ambiguous_count?: number
  warning_count?: number
  processed_notes?: number
  remaining_stale_notes?: number
}

export type NoteTaskListResponse = {
  tasks: NoteTask[]
  reconciliation: NoteTaskReconciliationSummary
}

export type NoteTaskCreateRequest = {
  text: string
  status?: NoteTaskStatus
  metadata?: NoteTaskMetadata
  expected_note_version: number
}

export type NoteTaskUpdateRequest = {
  text?: string
  metadata?: NoteTaskMetadata
  expected_task_version: number
  expected_note_version?: number | null
  record_only?: boolean
}

export type NoteTaskStatusUpdate = {
  task_id: string
  status: NoteTaskStatus
  expected_task_version: number
  expected_note_version?: number | null
  record_only?: boolean
}

export type NoteTaskDeleteRequest = {
  expected_task_version: number
  expected_note_version?: number | null
  record_only?: boolean
}

export type NoteTaskActivityEvent = {
  id: string
  task_id?: string | null
  note_id?: string | null
  event_type: string
  actor_type: string
  actor_id?: string | null
  tool_name?: string | null
  policy_mode?: string | null
  approval_id?: string | null
  old_value?: Record<string, unknown> | null
  new_value?: Record<string, unknown> | null
  created_at: string
  read_at?: string | null
  dismissed_at?: string | null
}

export type NoteTaskActivityListResponse = {
  events: NoteTaskActivityEvent[]
}

export type NoteTaskActivityPatch = {
  read?: boolean
  dismissed?: boolean
}

export type NoteTaskActivityState = {
  event_id: string
  user_id: string
  read_at?: string | null
  dismissed_at?: string | null
}

type QueryValue = string | number | boolean | null | undefined

const buildQuery = (params: Record<string, QueryValue>): string => {
  const query = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") continue
    query.set(key, String(value))
  }
  const serialized = query.toString()
  return serialized ? `?${serialized}` : ""
}

const pathWithQuery = (path: string, params: Record<string, QueryValue>): AllowedPath =>
  appendPathQuery(toAllowedPath(path), buildQuery(params))

const encodePathId = (value: string | number): string => encodeURIComponent(String(value))

export const listTasks = async (params: {
  status?: NoteTaskStatus | null
  projection_status?: NoteTaskProjectionStatus | null
  limit?: number
  reconcile_limit?: number
} = {}): Promise<NoteTaskListResponse> =>
  bgRequest<NoteTaskListResponse, AllowedPath, "GET">({
    path: pathWithQuery("/api/v1/notes/tasks", params),
    method: "GET"
  })

export const listNoteTasks = async (
  noteId: string | number,
  params: { limit?: number } = {}
): Promise<NoteTaskListResponse> =>
  bgRequest<NoteTaskListResponse, AllowedPath, "GET">({
    path: pathWithQuery(`/api/v1/notes/${encodePathId(noteId)}/tasks`, params),
    method: "GET"
  })

export const getTask = async (taskId: string | number): Promise<NoteTask> =>
  bgRequest<NoteTask, AllowedPath, "GET">({
    path: toAllowedPath(`/api/v1/notes/tasks/${encodePathId(taskId)}`),
    method: "GET"
  })

export const createNoteTask = async (
  noteId: string | number,
  body: NoteTaskCreateRequest
): Promise<NoteTask> =>
  bgRequest<NoteTask, AllowedPath, "POST">({
    path: toAllowedPath(`/api/v1/notes/${encodePathId(noteId)}/tasks`),
    method: "POST",
    body
  })

export const updateNoteTask = async (
  taskId: string | number,
  body: NoteTaskUpdateRequest
): Promise<NoteTask> =>
  bgRequest<NoteTask, AllowedPath, "PATCH">({
    path: toAllowedPath(`/api/v1/notes/tasks/${encodePathId(taskId)}`),
    method: "PATCH",
    body
  })

export const setNoteTaskStatus = async (
  updates: NoteTaskStatusUpdate[]
): Promise<{ tasks: NoteTask[] }> =>
  bgRequest<{ tasks: NoteTask[] }, AllowedPath, "POST">({
    path: toAllowedPath("/api/v1/notes/tasks/status"),
    method: "POST",
    body: { updates }
  })

export const deleteNoteTask = async (
  taskId: string | number,
  body: NoteTaskDeleteRequest
): Promise<NoteTask> =>
  bgRequest<NoteTask, AllowedPath, "DELETE">({
    path: pathWithQuery(`/api/v1/notes/tasks/${encodePathId(taskId)}`, body),
    method: "DELETE"
  })

export const reconcileNoteTasks = async (
  noteId: string | number
): Promise<NoteTaskReconciliationSummary> =>
  bgRequest<NoteTaskReconciliationSummary, AllowedPath, "POST">({
    path: toAllowedPath(`/api/v1/notes/${encodePathId(noteId)}/tasks/reconcile`),
    method: "POST"
  })

export const listTaskActivity = async (params: {
  note_id?: string | number | null
  limit?: number
} = {}): Promise<NoteTaskActivityListResponse> =>
  bgRequest<NoteTaskActivityListResponse, AllowedPath, "GET">({
    path: pathWithQuery("/api/v1/notes/tasks/activity", params),
    method: "GET"
  })

export const markTaskActivityRead = async (
  eventId: string | number,
  body: NoteTaskActivityPatch
): Promise<NoteTaskActivityState> =>
  bgRequest<NoteTaskActivityState, AllowedPath, "PATCH">({
    path: toAllowedPath(`/api/v1/notes/tasks/activity/${encodePathId(eventId)}`),
    method: "PATCH",
    body
  })
