import {
  WORKSPACE_SOURCE_SAVED_VIEW_DATE_FIELDS,
  WORKSPACE_SOURCE_SAVED_VIEW_LIFECYCLE_STATE_FILTERS,
  WORKSPACE_SOURCE_SAVED_VIEW_REVIEW_STATE_FILTERS,
  WORKSPACE_SOURCE_SAVED_VIEW_SORTS,
  WORKSPACE_SOURCE_SAVED_VIEW_STATUS_FILTERS,
  WORKSPACE_SOURCE_SAVED_VIEW_TYPE_FILTERS,
  type WorkspaceSourceSavedViewStateV1
} from "@/types/workspace-source-saved-view"
import {
  DEFAULT_SOURCE_LIST_VIEW_STATE,
  type SourceListViewState
} from "./source-list-view"

export const SOURCE_SAVED_VIEW_SCHEMA_VERSION = 1
export const LARGE_SOURCE_FILE_BYTES = 50 * 1024 * 1024

const WIRE_FIELDS = [
  "type_filters",
  "status_filters",
  "review_state_filters",
  "lifecycle_state_filters",
  "date_field",
  "date_from",
  "date_to",
  "require_url",
  "require_file_size",
  "require_duration",
  "require_page_count",
  "file_size_min",
  "file_size_max",
  "duration_min",
  "duration_max",
  "page_count_min",
  "page_count_max",
  "sort"
] as const

type WireField = (typeof WIRE_FIELDS)[number]

export interface SourceViewStateValidationIssue {
  field: string
  message: string
}

export type SourceViewStateValidationResult =
  | { ok: true; state: WorkspaceSourceSavedViewStateV1 }
  | { ok: false; issues: SourceViewStateValidationIssue[] }

type WireValidationResult = SourceViewStateValidationResult

const isPlainObject = (value: unknown): value is Record<string, unknown> => {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    return false
  }
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}

const hasOwn = (value: Record<string, unknown>, field: string): boolean =>
  Object.prototype.hasOwnProperty.call(value, field)

const readValue = (
  value: Record<string, unknown>,
  field: WireField,
  defaultValue: unknown
): unknown => (hasOwn(value, field) ? value[field] : defaultValue)

const readEnumArray = <T extends string>(
  input: Record<string, unknown>,
  field: WireField,
  order: readonly T[],
  issues: SourceViewStateValidationIssue[]
): T[] => {
  const value = readValue(input, field, [])
  if (
    !Array.isArray(value) ||
    value.some(
      (entry) => typeof entry !== "string" || !order.includes(entry as T)
    )
  ) {
    issues.push({ field, message: "Must contain only supported values." })
    return []
  }
  return order.filter((entry) => value.includes(entry))
}

const readEnum = <T extends string>(
  input: Record<string, unknown>,
  field: WireField,
  allowed: readonly T[],
  defaultValue: T,
  issues: SourceViewStateValidationIssue[]
): T => {
  const value = readValue(input, field, defaultValue)
  if (typeof value !== "string" || !allowed.includes(value as T)) {
    issues.push({ field, message: "Must be a supported value." })
    return defaultValue
  }
  return value as T
}

const isCalendarDate = (value: string): boolean => {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value) || value.startsWith("0000-")) {
    return false
  }
  const timestamp = Date.parse(`${value}T00:00:00.000Z`)
  return (
    Number.isFinite(timestamp) &&
    new Date(timestamp).toISOString().slice(0, 10) === value
  )
}

const readDate = (
  input: Record<string, unknown>,
  field: "date_from" | "date_to",
  issues: SourceViewStateValidationIssue[]
): string | null => {
  const value = readValue(input, field, null)
  if (value !== null && (typeof value !== "string" || !isCalendarDate(value))) {
    issues.push({
      field,
      message: "Must be a real YYYY-MM-DD calendar date or null."
    })
    return null
  }
  return value as string | null
}

const readBoolean = (
  input: Record<string, unknown>,
  field: WireField,
  issues: SourceViewStateValidationIssue[]
): boolean => {
  const value = readValue(input, field, false)
  if (typeof value !== "boolean") {
    issues.push({ field, message: "Must be a boolean." })
    return false
  }
  return value
}

const readNumber = (
  input: Record<string, unknown>,
  field: WireField,
  issues: SourceViewStateValidationIssue[]
): number | null => {
  const value = readValue(input, field, null)
  if (
    value !== null &&
    (typeof value !== "number" || !Number.isFinite(value) || value < 0)
  ) {
    issues.push({
      field,
      message: "Must be a finite nonnegative number or null."
    })
    return null
  }
  return value as number | null
}

const validateWireState = (input: unknown): WireValidationResult => {
  if (!isPlainObject(input)) {
    return {
      ok: false,
      issues: [{ field: "state", message: "Must be an object." }]
    }
  }

  const issues: SourceViewStateValidationIssue[] = []
  for (const field of Object.keys(input)) {
    if (!WIRE_FIELDS.includes(field as WireField)) {
      issues.push({ field, message: "Unknown field." })
    }
  }

  const state: WorkspaceSourceSavedViewStateV1 = {
    type_filters: readEnumArray(
      input,
      "type_filters",
      WORKSPACE_SOURCE_SAVED_VIEW_TYPE_FILTERS,
      issues
    ),
    status_filters: readEnumArray(
      input,
      "status_filters",
      WORKSPACE_SOURCE_SAVED_VIEW_STATUS_FILTERS,
      issues
    ),
    review_state_filters: readEnumArray(
      input,
      "review_state_filters",
      WORKSPACE_SOURCE_SAVED_VIEW_REVIEW_STATE_FILTERS,
      issues
    ),
    lifecycle_state_filters: readEnumArray(
      input,
      "lifecycle_state_filters",
      WORKSPACE_SOURCE_SAVED_VIEW_LIFECYCLE_STATE_FILTERS,
      issues
    ),
    date_field: readEnum(
      input,
      "date_field",
      WORKSPACE_SOURCE_SAVED_VIEW_DATE_FIELDS,
      "added_at",
      issues
    ),
    date_from: readDate(input, "date_from", issues),
    date_to: readDate(input, "date_to", issues),
    require_url: readBoolean(input, "require_url", issues),
    require_file_size: readBoolean(input, "require_file_size", issues),
    require_duration: readBoolean(input, "require_duration", issues),
    require_page_count: readBoolean(input, "require_page_count", issues),
    file_size_min: readNumber(input, "file_size_min", issues),
    file_size_max: readNumber(input, "file_size_max", issues),
    duration_min: readNumber(input, "duration_min", issues),
    duration_max: readNumber(input, "duration_max", issues),
    page_count_min: readNumber(input, "page_count_min", issues),
    page_count_max: readNumber(input, "page_count_max", issues),
    sort: readEnum(
      input,
      "sort",
      WORKSPACE_SOURCE_SAVED_VIEW_SORTS,
      "manual",
      issues
    )
  }

  const ranges = [
    ["file_size_min", "file_size_max"],
    ["duration_min", "duration_max"],
    ["page_count_min", "page_count_max"]
  ] as const
  for (const [minimumField, maximumField] of ranges) {
    const minimum = state[minimumField]
    const maximum = state[maximumField]
    if (minimum !== null && maximum !== null && minimum > maximum) {
      issues.push({
        field: maximumField,
        message: `Must be greater than or equal to ${minimumField}.`
      })
    }
  }

  if (
    state.date_from !== null &&
    state.date_to !== null &&
    state.date_from > state.date_to
  ) {
    issues.push({
      field: "date_to",
      message: "Must be on or after date_from."
    })
  }

  return issues.length > 0 ? { ok: false, issues } : { ok: true, state }
}

const WIRE_TO_LOCAL_FIELDS: Record<WireField, keyof SourceListViewState> = {
  type_filters: "typeFilters",
  status_filters: "statusFilters",
  review_state_filters: "reviewStateFilters",
  lifecycle_state_filters: "lifecycleStateFilters",
  date_field: "dateField",
  date_from: "dateFrom",
  date_to: "dateTo",
  require_url: "requireUrl",
  require_file_size: "requireFileSize",
  require_duration: "requireDuration",
  require_page_count: "requirePageCount",
  file_size_min: "fileSizeMin",
  file_size_max: "fileSizeMax",
  duration_min: "durationMin",
  duration_max: "durationMax",
  page_count_min: "pageCountMin",
  page_count_max: "pageCountMax",
  sort: "sort"
}

export const serializeSourceListViewState = (
  state: SourceListViewState
): SourceViewStateValidationResult => {
  if (!isPlainObject(state)) {
    return {
      ok: false,
      issues: [{ field: "state", message: "Must be an object." }]
    }
  }

  const wireState = {
    type_filters: state.typeFilters,
    status_filters: state.statusFilters,
    review_state_filters: state.reviewStateFilters,
    lifecycle_state_filters: state.lifecycleStateFilters,
    date_field:
      state.dateField === "addedAt"
        ? "added_at"
        : state.dateField === "sourceCreatedAt"
          ? "source_created_at"
          : state.dateField,
    date_from: state.dateFrom,
    date_to: state.dateTo,
    require_url: state.requireUrl,
    require_file_size: state.requireFileSize,
    require_duration: state.requireDuration,
    require_page_count: state.requirePageCount,
    file_size_min: state.fileSizeMin,
    file_size_max: state.fileSizeMax,
    duration_min: state.durationMin,
    duration_max: state.durationMax,
    page_count_min: state.pageCountMin,
    page_count_max: state.pageCountMax,
    sort: state.sort
  }
  const result = validateWireState(wireState)
  if (result.ok === false) {
    return {
      ok: false,
      issues: result.issues.map((issue) => ({
        ...issue,
        field: WIRE_TO_LOCAL_FIELDS[issue.field as WireField] ?? issue.field
      }))
    }
  }
  return result
}

export const deserializeSourceViewState = (
  payload: unknown
): WorkspaceSourceSavedViewStateV1 | null => {
  const result = validateWireState(payload)
  return result.ok ? result.state : null
}

export const applySavedSourceViewState = (
  current: SourceListViewState,
  saved: WorkspaceSourceSavedViewStateV1
): SourceListViewState => ({
  expanded: current.expanded,
  typeFilters: [...saved.type_filters],
  statusFilters: [...saved.status_filters],
  reviewStateFilters: [...saved.review_state_filters],
  lifecycleStateFilters: [...saved.lifecycle_state_filters],
  dateField: saved.date_field === "added_at" ? "addedAt" : "sourceCreatedAt",
  dateFrom: saved.date_from,
  dateTo: saved.date_to,
  requireUrl: saved.require_url,
  requireFileSize: saved.require_file_size,
  requireDuration: saved.require_duration,
  requirePageCount: saved.require_page_count,
  fileSizeMin: saved.file_size_min,
  fileSizeMax: saved.file_size_max,
  durationMin: saved.duration_min,
  durationMax: saved.duration_max,
  pageCountMin: saved.page_count_min,
  pageCountMax: saved.page_count_max,
  sort: saved.sort
})

const orderedWireState = (
  state: WorkspaceSourceSavedViewStateV1
): WorkspaceSourceSavedViewStateV1 => ({
  type_filters: state.type_filters,
  status_filters: state.status_filters,
  review_state_filters: state.review_state_filters,
  lifecycle_state_filters: state.lifecycle_state_filters,
  date_field: state.date_field,
  date_from: state.date_from,
  date_to: state.date_to,
  require_url: state.require_url,
  require_file_size: state.require_file_size,
  require_duration: state.require_duration,
  require_page_count: state.require_page_count,
  file_size_min: state.file_size_min,
  file_size_max: state.file_size_max,
  duration_min: state.duration_min,
  duration_max: state.duration_max,
  page_count_min: state.page_count_min,
  page_count_max: state.page_count_max,
  sort: state.sort
})

export const getSourceViewStateSignature = (
  state: unknown
): string | null => {
  const result = validateWireState(state)
  return result.ok ? JSON.stringify(orderedWireState(result.state)) : null
}

export const areSourceViewStatesEqual = (
  left: unknown,
  right: unknown
): boolean => {
  const leftSignature = getSourceViewStateSignature(left)
  const rightSignature = getSourceViewStateSignature(right)
  return (
    leftSignature !== null &&
    rightSignature !== null &&
    leftSignature === rightSignature
  )
}

export const getSourceListViewStateSignature = (
  state: SourceListViewState
): string | null => {
  const result = serializeSourceListViewState(state)
  return result.ok ? getSourceViewStateSignature(result.state) : null
}

export const isSourceListViewStateModified = (
  state: SourceListViewState,
  savedSignature: string
): boolean => getSourceListViewStateSignature(state) !== savedSignature

const freezePresetState = (
  fields: Partial<SourceListViewState>
): SourceListViewState => {
  const state: SourceListViewState = {
    ...DEFAULT_SOURCE_LIST_VIEW_STATE,
    ...fields,
    typeFilters: [...(fields.typeFilters ?? [])],
    statusFilters: [...(fields.statusFilters ?? [])],
    reviewStateFilters: [...(fields.reviewStateFilters ?? [])],
    lifecycleStateFilters: [...(fields.lifecycleStateFilters ?? [])]
  }
  Object.freeze(state.typeFilters)
  Object.freeze(state.statusFilters)
  Object.freeze(state.reviewStateFilters)
  Object.freeze(state.lifecycleStateFilters)
  return Object.freeze(state)
}

const preset = (label: string, fields: Partial<SourceListViewState>) =>
  Object.freeze({ label, state: freezePresetState(fields) })

export const SOURCE_VIEW_PRESETS = Object.freeze({
  needsReview: preset("Needs review", {
    reviewStateFilters: ["needs_review"]
  }),
  unreviewed: preset("Unreviewed", { reviewStateFilters: ["unset"] }),
  failedIngest: preset("Failed ingest", { statusFilters: ["error"] }),
  partiallyIndexed: preset("Partially indexed", {
    lifecycleStateFilters: ["partially_queryable"]
  }),
  pdfs: preset("PDFs", { typeFilters: ["pdf"] }),
  webCaptures: preset("Web captures", { typeFilters: ["website"] }),
  largeFiles: preset("Large files", { fileSizeMin: LARGE_SOURCE_FILE_BYTES })
})
