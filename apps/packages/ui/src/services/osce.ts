import { bgRequest } from "@/services/background-proxy"
import type { AllowedPath } from "@/services/tldw/openapi-guard"

export type OsceActivityType = "questions" | "osce"
export type OsceVerificationState =
  | "source_verified"
  | "modified_after_verification"
  | "manually_authored"
export type OsceAttemptState = "in_progress" | "self_assessment" | "completed"
export type OsceCitationSourceType = "media" | "document" | "url" | "note"
export type OsceStationOrigin = "generated" | "manual"
export type OsceChecklistSelection = "met" | "not_met"

export type OsceCitation = {
  source_type: OsceCitationSourceType
  source_id: string
  label?: string | null
  quote?: string | null
  media_id?: number | null
  chunk_id?: string | null
  timestamp_seconds?: number | null
  page_number?: number | null
  source_url?: string | null
}

export type OscePatientContext = {
  text: string
  citations: OsceCitation[]
}

export type OsceChecklistItemCreate = {
  label: string
  rationale?: string | null
  citations: OsceCitation[]
}

export type OsceChecklistItemStored = OsceChecklistItemCreate & { id: string }
export type OsceChecklistItemUpdate = OsceChecklistItemCreate & { id?: string }

export type OsceRubricLevelCreate = { label: string; description: string }
export type OsceRubricLevelStored = OsceRubricLevelCreate & { id: string }
export type OsceRubricLevelUpdate = OsceRubricLevelCreate & { id?: string }

export type OsceRubricDomainCreate = {
  label: string
  levels: OsceRubricLevelCreate[]
}

export type OsceRubricDomainStored = {
  id: string
  label: string
  levels: OsceRubricLevelStored[]
}

export type OsceRubricDomainUpdate = {
  id?: string
  label: string
  levels: OsceRubricLevelUpdate[]
}

export type OsceKeyPointCreate = { text: string; citations: OsceCitation[] }
export type OsceKeyPointStored = OsceKeyPointCreate & { id: string }
export type OsceKeyPointUpdate = OsceKeyPointCreate & { id?: string }

export type OsceStationCreateContent = {
  schema_version: "osce.station.v1"
  title: string
  candidate_instructions: string
  candidate_task: string
  patient_context: OscePatientContext
  recommended_duration_seconds: number
  checklist_items: OsceChecklistItemCreate[]
  rubric_domains: OsceRubricDomainCreate[]
  expected_key_points: OsceKeyPointCreate[]
}

export type OsceStationStoredContent = Omit<
  OsceStationCreateContent,
  "checklist_items" | "rubric_domains" | "expected_key_points"
> & {
  checklist_items: OsceChecklistItemStored[]
  rubric_domains: OsceRubricDomainStored[]
  expected_key_points: OsceKeyPointStored[]
}

export type OsceStationUpdateContent = Partial<Omit<
  OsceStationCreateContent,
  "checklist_items" | "rubric_domains" | "expected_key_points"
>> & {
  checklist_items?: OsceChecklistItemUpdate[]
  rubric_domains?: OsceRubricDomainUpdate[]
  expected_key_points?: OsceKeyPointUpdate[]
}

export type OsceStationSummary = {
  id: number
  quiz_id: number
  title: string
  recommended_duration_seconds: number
  order_index: number
  version: number
  checklist_count: number
  rubric_domain_count: number
  verification_state: OsceVerificationState
  created_at: string
  updated_at: string
}

export type OsceStationAuthoringResponse = {
  id: number
  quiz_id: number
  content: OsceStationStoredContent
  order_index: number
  version: number
  origin: OsceStationOrigin
  provenance: Record<string, unknown> | null
  source_bundle: Array<Record<string, unknown>>
  verification_state: OsceVerificationState
  verification_timestamp: string | null
  verification_summary: string | null
  deleted: boolean
  created_at: string
  updated_at: string
}

export type OscePagination = {
  total: number
  offset: number
  limit: number
  returned: number
  has_more: boolean
  next_offset: number | null
}

export type OsceOffsetPage<T> = {
  items: T[]
  count: number
  has_more: boolean | null
  next_offset: number | null
  pagination: OscePagination
}

export type OsceCandidateCitation = Pick<OsceCitation, "source_type" | "source_id" | "label">
export type OsceCandidateStation = Pick<
  OsceStationCreateContent,
  "schema_version" | "title" | "candidate_instructions" | "candidate_task" | "recommended_duration_seconds"
> & {
  patient_context: { text: string; citations: OsceCandidateCitation[] }
}

type OsceAttemptBase = {
  id: number
  quiz_id: number
  station_id: number
  client_attempt_id: string
  version: number
  notes: string
  started_at: string
  last_modified_at: string
  server_time: string
}

export type OsceCandidateAttempt = OsceAttemptBase & {
  state: "in_progress"
  station: OsceCandidateStation
}

export type OsceRevealedAttempt = OsceAttemptBase & {
  state: "self_assessment" | "completed"
  station: OsceStationStoredContent
  checklist_selections: Record<string, OsceChecklistSelection>
  rubric_selections: Record<string, string>
  self_assessment_started_at: string
  completed_at: string | null
  elapsed_seconds: number
}

export type OsceAttempt = OsceCandidateAttempt | OsceRevealedAttempt

export type OsceRubricResult = {
  domain_id: string
  domain_label: string
  level_id: string
  level_label: string
}

export type OsceAttemptSummary = {
  id: number
  quiz_id: number
  station_id: number
  client_attempt_id: string
  station_title: string
  state: OsceAttemptState
  version: number
  started_at: string
  self_assessment_started_at: string | null
  completed_at: string | null
  last_modified_at: string
  elapsed_seconds: number | null
  checklist_met_count: number | null
  checklist_total: number | null
  rubric_results: OsceRubricResult[]
}

export type OsceStationListParams = { limit?: number; offset?: number }
export type OsceAttemptFilters = {
  quiz_id?: number
  station_id?: number
  states?: OsceAttemptState[]
  limit?: number
  offset?: number
}
export type OsceStationCreateRequest = { content: OsceStationCreateContent; order_index?: number }
export type OsceStationPatchRequest = {
  expected_version: number
  content: OsceStationUpdateContent
  order_index?: number
}
export type OsceAttemptPatch = {
  expected_version: number
  notes?: string | null
  checklist_selections?: Record<string, OsceChecklistSelection> | null
  rubric_selections?: Record<string, string> | null
}

const request = <T, M extends "GET" | "POST" | "PATCH" | "DELETE">(
  path: string,
  method: M,
  body?: unknown,
  signal?: AbortSignal
): Promise<T> => bgRequest<T, AllowedPath, M>({
  path: path as AllowedPath,
  method,
  ...(body === undefined ? {} : { headers: { "Content-Type": "application/json" }, body }),
  abortSignal: signal
})

const appendNumber = (params: URLSearchParams, key: string, value?: number) => {
  if (value !== undefined) params.append(key, String(value))
}

export const listOsceStations = (
  quizId: number,
  params: OsceStationListParams = {},
  options?: { signal?: AbortSignal }
): Promise<OsceOffsetPage<OsceStationSummary>> => {
  const query = new URLSearchParams()
  appendNumber(query, "limit", params.limit)
  appendNumber(query, "offset", params.offset)
  const suffix = query.size > 0 ? `?${query.toString()}` : ""
  return request(`/api/v1/quizzes/${quizId}/osce-stations${suffix}`, "GET", undefined, options?.signal)
}

export const getOsceStation = (
  quizId: number,
  stationId: number,
  options?: { signal?: AbortSignal }
): Promise<OsceStationAuthoringResponse> =>
  request(`/api/v1/quizzes/${quizId}/osce-stations/${stationId}`, "GET", undefined, options?.signal)

export const createOsceStation = (
  quizId: number,
  input: OsceStationCreateRequest,
  options?: { signal?: AbortSignal }
): Promise<OsceStationAuthoringResponse> =>
  request(`/api/v1/quizzes/${quizId}/osce-stations`, "POST", input, options?.signal)

export const updateOsceStation = (
  quizId: number,
  stationId: number,
  input: OsceStationPatchRequest,
  options?: { signal?: AbortSignal }
): Promise<OsceStationAuthoringResponse> =>
  request(`/api/v1/quizzes/${quizId}/osce-stations/${stationId}`, "PATCH", input, options?.signal)

export const deleteOsceStation = (
  quizId: number,
  stationId: number,
  expectedVersion?: number,
  options?: { signal?: AbortSignal }
): Promise<{ status: "deleted" }> => {
  const suffix = expectedVersion === undefined ? "" : `?expected_version=${expectedVersion}`
  return request(`/api/v1/quizzes/${quizId}/osce-stations/${stationId}${suffix}`, "DELETE", undefined, options?.signal)
}

export const startOsceAttempt = (
  stationId: number,
  clientAttemptId: string,
  options?: { signal?: AbortSignal }
): Promise<OsceAttempt> => request(
  `/api/v1/quizzes/osce-stations/${stationId}/attempts`,
  "POST",
  { client_attempt_id: clientAttemptId },
  options?.signal
)

export const listOsceAttempts = (
  filters: OsceAttemptFilters = {},
  options?: { signal?: AbortSignal }
): Promise<OsceOffsetPage<OsceAttemptSummary>> => {
  const query = new URLSearchParams()
  appendNumber(query, "quiz_id", filters.quiz_id)
  appendNumber(query, "station_id", filters.station_id)
  filters.states?.forEach((state) => query.append("state", state))
  appendNumber(query, "limit", filters.limit)
  appendNumber(query, "offset", filters.offset)
  const suffix = query.size > 0 ? `?${query.toString()}` : ""
  return request(`/api/v1/quizzes/osce-attempts${suffix}`, "GET", undefined, options?.signal)
}

export const getOsceAttempt = (attemptId: number, options?: { signal?: AbortSignal }): Promise<OsceAttempt> =>
  request(`/api/v1/quizzes/osce-attempts/${attemptId}`, "GET", undefined, options?.signal)

export const patchOsceAttempt = (
  attemptId: number,
  input: OsceAttemptPatch,
  options?: { signal?: AbortSignal }
): Promise<OsceAttempt> =>
  request(`/api/v1/quizzes/osce-attempts/${attemptId}`, "PATCH", input, options?.signal)

export const beginOsceSelfAssessment = (
  attemptId: number,
  expectedVersion: number,
  options?: { signal?: AbortSignal }
): Promise<OsceAttempt> => request(
  `/api/v1/quizzes/osce-attempts/${attemptId}/begin-self-assessment`,
  "POST",
  { expected_version: expectedVersion },
  options?.signal
)

export const completeOsceAttempt = (
  attemptId: number,
  expectedVersion: number,
  options?: { signal?: AbortSignal }
): Promise<OsceAttempt> => request(
  `/api/v1/quizzes/osce-attempts/${attemptId}/complete`,
  "POST",
  { expected_version: expectedVersion },
  options?.signal
)
