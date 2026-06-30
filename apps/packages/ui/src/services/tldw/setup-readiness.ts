import { bgRequest } from "@/services/background-proxy"
import { toAllowedPath } from "@/services/tldw/path-utils"
import type { AllowedPath } from "@/services/tldw/openapi-guard"

export type SetupReadinessClientMode = "first-run" | "admin"

export type SetupReadinessAccess = {
  mode?: "first_run" | "admin" | string
  needs_setup?: boolean
  setup_completed?: boolean
  remote_access_active?: boolean
}

export type SetupReadinessLaneStatus =
  | "not_configured"
  | "previewed"
  | "provisioning"
  | "ready"
  | "ready_with_warnings"
  | "failed"
  | "blocked"
  | "skipped"
  | string

export type SetupReadinessLane = {
  lane_id: string
  label?: string
  status?: SetupReadinessLaneStatus
  primary_capability?: string
  secondary_capabilities?: string[]
  selection?: Record<string, unknown>
  warnings?: string[]
  blockers?: string[]
  consequences?: string[]
}

export type SetupReadinessProfile = {
  profile_id: string
  label: string
  description?: string
  lanes?: Record<string, Record<string, unknown>>
  advanced?: boolean
}

export type SetupReadinessProfilesResponse = {
  setup_access?: SetupReadinessAccess
  machine_profile?: Record<string, unknown>
  lane_ids?: string[]
  supported_statuses?: string[]
  supported_overlays?: string[]
  active_overlays?: string[]
  overlays?: string[]
  lanes?: SetupReadinessLane[]
  profiles?: SetupReadinessProfile[]
  recommended_profile_id?: string
}

export type SetupReadinessStatusResponse = SetupReadinessProfilesResponse & {
  readiness_status?: string
  operation_id?: string | null
  operation_status?: string | null
  started_at?: string | null
  completed_at?: string | null
  errors?: unknown[]
}

export type SetupReadinessSelection = {
  profile_id?: string | null
  lanes?: Record<string, Record<string, unknown>>
}

export type SetupReadinessPreviewRequest = SetupReadinessSelection

export type SetupReadinessPreviewResponse = {
  preview_id?: string | null
  profile_id?: string | null
  lane_ids?: string[]
  lanes?: Record<string, Record<string, unknown>>
  overlays?: string[]
  config_updates?: Record<string, Record<string, unknown>>
  secret_fields?: Array<{
    section?: string
    key?: string
    provider?: string | null
    state?: string
  }>
  install_plan?: Record<string, unknown>
  operation_required?: boolean
}

export type SetupReadinessProvisionRequest = {
  preview_id?: string | null
  selection?: SetupReadinessSelection | null
  confirmed?: boolean
}

export type SetupReadinessProvisionResponse = {
  operation_id: string
  operation_status: string
  status_url: string
  status: string
  lanes?: SetupReadinessLane[]
  overlays?: string[]
  install_plan_submitted?: boolean
  config_updates_applied?: boolean
  backup_path?: string | null
}

export type SetupReadinessVerifyRequest = {
  preview_id?: string | null
  selection?: SetupReadinessSelection | null
}

export type SetupReadinessVerifyResponse = {
  profile_id?: string | null
  lane_ids?: string[]
  lanes?: Record<string, Record<string, unknown>>
  overlays?: string[]
  status?: string
  verified_at?: string
}

export type SetupReadinessResponseEnvelope = {
  ok?: boolean
  status?: number
  data?: unknown
  error?: string
}

export type SetupReadinessRequestError = Error & {
  status?: number
  detail?: string
  data?: unknown
}

type RequestJsonInit = {
  method?: string
  headers?: Record<string, string>
  body?: string
  timeoutMs?: number
  signal?: AbortSignal
  responseType?: "json" | "text" | "arrayBuffer"
}

const FIRST_RUN_PATHS = {
  profiles: toAllowedPath("/api/v1/setup/readiness/profiles"),
  status: toAllowedPath("/api/v1/setup/readiness/status"),
  preview: toAllowedPath("/api/v1/setup/readiness/preview"),
  provision: toAllowedPath("/api/v1/setup/readiness/provision"),
  verify: toAllowedPath("/api/v1/setup/readiness/verify")
} satisfies Record<string, AllowedPath>

const ADMIN_PATHS = {
  profiles: toAllowedPath("/api/v1/setup/admin/readiness/profiles"),
  status: toAllowedPath("/api/v1/setup/admin/readiness/status"),
  preview: toAllowedPath("/api/v1/setup/admin/readiness/preview"),
  provision: toAllowedPath("/api/v1/setup/admin/readiness/provision"),
  verify: toAllowedPath("/api/v1/setup/admin/readiness/verify")
} satisfies Record<string, AllowedPath>

export const setupReadinessPathsForMode = (mode: SetupReadinessClientMode = "first-run") =>
  mode === "admin" ? ADMIN_PATHS : FIRST_RUN_PATHS

const extractDetail = (value: unknown): string => {
  if (typeof value === "string") return value
  if (Array.isArray(value)) {
    return value.map(extractDetail).filter(Boolean).join("; ")
  }
  if (!value || typeof value !== "object") return ""

  const record = value as Record<string, unknown>
  for (const key of ["detail", "error", "message"]) {
    const detail = extractDetail(record[key])
    if (detail) return detail
  }
  return ""
}

const toRequestError = async (
  response: SetupReadinessResponseEnvelope
): Promise<SetupReadinessRequestError> => {
  const detail = extractDetail(response?.data) || response?.error || ""
  const status = response?.status ?? 500
  const suffix = detail ? ` ${detail}` : ""
  const error = new Error(`Request failed: ${status}${suffix}`) as SetupReadinessRequestError
  error.status = status
  error.detail = detail
  error.data = response?.data
  return error
}

export const requestSetupReadinessJson = async <T,>(
  path: AllowedPath,
  init?: RequestJsonInit
): Promise<T> => {
  const response = await bgRequest<SetupReadinessResponseEnvelope>({
    path,
    method: (init?.method || "GET") as any,
    headers: init?.headers,
    body: init?.body,
    timeoutMs: init?.timeoutMs,
    abortSignal: init?.signal,
    responseType: init?.responseType,
    returnResponse: true
  })
  if (!response?.ok) {
    throw await toRequestError(response)
  }
  return response.data as T
}

const jsonPostInit = (body: unknown): RequestJsonInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body)
})

export const getSetupReadinessProfiles = (
  options: { mode?: SetupReadinessClientMode } = {}
) =>
  requestSetupReadinessJson<SetupReadinessProfilesResponse>(
    setupReadinessPathsForMode(options.mode).profiles
  )

export const getSetupReadinessStatus = (
  options: { mode?: SetupReadinessClientMode } = {}
) =>
  requestSetupReadinessJson<SetupReadinessStatusResponse>(
    setupReadinessPathsForMode(options.mode).status
  )

export const previewSetupReadiness = (
  request: SetupReadinessPreviewRequest,
  options: { mode?: SetupReadinessClientMode } = {}
) =>
  requestSetupReadinessJson<SetupReadinessPreviewResponse>(
    setupReadinessPathsForMode(options.mode).preview,
    jsonPostInit(request)
  )

export const provisionSetupReadiness = (
  request: SetupReadinessProvisionRequest,
  options: { mode?: SetupReadinessClientMode } = {}
) =>
  requestSetupReadinessJson<SetupReadinessProvisionResponse>(
    setupReadinessPathsForMode(options.mode).provision,
    jsonPostInit(request)
  )

export const verifySetupReadiness = (
  request: SetupReadinessVerifyRequest,
  options: { mode?: SetupReadinessClientMode } = {}
) =>
  requestSetupReadinessJson<SetupReadinessVerifyResponse>(
    setupReadinessPathsForMode(options.mode).verify,
    jsonPostInit(request)
  )
