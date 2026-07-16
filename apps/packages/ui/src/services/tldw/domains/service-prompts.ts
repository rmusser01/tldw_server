import { bgRequest } from "@/services/background-proxy"
import type { AllowedPath } from "@/services/tldw/openapi-guard"

export type KnownServicePromptId =
  | "chat.rag.answer"
  | "chat.rag.question_rewrite"
  | "chat.web_search.answer"
  | "media.text.translation"

export type ServicePromptSource = "user" | "packaged"
export type ServicePromptPartMode = "literal" | "template"

export type ServicePromptPart = {
  key: string
  label: string
  mode: ServicePromptPartMode
  required_variables: string[]
}

export type ServicePromptWorkflow = {
  id: string
  label: string
}

export type ServicePromptCatalogItem = {
  id: string
  label: string
  description: string
  parts: ServicePromptPart[]
  affected_workflows: ServicePromptWorkflow[]
}

export type ServicePromptDetail = ServicePromptCatalogItem & {
  default_parts: Record<string, string>
  saved_parts: Record<string, string> | null
  effective_parts: Record<string, string>
  source: ServicePromptSource
  revision: string | null
}

export type ServicePromptUpdateRequest = {
  parts: Record<string, string>
  expected_revision: string | null
}

type ServicePromptErrorDetail = {
  code?: unknown
  message?: unknown
  field_errors?: unknown
  current_revision?: unknown
  revision?: unknown
  can_reset?: unknown
}

const asRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null

const asOptionalString = (value: unknown): string | undefined =>
  typeof value === "string" ? value : undefined

const asOptionalRevision = (value: unknown): string | null | undefined =>
  value === null ? null : asOptionalString(value)

const asFieldErrors = (value: unknown): Record<string, string> | undefined => {
  const record = asRecord(value)
  if (!record) return undefined
  const entries = Object.entries(record)
  return entries.every(([, message]) => typeof message === "string")
    ? Object.fromEntries(entries) as Record<string, string>
    : undefined
}

export class ServicePromptApiError extends Error {
  status: number
  code?: string
  fieldErrors?: Record<string, string>
  currentRevision?: string | null
  revision?: string
  canReset?: boolean

  constructor(message: string, options: {
    status: number
    code?: string
    fieldErrors?: Record<string, string>
    currentRevision?: string | null
    revision?: string
    canReset?: boolean
  }) {
    super(message)
    this.name = "ServicePromptApiError"
    this.status = options.status
    this.code = options.code
    this.fieldErrors = options.fieldErrors
    this.currentRevision = options.currentRevision
    this.revision = options.revision
    this.canReset = options.canReset
  }
}

const normalizeServicePromptError = (error: unknown): ServicePromptApiError => {
  const candidate = error as {
    status?: unknown
    message?: unknown
    details?: { detail?: unknown }
    detail?: unknown
  } | null
  const detail = asRecord(
    candidate?.details?.detail ?? candidate?.detail
  ) as ServicePromptErrorDetail | null
  const status = typeof candidate?.status === "number" ? candidate.status : 0
  const message = asOptionalString(detail?.message)
    ?? (error instanceof Error ? error.message : "Service Prompt request failed.")

  return new ServicePromptApiError(message, {
    status,
    code: asOptionalString(detail?.code),
    fieldErrors: asFieldErrors(detail?.field_errors),
    currentRevision: asOptionalRevision(detail?.current_revision),
    revision: asOptionalString(detail?.revision),
    canReset: typeof detail?.can_reset === "boolean" ? detail.can_reset : undefined
  })
}

const request = async <T>(init: Parameters<typeof bgRequest>[0]): Promise<T> => {
  try {
    return await bgRequest<T>(init as never)
  } catch (error) {
    if ((error as { name?: unknown } | null)?.name === "AbortError") {
      throw error
    }
    throw normalizeServicePromptError(error)
  }
}

const detailPath = (id: string): AllowedPath =>
  `/api/v1/service-prompts/${encodeURIComponent(id)}`

export const servicePromptMethods = {
  async listServicePrompts(
    options?: { signal?: AbortSignal }
  ): Promise<ServicePromptCatalogItem[]> {
    return await request<ServicePromptCatalogItem[]>({
      path: "/api/v1/service-prompts",
      method: "GET",
      expectedStatuses: [404],
      abortSignal: options?.signal
    })
  },

  async getServicePrompt(
    id: string,
    options?: { signal?: AbortSignal }
  ): Promise<ServicePromptDetail> {
    return await request<ServicePromptDetail>({
      path: detailPath(id),
      method: "GET",
      expectedStatuses: [404, 500],
      abortSignal: options?.signal
    })
  },

  async saveServicePrompt(
    id: string,
    payload: ServicePromptUpdateRequest,
    options?: { signal?: AbortSignal }
  ): Promise<ServicePromptDetail> {
    return await request<ServicePromptDetail>({
      path: detailPath(id),
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: payload,
      expectedStatuses: [404, 409, 422, 500],
      abortSignal: options?.signal
    })
  },

  async resetServicePrompt(
    id: string,
    expectedRevision: string | null,
    options?: { signal?: AbortSignal }
  ): Promise<ServicePromptDetail> {
    const path = detailPath(id)
    return await request<ServicePromptDetail>({
      path: expectedRevision === null
        ? path
        : `${path}?expected_revision=${encodeURIComponent(expectedRevision)}`,
      method: "DELETE",
      expectedStatuses: [404, 409, 422, 500],
      abortSignal: options?.signal
    })
  }
}

export type ServicePromptMethods = typeof servicePromptMethods
