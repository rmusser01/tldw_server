import { bgRequest } from "@/services/background-proxy"
import type { AllowedPath } from "@/services/tldw/openapi-guard"
import type {
  ServicePromptTargetConfig
} from "@/services/tldw/TldwApiClient"

export type KnownServicePromptId =
  | "chat.rag.answer"
  | "chat.rag.question_rewrite"
  | "chat.web_search.answer"
  | "chat.title.generation"
  | "image.prompt.refinement"
  | "media.document.summarization"
  | "media.pdf.summarization"
  | "media.text.translation"
  | "notes.title.generate"

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

export type ServicePromptRequestScope = Readonly<{
  config: ServicePromptTargetConfig
  userId: string | number | null
}>

type ServicePromptRequestOptions = {
  signal?: AbortSignal
  requestScope?: ServicePromptRequestScope
}

type ServicePromptErrorDetail = {
  code?: unknown
  message?: unknown
  field_errors?: unknown
  current_revision?: unknown
  revision?: unknown
  can_reset?: unknown
}

export type ServicePromptRequestError = Readonly<{
  type: string
  loc: readonly (string | number)[]
  msg: string
}>

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
  return entries.length > 0 && entries.every(
    ([, message]) => typeof message === "string" && message.trim().length > 0
  )
    ? Object.fromEntries(entries) as Record<string, string>
    : undefined
}

const asRequestErrors = (
  value: unknown
): ServicePromptRequestError[] | undefined => {
  if (!Array.isArray(value) || value.length === 0) return undefined
  const errors: ServicePromptRequestError[] = []
  for (const item of value) {
    const record = asRecord(item)
    const type = asOptionalString(record?.type)?.trim()
    const msg = asOptionalString(record?.msg)?.trim()
    const loc = record?.loc
    if (!type || !msg || !Array.isArray(loc) || loc.length === 0 ||
      !loc.every((part) => typeof part === "string" || typeof part === "number")
    ) {
      return undefined
    }
    errors.push(Object.freeze({ type, loc: Object.freeze([...loc]), msg }))
  }
  return errors
}

const isStringArray = (value: unknown): value is string[] =>
  Array.isArray(value) && value.every((item) => typeof item === "string")

const isStringRecord = (value: unknown): value is Record<string, string> => {
  const record = asRecord(value)
  return record !== null && Object.values(record).every(
    (item) => typeof item === "string"
  )
}

const hasExactKeys = (
  record: Record<string, string>,
  keys: readonly string[]
): boolean => Object.keys(record).length === keys.length &&
  keys.every((key) => Object.prototype.hasOwnProperty.call(record, key))

const recordsEqual = (
  left: Record<string, string>,
  right: Record<string, string>
): boolean => {
  const keys = Object.keys(left)
  return hasExactKeys(right, keys) && keys.every((key) => left[key] === right[key])
}

const UUID_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i

const isServicePromptPart = (value: unknown): value is ServicePromptPart => {
  const record = asRecord(value)
  return record !== null &&
    typeof record.key === "string" &&
    typeof record.label === "string" &&
    (record.mode === "literal" || record.mode === "template") &&
    isStringArray(record.required_variables)
}

const isServicePromptWorkflow = (
  value: unknown
): value is ServicePromptWorkflow => {
  const record = asRecord(value)
  return record !== null &&
    typeof record.id === "string" &&
    typeof record.label === "string"
}

const isServicePromptCatalogItem = (
  value: unknown
): value is ServicePromptCatalogItem => {
  const record = asRecord(value)
  if (record === null ||
    typeof record.id !== "string" ||
    typeof record.label !== "string" ||
    typeof record.description !== "string" ||
    !Array.isArray(record.parts) ||
    !record.parts.every(isServicePromptPart) ||
    !Array.isArray(record.affected_workflows) ||
    !record.affected_workflows.every(isServicePromptWorkflow)
  ) {
    return false
  }
  const partKeys = record.parts.map((part) => part.key)
  return partKeys.every((key) => key.length > 0) &&
    new Set(partKeys).size === partKeys.length
}

const isServicePromptDetail = (
  value: unknown,
  expectedId: string
): value is ServicePromptDetail => {
  if (!isServicePromptCatalogItem(value) || value.id !== expectedId) {
    return false
  }
  const record = value as unknown as Record<string, unknown>
  const defaultParts = record.default_parts
  const savedParts = record.saved_parts
  const effectiveParts = record.effective_parts
  if (!isStringRecord(defaultParts) || !isStringRecord(effectiveParts)) {
    return false
  }
  if (savedParts !== null && !isStringRecord(savedParts)) {
    return false
  }
  const validatedSavedParts = savedParts as Record<string, string> | null
  const partKeys = value.parts.map((part) => part.key)
  if (!hasExactKeys(defaultParts, partKeys) ||
    !hasExactKeys(effectiveParts, partKeys) ||
    (validatedSavedParts !== null &&
      !hasExactKeys(validatedSavedParts, partKeys))
  ) {
    return false
  }
  if (record.source === "packaged") {
    return validatedSavedParts === null &&
      record.revision === null &&
      recordsEqual(effectiveParts, defaultParts)
  }
  return record.source === "user" &&
    validatedSavedParts !== null &&
    typeof record.revision === "string" &&
    UUID_PATTERN.test(record.revision) &&
    recordsEqual(effectiveParts, validatedSavedParts)
}

export class ServicePromptApiError extends Error {
  status: number
  code?: string
  fieldErrors?: Record<string, string>
  currentRevision?: string | null
  revision?: string
  canReset?: boolean
  requestErrors?: ServicePromptRequestError[]

  constructor(message: string, options: {
    status: number
    code?: string
    fieldErrors?: Record<string, string>
    currentRevision?: string | null
    revision?: string
    canReset?: boolean
    requestErrors?: ServicePromptRequestError[]
  }) {
    super(message)
    this.name = "ServicePromptApiError"
    this.status = options.status
    this.code = options.code
    this.fieldErrors = options.fieldErrors
    this.currentRevision = options.currentRevision
    this.revision = options.revision
    this.canReset = options.canReset
    this.requestErrors = options.requestErrors
  }
}

const invalidProtocolResponse = (): ServicePromptApiError =>
  new ServicePromptApiError(
    "Service Prompt server response was invalid.",
    { status: 0, code: "service_prompt_protocol_error" }
  )

const normalizeServicePromptError = (error: unknown): ServicePromptApiError => {
  const candidate = error as {
    status?: unknown
    message?: unknown
    details?: { detail?: unknown }
    detail?: unknown
  } | null
  const rawDetail = candidate?.details?.detail ?? candidate?.detail
  const detail = asRecord(rawDetail) as ServicePromptErrorDetail | null
  const status = typeof candidate?.status === "number" ? candidate.status : 0
  const code = asOptionalString(detail?.code)
  const message = asOptionalString(detail?.message)
    ?? (error instanceof Error ? error.message : "Service Prompt request failed.")

  return new ServicePromptApiError(message, {
    status,
    code,
    fieldErrors: code === "service_prompt_validation_failed"
      ? asFieldErrors(detail?.field_errors)
      : undefined,
    currentRevision: asOptionalRevision(detail?.current_revision),
    revision: asOptionalString(detail?.revision),
    canReset: typeof detail?.can_reset === "boolean" ? detail.can_reset : undefined,
    requestErrors: status === 422 ? asRequestErrors(rawDetail) : undefined
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

export const requestScopeFields = (
  requestScope?: ServicePromptRequestScope
): {
  servicePromptConfig?: ServicePromptTargetConfig
  headers?: Record<string, string>
} => requestScope
  ? {
      servicePromptConfig: {
        ...requestScope.config,
        expectedUserId: requestScope.userId
      },
      ...(requestScope.userId === null
        ? {}
        : {
            headers: {
              "X-TLDW-Expected-User-ID": String(requestScope.userId)
            }
          })
    }
  : {}

export const servicePromptMethods = {
  async listServicePrompts(
    options?: ServicePromptRequestOptions
  ): Promise<ServicePromptCatalogItem[]> {
    const scopeFields = requestScopeFields(options?.requestScope)
    const response = await request<unknown>({
      path: "/api/v1/service-prompts",
      method: "GET",
      expectedStatuses: [404, 412],
      abortSignal: options?.signal,
      ...scopeFields
    })
    if (!Array.isArray(response) || !response.every(isServicePromptCatalogItem)) {
      throw invalidProtocolResponse()
    }
    return response
  },

  async getServicePrompt(
    id: string,
    options?: ServicePromptRequestOptions
  ): Promise<ServicePromptDetail> {
    const response = await request<unknown>({
      path: detailPath(id),
      method: "GET",
      expectedStatuses: [404, 412, 500],
      abortSignal: options?.signal,
      ...requestScopeFields(options?.requestScope)
    })
    if (!isServicePromptDetail(response, id)) {
      throw invalidProtocolResponse()
    }
    return response
  },

  async saveServicePrompt(
    id: string,
    payload: ServicePromptUpdateRequest,
    options?: ServicePromptRequestOptions
  ): Promise<ServicePromptDetail> {
    const scopeFields = requestScopeFields(options?.requestScope)
    const response = await request<unknown>({
      path: detailPath(id),
      method: "PUT",
      headers: {
        "Content-Type": "application/json",
        ...scopeFields.headers
      },
      body: payload,
      expectedStatuses: [404, 409, 412, 422, 500],
      abortSignal: options?.signal,
      ...(scopeFields.servicePromptConfig
        ? { servicePromptConfig: scopeFields.servicePromptConfig }
        : {})
    })
    if (!isServicePromptDetail(response, id) ||
      response.source !== "user" ||
      response.saved_parts === null ||
      !recordsEqual(response.saved_parts, payload.parts) ||
      !recordsEqual(response.effective_parts, payload.parts)
    ) {
      throw invalidProtocolResponse()
    }
    return response
  },

  async resetServicePrompt(
    id: string,
    expectedRevision: string | null,
    options?: ServicePromptRequestOptions
  ): Promise<ServicePromptDetail> {
    const path = detailPath(id)
    const response = await request<unknown>({
      path: expectedRevision === null
        ? path
        : `${path}?expected_revision=${encodeURIComponent(expectedRevision)}`,
      method: "DELETE",
      expectedStatuses: [404, 409, 412, 422, 500],
      abortSignal: options?.signal,
      ...requestScopeFields(options?.requestScope)
    })
    if (!isServicePromptDetail(response, id) || response.source !== "packaged") {
      throw invalidProtocolResponse()
    }
    return response
  }
}

export type ServicePromptMethods = typeof servicePromptMethods
