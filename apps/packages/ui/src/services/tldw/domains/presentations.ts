import { bgRequest } from "@/services/background-proxy"
import { buildQuery, toTrimmedStringArray } from "../client-utils"
import type {
  PresentationStudioSlide,
  PresentationVisualStyleSnapshot,
  VisualStyleRecord,
  VisualStyleCreateInput,
  VisualStylePatchInput,
  PresentationStudioRecord,
  StructuredPresentationStudioRecord,
  StandaloneHtmlPresentationStudioRecord,
  UnsupportedPresentationStudioRecord,
  PresentationDetailResult,
  StandalonePresentationDetailResult,
  PresentationListResponse,
  PresentationSummary,
  PresentationMetadataResult,
  SlidesCapabilities,
  PresentationGenerationRequest,
  PresentationGenerationReceipt,
  PresentationGenerationStatusResult,
  PresentationRenderJob,
  PresentationRenderFormat,
  PresentationRenderArtifactList,
} from "../TldwApiClient"
import {
  clonePresentationVisualStyleSnapshot,
} from "../presentation-style"

const toOptionalString = (value: unknown): string | null => {
  if (typeof value !== "string") return null
  const trimmed = value.trim()
  return trimmed.length > 0 ? trimmed : null
}

const toRecord = (value: unknown): Record<string, unknown> => {
  if (value && typeof value === "object" && !Array.isArray(value)) {
    return value as Record<string, unknown>
  }
  return {}
}

const toOptionalNumber = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isFinite(value)) return value
  if (typeof value === "string") {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return null
}

const toFiniteNumber = (value: unknown, fallback = 0): number => {
  if (typeof value === "number" && Number.isFinite(value)) return value
  if (typeof value === "string") {
    const parsed = Number(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return fallback
}

const extractOffsetPaginationTotal = (value: unknown): number | null => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null
  }
  const record = value as Record<string, unknown>
  if (typeof record.total_count === "number" && Number.isFinite(record.total_count)) {
    return record.total_count
  }
  const pagination = record.pagination
  if (!pagination || typeof pagination !== "object" || Array.isArray(pagination)) {
    return null
  }
  const total = (pagination as Record<string, unknown>).total
  return typeof total === "number" && Number.isFinite(total) ? total : null
}

const normalizeVisualStyleSnapshot = (
  value: unknown
): PresentationVisualStyleSnapshot | null => {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return null
  }
  const snapshot = value as Record<string, unknown>
  const id = String(snapshot.id ?? "").trim()
  const scope = String(snapshot.scope ?? "").trim()
  const name = String(snapshot.name ?? "").trim()
  if (!id || !scope || !name) {
    return null
  }
  return clonePresentationVisualStyleSnapshot({
    id,
    scope,
    name,
    description: toOptionalString(snapshot.description),
    category: toOptionalString(snapshot.category),
    guide_number: toOptionalNumber(snapshot.guide_number),
    tags: toTrimmedStringArray(snapshot.tags),
    best_for: toTrimmedStringArray(snapshot.best_for),
    generation_rules: toRecord(snapshot.generation_rules),
    artifact_preferences: toTrimmedStringArray(snapshot.artifact_preferences),
    appearance_defaults: toRecord(snapshot.appearance_defaults),
    fallback_policy: toRecord(snapshot.fallback_policy),
    version: toOptionalNumber(snapshot.version)
  })
}

const normalizeVisualStyleRecord = (style: unknown): VisualStyleRecord => {
  const record = style && typeof style === "object" && !Array.isArray(style)
    ? (style as Record<string, unknown>)
    : {}
  return {
    id: String(record.id ?? ""),
    name: String(record.name ?? ""),
    scope: String(record.scope ?? ""),
    description: toOptionalString(record.description),
    category: toOptionalString(record.category),
    guide_number: toOptionalNumber(record.guide_number),
    tags: toTrimmedStringArray(record.tags),
    best_for: toTrimmedStringArray(record.best_for),
    generation_rules: toRecord(record.generation_rules),
    artifact_preferences: toTrimmedStringArray(record.artifact_preferences),
    appearance_defaults: toRecord(record.appearance_defaults),
    fallback_policy: toRecord(record.fallback_policy),
    version: toOptionalNumber(record.version),
    created_at: toOptionalString(record.created_at),
    updated_at: toOptionalString(record.updated_at)
  }
}

const ACCEPT_CONTENT_KINDS_HEADER = "X-Slides-Accept-Content-Kinds"
const ACCEPT_CONTENT_KINDS_VALUE = "structured_slides,standalone_html"
const MAX_STANDALONE_HTML_BYTES = 1_048_576
const HTML_ATTACHMENT_CONTENT_TYPE = "application/octet-stream"
const HTML_ATTACHMENT_DISPOSITION = 'attachment; filename="presentation.html"'
const MAX_GENERATION_RETRY_AFTER_MS = 60_000
const SAFE_ERROR_CODE = /^[a-z0-9_]{1,100}$/

const presentationNegotiationHeaders = (): Record<string, string> => ({
  [ACCEPT_CONTENT_KINDS_HEADER]: ACCEPT_CONTENT_KINDS_VALUE
})

const isRequiredString = (value: unknown): value is string =>
  typeof value === "string" && value.length > 0

const isNonBlankString = (value: unknown): value is string =>
  typeof value === "string" && value.trim().length > 0

const isPositiveInteger = (value: unknown): value is number =>
  typeof value === "number" && Number.isInteger(value) && value > 0

const isRecordValue = (value: unknown): value is Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value)

const normalizePresentationBase = (record: Record<string, unknown>) => ({
  id: String(record.id ?? ""),
  title: String(record.title ?? ""),
  description: toOptionalString(record.description),
  theme: String(record.theme ?? "black"),
  source_type: toOptionalString(record.source_type),
  source_ref: record.source_ref ?? null,
  source_query: toOptionalString(record.source_query),
  created_at: String(record.created_at ?? ""),
  last_modified: String(record.last_modified ?? ""),
  deleted: Boolean(record.deleted),
  client_id: toOptionalString(record.client_id) ?? undefined,
  version: toFiniteNumber(record.version, 0)
})

const normalizeStructuredPresentation = (
  record: Record<string, unknown>
): StructuredPresentationStudioRecord => ({
  ...normalizePresentationBase(record),
  content_kind: "structured_slides",
  marp_theme: toOptionalString(record.marp_theme),
  template_id: toOptionalString(record.template_id),
  visual_style_id: toOptionalString(record.visual_style_id),
  visual_style_scope: toOptionalString(record.visual_style_scope),
  visual_style_name: toOptionalString(record.visual_style_name),
  visual_style_version: toOptionalNumber(record.visual_style_version),
  visual_style_snapshot: normalizeVisualStyleSnapshot(record.visual_style_snapshot),
  settings: Object.keys(toRecord(record.settings)).length > 0 ? toRecord(record.settings) : null,
  studio_data:
    Object.keys(toRecord(record.studio_data)).length > 0 ? toRecord(record.studio_data) : null,
  slides: Array.isArray(record.slides) ? (record.slides as PresentationStudioSlide[]) : [],
  custom_css: toOptionalString(record.custom_css)
})

const normalizeStandalonePresentation = (
  record: Record<string, unknown>
): StandaloneHtmlPresentationStudioRecord => ({
  ...normalizePresentationBase(record),
  content_kind: "standalone_html",
  html_document: record.html_document as string,
  html_sha256: record.html_sha256 as string,
  html_bytes: record.html_bytes as number,
  html_slide_count: record.html_slide_count as number,
  generation_provenance: record.generation_provenance as Record<string, unknown>
})

const validateStandalonePresentation = (record: Record<string, unknown>): void => {
  const valid =
    isRequiredString(record.id) &&
    isRequiredString(record.title) &&
    isRequiredString(record.theme) &&
    isRequiredString(record.created_at) &&
    isRequiredString(record.last_modified) &&
    typeof record.deleted === "boolean" &&
    isRequiredString(record.client_id) &&
    isPositiveInteger(record.version) &&
    typeof record.html_document === "string" &&
    typeof record.html_sha256 === "string" &&
    /^[0-9a-f]{64}$/.test(record.html_sha256) &&
    isPositiveInteger(record.html_bytes) &&
    isPositiveInteger(record.html_slide_count) &&
    isRecordValue(record.generation_provenance)

  if (!valid) {
    throw new Error("Invalid presentation detail response")
  }
}

const normalizeUnsupportedPresentation = (
  record: Record<string, unknown>,
  unsupportedContentKind: string | null
): UnsupportedPresentationStudioRecord => ({
  ...normalizePresentationBase(record),
  content_kind: "unsupported",
  unsupported_content_kind: unsupportedContentKind,
  read_only: true
})

/** The single interpretation seam for presentation detail wire records. */
export const normalizePresentationStudioRecord = (
  presentation: unknown
): PresentationStudioRecord => {
  const record =
    presentation && typeof presentation === "object" && !Array.isArray(presentation)
      ? (presentation as Record<string, unknown>)
      : {}

  if (record.content_kind === "structured_slides") {
    if (!Array.isArray(record.slides)) {
      throw new Error("Invalid presentation detail response")
    }
    return normalizeStructuredPresentation(record)
  }
  if (record.content_kind === "standalone_html") {
    validateStandalonePresentation(record)
    return normalizeStandalonePresentation(record)
  }
  if (record.content_kind == null && Array.isArray(record.slides)) {
    return normalizeStructuredPresentation(record)
  }

  return normalizeUnsupportedPresentation(
    record,
    typeof record.content_kind === "string" ? record.content_kind : null
  )
}

const normalizeProvenanceSummary = (value: unknown) => {
  const record = toRecord(value)
  return {
    source_kind: toOptionalString(record.source_kind),
    provider: toOptionalString(record.provider),
    model: toOptionalString(record.model)
  }
}

const normalizePresentationSummary = (value: unknown): PresentationSummary => {
  const record = toRecord(value)
  const base = {
    id: String(record.id ?? ""),
    title: String(record.title ?? ""),
    description: toOptionalString(record.description),
    theme: String(record.theme ?? "black"),
    created_at: String(record.created_at ?? ""),
    last_modified: String(record.last_modified ?? ""),
    deleted: Boolean(record.deleted),
    version: toFiniteNumber(record.version, 0),
    provenance: normalizeProvenanceSummary(record.provenance)
  }
  if (record.content_kind === "structured_slides") {
    return {
      ...base,
      content_kind: "structured_slides",
      slide_count: toFiniteNumber(record.slide_count, 0)
    }
  }
  if (record.content_kind === "standalone_html") {
    return {
      ...base,
      content_kind: "standalone_html",
      html_slide_count: toFiniteNumber(record.html_slide_count, 0),
      html_bytes: toFiniteNumber(record.html_bytes, 0)
    }
  }
  return {
    ...base,
    content_kind: "unsupported",
    unsupported_content_kind: typeof record.content_kind === "string" ? record.content_kind : null,
    read_only: true
  }
}

const responseHeaders = (response: unknown): Headers =>
  new Headers(toRecord(toRecord(response).headers) as Record<string, string>)

const requireSuccessfulResponseData = (response: unknown): unknown => {
  const record = toRecord(response)
  if (
    record.ok !== true ||
    typeof record.status !== "number" ||
    record.status < 200 ||
    record.status >= 300
  ) {
    const data = toRecord(record.data)
    const detail = toRecord(data.detail)
    const candidate = [detail.error_code, detail.code, data.error_code, data.code]
      .find((value) => typeof value === "string" && SAFE_ERROR_CODE.test(value))
    const error = new Error("Invalid presentation response") as Error & {
      status?: number
      details?: { error_code: string }
      retryAfterMs?: number | null
    }
    if (typeof record.status === "number" && Number.isFinite(record.status)) {
      error.status = record.status
    }
    error.details = {
      error_code: typeof candidate === "string" ? candidate : "presentation_request_failed"
    }
    error.retryAfterMs =
      typeof record.retryAfterMs === "number" && Number.isFinite(record.retryAfterMs) && record.retryAfterMs >= 0
        ? Math.min(MAX_GENERATION_RETRY_AFTER_MS, Math.floor(record.retryAfterMs))
        : null
    throw error
  }
  return record.data
}

const responseEtag = (response: unknown): string | null => responseHeaders(response).get("etag")

const exactKeys = (
  record: Record<string, unknown>,
  required: readonly string[],
  optional: readonly string[] = []
): boolean => {
  const keys = Object.keys(record)
  const allowed = new Set([...required, ...optional])
  return (
    required.every((key) => Object.prototype.hasOwnProperty.call(record, key)) &&
    keys.every((key) => allowed.has(key))
  )
}

const hasExactNumericFields = (
  value: unknown,
  fields: readonly string[]
): value is Record<string, number> => {
  const record = toRecord(value)
  return exactKeys(record, fields) && fields.every((field) => isPositiveInteger(record[field]))
}

const GENERATION_REASON_CODES = new Set([
  "feature_disabled",
  "egress_disabled",
  "default_model_not_configured",
  "default_model_not_allowed",
  "default_endpoint_not_allowed",
  "prompt_asset_unavailable",
  "digest_key_unavailable",
  "generation_worker_unavailable",
  "generation_reconciler_overloaded",
  "validator_unavailable"
])
const GENERATION_SOURCE_KINDS = ["prompt", "chat", "media", "notes", "rag"]

const isNullableString = (value: unknown): value is string | null =>
  value === null || typeof value === "string"

const validateSlidesCapabilities = (value: unknown): SlidesCapabilities => {
  const root = toRecord(value)
  const contentKinds = toRecord(root.content_kinds)
  const structuredKind = toRecord(contentKinds.structured_slides)
  const htmlKind = toRecord(contentKinds.standalone_html)
  const htmlLimits = toRecord(htmlKind.limits)
  const generationModes = toRecord(root.generation_modes)
  const structuredGeneration = toRecord(generationModes.structured_slides)
  const htmlGeneration = toRecord(generationModes.standalone_html)
  const inputLimits = toRecord(htmlGeneration.input_limits)
  const outputLimits = toRecord(htmlGeneration.output_limits)
  const sourceKinds = htmlGeneration.source_kinds
  const contentReason = htmlKind.reason
  const generationReason = htmlGeneration.reason
  const validContentVariant =
    (contentReason === null && htmlKind.edit === true && htmlKind.export_attachment === true) ||
    (contentReason === "validator_unavailable" &&
      htmlKind.edit === false &&
      htmlKind.export_attachment === false)
  const generationTargets = [
    htmlGeneration.provider,
    htmlGeneration.model,
    htmlGeneration.adapter_id,
    htmlGeneration.endpoint_identity
  ]
  const validGenerationVariant =
    (htmlGeneration.enabled === true &&
      generationReason === null &&
      generationTargets.every(isNonBlankString) &&
      typeof htmlGeneration.generation_config_revision === "string" &&
      /^sha256:[0-9a-f]{64}$/.test(htmlGeneration.generation_config_revision)) ||
    (htmlGeneration.enabled === false &&
      typeof generationReason === "string" &&
      GENERATION_REASON_CODES.has(generationReason) &&
      generationTargets.every((target) => target === null) &&
      htmlGeneration.generation_config_revision === null)

  const valid =
    exactKeys(root, [
      "schema_version",
      "content_kind_request_header",
      "content_kinds",
      "generation_modes"
    ]) &&
    root.schema_version === 1 &&
    root.content_kind_request_header === ACCEPT_CONTENT_KINDS_HEADER &&
    exactKeys(contentKinds, ["structured_slides", "standalone_html"]) &&
    exactKeys(structuredKind, ["read", "edit"]) &&
    structuredKind.read === true &&
    structuredKind.edit === true &&
    exactKeys(htmlKind, [
      "read",
      "edit",
      "export_attachment",
      "draft_attachment",
      "reason",
      "limits"
    ]) &&
    htmlKind.read === true &&
    typeof htmlKind.edit === "boolean" &&
    typeof htmlKind.export_attachment === "boolean" &&
    htmlKind.draft_attachment === true &&
    validContentVariant &&
    hasExactNumericFields(htmlLimits, [
      "max_document_bytes",
      "max_source_write_bytes",
      "max_draft_attachment_bytes",
      "max_slides",
      "max_nesting_depth"
    ]) &&
    exactKeys(generationModes, ["structured_slides", "standalone_html"]) &&
    exactKeys(structuredGeneration, ["enabled", "transport"]) &&
    structuredGeneration.enabled === true &&
    structuredGeneration.transport === "existing_source_endpoints" &&
    exactKeys(htmlGeneration, [
      "enabled",
      "reason",
      "transport",
      "source_kinds",
      "provider",
      "model",
      "adapter_id",
      "endpoint_identity",
      "generation_config_revision",
      "input_limits",
      "output_limits"
    ]) &&
    validGenerationVariant &&
    htmlGeneration.transport === "slides_generation_job" &&
    Array.isArray(sourceKinds) &&
    sourceKinds.length === GENERATION_SOURCE_KINDS.length &&
    sourceKinds.every((kind, index) => kind === GENERATION_SOURCE_KINDS[index]) &&
    !(contentReason === "validator_unavailable" && htmlGeneration.enabled !== false) &&
    !(generationReason === "validator_unavailable" &&
      contentReason !== "validator_unavailable") &&
    hasExactNumericFields(inputLimits, [
      "max_request_bytes",
      "max_source_chars",
      "max_source_tokens",
      "max_audience_chars",
      "max_source_identifier_bytes",
      "max_note_ids",
      "max_rag_query_chars",
      "max_rag_top_k"
    ]) &&
    hasExactNumericFields(outputLimits, ["max_provider_response_bytes", "max_document_bytes"])

  if (!valid) {
    throw new Error("Invalid Slides capabilities response")
  }
  return value as SlidesCapabilities
}

const validateGenerationReceipt = (value: unknown): PresentationGenerationReceipt => {
  const record = toRecord(value)
  const commonValid = isRequiredString(record.generation_id) && isRequiredString(record.status_url)
  let valid = false
  if (record.status === "queued" || record.status === "running") {
    valid =
      exactKeys(
        record,
        ["generation_id", "status", "status_url", "presentation_id"],
        ["progress_text"]
      ) &&
      record.presentation_id === null &&
      (record.progress_text === undefined ||
        (isNullableString(record.progress_text) && (record.progress_text?.length ?? 0) <= 500))
  } else if (record.status === "completed") {
    valid =
      exactKeys(record, [
        "generation_id",
        "status",
        "status_url",
        "presentation_id",
        "content_kind"
      ]) &&
      (record.presentation_id === null || isRequiredString(record.presentation_id)) &&
      record.content_kind === "standalone_html"
  } else if (record.status === "failed") {
    valid =
      exactKeys(record, [
        "generation_id",
        "status",
        "status_url",
        "presentation_id",
        "error_code",
        "error_message"
      ]) &&
      record.presentation_id === null &&
      typeof record.error_code === "string" && SAFE_ERROR_CODE.test(record.error_code) &&
      isRequiredString(record.error_message) && record.error_message.length <= 1_000
  } else if (record.status === "cancelled") {
    valid =
      exactKeys(record, [
        "generation_id",
        "status",
        "status_url",
        "presentation_id",
        "error_code"
      ]) &&
      record.presentation_id === null &&
      record.error_code === "generation_cancelled"
  }
  if (!commonValid || !valid) {
    throw new Error("Invalid Slides generation response")
  }
  return value as PresentationGenerationReceipt
}

const validateStandaloneSource = (source: string): void => {
  if (source.includes("\u0000")) {
    throw new Error("Standalone HTML source contains U+0000")
  }
  for (let index = 0; index < source.length; index += 1) {
    const code = source.charCodeAt(index)
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = source.charCodeAt(index + 1)
      if (!(next >= 0xdc00 && next <= 0xdfff)) {
        throw new Error("Standalone HTML source must contain valid Unicode scalar values")
      }
      index += 1
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      throw new Error("Standalone HTML source must contain valid Unicode scalar values")
    }
  }
  if (new TextEncoder().encode(source).byteLength > MAX_STANDALONE_HTML_BYTES) {
    throw new Error("Standalone HTML source exceeds 1048576 UTF-8 bytes")
  }
}

const validateIfMatch = (ifMatch: string): void => {
  if (!ifMatch.trim()) {
    throw new Error("Standalone HTML save requires If-Match")
  }
}

const exactAttachmentBytes = (response: unknown): Uint8Array => {
  const record = toRecord(response)
  const headers = responseHeaders(response)
  if (
    record.ok !== true ||
    record.status !== 200 ||
    headers.get("content-type") !== HTML_ATTACHMENT_CONTENT_TYPE ||
    headers.get("content-disposition") !== HTML_ATTACHMENT_DISPOSITION ||
    headers.get("x-content-type-options") !== "nosniff" ||
    headers.get("x-download-options") !== "noopen" ||
    headers.get("cache-control") !== "private, no-store" ||
    headers.get("referrer-policy") !== "no-referrer" ||
    headers.get("cross-origin-resource-policy") !== "same-origin"
  ) {
    throw new Error("Invalid standalone HTML attachment response")
  }
  const data = record.data
  if (data instanceof ArrayBuffer) {
    return new Uint8Array(data)
  }
  if (ArrayBuffer.isView(data)) {
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
  }
  throw new Error("Invalid standalone HTML attachment response")
}

const requireStructuredRecord = (
  record: PresentationStudioRecord
): StructuredPresentationStudioRecord => {
  if (record.content_kind !== "structured_slides") {
    throw new Error("Structured presentation required")
  }
  return record
}

const requireStandaloneRecord = (
  record: PresentationStudioRecord
): StandaloneHtmlPresentationStudioRecord => {
  if (record.content_kind !== "standalone_html") {
    throw new Error("Standalone presentation required")
  }
  return record
}

/**
 * Minimal interface for the TldwApiClient methods referenced via `this`.
 */
export interface TldwApiClientCore {
  ensureConfigForRequest(requireAuth: boolean): Promise<any>
  request<T>(init: any, requireAuth?: boolean): Promise<T>
  resolveApiPath(key: string, candidates: string[]): Promise<string>
  fillPathParams(template: string, values: string | string[]): string
}

export const presentationsMethods = {
  async generateSlidesFromMedia(
    this: TldwApiClientCore,
    mediaId: number,
    options?: {
      titleHint?: string
      theme?: string
      visualStyleId?: string
      visualStyleScope?: string
      provider?: string
      model?: string
      claimsVerificationProvider?: string
      claimsVerificationModel?: string
      temperature?: number
      signal?: AbortSignal
    }
  ): Promise<{
    id: string
    title: string
    description?: string
    theme: string
    visual_style_id?: string | null
    visual_style_scope?: string | null
    visual_style_name?: string | null
    visual_style_version?: number | null
    visual_style_snapshot?: PresentationVisualStyleSnapshot | null
    studio_data?: Record<string, unknown> | null
    slides: Array<{
      order: number
      layout: string
      title?: string
      content: string
      speaker_notes?: string
    }>
    version: number
    created_at: string
  }> {
    const body: Record<string, unknown> = { media_id: mediaId }
    if (options?.titleHint) body.title_hint = options.titleHint
    if (options?.theme) body.theme = options.theme
    if (options?.visualStyleId) body.visual_style_id = options.visualStyleId
    if (options?.visualStyleScope) body.visual_style_scope = options.visualStyleScope
    if (options?.provider) body.provider = options.provider
    if (options?.model) body.model = options.model
    if (options?.claimsVerificationProvider) {
      body.claims_verification_provider = options.claimsVerificationProvider
    }
    if (options?.claimsVerificationModel) {
      body.claims_verification_model = options.claimsVerificationModel
    }
    if (options?.temperature != null) body.temperature = options.temperature
    return await this.request<any>({
      path: "/api/v1/slides/generate/from-media",
      method: "POST",
      body,
      abortSignal: options?.signal
    })
  },

  async listVisualStyles(
    this: TldwApiClientCore
  ): Promise<VisualStyleRecord[]> {
    const pageSize = 200
    const allStyles: VisualStyleRecord[] = []
    let offset = 0

    while (true) {
      const payload = await this.request<any>({
        path: `/api/v1/slides/styles?limit=${pageSize}&offset=${offset}`,
        method: "GET"
      })
      const styles = Array.isArray(payload?.styles) ? payload.styles : []
      allStyles.push(...styles.map((style: unknown) => normalizeVisualStyleRecord(style)))

      const totalCount = extractOffsetPaginationTotal(payload) ?? allStyles.length
      if (allStyles.length >= totalCount || styles.length === 0) {
        return allStyles
      }
      offset += styles.length
    }
  },

  async createVisualStyle(
    this: TldwApiClientCore,
    payload: VisualStyleCreateInput
  ): Promise<VisualStyleRecord> {
    const response = await this.request<any>({
      path: "/api/v1/slides/styles",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        name: payload.name,
        description: payload.description,
        generation_rules: payload.generation_rules ?? {},
        artifact_preferences: payload.artifact_preferences ?? [],
        appearance_defaults: payload.appearance_defaults ?? {},
        fallback_policy: payload.fallback_policy ?? {}
      }
    })
    return normalizeVisualStyleRecord(response)
  },

  async patchVisualStyle(
    this: TldwApiClientCore,
    styleId: string,
    payload: VisualStylePatchInput
  ): Promise<VisualStyleRecord> {
    const response = await this.request<any>({
      path: `/api/v1/slides/styles/${encodeURIComponent(styleId)}`,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: {
        name: payload.name,
        description: payload.description,
        generation_rules: payload.generation_rules,
        artifact_preferences: payload.artifact_preferences,
        appearance_defaults: payload.appearance_defaults,
        fallback_policy: payload.fallback_policy
      }
    })
    return normalizeVisualStyleRecord(response)
  },

  async deleteVisualStyle(
    this: TldwApiClientCore,
    styleId: string
  ): Promise<void> {
    await this.request<void>({
      path: `/api/v1/slides/styles/${encodeURIComponent(styleId)}`,
      method: "DELETE"
    })
  },

  async listPresentations(
    this: TldwApiClientCore,
    options?: {
      limit?: number
      offset?: number
      sort?: string
      includeDeleted?: boolean
    }
  ): Promise<PresentationListResponse> {
    const query = buildQuery({
      limit: options?.limit,
      offset: options?.offset,
      sort: options?.sort,
      include_deleted: options?.includeDeleted
    })
    const payload = toRecord(
      await this.request<unknown>({
        path: `/api/v1/slides/presentations${query}`,
        method: "GET",
        headers: presentationNegotiationHeaders()
      })
    )
    const pagination = toRecord(payload.pagination)
    const presentations = Array.isArray(payload.presentations)
      ? payload.presentations.map(normalizePresentationSummary)
      : []
    return {
      presentations,
      total: toFiniteNumber(payload.total, presentations.length),
      limit: toFiniteNumber(payload.limit, options?.limit ?? 50),
      offset: toFiniteNumber(payload.offset, options?.offset ?? 0),
      pagination: {
        mode: "offset",
        limit: toFiniteNumber(pagination.limit, options?.limit ?? 50),
        offset: toFiniteNumber(pagination.offset, options?.offset ?? 0),
        total: toFiniteNumber(pagination.total, presentations.length),
        has_more: Boolean(pagination.has_more),
        next_offset: toOptionalNumber(pagination.next_offset)
      },
      has_more:
        typeof payload.has_more === "boolean" ? payload.has_more : Boolean(pagination.has_more),
      next_offset: toOptionalNumber(payload.next_offset ?? pagination.next_offset)
    }
  },

  async getPresentation(
    this: TldwApiClientCore,
    presentationId: string,
    options?: { abortSignal?: AbortSignal }
  ): Promise<PresentationDetailResult> {
    const response = await this.request<unknown>({
      path: `/api/v1/slides/presentations/${encodeURIComponent(presentationId)}`,
      method: "GET",
      headers: presentationNegotiationHeaders(),
      ...(options?.abortSignal ? { abortSignal: options.abortSignal } : {}),
      returnResponse: true
    })
    return {
      record: normalizePresentationStudioRecord(requireSuccessfulResponseData(response)),
      etag: responseEtag(response)
    }
  },

  async getPresentationMetadata(
    this: TldwApiClientCore,
    presentationId: string
  ): Promise<PresentationMetadataResult> {
    const response = await this.request<unknown>({
      path: `/api/v1/slides/presentations/${encodeURIComponent(presentationId)}/metadata`,
      method: "GET",
      returnResponse: true
    })
    return {
      record: normalizePresentationSummary(requireSuccessfulResponseData(response)),
      etag: responseEtag(response)
    }
  },

  async getSlidesCapabilities(
    this: TldwApiClientCore,
    options?: { abortSignal?: AbortSignal }
  ): Promise<SlidesCapabilities> {
    const response = await this.request<unknown>({
      path: "/api/v1/slides/capabilities",
      method: "GET",
      ...(options?.abortSignal ? { abortSignal: options.abortSignal } : {}),
      returnResponse: true
    })
    const payload = requireSuccessfulResponseData(response)
    const cacheControl = responseHeaders(response).get("cache-control")?.trim().toLowerCase()
    if (cacheControl !== "private, no-store") {
      throw new Error("Invalid Slides capabilities cache policy")
    }
    return validateSlidesCapabilities(payload)
  },

  async submitPresentationGeneration(
    this: TldwApiClientCore,
    payload: PresentationGenerationRequest,
    options: { idempotencyKey: string; abortSignal?: AbortSignal }
  ): Promise<PresentationGenerationReceipt> {
    const response = await this.request<unknown>({
      path: "/api/v1/slides/generations",
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Idempotency-Key": options.idempotencyKey
      },
      body: payload,
      ...(options.abortSignal ? { abortSignal: options.abortSignal } : {})
    })
    return validateGenerationReceipt(response)
  },

  async getPresentationGeneration(
    this: TldwApiClientCore,
    generationId: string
  ): Promise<PresentationGenerationReceipt> {
    const response = await this.request<unknown>({
      path: `/api/v1/slides/generations/${encodeURIComponent(generationId)}`,
      method: "GET"
    })
    return validateGenerationReceipt(response)
  },

  async getPresentationGenerationStatus(
    this: TldwApiClientCore,
    generationId: string,
    options?: { abortSignal?: AbortSignal }
  ): Promise<PresentationGenerationStatusResult> {
    const response = await this.request<unknown>({
      path: `/api/v1/slides/generations/${encodeURIComponent(generationId)}`,
      method: "GET",
      ...(options?.abortSignal ? { abortSignal: options.abortSignal } : {}),
      returnResponse: true
    })
    const record = toRecord(response)
    const retryAfter = record.retryAfterMs
    return {
      receipt: validateGenerationReceipt(requireSuccessfulResponseData(response)),
      retryAfterMs:
        typeof retryAfter === "number" && Number.isFinite(retryAfter) && retryAfter >= 0
          ? Math.min(MAX_GENERATION_RETRY_AFTER_MS, Math.floor(retryAfter))
          : null
    }
  },

  async createPresentation(
    this: TldwApiClientCore,
    payload: {
      title: string
      description?: string | null
      theme?: string
      marp_theme?: string | null
      template_id?: string | null
      visual_style_id?: string | null
      visual_style_scope?: string | null
      visual_style_name?: string | null
      visual_style_version?: number | null
      visual_style_snapshot?: PresentationVisualStyleSnapshot | null
      settings?: Record<string, any> | null
      studio_data?: Record<string, any> | null
      slides: PresentationStudioSlide[]
      custom_css?: string | null
    }
  ): Promise<StructuredPresentationStudioRecord> {
    const path = await this.resolveApiPath("slides.presentations.create", [
      "/api/v1/slides/presentations"
    ])
    const body = {
      title: payload.title,
      description: payload.description,
      theme: payload.theme,
      marp_theme: payload.marp_theme,
      template_id: payload.template_id,
      visual_style_id: payload.visual_style_id,
      visual_style_scope: payload.visual_style_scope,
      visual_style_name: payload.visual_style_name,
      visual_style_version: payload.visual_style_version,
      visual_style_snapshot: clonePresentationVisualStyleSnapshot(payload.visual_style_snapshot),
      settings: payload.settings,
      studio_data: payload.studio_data,
      slides: payload.slides,
      custom_css: payload.custom_css
    }
    const response = await this.request<any>({
      path,
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...presentationNegotiationHeaders()
      },
      body
    })
    return requireStructuredRecord(normalizePresentationStudioRecord(response))
  },

  async patchPresentation(
    this: TldwApiClientCore,
    presentationId: string,
    payload: {
      title?: string | null
      description?: string | null
      theme?: string | null
      marp_theme?: string | null
      template_id?: string | null
      visual_style_id?: string | null
      visual_style_scope?: string | null
      visual_style_name?: string | null
      visual_style_version?: number | null
      visual_style_snapshot?: PresentationVisualStyleSnapshot | null
      settings?: Record<string, any> | null
      studio_data?: Record<string, any> | null
      slides?: PresentationStudioSlide[] | null
      custom_css?: string | null
    },
    options?: { ifMatch?: string | number | null }
  ): Promise<StructuredPresentationStudioRecord> {
    const template = await this.resolveApiPath("slides.presentations.patch", [
      "/api/v1/slides/presentations/{presentation_id}"
    ])
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      ...presentationNegotiationHeaders()
    }
    if (options?.ifMatch != null) {
      headers["If-Match"] = String(options.ifMatch)
    }
    const body = {
      title: payload.title,
      description: payload.description,
      theme: payload.theme,
      marp_theme: payload.marp_theme,
      template_id: payload.template_id,
      visual_style_id: payload.visual_style_id,
      visual_style_scope: payload.visual_style_scope,
      visual_style_name: payload.visual_style_name,
      visual_style_version: payload.visual_style_version,
      visual_style_snapshot: clonePresentationVisualStyleSnapshot(payload.visual_style_snapshot),
      settings: payload.settings,
      studio_data: payload.studio_data,
      slides: payload.slides,
      custom_css: payload.custom_css
    }
    const response = await this.request<any>({
      path: this.fillPathParams(template, presentationId),
      method: "PATCH",
      headers,
      body
    })
    return requireStructuredRecord(normalizePresentationStudioRecord(response))
  },

  async submitPresentationRenderJob(
    this: TldwApiClientCore,
    presentationId: string,
    payload: { format: PresentationRenderFormat },
    options: { ifMatch: string | number }
  ): Promise<PresentationRenderJob> {
    const template = await this.resolveApiPath("slides.presentations.render.create", [
      "/api/v1/slides/presentations/{presentation_id}/render-jobs"
    ])
    return await this.request<PresentationRenderJob>({
      path: this.fillPathParams(template, presentationId),
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "If-Match": String(options.ifMatch),
        ...presentationNegotiationHeaders()
      },
      body: payload
    })
  },

  async getPresentationRenderJob(
    this: TldwApiClientCore,
    jobId: number
  ): Promise<PresentationRenderJob> {
    const template = await this.resolveApiPath("slides.presentations.render.get", [
      "/api/v1/slides/render-jobs/{job_id}"
    ])
    return await this.request<PresentationRenderJob>({
      path: this.fillPathParams(template, String(jobId)),
      method: "GET",
      headers: presentationNegotiationHeaders()
    })
  },

  async listPresentationRenderArtifacts(
    this: TldwApiClientCore,
    presentationId: string
  ): Promise<PresentationRenderArtifactList> {
    const template = await this.resolveApiPath("slides.presentations.render.artifacts", [
      "/api/v1/slides/presentations/{presentation_id}/render-artifacts"
    ])
    return await this.request<PresentationRenderArtifactList>({
      path: this.fillPathParams(template, presentationId),
      method: "GET",
      headers: presentationNegotiationHeaders()
    })
  },

  async saveStandaloneHtmlSource(
    this: TldwApiClientCore,
    presentationId: string,
    source: string,
    options: { ifMatch: string; abortSignal?: AbortSignal }
  ): Promise<StandalonePresentationDetailResult> {
    validateIfMatch(options.ifMatch)
    validateStandaloneSource(source)
    const response = await this.request<unknown>({
      path: `/api/v1/slides/presentations/${encodeURIComponent(presentationId)}/html-source`,
      method: "PUT",
      headers: {
        "Content-Type": "application/octet-stream",
        "If-Match": options.ifMatch,
        ...presentationNegotiationHeaders()
      },
      body: source,
      ...(options.abortSignal ? { abortSignal: options.abortSignal } : {}),
      returnResponse: true
    })
    return {
      record: requireStandaloneRecord(
        normalizePresentationStudioRecord(requireSuccessfulResponseData(response))
      ),
      etag: responseEtag(response)
    }
  },

  async downloadStandaloneHtmlDraft(
    this: TldwApiClientCore,
    presentationId: string,
    source: string,
    options?: { abortSignal?: AbortSignal }
  ): Promise<Uint8Array> {
    validateStandaloneSource(source)
    const response = await this.request<unknown>({
      path: `/api/v1/slides/presentations/${encodeURIComponent(presentationId)}/draft-attachment`,
      method: "POST",
      headers: {
        "Content-Type": "application/octet-stream",
        Accept: "application/octet-stream",
        ...presentationNegotiationHeaders()
      },
      body: source,
      ...(options?.abortSignal ? { abortSignal: options.abortSignal } : {}),
      responseType: "arrayBuffer",
      returnResponse: true
    })
    return exactAttachmentBytes(response)
  },

  async downloadStandaloneHtmlPresentation(
    this: TldwApiClientCore,
    presentationId: string
  ): Promise<Uint8Array> {
    const response = await this.request<unknown>({
      path: `/api/v1/slides/presentations/${encodeURIComponent(presentationId)}/export?format=html`,
      method: "GET",
      headers: {
        Accept: "application/octet-stream",
        ...presentationNegotiationHeaders()
      },
      responseType: "arrayBuffer",
      returnResponse: true
    })
    return exactAttachmentBytes(response)
  },

  async exportPresentation(
    this: TldwApiClientCore,
    presentationId: string,
    format: "revealjs" | "markdown" | "json" | "pdf"
  ): Promise<Blob> {
    await this.ensureConfigForRequest(true)

    const response = await this.request<any>({
      path: `/api/v1/slides/presentations/${encodeURIComponent(presentationId)}/export?format=${encodeURIComponent(format)}`,
      method: "GET",
      headers: presentationNegotiationHeaders(),
      responseType: "arrayBuffer",
      returnResponse: true
    })

    if (!response) {
      throw new Error("Export failed")
    }

    // Handle response data
    let data: ArrayBuffer
    if (response.data instanceof ArrayBuffer) {
      data = response.data
    } else if (response.data instanceof Uint8Array) {
      data = response.data.buffer.slice(
        response.data.byteOffset,
        response.data.byteOffset + response.data.byteLength
      )
    } else if (typeof response.data === "string") {
      const encoder = new TextEncoder()
      data = encoder.encode(response.data).buffer
    } else if (response.data && typeof response.data === "object") {
      // Handle JSON response
      const encoder = new TextEncoder()
      data = encoder.encode(JSON.stringify(response.data)).buffer
    } else {
      throw new Error("Invalid export response")
    }

    // Determine MIME type based on format
    let mimeType: string
    switch (format) {
      case "revealjs":
        mimeType = "application/zip"
        break
      case "markdown":
        mimeType = "text/markdown"
        break
      case "json":
        mimeType = "application/json"
        break
      case "pdf":
        mimeType = "application/pdf"
        break
      default:
        mimeType = "application/octet-stream"
    }

    return new Blob([data], { type: mimeType })
  },
}

export type PresentationsMethods = typeof presentationsMethods
