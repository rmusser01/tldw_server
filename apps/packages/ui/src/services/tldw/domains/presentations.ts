import { bgRequest } from "@/services/background-proxy"
import { buildQuery, toTrimmedStringArray } from "../client-utils"
import type {
  PresentationStudioSlide,
  PresentationVisualStyleSnapshot,
  VisualStyleRecord,
  VisualStyleCreateInput,
  VisualStylePatchInput,
  PresentationStudioRecord,
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

const normalizePresentationStudioRecord = (presentation: unknown): PresentationStudioRecord => {
  const record =
    presentation && typeof presentation === "object" && !Array.isArray(presentation)
      ? (presentation as Record<string, unknown>)
      : {}
  const slides = Array.isArray(record.slides)
    ? (record.slides as PresentationStudioSlide[])
    : []
  return {
    id: String(record.id ?? ""),
    title: String(record.title ?? ""),
    description: toOptionalString(record.description),
    theme: String(record.theme ?? "black"),
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
    slides,
    custom_css: toOptionalString(record.custom_css),
    source_type: toOptionalString(record.source_type),
    source_ref: record.source_ref ?? null,
    source_query: toOptionalString(record.source_query),
    created_at: String(record.created_at ?? ""),
    last_modified: String(record.last_modified ?? ""),
    deleted: Boolean(record.deleted),
    client_id: toOptionalString(record.client_id) ?? undefined,
    version: toFiniteNumber(record.version, 0)
  }
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

      const totalCount =
        typeof payload?.total_count === "number" && Number.isFinite(payload.total_count)
          ? payload.total_count
          : allStyles.length
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

  async getPresentation(
    this: TldwApiClientCore,
    presentationId: string
  ): Promise<PresentationStudioRecord> {
    const payload = await this.request<any>({
      path: `/api/v1/slides/presentations/${encodeURIComponent(presentationId)}`,
      method: "GET"
    })
    return normalizePresentationStudioRecord(payload)
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
  ): Promise<PresentationStudioRecord> {
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
      headers: { "Content-Type": "application/json" },
      body
    })
    return normalizePresentationStudioRecord(response)
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
  ): Promise<PresentationStudioRecord> {
    const template = await this.resolveApiPath("slides.presentations.patch", [
      "/api/v1/slides/presentations/{presentation_id}"
    ])
    const headers: Record<string, string> = { "Content-Type": "application/json" }
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
    return normalizePresentationStudioRecord(response)
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
        "If-Match": String(options.ifMatch)
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
      method: "GET"
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
      method: "GET"
    })
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
