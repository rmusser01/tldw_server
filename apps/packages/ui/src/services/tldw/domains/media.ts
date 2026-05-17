import { bgRequest, bgUpload } from "@/services/background-proxy"
import { inferUploadMediaTypeFromUrl } from "@/services/tldw/media-routing"
import {
  buildContentPayload,
  mapApiDetailToUi,
  mapApiListToUi,
  mapUiSourceToApi,
  type ApiDataTableGenerateResponse,
  type ApiDataTableJobStatus
} from "@/services/tldw/data-tables"
import type { DataTableColumn } from "@/types/data-tables"
import type { AllowedPath, PathOrUrl } from "@/services/tldw/openapi-guard"
import type {
  FileCreateResponse,
  ImageArtifactRequest,
  ReferenceImageCandidate,
  ReferenceImageListResponse
} from "../TldwApiClient"
import {
  normalizePlaylistPreflightResponse,
  type ApiPlaylistPreflightResponse,
  type PlaylistPreflightResult
} from "@/services/tldw/playlist-preflight"
import {
  normalizeMediaCollectionItem,
  normalizeMediaCollectionListResponse,
  normalizeMediaCollectionResponse,
  type ApiMediaCollection,
  type ApiMediaCollectionItem,
  type ApiMediaCollectionListResponse,
  type MediaCollection,
  type MediaCollectionItem,
  type MediaCollectionList
} from "@/services/tldw/conference-collections"

/**
 * Builds a query string from a record of parameters.
 * Replaces `this.buildQuery(...)` from TldwApiClient.
 */
function buildQuery(params?: Record<string, any>): string {
  if (!params || Object.keys(params).length === 0) {
    return ''
  }
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null) continue
    if (Array.isArray(value)) {
      value.forEach((entry) => search.append(key, String(entry)))
      continue
    }
    search.append(key, String(value))
  }
  const query = search.toString()
  return query ? `?${query}` : ''
}

const normalizeReferenceImageCandidate = (
  item: any
): ReferenceImageCandidate => ({
  file_id: Number(item?.file_id ?? 0),
  title: String(item?.title ?? ""),
  mime_type: String(item?.mime_type ?? ""),
  width:
    typeof item?.width === "number" && Number.isFinite(item.width)
      ? item.width
      : null,
  height:
    typeof item?.height === "number" && Number.isFinite(item.height)
      ? item.height
      : null,
  created_at: String(item?.created_at ?? "")
})

const normalizeReferenceImageListResponse = (
  payload: any
): ReferenceImageListResponse => {
  const items = Array.isArray(payload?.items)
    ? payload.items.map((item: any) => normalizeReferenceImageCandidate(item))
    : []
  return { items }
}

export const mediaMethods = {
  // ─────────────────────────────────────────────────────────────────────────────
  // Media Ingestion & CRUD
  // ─────────────────────────────────────────────────────────────────────────────

  async addMedia(url: string, metadata?: any): Promise<any> {
    const sourceUrl = String(url || "").trim()
    const {
      timeoutMs,
      media_type,
      urls: rawUrls,
      ...rest
    } = metadata || {}
    const urls = Array.isArray(rawUrls)
      ? rawUrls
          .map((item: unknown) => String(item || "").trim())
          .filter((item: string) => item.length > 0)
      : typeof rawUrls === "string" && rawUrls.trim()
        ? [rawUrls.trim()]
        : sourceUrl
          ? [sourceUrl]
          : []
    if (urls.length === 0) {
      throw new Error("addMedia requires a URL")
    }
    const resolvedMediaType =
      typeof media_type === "string" && media_type.trim()
        ? media_type.trim()
        : inferUploadMediaTypeFromUrl(urls[0])

    return await bgUpload<any>({
      path: "/api/v1/media/add",
      method: "POST",
      fields: {
        ...rest,
        media_type: resolvedMediaType,
        urls
      },
      timeoutMs
    })
  },

  async submitMediaIngestJobs(fields?: Record<string, any>): Promise<any> {
    const { timeoutMs, ...rest } = fields || {}
    const normalized: Record<string, any> = {}
    for (const [k, v] of Object.entries(rest || {})) {
      if (typeof v === "undefined" || v === null) continue
      normalized[k] = v
    }
    return await bgUpload<any>({
      path: "/api/v1/media/ingest/jobs",
      method: "POST",
      fields: normalized,
      timeoutMs
    })
  },

  async preflightPlaylist(payload: {
    url: string
    max_items?: number
    maxItems?: number
    timeoutMs?: number
  }): Promise<PlaylistPreflightResult> {
    const { timeoutMs, maxItems, max_items, ...rest } = payload || {}
    const body = {
      ...rest,
      max_items: max_items ?? maxItems
    }
    if (typeof body.max_items === "undefined") {
      delete (body as Record<string, unknown>).max_items
    }
    const response = await bgRequest<ApiPlaylistPreflightResponse>({
      path: "/api/v1/media/playlists/preflight",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      timeoutMs
    })
    return normalizePlaylistPreflightResponse(response)
  },

  async createMediaCollection(
    payload: Record<string, any>,
    options?: { timeoutMs?: number }
  ): Promise<MediaCollection> {
    const response = await bgRequest<ApiMediaCollection>({
      path: "/api/v1/media/collections",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
      timeoutMs: options?.timeoutMs
    })
    return normalizeMediaCollectionResponse(response)
  },

  async listMediaCollections(
    params?: {
      kind?: string
      page?: number
      size?: number
    },
    options?: { timeoutMs?: number; signal?: AbortSignal }
  ): Promise<MediaCollectionList> {
    const query = buildQuery(params as Record<string, any>)
    const response = await bgRequest<ApiMediaCollectionListResponse>({
      path: `/api/v1/media/collections${query}`,
      method: "GET",
      timeoutMs: options?.timeoutMs,
      abortSignal: options?.signal
    })
    return normalizeMediaCollectionListResponse(response)
  },

  async getMediaCollection(
    collectionId: number | string,
    options?: { timeoutMs?: number; signal?: AbortSignal }
  ): Promise<MediaCollection> {
    const id = encodeURIComponent(String(collectionId))
    const response = await bgRequest<ApiMediaCollection>({
      path: `/api/v1/media/collections/${id}`,
      method: "GET",
      timeoutMs: options?.timeoutMs,
      abortSignal: options?.signal
    })
    return normalizeMediaCollectionResponse(response)
  },

  async updateMediaCollection(
    collectionId: number | string,
    payload: Record<string, any>,
    options?: { timeoutMs?: number }
  ): Promise<MediaCollection> {
    const id = encodeURIComponent(String(collectionId))
    const response = await bgRequest<ApiMediaCollection>({
      path: `/api/v1/media/collections/${id}`,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: payload,
      timeoutMs: options?.timeoutMs
    })
    return normalizeMediaCollectionResponse(response)
  },

  async addMediaCollectionItem(
    collectionId: number | string,
    payload: Record<string, any>,
    options?: { timeoutMs?: number }
  ): Promise<MediaCollectionItem> {
    const id = encodeURIComponent(String(collectionId))
    const response = await bgRequest<ApiMediaCollectionItem>({
      path: `/api/v1/media/collections/${id}/items`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
      timeoutMs: options?.timeoutMs
    })
    return normalizeMediaCollectionItem(response)
  },

  async updateMediaCollectionItem(
    collectionId: number | string,
    itemId: number | string,
    payload: Record<string, any>,
    options?: { timeoutMs?: number }
  ): Promise<MediaCollectionItem> {
    const id = encodeURIComponent(String(collectionId))
    const item = encodeURIComponent(String(itemId))
    const response = await bgRequest<ApiMediaCollectionItem>({
      path: `/api/v1/media/collections/${id}/items/${item}`,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: payload,
      timeoutMs: options?.timeoutMs
    })
    return normalizeMediaCollectionItem(response)
  },

  async getMediaIngestJob(
    jobId: number | string,
    options?: { timeoutMs?: number }
  ): Promise<any> {
    return await bgRequest<any>({
      path: `/api/v1/media/ingest/jobs/${encodeURIComponent(String(jobId))}`,
      method: "GET",
      timeoutMs: options?.timeoutMs
    })
  },

  async listMediaIngestJobs(
    params: {
      batch_id: string
      limit?: number
    },
    options?: { timeoutMs?: number }
  ): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({
      path: `/api/v1/media/ingest/jobs${query}`,
      method: "GET",
      timeoutMs: options?.timeoutMs
    })
  },

  async addMediaForm(fields: Record<string, any>): Promise<any> {
    // Multipart form for rich ingest parameters
    // Accepts a flat fields map; callers may pass booleans/strings and they will be converted
    const normalized: Record<string, any> = {}
    for (const [k, v] of Object.entries(fields || {})) {
      if (typeof v === 'undefined' || v === null) continue
      if (typeof v === 'boolean') normalized[k] = v ? 'true' : 'false'
      else normalized[k] = v
    }
    return await bgUpload<any>({ path: '/api/v1/media/add', method: 'POST', fields: normalized })
  },

  async uploadMedia(
    file: File,
    fields?: Record<string, any>,
    getConfig?: () => Promise<any>
  ): Promise<any> {
    const data = await file.arrayBuffer()
    const name = file.name || 'upload'
    const type = file.type || 'application/octet-stream'
    const normalized: Record<string, any> = {}
    for (const [k, v] of Object.entries(fields || {})) {
      if (typeof v === 'undefined' || v === null) continue
      if (typeof v === 'boolean') normalized[k] = v ? 'true' : 'false'
      else normalized[k] = v
    }
    let uploadTimeoutMs = 60000
    if (getConfig) {
      const cfg = await getConfig().catch(() => null)
      if (cfg && typeof (cfg as any).uploadRequestTimeoutMs === "number") {
        const cfgTimeout = Number((cfg as any).uploadRequestTimeoutMs)
        if (cfgTimeout > 0) {
          uploadTimeoutMs = cfgTimeout
        }
      }
    }
    uploadTimeoutMs = Math.max(uploadTimeoutMs, 5000)
    return await bgUpload<any>({
      path: '/api/v1/media/add',
      method: 'POST',
      fields: normalized,
      file: { name, type, data },
      fileFieldName: 'files',
      timeoutMs: uploadTimeoutMs
    })
  },

  async listMedia(
    params?: {
      page?: number
      results_per_page?: number
      include_keywords?: boolean
    },
    options?: { signal?: AbortSignal }
  ): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({
      path: `/api/v1/media${query}`,
      method: "GET",
      abortSignal: options?.signal
    })
  },

  async listReferenceImageCandidates(
    options?: { signal?: AbortSignal }
  ): Promise<ReferenceImageListResponse> {
    const response = await bgRequest<any>({
      path: "/api/v1/files/reference-images" as AllowedPath,
      method: "GET",
      abortSignal: options?.signal
    })
    return normalizeReferenceImageListResponse(response)
  },

  async searchMedia(
    payload: {
      query?: string
      fields?: string[]
      exact_phrase?: string
      media_types?: string[]
      date_range?: Record<string, any>
      must_have?: string[]
      must_not_have?: string[]
      sort_by?: string
      boost_fields?: Record<string, number>
    },
    params?: { page?: number; results_per_page?: number },
    options?: { signal?: AbortSignal }
  ): Promise<any> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<any>({
      path: `/api/v1/media/search${query}`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload,
      abortSignal: options?.signal
    })
  },

  async updateMediaKeywords(
    mediaId: string | number,
    payload: { keywords: string[]; mode?: "add" | "remove" | "set" }
  ): Promise<{ media_id: number; keywords: string[] }> {
    const id = encodeURIComponent(String(mediaId))
    return await bgRequest<{ media_id: number; keywords: string[] }>({
      path: `/api/v1/media/${id}/keywords`,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async bulkUpdateMediaKeywords(payload: {
    media_ids: number[]
    keywords: string[]
    mode?: "add" | "remove" | "set"
  }): Promise<{
    endpoint: "bulk" | "fallback"
    updated: number
    failed: number
    results: Array<{
      media_id: number
      success: boolean
      keywords: string[] | null
      error: string | null
    }>
  }> {
    const rawIds = Array.isArray(payload.media_ids) ? payload.media_ids : []
    const mediaIds = Array.from(
      new Set(
        rawIds
          .map((id) => Number(id))
          .filter((id) => Number.isFinite(id) && id > 0)
          .map((id) => Math.trunc(id))
      )
    )
    if (mediaIds.length === 0) {
      throw new Error("media_ids_required")
    }

    const keywords = Array.isArray(payload.keywords)
      ? payload.keywords
          .map((keyword) => String(keyword ?? "").trim())
          .filter((keyword) => keyword.length > 0)
      : []
    const mode = payload.mode ?? "add"

    const requestPayload = {
      media_ids: mediaIds,
      keywords,
      mode
    } as const

    try {
      const response = await bgRequest<any>({
        path: "/api/v1/media/bulk/keyword-update",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: requestPayload
      })
      const results = Array.isArray(response?.results)
        ? response.results.map((entry: any) => ({
            media_id: Number(entry?.media_id ?? entry?.id ?? 0),
            success: Boolean(entry?.success ?? true),
            keywords: Array.isArray(entry?.keywords) ? entry.keywords.map(String) : null,
            error:
              typeof entry?.error === "string"
                ? entry.error
                : typeof entry?.detail === "string"
                  ? entry.detail
                  : null
          }))
        : mediaIds.map((mediaId) => ({
            media_id: mediaId,
            success: true,
            keywords: null,
            error: null
          }))
      const updatedCount =
        typeof response?.updated === "number"
          ? Math.max(0, Math.trunc(response.updated))
          : results.filter((entry) => entry.success).length
      const failedCount =
        typeof response?.failed === "number"
          ? Math.max(0, Math.trunc(response.failed))
          : Math.max(0, results.length - updatedCount)

      return {
        endpoint: "bulk",
        updated: updatedCount,
        failed: failedCount,
        results
      }
    } catch (error) {
      const candidate = error as
        | { status?: number; response?: { status?: number }; statusCode?: number }
        | undefined
      const statusCode = Number(
        candidate?.status ?? candidate?.response?.status ?? candidate?.statusCode
      )
      if (!Number.isFinite(statusCode) || (statusCode !== 404 && statusCode !== 405)) {
        throw error
      }
    }

    const settled = await Promise.allSettled(
      mediaIds.map(async (mediaId) => {
        const updated = await mediaMethods.updateMediaKeywords(mediaId, {
          keywords,
          mode
        })
        return {
          media_id: mediaId,
          success: true,
          keywords: Array.isArray(updated?.keywords) ? updated.keywords : [],
          error: null as string | null
        }
      })
    )

    const results = settled.map((entry, index) => {
      const mediaId = mediaIds[index]
      if (entry.status === "fulfilled") {
        return entry.value
      }
      const reason = entry.reason
      const detail =
        typeof reason?.message === "string"
          ? reason.message
          : typeof reason === "string"
            ? reason
            : "keyword_update_failed"
      return {
        media_id: mediaId,
        success: false,
        keywords: null,
        error: detail
      }
    })
    const updated = results.filter((entry) => entry.success).length

    return {
      endpoint: "fallback",
      updated,
      failed: results.length - updated,
      results
    }
  },

  async deleteMedia(mediaId: string | number): Promise<void> {
    const id = encodeURIComponent(String(mediaId))
    await bgRequest<void>({
      path: `/api/v1/media/${id}`,
      method: "DELETE"
    })
  },

  async restoreMedia(mediaId: string | number): Promise<any> {
    const id = encodeURIComponent(String(mediaId))
    return await bgRequest<any>({
      path: `/api/v1/media/${id}/restore`,
      method: "POST"
    })
  },

  async permanentlyDeleteMedia(mediaId: string | number): Promise<void> {
    const id = encodeURIComponent(String(mediaId))
    await bgRequest<void>({
      path: `/api/v1/media/${id}/permanent`,
      method: "DELETE"
    })
  },

  async reprocessMedia(
    mediaId: string | number,
    options?: Record<string, unknown>
  ): Promise<any> {
    const id = encodeURIComponent(String(mediaId))
    return await bgRequest<any>({
      path: `/api/v1/media/${id}/reprocess`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: options || {}
    })
  },

  async getMediaStatistics(): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/media/statistics",
      method: "GET"
    })
  },

  async getMediaDetails(
    mediaId: string | number,
    options?: {
      include_content?: boolean
      include_versions?: boolean
      include_version_content?: boolean
      signal?: AbortSignal
    }
  ): Promise<any> {
    const id = encodeURIComponent(String(mediaId))
    const query = buildQuery({
      include_content: options?.include_content ?? true,
      include_versions: options?.include_versions ?? false,
      include_version_content: options?.include_version_content ?? false
    })
    return await bgRequest<any>({
      path: `/api/v1/media/${id}${query}`,
      method: "GET",
      abortSignal: options?.signal
    })
  },

  async getDocumentOutline(
    mediaId: string | number,
    options?: { signal?: AbortSignal }
  ): Promise<{
    media_id: number
    has_outline: boolean
    entries: Array<{ level: number; title: string; page: number }>
    total_pages: number
  }> {
    const id = encodeURIComponent(String(mediaId))
    return await bgRequest<{
      media_id: number
      has_outline: boolean
      entries: Array<{ level: number; title: string; page: number }>
      total_pages: number
    }>({
      path: `/api/v1/media/${id}/outline`,
      method: "GET",
      abortSignal: options?.signal
    })
  },

  async generateDocumentInsights(
    mediaId: string | number,
    options?: {
      categories?: string[]
      model?: string
      max_content_length?: number
      force?: boolean
      signal?: AbortSignal
    }
  ): Promise<{
    media_id: number
    insights: Array<{
      category: string
      title: string
      content: string
      confidence?: number
    }>
    model_used: string
    cached: boolean
  }> {
    const id = encodeURIComponent(String(mediaId))
    const body: Record<string, unknown> = {}
    if (options?.categories) body.categories = options.categories
    if (options?.model) body.model = options.model
    if (options?.max_content_length) body.max_content_length = options.max_content_length
    if (options?.force) body.force = options.force

    return await bgRequest<{
      media_id: number
      insights: Array<{
        category: string
        title: string
        content: string
        confidence?: number
      }>
      model_used: string
      cached: boolean
    }>({
      path: `/api/v1/media/${id}/insights`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      abortSignal: options?.signal
    })
  },

  async getDocumentFigures(
    mediaId: string | number,
    options?: {
      minSize?: number
      signal?: AbortSignal
    }
  ): Promise<{
    media_id: number
    has_figures: boolean
    figures: Array<{
      id: string
      page: number
      width: number
      height: number
      format: string
      data_url?: string
      caption?: string
    }>
    total_count: number
  }> {
    const id = encodeURIComponent(String(mediaId))
    const minSize = options?.minSize ?? 50
    return await bgRequest<{
      media_id: number
      has_figures: boolean
      figures: Array<{
        id: string
        page: number
        width: number
        height: number
        format: string
        data_url?: string
        caption?: string
      }>
      total_count: number
    }>({
      path: `/api/v1/media/${id}/figures?min_size=${minSize}`,
      method: "GET",
      abortSignal: options?.signal
    })
  },

  async getDocumentReferences(
    mediaId: string | number,
    options?: {
      enrich?: boolean
      referenceIndex?: number
      offset?: number
      limit?: number
      parseCap?: number
      search?: string
      signal?: AbortSignal
    }
  ): Promise<{
    media_id: number
    has_references: boolean
    references: Array<{
      raw_text: string
      title?: string
      authors?: string
      year?: number
      venue?: string
      doi?: string
      arxiv_id?: string
      url?: string
      citation_count?: number
      semantic_scholar_id?: string
      open_access_pdf?: string
    }>
    enrichment_source?: string
    enriched_count?: number
    enrichment_limited?: boolean
    total_detected?: number
    truncated?: boolean
    offset?: number
    limit?: number
    returned_count?: number
    total_available?: number
    has_more?: boolean
    next_offset?: number | null
  }> {
    const id = encodeURIComponent(String(mediaId))
    const enrich = options?.enrich !== false
    const referenceIndex =
      typeof options?.referenceIndex === "number"
        ? `&reference_index=${options.referenceIndex}`
        : ""
    const offset =
      typeof options?.offset === "number" ? `&offset=${Math.max(0, options.offset)}` : ""
    const limit =
      typeof options?.limit === "number" ? `&limit=${Math.max(1, options.limit)}` : ""
    const parseCap =
      typeof options?.parseCap === "number" ? `&parse_cap=${Math.max(1, options.parseCap)}` : ""
    const search =
      typeof options?.search === "string" && options.search.trim().length > 0
        ? `&search=${encodeURIComponent(options.search.trim())}`
        : ""
    return await bgRequest<{
      media_id: number
      has_references: boolean
      references: Array<{
        raw_text: string
        title?: string
        authors?: string
        year?: number
        venue?: string
        doi?: string
        arxiv_id?: string
        url?: string
        citation_count?: number
        semantic_scholar_id?: string
        open_access_pdf?: string
      }>
      enrichment_source?: string
      enriched_count?: number
      enrichment_limited?: boolean
      total_detected?: number
      truncated?: boolean
      offset?: number
      limit?: number
      returned_count?: number
      total_available?: number
      has_more?: boolean
      next_offset?: number | null
    }>({
      path: `/api/v1/media/${id}/references?enrich=${enrich}${referenceIndex}${offset}${limit}${parseCap}${search}`,
      method: "GET",
      abortSignal: options?.signal,
      timeoutMs: 45000
    })
  },

  // ─────────────────────────────────────────────────────────────────────────────
  // Translation
  // ─────────────────────────────────────────────────────────────────────────────

  async translate(
    text: string,
    targetLanguage: string = "English",
    options?: { model?: string; provider?: string }
  ): Promise<{
    translated_text: string
    target_language: string
    model_used: string
    detected_source_language?: string
  }> {
    return await bgRequest<{
      translated_text: string
      target_language: string
      model_used: string
      detected_source_language?: string
    }>({
      path: "/api/v1/translate",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        text,
        target_language: targetLanguage,
        ...(options?.model && { model: options.model }),
        ...(options?.provider && { provider: options.provider })
      }
    })
  },

  // ─────────────────────────────────────────────────────────────────────────────
  // Document Annotations
  // ─────────────────────────────────────────────────────────────────────────────

  async listAnnotations(
    mediaId: string | number,
    options?: { signal?: AbortSignal }
  ): Promise<{
    media_id: number
    annotations: Array<{
      id: string
      media_id: number
      location: string
      text: string
      color: "yellow" | "green" | "blue" | "pink"
      note?: string
      annotation_type: "highlight" | "page_note"
      created_at: string
      updated_at: string
    }>
    total_count: number
  }> {
    const id = encodeURIComponent(String(mediaId))
    return await bgRequest<{
      media_id: number
      annotations: Array<{
        id: string
        media_id: number
        location: string
        text: string
        color: "yellow" | "green" | "blue" | "pink"
        note?: string
        annotation_type: "highlight" | "page_note"
        created_at: string
        updated_at: string
      }>
      total_count: number
    }>({
      path: `/api/v1/media/${id}/annotations`,
      method: "GET",
      abortSignal: options?.signal
    })
  },

  async createAnnotation(
    mediaId: string | number,
    annotation: {
      location: string
      text: string
      color?: "yellow" | "green" | "blue" | "pink"
      note?: string
      annotation_type?: "highlight" | "page_note"
    },
    options?: { signal?: AbortSignal }
  ): Promise<{
    id: string
    media_id: number
    location: string
    text: string
    color: "yellow" | "green" | "blue" | "pink"
    note?: string
    annotation_type: "highlight" | "page_note"
    created_at: string
    updated_at: string
  }> {
    const id = encodeURIComponent(String(mediaId))
    return await bgRequest<{
      id: string
      media_id: number
      location: string
      text: string
      color: "yellow" | "green" | "blue" | "pink"
      note?: string
      annotation_type: "highlight" | "page_note"
      created_at: string
      updated_at: string
    }>({
      path: `/api/v1/media/${id}/annotations`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: annotation,
      abortSignal: options?.signal
    })
  },

  async updateAnnotation(
    mediaId: string | number,
    annotationId: string,
    updates: {
      text?: string
      color?: "yellow" | "green" | "blue" | "pink"
      note?: string
    },
    options?: { signal?: AbortSignal }
  ): Promise<{
    id: string
    media_id: number
    location: string
    text: string
    color: "yellow" | "green" | "blue" | "pink"
    note?: string
    annotation_type: "highlight" | "page_note"
    created_at: string
    updated_at: string
  }> {
    const id = encodeURIComponent(String(mediaId))
    const annId = encodeURIComponent(annotationId)
    return await bgRequest<{
      id: string
      media_id: number
      location: string
      text: string
      color: "yellow" | "green" | "blue" | "pink"
      note?: string
      annotation_type: "highlight" | "page_note"
      created_at: string
      updated_at: string
    }>({
      path: `/api/v1/media/${id}/annotations/${annId}`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: updates,
      abortSignal: options?.signal
    })
  },

  async deleteAnnotation(
    mediaId: string | number,
    annotationId: string,
    options?: { signal?: AbortSignal }
  ): Promise<void> {
    const id = encodeURIComponent(String(mediaId))
    const annId = encodeURIComponent(annotationId)
    await bgRequest<void>({
      path: `/api/v1/media/${id}/annotations/${annId}`,
      method: "DELETE",
      abortSignal: options?.signal
    })
  },

  async syncAnnotations(
    mediaId: string | number,
    annotations: Array<{
      location: string
      text: string
      color?: "yellow" | "green" | "blue" | "pink"
      note?: string
      annotation_type?: "highlight" | "page_note"
    }>,
    clientIds?: string[],
    options?: { signal?: AbortSignal }
  ): Promise<{
    media_id: number
    synced_count: number
    annotations: Array<{
      id: string
      media_id: number
      location: string
      text: string
      color: "yellow" | "green" | "blue" | "pink"
      note?: string
      annotation_type: "highlight" | "page_note"
      created_at: string
      updated_at: string
    }>
    id_mapping?: Record<string, string>
  }> {
    const id = encodeURIComponent(String(mediaId))
    return await bgRequest<{
      media_id: number
      synced_count: number
      annotations: Array<{
        id: string
        media_id: number
        location: string
        text: string
        color: "yellow" | "green" | "blue" | "pink"
        note?: string
        annotation_type: "highlight" | "page_note"
        created_at: string
        updated_at: string
      }>
      id_mapping?: Record<string, string>
    }>({
      path: `/api/v1/media/${id}/annotations/sync`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        annotations,
        ...(clientIds && { client_ids: clientIds })
      },
      abortSignal: options?.signal
    })
  },

  // ─────────────────────────────────────────────────────────────────────────────
  // Reading Progress
  // ─────────────────────────────────────────────────────────────────────────────

  async getReadingProgress(
    mediaId: string | number,
    options?: { signal?: AbortSignal }
  ): Promise<{
    media_id: number
    has_progress?: boolean
    current_page?: number
    total_pages?: number
    zoom_level?: number
    view_mode?: "single" | "continuous" | "thumbnails"
    percent_complete?: number
    cfi?: string
    last_read_at?: string
  }> {
    const id = encodeURIComponent(String(mediaId))
    return await bgRequest<{
      media_id: number
      has_progress?: boolean
      current_page?: number
      total_pages?: number
      zoom_level?: number
      view_mode?: "single" | "continuous" | "thumbnails"
      percent_complete?: number
      cfi?: string
      last_read_at?: string
    }>({
      path: `/api/v1/media/${id}/progress`,
      method: "GET",
      abortSignal: options?.signal
    })
  },

  async updateReadingProgress(
    mediaId: string | number,
    progress: {
      current_page: number
      total_pages: number
      zoom_level?: number
      view_mode?: "single" | "continuous" | "thumbnails"
      cfi?: string
      percentage?: number
    },
    options?: { signal?: AbortSignal }
  ): Promise<{
    media_id: number
    current_page: number
    total_pages: number
    zoom_level: number
    view_mode: "single" | "continuous" | "thumbnails"
    percent_complete: number
    cfi?: string
    last_read_at: string
  }> {
    const id = encodeURIComponent(String(mediaId))
    return await bgRequest<{
      media_id: number
      current_page: number
      total_pages: number
      zoom_level: number
      view_mode: "single" | "continuous" | "thumbnails"
      percent_complete: number
      cfi?: string
      last_read_at: string
    }>({
      path: `/api/v1/media/${id}/progress`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: progress,
      abortSignal: options?.signal
    })
  },

  async deleteReadingProgress(
    mediaId: string | number,
    options?: { signal?: AbortSignal }
  ): Promise<void> {
    const id = encodeURIComponent(String(mediaId))
    await bgRequest<void>({
      path: `/api/v1/media/${id}/progress`,
      method: "DELETE",
      abortSignal: options?.signal
    })
  },

  // ─────────────────────────────────────────────────────────────────────────────
  // Data Tables
  // ─────────────────────────────────────────────────────────────────────────────

  async listDataTables(params?: {
    page?: number
    page_size?: number
    limit?: number
    offset?: number
    search?: string
    status?: string
    workspace_tag?: string
  }): Promise<{ tables: any[]; total: number }> {
    const limit = params?.limit ?? params?.page_size ?? 20
    const page = params?.page ?? 1
    const offset = params?.offset ?? Math.max(0, (page - 1) * limit)
    const query = buildQuery({
      limit,
      offset,
      search: params?.search,
      status_filter: params?.status,
      workspace_tag: params?.workspace_tag
    } as Record<string, any>)
    const response = await bgRequest<any>({
      path: `/api/v1/data-tables${query}`,
      method: "GET"
    })
    return mapApiListToUi(response)
  },

  async getDataTable(
    tableId: string,
    params?: {
      rows_limit?: number
      rows_offset?: number
      include_rows?: boolean
      include_sources?: boolean
    }
  ): Promise<any> {
    const id = encodeURIComponent(tableId)
    const query = buildQuery({
      rows_limit: params?.rows_limit,
      rows_offset: params?.rows_offset,
      include_rows: params?.include_rows,
      include_sources: params?.include_sources
    } as Record<string, any>)
    const response = await bgRequest<any>({
      path: `/api/v1/data-tables/${id}${query}`,
      method: "GET"
    })
    return response?.table ? mapApiDetailToUi(response) : response
  },

  async generateDataTable(payload: {
    name: string
    prompt: string
    workspace_tag?: string
    sources: Array<{ type: string; id: string; title: string; snippet?: string }>
    column_hints?: Array<{ name?: string; type?: string; description?: string; format?: string }>
    model?: string
    max_rows?: number
  }): Promise<ApiDataTableGenerateResponse> {
    const body = {
      name: payload.name,
      prompt: payload.prompt,
      workspace_tag: payload.workspace_tag,
      sources: payload.sources.map(mapUiSourceToApi),
      column_hints: payload.column_hints,
      model: payload.model,
      max_rows: payload.max_rows
    }
    return await bgRequest<ApiDataTableGenerateResponse>({
      path: "/api/v1/data-tables/generate",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body
    })
  },

  async updateDataTable(
    tableId: string,
    payload: { name?: string; description?: string }
  ): Promise<any> {
    const id = encodeURIComponent(tableId)
    return await bgRequest<any>({
      path: `/api/v1/data-tables/${id}`,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
  },

  async saveDataTableContent(
    tableId: string,
    payload: {
      columns: DataTableColumn[]
      rows: Record<string, any>[]
    }
  ): Promise<any> {
    const id = encodeURIComponent(tableId)
    const body = buildContentPayload(payload.columns, payload.rows)
    const response = await bgRequest<any>({
      path: `/api/v1/data-tables/${id}/content`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body
    })
    return response?.table ? mapApiDetailToUi(response) : response
  },

  async deleteDataTable(tableId: string): Promise<void> {
    const id = encodeURIComponent(tableId)
    await bgRequest<void>({
      path: `/api/v1/data-tables/${id}`,
      method: "DELETE"
    })
  },

  async getDataTableJob(jobId: number): Promise<ApiDataTableJobStatus> {
    return await bgRequest<ApiDataTableJobStatus>({
      path: `/api/v1/data-tables/jobs/${encodeURIComponent(String(jobId))}`,
      method: "GET"
    })
  },

  async regenerateDataTable(
    tableId: string,
    payload?: { prompt?: string; model?: string; max_rows?: number }
  ): Promise<ApiDataTableGenerateResponse> {
    const id = encodeURIComponent(tableId)
    return await bgRequest<ApiDataTableGenerateResponse>({
      path: `/api/v1/data-tables/${id}/regenerate`,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload || {}
    })
  },

  async exportDataTable(
    tableId: string,
    format: "csv" | "xlsx" | "json",
    ensureConfigForRequest: (requireAuth: boolean) => Promise<void>
  ): Promise<{ blob: Blob; filename: string }> {
    await ensureConfigForRequest(true)

    const fallbackFilename = `data-table-${tableId}.${format}`
    const resolveFilename = (res: Response) => {
      const disposition = res.headers.get("content-disposition")
      if (!disposition) return fallbackFilename
      const utfMatch = disposition.match(/filename\*=UTF-8''([^;]+)/i)
      const plainMatch = disposition.match(/filename="?([^\";]+)"?/i)
      const raw = utfMatch?.[1] || plainMatch?.[1]
      if (!raw) return fallbackFilename
      try {
        return decodeURIComponent(raw)
      } catch {
        return raw
      }
    }
    const readErrorDetail = async (res: Response) => {
      try {
        const data = await res.json()
        return data?.detail || data?.error || data?.message
      } catch {
        return undefined
      }
    }
    const bytesToArrayBuffer = (bytes: Uint8Array): ArrayBuffer => {
      if (bytes.buffer instanceof ArrayBuffer) {
        return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
      }
      return new Uint8Array(bytes).buffer as ArrayBuffer
    }
    const requestWithAuth = async (
      path: PathOrUrl,
      options?: {
        method?: "GET" | "POST" | "PUT" | "DELETE"
        body?: unknown
        signal?: AbortSignal
      }
    ) => {
      const response = await bgRequest<{
        ok: boolean
        status: number
        data?: unknown
        error?: string
        headers?: Record<string, string>
      }, PathOrUrl>({
        path,
        method: options?.method ?? "GET",
        body: options?.body,
        abortSignal: options?.signal,
        responseType: "arrayBuffer",
        returnResponse: true
      })
      if (!response) {
        throw new Error(`Request failed (${options?.method ?? "GET"} ${path})`)
      }
      if (!response.ok && response.status === 0) {
        throw new Error(response.error || "Network error")
      }
      const headers = new Headers(response.headers || {})
      let body: BodyInit | null = null
      if (response.data instanceof ArrayBuffer) {
        body = response.data
      } else if (response.data instanceof Uint8Array) {
        body = bytesToArrayBuffer(response.data)
      } else if (response.data instanceof Blob) {
        body = response.data
      } else if (typeof response.data === "string") {
        body = response.data
      } else if (response.data != null) {
        body = JSON.stringify(response.data)
        if (!headers.has("content-type")) {
          headers.set("content-type", "application/json")
        }
      }
      return new Response(body, { status: response.status, headers })
    }
    const readBlobResponse = async (res: Response) => {
      const blob = await res.blob()
      return { blob, filename: resolveFilename(res) }
    }
    const decodeBase64Blob = (data: string, contentType?: string | null) => {
      const binary = atob(data)
      const bytes = new Uint8Array(binary.length)
      for (let i = 0; i < binary.length; i += 1) {
        bytes[i] = binary.charCodeAt(i)
      }
      return new Blob([bytes], {
        type: contentType || "application/octet-stream"
      })
    }
    const waitForExportReady = async (fileId: number) => {
      const timeoutMs = 5 * 60 * 1000
      const intervalMs = 1500
      const start = Date.now()
      while (Date.now() - start < timeoutMs) {
        const statusRes = await requestWithAuth(`/api/v1/files/${fileId}`)
        if (!statusRes.ok) {
          const detail = await readErrorDetail(statusRes)
          throw new Error(detail || `Export status failed: ${statusRes.status}`)
        }
        const payload = await statusRes.json()
        const exportInfo = payload?.artifact?.export || payload?.export
        if (exportInfo?.status === "ready") {
          return exportInfo
        }
        if (exportInfo?.status && exportInfo.status !== "pending") {
          throw new Error("Export failed")
        }
        await new Promise((resolve) => setTimeout(resolve, intervalMs))
      }
      throw new Error("Export timed out")
    }
    const downloadFromUrl = async (url: string) => {
      const resolved = url.startsWith("http")
        ? (url as PathOrUrl)
        : ((url.startsWith("/") ? url : `/${url}`) as PathOrUrl)
      const fileRes = await requestWithAuth(resolved)
      if (!fileRes.ok) {
        const detail = await readErrorDetail(fileRes)
        throw new Error(detail || `Export download failed: ${fileRes.status}`)
      }
      return await readBlobResponse(fileRes)
    }
    const exportViaArtifact = async () => {
      const exportUrl =
        `/api/v1/data-tables/${encodeURIComponent(tableId)}/export?format=${encodeURIComponent(
          format
        )}&async_mode=auto&mode=url` as AllowedPath
      const exportRes = await requestWithAuth(exportUrl)
      if (!exportRes.ok) {
        const detail = await readErrorDetail(exportRes)
        throw new Error(detail || `Export failed: ${exportRes.status}`)
      }
      const contentType = exportRes.headers.get("content-type") || ""
      if (!contentType.includes("application/json")) {
        return await readBlobResponse(exportRes)
      }
      const payload = await exportRes.json()
      const exportInfo = payload?.export || payload?.artifact?.export
      const fileId = payload?.file_id || payload?.artifact?.file_id
      if (exportInfo?.content_b64) {
        const blob = decodeBase64Blob(
          exportInfo.content_b64,
          exportInfo.content_type
        )
        return { blob, filename: resolveFilename(exportRes) }
      }
      if (!fileId) {
        throw new Error("Export response missing file id")
      }
      const resolvedExport =
        exportInfo?.status === "pending" ? await waitForExportReady(fileId) : exportInfo
      if (!resolvedExport?.url) {
        throw new Error("Export URL missing")
      }
      return await downloadFromUrl(resolvedExport.url)
    }

    const url =
      `/api/v1/data-tables/${encodeURIComponent(tableId)}/export?format=${encodeURIComponent(
        format
      )}&download=true` as AllowedPath
    const res = await requestWithAuth(url)

    if (!res.ok) {
      const detail = await readErrorDetail(res)
      if (res.status === 422 && detail === "export_size_exceeded") {
        return await exportViaArtifact()
      }
      throw new Error(detail || `Export failed: ${res.status}`)
    }

    return await readBlobResponse(res)
  },

  // ─────────────────────────────────────────────────────────────────────────────
  // File Artifacts
  // ─────────────────────────────────────────────────────────────────────────────

  async createImageArtifact(
    request: ImageArtifactRequest,
    ensureAndRequest: <T>(init: any, requireAuth?: boolean) => Promise<T>
  ): Promise<FileCreateResponse> {
    const payload: Record<string, unknown> = {
      backend: request.backend,
      prompt: request.prompt
    }
    if (request.negativePrompt) payload.negative_prompt = request.negativePrompt
    if (typeof request.referenceFileId === "number") {
      payload.reference_file_id = request.referenceFileId
    }
    if (typeof request.width === "number") payload.width = request.width
    if (typeof request.height === "number") payload.height = request.height
    if (typeof request.steps === "number") payload.steps = request.steps
    if (typeof request.cfgScale === "number") payload.cfg_scale = request.cfgScale
    if (typeof request.seed === "number") payload.seed = request.seed
    if (request.sampler) payload.sampler = request.sampler
    if (request.model) payload.model = request.model
    if (request.extraParams) payload.extra_params = request.extraParams

    const body: Record<string, unknown> = {
      file_type: "image",
      payload,
      export: {
        format: request.format || "png",
        mode: "inline",
        async_mode: "sync"
      },
      options: {
        persist: typeof request.persist === "boolean" ? request.persist : true
      }
    }
    if (request.title) {
      body.title = request.title
    }

    return await ensureAndRequest<FileCreateResponse>({
      path: "/api/v1/files/create",
      method: "POST",
      body,
      timeoutMs: request.timeoutMs
    })
  }
}

export type MediaMethods = typeof mediaMethods
