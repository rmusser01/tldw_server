import { bgRequest } from "@/services/background-proxy"
import { getTldwTTSModel, getTldwTTSVoice } from "@/services/tts"
import { buildQuery } from "../client-utils"
import type {
  CreateReadingSavedSearchRequest,
  CreateReadingDigestScheduleRequest,
  ImportSource,
  ReadingNoteLink,
  ReadingSavedSearch,
  ReadingSavedSearchListResponse,
  ReadingDigestSchedule,
  ReadingImportJobResponse,
  ReadingImportJobStatus,
  ReadingImportJobsListResponse,
  UpdateReadingSavedSearchRequest,
  UpdateReadingDigestScheduleRequest
} from "@/types/collections"
import type {
  CreateIngestionSourceRequest,
  IngestionSourceItem,
  IngestionSourceItemFilters,
  IngestionSourceItemsListResponse,
  IngestionSourceListResponse,
  IngestionSourceSummary,
  IngestionSourceSyncTriggerResponse,
  UpdateIngestionSourceRequest
} from "@/types/ingestion-sources"
import {
  normalizeIngestionSource,
  normalizeIngestionSourceItem,
  normalizeIngestionSourceItemsListResponse,
  normalizeIngestionSourceListResponse,
  normalizeIngestionSourceSyncTrigger,
  normalizeReadingDigestSchedule
} from "../collections-normalizers"

export interface TldwApiClientCore {
  resolveApiPath(key: string, candidates: string[]): Promise<string>
  fillPathParams(template: string, values: string | string[]): string
  ensureConfigForRequest(requireAuth: boolean): Promise<any>
  request<T>(init: any, requireAuth?: boolean): Promise<T>
  upload<T>(init: any, requireAuth?: boolean): Promise<T>
  listNotes(
    params?: {
      page?: number
      results_per_page?: number
      limit?: number
      offset?: number
      include_keywords?: boolean
    },
    options?: { signal?: AbortSignal }
  ): Promise<any>
}

export type PromptPayload = {
  name?: string
  title?: string
  author?: string
  details?: string
  system_prompt?: string | null
  user_prompt?: string | null
  keywords?: string[]
  content?: string
  is_system?: boolean
}

export const collectionsMethods = {
  // ── Notes ──

  async createNote(content: string, metadata?: any): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/notes/",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { content, ...metadata }
    })
  },

  async listNotes(
    this: TldwApiClientCore,
    params?: {
      page?: number
      results_per_page?: number
      limit?: number
      offset?: number
      include_keywords?: boolean
    },
    options?: { signal?: AbortSignal }
  ): Promise<any> {
    const limit = params?.limit ?? params?.results_per_page
    const offset =
      params?.offset ??
      (params?.page != null && limit != null
        ? Math.max(0, (params.page - 1) * limit)
        : undefined)
    const query = buildQuery({
      limit,
      offset,
      include_keywords: params?.include_keywords
    } as Record<string, any>)
    return await bgRequest<any>({
      path: `/api/v1/notes/${query}`,
      method: "GET",
      abortSignal: options?.signal
    })
  },

  async searchNotes(this: TldwApiClientCore, query: string): Promise<any> {
    const normalized = query.trim()
    if (!normalized) {
      return await this.listNotes()
    }
    const queryString = buildQuery({
      query: normalized
    })
    return await bgRequest<any>({
      path: `/api/v1/notes/search/${queryString}`,
      method: "GET"
    })
  },

  // ── Prompts ──

  async getPrompts(this: TldwApiClientCore): Promise<any> {
    const path = await this.resolveApiPath("prompts.list", [
      "/api/v1/prompts",
      "/api/v1/prompts/"
    ])
    return await bgRequest<any>({ path, method: "GET" })
  },

  async searchPrompts(query: string): Promise<any> {
    // TODO: confirm trailing slash per OpenAPI (`/api/v1/prompts/search` exists without slash)
    return await bgRequest<any>({
      path: "/api/v1/prompts/search",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { query }
    })
  },

  async createPrompt(this: TldwApiClientCore, payload: PromptPayload): Promise<any> {
    const name = payload.name || payload.title || "Untitled"
    const system_prompt = payload.system_prompt ?? (payload.is_system ? payload.content : undefined)
    const user_prompt = payload.user_prompt ?? (!payload.is_system ? payload.content : undefined)
    const keywords = payload.keywords
    const normalized: Record<string, any> = {
      name,
      author: payload.author,
      details: payload.details,
      system_prompt,
      user_prompt,
      keywords
    }

    Object.keys(normalized).forEach((key) => {
      if (typeof normalized[key] === "undefined") delete normalized[key]
    })

    const path = await this.resolveApiPath("prompts.create", [
      "/api/v1/prompts",
      "/api/v1/prompts/"
    ])
    return await bgRequest<any>({
      path,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: normalized
    })
  },

  async updatePrompt(
    this: TldwApiClientCore,
    id: string | number,
    payload: PromptPayload
  ): Promise<any> {
    const pid = String(id)
    const name = payload.name || payload.title || "Untitled"
    const system_prompt = payload.system_prompt ?? (payload.is_system ? payload.content : undefined)
    const user_prompt = payload.user_prompt ?? (!payload.is_system ? payload.content : undefined)
    const keywords = payload.keywords

    const normalized: Record<string, any> = {
      name,
      author: payload.author,
      details: payload.details,
      system_prompt,
      user_prompt,
      keywords
    }

    Object.keys(normalized).forEach((key) => {
      if (typeof normalized[key] === "undefined") delete normalized[key]
    })

    // Path per OpenAPI: /api/v1/prompts/{prompt_identifier}
    return await bgRequest<any>({
      path: `/api/v1/prompts/${pid}`,
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: normalized
    })
  },

  // ── Items ──

  async getItems(params?: {
    page?: number
    size?: number
    q?: string
    status_filter?: string | string[]
    tags?: string[]
    favorite?: boolean
    domain?: string
    date_from?: string
    date_to?: string
    origin?: string
    job_id?: number
    run_id?: number
  }): Promise<any> {
    const query = new URLSearchParams()
    if (params?.page) query.set("page", String(params.page))
    if (params?.size) query.set("size", String(params.size))
    if (params?.q) query.set("q", params.q)
    if (params?.status_filter) {
      const statuses = Array.isArray(params.status_filter)
        ? params.status_filter
        : [params.status_filter]
      statuses.filter(Boolean).forEach((status) => query.append("status_filter", status))
    }
    if (params?.tags?.length) params.tags.forEach((tag) => query.append("tags", tag))
    if (params?.favorite !== undefined) query.set("favorite", String(params.favorite))
    if (params?.domain) query.set("domain", params.domain)
    if (params?.date_from) query.set("date_from", params.date_from)
    if (params?.date_to) query.set("date_to", params.date_to)
    if (params?.origin) query.set("origin", params.origin)
    if (params?.job_id !== undefined) query.set("job_id", String(params.job_id))
    if (params?.run_id !== undefined) query.set("run_id", String(params.run_id))
    const qs = query.toString()
    const path = `/api/v1/items${qs ? `?${qs}` : ""}` as const
    const data = await bgRequest<any>({ path, method: "GET" })
    const items = Array.isArray(data?.items)
      ? data.items.map((item: any) => ({
          ...item,
          id: String(item?.id),
          content_item_id:
            item?.content_item_id === null || typeof item?.content_item_id === "undefined"
              ? undefined
              : String(item.content_item_id),
          media_id:
            item?.media_id === null || typeof item?.media_id === "undefined"
              ? undefined
              : String(item.media_id),
          title: item?.title || item?.url || "Untitled",
          tags: Array.isArray(item?.tags) ? item.tags : []
        }))
      : []
    return {
      ...data,
      items,
      total: data?.total ?? items.length,
      page: data?.page ?? params?.page ?? 1,
      size: data?.size ?? params?.size ?? items.length
    }
  },

  async bulkUpdateItems(data: {
    item_ids: string[]
    action: "set_status" | "set_favorite" | "add_tags" | "remove_tags" | "replace_tags" | "delete"
    status?: string
    favorite?: boolean
    tags?: string[]
    hard?: boolean
  }): Promise<{
    total: number
    succeeded: number
    failed: number
    results: Array<{ item_id: string; success: boolean; error?: string | null }>
  }> {
    const itemIds = (data.item_ids || [])
      .map((id) => Number(id))
      .filter((id) => Number.isFinite(id) && id > 0)
      .map((id) => Math.floor(id))
    if (itemIds.length === 0) {
      throw new Error("item_ids_required")
    }

    const response = await bgRequest<any>({
      path: "/api/v1/items/bulk",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        item_ids: itemIds,
        action: data.action,
        status: data.status,
        favorite: data.favorite,
        tags: data.tags,
        hard: data.hard
      }
    })

    const results = Array.isArray(response?.results)
      ? response.results.map((entry: any) => ({
          item_id: String(entry?.item_id),
          success: Boolean(entry?.success),
          error: entry?.error ?? null
        }))
      : []

    return {
      total: response?.total ?? itemIds.length,
      succeeded: response?.succeeded ?? results.filter((entry) => entry.success).length,
      failed: response?.failed ?? results.filter((entry) => !entry.success).length,
      results
    }
  },

  // ── Reading List ──

  async getReadingList(params?: {
    page?: number
    size?: number
    q?: string
    status?: string | string[]
    tags?: string[]
    favorite?: boolean
    sort?: string
    domain?: string
    date_from?: string
    date_to?: string
  }): Promise<any> {
    const query = new URLSearchParams()
    if (params?.page) query.set("page", String(params.page))
    if (params?.size) query.set("size", String(params.size))
    if (params?.q) query.set("q", params.q)
    if (params?.status) {
      const statuses = Array.isArray(params.status) ? params.status : [params.status]
      statuses.filter(Boolean).forEach((status) => query.append("status", status))
    }
    if (params?.tags?.length) params.tags.forEach((tag) => query.append("tags", tag))
    if (params?.favorite !== undefined) query.set("favorite", String(params.favorite))
    if (params?.sort) query.set("sort", params.sort)
    if (params?.domain) query.set("domain", params.domain)
    if (params?.date_from) query.set("date_from", params.date_from)
    if (params?.date_to) query.set("date_to", params.date_to)
    const qs = query.toString()
    const path = `/api/v1/reading/items${qs ? `?${qs}` : ""}` as const
    const data = await bgRequest<any>({ path, method: "GET" })
    const items = Array.isArray(data?.items)
      ? data.items.map((item: any) => ({
          id: String(item.id),
          title: item.title || item.url || "Untitled",
          url: item.url,
          canonical_url: item.canonical_url,
          domain: item.domain,
          summary: item.summary ?? undefined,
          notes: item.notes ?? undefined,
          status: item.status ?? "saved",
          favorite: Boolean(item.favorite),
          tags: Array.isArray(item.tags) ? item.tags : [],
          reading_time_minutes: item.reading_time_minutes,
          created_at: item.created_at,
          updated_at: item.updated_at,
          published_at: item.published_at
        }))
      : []
    return {
      ...data,
      items,
      total: data?.total ?? items.length,
      page: data?.page ?? params?.page ?? 1,
      size: data?.size ?? params?.size ?? items.length
    }
  },

  async getReadingItem(itemId: string): Promise<any> {
    const path = `/api/v1/reading/items/${encodeURIComponent(itemId)}` as const
    const item = await bgRequest<any>({ path, method: "GET" })
    return {
      ...item,
      id: String(item?.id),
      media_id: item?.media_id ? String(item.media_id) : undefined,
      favorite: Boolean(item?.favorite),
      tags: Array.isArray(item?.tags) ? item.tags : []
    }
  },

  async addReadingItem(data: {
    url: string
    title?: string
    tags?: string[]
    notes?: string
    archive_mode?: "use_default" | "always" | "never"
    status?: string
    favorite?: boolean
    summary?: string
    content?: string
  }): Promise<any> {
    return await bgRequest<any>({
      path: "/api/v1/reading/save",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: data
    })
  },

  async updateReadingItem(
    itemId: string,
    data: {
      status?: string
      favorite?: boolean
      tags?: string[]
      notes?: string
      title?: string
    }
  ): Promise<any> {
    const path = `/api/v1/reading/items/${encodeURIComponent(itemId)}` as const
    return await bgRequest<any>({
      path,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: data
    })
  },

  async bulkUpdateReadingItems(data: {
    item_ids: string[]
    action: "set_status" | "set_favorite" | "add_tags" | "remove_tags" | "replace_tags" | "delete"
    status?: string
    favorite?: boolean
    tags?: string[]
    hard?: boolean
  }): Promise<{
    total: number
    succeeded: number
    failed: number
    results: Array<{ item_id: string; success: boolean; error?: string | null }>
  }> {
    const itemIds = (data.item_ids || [])
      .map((id) => Number(id))
      .filter((id) => Number.isFinite(id) && id > 0)
      .map((id) => Math.floor(id))
    if (itemIds.length === 0) {
      throw new Error("item_ids_required")
    }

    const response = await bgRequest<any>({
      path: "/api/v1/reading/items/bulk",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        item_ids: itemIds,
        action: data.action,
        status: data.status,
        favorite: data.favorite,
        tags: data.tags,
        hard: data.hard
      }
    })

    const results = Array.isArray(response?.results)
      ? response.results.map((entry: any) => ({
          item_id: String(entry?.item_id),
          success: Boolean(entry?.success),
          error: entry?.error ?? null
        }))
      : []

    return {
      total: response?.total ?? itemIds.length,
      succeeded: response?.succeeded ?? results.filter((entry) => entry.success).length,
      failed: response?.failed ?? results.filter((entry) => !entry.success).length,
      results
    }
  },

  async deleteReadingItem(itemId: string, options?: { hard?: boolean }): Promise<void> {
    const query = new URLSearchParams()
    if (options?.hard !== undefined) query.set("hard", String(options.hard))
    const qs = query.toString()
    const path = `/api/v1/reading/items/${encodeURIComponent(itemId)}${qs ? `?${qs}` : ""}` as const
    await bgRequest<void>({ path, method: "DELETE" })
  },

  // ── Saved Searches ──

  async createReadingSavedSearch(
    data: CreateReadingSavedSearchRequest
  ): Promise<ReadingSavedSearch> {
    const row = await bgRequest<any>({
      path: "/api/v1/reading/saved-searches",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: data
    })
    return {
      id: String(row?.id ?? ""),
      name: String(row?.name ?? ""),
      query:
        row?.query && typeof row.query === "object" && !Array.isArray(row.query)
          ? row.query
          : {},
      sort: row?.sort ?? undefined,
      created_at: row?.created_at ?? undefined,
      updated_at: row?.updated_at ?? undefined
    }
  },

  async listReadingSavedSearches(
    params?: { limit?: number; offset?: number }
  ): Promise<ReadingSavedSearchListResponse> {
    const query = new URLSearchParams()
    if (typeof params?.limit === "number" && Number.isFinite(params.limit)) {
      query.set("limit", String(Math.max(1, Math.floor(params.limit))))
    }
    if (typeof params?.offset === "number" && Number.isFinite(params.offset)) {
      query.set("offset", String(Math.max(0, Math.floor(params.offset))))
    }
    const qs = query.toString()
    const path = `/api/v1/reading/saved-searches${qs ? `?${qs}` : ""}` as const
    const data = await bgRequest<any>({ path, method: "GET" })
    const items: ReadingSavedSearch[] = Array.isArray(data?.items)
      ? data.items.map((row: any) => ({
          id: String(row?.id ?? ""),
          name: String(row?.name ?? ""),
          query:
            row?.query && typeof row.query === "object" && !Array.isArray(row.query)
              ? row.query
              : {},
          sort: row?.sort ?? undefined,
          created_at: row?.created_at ?? undefined,
          updated_at: row?.updated_at ?? undefined
        }))
      : []
    return {
      items,
      total: Number.isFinite(data?.total) ? Number(data.total) : items.length,
      limit:
        Number.isFinite(data?.limit) && Number(data.limit) > 0
          ? Number(data.limit)
          : params?.limit ?? 50,
      offset:
        Number.isFinite(data?.offset) && Number(data.offset) >= 0
          ? Number(data.offset)
          : params?.offset ?? 0
    }
  },

  async updateReadingSavedSearch(
    searchId: string,
    data: UpdateReadingSavedSearchRequest
  ): Promise<ReadingSavedSearch> {
    const path = `/api/v1/reading/saved-searches/${encodeURIComponent(searchId)}` as const
    const row = await bgRequest<any>({
      path,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: data
    })
    return {
      id: String(row?.id ?? ""),
      name: String(row?.name ?? ""),
      query:
        row?.query && typeof row.query === "object" && !Array.isArray(row.query)
          ? row.query
          : {},
      sort: row?.sort ?? undefined,
      created_at: row?.created_at ?? undefined,
      updated_at: row?.updated_at ?? undefined
    }
  },

  async deleteReadingSavedSearch(searchId: string): Promise<{ ok: boolean }> {
    const path = `/api/v1/reading/saved-searches/${encodeURIComponent(searchId)}` as const
    const response = await bgRequest<{ ok?: boolean }>({ path, method: "DELETE" })
    return { ok: Boolean(response?.ok) }
  },

  // ── Note Links ──

  async linkReadingItemToNote(itemId: string, noteId: string): Promise<ReadingNoteLink> {
    const path = `/api/v1/reading/items/${encodeURIComponent(itemId)}/links/note` as const
    const row = await bgRequest<any>({
      path,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: { note_id: noteId }
    })
    return {
      item_id: String(row?.item_id ?? itemId),
      note_id: String(row?.note_id ?? noteId),
      created_at: row?.created_at ?? undefined
    }
  },

  async listReadingItemNoteLinks(itemId: string): Promise<ReadingNoteLink[]> {
    const path = `/api/v1/reading/items/${encodeURIComponent(itemId)}/links` as const
    const data = await bgRequest<any>({ path, method: "GET" })
    if (!Array.isArray(data?.links)) {
      return []
    }
    return data.links.map((row: any) => ({
      item_id: String(row?.item_id ?? itemId),
      note_id: String(row?.note_id ?? ""),
      created_at: row?.created_at ?? undefined
    }))
  },

  async unlinkReadingItemNote(itemId: string, noteId: string): Promise<{ ok: boolean }> {
    const path = `/api/v1/reading/items/${encodeURIComponent(itemId)}/links/note/${encodeURIComponent(noteId)}` as const
    const response = await bgRequest<{ ok?: boolean }>({ path, method: "DELETE" })
    return { ok: Boolean(response?.ok) }
  },

  // ── Summarize & TTS ──

  async summarizeReadingItem(
    itemId: string,
    options?: {
      provider?: string
      model?: string
      prompt?: string
      system_prompt?: string
      temperature?: number
      recursive?: boolean
      chunked?: boolean
    }
  ): Promise<{ summary: string; provider: string; model?: string }> {
    const path = `/api/v1/reading/items/${encodeURIComponent(itemId)}/summarize` as const
    return await bgRequest<any>({
      path,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: options || {}
    })
  },

  async generateReadingItemTts(
    itemId: string,
    options?: { model?: string; voice?: string }
  ): Promise<{ audio_url: string }> {
    const path = `/api/v1/reading/items/${encodeURIComponent(itemId)}/tts` as const
    const model = options?.model || (await getTldwTTSModel())
    const voice = options?.voice || (await getTldwTTSVoice())
    const data = await bgRequest<ArrayBuffer>({
      path,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        model,
        voice,
        response_format: "mp3",
        stream: false,
      },
      responseType: "arrayBuffer"
    })
    const blob = new Blob([data], { type: "audio/mpeg" })
    return { audio_url: URL.createObjectURL(blob) }
  },

  // ── Highlights ──

  async getHighlights(itemId: string): Promise<any[]> {
    const path = `/api/v1/reading/items/${encodeURIComponent(itemId)}/highlights` as const
    const data = await bgRequest<any>({ path, method: "GET" })
    return Array.isArray(data)
      ? data.map((highlight) => ({
          ...highlight,
          id: String(highlight.id),
          item_id: String(highlight.item_id),
          color: highlight.color || "yellow",
          state: highlight.state || "active",
          anchor_strategy: highlight.anchor_strategy || "fuzzy_quote"
        }))
      : []
  },

  async createHighlight(data: {
    item_id: string
    quote: string
    note?: string
    color?: string
    start_offset?: number
    end_offset?: number
    anchor_strategy?: string
  }): Promise<any> {
    const path = `/api/v1/reading/items/${encodeURIComponent(data.item_id)}/highlight` as const
    const payload = {
      item_id: Number(data.item_id),
      quote: data.quote,
      note: data.note,
      color: data.color,
      start_offset: data.start_offset,
      end_offset: data.end_offset,
      anchor_strategy: data.anchor_strategy
    }
    const highlight = await bgRequest<any>({
      path,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
    return {
      ...highlight,
      id: String(highlight.id),
      item_id: String(highlight.item_id),
      color: highlight.color || "yellow",
      state: highlight.state || "active",
      anchor_strategy: highlight.anchor_strategy || "fuzzy_quote"
    }
  },

  async updateHighlight(
    highlightId: string,
    data: { note?: string; color?: string; state?: string }
  ): Promise<any> {
    const path = `/api/v1/reading/highlights/${encodeURIComponent(highlightId)}` as const
    const highlight = await bgRequest<any>({
      path,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: data
    })
    return {
      ...highlight,
      id: String(highlight.id),
      item_id: String(highlight.item_id),
      color: highlight.color || "yellow",
      state: highlight.state || "active",
      anchor_strategy: highlight.anchor_strategy || "fuzzy_quote"
    }
  },

  async deleteHighlight(highlightId: string): Promise<void> {
    const path = `/api/v1/reading/highlights/${encodeURIComponent(highlightId)}` as const
    await bgRequest<void>({ path, method: "DELETE" })
  },

  // ── Output Templates ──

  async getOutputTemplates(params?: {
    q?: string
    limit?: number
    offset?: number
  }): Promise<any> {
    const query = new URLSearchParams()
    if (params?.q) query.set("q", params.q)
    if (params?.limit) query.set("limit", String(params.limit))
    if (params?.offset !== undefined) query.set("offset", String(params.offset))
    const qs = query.toString()
    const path = `/api/v1/outputs/templates${qs ? `?${qs}` : ""}` as const
    const data = await bgRequest<any>({ path, method: "GET" })
    const items = Array.isArray(data?.items)
      ? data.items.map((template: any) => ({
          ...template,
          id: String(template.id)
        }))
      : []
    return {
      ...data,
      items,
      total: data?.total ?? items.length
    }
  },

  async createOutputTemplate(data: {
    name: string
    description?: string
    type: string
    format: string
    body: string
    is_default?: boolean
  }): Promise<any> {
    const template = await bgRequest<any>({
      path: "/api/v1/outputs/templates",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: data
    })
    return { ...template, id: String(template.id) }
  },

  async updateOutputTemplate(
    templateId: string,
    data: {
      name?: string
      description?: string
      body?: string
      is_default?: boolean
      type?: string
      format?: string
    }
  ): Promise<any> {
    const path = `/api/v1/outputs/templates/${encodeURIComponent(templateId)}` as const
    const template = await bgRequest<any>({
      path,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: data
    })
    return { ...template, id: String(template.id) }
  },

  async deleteOutputTemplate(templateId: string): Promise<void> {
    const path = `/api/v1/outputs/templates/${encodeURIComponent(templateId)}` as const
    await bgRequest<void>({ path, method: "DELETE" })
  },

  async previewTemplate(data: {
    template_id: string
    item_ids?: string[]
    run_id?: string
    limit?: number
    data?: Record<string, unknown>
  }): Promise<{ rendered: string; format: string }> {
    const path = `/api/v1/outputs/templates/${encodeURIComponent(data.template_id)}/preview` as const
    return await bgRequest<any>({
      path,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        template_id: Number(data.template_id),
        item_ids: data.item_ids?.map((id) => Number(id)),
        run_id: data.run_id ? Number(data.run_id) : undefined,
        limit: data.limit,
        data: data.data
      }
    })
  },

  // ── Outputs ──

  async listOutputs(
    this: TldwApiClientCore,
    params?: {
      page?: number
      size?: number
      job_id?: number
      run_id?: number
      type?: string
      workspace_tag?: string
      include_deleted?: boolean
    }
  ): Promise<{ items: any[]; total: number; page?: number; size?: number }> {
    const query = buildQuery(params as Record<string, any>)
    const response = await bgRequest<any>({
      path: `/api/v1/outputs${query}`,
      method: "GET"
    })
    const items = Array.isArray(response?.items)
      ? response.items.map((item: any) => ({
          ...item,
          id: String(item.id),
          media_item_id:
            item.media_item_id === null || typeof item.media_item_id === "undefined"
              ? undefined
              : String(item.media_item_id)
        }))
      : []
    return {
      ...response,
      items,
      total: response?.total ?? items.length
    }
  },

  async generateOutput(data: {
    template_id: string
    item_ids?: string[]
    run_id?: string
    title?: string
    workspace_tag?: string
    data?: Record<string, unknown>
  }): Promise<any> {
    const output = await bgRequest<any>({
      path: "/api/v1/outputs",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        template_id: Number(data.template_id),
        item_ids: data.item_ids?.map((id) => Number(id)),
        run_id: data.run_id ? Number(data.run_id) : undefined,
        title: data.title,
        workspace_tag: data.workspace_tag,
        data: data.data
      }
    })
    return { ...output, id: String(output.id) }
  },

  async downloadOutput(outputId: string, format?: string): Promise<Blob> {
    const path = `/api/v1/outputs/${encodeURIComponent(outputId)}/download` as const
    const data = await bgRequest<ArrayBuffer>({
      path,
      method: "GET",
      responseType: "arrayBuffer"
    })
    const mime =
      format === "html"
        ? "text/html"
        : format === "md"
          ? "text/markdown"
          : format === "mp3"
            ? "audio/mpeg"
            : "application/octet-stream"
    return new Blob([data], { type: mime })
  },

  // ── Ingestion Sources ──

  async listIngestionSources(this: TldwApiClientCore): Promise<IngestionSourceListResponse> {
    const response = await this.request<any>({
      path: "/api/v1/ingestion-sources",
      method: "GET"
    })
    return normalizeIngestionSourceListResponse(response)
  },

  async getIngestionSource(this: TldwApiClientCore, sourceId: string): Promise<IngestionSourceSummary> {
    const encodedSourceId = encodeURIComponent(sourceId)
    const response = await this.request<any>({
      path: `/api/v1/ingestion-sources/${encodedSourceId}`,
      method: "GET"
    })
    return normalizeIngestionSource(response)
  },

  async listIngestionSourceItems(
    this: TldwApiClientCore,
    sourceId: string,
    filters?: IngestionSourceItemFilters
  ): Promise<IngestionSourceItemsListResponse> {
    const encodedSourceId = encodeURIComponent(sourceId)
    const query = buildQuery(filters as Record<string, any> | undefined)
    const response = await this.request<any>({
      path: `/api/v1/ingestion-sources/${encodedSourceId}/items${query}`,
      method: "GET"
    })
    return normalizeIngestionSourceItemsListResponse(response)
  },

  async createIngestionSource(
    this: TldwApiClientCore,
    payload: CreateIngestionSourceRequest
  ): Promise<IngestionSourceSummary> {
    const response = await this.request<any>({
      path: "/api/v1/ingestion-sources",
      method: "POST",
      body: payload
    })
    return normalizeIngestionSource(response)
  },

  async updateIngestionSource(
    this: TldwApiClientCore,
    sourceId: string,
    payload: UpdateIngestionSourceRequest
  ): Promise<IngestionSourceSummary> {
    const encodedSourceId = encodeURIComponent(sourceId)
    const response = await this.request<any>({
      path: `/api/v1/ingestion-sources/${encodedSourceId}`,
      method: "PATCH",
      body: payload
    })
    return normalizeIngestionSource(response)
  },

  async syncIngestionSource(
    this: TldwApiClientCore,
    sourceId: string
  ): Promise<IngestionSourceSyncTriggerResponse> {
    const encodedSourceId = encodeURIComponent(sourceId)
    const response = await this.request<any>({
      path: `/api/v1/ingestion-sources/${encodedSourceId}/sync`,
      method: "POST"
    })
    return normalizeIngestionSourceSyncTrigger(response)
  },

  async uploadIngestionSourceArchive(
    this: TldwApiClientCore,
    sourceId: string,
    file: File
  ): Promise<IngestionSourceSyncTriggerResponse> {
    const encodedSourceId = encodeURIComponent(sourceId)
    const data = await file.arrayBuffer()
    const response = await this.upload<any>({
      path: `/api/v1/ingestion-sources/${encodedSourceId}/archive`,
      method: "POST",
      fileFieldName: "archive",
      file: {
        name: file.name || "archive-upload",
        type: file.type || "application/octet-stream",
        data
      }
    })
    return normalizeIngestionSourceSyncTrigger(response)
  },

  async reattachIngestionSourceItem(
    this: TldwApiClientCore,
    sourceId: string,
    itemId: string
  ): Promise<IngestionSourceItem> {
    const encodedSourceId = encodeURIComponent(sourceId)
    const encodedItemId = encodeURIComponent(itemId)
    const response = await this.request<any>({
      path: `/api/v1/ingestion-sources/${encodedSourceId}/items/${encodedItemId}/reattach`,
      method: "POST"
    })
    return normalizeIngestionSourceItem(response)
  },

  // ── Import/Export ──

  async importReadingList(
    this: TldwApiClientCore,
    data: {
      source: ImportSource
      file: File
      merge_tags?: boolean
    }
  ): Promise<ReadingImportJobResponse> {
    const buffer = await data.file.arrayBuffer()
    const fileData = Array.from(new Uint8Array(buffer))
    return await this.upload<ReadingImportJobResponse>({
      path: "/api/v1/reading/import",
      method: "POST",
      fileFieldName: "file",
      file: {
        name: data.file.name,
        type: data.file.type || "application/octet-stream",
        data: fileData
      },
      fields: {
        source: data.source,
        merge_tags: data.merge_tags ?? true
      }
    })
  },

  async listReadingImportJobs(params?: {
    status?: string
    limit?: number
    offset?: number
  }): Promise<ReadingImportJobsListResponse> {
    const query = buildQuery(params as Record<string, any>)
    return await bgRequest<ReadingImportJobsListResponse>({
      path: `/api/v1/reading/import/jobs${query}`,
      method: "GET"
    })
  },

  async getReadingImportJob(job_id: number | string): Promise<ReadingImportJobStatus> {
    const id = String(job_id)
    return await bgRequest<ReadingImportJobStatus>({
      path: `/api/v1/reading/import/jobs/${id}`,
      method: "GET"
    })
  },

  async exportReadingList(params: {
    format: string
    status?: string[]
    tags?: string[]
    favorite?: boolean
    q?: string
    domain?: string
    page?: number
    size?: number
    include_highlights?: boolean
    include_notes?: boolean
  }): Promise<{ blob: Blob; filename: string }> {
    const query = new URLSearchParams()
    query.set("format", params.format)
    if (params?.status?.length) params.status.forEach((status) => query.append("status", status))
    if (params?.tags?.length) params.tags.forEach((tag) => query.append("tags", tag))
    if (params?.favorite !== undefined) query.set("favorite", String(params.favorite))
    if (params?.q) query.set("q", params.q)
    if (params?.domain) query.set("domain", params.domain)
    if (params?.page) query.set("page", String(params.page))
    if (params?.size) query.set("size", String(params.size))
    if (params?.include_highlights !== undefined) {
      query.set("include_highlights", String(params.include_highlights))
    }
    if (params?.include_notes !== undefined) {
      query.set("include_notes", String(params.include_notes))
    }
    const qs = query.toString()
    const path = `/api/v1/reading/export${qs ? `?${qs}` : ""}` as const
    const response = await bgRequest<any>({
      path,
      method: "GET",
      responseType: "arrayBuffer",
      returnResponse: true
    })
    if (!response) {
      throw new Error("Export failed")
    }
    if (!response.ok) {
      const msg = response.error || `Export failed: ${response.status}`
      throw new Error(msg)
    }
    const headers = new Headers(response.headers || {})
    const contentDisposition = headers.get("content-disposition") || ""
    const filenameMatch = /filename="?([^"]+)"?/i.exec(contentDisposition)
    const filename = filenameMatch?.[1] || "reading_export.jsonl"
    const blob = new Blob([response.data], {
      type: headers.get("content-type") || "application/octet-stream"
    })
    return { blob, filename }
  },

  // ── Digest Schedules ──

  async createReadingDigestSchedule(
    data: CreateReadingDigestScheduleRequest
  ): Promise<{ id: string }> {
    const payload: Record<string, unknown> = { ...data }
    if (!payload.format) payload.format = "md"
    if (typeof payload.enabled !== "boolean") payload.enabled = true
    if (typeof payload.require_online !== "boolean") payload.require_online = false
    const response = await bgRequest<{ id: string }>({
      path: "/api/v1/reading/digests/schedules",
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: payload
    })
    return { id: String(response?.id ?? "") }
  },

  async listReadingDigestSchedules(params?: {
    limit?: number
    offset?: number
  }): Promise<ReadingDigestSchedule[]> {
    const query = new URLSearchParams()
    if (typeof params?.limit === "number" && Number.isFinite(params.limit)) {
      query.set("limit", String(Math.max(1, Math.floor(params.limit))))
    }
    if (typeof params?.offset === "number" && Number.isFinite(params.offset)) {
      query.set("offset", String(Math.max(0, Math.floor(params.offset))))
    }
    const qs = query.toString()
    const path = `/api/v1/reading/digests/schedules${qs ? `?${qs}` : ""}` as const
    const rows = await bgRequest<any>({ path, method: "GET" })
    if (!Array.isArray(rows)) {
      return []
    }
    return rows.map((row) => normalizeReadingDigestSchedule(row))
  },

  async getReadingDigestSchedule(scheduleId: string): Promise<ReadingDigestSchedule> {
    const path = `/api/v1/reading/digests/schedules/${encodeURIComponent(scheduleId)}` as const
    const schedule = await bgRequest<any>({ path, method: "GET" })
    return normalizeReadingDigestSchedule(schedule)
  },

  async updateReadingDigestSchedule(
    scheduleId: string,
    data: UpdateReadingDigestScheduleRequest
  ): Promise<ReadingDigestSchedule> {
    const path = `/api/v1/reading/digests/schedules/${encodeURIComponent(scheduleId)}` as const
    const schedule = await bgRequest<any>({
      path,
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: data
    })
    return normalizeReadingDigestSchedule(schedule)
  },

  async deleteReadingDigestSchedule(scheduleId: string): Promise<{ ok: boolean }> {
    const path = `/api/v1/reading/digests/schedules/${encodeURIComponent(scheduleId)}` as const
    const response = await bgRequest<{ ok?: boolean }>({ path, method: "DELETE" })
    return { ok: Boolean(response?.ok) }
  }
}

export type CollectionsMethods = typeof collectionsMethods
