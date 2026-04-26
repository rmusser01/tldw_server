import { bgRequest, bgStream } from '@/services/background-proxy'
import { buildQuery } from '../client-utils'
import { appendPathQuery } from '../path-utils'
import { createSafeStorage } from '@/utils/safe-storage'
import { tldwRequest } from '@/services/tldw/request-core'
import type { AllowedPath } from '@/services/tldw/openapi-guard'
import type { TldwApiClientCore } from '../TldwApiClient'
import type {
  TldwConfig,
  CharacterListQueryParams,
  CharacterListQueryResponse,
  CharacterVersionEntry,
  CharacterVersionListResponse,
  CharacterVersionDiffResponse,
  PersonaProfileSummary,
  PersonaProfile,
  PersonaExemplar,
  PersonaExemplarInput,
  PersonaExemplarListOptions,
  PersonaExemplarImportInput,
  PersonaExemplarReviewInput,
} from '../TldwApiClient'
import type { PersonaBuddySummary } from '@/types/persona-buddy'
import {
  normalizePersonaProfile,
  normalizePersonaExemplar,
} from '../persona-normalizers'

const CHARACTER_CACHE_TTL_MS = 5 * 60 * 1000
type PersonaProfileWithBuddy = PersonaProfileSummary & {
  buddy_summary?: PersonaBuddySummary | null
}

export const characterMethods = {
  normalizeCharacterListResponse(this: TldwApiClientCore, payload: unknown): any[] {
    if (Array.isArray(payload)) {
      return payload
    }
    if (!payload || typeof payload !== "object") {
      return []
    }

    const objectPayload = payload as Record<string, unknown>
    const candidateLists = [
      objectPayload.items,
      objectPayload.characters,
      objectPayload.results,
      objectPayload.data
    ]

    for (const candidate of candidateLists) {
      if (Array.isArray(candidate)) {
        return candidate
      }
    }

    return []
  },

  async listCharacters(this: TldwApiClientCore, params?: Record<string, any>): Promise<any[]> {
    const query = buildQuery(params)
    const listPathCandidates = ["/api/v1/characters", "/api/v1/characters/"] as const
    const base = await this.resolveApiPath("characters.list", [...listPathCandidates])
    const requestList = async (path: string) =>
      this.normalizeCharacterListResponse(
        await bgRequest<any>({
          path: appendPathQuery(path as AllowedPath, query),
          method: "GET"
        })
      )

    try {
      return await requestList(base)
    } catch (error) {
      const candidate = error as
        | {
            status?: unknown
            response?: { status?: unknown }
            message?: unknown
            details?: unknown
          }
        | null
        | undefined
      const rawStatus = candidate?.status ?? candidate?.response?.status
      const statusCodeFromNumberLike =
        typeof rawStatus === "number"
          ? rawStatus
          : typeof rawStatus === "string"
            ? Number(rawStatus)
            : Number.NaN
      const statusCodeFromMessage = String(candidate?.message || "").match(
        /\b(301|302|307|308|404|405|422)\b/
      )
      const statusCode = Number.isFinite(statusCodeFromNumberLike)
        ? statusCodeFromNumberLike
        : statusCodeFromMessage
          ? Number(statusCodeFromMessage[1])
          : Number.NaN
      const normalizedMessage = String(candidate?.message || "").toLowerCase()
      const normalizedDetails = (() => {
        const details = candidate?.details
        if (typeof details === "string") return details.toLowerCase()
        if (details == null) return ""
        try {
          return JSON.stringify(details).toLowerCase()
        } catch {
          return String(details).toLowerCase()
        }
      })()
      const shouldTryAlternatePath =
        statusCode === 301 ||
        statusCode === 302 ||
        statusCode === 307 ||
        statusCode === 308 ||
        statusCode === 404 ||
        statusCode === 405 ||
        statusCode === 422 ||
        normalizedMessage.includes("path.character_id") ||
        normalizedMessage.includes("unable to parse string as an integer") ||
        normalizedMessage.includes('input":"query"') ||
        normalizedMessage.includes("/api/v1/characters/query") ||
        normalizedDetails.includes("path.character_id") ||
        normalizedDetails.includes("unable to parse string as an integer") ||
        normalizedDetails.includes('input":"query"') ||
        normalizedDetails.includes("/api/v1/characters/query")

      if (!shouldTryAlternatePath) {
        throw error
      }

      const alternatePath = listPathCandidates.find((path) => path !== base)
      if (!alternatePath) {
        throw error
      }

      try {
        return await requestList(alternatePath)
      } catch {
        throw error
      }
    }
  },

  async listCharactersPage(
    this: TldwApiClientCore,
    params?: CharacterListQueryParams
  ): Promise<CharacterListQueryResponse> {
    const query = buildQuery(params as Record<string, any> | undefined)
    const base = await this.resolveApiPath("characters.query", [
      "/api/v1/characters/query",
      "/api/v1/characters/query/"
    ])
    const requestedPage =
      typeof params?.page === "number" && Number.isFinite(params.page)
        ? Math.max(1, Math.floor(params.page))
        : 1
    const requestedPageSize =
      typeof params?.page_size === "number" && Number.isFinite(params.page_size)
        ? Math.max(1, Math.floor(params.page_size))
        : 25

    const buildLegacyListFallback = async (): Promise<CharacterListQueryResponse> => {
      const offset = (requestedPage - 1) * requestedPageSize
      const legacyResponse = await this.listCharacters({
        limit: requestedPageSize,
        offset,
        query: params?.query,
        tags: params?.tags,
        match_all_tags: params?.match_all_tags,
        creator: params?.creator,
        has_conversations: params?.has_conversations,
        favorite_only: params?.favorite_only,
        include_deleted: params?.include_deleted,
        deleted_only: params?.deleted_only,
        sort_by: params?.sort_by,
        sort_order: params?.sort_order,
        include_image_base64: params?.include_image_base64
      })
      const legacyCandidate = legacyResponse as
        | {
            items?: unknown
            total?: unknown
            has_more?: unknown
          }
        | null
        | undefined
      const legacyItems = Array.isArray(legacyCandidate?.items)
        ? legacyCandidate.items
        : Array.isArray(legacyResponse)
          ? legacyResponse
          : []
      const legacyHasMore =
        typeof legacyCandidate?.has_more === "boolean"
          ? legacyCandidate.has_more
          : legacyItems.length >= requestedPageSize
      const legacyTotal =
        typeof legacyCandidate?.total === "number" &&
        Number.isFinite(legacyCandidate.total)
          ? legacyCandidate.total
          : legacyHasMore
            ? offset + legacyItems.length + 1
            : offset + legacyItems.length

      return {
        items: legacyItems,
        total: legacyTotal,
        page: requestedPage,
        page_size: requestedPageSize,
        has_more: legacyHasMore
      }
    }

    const isQueryRouteConflict = (error: unknown): boolean => {
      const candidate = error as
        | {
            status?: unknown
            response?: { status?: unknown }
            message?: unknown
            details?: unknown
          }
        | null
        | undefined
      const rawStatus = candidate?.status ?? candidate?.response?.status
      const statusCodeFromNumberLike =
        typeof rawStatus === "number"
          ? rawStatus
          : typeof rawStatus === "string"
            ? Number(rawStatus)
            : Number.NaN
      const statusCodeFromMessage = String(candidate?.message || "").match(
        /\b(404|405|422)\b/
      )
      const statusCode = Number.isFinite(statusCodeFromNumberLike)
        ? statusCodeFromNumberLike
        : statusCodeFromMessage
          ? Number(statusCodeFromMessage[1])
          : Number.NaN
      const normalizedMessage = String(candidate?.message || "").toLowerCase()
      const normalizedDetails = (() => {
        const details = candidate?.details
        if (typeof details === "string") return details.toLowerCase()
        if (details == null) return ""
        try {
          return JSON.stringify(details).toLowerCase()
        } catch {
          return String(details).toLowerCase()
        }
      })()
      return (
        statusCode === 404 ||
        statusCode === 405 ||
        statusCode === 422 ||
        normalizedMessage.includes("path.character_id") ||
        normalizedMessage.includes("unable to parse string as an integer") ||
        normalizedMessage.includes('input":"query"') ||
        normalizedMessage.includes("/api/v1/characters/query") ||
        normalizedDetails.includes("path.character_id") ||
        normalizedDetails.includes("unable to parse string as an integer") ||
        normalizedDetails.includes('input":"query"') ||
        normalizedDetails.includes("/api/v1/characters/query")
      )
    }

    let response: any
    try {
      response = await bgRequest<any>({
        path: appendPathQuery(base, query),
        method: "GET"
      })
    } catch (error) {
      if (!isQueryRouteConflict(error)) {
        throw error
      }
      return await buildLegacyListFallback()
    }

    if (Array.isArray(response)) {
      return {
        items: response,
        total: response.length,
        page: Number(params?.page || 1),
        page_size: Number(params?.page_size || response.length || 25),
        has_more: false
      }
    }

    const responseLooksLikeRouteConflict =
      response &&
      typeof response === "object" &&
      !Array.isArray(response) &&
      !Array.isArray((response as any).items) &&
      (() => {
        try {
          const payload = JSON.stringify(response).toLowerCase()
          return (
            payload.includes("path.character_id") ||
            payload.includes("unable to parse string as an integer") ||
            payload.includes('input":"query"') ||
            payload.includes("/api/v1/characters/query")
          )
        } catch {
          return false
        }
      })()

    if (responseLooksLikeRouteConflict) {
      return await buildLegacyListFallback()
    }

    const items = Array.isArray(response?.items) ? response.items : []
    const total =
      typeof response?.total === "number" && Number.isFinite(response.total)
        ? response.total
        : items.length
    const page =
      typeof response?.page === "number" && Number.isFinite(response.page)
        ? response.page
        : Number(params?.page || 1)
    const pageSize =
      typeof response?.page_size === "number" &&
      Number.isFinite(response.page_size)
        ? response.page_size
        : Number(params?.page_size || 25)

    return {
      items,
      total,
      page,
      page_size: pageSize,
      has_more: Boolean(response?.has_more)
    }
  },

  getCharacterListIdentity(this: TldwApiClientCore, character: any, fallbackIndex: number): string {
    const id = character?.id ?? character?.character_id ?? character?.characterId
    if (id !== undefined && id !== null && String(id).trim().length > 0) {
      return `id:${String(id)}`
    }

    const slug = character?.slug
    if (typeof slug === "string" && slug.trim().length > 0) {
      return `slug:${slug}`
    }

    const name = character?.name ?? character?.title
    if (typeof name === "string" && name.trim().length > 0) {
      return `name:${name}`
    }

    return `idx:${fallbackIndex}`
  },

  async listAllCharacters(
    this: TldwApiClientCore,
    options?: {
      pageSize?: number
      maxPages?: number
    }
  ): Promise<any[]> {
    const requestedPageSize =
      typeof options?.pageSize === "number" && Number.isFinite(options.pageSize)
        ? Math.floor(options.pageSize)
        : 1000
    const requestedMaxPages =
      typeof options?.maxPages === "number" && Number.isFinite(options.maxPages)
        ? Math.floor(options.maxPages)
        : 20
    const pageSize = Math.min(1000, Math.max(1, requestedPageSize))
    const maxPages = Math.min(200, Math.max(1, requestedMaxPages))

    const characters: any[] = []
    const seen = new Set<string>()

    for (let pageIndex = 0; pageIndex < maxPages; pageIndex += 1) {
      const offset = pageIndex * pageSize
      const page = await this.listCharacters({ limit: pageSize, offset })
      const pageList = Array.isArray(page) ? page : []
      if (pageList.length === 0) {
        break
      }

      let addedFromPage = 0
      for (const character of pageList) {
        const identity = this.getCharacterListIdentity(
          character,
          characters.length + addedFromPage
        )
        if (seen.has(identity)) continue
        seen.add(identity)
        characters.push(character)
        addedFromPage += 1
      }

      // Stop if we reached the final partial page, or the backend ignored offset
      // and returned only already-seen entries.
      if (pageList.length < pageSize || addedFromPage === 0) {
        break
      }
    }

    return characters
  },

  async searchCharacters(this: TldwApiClientCore, query: string, params?: Record<string, any>): Promise<any[]> {
    const qp = buildQuery({ query, ...(params || {}) })
    const base = await this.resolveApiPath("characters.search", [
      "/api/v1/characters/search",
      "/api/v1/characters/search/"
    ])
    return await bgRequest<any[]>({
      path: appendPathQuery(base, qp),
      method: 'GET'
    })
  },

  async filterCharactersByTags(
    this: TldwApiClientCore,
    tags: string[],
    options?: { match_all?: boolean; limit?: number; offset?: number }
  ): Promise<any[]> {
    const qp = buildQuery({
      tags,
      ...(options || {})
    })
    const base = await this.resolveApiPath("characters.filter", [
      "/api/v1/characters/filter",
      "/api/v1/characters/filter/"
    ])
    return await bgRequest<any[]>({
      path: appendPathQuery(base, qp),
      method: 'GET'
    })
  },

  async getCharacter(this: TldwApiClientCore, id: string | number, options?: { forceRefresh?: boolean }): Promise<any> {
    const cid = String(id)
    const forceRefresh = options?.forceRefresh === true
    if (!forceRefresh) {
      const cached = this.characterCache.get(cid)
      if (cached && cached.expiresAt > Date.now()) {
        return cached.value
      }
    }
    const inFlight = this.characterInFlight.get(cid)
    if (inFlight) return inFlight

    const request = (async () => {
      try {
        const template = await this.resolveApiPath("characters.get", [
          "/api/v1/characters/{id}",
          "/api/v1/characters/{id}/"
        ])
        const path = this.fillPathParams(template, cid)
        const value = await bgRequest<any>({
          path,
          method: 'GET'
        })
        this.characterCache.set(cid, {
          value,
          expiresAt: Date.now() + CHARACTER_CACHE_TTL_MS
        })
        return value
      } finally {
        this.characterInFlight.delete(cid)
      }
    })()

    this.characterInFlight.set(cid, request)
    return request
  },

  async listCharacterVersions(
    this: TldwApiClientCore,
    id: string | number,
    options?: { limit?: number }
  ): Promise<CharacterVersionListResponse> {
    const cid = String(id)
    const query = buildQuery({
      limit: options?.limit ?? 50
    })
    const template = await this.resolveApiPath("characters.versions", [
      "/api/v1/characters/{id}/versions",
      "/api/v1/characters/{id}/versions/"
    ])
    const path = appendPathQuery(this.fillPathParams(template, cid), query)
    const response = await bgRequest<any>({
      path,
      method: "GET"
    })

    const items = Array.isArray(response?.items)
      ? response.items
      : Array.isArray(response)
        ? response
        : []
    return {
      items: items.map((item: any) => ({
        change_id:
          typeof item?.change_id === "number" && Number.isFinite(item.change_id)
            ? item.change_id
            : Number(item?.change_id || 0),
        version:
          typeof item?.version === "number" && Number.isFinite(item.version)
            ? item.version
            : Number(item?.version || 0),
        operation: String(item?.operation || "update"),
        timestamp: item?.timestamp ?? null,
        client_id: item?.client_id ?? null,
        payload:
          item?.payload && typeof item.payload === "object" && !Array.isArray(item.payload)
            ? item.payload
            : {}
      })),
      total:
        typeof response?.total === "number" && Number.isFinite(response.total)
          ? response.total
          : items.length
    }
  },

  async diffCharacterVersions(
    this: TldwApiClientCore,
    id: string | number,
    fromVersion: number,
    toVersion: number
  ): Promise<CharacterVersionDiffResponse> {
    const cid = String(id)
    const query = buildQuery({
      from_version: fromVersion,
      to_version: toVersion
    })
    const template = await this.resolveApiPath("characters.versionDiff", [
      "/api/v1/characters/{id}/versions/diff",
      "/api/v1/characters/{id}/versions/diff/"
    ])
    const path = appendPathQuery(this.fillPathParams(template, cid), query)
    const response = await bgRequest<any>({
      path,
      method: "GET"
    })

    const normalizeVersionEntry = (entry: any): CharacterVersionEntry => ({
      change_id:
        typeof entry?.change_id === "number" && Number.isFinite(entry.change_id)
          ? entry.change_id
          : Number(entry?.change_id || 0),
      version:
        typeof entry?.version === "number" && Number.isFinite(entry.version)
          ? entry.version
          : Number(entry?.version || 0),
      operation: String(entry?.operation || "update"),
      timestamp: entry?.timestamp ?? null,
      client_id: entry?.client_id ?? null,
      payload:
        entry?.payload && typeof entry.payload === "object" && !Array.isArray(entry.payload)
          ? entry.payload
          : {}
    })

    const changedFields = Array.isArray(response?.changed_fields)
      ? response.changed_fields
      : []

    return {
      character_id:
        typeof response?.character_id === "number" && Number.isFinite(response.character_id)
          ? response.character_id
          : Number(response?.character_id || 0),
      from_entry: normalizeVersionEntry(response?.from_entry),
      to_entry: normalizeVersionEntry(response?.to_entry),
      changed_fields: changedFields.map((field: any) => ({
        field: String(field?.field || ""),
        old_value: field?.old_value,
        new_value: field?.new_value
      })),
      changed_count:
        typeof response?.changed_count === "number" && Number.isFinite(response.changed_count)
          ? response.changed_count
          : changedFields.length
    }
  },

  async revertCharacter(
    this: TldwApiClientCore,
    id: string | number,
    targetVersion: number
  ): Promise<any> {
    const cid = String(id)
    const template = await this.resolveApiPath("characters.revert", [
      "/api/v1/characters/{id}/revert",
      "/api/v1/characters/{id}/revert/"
    ])
    const path = this.fillPathParams(template, cid)
    const response = await bgRequest<any>({
      path,
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: {
        target_version: targetVersion
      }
    })
    this.characterCache.delete(cid)
    return response
  },

  async createCharacter(this: TldwApiClientCore, payload: Record<string, any>): Promise<any> {
    const pathCandidates = [
      "/api/v1/characters/",
      "/api/v1/characters"
    ] as const
    const path = await this.resolveApiPath("characters.create", [...pathCandidates])
    await this.ensureConfigForRequest(true)

    const requestCreate = async (requestPath: string) =>
      await bgRequest<any>({
        path: requestPath as AllowedPath,
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: payload
      })

    const requestCreateDirect = async (
      requestPath: string
    ): Promise<any> => {
      const storage = createSafeStorage()
      const response = await tldwRequest(
        {
          path: requestPath as AllowedPath,
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: payload
        },
        {
          getConfig: () =>
            storage.get<TldwConfig>("tldwConfig").catch(() => null)
        }
      )
      if (response?.ok) {
        return response.data
      }

      const error = new Error(
        typeof response?.error === "string" && response.error.trim().length > 0
          ? response.error
          : `Request failed: ${response?.status ?? 0}`
      ) as Error & {
        status?: number
        details?: unknown
      }
      error.status = response?.status
      if (typeof response?.data !== "undefined") {
        error.details = response.data
      }
      throw error
    }

    const readErrorText = (error: unknown): string => {
      const candidate = error as
        | {
            message?: unknown
            details?: unknown
          }
        | null
        | undefined
      const message = String(candidate?.message || "")
      const details = (() => {
        const value = candidate?.details
        if (typeof value === "string") return value
        if (value == null) return ""
        try {
          return JSON.stringify(value)
        } catch {
          return String(value)
        }
      })()
      return `${message} ${details}`.toLowerCase()
    }

    const getErrorStatusCode = (error: unknown): number | null => {
      const candidate = error as
        | {
            status?: unknown
            response?: { status?: unknown }
            message?: unknown
            details?: unknown
          }
        | null
        | undefined
      const rawStatus = candidate?.status ?? candidate?.response?.status
      const statusCodeFromNumberLike =
        typeof rawStatus === "number"
          ? rawStatus
          : typeof rawStatus === "string"
            ? Number(rawStatus)
            : Number.NaN
      if (Number.isFinite(statusCodeFromNumberLike)) {
        return statusCodeFromNumberLike
      }
      const statusFromText = readErrorText(error).match(
        /\b(301|302|307|308|404|405|422)\b/
      )
      if (!statusFromText) return null
      const parsedStatus = Number(statusFromText[1])
      return Number.isFinite(parsedStatus) ? parsedStatus : null
    }

    const isExtensionTimeoutError = (error: unknown): boolean => {
      return Boolean(
        (error as { __tldwExtensionTimeout?: boolean } | null)
          ?.__tldwExtensionTimeout
      ) || readErrorText(error).includes("extension messaging timeout")
    }

    const shouldTryAlternatePath = (error: unknown): boolean => {
      const statusCode = getErrorStatusCode(error)
      if (
        statusCode === 301 ||
        statusCode === 302 ||
        statusCode === 307 ||
        statusCode === 308 ||
        statusCode === 404 ||
        statusCode === 405 ||
        statusCode === 422
      ) {
        return true
      }
      const normalizedText = readErrorText(error)
      return (
        normalizedText.includes("path.character_id") ||
        normalizedText.includes("unable to parse string as an integer") ||
        normalizedText.includes("/api/v1/characters/query")
      )
    }

    const runCreateWithTimeoutRetry = async (
      requestPath: string
    ): Promise<any> => {
      try {
        return await requestCreate(requestPath)
      } catch (error) {
        if (!isExtensionTimeoutError(error)) {
          throw error
        }
        try {
          return await requestCreate(requestPath)
        } catch (retryError) {
          if (!isExtensionTimeoutError(retryError)) {
            throw retryError
          }
          return await requestCreateDirect(requestPath)
        }
      }
    }

    try {
      return await runCreateWithTimeoutRetry(path)
    } catch (error) {
      if (!shouldTryAlternatePath(error)) {
        throw error
      }

      const alternatePath = pathCandidates.find(
        (candidate) => candidate !== path
      )
      if (!alternatePath) {
        throw error
      }

      return await runCreateWithTimeoutRetry(alternatePath)
    }
  },

  async importCharacterFile(
    this: TldwApiClientCore,
    file: File,
    options?: { allowImageOnly?: boolean }
  ): Promise<any> {
    const data = await file.arrayBuffer()
    const name = file.name || "character-card"
    const type = file.type || "application/octet-stream"
    const path = await this.resolveApiPath("characters.import", [
      "/api/v1/characters/import",
      "/api/v1/characters/import/"
    ])
    const fields = options?.allowImageOnly
      ? { allow_image_only: true }
      : undefined
    return await this.upload<any>({
      path,
      method: "POST",
      fileFieldName: "character_file",
      file: { name, type, data },
      fields
    })
  },

  async exportCharacter(
    this: TldwApiClientCore,
    id: string | number,
    options?: { format?: 'v3' | 'v2' | 'json'; includeWorldBooks?: boolean }
  ): Promise<any> {
    const cid = String(id)
    const params = new URLSearchParams()
    if (options?.format) {
      params.set('format', options.format)
    }
    if (options?.includeWorldBooks) {
      params.set('include_world_books', 'true')
    }
    const qp = params.toString() ? `?${params.toString()}` : ''
    const template = await this.resolveApiPath("characters.export", [
      "/api/v1/characters/{id}/export",
      "/api/v1/characters/{id}/export/"
    ])
    const path = appendPathQuery(this.fillPathParams(template, cid), qp)
    return await bgRequest<any>({
      path,
      method: 'GET'
    })
  },

  async updateCharacter(this: TldwApiClientCore, id: string | number, payload: Record<string, any>, expectedVersion?: number): Promise<any> {
    const cid = String(id)
    const qp = expectedVersion != null ? `?expected_version=${encodeURIComponent(String(expectedVersion))}` : ''
    const template = await this.resolveApiPath("characters.update", [
      "/api/v1/characters/{id}",
      "/api/v1/characters/{id}/"
    ])
    const path = appendPathQuery(this.fillPathParams(template, cid), qp)
    const res = await bgRequest<any>({
      path,
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: payload
    })
    this.characterCache.delete(cid)
    return res
  },

  async deleteCharacter(this: TldwApiClientCore, id: string | number, expectedVersion?: number): Promise<void> {
    const cid = String(id)
    let resolvedVersion = Number(expectedVersion)
    if (!Number.isInteger(resolvedVersion) || resolvedVersion < 0) {
      const character = await this.getCharacter(cid, { forceRefresh: true })
      const fetchedVersion = Number(character?.version)
      if (!Number.isInteger(fetchedVersion) || fetchedVersion < 0) {
        throw new Error("Character delete failed: missing expected version")
      }
      resolvedVersion = fetchedVersion
    }
    const template = await this.resolveApiPath("characters.delete", [
      "/api/v1/characters/{id}",
      "/api/v1/characters/{id}/"
    ])
    const path = appendPathQuery(
      this.fillPathParams(template, cid),
      `?expected_version=${encodeURIComponent(String(resolvedVersion))}`
    )
    await bgRequest<void>({ path, method: 'DELETE' })
    this.characterCache.delete(cid)
  },

  async restoreCharacter(this: TldwApiClientCore, id: string | number, expectedVersion: number): Promise<any> {
    const cid = String(id)
    const template = await this.resolveApiPath("characters.restore", [
      "/api/v1/characters/{id}/restore",
      "/api/v1/characters/{id}/restore/"
    ])
    const path = appendPathQuery(
      this.fillPathParams(template, cid),
      `?expected_version=${expectedVersion}`
    )
    const res = await bgRequest<any>({ path, method: 'POST' })
    this.characterCache.delete(cid)
    return res
  },

  // Character chat sessions
  async listCharacterChatSessions(this: TldwApiClientCore): Promise<any[]> {
    const path = await this.resolveApiPath("characterChatSessions.list", [
      "/api/v1/character-chat/sessions",
      "/api/v1/character_chat_sessions",
      "/api/v1/character_chat_sessions/"
    ])
    return await bgRequest<any[]>({ path, method: 'GET' })
  },

  async createCharacterChatSession(this: TldwApiClientCore, character_id: string): Promise<any> {
    const body = { character_id }
    const path = await this.resolveApiPath("characterChatSessions.create", [
      "/api/v1/character-chat/sessions",
      "/api/v1/character_chat_sessions",
      "/api/v1/character_chat_sessions/"
    ])
    return await bgRequest<any>({
      path,
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body
    })
  },

  async deleteCharacterChatSession(this: TldwApiClientCore, session_id: string | number): Promise<void> {
    const sid = String(session_id)
    const template = await this.resolveApiPath("characterChatSessions.delete", [
      "/api/v1/character-chat/sessions/{session_id}",
      "/api/v1/character_chat_sessions/{session_id}",
      "/api/v1/character_chat_sessions/{session_id}/"
    ])
    const path = this.fillPathParams(template, sid)
    await bgRequest<void>({ path, method: 'DELETE' })
  },

  // Character messages
  async listCharacterMessages(this: TldwApiClientCore, session_id: string | number): Promise<any[]> {
    const sid = String(session_id)
    const query = buildQuery({ session_id: sid })
    const template = await this.resolveApiPath("characterChatMessages.list", [
      "/api/v1/character-chat/sessions/{session_id}/messages",
      "/api/v1/character-messages",
      "/api/v1/character_messages"
    ])
    const path = template.includes("{")
      ? this.fillPathParams(template, sid)
      : appendPathQuery(template, query)
    return await bgRequest<any[]>({ path, method: 'GET' })
  },

  async sendCharacterMessage(this: TldwApiClientCore, session_id: string | number, content: string, options?: { extra?: Record<string, any> }): Promise<any> {
    const sid = String(session_id)
    const body = { content, session_id: sid, ...(options?.extra || {}) }
    const template = await this.resolveApiPath("characterChatMessages.send", [
      "/api/v1/character-chat/sessions/{session_id}/messages",
      "/api/v1/character_messages",
      "/api/v1/character-messages"
    ])
    const path = template.includes("{")
      ? this.fillPathParams(template, sid)
      : template
    return await bgRequest<any>({
      path,
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body
    })
  },

  async * streamCharacterMessage(this: TldwApiClientCore, session_id: string | number, content: string, options?: { extra?: Record<string, any> }): AsyncGenerator<any> {
    const sid = String(session_id)
    const body = { content, session_id: sid, ...(options?.extra || {}) }
    const template = await this.resolveApiPath("characterChatMessages.stream", [
      "/api/v1/character-chat/sessions/{session_id}/messages/stream",
      "/api/v1/character_messages/stream",
      "/api/v1/character-messages/stream"
    ])
    const path = this.fillPathParams(template, sid)
    for await (const line of bgStream({ path, method: 'POST', headers: { 'Content-Type': 'application/json' }, body })) {
      try { yield JSON.parse(line) } catch {}
    }
  },

  async listPersonaProfiles(this: TldwApiClientCore): Promise<PersonaProfileSummary[]> {
    const payload = await this.request<any>({
      path: "/api/v1/persona/catalog",
      method: "GET"
    })
    const list = Array.isArray(payload) ? payload : []
    return list.map((item) =>
      normalizePersonaProfile(item as PersonaProfileWithBuddy)
    )
  },

  async getPersonaProfile(this: TldwApiClientCore, id: string | number): Promise<PersonaProfile> {
    const personaId = encodeURIComponent(String(id))
    const payload = await this.request<any>({
      path: `/api/v1/persona/profiles/${personaId}`,
      method: "GET"
    })
    return normalizePersonaProfile(
      payload as PersonaProfileWithBuddy | null | undefined
    )
  },

  async listPersonaExemplars(
    this: TldwApiClientCore,
    personaId: string | number,
    options?: PersonaExemplarListOptions
  ): Promise<PersonaExemplar[]> {
    const encodedPersonaId = encodeURIComponent(String(personaId))
    const query = new URLSearchParams()
    if (options?.includeDisabled) query.set("include_disabled", "true")
    if (options?.includeDeleted) query.set("include_deleted", "true")
    if (options?.includeDeletedPersonas) {
      query.set("include_deleted_personas", "true")
    }
    const payload = await this.request<any>({
      path: appendPathQuery(
        `/api/v1/persona/profiles/${encodedPersonaId}/exemplars`,
        query.toString() ? `?${query.toString()}` : ""
      ),
      method: "GET"
    })
    const list = Array.isArray(payload)
      ? payload
      : Array.isArray(payload?.items)
        ? payload.items
        : []
    return list.map((item) =>
      normalizePersonaExemplar(item as Record<string, unknown>)
    )
  },

  async createPersonaExemplar(
    this: TldwApiClientCore,
    personaId: string | number,
    payload: PersonaExemplarInput
  ): Promise<PersonaExemplar> {
    const encodedPersonaId = encodeURIComponent(String(personaId))
    const response = await this.request<any>({
      path: `/api/v1/persona/profiles/${encodedPersonaId}/exemplars`,
      method: "POST",
      body: payload
    })
    return normalizePersonaExemplar(response as Record<string, unknown>)
  },

  async importPersonaExemplars(
    this: TldwApiClientCore,
    personaId: string | number,
    payload: PersonaExemplarImportInput
  ): Promise<PersonaExemplar[]> {
    const encodedPersonaId = encodeURIComponent(String(personaId))
    const response = await this.request<any>({
      path: `/api/v1/persona/profiles/${encodedPersonaId}/exemplars/import`,
      method: "POST",
      body: payload
    })
    const list = Array.isArray(response)
      ? response
      : Array.isArray(response?.items)
        ? response.items
        : []
    return list.map((item) =>
      normalizePersonaExemplar(item as Record<string, unknown>)
    )
  },

  async updatePersonaExemplar(
    this: TldwApiClientCore,
    personaId: string | number,
    exemplarId: string | number,
    payload: Partial<PersonaExemplarInput>
  ): Promise<PersonaExemplar> {
    const encodedPersonaId = encodeURIComponent(String(personaId))
    const encodedExemplarId = encodeURIComponent(String(exemplarId))
    const response = await this.request<any>({
      path: `/api/v1/persona/profiles/${encodedPersonaId}/exemplars/${encodedExemplarId}`,
      method: "PATCH",
      body: payload
    })
    return normalizePersonaExemplar(response as Record<string, unknown>)
  },

  async reviewPersonaExemplar(
    this: TldwApiClientCore,
    personaId: string | number,
    exemplarId: string | number,
    payload: PersonaExemplarReviewInput
  ): Promise<PersonaExemplar> {
    const encodedPersonaId = encodeURIComponent(String(personaId))
    const encodedExemplarId = encodeURIComponent(String(exemplarId))
    const response = await this.request<any>({
      path: `/api/v1/persona/profiles/${encodedPersonaId}/exemplars/${encodedExemplarId}/review`,
      method: "POST",
      body: payload
    })
    return normalizePersonaExemplar(response as Record<string, unknown>)
  },

  async deletePersonaExemplar(
    this: TldwApiClientCore,
    personaId: string | number,
    exemplarId: string | number
  ): Promise<void> {
    const encodedPersonaId = encodeURIComponent(String(personaId))
    const encodedExemplarId = encodeURIComponent(String(exemplarId))
    await this.request<void>({
      path: `/api/v1/persona/profiles/${encodedPersonaId}/exemplars/${encodedExemplarId}`,
      method: "DELETE"
    })
  },
}

export type CharacterMethods = typeof characterMethods
