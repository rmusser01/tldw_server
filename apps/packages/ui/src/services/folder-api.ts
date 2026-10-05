/**
 * Folder System API Service
 *
 * Handles communication with tldw_server for folder (keyword_collections)
 * and keyword management. Uses bgRequestClient for proper auth handling.
 *
 * All methods in this file return a typed response envelope and use a
 * single-flight pattern to deduplicate concurrent identical requests.
 */

import { requestScopeFields, type ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"
import { bgRequestClient } from "@/services/background-proxy"
import type {
  AllowedPath,
  ClientPathOrUrlWithQuery
} from "@/services/tldw/openapi-guard"
import type { Folder, Keyword, FolderKeywordLink, ConversationKeywordLink } from "@/db/dexie/types"
import {
  buildQuery,
  createResourceClient,
  type RequestFn
} from "@/services/resource-client"

const foldersClient = createResourceClient({
  basePath: "/api/v1/notes/collections" as AllowedPath,
  request: bgRequestClient as unknown as RequestFn
})

const keywordsClient = createResourceClient({
  basePath: "/api/v1/notes/keywords" as AllowedPath,
  request: bgRequestClient as unknown as RequestFn
})

type ArrayOrWrapped<T, K extends string> = T[] | { [key in K]: T[] } | null | undefined

export interface ApiResult<T> {
  ok: boolean
  data?: T
  error?: string
  status?: number
}

export type FolderRequestOptions = {
  abortSignal?: AbortSignal
  timeoutMs?: number
  requestScope?: ServicePromptRequestScope
}

// Captured scope and cancellation lifetimes never join another account or a
// canceled picker. Object identities keep credentials out of request keys.
const requestLifetimeIds = new WeakMap<object, number>()
let nextRequestLifetimeId = 0
const requestLifetimeId = (value?: object): number | "-" => {
  if (!value) return "-"
  let id = requestLifetimeIds.get(value)
  if (id === undefined) {
    id = ++nextRequestLifetimeId
    requestLifetimeIds.set(value, id)
  }
  return id
}
const scopedRequestKey = (key: string, options?: FolderRequestOptions): string =>
  `${key} scope:${requestLifetimeId(options?.requestScope)} signal:${requestLifetimeId(options?.abortSignal)}`
const requestOptions = (options?: FolderRequestOptions) => ({
  abortSignal: options?.abortSignal,
  timeoutMs: options?.timeoutMs,
  ...requestScopeFields(options?.requestScope)
})

const inFlightRequests = new Map<string, Promise<ApiResult<unknown>>>()

const normalizeArrayResponse = <T, K extends string>(
  response: ArrayOrWrapped<T, K>,
  key: K
): T[] => {
  if (!response) return []
  if (Array.isArray(response)) return response

  if (typeof response === 'object' && response !== null && key in response) {
    const wrapped = response[key]
    if (Array.isArray(wrapped)) return wrapped
  }

  const tag = Object.prototype.toString.call(response)
  const keys =
    typeof response === 'object' && response !== null ? Object.keys(response) : []
  const keysPart = keys.length
    ? ` keys=[${keys.slice(0, 8).join(', ')}${keys.length > 8 ? ', …' : ''}]`
    : ''
  throw new Error(`Unexpected response shape (expected array or { ${key}: [] }, got ${tag}${keysPart})`)
}

const ORGANIZATION_PAGE_SIZE = 1000

// Complete the existing offset API before publishing a usable picker snapshot.
// A later-page failure rejects the whole read; it is not an empty/full result.
const readAllPages = async <T, K extends string>(
  key: K,
  read: (offset: number, limit: number) => Promise<unknown>,
  signal?: AbortSignal
): Promise<T[]> => {
  const rows: T[] = []
  let offset = 0
  while (true) {
    signal?.throwIfAborted()
    const response = await read(offset, ORGANIZATION_PAGE_SIZE)
    signal?.throwIfAborted()
    const page = normalizeArrayResponse<T, K>(response as ArrayOrWrapped<T, K>, key)
    rows.push(...page)
    const metadata = response && typeof response === "object" && !Array.isArray(response)
      ? response as Record<string, unknown> : {}
    const pagination = metadata.pagination && typeof metadata.pagination === "object"
      ? metadata.pagination as Record<string, unknown> : {}
    const total = metadata.total ?? pagination.total
    const hasMore = metadata.has_more ?? pagination.has_more
    const next = metadata.next_offset ?? pagination.next_offset
    const more = hasMore === true || (typeof total === "number" && offset + page.length < total) ||
      (hasMore === undefined && total == null && page.length === ORGANIZATION_PAGE_SIZE)
    if (!more) return rows
    const nextOffset = typeof next === "number" ? next : offset + page.length
    if (page.length === 0 || !Number.isSafeInteger(nextOffset) || nextOffset <= offset) {
      throw new Error("Incomplete folder/tag page sequence; refresh and try again.")
    }
    offset = nextOffset
  }
}

const buildRequestKey = (method: string, path: string, body?: unknown): string => {
  const m = method.toUpperCase()
  if (body === undefined || body === null) {
    return `${m} ${path}`
  }
  try {
    const json = JSON.stringify(body)
    if (json !== undefined) {
      return `${m} ${path} ${json}`
    }
  } catch {
    // ignore; fall back to a readable string representation
  }
  try {
    return `${m} ${path} ${String(body)}`
  } catch {
    try {
      return `${m} ${path} ${Object.prototype.toString.call(body)}`
    } catch {
      return `${m} ${path} [unstringifiable body]`
    }
  }
}

const extractStatusFromError = (error: unknown): number | undefined => {
  const asHttpStatus = (value: unknown): number | undefined => {
    const num =
      typeof value === 'number'
        ? value
        : typeof value === 'string'
          ? Number(value.trim())
          : NaN
    if (!Number.isFinite(num) || !Number.isInteger(num)) return undefined
    return num >= 100 && num <= 599 ? num : undefined
  }

  const fromRecord = (record: Record<string, unknown>): number | undefined => {
    return (
      asHttpStatus(record.status) ??
      asHttpStatus(record.statusCode) ??
      asHttpStatus(record.code)
    )
  }

  if (error && typeof error === 'object') {
    const record = error as Record<string, unknown>
    const direct = fromRecord(record)
    if (direct !== undefined) return direct

    const response = record.response
    if (response && typeof response === 'object') {
      const nested = fromRecord(response as Record<string, unknown>)
      if (nested !== undefined) return nested
    }
  }

  const msg = error instanceof Error ? error.message : String(error)
  const matches = [
    msg.match(/\b(?:status|statusCode)\b\s*[:=]?\s*(\d{3})\b/i)?.[1],
    msg.match(/\bHTTP(?:\/\d(?:\.\d)?)?\s*(\d{3})\b/i)?.[1],
    msg.match(/\b(?:request|upload)\s+failed\b\s*[:(]\s*(\d{3})\b/i)?.[1],
    msg.match(/\((\d{3})\)/)?.[1]
  ]

  for (const match of matches) {
    const status = asHttpStatus(match)
    if (status !== undefined) return status
  }

  return undefined
}

const singleFlight = async <T>(
  key: string,
  executor: () => Promise<T>
): Promise<ApiResult<T>> => {
  const existing = inFlightRequests.get(key) as Promise<ApiResult<T>> | undefined
  if (existing) return existing

  const promise: Promise<ApiResult<T>> = (async () => {
    try {
      const data = await executor()
      return {
        ok: true,
        data,
        // bgRequest/bgRequestClient does not currently expose HTTP status on success,
        // so use 200 as a generic indicator (this will mask 201/204/etc until status is plumbed through).
        status: 200
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      return {
        ok: false,
        error: message,
        status: extractStatusFromError(error)
      }
    }
  })()

  inFlightRequests.set(key, promise as Promise<ApiResult<unknown>>)

  const cleanup = () => {
    const current = inFlightRequests.get(key)
    if (current === promise) {
      inFlightRequests.delete(key)
    }
  }

  promise.then(cleanup, cleanup)

  return promise
}

// ─────────────────────────────────────────────────────────────────────────────
// Folder (keyword_collections) API
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Fetch all folders from server
 */
export const fetchFolders = async (
  options?: FolderRequestOptions
): Promise<ApiResult<Folder[]>> => {
  const path = '/api/v1/notes/collections' as ClientPathOrUrlWithQuery
  const key = scopedRequestKey(buildRequestKey('GET', path), options)
  return singleFlight<Folder[]>(key, () => readAllPages<Folder, "collections">(
    "collections", (offset, limit) => foldersClient.list({ offset, limit }, requestOptions(options)), options?.abortSignal
  ))
}

/**
 * Create a new folder
 */
export const createFolder = async (
  name: string,
  parentId?: number | null,
  options?: FolderRequestOptions
): Promise<ApiResult<Folder>> => {
  const path = '/api/v1/notes/collections' as ClientPathOrUrlWithQuery
  const body = { name, parent_id: parentId ?? null }
  const key = scopedRequestKey(buildRequestKey('POST', path, body), options)
  return singleFlight<Folder>(key, async () => {
    return foldersClient.create<Folder>(body, requestOptions(options))
  })
}

/**
 * Update an existing folder
 */
export const updateFolder = async (
  id: number,
  data: { name?: string; parent_id?: number | null },
  options?: FolderRequestOptions
): Promise<ApiResult<Folder>> => {
  const path = `/api/v1/notes/collections/${id}` as ClientPathOrUrlWithQuery
  const key = scopedRequestKey(buildRequestKey('PATCH', path, data), options)
  return singleFlight<Folder>(key, async () => {
    return foldersClient.update<Folder>(id, data, requestOptions(options))
  })
}

/**
 * Delete a folder (soft delete on server)
 */
export const deleteFolder = async (
  id: number,
  options?: FolderRequestOptions
): Promise<ApiResult<void>> => {
  const path = `/api/v1/notes/collections/${id}` as ClientPathOrUrlWithQuery
  const key = scopedRequestKey(buildRequestKey('DELETE', path), options)
  return singleFlight<void>(key, async () => {
    await foldersClient.remove<void>(id, undefined, requestOptions(options))
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Keyword API
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Fetch all keywords from server
 */
export const fetchKeywords = async (
  options?: FolderRequestOptions
): Promise<ApiResult<Keyword[]>> => {
  const path = '/api/v1/notes/keywords' as ClientPathOrUrlWithQuery
  const key = scopedRequestKey(buildRequestKey('GET', path), options)
  return singleFlight<Keyword[]>(key, () => readAllPages<Keyword, "keywords">(
    "keywords", (offset, limit) => keywordsClient.list({ offset, limit }, requestOptions(options)), options?.abortSignal
  ))
}

/**
 * Create a new keyword
 */
export const createKeyword = async (
  keyword: string,
  options?: FolderRequestOptions
): Promise<ApiResult<Keyword>> => {
  const path = '/api/v1/notes/keywords' as ClientPathOrUrlWithQuery
  const body = { keyword }
  const key = scopedRequestKey(buildRequestKey('POST', path, body), options)
  return singleFlight<Keyword>(key, async () => {
    return keywordsClient.create<Keyword>(body, requestOptions(options))
  })
}

/**
 * Delete a keyword (soft delete on server)
 */
export const deleteKeyword = async (
  id: number,
  options?: FolderRequestOptions
): Promise<ApiResult<void>> => {
  const path = `/api/v1/notes/keywords/${id}` as ClientPathOrUrlWithQuery
  const key = scopedRequestKey(buildRequestKey('DELETE', path), options)
  return singleFlight<void>(key, async () => {
    await keywordsClient.remove<void>(id, undefined, requestOptions(options))
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Folder-Keyword Linking API
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Link a keyword to a folder
 */
export const linkKeywordToFolder = async (
  folderId: number,
  keywordId: number,
  options?: FolderRequestOptions
): Promise<ApiResult<void>> => {
  const path = `/api/v1/notes/collections/${folderId}/keywords/${keywordId}` as ClientPathOrUrlWithQuery
  const key = scopedRequestKey(buildRequestKey('POST', path), options)
  return singleFlight<void>(key, async () => {
    await bgRequestClient({
      path,
      method: 'POST',
      ...requestOptions(options)
    })
  })
}

/**
 * Unlink a keyword from a folder
 */
export const unlinkKeywordFromFolder = async (
  folderId: number,
  keywordId: number,
  options?: FolderRequestOptions
): Promise<ApiResult<void>> => {
  const path = `/api/v1/notes/collections/${folderId}/keywords/${keywordId}` as ClientPathOrUrlWithQuery
  const key = scopedRequestKey(buildRequestKey('DELETE', path), options)
  return singleFlight<void>(key, async () => {
    await bgRequestClient({
      path,
      method: 'DELETE',
      ...requestOptions(options)
    })
  })
}

/**
 * Get all keywords for a folder
 */
export const getKeywordsForFolder = async (
  folderId: number,
  options?: FolderRequestOptions
): Promise<ApiResult<Keyword[]>> => {
  const path = `/api/v1/notes/collections/${folderId}/keywords` as ClientPathOrUrlWithQuery
  const key = scopedRequestKey(buildRequestKey('GET', path), options)
  return singleFlight<Keyword[]>(key, async () => {
    const response = await bgRequestClient<ArrayOrWrapped<Keyword, "keywords">>({
      path,
      method: 'GET',
      ...requestOptions(options)
    })
    return normalizeArrayResponse(response, "keywords")
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Conversation-Keyword Linking API
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Link a keyword to a conversation
 */
export const linkKeywordToConversation = async (
  conversationId: string,
  keywordId: number,
  options?: FolderRequestOptions
): Promise<ApiResult<void>> => {
  const cid = encodeURIComponent(conversationId)
  const path = `/api/v1/notes/conversations/${cid}/keywords/${keywordId}` as ClientPathOrUrlWithQuery
  const key = scopedRequestKey(buildRequestKey('POST', path), options)
  return singleFlight<void>(key, async () => {
    await bgRequestClient({
      path,
      method: 'POST',
      ...requestOptions(options)
    })
  })
}

/**
 * Unlink a keyword from a conversation
 */
export const unlinkKeywordFromConversation = async (
  conversationId: string,
  keywordId: number,
  options?: FolderRequestOptions
): Promise<ApiResult<void>> => {
  const cid = encodeURIComponent(conversationId)
  const path = `/api/v1/notes/conversations/${cid}/keywords/${keywordId}` as ClientPathOrUrlWithQuery
  const key = scopedRequestKey(buildRequestKey('DELETE', path), options)
  return singleFlight<void>(key, async () => {
    await bgRequestClient({
      path,
      method: 'DELETE',
      ...requestOptions(options)
    })
  })
}

/**
 * Get all keywords for a conversation
 */
export const getKeywordsForConversation = async (
  conversationId: string,
  options?: FolderRequestOptions
): Promise<ApiResult<Keyword[]>> => {
  const cid = encodeURIComponent(conversationId)
  const path = `/api/v1/notes/conversations/${cid}/keywords` as ClientPathOrUrlWithQuery
  const key = scopedRequestKey(buildRequestKey('GET', path), options)
  return singleFlight<Keyword[]>(key, async () => {
    const response = await bgRequestClient<ArrayOrWrapped<Keyword, "keywords">>({
      path,
      method: 'GET',
      ...requestOptions(options)
    })
    return normalizeArrayResponse(response, "keywords")
  })
}

// ─────────────────────────────────────────────────────────────────────────────
// Bulk Operations
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Fetch all folder-keyword links (for building the tree)
 */
export const fetchFolderKeywordLinks = async (
  options?: FolderRequestOptions
): Promise<ApiResult<FolderKeywordLink[]>> => {
  const path = '/api/v1/notes/collections/keyword-links' as ClientPathOrUrlWithQuery
  const key = scopedRequestKey(buildRequestKey('GET', path), options)
  return singleFlight<FolderKeywordLink[]>(key, async () => {
    const links = await readAllPages<{ collection_id: number; keyword_id: number }, "links">(
      "links", (offset, limit) => bgRequestClient({
        path: `${path}${buildQuery({ offset, limit })}` as ClientPathOrUrlWithQuery,
        method: 'GET', ...requestOptions(options)
      }), options?.abortSignal)
    return links.map((link) => {
      if (!Number.isSafeInteger(link.collection_id) || !Number.isSafeInteger(link.keyword_id)) {
        throw new Error("Invalid collection-keyword link response.")
      }
      return { folder_id: link.collection_id, keyword_id: link.keyword_id }
    })
  })
}

/**
 * Fetch all conversation-keyword links for a set of conversations
 */
export const fetchConversationKeywordLinks = async (
  conversationIds?: string[],
  options?: FolderRequestOptions
): Promise<ApiResult<ConversationKeywordLink[]>> => {
  const ids = conversationIds?.length ? conversationIds : undefined
  const path = `/api/v1/notes/conversations/keyword-links${buildQuery({ ids }, { arrayFormat: "comma" })}` as ClientPathOrUrlWithQuery
  const key = scopedRequestKey(buildRequestKey('GET', path), options)
  return singleFlight<ConversationKeywordLink[]>(key, () => readAllPages<ConversationKeywordLink, "links">(
    "links", (offset, limit) => bgRequestClient({
      path: `/api/v1/notes/conversations/keyword-links${buildQuery({ ids, offset, limit }, { arrayFormat: "comma" })}` as ClientPathOrUrlWithQuery,
      method: 'GET', ...requestOptions(options)
    }), options?.abortSignal
  ))
}
