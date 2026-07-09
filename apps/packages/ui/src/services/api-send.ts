import { browser } from 'wxt/browser'
import { createSafeStorage } from '@/utils/safe-storage'
import { tldwRequest } from '@/services/tldw/request-core'
import type { PathOrUrl, AllowedMethodFor, UpperLower } from '@/services/tldw/openapi-guard'

export interface ApiSendPayload<P extends PathOrUrl = PathOrUrl, M extends AllowedMethodFor<P> = AllowedMethodFor<P>> {
  path: P
  method?: UpperLower<M>
  headers?: Record<string, string>
  body?: any
  noAuth?: boolean
  timeoutMs?: number
  responseType?: "json" | "text" | "arrayBuffer"
}

export interface ApiSendResponse<T = any> {
  ok: boolean
  status: number
  data?: T
  error?: string
  headers?: Record<string, string>
  retryAfterMs?: number | null
}

const isSafeFallbackMethod = (method?: string): boolean => {
  const methodUpper = String(method || "GET").toUpperCase()
  return methodUpper === "GET" || methodUpper === "HEAD" || methodUpper === "OPTIONS"
}

const inFlightGetRequests = new Map<string, Promise<ApiSendResponse<unknown>>>()

const normalizeHeaders = (
  headers?: Record<string, string>
): Record<string, string> | null => {
  if (!headers) return null
  return Object.keys(headers)
    .sort()
    .reduce<Record<string, string>>((acc, headerKey) => {
      acc[headerKey.toLowerCase()] = headers[headerKey]
      return acc
    }, {})
}

const getCoalescingKey = (payload: ApiSendPayload): string | null => {
  const method = String(payload.method || "GET").toUpperCase()
  if (
    method !== "GET" ||
    payload.body ||
    payload.responseType
  ) {
    return null
  }

  return JSON.stringify({
    method,
    path: payload.path,
    headers: normalizeHeaders(payload.headers),
    noAuth: Object.prototype.hasOwnProperty.call(payload, "noAuth")
      ? payload.noAuth
      : "__omitted__",
    timeoutMs: payload.timeoutMs ?? null
  })
}

export async function apiSend<T = any, P extends PathOrUrl = PathOrUrl, M extends AllowedMethodFor<P> = AllowedMethodFor<P>>(
  payload: ApiSendPayload<P, M>
): Promise<ApiSendResponse<T>> {
  const coalescingKey = getCoalescingKey(payload)
  if (coalescingKey) {
    const existing = inFlightGetRequests.get(coalescingKey)
    if (existing) return existing as Promise<ApiSendResponse<T>>

    const pending = apiSendImpl<T, P, M>(payload).finally(() => {
      if (inFlightGetRequests.get(coalescingKey) === pending) {
        inFlightGetRequests.delete(coalescingKey)
      }
    })
    inFlightGetRequests.set(
      coalescingKey,
      pending as Promise<ApiSendResponse<unknown>>
    )
    return pending
  }

  return apiSendImpl(payload)
}

async function apiSendImpl<T = any, P extends PathOrUrl = PathOrUrl, M extends AllowedMethodFor<P> = AllowedMethodFor<P>>(
  payload: ApiSendPayload<P, M>
): Promise<ApiSendResponse<T>> {
  const methodIsSafeFallback = isSafeFallbackMethod(
    payload?.method ? String(payload.method) : "GET"
  )
  try {
    // In web mode the wxt/browser shim provides sendMessage but no runtime.id.
    // Only use extension messaging when a real runtime is present.
    if (browser?.runtime?.sendMessage && browser?.runtime?.id) {

      // Add timeout to extension messaging - if it doesn't respond quickly, fall back to direct request
      // Must be less than CONNECTION_TIMEOUT_MS (20s) so health checks can fall back to direct fetch
      // Increased from 5s to 10s to reduce premature fallbacks during slow operations
      const extensionTimeout = 10000 // 10 second timeout for extension messaging
      const extensionPromise = browser.runtime.sendMessage({ type: 'tldw:request', payload })
      const timeoutPromise = new Promise<null>((resolve) => {
        setTimeout(() => resolve(null), extensionTimeout)
      })

      const resp = await Promise.race([extensionPromise, timeoutPromise])

      if (resp) {
        return resp as ApiSendResponse<T>
      }
      if (!methodIsSafeFallback) {
        throw new Error("Extension messaging timeout")
      }
      // If resp is null (timeout), fall through to direct request for safe methods.
    }
  } catch (err) {
    const message = err instanceof Error ? err.message.toLowerCase() : String(err || "").toLowerCase()
    if (
      !methodIsSafeFallback &&
      message.includes("extension messaging timeout")
    ) {
      throw err
    }
    // fall through to direct request
  }
  const storage = createSafeStorage()
  return await tldwRequest(payload, {
    // IMPORTANT: getConfig must fetch fresh config each time it's called
    // (not pre-fetch once), because the config may change or not be seeded yet.
    getConfig: async () => {
      const config = (await storage.get('tldwConfig').catch(() => null)) as
        | { serverUrl?: string }
        | null
      return config
    }
  })
}
