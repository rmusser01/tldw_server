import type { RecipeDeliveryReceipt } from "@/services/recipe-persistence-registry"
import {
  directRecipeRequestAuthority,
  hasRecipeExtensionRuntime
} from "@/services/recipe-persistence-uncertainty"
import { createDirectRuntime } from "@/services/background-proxy"
import { captureDocumentGetSignal } from "./document-get-scope"
import type {
  AllowedMethodFor,
  PathOrUrl,
  UpperLower
} from "@/services/tldw/openapi-guard"
import type {
  RecipePersistenceDispatch,
  RecipePersistenceRequestPolicy
} from "@/services/tldw/recipe-request-snapshot"
import { tldwRequest } from "@/services/tldw/request-core"
import { createSafeStorage } from "@/utils/safe-storage"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"
import { createServicePromptScopeChangedError } from "@/services/tldw/service-prompt-scope-error"
import { deriveScopedUserId } from "@/utils/media-navigation-scope"
import { browser } from "wxt/browser"

export interface ApiSendPayload<
  P extends PathOrUrl = PathOrUrl,
  M extends AllowedMethodFor<P> = AllowedMethodFor<P>
> {
  path: P
  method?: UpperLower<M>
  headers?: Record<string, string>
  body?: any
  noAuth?: boolean
  timeoutMs?: number
  responseType?: "json" | "text" | "arrayBuffer"
  recipePersistence?: RecipePersistenceRequestPolicy
}

export interface ApiSendResponse<T = any> {
  ok: boolean
  status: number
  data?: T
  error?: string
  code?: string
  headers?: Record<string, string>
  retryAfterMs?: number | null
  /** Transport metadata, never supplied by the server response body. */
  recipePersistence?: RecipePersistenceDispatch
  /** Client-only operation guard; sync acknowledges after local settlement. */
  recipeDelivery?: RecipeDeliveryReceipt
}

const isSafeFallbackMethod = (method?: string): boolean => {
  const methodUpper = String(method || "GET").toUpperCase()
  return (
    methodUpper === "GET" || methodUpper === "HEAD" || methodUpper === "OPTIONS"
  )
}

const inFlightGetRequests = new Map<string, { promise: Promise<ApiSendResponse<unknown>>; documentSignal?: AbortSignal }>()

interface ApiSendOptions {
  coalesce?: boolean
  /** Captured connection owner; only the authenticated readiness GET may refresh. */
  readiness?: { config: TldwConfig; isCurrent: () => boolean }
}

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
    payload.recipePersistence ||
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

export async function apiSend<
  T = any,
  P extends PathOrUrl = PathOrUrl,
  M extends AllowedMethodFor<P> = AllowedMethodFor<P>
>(
  payload: ApiSendPayload<P, M>,
  // Client-local scheduling policy: never forwarded to a transport or server.
  options: ApiSendOptions = {}
): Promise<ApiSendResponse<T>> {
  const documentSignal = browser?.runtime?.sendMessage && browser?.runtime?.id
    ? undefined : captureDocumentGetSignal(payload.method)
  documentSignal?.throwIfAborted()
  const coalescingKey =
    options.coalesce === false || options.readiness ? null : getCoalescingKey(payload)
  if (coalescingKey) {
    const existing = inFlightGetRequests.get(coalescingKey)
    if (existing && !existing.documentSignal?.aborted) {
      try { return await existing.promise as ApiSendResponse<T> }
      finally { documentSignal?.throwIfAborted() }
    }

    const pending = apiSendImpl<T, P, M>(payload, options, documentSignal).finally(() => {
      if (inFlightGetRequests.get(coalescingKey)?.promise === pending) {
        inFlightGetRequests.delete(coalescingKey)
      }
    })
    inFlightGetRequests.set(
      coalescingKey,
      { promise: pending as Promise<ApiSendResponse<unknown>>, documentSignal }
    )
    try { return await pending }
    finally { documentSignal?.throwIfAborted() }
  }

  const response = await apiSendImpl(payload, options, documentSignal)
  documentSignal?.throwIfAborted()
  return response
}

async function apiSendImpl<
  T = any,
  P extends PathOrUrl = PathOrUrl,
  M extends AllowedMethodFor<P> = AllowedMethodFor<P>
>(payload: ApiSendPayload<P, M>, options: ApiSendOptions, documentSignal?: AbortSignal): Promise<ApiSendResponse<T>> {
  const recipeExtension = Boolean(
    payload.recipePersistence && hasRecipeExtensionRuntime()
  )
  if (
    recipeExtension &&
    !(browser?.runtime?.sendMessage && browser?.runtime?.id)
  ) {
    return {
      ok: false,
      status: 0,
      error: "Recipe background authority unavailable",
      recipePersistence: { state: "not_dispatched", actualOwnerId: null }
    }
  }
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
      const extensionPromise = browser.runtime.sendMessage({
        type: "tldw:request",
        payload
      })
      const timeoutPromise = new Promise<null>((resolve) => {
        setTimeout(() => resolve(null), extensionTimeout)
      })

      const resp = await Promise.race([extensionPromise, timeoutPromise])

      if (resp) {
        return resp as ApiSendResponse<T>
      }
      if (!methodIsSafeFallback || recipeExtension) {
        throw new Error("Extension messaging timeout")
      }
      // If resp is null (timeout), fall through to direct request for safe methods.
    }
  } catch (err) {
    if (
      payload.recipePersistence &&
      (!methodIsSafeFallback || recipeExtension)
    ) {
      // The background may already have written. Never replay this mutation or
      // invent its owner from the page's independently selected connection.
      return {
        ok: false,
        status: 0,
        error: "Extension request outcome is unknown",
        recipePersistence: { state: "unknown", actualOwnerId: null }
      }
    }
    const message =
      err instanceof Error
        ? err.message.toLowerCase()
        : String(err || "").toLowerCase()
    if (
      !methodIsSafeFallback &&
      message.includes("extension messaging timeout")
    ) {
      throw err
    }
    // fall through to direct request
  }
  const storage = createSafeStorage({ area: "local" })
  const readiness = options.readiness
  if (readiness && String(payload.path) === "/api/v1/auth/sessions" &&
    String(payload.method || "GET").toUpperCase() === "GET" &&
    !payload.noAuth && !payload.recipePersistence &&
    readiness.config.authMode === "multi-user" && readiness.config.refreshToken) {
    const assertCurrent = () => {
      if (!readiness.isCurrent()) throw createServicePromptScopeChangedError()
    }
    const guardedStorage = {
      get: async <V,>(key: string): Promise<V> => {
        assertCurrent()
        const value = await storage.get<V>(key)
        assertCurrent()
        return value
      },
      set: async <V,>(key: string, value: V) => {
        assertCurrent()
        const result = await storage.set(key, value)
        assertCurrent()
        return result
      },
      remove: async (key: string) => {
        assertCurrent()
        const result = await storage.remove(key)
        assertCurrent()
        return result
      }
    }
    const runtime = createDirectRuntime(guardedStorage, {
      ...readiness.config,
      expectedUserId: deriveScopedUserId(readiness.config).replace(/^user:/, "")
    }, undefined, undefined, documentSignal)
    return await tldwRequest(payload, {
      ...runtime,
      getConfig: async () => {
        assertCurrent()
        const config = await runtime.getConfig()
        assertCurrent()
        return config
      }
    })
  }
  return await tldwRequest(payload, {
    ...createDirectRuntime(storage, undefined, undefined, undefined, documentSignal),
    ...directRecipeRequestAuthority,
    // Only the guarded readiness GET above opts into automatic refresh/replay.
    refreshAuth: undefined
  })
}
