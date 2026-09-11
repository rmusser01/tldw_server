import { buildChatSurfaceScopeKeyFromConfig } from "@/services/chat-surface-scope"
import type { RecipePersistenceOwnerView } from "@/services/recipe-persistence-owner"
import {
  RecipePersistenceRegistry,
  type RecipeUncertaintyState
} from "@/services/recipe-persistence-registry"
import { resolveDirectBrowserConfig } from "@/services/tldw/direct-browser-config"
import {
  type RecipeRequestTransportSnapshot,
  resolveRecipeRequestSnapshot
} from "@/services/tldw/recipe-request-snapshot"
import { readBrowserCookie, tldwRequest } from "@/services/tldw/request-core"
import { getRuntimeSingleUserApiKeyOverride } from "@/services/tldw/runtime-auth-override"
import { deriveScopedUserId } from "@/utils/media-navigation-scope"
import { createSafeStorage } from "@/utils/safe-storage"
import { browser } from "wxt/browser"

export type { RecipeUncertaintyState } from "@/services/recipe-persistence-registry"

const directRegistry = new RecipePersistenceRegistry()
const isOwnerId = (value: unknown): value is string =>
  typeof value === "string" && /^recipe-owner:sha256:[0-9a-f]{64}$/.test(value)
const isLocalId = (value: unknown): value is string =>
  typeof value === "string" && value.trim().length > 0 && value.length <= 512

/** Dispatch markers must remain addressable by the authority's read/clear protocol. */
export function assertRecipeDispatchMarker(id: string, ownerId: string): void {
  if (!isLocalId(id) || !isOwnerId(ownerId))
    throw new Error("Invalid recipe dispatch marker")
}

export type RecipePersistenceMessage =
  | { type: "tldw:recipe-owner:resolve" }
  | { type: "tldw:recipe-uncertainty:read"; id: string; ownerId: string | null }
  | { type: "tldw:recipe-uncertainty:mark-scoped"; id: string; ownerId: string }
  | { type: "tldw:recipe-uncertainty:mark-unknown"; id: string }
  | {
      type: "tldw:recipe-uncertainty:clear-scoped"
      id: string
      ownerId: string
    }
  | { type: "tldw:recipe-uncertainty:forget-unknown"; id: string }

/** Reject extra fields too: page messages never supply credentials or owner material. */
export function isRecipePersistenceMessage(
  value: unknown
): value is RecipePersistenceMessage {
  if (!value || typeof value !== "object" || Array.isArray(value)) return false
  const message = value as Record<string, unknown>
  const keys = Object.keys(message)
  if (message.type === "tldw:recipe-owner:resolve") return keys.length === 1
  if (!isLocalId(message.id)) return false
  switch (message.type) {
    case "tldw:recipe-uncertainty:read":
      return (
        keys.length === 3 &&
        (message.ownerId === null || isOwnerId(message.ownerId))
      )
    case "tldw:recipe-uncertainty:mark-scoped":
    case "tldw:recipe-uncertainty:clear-scoped":
      return keys.length === 3 && isOwnerId(message.ownerId)
    case "tldw:recipe-uncertainty:mark-unknown":
    case "tldw:recipe-uncertainty:forget-unknown":
      return keys.length === 2
    default:
      return false
  }
}

/** Bind /auth/me to the already-resolved transport, never current mutable config.
 * This is non-captured local request-core I/O, with no refresh or background message.
 */
export async function getRecipeAuthenticatedPrincipal(
  snapshot: RecipeRequestTransportSnapshot
): Promise<string | number | null> {
  try {
    const url = new URL("api/v1/auth/me", snapshot.effectiveBase).toString()
    const response = await tldwRequest(
      {
        path: url as `https://${string}`,
        method: "GET",
        noAuth: true,
        headers: { ...snapshot.headers },
        timeoutMs: 10_000
      },
      {
        getConfig: async () => ({ serverUrl: snapshot.effectiveBase }),
        useRuntimeAuthOverride: false,
        // noAuth prevents credential reconstruction; explicitly preserve cookie policy.
        fetchFn: (input, init) =>
          fetch(input, { ...init, credentials: snapshot.credentials })
      }
    )
    const id = response.ok
      ? (response.data as { id?: unknown } | null)?.id
      : null
    if (typeof id === "string" && id.trim()) return id
    if (typeof id === "number" && Number.isFinite(id)) return id
    return null
  } catch {
    return null
  }
}

type OwnerConfig = Parameters<typeof resolveRecipeRequestSnapshot>[0]["config"]

/** Resolve the same unsafe representative route used by recipe v2 create. */
export async function resolveRecipeOwnerWithConfig(
  getConfig: () => Promise<OwnerConfig>
): Promise<RecipePersistenceOwnerView | null> {
  try {
    const config = await getConfig()
    const input = {
      config: config ? { ...config } : config,
      path: "/api/v1/prompts/",
      method: "POST",
      runtimeApiKey: getRuntimeSingleUserApiKeyOverride(),
      csrfToken: readBrowserCookie("csrf_token")
    }
    const resolved = resolveRecipeRequestSnapshot(input)
    if (resolved.authenticationError) return null
    if (resolved.view) return resolved.view
    if (
      resolved.snapshot.credentials === "same-origin" ||
      resolved.snapshot.headers.Authorization
    ) {
      const principal = await getRecipeAuthenticatedPrincipal(resolved.snapshot)
      return resolveRecipeRequestSnapshot({
        ...input,
        authenticatedPrincipalId: principal
      }).view
    }
    return null
  } catch {
    return null
  }
}

/** Used by the real direct request path, not a component-owned copy. */
export const directRecipeRequestAuthority = {
  getAuthenticatedPrincipal: getRecipeAuthenticatedPrincipal,
  dispatchAuthority: {
    markDispatched: (id: string, ownerId: string): void => {
      assertRecipeDispatchMarker(id, ownerId)
      directRegistry.markScoped(id, ownerId)
    }
  }
}

export function hasRecipeExtensionRuntime(): boolean {
  if (browser?.runtime?.id) return true
  const protocol =
    typeof window === "undefined" ? "" : window.location?.protocol
  return protocol === "chrome-extension:" || protocol === "moz-extension:"
}

async function sendRecipeMessage(
  message: RecipePersistenceMessage
): Promise<unknown> {
  if (!isRecipePersistenceMessage(message))
    throw new Error("Invalid recipe authority message")
  if (!browser?.runtime?.sendMessage)
    throw new Error("Recipe background authority unavailable")
  let timer: ReturnType<typeof setTimeout> | undefined
  try {
    return await Promise.race([
      browser.runtime.sendMessage(message),
      new Promise<never>((_, reject) => {
        timer = setTimeout(
          () => reject(new Error("Recipe background authority unavailable")),
          3_000
        )
      })
    ])
  } finally {
    if (timer) clearTimeout(timer)
  }
}

export async function resolveRecipePersistenceOwnerView(): Promise<RecipePersistenceOwnerView | null> {
  if (!hasRecipeExtensionRuntime()) {
    return resolveRecipeOwnerWithConfig(() =>
      resolveDirectBrowserConfig(createSafeStorage({ area: "local" }))
    )
  }
  try {
    const value = await sendRecipeMessage({ type: "tldw:recipe-owner:resolve" })
    if (!value || typeof value !== "object" || Object.keys(value).length !== 2)
      return null
    const view = value as RecipePersistenceOwnerView
    return isOwnerId(view.ownerId) &&
      typeof view.authorizationRevision === "string" &&
      /^recipe-authorization:sha256:[0-9a-f]{64}$/.test(
        view.authorizationRevision
      )
      ? {
          ownerId: view.ownerId,
          authorizationRevision: view.authorizationRevision
        }
      : null
  } catch {
    return null
  }
}

export async function readRecipePersistenceUncertainty(
  id: string,
  ownerId: string | null
): Promise<RecipeUncertaintyState> {
  const message: RecipePersistenceMessage = {
    type: "tldw:recipe-uncertainty:read",
    id,
    ownerId
  }
  if (!isRecipePersistenceMessage(message))
    throw new Error("Invalid recipe authority message")
  if (!hasRecipeExtensionRuntime()) return directRegistry.read(id, ownerId)
  const value = await sendRecipeMessage(message)
  if (value === "clear" || value === "scoped" || value === "unknown_owner")
    return value
  throw new Error("Recipe background authority unavailable")
}

async function mutate(
  message: RecipePersistenceMessage,
  direct: () => void
): Promise<void> {
  if (!isRecipePersistenceMessage(message))
    throw new Error("Invalid recipe authority message")
  if (!hasRecipeExtensionRuntime()) return direct()
  const response = await sendRecipeMessage(message)
  if (
    !response ||
    typeof response !== "object" ||
    Object.keys(response).length !== 1 ||
    (response as { ok?: unknown }).ok !== true
  ) {
    throw new Error("Recipe background authority unavailable")
  }
}

export const markRecipePersistenceScoped = (
  id: string,
  ownerId: string
): Promise<void> =>
  mutate({ type: "tldw:recipe-uncertainty:mark-scoped", id, ownerId }, () =>
    directRegistry.markScoped(id, ownerId)
  )
export const clearRecipePersistenceScoped = (
  id: string,
  ownerId: string
): Promise<void> =>
  mutate({ type: "tldw:recipe-uncertainty:clear-scoped", id, ownerId }, () =>
    directRegistry.clearScoped(id, ownerId)
  )
export const markRecipePersistenceUnknown = (id: string): Promise<void> =>
  mutate({ type: "tldw:recipe-uncertainty:mark-unknown", id }, () =>
    directRegistry.markUnknown(id)
  )
export const forgetRecipePersistenceUnknown = (id: string): Promise<void> =>
  mutate({ type: "tldw:recipe-uncertainty:forget-unknown", id }, () =>
    directRegistry.forgetUnknown(id)
  )

// Deprecated Task 4/5 transition wrappers: preserve synchronous callers' types.
// These are direct-only compatibility state, never the extension async authority.
export const getRecipePersistenceScope = async (): Promise<string | null> => {
  try {
    const { tldwClient } = await import("@/services/tldw/TldwApiClient")
    const config = await tldwClient.getConfig()
    if (
      !config?.serverUrl?.trim() ||
      (config.authMode !== "single-user" && config.authMode !== "multi-user") ||
      deriveScopedUserId({
        authMode: config.authMode,
        accessToken: config.accessToken
      }) === "user:anonymous"
    )
      return null
    return buildChatSurfaceScopeKeyFromConfig(config)
  } catch {
    return null
  }
}
export const markRecipePersistenceUncertain = (
  id: string,
  ownerId: string | null
): void => {
  if (ownerId) directRegistry.markScoped(id, ownerId)
}
export const markRecipePersistenceOwnerUnknown = (id: string): void =>
  directRegistry.markUnknown(id)
export const clearRecipePersistenceUncertainty = (
  id: string,
  ownerId: string | null
): void => {
  if (ownerId) directRegistry.clearScoped(id, ownerId)
}
export const isRecipePersistenceUncertain = (
  id: string,
  ownerId: string | null
): boolean => !ownerId || directRegistry.read(id, ownerId) !== "clear"
