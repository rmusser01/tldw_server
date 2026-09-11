import { buildChatSurfaceScopeKeyFromConfig } from "@/services/chat-surface-scope"
import type { RecipePersistenceOwnerView } from "@/services/recipe-persistence-owner"
import { deriveScopedUserId } from "@/utils/media-navigation-scope"

const uncertainRecipes = new Set<string>()
const unknownOwnerRecipes = new Set<string>()
type RecipePersistenceOwnerId = RecipePersistenceOwnerView["ownerId"]

export const recipePersistenceScopeFromConfig = (
  config: Parameters<typeof buildChatSurfaceScopeKeyFromConfig>[0]
): string | null => {
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
}

// The stable surface identity deliberately survives same-subject token refresh.
// Capture it before asynchronous work, never from credentials at settlement.
export const getRecipePersistenceScope = async (): Promise<string | null> => {
  try {
    const { tldwClient } = await import("@/services/tldw/TldwApiClient")
    const config = await tldwClient.getConfig()
    return recipePersistenceScopeFromConfig(config)
  } catch {
    // Unavailable ownership must not release another connection's marker.
    return null
  }
}

const uncertaintyKey = (id: string, scope: RecipePersistenceOwnerId) =>
  JSON.stringify([scope, id])

export const markRecipePersistenceUncertain = (
  id: string,
  scope: RecipePersistenceOwnerId | null
) => {
  if (scope) uncertainRecipes.add(uncertaintyKey(id, scope))
}

// No connection can reconcile an outcome whose dispatch owner is unknown.
// Keep this exact-ID quarantine for the session, even if durable marking fails.
export const markRecipePersistenceOwnerUnknown = (id: string) => {
  unknownOwnerRecipes.add(id)
}

export const clearRecipePersistenceUncertainty = (
  id: string,
  scope: RecipePersistenceOwnerId | null
) => {
  if (scope) uncertainRecipes.delete(uncertaintyKey(id, scope))
}

export const isRecipePersistenceUncertain = (
  id: string,
  scope: RecipePersistenceOwnerId | null
) =>
  !scope ||
  unknownOwnerRecipes.has(id) ||
  uncertainRecipes.has(uncertaintyKey(id, scope))
