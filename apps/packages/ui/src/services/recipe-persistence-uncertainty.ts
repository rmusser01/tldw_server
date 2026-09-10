import { buildChatSurfaceScopeKeyFromConfig } from "@/services/chat-surface-scope"
import { deriveScopedUserId } from "@/utils/media-navigation-scope"

const uncertainRecipes = new Set<string>()

// The stable surface identity deliberately survives same-subject token refresh.
// Capture it before asynchronous work, never from credentials at settlement.
export const getRecipePersistenceScope = async (): Promise<string | null> => {
  try {
    const { tldwClient } = await import("@/services/tldw/TldwApiClient")
    const config = await tldwClient.getConfig()
    if (
      !config?.serverUrl?.trim() ||
      (config.authMode !== "single-user" && config.authMode !== "multi-user") ||
      deriveScopedUserId(config) === "user:anonymous"
    )
      return null
    return buildChatSurfaceScopeKeyFromConfig(config)
  } catch {
    // Unavailable ownership must not release another connection's marker.
    return null
  }
}

const uncertaintyKey = (id: string, scope: string) =>
  JSON.stringify([scope, id])

export const markRecipePersistenceUncertain = (
  id: string,
  scope: string | null
) => {
  if (scope) uncertainRecipes.add(uncertaintyKey(id, scope))
}

export const clearRecipePersistenceUncertainty = (
  id: string,
  scope: string | null
) => {
  if (scope) uncertainRecipes.delete(uncertaintyKey(id, scope))
}

export const isRecipePersistenceUncertain = (
  id: string,
  scope: string | null
) => !scope || uncertainRecipes.has(uncertaintyKey(id, scope))
