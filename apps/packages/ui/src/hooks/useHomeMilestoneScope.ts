import { useEffect, useState } from "react"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { buildChatSurfaceScopeKeyFromConfig } from "@/services/chat-surface-scope"
import { deriveScopedUserId } from "@/utils/media-navigation-scope"

/** Current account/server identity for Home progress and source ownership. */
export function useHomeMilestoneScope({ verifyCookieSession = false } = {}): string | null {
  const [scope, setScope] = useState<string | null>(null)
  useEffect(() => {
    let generation = 0
    const refresh = (reloadConfig = false) => {
      const request = ++generation
      setScope(null)
      void (
        reloadConfig
          ? tldwClient.initialize().then(() => tldwClient.getConfig())
          : tldwClient.getConfig()
      )
        .then(async (config) => {
          if (request !== generation || !config?.serverUrl) return
          if (verifyCookieSession && config.authMode === "multi-user" && config.authSource === "cookie-session") {
            const { resolveServicePromptScope } = await import("@/services/service-prompts")
            if (request !== generation) return
            const { scopeKey } = await resolveServicePromptScope()
            if (request === generation) setScope(scopeKey)
            return
          }
          if (deriveScopedUserId(config) === "user:anonymous") return
          // Single-user keys authenticate the same server account; manual-key
          // recovery must not change the account's setup progress identity.
          setScope(
            buildChatSurfaceScopeKeyFromConfig({ ...config, apiKey: undefined })
          )
        })
        .catch(() => undefined)
    }
    const configUpdated = () => refresh()
    const principalChanged = (event: Event) => {
      if ((event as CustomEvent).detail?.kind === "logout") {
        generation += 1
        setScope(null)
      } else {
        refresh(true)
      }
    }
    const storageChanged = (event: StorageEvent) => {
      const key = event.key?.replace(/^plasmo-(?:local|sync):/, "")
      if (
        key == null ||
        [
          "tldwConfig",
          "tldwCookieSessionConfig",
          "serverUrl",
          "authMode",
          "apiKey",
          "accessToken"
        ].includes(key)
      )
        refresh(true)
    }
    refresh()
    window.addEventListener("tldw:config-updated", configUpdated)
    window.addEventListener("tldw:auth-principal-changed", principalChanged)
    window.addEventListener("storage", storageChanged)
    return () => {
      generation += 1
      window.removeEventListener("tldw:config-updated", configUpdated)
      window.removeEventListener(
        "tldw:auth-principal-changed",
        principalChanged
      )
      window.removeEventListener("storage", storageChanged)
    }
  }, [verifyCookieSession])
  return scope
}
