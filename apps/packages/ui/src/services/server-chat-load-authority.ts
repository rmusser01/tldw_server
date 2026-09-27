import { browser } from "wxt/browser"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
import { servicePromptPrincipalMatches, servicePromptSingleUserApiKeyScopeMatches, servicePromptTargetsMatch } from "@/services/tldw/service-prompt-scope-error"
import { REFRESH_ROTATION_KEY, REFRESH_SESSION_INVALIDATION_PREFIX } from "@/services/tldw/single-user-credential"

/** Keep a pending saved-chat read tied to its canonical credential generation. */
export const watchServerChatLoadAuthority = (snapshot: ServicePromptSnapshot, controller: AbortController): (() => void) => {
  let released = false
  let generation = 0
  const revalidate = async () => {
    const check = ++generation
    const current = () => !released && !controller.signal.aborted && check === generation
    try {
      const config = await tldwClient.ensureConfigForRequest(true)
      if (!current()) return
      const expected = snapshot.requestScope
      if (!servicePromptTargetsMatch(config, expected.config) ||
        (config.authSource !== "cookie-session" &&
          (!servicePromptPrincipalMatches(config, expected.userId) ||
            !servicePromptSingleUserApiKeyScopeMatches(config, expected.config.expectedSingleUserApiKeyScope)))) controller.abort()
    } catch {
      if (current()) controller.abort()
    }
  }
  const changed = () => { void revalidate() }
  const principalChanged = () => controller.abort()
  const relevantKey = (key: string) => key === "tldwConfig" || key === "tldwCookieSessionConfig" || key === REFRESH_ROTATION_KEY || key.startsWith(REFRESH_SESSION_INVALIDATION_PREFIX)
  const storageChanged = (event: StorageEvent) => { if (!event.key || relevantKey(event.key)) changed() }
  const extensionChanged = (changes: Record<string, unknown>, area: string) => {
    if (area === "local" && Object.keys(changes).some(relevantKey)) changed()
  }
  window.addEventListener("tldw:config-updated", changed)
  window.addEventListener("tldw:auth-principal-changed", principalChanged)
  window.addEventListener("storage", storageChanged)
  browser?.storage?.onChanged?.addListener(extensionChanged)
  return () => {
    released = true
    generation++
    window.removeEventListener("tldw:config-updated", changed)
    window.removeEventListener("tldw:auth-principal-changed", principalChanged)
    window.removeEventListener("storage", storageChanged)
    browser?.storage?.onChanged?.removeListener(extensionChanged)
  }
}
