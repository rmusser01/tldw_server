import type { TldwConfig } from "@/services/tldw/TldwApiClient"

type TldwAuthLike = {
  getCurrentUser: () => Promise<unknown>
  logout?: () => Promise<void>
}

/** The shell must observe another tab's login even when the client caches signed-out state. */
export const loadConfiguredAuthConfig = async (): Promise<TldwConfig | null> => {
  const [{ resolveDirectBrowserConfig }, { createSafeStorage }] = await Promise.all([
    import("@/services/tldw/direct-browser-config"),
    import("@/utils/safe-storage")
  ])
  return resolveDirectBrowserConfig(createSafeStorage({ area: "local" }))
}

export const loadTldwAuth = async (): Promise<TldwAuthLike> => {
  const authModule = await import("@/services/tldw/TldwAuth")
  return authModule.tldwAuth as TldwAuthLike
}
