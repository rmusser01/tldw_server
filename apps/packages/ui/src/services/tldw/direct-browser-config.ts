import { createSafeStorage } from "@/utils/safe-storage"
import { COOKIE_SESSION_CONFIG_KEY } from "@/services/tldw/browser-networking"
import { isCookieSessionConfigInvalidated } from "@/services/tldw/runtime-auth-override"
import { resolveEffectiveTldwConfig } from "@/services/tldw/single-user-credential"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"

export type DirectRuntimeStorage = Pick<
  ReturnType<typeof createSafeStorage>,
  "get" | "set" | "remove"
>

const getQuickstartWebUiCookie = async (
  storage: DirectRuntimeStorage
): Promise<
  | { cookieSession: TldwConfig | null; expectedCookieOrigin: string }
  | undefined
> => {
  if (isCookieSessionConfigInvalidated() || typeof window === "undefined") {
    return undefined
  }
  const protocol = String(window.location?.protocol || "").toLowerCase()
  if (protocol !== "http:" && protocol !== "https:") return undefined
  const deploymentMode = String(
    typeof process !== "undefined"
      ? process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE || ""
      : ""
  ).trim()
  if (deploymentMode !== "quickstart") return undefined
  const expectedCookieOrigin = String(window.location?.origin || "").trim()
  if (!expectedCookieOrigin) return undefined
  const cookieSession = await storage
    .get<TldwConfig>(COOKIE_SESSION_CONFIG_KEY)
    .catch(() => null)
  return { cookieSession, expectedCookieOrigin }
}

export const resolveDirectBrowserConfig = async (
  storage: DirectRuntimeStorage
): Promise<TldwConfig | null> =>
  await resolveEffectiveTldwConfig(
    {
      persistent: storage,
      session: createSafeStorage({ area: "session" })
    },
    await getQuickstartWebUiCookie(storage)
  )
