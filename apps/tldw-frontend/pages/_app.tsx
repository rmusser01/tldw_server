import "../styles/globals.css"
import "@/assets/react-pdf.css"
import "@web/extension/shims/runtime-bootstrap"
// Use web-specific i18n that works with SSR/static generation
import "@web/lib/i18n-web"
import type { AppProps } from "next/app"
import dynamic from "next/dynamic"
import { useRouter } from "next/router"
import React from "react"
import { BackendRecoveryUiProvider } from "@/components/Common/BackendRecoveryUiContext"
import { FirstRunGate } from "@/components/PersonaGarden/FirstRunGate"
import { AppProviders } from "@web/components/AppProviders"
import ErrorBoundary from "@web/components/ErrorBoundary"
import { ConfigurationGuard } from "@web/components/networking/ConfigurationGuard"
import { ServerReadinessGate } from "@web/components/networking/ServerReadinessGate"
import { hasEnvApiAuth } from "@web/lib/authStorage"
import { loadTldwAuth, loadTldwClient } from "@web/lib/configured-auth-state"

const OptionLayout = dynamic(
  () => import("@web/components/layout/WebLayout"),
  { ssr: false }
)

// Ordered to match high-traffic navigation:
// - Route-registry eager imports (chat/media/media-multi/research)
// - Default sidebar shortcut selections (prompts/characters/dictionaries/world-books/knowledge)
const PRIMARY_WARM_PREFETCH_ROUTES = [
  "/chat",
  "/media",
  "/media-multi",
  "/research",
  "/knowledge",
  "/prompts",
  "/characters",
  "/dictionaries",
  "/world-books",
  "/settings"
] as const

// Secondary warmups for power-user paths; skipped on data saver / very slow networks.
const SECONDARY_WARM_PREFETCH_ROUTES = [
  "/document-workspace"
] as const

const PREFETCH_STEP_DELAY_MS = 250
const PREFETCH_IDLE_TIMEOUT_MS = 2000
const PREFETCH_FALLBACK_DELAY_MS = 1200
const SLOW_EFFECTIVE_TYPES = new Set(["slow-2g", "2g"])

type ConfiguredAuthState = {
  hasConfig: boolean
  authMode?: "single-user" | "multi-user"
  isAuthenticated: boolean
}

const getConfiguredAuthState = async (): Promise<ConfiguredAuthState> => {
  try {
    const tldwClient = await loadTldwClient()
    const config = await tldwClient.getConfig()
    if (!config) {
      return {
        hasConfig: false,
        isAuthenticated: false
      }
    }

    if (config.authMode === "multi-user") {
      const hasAccessToken =
        typeof config.accessToken === "string" &&
        config.accessToken.trim().length > 0
      if (!hasAccessToken) {
        return {
          hasConfig: true,
          authMode: "multi-user",
          isAuthenticated: false
        }
      }

      try {
        const tldwAuth = await loadTldwAuth()
        await tldwAuth.getCurrentUser()
        return {
          hasConfig: true,
          authMode: "multi-user",
          isAuthenticated: true
        }
      } catch {
        return {
          hasConfig: true,
          authMode: "multi-user",
          isAuthenticated: false
        }
      }
    }

    return {
      hasConfig: true,
      authMode: "single-user",
      isAuthenticated:
        typeof config.apiKey === "string" && config.apiKey.trim().length > 0
    }
  } catch {
    return {
      hasConfig: false,
      isAuthenticated: false
    }
  }
}

export default function App({ Component, pageProps }: AppProps) {
  const router = useRouter()
  const pathname = router.pathname || ""
  const routePath =
    pathname.length > 1 && pathname.endsWith("/")
      ? pathname.slice(0, -1)
      : pathname
  const isPublicAuthRoute = routePath === "/login"
  const isSettingsRoute =
    routePath === "/settings" || routePath.startsWith("/settings/")
  const [isAuthenticated, setIsAuthenticated] = React.useState(false)
  const [authResolved, setAuthResolved] = React.useState(false)
  const didWarmRoutePrefetch = React.useRef(false)

  React.useEffect(() => {
    if (typeof window === "undefined") return

    let cancelled = false
    const refreshAuthState = async () => {
      const envAuthed = hasEnvApiAuth()
      const configuredAuth = await getConfiguredAuthState()
      const authed = configuredAuth.hasConfig
        ? configuredAuth.authMode === "multi-user"
          ? configuredAuth.isAuthenticated
          : configuredAuth.isAuthenticated || envAuthed
        : envAuthed

      if (!cancelled) {
        setIsAuthenticated(authed)
        setAuthResolved(true)
      }
    }

    void refreshAuthState()

    const onConfigUpdated = () => {
      void refreshAuthState()
    }
    const onStorage = (event: StorageEvent) => {
      if (!event.key || event.key === "tldwConfig") {
        void refreshAuthState()
      }
    }

    window.addEventListener("tldw:config-updated", onConfigUpdated)
    window.addEventListener("focus", onConfigUpdated)
    window.addEventListener("storage", onStorage)

    return () => {
      cancelled = true
      window.removeEventListener("tldw:config-updated", onConfigUpdated)
      window.removeEventListener("focus", onConfigUpdated)
      window.removeEventListener("storage", onStorage)
    }
  }, [router.asPath])

  React.useEffect(() => {
    if (typeof window === "undefined") return
    if (!authResolved || !isAuthenticated || isPublicAuthRoute) return
    if (didWarmRoutePrefetch.current) return

    const prefetchRoute = router.prefetch?.bind(router)
    if (typeof prefetchRoute !== "function") return

    const connection = (navigator as Navigator & {
      connection?: {
        saveData?: boolean
        effectiveType?: string
      }
    }).connection

    const shouldReducePrefetch =
      connection?.saveData === true ||
      (typeof connection?.effectiveType === "string" &&
        SLOW_EFFECTIVE_TYPES.has(connection.effectiveType))

    const warmPrefetchRoutes = shouldReducePrefetch
      ? PRIMARY_WARM_PREFETCH_ROUTES
      : [...PRIMARY_WARM_PREFETCH_ROUTES, ...SECONDARY_WARM_PREFETCH_ROUTES]

    const routesToPrefetch = warmPrefetchRoutes.filter(
      (targetRoute, index, allRoutes) =>
        targetRoute !== routePath && allRoutes.indexOf(targetRoute) === index
    )
    if (routesToPrefetch.length === 0) return

    didWarmRoutePrefetch.current = true
    let cancelled = false
    // In this mixed DOM + Node type environment, using an explicit numeric
    // handle avoids NodeJS.Timeout incompatibilities with window.setTimeout.
    let prefetchTimeout: number | undefined
    const windowWithIdle = window as Window & {
      requestIdleCallback?: (
        callback: () => void,
        options?: { timeout: number }
      ) => number
      cancelIdleCallback?: (handle: number) => void
    }

    const prefetchRouteAtIndex = (index: number) => {
      if (cancelled || index >= routesToPrefetch.length) return
      void prefetchRoute(routesToPrefetch[index])
        .catch(() => undefined)
        .finally(() => {
          if (cancelled) return
          prefetchTimeout = window.setTimeout(() => {
            prefetchRouteAtIndex(index + 1)
          }, PREFETCH_STEP_DELAY_MS)
        })
    }

    const startPrefetch = () => {
      prefetchRouteAtIndex(0)
    }

    let idleHandle: number | undefined
    if (typeof windowWithIdle.requestIdleCallback === "function") {
      idleHandle = windowWithIdle.requestIdleCallback(startPrefetch, {
        timeout: PREFETCH_IDLE_TIMEOUT_MS
      })
    } else {
      prefetchTimeout = window.setTimeout(
        startPrefetch,
        PREFETCH_FALLBACK_DELAY_MS
      )
    }

    return () => {
      cancelled = true
      if (prefetchTimeout) {
        window.clearTimeout(prefetchTimeout)
      }
      if (
        idleHandle !== undefined &&
        typeof windowWithIdle.cancelIdleCallback === "function"
      ) {
        windowWithIdle.cancelIdleCallback(idleHandle)
      }
    }
  }, [authResolved, isAuthenticated, isPublicAuthRoute, routePath, router])

  const hideShellNav = !authResolved || !isAuthenticated
  const shouldBypassGates = isPublicAuthRoute || isSettingsRoute

  const handleStartSetup = React.useCallback(() => {
    void router.push("/persona")
  }, [router])

  const layoutProps = React.useMemo(
    () => ({
      hideHeader: hideShellNav,
      hideSidebar: hideShellNav || isSettingsRoute,
      allowNestedHideHeader: !isSettingsRoute
    }),
    [hideShellNav, isSettingsRoute]
  )

  const layoutContent = (
    <OptionLayout {...layoutProps}>
      <Component {...pageProps} />
    </OptionLayout>
  )

  const gatedContent = isPublicAuthRoute ? (
    <Component {...pageProps} />
  ) : shouldBypassGates ? (
    layoutContent
  ) : (
    <FirstRunGate onStartSetup={handleStartSetup}>
      {layoutContent}
    </FirstRunGate>
  )

  return (
    <AppProviders>
      <ConfigurationGuard>
        <BackendRecoveryUiProvider routeRecoveryEnabled>
          <ErrorBoundary>
            <ServerReadinessGate bypass={shouldBypassGates}>
              {gatedContent}
            </ServerReadinessGate>
          </ErrorBoundary>
        </BackendRecoveryUiProvider>
      </ConfigurationGuard>
    </AppProviders>
  )
}
