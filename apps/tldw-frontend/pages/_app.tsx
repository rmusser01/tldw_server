import "../styles/globals.css"
import "@/assets/react-pdf.css"
import { runtimeBootstrapReady } from "@web/extension/shims/runtime-bootstrap"
// Use web-specific i18n that works with SSR/static generation
import { i18nNamespacesReady } from "@web/lib/i18n-web"
import type { AppProps } from "next/app"
import dynamic from "next/dynamic"
import { useRouter } from "next/router"
import React from "react"
import { BackendRecoveryUiProvider } from "@/components/Common/BackendRecoveryUiContext"
import { PageAssistLoader } from "@/components/Common/PageAssistLoader"
import { FirstRunGate } from "@/components/PersonaGarden/FirstRunGate"
import { AppProviders } from "@web/components/AppProviders"
import ErrorBoundary from "@web/components/ErrorBoundary"
import { ConfigurationGuard } from "@web/components/networking/ConfigurationGuard"
import { ServerReadinessGate } from "@web/components/networking/ServerReadinessGate"
import {
  getRuntimeApiBearer,
  getRuntimeApiKey,
  hasActiveCookieSessionAuth,
  hasEnvApiAuth
} from "@web/lib/authStorage"
import { loadTldwAuth, loadTldwClient } from "@web/lib/configured-auth-state"
import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode"
import {
  buildFirstRunOnboardingRoute,
  CHARACTER_CHAT_ONBOARDING_INTENT,
  getOnboardingReturnToFromSearch,
  resolveOnboardingEntryIntent
} from "@/utils/onboarding-route-intent"

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

const DEGRADED_READINESS_ROUTES = new Set(["/chat", "/research-workspace"])

const PREFETCH_STEP_DELAY_MS = 250
const PREFETCH_IDLE_TIMEOUT_MS = 2000
const PREFETCH_FALLBACK_DELAY_MS = 1200
const SLOW_EFFECTIVE_TYPES = new Set(["slow-2g", "2g"])

type ConfiguredAuthState = {
  hasConfig: boolean
  authMode?: "single-user" | "multi-user"
  isAuthenticated: boolean
  serverUrl?: string | null
}

const getErrorStatus = (error: unknown): number | null => {
  const candidate = error as
    | {
        status?: unknown
        statusCode?: unknown
        response?: { status?: unknown }
      }
    | null
  const rawStatus =
    candidate?.status ?? candidate?.statusCode ?? candidate?.response?.status
  return typeof rawStatus === "number" && Number.isFinite(rawStatus)
    ? rawStatus
    : null
}

const isAuthValidationFailure = (error: unknown): boolean => {
  const status = getErrorStatus(error)
  if (status === 401 || status === 403) return true
  const candidate = error as { message?: unknown } | null
  const message =
    error instanceof Error
      ? error.message
      : typeof candidate?.message === "string"
        ? candidate.message
        : ""
  return message.trim() === "Not authenticated"
}

const splitRouteAsPath = (asPath: string) => {
  const fallback = asPath || "/"
  const hashIndex = fallback.indexOf("#")
  const withoutHash = hashIndex >= 0 ? fallback.slice(0, hashIndex) : fallback
  const hash = hashIndex >= 0 ? fallback.slice(hashIndex) : ""
  const searchIndex = withoutHash.indexOf("?")
  const pathname =
    searchIndex >= 0 ? withoutHash.slice(0, searchIndex) || "/" : withoutHash || "/"
  const search = searchIndex >= 0 ? withoutHash.slice(searchIndex) : ""

  return {
    pathname,
    search,
    hash
  }
}

const buildFirstRunSetupRoute = (asPath: string): string => {
  const routeParts = splitRouteAsPath(asPath)
  const entryIntent = resolveOnboardingEntryIntent(routeParts)

  if (entryIntent !== CHARACTER_CHAT_ONBOARDING_INTENT) {
    return "/"
  }

  if (routeParts.pathname === "/") {
    const returnTo = getOnboardingReturnToFromSearch(routeParts.search)
    if (returnTo) {
      const params = new URLSearchParams({
        intent: CHARACTER_CHAT_ONBOARDING_INTENT
      })
      params.set("returnTo", returnTo)
      return `/?${params.toString()}`
    }
  }

  return buildFirstRunOnboardingRoute(routeParts)
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
    const serverUrl =
      typeof config.serverUrl === "string" ? config.serverUrl : null

    if (config.authMode === "multi-user") {
      const hostedMode = isHostedTldwDeployment()
      const hasAccessToken =
        typeof config.accessToken === "string" &&
        config.accessToken.trim().length > 0
      if (!hasAccessToken && !hostedMode) {
        return {
          hasConfig: true,
          authMode: "multi-user",
          isAuthenticated: false,
          serverUrl
        }
      }

      const tldwAuth = await loadTldwAuth()
      try {
        await tldwAuth.getCurrentUser()
        return {
          hasConfig: true,
          authMode: "multi-user",
          isAuthenticated: true,
          serverUrl
        }
      } catch (error) {
        if (!isAuthValidationFailure(error)) {
          return {
            hasConfig: true,
            authMode: "multi-user",
            isAuthenticated: true,
            serverUrl
          }
        }
        try {
          await tldwAuth.logout?.()
        } catch (logoutError) {
          console.warn("Failed to clear stale tldw auth session:", logoutError)
        }
        return {
          hasConfig: true,
          authMode: "multi-user",
          isAuthenticated: false,
          serverUrl
        }
      }
    }

    return {
      hasConfig: true,
      authMode: "single-user",
      serverUrl,
      isAuthenticated:
        hasActiveCookieSessionAuth(config) ||
        (typeof config.apiKey === "string" && config.apiKey.trim().length > 0)
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
  const isSetupRoute = routePath === "/setup"
  const isDebugRoute =
    routePath === "/__debug__" || routePath.startsWith("/__debug__/")
  const isSidepanelDebugRoute = routePath === "/__debug__/sidepanel-chat"
  const isSettingsRoute =
    routePath === "/settings" || routePath.startsWith("/settings/")
  const shouldBypassGates =
    isPublicAuthRoute || isSettingsRoute || isSetupRoute || isDebugRoute
  const [isAuthenticated, setIsAuthenticated] = React.useState(false)
  const [configuredServerUrl, setConfiguredServerUrl] =
    React.useState<string | null>(null)
  const [authResolved, setAuthResolved] = React.useState(false)
  const didWarmRoutePrefetch = React.useRef(false)

  React.useEffect(() => {
    if (typeof window === "undefined") return

    let cancelled = false
    const refreshAuthState = async () => {
      // Translation namespaces load as separate chunks; wait for them here so no
      // page renders before its namespace exists. These fetches run in parallel
      // with the bootstrap below, so they add no round trip of their own.
      await Promise.all([
        Promise.resolve(runtimeBootstrapReady).catch(() => undefined),
        // Promise.resolve() so a missing or failed i18n module degrades to
        // untranslated keys rather than pinning the app on the loading screen.
        Promise.resolve(i18nNamespacesReady).catch(() => undefined)
      ])
      if (cancelled) return

      const envAuthed =
        hasEnvApiAuth() ||
        Boolean(getRuntimeApiKey()) ||
        Boolean(getRuntimeApiBearer())
      const configuredAuth = await getConfiguredAuthState()
      const authed = configuredAuth.hasConfig
        ? configuredAuth.authMode === "multi-user"
          ? configuredAuth.isAuthenticated
          : configuredAuth.isAuthenticated || envAuthed
        : envAuthed

      if (!cancelled) {
        setIsAuthenticated(authed)
        setConfiguredServerUrl(configuredAuth.serverUrl ?? null)
        setAuthResolved(true)
        if (
          !authed &&
          configuredAuth.hasConfig &&
          configuredAuth.authMode === "multi-user" &&
          !shouldBypassGates
        ) {
          void router.push("/login")
        }
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
  }, [router, router.asPath, shouldBypassGates])

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
  const shouldAllowDegradedReadiness = DEGRADED_READINESS_ROUTES.has(routePath)
  const firstRunRouteParts = React.useMemo(
    () => splitRouteAsPath(router.asPath || routePath || "/"),
    [routePath, router.asPath]
  )
  const firstRunEntryIntent = React.useMemo(
    () => resolveOnboardingEntryIntent(firstRunRouteParts),
    [firstRunRouteParts]
  )
  const firstRunSetupRoute = React.useMemo(
    () => buildFirstRunSetupRoute(router.asPath || routePath || "/"),
    [routePath, router.asPath]
  )
  const shouldBypassFirstRunOverlay =
    !shouldBypassGates &&
    (routePath === "/" ||
      routePath === "/research-workspace" ||
      firstRunEntryIntent === CHARACTER_CHAT_ONBOARDING_INTENT)

  const handleStartSetup = React.useCallback(() => {
    void router.push(firstRunSetupRoute)
  }, [firstRunSetupRoute, router])

  const layoutProps = React.useMemo(
    () => ({
      hideHeader: hideShellNav || isSetupRoute || isSidepanelDebugRoute,
      hideSidebar:
        hideShellNav || isSettingsRoute || isSetupRoute || isSidepanelDebugRoute,
      allowNestedHideHeader: !isSettingsRoute
    }),
    [hideShellNav, isSettingsRoute, isSetupRoute, isSidepanelDebugRoute]
  )
  const enableNotifications =
    authResolved && isAuthenticated && !isPublicAuthRoute && !isSetupRoute

  if (!authResolved) {
    return <PageAssistLoader label="Loading..." autoFocus={false} />
  }

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
    <FirstRunGate
      onStartSetup={handleStartSetup}
      bypass={shouldBypassFirstRunOverlay}>
      {layoutContent}
    </FirstRunGate>
  )

  return (
    <AppProviders enableNotifications={enableNotifications}>
      <ConfigurationGuard>
        <BackendRecoveryUiProvider routeRecoveryEnabled>
          <ErrorBoundary>
            <ServerReadinessGate
              bypass={shouldBypassGates}
              allowDegraded={shouldAllowDegradedReadiness}
              configuredServerUrl={configuredServerUrl}>
              {gatedContent}
            </ServerReadinessGate>
          </ErrorBoundary>
        </BackendRecoveryUiProvider>
      </ConfigurationGuard>
    </AppProviders>
  )
}
