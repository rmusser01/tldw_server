import {
  type RecipePersistenceOwnerView,
  deriveRecipePersistenceOwner
} from "@/services/recipe-persistence-owner"
import type { RecipeDeliveryReceipt } from "@/services/recipe-persistence-registry"
import {
  type BrowserSurface,
  buildBrowserHttpBase,
  isCookieSessionBrowserTransport,
  resolveBrowserTransport
} from "@/services/tldw/browser-networking"
import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode"
import { isPlaceholderApiKey } from "@/utils/api-key"

export type RecipePersistenceRequestPolicy =
  | Readonly<{ mode: "capture" }>
  | Readonly<{
      mode: "require"
      expectedOwnerId: string
      localId: string
    }>

export type RecipePersistenceDispatch =
  | Readonly<{ state: "not_dispatched"; actualOwnerId: null }>
  | Readonly<{ state: "dispatched"; actualOwnerId: string | null }>
  | Readonly<{ state: "unknown"; actualOwnerId: null }>

export type RecipeRequestTransportSnapshot = Readonly<{
  url: string
  effectiveBase: string
  headers: Readonly<Record<string, string>>
  credentials?: RequestCredentials
}>

export type RecipeOwnerResolution = Readonly<{
  view: RecipePersistenceOwnerView
  snapshot: RecipeRequestTransportSnapshot
}>

export type RecipeDispatchAuthority = Readonly<{
  markDispatched(
    localId: string,
    ownerId: string
  ): RecipeDeliveryReceipt | void | Promise<RecipeDeliveryReceipt | void>
}>

export type BrowserRequestTransport = {
  mode: "hosted" | "quickstart" | "advanced"
  kind: "same-origin" | "absolute"
  url: string
}

type SnapshotAuthenticationError = Readonly<{
  status: number
  error: string
}>

export type RecipeRequestSnapshotResolution = Readonly<{
  view: RecipePersistenceOwnerView | null
  snapshot: RecipeRequestTransportSnapshot
  authenticationError?: SnapshotAuthenticationError
}>

type RecipeRequestSnapshotInput = Readonly<{
  config: RecipeRequestConfig | null | undefined
  path: string
  method: string
  headers?: Readonly<Record<string, string>>
  noAuth?: boolean
  runtimeApiKey?: string | null
  authenticatedPrincipalId?: string | number | null
  cookieSessionRevision?: string | null
  csrfToken?: string | null
  pageOrigin?: string | null
  absoluteAuthAllowed?: boolean
  cookieSessionTransport?: boolean
}>

type RecipeRequestConfig = Readonly<{
  serverUrl?: unknown
  authMode?: unknown
  authSource?: unknown
  apiKey?: unknown
  accessToken?: unknown
  orgId?: unknown
}>

const toHostedProxyPath = (path: string): string => {
  const [pathname, search = ""] = path.split("?")
  if (pathname.startsWith("/api/v1/")) {
    const proxiedPath = pathname.replace(/^\/api\/v1\//, "/api/proxy/")
    return search ? `${proxiedPath}?${search}` : proxiedPath
  }
  return path
}

const getCurrentBrowserSurface = (): BrowserSurface => {
  if (typeof window === "undefined") return "extension"
  try {
    const protocol = String(window.location?.protocol || "")
      .trim()
      .toLowerCase()
    if (protocol === "chrome-extension:" || protocol === "moz-extension:") {
      return "extension"
    }
    if (protocol === "http:" || protocol === "https:") return "webui-page"
  } catch {
    // Fall through to the browser-app default.
  }
  return "browser-app"
}

const normalizeHttpBase = (value: string): string => {
  try {
    const url = new URL(value.trim())
    if (url.protocol !== "http:" && url.protocol !== "https:")
      return value.trim()
    url.pathname = url.pathname.replace(/\/+$/, "") || "/"
    url.search = ""
    url.hash = ""
    return url.toString().replace(/\/$/, "")
  } catch {
    return value.trim().replace(/\/+$/, "")
  }
}

const joinBaseAndPath = (base: string, path: string): string =>
  `${base.replace(/\/+$/, "")}${path.startsWith("/") ? "" : "/"}${path}`

export const resolveBrowserRequestTransport = ({
  config,
  path,
  pageOrigin
}: {
  config: RecipeRequestConfig | null | undefined
  path: string
  pageOrigin?: string | null
}): BrowserRequestTransport => {
  if (isHostedTldwDeployment()) {
    return {
      mode: "hosted",
      kind: "same-origin",
      url: toHostedProxyPath(path)
    }
  }

  const configuredServerUrl = String(config?.serverUrl || "").trim()
  const surface = getCurrentBrowserSurface()
  if (surface === "webui-page") {
    try {
      const resolved = resolveBrowserTransport({
        surface,
        deploymentMode: process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE,
        pageOrigin:
          pageOrigin ??
          (typeof window === "undefined"
            ? null
            : String(window.location?.origin || "").trim()),
        apiOrigin: configuredServerUrl || process.env.NEXT_PUBLIC_API_URL
      })
      const browserHttpBase = buildBrowserHttpBase(resolved)
      if (!browserHttpBase) {
        return { mode: "quickstart", kind: "same-origin", url: path }
      }
      return {
        mode: "advanced",
        kind: "absolute",
        url: joinBaseAndPath(browserHttpBase, path)
      }
    } catch {
      // Fall through to explicit configured server handling below.
    }
  }

  return {
    mode: "advanced",
    kind: "absolute",
    url: joinBaseAndPath(normalizeHttpBase(configuredServerUrl), path)
  }
}

export const isUnsafeMethod = (method: string): boolean =>
  !new Set(["GET", "HEAD", "OPTIONS", "TRACE"]).has(method.toUpperCase())

const normalizeBaseWithTrailingSlash = (value: string): string => {
  const url = new URL(value)
  url.pathname = `${url.pathname.replace(/\/+$/, "")}/`
  url.search = ""
  url.hash = ""
  return url.toString()
}

const effectiveBaseFor = (
  transport: BrowserRequestTransport | null,
  path: string,
  pageOrigin: string | null,
  absolutePath: boolean
): string => {
  if (transport?.kind === "same-origin") {
    return normalizeBaseWithTrailingSlash(pageOrigin || "http://localhost")
  }
  const requestUrl = transport?.url || path
  if (absolutePath) {
    const absolute = new URL(requestUrl)
    return normalizeBaseWithTrailingSlash(absolute.origin)
  }
  const suffix = path.startsWith("/") ? path : `/${path}`
  const withoutPath = requestUrl.endsWith(suffix)
    ? requestUrl.slice(0, -suffix.length)
    : new URL(requestUrl).origin
  return normalizeBaseWithTrailingSlash(withoutPath)
}

const cookieRevision = (provided?: string | null): string => {
  if (typeof provided === "string" && provided.trim()) return provided
  if (
    typeof crypto !== "undefined" &&
    typeof crypto.randomUUID === "function"
  ) {
    return crypto.randomUUID()
  }
  return `${Date.now()}:${Math.random()}`
}

const deleteManagedAuthHeaders = (headers: Record<string, string>): void => {
  for (const key of Object.keys(headers)) {
    const normalized = key.toLowerCase()
    if (
      normalized === "x-api-key" ||
      normalized === "authorization" ||
      normalized === "x-csrf-token" ||
      normalized === "x-tldw-org-id"
    ) {
      delete headers[key]
    }
  }
}

export const resolveRecipeRequestSnapshot = (
  input: RecipeRequestSnapshotInput
): RecipeRequestSnapshotResolution => {
  const config = input.config || {}
  const pageOrigin =
    input.pageOrigin ??
    (typeof window === "undefined"
      ? null
      : String(window.location?.origin || ""))
  const absolutePath = /^https?:/i.test(input.path)
  const transport = absolutePath
    ? null
    : resolveBrowserRequestTransport({ config, path: input.path, pageOrigin })
  const url = absolutePath ? input.path : transport?.url || input.path
  const effectiveBase = effectiveBaseFor(
    transport,
    input.path,
    pageOrigin,
    absolutePath
  )
  const headers: Record<string, string> = { ...(input.headers || {}) }
  const hostedMode = transport?.mode === "hosted"
  const cookieSession =
    input.cookieSessionTransport ??
    isCookieSessionBrowserTransport({
      authMode: config.authMode,
      authSource: config.authSource,
      transportMode: transport?.mode,
      transportKind: transport?.kind,
      pageOrigin
    })
  const shouldSkipAuth = Boolean(
    input.noAuth || (absolutePath && !input.absoluteAuthAllowed)
  )
  const ownerEligible = !hostedMode && !input.noAuth && !absolutePath
  const normalizedRuntimeKey = String(input.runtimeApiKey || "").trim()
  const runtimeKeyEligible =
    !hostedMode &&
    !shouldSkipAuth &&
    normalizedRuntimeKey.length > 0 &&
    !isPlaceholderApiKey(normalizedRuntimeKey)
  const configuredOrgId = String(config.orgId ?? "").trim() || null
  const effectiveOrgId =
    !hostedMode && !shouldSkipAuth && (!cookieSession || runtimeKeyEligible)
      ? configuredOrgId
      : null
  let credentials: RequestCredentials | undefined
  let view: RecipePersistenceOwnerView | null = null
  let authenticationError: SnapshotAuthenticationError | undefined

  deleteManagedAuthHeaders(headers)
  if (runtimeKeyEligible) {
    headers["X-API-KEY"] = normalizedRuntimeKey
    view = deriveRecipePersistenceOwner(
      {
        effectiveBase,
        authMode: "single-user",
        authSource: "runtime_api_key",
        orgId: effectiveOrgId,
        principalKind: "api_key",
        principal: normalizedRuntimeKey
      },
      normalizedRuntimeKey
    )
  } else if (cookieSession && !shouldSkipAuth) {
    if (isUnsafeMethod(input.method) && input.csrfToken) {
      headers["X-CSRF-Token"] = input.csrfToken
    }
    credentials = "same-origin"
    const principal = String(input.authenticatedPrincipalId ?? "").trim()
    if (principal) {
      view = deriveRecipePersistenceOwner(
        {
          effectiveBase,
          authMode: "multi-user",
          authSource: "cookie_session",
          orgId: effectiveOrgId,
          principalKind: "user",
          principal
        },
        cookieRevision(input.cookieSessionRevision)
      )
    }
  } else if (!hostedMode && !shouldSkipAuth) {
    const authSourceSupported =
      config.authSource === null ||
      config.authSource === undefined ||
      config.authSource === "" ||
      config.authSource === "manual" ||
      config.authSource === "cookie-session"
    if (config.authMode === "single-user") {
      const key = String(config.apiKey || "").trim()
      if (!key) {
        authenticationError = {
          status: 401,
          error: normalizedRuntimeKey
            ? "tldw server API key is still set to a placeholder value. Replace it with your real API key in Settings -> tldw server before continuing."
            : "Add or update your API key in Settings -> tldw server, then try again."
        }
      } else if (isPlaceholderApiKey(key)) {
        authenticationError = {
          status: 401,
          error:
            "tldw server API key is still set to a placeholder value. Replace it with your real API key in Settings -> tldw server before continuing."
        }
      } else {
        headers["X-API-KEY"] = key
        if (authSourceSupported) {
          view = deriveRecipePersistenceOwner(
            {
              effectiveBase,
              authMode: "single-user",
              authSource: "manual_api_key",
              orgId: effectiveOrgId,
              principalKind: "api_key",
              principal: key
            },
            key
          )
        }
      }
    } else if (config.authMode === "multi-user") {
      const token = String(config.accessToken || "").trim()
      if (!token) {
        authenticationError = {
          status: 401,
          error: "Not authenticated. Please login under Settings > tldw."
        }
      } else {
        headers.Authorization = `Bearer ${token}`
        const principal = String(input.authenticatedPrincipalId ?? "").trim()
        if (principal && authSourceSupported) {
          view = deriveRecipePersistenceOwner(
            {
              effectiveBase,
              authMode: "multi-user",
              authSource: "manual_bearer",
              orgId: effectiveOrgId,
              principalKind: "user",
              principal
            },
            token
          )
        }
      }
    }
  }

  if (effectiveOrgId) {
    headers["X-TLDW-Org-Id"] = effectiveOrgId
  }
  if (!ownerEligible) view = null

  const snapshot = Object.freeze({
    url,
    effectiveBase,
    headers: Object.freeze({ ...headers }),
    ...(credentials ? { credentials } : {})
  })
  return Object.freeze({
    view: view ? Object.freeze(view) : null,
    snapshot,
    ...(authenticationError ? { authenticationError } : {})
  })
}
