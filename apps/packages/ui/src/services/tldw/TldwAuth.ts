import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import { assertAuthConnectionAttempt, normalizeAuthServerUrl, type AuthConnectionAttempt } from "./auth-connection-target"
import type { PathOrUrl } from "./openapi-guard"
import type { TldwConfig } from "./TldwApiClient"
import { tldwClient } from "./TldwApiClient"
import { bgRequest } from "@/services/background-proxy"
import { emitSplashAfterLoginSuccess } from "@/services/splash-events"
import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode"
import { getRuntimeSingleUserApiKeyOverride } from "@/services/tldw/runtime-auth-override"
import { clearSourceReviewHandoffs } from "@/services/tldw/source-review-handoff"
import { clearFlashcardsGenerateHandoffs } from "@/services/tldw/flashcards-generate-handoff"
import { createServicePromptScopeChangedError, isRequestConfigScopeChangedError } from "@/services/tldw/service-prompt-scope-error"
import { clearStandaloneHtmlSessionRecords } from "@/services/tldw/standalone-html-session-records"
import { deriveScopedUserId, deriveTokenOrgId } from "@/utils/media-navigation-scope"

export interface LoginCredentials {
  username: string
  password: string
}

export interface TokenResponse {
  access_token: string
  refresh_token?: string
  token_type: string
  expires_in?: number
}

type OrgListResponse = {
  items?: Array<{ id: number }>
}

export interface UserInfo {
  id: number
  username: string
  email?: string
  role?: string
  is_active: boolean
}

const API_KEY_PROFILE_PATH = "/api/v1/users/me/profile"
const API_KEY_VALIDATION_TIMEOUT_MS = 30000

const emitLogoutPrincipalBoundary = (): void => {
  void clearFlashcardsGenerateHandoffs().catch(() => console.warn("Could not clear private Flashcards transfers during sign-out."))
  if (typeof window === "undefined") return
  window.dispatchEvent(
    new CustomEvent("tldw:auth-principal-changed", {
      detail: { kind: "logout" }
    })
  )
}

const buildApiKeyValidationUrl = (serverUrl: string): string => {
  const trimmed = String(serverUrl || "").trim()
  if (!trimmed) {
    throw new Error("tldw server not configured")
  }
  return new URL(API_KEY_PROFILE_PATH, `${trimmed.replace(/\/+$/, "")}/`).toString()
}

export class TldwAuthService {
  private refreshTimer: NodeJS.Timeout | null = null
  private refreshInFlight: Promise<TokenResponse> | null = null

  constructor() {
  }

  private isHostedMode(): boolean {
    return isHostedTldwDeployment()
  }

  private async ensureHostedOrgId(
    expected?: AuthConnectionAttempt,
    assertCurrent?: (config: Partial<TldwConfig> | null) => void,
    assertCommitted?: (config: Partial<TldwConfig> | null) => void
  ): Promise<void> {
    const checkCurrent = async () => {
      if (assertCurrent) assertCurrent(await this.readLoginConfig(expected))
    }
    const saveOrg = async (orgId: number) => {
      if (assertCurrent) await tldwClient.updateConfig({ orgId }, assertCurrent, assertCommitted)
      else await tldwClient.updateConfig({ orgId })
    }
    await checkCurrent()
    try {
      const orgs = await bgRequest<OrgListResponse>({
        path: "/api/v1/orgs",
        method: "GET"
      })
      await checkCurrent()
      const existingId = orgs?.items?.[0]?.id
      if (existingId) {
        await saveOrg(existingId)
        return
      }
    } catch (error) {
      if (isRequestConfigScopeChangedError(error)) throw error
      // Continue to the hosted profile fallback below.
    }

    await checkCurrent()
    try {
      const profile = await tldwClient.getCurrentUserProfile({
        includeRaw: true
      })
      await checkCurrent()
      const activeOrgId = Number(
        profile?.active_org_id ??
        profile?.org_id ??
        profile?.raw?.active_org_id ??
        0
      )
      if (Number.isFinite(activeOrgId) && activeOrgId > 0) {
        await saveOrg(activeOrgId)
      }
    } catch (error) {
      if (isRequestConfigScopeChangedError(error)) throw error
      // best-effort only
    }
  }

  /** Bind Settings sign-in to its visible target across service and storage awaits. */
  private async withConnectionAttempt<T>(
    expected: AuthConnectionAttempt | undefined,
    run: (
      assertCurrent?: (config: Partial<TldwConfig> | null) => void,
      assertCommitted?: (config: Partial<TldwConfig> | null) => void
    ) => Promise<T>
  ): Promise<T> {
    if (!expected) return run()
    const target = { ...expected.target }
    let invalidated = false
    const unwatch = watchChatAccountChanges(changed => { if (changed) invalidated = true })
    const assertCurrent = (config: Partial<TldwConfig> | null) => {
      if (invalidated) throw createServicePromptScopeChangedError()
      assertAuthConnectionAttempt({ ...expected, target }, config)
    }
    try {
      return await run(assertCurrent, config => {
        // The write itself can change account authority. Its own event is valid;
        // the view lifetime and the actual committed target must still match.
        assertAuthConnectionAttempt({ target, signal: expected.signal }, config)
      })
    } finally {
      unwatch()
    }
  }

  private async readLoginConfig(expected?: AuthConnectionAttempt): Promise<TldwConfig | null> {
    // getConfig can return a cached target. Settings authorizes committed storage.
    if (!expected) return tldwClient.getConfig()
    try {
      await tldwClient.initialize()
      return await tldwClient.getConfig()
    } catch {
      throw createServicePromptScopeChangedError()
    }
  }

  private authPath(path: `/api/v1/auth/${string}`, hostedMode: boolean, expected?: AuthConnectionAttempt): PathOrUrl {
    if (hostedMode) return path.replace("/api/v1/auth/", "/api/auth/") as PathOrUrl
    return expected ? (normalizeAuthServerUrl(expected.target.serverUrl) + path) as PathOrUrl : path
  }

  /** Login for multi-user mode. */
  async login(credentials: LoginCredentials, expected?: AuthConnectionAttempt): Promise<TokenResponse> {
    return this.withConnectionAttempt(expected, async (assertCurrent, assertCommitted) => {
      const hostedMode = this.isHostedMode()
      const config = await this.readLoginConfig(expected)
      assertCurrent?.(config)
      if (!config && !hostedMode) throw new Error("tldw server not configured")
      const formData = new URLSearchParams({ username: credentials.username, password: credentials.password })
      const tokens = await bgRequest<TokenResponse, PathOrUrl>({
        path: this.authPath("/api/v1/auth/login", hostedMode, expected),
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: formData.toString(),
        noAuth: true
      })
      assertCurrent?.(config)
      assertCurrent?.(await this.readLoginConfig(expected))
      await this.publishLogin(tokens, hostedMode, assertCurrent, assertCommitted, expected)
      return tokens
    })
  }

  /** Request a magic link sign-in email. */
  async requestMagicLink(email: string, expected?: AuthConnectionAttempt): Promise<void> {
    return this.withConnectionAttempt(expected, async assertCurrent => {
      const hostedMode = this.isHostedMode()
      const config = await this.readLoginConfig(expected)
      assertCurrent?.(config)
      if (!config && !hostedMode) throw new Error("tldw server not configured")
      await bgRequest({
        path: this.authPath("/api/v1/auth/magic-link/request", hostedMode, expected),
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: { email },
        noAuth: true
      })
      assertCurrent?.(config)
      assertCurrent?.(await this.readLoginConfig(expected))
    })
  }

  /** Verify a magic link token and sign in. */
  async verifyMagicLink(token: string, expected?: AuthConnectionAttempt): Promise<TokenResponse> {
    return this.withConnectionAttempt(expected, async (assertCurrent, assertCommitted) => {
      const hostedMode = this.isHostedMode()
      const config = await this.readLoginConfig(expected)
      assertCurrent?.(config)
      if (!config && !hostedMode) throw new Error("tldw server not configured")
      const tokens = await bgRequest<TokenResponse, PathOrUrl>({
        path: this.authPath("/api/v1/auth/magic-link/verify", hostedMode, expected),
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: { token },
        noAuth: true
      })
      assertCurrent?.(config)
      assertCurrent?.(await this.readLoginConfig(expected))
      await this.publishLogin(tokens, hostedMode, assertCurrent, assertCommitted, expected)
      return tokens
    })
  }

  private async publishLogin(
    tokens: TokenResponse,
    hostedMode: boolean,
    assertCurrent?: (config: Partial<TldwConfig> | null) => void,
    assertCommitted?: (config: Partial<TldwConfig> | null) => void,
    expected?: AuthConnectionAttempt
  ): Promise<void> {
    const update: Partial<TldwConfig> = {
      authMode: "multi-user",
      accessToken: hostedMode ? undefined : tokens.access_token,
      refreshToken: hostedMode ? undefined : tokens.refresh_token,
      ...(hostedMode ? {} : { orgId: deriveTokenOrgId(tokens.access_token) })
    }
    if (assertCurrent) await tldwClient.updateConfig(update, assertCurrent, assertCommitted)
    else await tldwClient.updateConfig(update)
    if (hostedMode) {
      // The login write is now authoritative. Start the metadata phase from it,
      // keeping lifecycle cancellation while observing subsequent account changes.
      const committedAttempt = expected ? { target: { ...expected.target }, signal: expected.signal } : undefined
      await this.withConnectionAttempt(committedAttempt, (current, committed) =>
        this.ensureHostedOrgId(committedAttempt, current, committed))
    }
    if (expected) assertAuthConnectionAttempt({ target: expected.target, signal: expected.signal }, expected.target)
    if (!hostedMode && tokens.expires_in) this.setupTokenRefresh(tokens.expires_in)
    emitSplashAfterLoginSuccess()
  }

  /**
   * Logout and clear tokens
   */
  async logout(): Promise<void> {
    const config = await tldwClient.getConfig()
    if (!config) {
      return
    }
    if (config.authMode === 'single-user') {
      if (config.authSource === 'cookie-session') {
        await bgRequest<any>({
          path: '/api/v1/auth/single-user/session',
          method: 'DELETE'
        })
        await tldwClient.clearCookieSingleUserSession()
        emitLogoutPrincipalBoundary()
        return
      }
      await tldwClient.clearManualSingleUserCredentials()
      emitLogoutPrincipalBoundary()
      return
    }

    try {
      await bgRequest<any>({
        path: this.isHostedMode() ? '/api/auth/logout' : '/api/v1/auth/logout',
        method: 'POST'
      })
    } catch {
      console.warn('Server logout unavailable; continuing local sign-out without confirmed remote revocation.')
    }

    clearSourceReviewHandoffs()

    // Clear local tokens
    await tldwClient.updateConfig({
      accessToken: undefined,
      refreshToken: undefined
    })

    // Clear refresh timer
    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer)
      this.refreshTimer = null
    }

    clearStandaloneHtmlSessionRecords()
    emitLogoutPrincipalBoundary()
  }

  /**
   * Refresh access token using refresh token.
   *
   * Single-flighted: concurrent callers (e.g. the pre-expiry timer racing a 401
   * refresh) share one in-flight request so the backend's rotating refresh
   * token is not spent twice, which would persist a dead token.
   */
  async refreshToken(): Promise<TokenResponse> {
    if (this.refreshInFlight) {
      return this.refreshInFlight
    }
    this.refreshInFlight = this.performTokenRefresh().finally(() => {
      this.refreshInFlight = null
    })
    return this.refreshInFlight
  }

  private async performTokenRefresh(): Promise<TokenResponse> {
    const refreshTimer = this.refreshTimer
    await tldwClient.initialize()
    const config = await tldwClient.getConfig()
    if (!config || !config.refreshToken) {
      throw new Error('No refresh token available')
    }

    const scopedUserId = deriveScopedUserId({
      userId: null,
      authMode: config.authMode,
      accessToken: config.accessToken ?? null
    })
    const expectedUserId = scopedUserId === "user:anonymous"
      ? null
      : scopedUserId.slice("user:".length)

    let tokens: TokenResponse
    try {
      tokens = await bgRequest<TokenResponse>({
        path: '/api/v1/auth/refresh',
        method: 'POST',
        // A background worker cannot send this request back to its own listener.
        preferDirect: typeof window === "undefined",
        headers: { 'Content-Type': 'application/json' },
        body: { refresh_token: config.refreshToken },
        servicePromptConfig: {
          serverUrl: config.serverUrl,
          authMode: config.authMode,
          authSource: config.authSource,
          orgId: config.orgId,
          expectedUserId,
          expectedRefreshToken: config.refreshToken
        }
      })
    } catch (error) {
      if ((error as { status?: number } | null)?.status === 401) {
        if (!await tldwClient.invalidateRefreshSession(config)) {
          throw createServicePromptScopeChangedError()
        }
        if (this.refreshTimer && this.refreshTimer === refreshTimer) {
          clearTimeout(this.refreshTimer)
          this.refreshTimer = null
        }
      }
      throw error
    }

    const committed = await tldwClient.commitTokenRefresh(
      config,
      config.refreshToken,
      {
        accessToken: tokens.access_token,
        refreshToken: tokens.refresh_token || config.refreshToken
      }
    )
    if (!committed) throw createServicePromptScopeChangedError()

    // Set up auto-refresh if expires_in is provided
    if (tokens.expires_in) {
      this.setupTokenRefresh(tokens.expires_in)
    }

    return tokens
  }

  /**
   * (Re-)arm the pre-expiry refresh timer after a page load.
   *
   * The timer set during login/verify/refresh is discarded on reload, so a
   * multi-user user who reloads would otherwise never auto-refresh and every
   * request would 401 once the access token expired. Safe to call repeatedly:
   * it is a no-op in hosted mode, when a timer is already armed, or when no
   * refresh token is present. When a refresh token exists but no timer is
   * scheduled it performs a single refresh, which rotates the access token and
   * re-arms the timer via setupTokenRefresh.
   */
  async initTokenRefresh(): Promise<void> {
    if (this.isHostedMode()) return
    if (this.refreshTimer) return
    const config = await tldwClient.getConfig()
    if (!config || config.authMode !== 'multi-user') return
    if (!config.refreshToken) return
    try {
      await this.refreshToken()
    } catch (error) {
      console.error('Token refresh on init failed:', error)
    }
  }

  /**
   * Get current user information
   */
  async getCurrentUser(): Promise<UserInfo> {
    const hostedMode = this.isHostedMode()
    const config = await tldwClient.getConfig()
    if (!config && !hostedMode) {
      throw new Error('tldw server not configured')
    }

    if (hostedMode) {
      const session = await bgRequest<{
        authenticated?: boolean
        user?: UserInfo
      }>({
        path: '/api/auth/session',
        method: 'GET',
        noAuth: true
      })
      if (!session?.authenticated || !session.user) {
        throw Object.assign(new Error('Not authenticated'), {
          status: 401,
          code: 'not_authenticated'
        })
      }
      return session.user
    }

    const me = await bgRequest<UserInfo>({ path: '/api/v1/auth/me', method: 'GET' })
    return me
  }

  /**
   * Register a new user (if registration is enabled)
   */
  async register(username: string, password: string, email?: string, registrationCode?: string): Promise<any> {
    const hostedMode = this.isHostedMode()
    const config = await tldwClient.getConfig()
    if (!config && !hostedMode) {
      throw new Error('tldw server not configured')
    }

    try {
      const data = await bgRequest<any>({
        path: hostedMode ? '/api/auth/register' : '/api/v1/auth/register',
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: { username, password, email, registration_code: registrationCode },
        noAuth: true
      })
      return data
    } catch (e: any) {
      throw new Error(e?.message || 'Registration failed')
    }
  }

  /**
   * Test API key for single-user mode
   */
  async testApiKey(serverUrl: string, apiKey: string): Promise<boolean> {
    // Validate against the candidate setup endpoint, not the currently
    // persisted config. First-run setup may be correcting a bad saved URL.
    const controller = new AbortController()
    const timeoutId = setTimeout(
      () => controller.abort(),
      API_KEY_VALIDATION_TIMEOUT_MS
    )
    try {
      const validationUrl = buildApiKeyValidationUrl(serverUrl)
      const response = await fetch(validationUrl, {
        method: 'GET',
        headers: { 'X-API-KEY': apiKey },
        credentials: 'omit',
        signal: controller.signal
      })

      if (response.ok) {
        return true
      }

      if (response.status === 401 || response.status === 403) {
        return false
      }

      const message =
        (await response.text().catch(() => "")) ||
        response.statusText ||
        `HTTP ${response.status}`
      const responseError = new Error(message) as Error & { status?: number }
      responseError.status = response.status
      throw responseError
    } catch (error: any) {
      const status = Number(
        error?.status ?? error?.statusCode ?? error?.response?.status ?? 0
      )
      const message = String(error?.message || error || "")
      const normalized = message.toLowerCase()
      const isAbort =
        error?.name === "AbortError" ||
        normalized.includes("aborted") ||
        normalized.includes("timeout")

      console.error("API key test failed:", message || error)

      if (status === 401 || status === 403) {
        return false
      }

      if (isAbort) {
        const connectionError = new Error(
          "API key validation timed out or was aborted. Verify server URL/connectivity and try again."
        ) as Error & { status?: number }
        connectionError.status = 0
        throw connectionError
      }

      throw error
    } finally {
      clearTimeout(timeoutId)
    }
  }

  /**
   * Set up automatic token refresh
   */
  private setupTokenRefresh(expiresIn: number): void {
    // Clear existing timer
    if (this.refreshTimer) {
      clearTimeout(this.refreshTimer)
    }

    // Refresh 5 minutes before expiry
    const refreshIn = Math.max(0, (expiresIn - 300) * 1000)

    this.refreshTimer = setTimeout(async () => {
      try {
        await this.refreshToken()
      } catch (error) {
        console.error('Auto token refresh failed:', error)
        // Could emit an event here to notify UI
      }
    }, refreshIn)
  }

  /**
   * Check if user is authenticated
   */
  async isAuthenticated(): Promise<boolean> {
    const config = await tldwClient.getConfig()
    if (!config) {
      return false
    }

    if (this.isHostedMode()) {
      return config.authMode === 'multi-user'
    }

    if (config.authMode === 'single-user') {
      return config.authSource === 'cookie-session' || !!config.apiKey
    } else if (config.authMode === 'multi-user') {
      return !!config.accessToken
    }

    return false
  }

  /**
   * Get authentication headers
   */
  async getAuthHeaders(): Promise<HeadersInit> {
    const config = await tldwClient.getConfig()
    const headers: HeadersInit = {}
    const hostedMode = this.isHostedMode()

    if (!hostedMode) {
      const runtimeApiKey = getRuntimeSingleUserApiKeyOverride()
      if (runtimeApiKey) {
        headers['X-API-KEY'] = runtimeApiKey
        return headers
      }
    }

    if (!config) {
      return headers
    }

    if (hostedMode) {
      if (config.orgId) {
        headers['X-TLDW-Org-Id'] = String(config.orgId)
      }
      return headers
    }

    if (config.authMode === 'single-user' && config.apiKey) {
      headers['X-API-KEY'] = config.apiKey
    } else if (config.authMode === 'multi-user' && config.accessToken) {
      headers['Authorization'] = `Bearer ${config.accessToken}`
    }

    return headers
  }
}

// Singleton instance
export const tldwAuth = new TldwAuthService()
