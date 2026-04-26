import { tldwClient } from "./TldwApiClient"
import { bgRequest } from "@/services/background-proxy"
import { emitSplashAfterLoginSuccess } from "@/services/splash-events"
import { isHostedTldwDeployment } from "@/services/tldw/deployment-mode"

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

type OrgDetailResponse = {
  id: number
}

export interface UserInfo {
  id: number
  username: string
  email?: string
  role?: string
  is_active: boolean
}

export class TldwAuthService {
  private refreshTimer: NodeJS.Timeout | null = null

  constructor() {
  }

  private isHostedMode(): boolean {
    return isHostedTldwDeployment()
  }

  private async ensureOrgId(): Promise<void> {
    try {
      const orgs = await bgRequest<OrgListResponse>({
        path: "/api/v1/orgs",
        method: "GET"
      })
      const existingId = orgs?.items?.[0]?.id
      if (existingId) {
        await tldwClient.updateConfig({ orgId: existingId })
        return
      }
    } catch {
      // ignore and continue to hosted fallback or self-host create below
    }

    if (this.isHostedMode()) {
      try {
        const profile = await tldwClient.getCurrentUserProfile({
          includeRaw: true
        })
        const activeOrgId = Number(
          profile?.active_org_id ??
          profile?.org_id ??
          profile?.raw?.active_org_id ??
          0
        )
        if (Number.isFinite(activeOrgId) && activeOrgId > 0) {
          await tldwClient.updateConfig({ orgId: activeOrgId })
        }
      } catch {
        // best-effort only
      }
      return
    }

    try {
      const created = await bgRequest<OrgDetailResponse>({
        path: "/api/v1/orgs",
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: { name: "Personal Workspace" }
      })
      if (created?.id) {
        await tldwClient.updateConfig({ orgId: created.id })
      }
    } catch {
      // best-effort only
    }
  }

  /**
   * Login for multi-user mode
   */
  async login(credentials: LoginCredentials): Promise<TokenResponse> {
    const hostedMode = this.isHostedMode()
    const config = await tldwClient.getConfig()
    if (!config && !hostedMode) {
      throw new Error('tldw server not configured')
    }

    const formData = new URLSearchParams()
    formData.append('username', credentials.username)
    formData.append('password', credentials.password)

    const response = await bgRequest<any>({
      path: hostedMode ? '/api/auth/login' : '/api/v1/auth/login',
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: formData.toString(),
      noAuth: true
    })
    const tokens = response as TokenResponse

    await tldwClient.updateConfig({
      authMode: 'multi-user',
      accessToken: hostedMode ? undefined : tokens.access_token,
      refreshToken: hostedMode ? undefined : tokens.refresh_token
    })

    await this.ensureOrgId()

    if (!hostedMode && tokens.expires_in) {
      this.setupTokenRefresh(tokens.expires_in)
    }

    emitSplashAfterLoginSuccess()
    return tokens
  }

  /**
   * Request a magic link sign-in email
   */
  async requestMagicLink(email: string): Promise<void> {
    const hostedMode = this.isHostedMode()
    const config = await tldwClient.getConfig()
    if (!config && !hostedMode) {
      throw new Error('tldw server not configured')
    }
    await bgRequest<any>({
      path: hostedMode ? '/api/auth/magic-link/request' : '/api/v1/auth/magic-link/request',
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: { email },
      noAuth: true
    })
  }

  /**
   * Verify a magic link token and sign in
   */
  async verifyMagicLink(token: string): Promise<TokenResponse> {
    const hostedMode = this.isHostedMode()
    const config = await tldwClient.getConfig()
    if (!config && !hostedMode) {
      throw new Error('tldw server not configured')
    }

    const tokens = await bgRequest<TokenResponse>({
      path: hostedMode ? '/api/auth/magic-link/verify' : '/api/v1/auth/magic-link/verify',
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: { token },
      noAuth: true
    })

    await tldwClient.updateConfig({
      authMode: 'multi-user',
      accessToken: hostedMode ? undefined : tokens.access_token,
      refreshToken: hostedMode ? undefined : tokens.refresh_token
    })

    await this.ensureOrgId()

    if (!hostedMode && tokens.expires_in) {
      this.setupTokenRefresh(tokens.expires_in)
    }

    emitSplashAfterLoginSuccess()
    return tokens
  }

  /**
   * Logout and clear tokens
   */
  async logout(): Promise<void> {
    const config = await tldwClient.getConfig()
    if (!config || config.authMode !== 'multi-user') {
      return
    }

    try {
      await bgRequest<any>({
        path: this.isHostedMode() ? '/api/auth/logout' : '/api/v1/auth/logout',
        method: 'POST'
      })
    } catch (error) {
      console.error('Server logout failed:', error)
    }

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
  }

  /**
   * Refresh access token using refresh token
   */
  async refreshToken(): Promise<TokenResponse> {
    const config = await tldwClient.getConfig()
    if (!config || !config.refreshToken) {
      throw new Error('No refresh token available')
    }

    const tokens = await bgRequest<TokenResponse>({
      path: '/api/v1/auth/refresh',
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: { refresh_token: config.refreshToken }
    })

    const latestConfig = await tldwClient.getConfig()

    // Persist rotated refresh tokens; the backend rotates them by default.
    await tldwClient.updateConfig({
      accessToken: tokens.access_token,
      refreshToken:
        tokens.refresh_token ||
        latestConfig?.refreshToken ||
        config.refreshToken
    })

    // Set up auto-refresh if expires_in is provided
    if (tokens.expires_in) {
      this.setupTokenRefresh(tokens.expires_in)
    }

    return tokens
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
        throw new Error('Not authenticated')
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
  async testApiKey(_serverUrl: string, apiKey: string): Promise<boolean> {
    // Validate against a protected endpoint that requires auth.
    // Keep this as a relative path so request-core does not apply
    // absolute URL allowlist policy during onboarding validation.
    try {
      // Use /api/v1/users/me/profile which requires valid authentication
      await bgRequest<any>({
        path: '/api/v1/users/me/profile' as any,
        method: 'GET' as any,
        headers: { 'X-API-KEY': apiKey },
        noAuth: true,
        timeoutMs: 30000
      })
      return true
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
      return !!config.apiKey
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

    if (!config) {
      return headers
    }

    if (this.isHostedMode()) {
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
