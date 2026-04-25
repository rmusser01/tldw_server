/**
 * ACP REST and WebSocket Client
 */

import type {
  ACPAgentListResponse,
  ACPSessionDetailResponse,
  ACPSessionForkRequest,
  ACPSessionForkResponse,
  ACPSessionListResponse,
  ACPSessionNewRequest,
  ACPSessionNewResponse,
  ACPSessionPromptRequest,
  ACPSessionPromptResponse,
  ACPSessionUsageResponse,
  ACPWSClientMessage,
  ACPWSServerMessage,
} from "./types"
import { shouldRetryACPWebSocketClose } from "./constants"
import { resolveBrowserRequestTransport } from "@/services/tldw/request-core"
import { resolveBrowserWebSocketBase } from "@/services/tldw/browser-websocket"

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

export interface ACPClientConfig {
  serverUrl: string
  getAuthHeaders: () => Promise<Record<string, string>>
  getAuthParams: () => Promise<{ token?: string; api_key?: string }>
}

// -----------------------------------------------------------------------------
// REST Client
// -----------------------------------------------------------------------------

export class ACPRestClient {
  constructor(private config: ACPClientConfig) {}

  private buildQuery(params?: Record<string, string | number | boolean | null | undefined>): string {
    if (!params) {
      return ""
    }

    const searchParams = new URLSearchParams()
    for (const [key, value] of Object.entries(params)) {
      if (value === undefined || value === null || value === "") {
        continue
      }
      searchParams.set(key, String(value))
    }

    const query = searchParams.toString()
    return query ? `?${query}` : ""
  }

  private async request<T>(
    path: string,
    options: RequestInit = {}
  ): Promise<T> {
    const transport = resolveBrowserRequestTransport({
      config: { serverUrl: this.config.serverUrl },
      path
    })
    const headers =
      transport.mode === "hosted" ? {} : await this.config.getAuthHeaders()

    const response = await fetch(transport.url, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...headers,
        ...options.headers,
      },
    })

    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({ detail: response.statusText }))
      const detail =
        errorBody && typeof errorBody === "object" && "detail" in errorBody
          ? errorBody.detail
          : response.statusText
      const requestError = new Error(
        typeof detail === "string" && detail.trim().length > 0
          ? detail
          : `HTTP ${response.status}`
      ) as Error & {
        status?: number
        code?: string
        detail?: unknown
        details?: unknown
      }
      requestError.status = response.status
      requestError.code =
        errorBody && typeof errorBody === "object"
          ? typeof (errorBody as Record<string, unknown>).error_code === "string"
            ? (errorBody as Record<string, unknown>).error_code as string
            : typeof (errorBody as Record<string, unknown>).code === "string"
              ? (errorBody as Record<string, unknown>).code as string
              : undefined
          : undefined
      requestError.detail = detail
      requestError.details = errorBody
      throw requestError
    }

    return response.json()
  }

  /**
   * Get list of available agents and their configuration status
   */
  async getAvailableAgents(): Promise<ACPAgentListResponse> {
    return this.request<ACPAgentListResponse>("/api/v1/acp/agents")
  }

  /**
   * Create a new ACP session
   */
  async createSession(request: ACPSessionNewRequest): Promise<ACPSessionNewResponse> {
    return this.request<ACPSessionNewResponse>("/api/v1/acp/sessions/new", {
      method: "POST",
      body: JSON.stringify(request),
    })
  }

  /**
   * Send a prompt to an existing session
   */
  async sendPrompt(request: ACPSessionPromptRequest): Promise<ACPSessionPromptResponse> {
    return this.request<ACPSessionPromptResponse>("/api/v1/acp/sessions/prompt", {
      method: "POST",
      body: JSON.stringify(request),
    })
  }

  /**
   * Cancel the current operation in a session
   */
  async cancelSession(sessionId: string): Promise<void> {
    await this.request<{ status: string }>("/api/v1/acp/sessions/cancel", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId }),
    })
  }

  /**
   * Close and cleanup a session
   */
  async closeSession(sessionId: string): Promise<void> {
    await this.request<{ status: string }>("/api/v1/acp/sessions/close", {
      method: "POST",
      body: JSON.stringify({ session_id: sessionId }),
    })
  }

  /**
   * List ACP sessions for the authenticated user
   */
  async listSessions(params?: {
    status?: string
    agent_type?: string
    limit?: number
    offset?: number
  }): Promise<ACPSessionListResponse> {
    const query = this.buildQuery(params)
    return this.request<ACPSessionListResponse>(`/api/v1/acp/sessions${query}`)
  }

  /**
   * Get ACP session detail including message history
   */
  async getSessionDetail(sessionId: string): Promise<ACPSessionDetailResponse> {
    return this.request<ACPSessionDetailResponse>(
      `/api/v1/acp/sessions/${encodeURIComponent(sessionId)}/detail`
    )
  }

  /**
   * Get ACP session usage totals
   */
  async getSessionUsage(sessionId: string): Promise<ACPSessionUsageResponse> {
    return this.request<ACPSessionUsageResponse>(
      `/api/v1/acp/sessions/${encodeURIComponent(sessionId)}/usage`
    )
  }

  /**
   * Fork an existing ACP session
   */
  async forkSession(
    sessionId: string,
    request: ACPSessionForkRequest
  ): Promise<ACPSessionForkResponse> {
    return this.request<ACPSessionForkResponse>(
      `/api/v1/acp/sessions/${encodeURIComponent(sessionId)}/fork`,
      {
        method: "POST",
        body: JSON.stringify(request),
      }
    )
  }

  /**
   * Poll for updates (fallback when WebSocket unavailable)
   */
  async getUpdates(sessionId: string, limit = 100): Promise<Array<Record<string, unknown>>> {
    const response = await this.request<{ updates: Array<Record<string, unknown>> }>(
      `/api/v1/acp/sessions/${sessionId}/updates?limit=${limit}`
    )
    return response.updates
  }
}

// -----------------------------------------------------------------------------
// WebSocket Client
// -----------------------------------------------------------------------------

export interface ACPWebSocketCallbacks {
  onOpen?: () => void
  onClose?: (code: number, reason: string) => void
  onError?: (error: Event) => void
  onMessage?: (message: ACPWSServerMessage) => void
}

export class ACPWebSocketClient {
  private ws: WebSocket | null = null
  private reconnectAttempts = 0
  private reconnectTimeout: ReturnType<typeof setTimeout> | null = null
  private isClosedManually = false

  constructor(
    private config: ACPClientConfig,
    private callbacks: ACPWebSocketCallbacks = {}
  ) {}

  /**
   * Connect to the WebSocket endpoint for a session
   */
  async connect(sessionId: string): Promise<void> {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      console.warn("WebSocket already connected")
      return
    }

    this.isClosedManually = false
    const authParams = await this.config.getAuthParams()

    // Build WebSocket URL
    const wsUrl = resolveBrowserWebSocketBase(this.config.serverUrl)
    const params = new URLSearchParams()
    if (authParams.token) params.set("token", authParams.token)
    if (authParams.api_key) params.set("api_key", authParams.api_key)

    const url = `${wsUrl}/api/v1/acp/sessions/${sessionId}/stream?${params.toString()}`

    this.ws = new WebSocket(url)

    this.ws.onopen = () => {
      this.reconnectAttempts = 0
      this.callbacks.onOpen?.()
    }

    this.ws.onclose = (event) => {
      this.callbacks.onClose?.(event.code, event.reason)

      // Attempt reconnection if not manually closed
      if (!this.isClosedManually && shouldRetryACPWebSocketClose(event.code)) {
        this.scheduleReconnect(sessionId)
      }
    }

    this.ws.onerror = (event) => {
      this.callbacks.onError?.(event)
    }

    this.ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as ACPWSServerMessage
        this.callbacks.onMessage?.(message)
      } catch (e) {
        console.error("Failed to parse WebSocket message:", e)
      }
    }
  }

  /**
   * Send a message to the server
   */
  send(message: ACPWSClientMessage): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error("WebSocket not connected")
    }
    this.ws.send(JSON.stringify(message))
  }

  /**
   * Send a permission response
   */
  respondToPermission(
    requestId: string,
    approved: boolean,
    batchApproveTier?: string
  ): void {
    this.send({
      type: "permission_response",
      request_id: requestId,
      approved,
      batch_approve_tier: batchApproveTier as any,
    })
  }

  /**
   * Send a cancel request
   */
  cancelOperation(sessionId: string): void {
    this.send({
      type: "cancel",
      session_id: sessionId,
    })
  }

  /**
   * Send a prompt via WebSocket
   */
  sendPrompt(
    sessionId: string,
    prompt: Array<{ role: "system" | "user" | "assistant"; content: string }>
  ): void {
    this.send({
      type: "prompt",
      session_id: sessionId,
      prompt,
    })
  }

  /**
   * Close the WebSocket connection
   */
  close(): void {
    this.isClosedManually = true
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout)
      this.reconnectTimeout = null
    }
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }

  /**
   * Check if connected
   */
  get isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN
  }

  /**
   * Get the current WebSocket state
   */
  get state(): number | null {
    return this.ws?.readyState ?? null
  }

  private scheduleReconnect(sessionId: string): void {
    if (this.reconnectAttempts >= 10) {
      console.error("Max reconnection attempts reached")
      return
    }

    const delay = Math.min(
      1000 * Math.pow(1.5, this.reconnectAttempts),
      30000
    )

    this.reconnectTimeout = setTimeout(() => {
      this.reconnectAttempts++
      console.log(`Attempting reconnection ${this.reconnectAttempts}/10...`)
      this.connect(sessionId).catch(console.error)
    }, delay)
  }
}

// -----------------------------------------------------------------------------
// Combined Client Factory
// -----------------------------------------------------------------------------

export function createACPClient(config: ACPClientConfig) {
  return {
    rest: new ACPRestClient(config),
    createWebSocket: (callbacks: ACPWebSocketCallbacks = {}) =>
      new ACPWebSocketClient(config, callbacks),
  }
}
