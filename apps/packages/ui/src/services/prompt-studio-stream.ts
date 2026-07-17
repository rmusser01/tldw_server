import type { TldwConfig } from "@/services/tldw/TldwApiClient"
import {
  resolveCookieSessionWebSocketBase,
  resolveBrowserWebSocketBase,
} from "@/services/tldw/browser-websocket"

export const buildPromptStudioWebSocketUrl = (
  config: Pick<
    TldwConfig,
    "serverUrl" | "authMode" | "authSource" | "apiKey" | "accessToken"
  >,
  projectId?: number | null
): string => {
  const serverUrl = String(config.serverUrl || "").trim()
  if (!serverUrl) {
    throw new Error("tldw server is not configured")
  }

  const cookieBase = resolveCookieSessionWebSocketBase(config)
  const cookieSession = Boolean(cookieBase)
  const base = cookieBase || resolveBrowserWebSocketBase(serverUrl)
  if (!base) throw new Error("WebUI origin is not available")
  const params = new URLSearchParams()

  if (!cookieSession) {
    if (config.authMode === "multi-user") {
      const token = String(config.accessToken || "").trim()
      if (!token) {
        throw new Error("Not authenticated. Please log in under Settings.")
      }
      params.set("token", token)
    } else {
      const apiKey = String(config.apiKey || "").trim()
      if (!apiKey) {
        throw new Error("API key missing. Update Settings -> tldw server.")
      }
      params.set("api_key", apiKey)
    }
  }

  if (typeof projectId === "number" && Number.isFinite(projectId)) {
    params.set("project_id", String(projectId))
  }

  const query = params.toString()
  return `${base}/api/v1/prompt-studio/ws${query ? `?${query}` : ""}`
}

const STATUS_EVENT_TYPES = new Set([
  "job_created",
  "job_started",
  "job_progress",
  "job_completed",
  "job_failed",
  "job_cancelled",
  "job_retrying",
  "evaluation_started",
  "evaluation_progress",
  "evaluation_completed",
  "optimization_started",
  "optimization_iteration",
  "optimization_completed",
  "subscribed",
  "job_update"
])

export const isPromptStudioStatusEvent = (payload: unknown): boolean => {
  if (!payload || typeof payload !== "object" || Array.isArray(payload)) return false
  const type = (payload as Record<string, unknown>).type
  return typeof type === "string" && STATUS_EVENT_TYPES.has(type)
}
