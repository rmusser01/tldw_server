import type { TldwConfig } from "@/services/tldw/TldwApiClient"
import {
  resolveBrowserWebSocketBase,
  resolveCurrentPageWebSocketBase
} from "@/services/tldw/browser-websocket"

type StreamEventType = "snapshot" | "run_update" | "log" | "complete" | "heartbeat"

interface StreamRunPayload {
  id: number
  job_id: number
  status: string
  started_at?: string | null
  finished_at?: string | null
}

export interface WatchlistsRunStreamSnapshot {
  type: "snapshot"
  run: StreamRunPayload
  stats: Record<string, number>
  error_msg?: string | null
  log_tail?: string | null
  log_truncated?: boolean
}

export interface WatchlistsRunStreamUpdate {
  type: "run_update"
  run: StreamRunPayload
  stats: Record<string, number>
  error_msg?: string | null
}

export interface WatchlistsRunStreamLog {
  type: "log"
  text: string
}

export interface WatchlistsRunStreamComplete {
  type: "complete"
  status?: string
}

export interface WatchlistsRunStreamHeartbeat {
  type: "heartbeat"
}

export type WatchlistsRunStreamEvent =
  | WatchlistsRunStreamSnapshot
  | WatchlistsRunStreamUpdate
  | WatchlistsRunStreamLog
  | WatchlistsRunStreamComplete
  | WatchlistsRunStreamHeartbeat

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value)

const isStreamType = (value: unknown): value is StreamEventType =>
  typeof value === "string" &&
  ["snapshot", "run_update", "log", "complete", "heartbeat"].includes(value)

const asInteger = (value: unknown): number | null => {
  if (typeof value === "number" && Number.isInteger(value)) return value
  if (typeof value === "string" && value.trim().length > 0) {
    const parsed = Number.parseInt(value, 10)
    if (Number.isInteger(parsed)) return parsed
  }
  return null
}

const asRunPayload = (value: unknown): StreamRunPayload | null => {
  if (!isRecord(value)) return null
  const id = asInteger(value.id)
  const jobId = asInteger(value.job_id)
  const status = typeof value.status === "string" ? value.status : null
  if (id == null || jobId == null || !status) return null

  return {
    id,
    job_id: jobId,
    status,
    started_at: typeof value.started_at === "string" ? value.started_at : null,
    finished_at: typeof value.finished_at === "string" ? value.finished_at : null
  }
}

const asStats = (value: unknown): Record<string, number> => {
  if (!isRecord(value)) return {}
  const out: Record<string, number> = {}
  Object.entries(value).forEach(([key, raw]) => {
    if (typeof raw === "number" && Number.isFinite(raw)) {
      out[key] = raw
      return
    }
    if (typeof raw === "string" && raw.trim().length > 0) {
      const numeric = Number(raw)
      if (Number.isFinite(numeric)) out[key] = numeric
    }
  })
  return out
}

export const parseWatchlistsRunStreamPayload = (
  payload: unknown
): WatchlistsRunStreamEvent | null => {
  if (!isRecord(payload) || !isStreamType(payload.type)) return null

  if (payload.type === "log") {
    if (typeof payload.text !== "string") return null
    return { type: "log", text: payload.text }
  }

  if (payload.type === "heartbeat") {
    return { type: "heartbeat" }
  }

  if (payload.type === "complete") {
    return {
      type: "complete",
      status: typeof payload.status === "string" ? payload.status : undefined
    }
  }

  if (payload.type === "snapshot") {
    const run = asRunPayload(payload.run)
    if (!run) return null
    return {
      type: "snapshot",
      run,
      stats: asStats(payload.stats),
      error_msg: typeof payload.error_msg === "string" ? payload.error_msg : null,
      log_tail: typeof payload.log_tail === "string" ? payload.log_tail : null,
      log_truncated: payload.log_truncated === true
    }
  }

  const run = asRunPayload(payload.run)
  if (!run) return null
  return {
    type: "run_update",
    run,
    stats: asStats(payload.stats),
    error_msg: typeof payload.error_msg === "string" ? payload.error_msg : null
  }
}

export const buildWatchlistsRunWebSocketUrl = (
  config: Pick<
    TldwConfig,
    "serverUrl" | "authMode" | "authSource" | "apiKey" | "accessToken"
  >,
  runId: number
): string => {
  const normalizedRunId = Math.floor(Number(runId))
  if (!Number.isInteger(normalizedRunId) || normalizedRunId <= 0) {
    throw new Error("Invalid run id")
  }

  const serverUrl = String(config.serverUrl || "").trim()
  if (!serverUrl) {
    throw new Error("tldw server is not configured")
  }

  const cookieSession = config.authSource === "cookie-session"
  const base = cookieSession
    ? resolveCurrentPageWebSocketBase()
    : resolveBrowserWebSocketBase(serverUrl)
  if (!base) throw new Error("WebUI origin is not available")
  if (cookieSession) {
    return `${base}/api/v1/watchlists/runs/${normalizedRunId}/stream`
  }

  const params = new URLSearchParams()

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

  return `${base}/api/v1/watchlists/runs/${normalizedRunId}/stream?${params.toString()}`
}
