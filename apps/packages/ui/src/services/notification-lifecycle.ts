import { buildChatSurfaceScopeKey, type ChatSurfaceScopeInput } from "@/services/chat-surface-scope"

export type NotificationLifecycleState =
  | "idle"
  | "connecting"
  | "active"
  | "degraded"
  | "auth-required"
  | "unavailable"

export type NotificationLifecycleAction =
  | { type: "start" }
  | { type: "open" }
  | { type: "retry" }
  | { type: "reconnect" }
  | { type: "auth-required" }
  | { type: "unavailable" }
  | { type: "stop" }

export type NotificationErrorClassification =
  | { kind: "idle" }
  | { kind: "auth-required" }
  | { kind: "unavailable" }
  | { kind: "retry"; delayMs: number }

export type NotificationReconnectDelayOptions = {
  attempt: number
  baseDelayMs?: number
  maxDelayMs?: number
  retryAfterSeconds?: number | null
  jitter?: number
}

export type NotificationScopeInput = ChatSurfaceScopeInput

const DEFAULT_RECONNECT_DELAY_MS = 1_200
const MAX_RECONNECT_DELAY_MS = 30_000
const RETRYABLE_STATUSES = new Set([408, 425, 429])

const asRecord = (value: unknown): Record<string, unknown> | null =>
  value !== null && typeof value === "object" ? (value as Record<string, unknown>) : null

const asHttpStatus = (value: unknown): number | null => {
  if (typeof value !== "number" || !Number.isFinite(value)) return null
  const status = Math.trunc(value)
  return status >= 100 && status <= 599 ? status : null
}

const readRetryAfterSeconds = (error: unknown): number | null => {
  const value = asRecord(error)?.retryAfter
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : null
}

const isAbortError = (error: unknown): boolean => {
  const record = asRecord(error)
  return record?.name === "AbortError" || record?.code === "REQUEST_ABORTED"
}

export const readHttpStatus = (error: unknown): number | null => {
  const record = asRecord(error)
  if (!record) return null
  return asHttpStatus(record.status) ?? asHttpStatus(record.statusCode)
}

export const nextReconnectDelay = (options: NotificationReconnectDelayOptions): number => {
  const attempt = Math.max(0, Math.trunc(options.attempt))
  const baseDelayMs = Math.max(0, options.baseDelayMs ?? DEFAULT_RECONNECT_DELAY_MS)
  const maxDelayMs = Math.max(baseDelayMs, options.maxDelayMs ?? MAX_RECONNECT_DELAY_MS)
  const jitter = Math.min(1, Math.max(0, options.jitter ?? 0.5))
  const jitterFactor = 0.8 + jitter * 0.4
  const exponentialDelay = baseDelayMs * 2 ** attempt
  const boundedDelay = Math.min(maxDelayMs, Math.round(exponentialDelay * jitterFactor))
  const retryAfterMs =
    typeof options.retryAfterSeconds === "number" &&
    Number.isFinite(options.retryAfterSeconds) &&
    options.retryAfterSeconds > 0
      ? Math.round(options.retryAfterSeconds * 1_000)
      : 0

  return Math.max(boundedDelay, retryAfterMs)
}

export const classifyNotificationError = (
  error: unknown,
  options: Omit<NotificationReconnectDelayOptions, "retryAfterSeconds"> = {
    attempt: 0
  }
): NotificationErrorClassification => {
  if (isAbortError(error)) return { kind: "idle" }

  const status = readHttpStatus(error)
  if (status === 401) return { kind: "auth-required" }
  if (status !== null && status >= 400 && status < 500 && !RETRYABLE_STATUSES.has(status)) {
    return { kind: "unavailable" }
  }
  if (status === null || RETRYABLE_STATUSES.has(status) || status >= 500) {
    return {
      kind: "retry",
      delayMs: nextReconnectDelay({
        ...options,
        retryAfterSeconds: readRetryAfterSeconds(error)
      })
    }
  }
  return { kind: "unavailable" }
}

export const reduceNotificationLifecycle = (
  state: NotificationLifecycleState,
  action: NotificationLifecycleAction
): NotificationLifecycleState => {
  switch (action.type) {
    case "start":
    case "reconnect":
      return "connecting"
    case "open":
      return "active"
    case "retry":
      return "degraded"
    case "auth-required":
      return "auth-required"
    case "unavailable":
      return "unavailable"
    case "stop":
      return "idle"
    default:
      return state
  }
}

export const buildNotificationScopeKey = (input: NotificationScopeInput): string =>
  `notifications:${buildChatSurfaceScopeKey(input)}`
