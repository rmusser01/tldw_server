/**
 * Per-model readiness for the chat model chip (CC-01, #3111).
 *
 * The chip used to show the tldw server connection as "Healthy", even when the
 * selected model's provider was not running. Readiness here describes the
 * selected model only, from the evidence that exists:
 *   - what happened the last time this session sent to the model, and
 *   - the server's provider health report (/api/v1/llm/providers `health`).
 * Without either, the model is "not checked" rather than presumed healthy.
 */
import {
  normalizeProviderAvailabilityKey,
  toProviderHealthSignal
} from "@/services/tldw/model-provider-availability"
import { splitChatModelId } from "@/utils/chat-model-id"

export type ChatModelSendOutcome = "working" | "unreachable"

export type ChatModelReadinessState = "not_checked" | "working" | "unreachable"

const outcomes = new Map<string, ChatModelSendOutcome>()
const listeners = new Set<() => void>()
let outcomesVersion = 0

const toTrimmedString = (value: unknown): string => {
  if (typeof value !== "string" && typeof value !== "number") return ""
  return String(value).trim()
}

/**
 * A stable key for one provider's model. Accepts a bare model id with a
 * separate provider, or a provider-qualified id such as `ollama:llama3`. Ollama
 * ids such as `mistral:7b` and `qwen:7b` stay whole, so they get their own keys.
 */
export const chatModelReadinessKey = (
  provider: unknown,
  model: unknown
): string | null => {
  const parts = splitChatModelId(model, provider)
  const modelId = parts.modelId.toLowerCase()
  if (!modelId) return null
  const providerKey =
    normalizeProviderAvailabilityKey(parts.provider || toTrimmedString(provider)) ?? ""
  return `${providerKey}:${modelId}`
}

const notify = () => {
  outcomesVersion += 1
  for (const listener of listeners) listener()
}

/** Remember how the latest send to a model ended, for this page session. */
export const recordChatModelSendOutcome = (
  provider: unknown,
  model: unknown,
  outcome: ChatModelSendOutcome
) => {
  const key = chatModelReadinessKey(provider, model)
  if (!key || outcomes.get(key) === outcome) return
  outcomes.set(key, outcome)
  notify()
}

export const getChatModelSendOutcome = (
  provider: unknown,
  model: unknown
): ChatModelSendOutcome | null => {
  const key = chatModelReadinessKey(provider, model)
  return key ? outcomes.get(key) ?? null : null
}

export const subscribeChatModelSendOutcomes = (listener: () => void) => {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

export const getChatModelSendOutcomesVersion = () => outcomesVersion

/** Test helper: forget every recorded outcome. */
export const resetChatModelSendOutcomes = () => {
  if (outcomes.size === 0) return
  outcomes.clear()
  notify()
}

const UNREACHABLE_ERROR_CODES = new Set([
  "provider_unavailable",
  "provider_error",
  "upstream_error",
  "upstream_unavailable",
  "service_unavailable",
  "bad_gateway",
  "gateway_timeout"
])

const readStatus = (value: Record<string, unknown>): number | null => {
  for (const candidate of [
    value.status,
    value.statusCode,
    (value.response as Record<string, unknown> | undefined)?.status
  ]) {
    if (typeof candidate === "number" && Number.isFinite(candidate)) {
      return Math.trunc(candidate)
    }
  }
  const message = typeof value.message === "string" ? value.message : ""
  const match = message.match(/\bhttp\s+(\d{3})\b/i)
  return match ? Number(match[1]) : null
}

const collectErrorCodes = (value: unknown, codes: string[], depth = 0) => {
  if (!value || typeof value !== "object" || depth > 3) return
  const record = value as Record<string, unknown>
  for (const key of ["code", "error_code", "type"]) {
    const code = record[key]
    if (typeof code === "string" && code.trim()) codes.push(code.trim().toLowerCase())
  }
  for (const key of ["detail", "details", "error", "data", "body"]) {
    collectErrorCodes(record[key], codes, depth + 1)
  }
}

/**
 * True when a failed chat request shows the model's provider could not answer:
 * a 5xx from the server or a provider-unavailable error code. Client-side
 * aborts, timeouts and validation errors are not evidence either way.
 */
export const isChatModelUnreachableError = (error: unknown): boolean => {
  let current: unknown = error
  for (let depth = 0; depth < 5 && current; depth += 1) {
    if (typeof current !== "object") return false
    const record = current as Record<string, unknown>
    const status = readStatus(record)
    if (status !== null && status >= 500 && status <= 599) return true
    const codes: string[] = []
    collectErrorCodes(record, codes)
    if (codes.some((code) => UNREACHABLE_ERROR_CODES.has(code))) return true
    current = record.cause
  }
  return false
}

type ReadinessModelDescriptor = Record<string, unknown> | null | undefined

/** Readiness of one model from this session's outcome and the provider report. */
export const resolveChatModelReadinessState = ({
  model,
  provider,
  modelId
}: {
  model?: ReadinessModelDescriptor
  provider: unknown
  modelId: unknown
}): ChatModelReadinessState => {
  const outcome = getChatModelSendOutcome(provider, modelId)
  if (outcome) return outcome
  const health = toProviderHealthSignal(model?.provider_health)
  if (health === "working") return "working"
  if (health === "unreachable") return "unreachable"
  return "not_checked"
}
