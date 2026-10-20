/**
 * What the client knows about a chat send that got no reply (CC-02, CC-N2,
 * CM-06): the HTTP status, error code and message the server answered with,
 * the provider and model it went to, or how long the client waited before it
 * gave up. Used to word the error a user sees and to keep it with a retained
 * turn, so the same explanation survives a reload.
 */
import type { HistoryTurnFailure } from "@/db/dexie/types"
import { ChatStreamTimeoutError } from "@/services/tldw/chat-timeouts"
import {
  clearModelProviderFailure,
  recordModelProviderFailure
} from "@/store/model-provider-failures"
import { getProviderDisplayName } from "@/utils/provider-registry"
import { parseProviderQualifiedModelSelection } from "@/utils/resolve-api-provider"

const asRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null

const causes = function* (error: unknown) {
  const seen = new Set<unknown>()
  let current = error
  while (current && typeof current === "object" && !seen.has(current)) {
    seen.add(current)
    yield current as Record<string, unknown>
    current = (current as { cause?: unknown }).cause
  }
}

/**
 * The HTTP status, error code and message of a failed request, read from the
 * error or the errors it wraps (the chat client wraps transport errors in
 * `cause`). The server's JSON body is in `details`, e.g.
 * `{ detail: { error_code: "provider_unavailable", message: "..." } }`.
 */
export const findHttpFailure = (
  error: unknown
): Pick<HistoryTurnFailure, "status" | "code" | "message"> => {
  const found: Pick<HistoryTurnFailure, "status" | "code" | "message"> = {}
  for (const entry of causes(error)) {
    if (
      found.status === undefined &&
      typeof entry.status === "number" &&
      entry.status >= 400 &&
      entry.status <= 599
    )
      found.status = entry.status
    const details = asRecord(entry.details)
    for (const body of [details, asRecord(details?.detail), asRecord(entry.detail)]) {
      if (!body) continue
      const code = body.error_code ?? body.code
      if (!found.code && typeof code === "string" && code.trim()) found.code = code.trim()
      if (!found.message && typeof body.message === "string" && body.message.trim())
        found.message = body.message.trim()
    }
  }
  return found
}

/** The provider and model a selected model id names, for messages. */
export const sendFailureTarget = (
  selectedModel: string | null | undefined
): Pick<HistoryTurnFailure, "provider" | "model"> => {
  const selection = parseProviderQualifiedModelSelection(selectedModel)
  const model = selection.modelId?.trim()
  const provider = selection.provider?.trim()
  return {
    ...(provider ? { provider } : {}),
    ...(model ? { model } : {})
  }
}

/** A provider's display name, e.g. "custom-openai-api" -> "Custom OpenAI API". */
export const providerDisplayLabel = (provider: string | undefined): string | undefined => {
  if (!provider) return undefined
  const direct = getProviderDisplayName(provider)
  if (direct !== provider) return direct
  const underscored = getProviderDisplayName(provider.replace(/-/g, "_"))
  return underscored !== provider.replace(/-/g, "_") ? underscored : provider
}

/** What is known about why a send got no reply, or undefined when nothing is. */
export const describeSendFailure = (
  error: unknown,
  target: Pick<HistoryTurnFailure, "provider" | "model"> = {}
): HistoryTurnFailure | undefined => {
  for (const entry of causes(error)) {
    if (entry instanceof ChatStreamTimeoutError && entry.phase === "startup")
      return { ...target, timeout_ms: entry.timeoutMs }
  }
  const http = findHttpFailure(error)
  return http.status === undefined ? undefined : { ...http, ...target }
}

/**
 * A failure that settles the turn: the server answered with an error before
 * any reply arrived, so no reply exists to recover. Conflicts (409) keep their
 * own history handling. A client-side startup timeout settles a turn whose
 * reply only the client would save; a server-settled reply may still arrive.
 */
export const isDefinitiveSendFailure = (
  failure: HistoryTurnFailure | undefined,
  persistence?: "server" | "client"
): failure is HistoryTurnFailure =>
  Boolean(failure) &&
  ((failure!.status !== undefined && failure!.status !== 409) ||
    (failure!.timeout_ms !== undefined && persistence !== "server"))

/** A provider failure: the tldw server answered, the model's provider did not. */
export const isProviderUnavailableFailure = (
  failure: Pick<HistoryTurnFailure, "status" | "code" | "message"> | undefined
): boolean => {
  if (!failure) return false
  if (failure.code === "provider_authentication_failed") return false
  return (
    failure.code === "provider_unavailable" ||
    failure.code === "provider_error" ||
    failure.status === 502 ||
    failure.status === 503 ||
    /provider is currently unavailable/i.test(failure.message ?? "")
  )
}

/** Rebuild an error equivalent to a retained failure, for its error message. */
export const sendFailureError = (failure: HistoryTurnFailure): Error => {
  if (failure.timeout_ms !== undefined)
    return new ChatStreamTimeoutError("startup", failure.timeout_ms)
  const error = new Error(failure.message || `HTTP ${failure.status ?? ""}`.trim()) as Error & {
    status?: number
    details?: unknown
  }
  if (failure.status !== undefined) error.status = failure.status
  error.details = {
    detail: {
      ...(failure.code ? { error_code: failure.code } : {}),
      ...(failure.message ? { message: failure.message } : {})
    }
  }
  return error
}

/**
 * Note whether a model's provider answered a send (CC-N2): a provider failure
 * marks the model unreachable for this session, and a reply clears that. A
 * slow first token is not counted: the provider may simply be busy.
 */
export const noteModelProviderOutcome = (
  selectedModel: string | null | undefined,
  error?: unknown
) => {
  if (error === undefined) {
    clearModelProviderFailure(selectedModel)
    return
  }
  const failure = describeSendFailure(error, sendFailureTarget(selectedModel))
  if (failure && failure.timeout_ms === undefined && isProviderUnavailableFailure(failure))
    recordModelProviderFailure(selectedModel, {
      ...(failure.provider ? { provider: failure.provider } : {}),
      ...(failure.model ? { model: failure.model } : {})
    })
}
