import {
  PUBLIC_RAG_PROVIDER_ERROR_MESSAGES,
  asPublicRagProviderErrorCode,
  asValidatedHttpStatus,
  getValidatedHttpStatus,
} from "@/services/rag/provider-error-contract"

const toErrorString = (error: unknown): string => {
  if (error instanceof Error) return error.message
  if (typeof error === "string") return error
  return ""
}

const isConnectionError = (message: string): boolean =>
  /network|offline|failed to fetch|connection|unreachable/i.test(message)

const isTimeoutError = (message: string): boolean =>
  /timeout|timed out|etimedout/i.test(message)

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const getTerminalErrorEvent = (
  error: unknown
): Record<string, unknown> | null => {
  if (
    !isRecord(error) ||
    !isRecord(error.event) ||
    error.event.type !== "error"
  ) {
    return null
  }
  return error.event
}

const getPublicProviderCode = (error: unknown) => {
  const terminalEvent = getTerminalErrorEvent(error)
  if (terminalEvent) {
    return (
      asPublicRagProviderErrorCode(terminalEvent.code) ?? "provider_unavailable"
    )
  }
  return isRecord(error) ? asPublicRagProviderErrorCode(error.code) : null
}

const getSearchErrorStatus = (error: unknown): number | undefined => {
  const terminalEvent = getTerminalErrorEvent(error)
  return terminalEvent
    ? asValidatedHttpStatus(terminalEvent.status_code)
    : getValidatedHttpStatus(error)
}

export const getKnowledgeQaSearchErrorLogCode = (error: unknown): string => {
  const providerCode = getPublicProviderCode(error)
  if (providerCode) return providerCode

  const status = getSearchErrorStatus(error)
  if (status) return `http_${status}`
  if (isRecord(error) && "event" in error) return "RagTerminalStreamError"
  return error instanceof Error ? "Error" : "UnknownError"
}

export const mapKnowledgeQaSearchErrorMessage = (
  error: unknown,
  fallback: string = "Search failed"
): string => {
  const providerCode = getPublicProviderCode(error)
  if (providerCode) {
    return PUBLIC_RAG_PROVIDER_ERROR_MESSAGES[providerCode]
  }

  const message = toErrorString(error)
  if (isTimeoutError(message)) {
    return "Search timed out. Try the Fast preset or reduce sources."
  }
  if (isConnectionError(message)) {
    return "Cannot reach server. Check your connection and try again."
  }
  if (/no results|no relevant/i.test(message)) {
    return "No relevant documents found. Try broadening your query."
  }

  const status = getSearchErrorStatus(error)
  if (status === 408) {
    return "Search timed out. Try the Fast preset or reduce sources."
  }
  if (status === 400 || status === 422) {
    return "RAG search request is invalid."
  }
  if (status === 401) {
    return "RAG search failed. Authentication is required."
  }
  if (status === 403) {
    return "RAG search failed. Access was denied."
  }
  if (status === 404) {
    return "RAG search endpoint is unavailable."
  }
  if (status === 429) {
    return "RAG search is rate limited. Please wait and try again."
  }
  if (status && status >= 500) {
    return "RAG search failed due to a server error."
  }
  if (isRecord(error) && "event" in error) {
    return "Invalid RAG terminal stream event."
  }
  return fallback
}

export const mapKnowledgeQaExportErrorMessage = (
  error: unknown,
  fallback: string = "Chatbook export failed. Please try again."
): string => {
  const message = toErrorString(error)
  if (!message) return fallback
  if (isConnectionError(message)) {
    return "Chatbook export failed. Cannot reach server."
  }
  if (isTimeoutError(message)) {
    return "Chatbook export timed out. Please retry in a moment."
  }
  if (/404|not found|thread/i.test(message)) {
    return "Chatbook export failed. Thread was not found."
  }
  if (/401|unauthorized/i.test(message)) {
    return "Chatbook export failed. You are not authorized to export this thread."
  }
  if (/403|forbidden/i.test(message)) {
    return "Chatbook export failed. You do not have permission to export this thread."
  }
  if (/400|422|unprocessable|validation|required field|invalid request|invalid payload/i.test(message)) {
    return "Chatbook export failed. Export request is invalid. Check the selected thread and try again."
  }
  if (/429|rate limit|too many/i.test(message)) {
    return "Chatbook export failed. Too many export requests. Please wait and try again."
  }
  if (/5\d\d|server error|internal server/i.test(message)) {
    return "Chatbook export failed due to a server error. Please try again."
  }
  return `Chatbook export failed. ${message}`
}
