const MAX_PROVIDER_ERROR_MESSAGE_LENGTH = 240

export const PUBLIC_RAG_PROVIDER_ERROR_MESSAGES = {
  provider_request_invalid: "The selected provider or model is invalid.",
  provider_authentication_failed:
    "The selected provider credentials could not be authenticated.",
  invalid_provider_credentials: "The selected provider credentials are invalid.",
  missing_provider_credentials:
    "The selected provider credentials are not configured.",
  credential_store_unavailable:
    "Provider credential storage is temporarily unavailable.",
  credential_scope_revoked:
    "The selected provider credential scope is no longer available.",
  provider_disabled:
    "The selected provider is disabled by administrator policy.",
  model_not_allowed: "The selected model is not allowed for this provider.",
  provider_configuration_invalid:
    "The selected provider configuration is invalid.",
  provider_unavailable: "The selected provider is currently unavailable.",
} as const

export type PublicRagProviderErrorCode =
  keyof typeof PUBLIC_RAG_PROVIDER_ERROR_MESSAGES

type PublicRagProviderError = {
  code: PublicRagProviderErrorCode
  message: string
}

export type SanitizedRagProviderFailure = {
  message: string
  status?: number
  code?: PublicRagProviderErrorCode
  details?: {
    detail: {
      error_code: PublicRagProviderErrorCode
      message: string
    }
  }
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

export const asPublicRagProviderErrorCode = (
  value: unknown
): PublicRagProviderErrorCode | null =>
  typeof value === "string" &&
  Object.prototype.hasOwnProperty.call(PUBLIC_RAG_PROVIDER_ERROR_MESSAGES, value)
    ? (value as PublicRagProviderErrorCode)
    : null

export const asValidatedHttpStatus = (value: unknown): number | undefined =>
  typeof value === "number" &&
  Number.isInteger(value) &&
  value >= 100 &&
  value <= 599
    ? value
    : undefined

export const getValidatedHttpStatus = (error: unknown): number | undefined => {
  if (!isRecord(error)) return undefined
  const response = isRecord(error.response) ? error.response : null
  return (
    asValidatedHttpStatus(error.status) ??
    asValidatedHttpStatus(response?.status) ??
    asValidatedHttpStatus(error.statusCode)
  )
}

export const getStructuredPublicRagProviderError = (
  error: unknown
): PublicRagProviderError | null => {
  if (!isRecord(error)) return null

  const details = isRecord(error.details) ? error.details : null
  let detail: Record<string, unknown> | null = null
  if ("detail" in error) {
    detail = isRecord(error.detail) ? error.detail : null
  } else if (details && "detail" in details) {
    detail = isRecord(details.detail) ? details.detail : null
  } else {
    detail = details
  }
  if (!detail) return null

  const code = asPublicRagProviderErrorCode(detail.error_code)
  const rawMessage = detail.message
  if (
    !code ||
    typeof rawMessage !== "string" ||
    rawMessage.trim().length === 0 ||
    rawMessage.length > MAX_PROVIDER_ERROR_MESSAGE_LENGTH
  ) {
    return null
  }

  return {
    code,
    message: PUBLIC_RAG_PROVIDER_ERROR_MESSAGES[code],
  }
}

const getStructuredProviderErrorFromFailure = (
  error: unknown
): PublicRagProviderError | null => {
  const direct = getStructuredPublicRagProviderError(error)
  if (direct || !isRecord(error)) return direct
  const response = isRecord(error.response) ? error.response : null

  return (
    getStructuredPublicRagProviderError(error.data) ??
    getStructuredPublicRagProviderError({ details: error.data }) ??
    getStructuredPublicRagProviderError(response?.data) ??
    getStructuredPublicRagProviderError({ details: response?.data })
  )
}

const getFailureMessage = (error: unknown): string => {
  if (error instanceof Error) return error.message
  if (typeof error === "string") return error
  if (!isRecord(error)) return ""
  if (typeof error.message === "string") return error.message
  return typeof error.error === "string" ? error.error : ""
}

const isConnectionFailure = (message: string): boolean =>
  /network|offline|failed to fetch|connection|unreachable/i.test(message)

const isTimeoutFailure = (message: string): boolean =>
  /timeout|timed out|etimedout/i.test(message)

/**
 * Reduces an arbitrary RAG provider failure to client-owned, bounded fields.
 * Raw provider text is inspected only for broad network/timeout classification
 * and is never included in the returned diagnostic payload.
 */
export const sanitizeRagProviderFailure = (
  error: unknown
): SanitizedRagProviderFailure => {
  const status = getValidatedHttpStatus(error)
  const providerError = getStructuredProviderErrorFromFailure(error)
  const rawMessage = getFailureMessage(error)

  let message = "RAG search failed."
  if (providerError) {
    message = providerError.message
  } else if (isConnectionFailure(rawMessage)) {
    message = "Cannot reach server. Check your connection and try again."
  } else if (isTimeoutFailure(rawMessage) || status === 408) {
    message = "RAG search timed out. Try again."
  } else if (status === 400 || status === 422) {
    message = "RAG search request is invalid."
  } else if (status === 401) {
    message = "RAG search failed. Authentication is required."
  } else if (status === 403) {
    message = "RAG search failed. Access was denied."
  } else if (status === 404) {
    message = "RAG search endpoint is unavailable."
  } else if (status === 429) {
    message = "RAG search is rate limited. Please wait and try again."
  } else if (typeof status === "number" && status >= 500) {
    message = "RAG search failed due to a server error."
  }

  const sanitized: SanitizedRagProviderFailure = { message }
  if (typeof status === "number") {
    sanitized.status = status
  }
  if (providerError) {
    sanitized.code = providerError.code
    sanitized.details = {
      detail: {
        error_code: providerError.code,
        message: providerError.message,
      },
    }
  }
  return sanitized
}
