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
