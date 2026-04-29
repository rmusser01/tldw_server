export type ApiEnvelopeMetadata = Record<string, unknown>

export type ApiResponseEnvelope<T> = {
  success: boolean
  data?: T | null
  error?: string | null
  error_code?: string | null
  metadata?: ApiEnvelopeMetadata | null
}

export type ApiResponseDataWrapper<T> = {
  data?: T | null
  error?: string | null
  error_code?: string | null
  metadata?: ApiEnvelopeMetadata | null
}

const hasOwn = (value: Record<string, unknown>, key: string): boolean =>
  Object.prototype.hasOwnProperty.call(value, key)

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const envelopeKeys = ["data", "error", "error_code", "metadata"] as const
const envelopeKeySet = new Set<string>(envelopeKeys)

const hasEnvelopeKey = (value: Record<string, unknown>): boolean =>
  envelopeKeys.some((key) => hasOwn(value, key))

const hasOnlyEnvelopeKeys = (value: Record<string, unknown>): boolean =>
  Object.keys(value).every((key) => envelopeKeySet.has(key))

const isApiResponseDataWrapper = (
  value: unknown
): value is ApiResponseDataWrapper<unknown> =>
  isRecord(value) &&
  !hasOwn(value, "success") &&
  hasEnvelopeKey(value) &&
  hasOnlyEnvelopeKeys(value)

/**
 * Detects the canonical opt-in API response envelope without treating legacy
 * success-shaped payloads as envelopes.
 */
export const isApiResponseEnvelope = (
  value: unknown
): value is ApiResponseEnvelope<unknown> => {
  if (!isRecord(value) || typeof value.success !== "boolean") {
    return false
  }
  return hasEnvelopeKey(value)
}

/**
 * Unwraps canonical response envelopes while preserving non-envelope payloads.
 */
export function unwrapApiResponseEnvelope<T>(value: ApiResponseEnvelope<T>): T | null
export function unwrapApiResponseEnvelope<T>(value: T): T
export function unwrapApiResponseEnvelope(value: null): null
export function unwrapApiResponseEnvelope(value: undefined): undefined
export function unwrapApiResponseEnvelope<T>(
  value: ApiResponseEnvelope<T> | T | null | undefined
): T | null | undefined {
  if (isApiResponseEnvelope(value)) {
    return (value as ApiResponseEnvelope<T>).data ?? null
  }
  return value as T | null | undefined
}

/**
 * Unwraps canonical envelopes and transitional response-body data wrappers.
 */
export function unwrapApiResponseData<T>(
  value: ApiResponseEnvelope<T> | ApiResponseDataWrapper<T>
): T | null
export function unwrapApiResponseData<T>(value: T): T
export function unwrapApiResponseData(value: null): null
export function unwrapApiResponseData(value: undefined): undefined
export function unwrapApiResponseData<T>(
  value: ApiResponseEnvelope<T> | ApiResponseDataWrapper<T> | T | null | undefined
): T | null | undefined {
  if (isApiResponseEnvelope(value)) {
    return (value as ApiResponseEnvelope<T>).data ?? null
  }
  if (isApiResponseDataWrapper(value)) {
    return (value as ApiResponseDataWrapper<T>).data ?? null
  }
  return value as T | null | undefined
}
