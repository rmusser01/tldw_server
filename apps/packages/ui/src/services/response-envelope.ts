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

export const isApiResponseEnvelope = (
  value: unknown
): value is ApiResponseEnvelope<unknown> => {
  if (!isRecord(value) || typeof value.success !== "boolean") {
    return false
  }
  return hasOwn(value, "data") || hasOwn(value, "error") || hasOwn(value, "error_code")
}

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
  if (isRecord(value) && hasOwn(value, "data")) {
    return (value as ApiResponseDataWrapper<T>).data ?? null
  }
  return value as T | null | undefined
}
