export interface StructuredApiErrorDetail {
  category?: string
  code?: string
  frontend_state?: string
  message?: string
  recovery_action?: "retry" | "refresh" | "reselect_sources"
  retry_after_ms?: number
  retryable?: boolean
  [key: string]: unknown
}

export class TldwApiError extends Error {
  status: number
  detail: unknown

  constructor(message: string, status: number, detail: unknown) {
    super(message)
    this.name = "TldwApiError"
    this.status = status
    this.detail = detail
  }
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const apiErrorMessage = (detail: unknown, fallback: string): string => {
  if (typeof detail === "string" && detail.trim()) {
    return detail
  }
  if (isRecord(detail)) {
    const message = detail.message
    if (typeof message === "string" && message.trim()) {
      return message
    }
  }
  return fallback
}

export const buildTldwApiError = async (
  response: Response,
  fallback = "Request failed"
): Promise<TldwApiError> => {
  const body = await response
    .json()
    .catch(() => ({ detail: response.statusText }))
  const detail = isRecord(body) && "detail" in body ? body.detail : body
  const message = apiErrorMessage(detail, response.statusText || fallback)
  return new TldwApiError(message, response.status, detail)
}

export const getStructuredApiErrorDetail = (
  error: unknown
): StructuredApiErrorDetail | null => {
  if (!isRecord(error)) {
    return null
  }

  const detail = error.detail
  if (!isRecord(detail)) {
    return null
  }

  const recoveryAction = detail.recovery_action
  return {
    ...detail,
    code:
      typeof detail.code === "string" && detail.code.trim()
        ? detail.code.trim()
        : undefined,
    message: typeof detail.message === "string" ? detail.message : undefined,
    retryable:
      typeof detail.retryable === "boolean" ? detail.retryable : undefined,
    recovery_action:
      recoveryAction === "retry" ||
      recoveryAction === "refresh" ||
      recoveryAction === "reselect_sources"
        ? recoveryAction
        : undefined,
    retry_after_ms:
      typeof detail.retry_after_ms === "number" &&
      Number.isFinite(detail.retry_after_ms) &&
      detail.retry_after_ms >= 0
        ? detail.retry_after_ms
        : undefined,
    category:
      typeof detail.category === "string" ? detail.category : undefined,
    frontend_state:
      typeof detail.frontend_state === "string"
        ? detail.frontend_state
        : undefined
  }
}
