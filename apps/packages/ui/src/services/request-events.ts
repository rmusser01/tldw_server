export const BACKEND_UNREACHABLE_EVENT = "tldw:backend-unreachable"

const EXPLICIT_CANCELLATION_MESSAGE = /^(?:abort(?:ed|error)?|request_aborted|(?:the|this) operation was aborted|(?:the )?signal is aborted without reason|the user aborted (?:a|the) request|(?:[a-z0-9_-]+(?:\s+[a-z0-9_-]+)*\s+)?request (?:was )?aborted)\.?$/i

type RequestCancellationLike = {
  name?: unknown
  code?: unknown
  message?: unknown
}

export const isExplicitRequestCancellation = (value: unknown): boolean => {
  if (value === null || value === undefined) return false

  const candidate: RequestCancellationLike =
    typeof value === "object" ? value as RequestCancellationLike : {}
  const name = typeof candidate.name === "string" ? candidate.name.trim() : ""
  const code = typeof candidate.code === "string" ? candidate.code.trim() : ""
  if (/^AbortError$/i.test(name) || /^REQUEST_ABORTED$/i.test(code)) {
    return true
  }

  const message = typeof value === "string"
    ? value.trim()
    : typeof candidate.message === "string"
      ? candidate.message.trim()
      : ""
  return EXPLICIT_CANCELLATION_MESSAGE.test(message)
}

export type BackendUnreachableDetail = {
  method: string
  path: string
  status?: number
  code?: string
  message: string
  source: "background" | "direct"
  timestamp: number
}
