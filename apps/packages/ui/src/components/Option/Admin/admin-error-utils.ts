export type AdminGuardState = "forbidden" | "notFound" | null

// Codes that mean the admin endpoint itself does not exist on this server.
// 5xx availability codes deliberately do NOT belong here: a 503 from e.g. the
// llama.cpp service means "the backing runtime is down", not "this tldw server
// lacks the /admin endpoints" — conflating them misdiagnoses the outage.
const ADMIN_NOT_FOUND_CODES = new Set(["404", "405", "410", "501"])

const SERVICE_UNAVAILABLE_CODES = new Set(["502", "503", "504"])

const normalizeErrorMessage = (error: unknown): string => {
  if (typeof error === "string") return error
  if (error && typeof error === "object" && "message" in error) {
    return String((error as { message?: unknown }).message ?? "")
  }
  return ""
}

const extractStatusCode = (error: unknown): string => {
  const rawMessage = normalizeErrorMessage(error)
  const statusFromField =
    error && typeof error === "object" && "status" in error
      ? String((error as { status?: unknown }).status ?? "")
      : ""
  const statusMatch =
    rawMessage.match(/Request failed:\s*(\d{3})/i) ||
    rawMessage.match(/\b(403|404|405|410|501|502|503|504)\b/)
  return statusFromField || statusMatch?.[1] || ""
}

export const deriveAdminGuardFromError = (error: unknown): AdminGuardState => {
  const statusCode = extractStatusCode(error)

  if (statusCode === "403") {
    return "forbidden"
  }
  if (statusCode && ADMIN_NOT_FOUND_CODES.has(statusCode)) {
    return "notFound"
  }
  return null
}

/**
 * True when the failure is a temporary upstream/service availability problem
 * (502/503/504) rather than a missing or forbidden admin API. Pages should
 * treat this as "the backing service is down — retry / start it", never as
 * "this server has no admin endpoints".
 */
export const isServiceUnavailableError = (error: unknown): boolean =>
  SERVICE_UNAVAILABLE_CODES.has(extractStatusCode(error))

export const sanitizeAdminErrorMessage = (
  error: unknown,
  fallbackMessage: string
): string => {
  const rawMessage = normalizeErrorMessage(error)
  if (!rawMessage.trim()) return fallbackMessage

  const firstLine = rawMessage
    .split("\n")
    .map((line) => line.trim())
    .find((line) => line.length > 0)

  let cleaned = (firstLine || rawMessage).replace(/^Error:\s*/i, "")

  // Avoid surfacing raw endpoint paths in user-facing admin errors.
  cleaned = cleaned.replace(
    /\b(GET|POST|PUT|PATCH|DELETE)\s+\/api\/v1\/[^\s)]+/gi,
    "$1 [admin-endpoint]"
  )
  cleaned = cleaned.replace(/\b\/api\/v1\/[^\s)]+/gi, "[admin-endpoint]")

  // Redact filesystem paths from backend traces/messages.
  cleaned = cleaned.replace(
    /\/(?:Users|home|var|etc|opt|tmp|private|srv)\/[^\s)]+/g,
    "[redacted-path]"
  )
  cleaned = cleaned.replace(
    /[A-Za-z]:\\(?:[^\\\s]+\\)+[^\\\s)]+/g,
    "[redacted-path]"
  )

  // A parenthetical that only carried the (now redacted) endpoint adds no
  // information for the reader — drop it instead of showing "[admin-endpoint]".
  cleaned = cleaned
    .replace(
      /\s*\(\s*(?:(?:GET|POST|PUT|PATCH|DELETE)\s+)?\[admin-endpoint\]\s*\)/gi,
      ""
    )
    .trim()

  if (cleaned.length > 220) {
    cleaned = `${cleaned.slice(0, 217)}...`
  }

  return cleaned || fallbackMessage
}
