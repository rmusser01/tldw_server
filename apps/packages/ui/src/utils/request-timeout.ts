const TIMEOUT_HINTS = [
  "timeout",
  "timed out",
  "aborterror",
  "request_aborted",
  "aborted",
  "extension messaging timeout"
]

export const isTimeoutLikeError = (error: unknown): boolean => {
  if (error instanceof Error && error.name === "AbortError") return true

  const message =
    error instanceof Error
      ? `${error.name} ${error.message}`
      : typeof error === "string"
        ? error
        : ""

  const normalized = message.toLowerCase()
  return TIMEOUT_HINTS.some((hint) => normalized.includes(hint))
}

/** Config values may come from storage; preserve numeric-string timeout settings. */
type RequestTimeoutConfig = Readonly<{
  requestTimeoutMs?: unknown
  chatRequestTimeoutMs?: unknown
  ragRequestTimeoutMs?: unknown
  mediaRequestTimeoutMs?: unknown
}> | null | undefined

const isMediaApiPath = (path: string): boolean => /\/api\/v1\/media(?:\/|\?|$)/.test(path)
const isFilesApiPath = (path: string): boolean => /\/api\/v1\/files(?:\/|\?|$)/.test(path)
const isSlidesApiPath = (path: string): boolean => /\/api\/v1\/slides(?:\/|\?|$)/.test(path)
const SLIDES_REQUEST_TIMEOUT_FLOOR_MS = 120000
const MODEL_METADATA_REQUEST_TIMEOUT_FLOOR_MS = 60000
// LLM generation and RAG endpoints routinely run far longer than the generic
// 10s request default. Using the short default aborts normal generations
// mid-response and surfaces as a spurious "Network error". Default these paths
// to a generation-appropriate timeout instead (still overridable via config).
const GENERATION_REQUEST_TIMEOUT_DEFAULT_MS = 120000

export const resolveRequestTimeout = (
  cfg: RequestTimeoutConfig,
  path: string,
  override?: number
): number => {
  if (override && override > 0) return override
  const p = path
  if (p.includes("/api/v1/chat/completions")) {
    return Number(cfg?.chatRequestTimeoutMs) > 0
      ? Number(cfg?.chatRequestTimeoutMs)
      : Number(cfg?.requestTimeoutMs) > 0
        ? Number(cfg?.requestTimeoutMs)
        : GENERATION_REQUEST_TIMEOUT_DEFAULT_MS
  }
  if (p.includes("/api/v1/rag/")) {
    return Number(cfg?.ragRequestTimeoutMs) > 0
      ? Number(cfg?.ragRequestTimeoutMs)
      : Number(cfg?.requestTimeoutMs) > 0
        ? Number(cfg?.requestTimeoutMs)
        : GENERATION_REQUEST_TIMEOUT_DEFAULT_MS
  }
  if (/\/api\/v1\/llm\/models\/metadata(?:[/?#]|$)/.test(p)) {
    const configuredTimeout =
      Number(cfg?.requestTimeoutMs) > 0 ? Number(cfg?.requestTimeoutMs) : 0
    return Math.max(configuredTimeout, MODEL_METADATA_REQUEST_TIMEOUT_FLOOR_MS)
  }
  if (isMediaApiPath(p)) {
    return Number(cfg?.mediaRequestTimeoutMs) > 0
      ? Number(cfg?.mediaRequestTimeoutMs)
      : Number(cfg?.requestTimeoutMs) > 0
        ? Number(cfg?.requestTimeoutMs)
        : 10000
  }
  if (isFilesApiPath(p)) {
    return Number(cfg?.mediaRequestTimeoutMs) > 0
      ? Number(cfg?.mediaRequestTimeoutMs)
      : Number(cfg?.requestTimeoutMs) > 0
        ? Number(cfg?.requestTimeoutMs)
        : 10000
  }
  if (isSlidesApiPath(p)) {
    const configuredTimeout =
      Number(cfg?.requestTimeoutMs) > 0 ? Number(cfg?.requestTimeoutMs) : 0
    return Math.max(configuredTimeout, SLIDES_REQUEST_TIMEOUT_FLOOR_MS)
  }
  return Number(cfg?.requestTimeoutMs) > 0
    ? Number(cfg?.requestTimeoutMs)
    : 10000
}
