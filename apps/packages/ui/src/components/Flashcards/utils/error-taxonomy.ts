import { formatErrorMessage } from "@/utils/format-error-message"

export type FlashcardsUiErrorCode =
  | "FLASHCARDS_GROUNDING"
  | "FLASHCARDS_VERSION_CONFLICT"
  | "FLASHCARDS_NETWORK"
  | "FLASHCARDS_VALIDATION"
  | "FLASHCARDS_NOT_FOUND"
  | "FLASHCARDS_SERVER"
  | "FLASHCARDS_UNKNOWN"

export interface FlashcardsUiError {
  code: FlashcardsUiErrorCode
  message: string
  actionLabel: string
  status?: number
  rawMessage: string
  grounding?: FlashcardsGroundingDiagnostics
}

export interface FlashcardsGroundingDiagnostics {
  verdict: string
  generationModel?: string
  verificationModel?: string
  claims: {
    cardNumber: number
    field: string
    claim: string
    status: string
    rationale?: string
    evidence: string[]
  }[]
}

export interface FlashcardsErrorMappingOptions {
  operation: string
  fallback: string
}

const NETWORK_ERROR_PATTERN =
  /(networkerror|failed to fetch|network error|load failed|err_connection|connection refused|receiving end does not exist|messaging timeout|timed out|timeout)/i
const VERSION_CONFLICT_PATTERN =
  /(version mismatch|version changed|optimistic lock|expected version|conflict)/i
const VALIDATION_PATTERN =
  /(invalid|missing required|required field|unprocessable|validation|bad request|must contain)/i

const asHttpStatus = (value: unknown): number | undefined => {
  const num =
    typeof value === "number"
      ? value
      : typeof value === "string"
        ? Number(value.trim())
        : Number.NaN
  if (!Number.isFinite(num) || !Number.isInteger(num)) return undefined
  if (num < 100 || num > 599) return undefined
  return num
}

const extractStatusFromRecord = (
  record: Record<string, unknown>
): number | undefined => {
  return (
    asHttpStatus(record.status) ??
    asHttpStatus(record.statusCode) ??
    asHttpStatus(record.code)
  )
}

export const extractFlashcardsErrorStatus = (error: unknown): number | undefined => {
  if (error && typeof error === "object") {
    const record = error as Record<string, unknown>
    const direct = extractStatusFromRecord(record)
    if (direct !== undefined) return direct

    if (record.response && typeof record.response === "object") {
      const nested = extractStatusFromRecord(record.response as Record<string, unknown>)
      if (nested !== undefined) return nested
    }
  }

  const message = error instanceof Error ? error.message : String(error || "")
  const matches = [
    message.match(/\b(?:status|statusCode)\b\s*[:=]?\s*(\d{3})\b/i)?.[1],
    message.match(/\bHTTP(?:\/\d(?:\.\d)?)?\s*(\d{3})\b/i)?.[1],
    message.match(/\b(?:request|upload)\s+failed\b\s*[:(]\s*(\d{3})\b/i)?.[1],
    message.match(/\((\d{3})\)/)?.[1]
  ]

  for (const candidate of matches) {
    const status = asHttpStatus(candidate)
    if (status !== undefined) return status
  }

  return undefined
}

const stripMethodPathSuffix = (message: string): string =>
  message
    .replace(/\s+\((GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+\/api\/[^\)]*\)$/i, "")
    .trim()

const appendRetryHint = (message: string): string =>
  message.endsWith(".") ? `${message} Please retry.` : `${message}. Please retry.`

const asRecord = (value: unknown): Record<string, unknown> | undefined =>
  value !== null && typeof value === "object" && !Array.isArray(value)
    ? value as Record<string, unknown>
    : undefined

const reportText = (value: unknown): string | undefined =>
  typeof value === "string" && value.trim() ? value.trim() : undefined

const VERDICT_LABELS = new Map([
  ["grounded", "Grounded"], ["needs_revision", "Needs revision"], ["failed", "Failed"]
])
const STATUS_LABELS = new Map([
  ["verified", "Verified"], ["refuted", "Refuted"], ["unverified", "Unverified"],
  ["hallucination", "Unsupported claim"], ["numerical_error", "Numerical error"],
  ["misquoted", "Misquoted"], ["citation_not_found", "Citation not found"],
  ["contested", "Contested"], ["misleading", "Misleading"]
])

// Request errors already retain a sanitized response. Project only the report
// fields this panel explains; identifiers are used for links, never displayed.
const groundingDiagnostics = (error: unknown): FlashcardsGroundingDiagnostics | undefined => {
  const detail = asRecord(asRecord(asRecord(error)?.details)?.detail)
  if (detail?.code !== "claim_verification_failed") return undefined
  const verification = asRecord(detail.claimVerification)
  const report = asRecord(verification?.report)
  const verdict = VERDICT_LABELS.get(reportText(verification?.verdict) ?? "")
  if (!verdict || !Array.isArray(report?.claims) || !Array.isArray(verification?.unitResults)) return undefined
  const claimsById = new Map<string, Record<string, unknown>>()
  for (const entry of report.claims) {
    const claim = asRecord(entry)
    const id = reportText(claim?.claim_id)
    if (id && claim) claimsById.set(id, claim)
  }
  const claims: FlashcardsGroundingDiagnostics["claims"] = []
  for (const entry of verification.unitResults) {
    const unit = asRecord(entry)
    const match = reportText(unit?.unit_id)?.match(/^flashcard:([1-9]\d*):(front|back|notes|extra)$/)
    if (!match || !Array.isArray(unit?.claim_ids)) continue
    const cardNumber = Number(match[1])
    if (!Number.isSafeInteger(cardNumber) || cardNumber > 100) continue
    for (const id of unit.claim_ids) {
      const claim = typeof id === "string" ? claimsById.get(id) : undefined
      const text = reportText(claim?.claim_text)
      if (!text || !claim) continue
      claims.push({
        cardNumber,
        field: match[2][0].toUpperCase() + match[2].slice(1),
        claim: text,
        status: STATUS_LABELS.get(reportText(claim.status) ?? "") ?? "Not reported",
        rationale: reportText(claim.rationale),
        evidence: Array.isArray(claim.evidence)
          ? claim.evidence.flatMap(entry => {
              const snippet = reportText(asRecord(entry)?.snippet)
              return snippet ? [snippet] : []
            })
          : []
      })
    }
  }
  const metadata = asRecord(verification.metadata)
  const modelLabel = (provider: unknown, model: unknown) =>
    [reportText(provider), reportText(model)].filter(Boolean).join(" / ") || undefined
  return {
    verdict,
    generationModel: modelLabel(metadata?.generation_provider, metadata?.generation_model),
    verificationModel: modelLabel(metadata?.verification_provider, metadata?.verification_model),
    claims
  }
}

export const mapFlashcardsUiError = (
  error: unknown,
  options: FlashcardsErrorMappingOptions
): FlashcardsUiError => {
  const status = extractFlashcardsErrorStatus(error)
  const rawMessage = stripMethodPathSuffix(
    formatErrorMessage(error, options.fallback)
  )
  const normalized = rawMessage.toLowerCase()

  if (/\bclaim_verification_failed\b/.test(normalized)) {
    const grounding = groundingDiagnostics(error)
    return {
      code: "FLASHCARDS_GROUNDING",
      message: "The generated cards could not be verified against your source. Add clearer source details or request fewer cards, then generate again.",
      actionLabel: "Revise source",
      status,
      rawMessage,
      ...(grounding ? { grounding } : {})
    }
  }

  if (status === 409 || VERSION_CONFLICT_PATTERN.test(normalized)) {
    return {
      code: "FLASHCARDS_VERSION_CONFLICT",
      message: "This card was modified since you opened it. Refresh and try again.",
      actionLabel: "Refresh",
      status,
      rawMessage
    }
  }

  if (status === 0 || NETWORK_ERROR_PATTERN.test(normalized)) {
    return {
      code: "FLASHCARDS_NETWORK",
      message: `Couldn't reach the server while ${options.operation}. Check your connection and retry.`,
      actionLabel: "Retry",
      status,
      rawMessage
    }
  }

  if (status === 400 || status === 422 || VALIDATION_PATTERN.test(normalized)) {
    const baseMessage = normalized === "flashcard_generate_error" ? options.fallback : rawMessage || options.fallback
    return {
      code: "FLASHCARDS_VALIDATION",
      message: `${baseMessage} Fix the input and retry.`,
      actionLabel: "Fix input",
      status,
      rawMessage
    }
  }

  if (status === 404) {
    return {
      code: "FLASHCARDS_NOT_FOUND",
      message: "This card is no longer available. Reload and try again.",
      actionLabel: "Reload card",
      status,
      rawMessage
    }
  }

  if (typeof status === "number" && status >= 500) {
    return {
      code: "FLASHCARDS_SERVER",
      message: `Server error while ${options.operation}. Please retry in a moment.`,
      actionLabel: "Retry",
      status,
      rawMessage
    }
  }

  return {
    code: "FLASHCARDS_UNKNOWN",
    message: appendRetryHint(options.fallback),
    actionLabel: "Retry",
    status,
    rawMessage
  }
}

export const formatFlashcardsUiErrorMessage = (error: FlashcardsUiError): string =>
  `${error.message} [${error.code}]`
