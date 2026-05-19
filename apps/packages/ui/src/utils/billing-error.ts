/**
 * Utilities for detecting and parsing billing limit errors (402/429)
 * returned by the tldw server billing enforcement layer.
 */

export type BillingLimitErrorInfo = {
  errorType: "limit_exceeded" | "feature_not_available"
  category?: string
  feature?: string
  current?: number
  limit?: number
  message: string
  upgradeUrl: string
  retryAfterSeconds?: number
}

/**
 * Attempt to extract structured billing limit info from an error object.
 * Returns null if the error is not a billing limit error.
 *
 * Handles:
 * - Errors with `.status` (402/429) and `.details` from background-proxy
 * - Errors with message string containing billing keywords (fallback)
 */
export function parseBillingLimitError(
  error: unknown
): BillingLimitErrorInfo | null {
  if (!error) return null

  const status = (error as { status?: number }).status
  const details = (error as { details?: Record<string, unknown> }).details
  const message =
    error instanceof Error ? error.message : String(error || "")

  // Try structured details first (from background-proxy or API client)
  if (details && typeof details === "object") {
    const d = details as Record<string, unknown>
    // The backend sends { error: "limit_exceeded", category, current, limit, message, upgrade_url }
    // or { error: "feature_not_available", feature, message, upgrade_url }
    const errorType = d.error ?? (d.detail as Record<string, unknown>)?.error
    if (
      errorType === "limit_exceeded" ||
      errorType === "feature_not_available"
    ) {
      const inner =
        typeof d.detail === "object" && d.detail ? (d.detail as Record<string, unknown>) : d
      return {
        errorType: errorType as "limit_exceeded" | "feature_not_available",
        category: String(inner.category || ""),
        feature: String(inner.feature || ""),
        current: typeof inner.current === "number" ? inner.current : undefined,
        limit: typeof inner.limit === "number" ? inner.limit : undefined,
        message: String(inner.message || message),
        upgradeUrl: String(inner.upgrade_url || "/billing/plans"),
        retryAfterSeconds:
          typeof inner.retry_after === "number" ? inner.retry_after : undefined,
      }
    }
  }

  // Check status code
  if (status === 402 || status === 429) {
    const lower = message.toLowerCase()
    const isFeatureGate =
      lower.includes("feature_not_available") ||
      lower.includes("feature not available")
    if (
      lower.includes("limit_exceeded") ||
      lower.includes("limit exceeded") ||
      isFeatureGate ||
      lower.includes("quota exceeded") ||
      lower.includes("upgrade")
    ) {
      return {
        errorType: isFeatureGate
          ? "feature_not_available"
          : "limit_exceeded",
        message,
        upgradeUrl: "/billing/plans",
      }
    }
  }

  // Message-only fallback (for degraded paths where status is lost)
  const lower = message.toLowerCase()
  const isFeatureGateFallback =
    lower.includes("feature_not_available") ||
    lower.includes("feature not available")
  if (
    lower.includes("limit_exceeded") ||
    isFeatureGateFallback
  ) {
    return {
      errorType: isFeatureGateFallback
        ? "feature_not_available"
        : "limit_exceeded",
      message,
      upgradeUrl: "/billing/plans",
    }
  }

  return null
}

export function isBillingLimitError(error: unknown): boolean {
  return parseBillingLimitError(error) !== null
}
