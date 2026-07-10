import type { ErrorClassification } from "./types"

/**
 * Structured error category returned by `classifyError()`.
 */
export type ErrorCategory = {
  /** Classification bucket from the wizard type system. */
  classification: ErrorClassification
  /** Whether the error is worth retrying. */
  retryable: boolean
  /** Short label shown inside a badge. */
  badgeLabel: string
  /** Tailwind color token for the badge background. */
  badgeColor: string
  /** Plain-language explanation of the error for end users. */
  userMessage: string
  /** Actionable suggestion the user can follow. */
  suggestion: string
}

// ---------------------------------------------------------------------------
// Internal pattern table
// ---------------------------------------------------------------------------

type PatternEntry = {
  patterns: RegExp
  category: ErrorCategory
}

const NETWORK_CATEGORY: ErrorCategory = {
  classification: "network",
  retryable: true,
  badgeLabel: "Network \u00b7 Retryable",
  badgeColor: "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300",
  userMessage: "The server couldn't be reached. This is usually temporary.",
  suggestion: "Check your connection and retry.",
}

const AUTH_CATEGORY: ErrorCategory = {
  classification: "auth",
  retryable: false,
  badgeLabel: "Auth \u00b7 Check Config",
  badgeColor: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300",
  userMessage: "Authentication failed.",
  suggestion: "Check your API key or server configuration.",
}

const CONFIG_CATEGORY: ErrorCategory = {
  classification: "auth",
  retryable: false,
  badgeLabel: "Config \u00b7 Check Server",
  badgeColor: "bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300",
  userMessage: "The tldw server is not configured.",
  suggestion: "Set the server URL and API key in Settings, then try again.",
}

const VALIDATION_CATEGORY: ErrorCategory = {
  classification: "validation",
  retryable: false,
  badgeLabel: "Format \u00b7 Permanent",
  badgeColor: "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300",
  userMessage: "The input format is not supported or invalid.",
  suggestion: "Check the file format and try again.",
}

const SERVER_CATEGORY: ErrorCategory = {
  classification: "server",
  retryable: true,
  badgeLabel: "Server \u00b7 Retryable",
  badgeColor: "bg-rose-100 text-rose-800 dark:bg-rose-900/30 dark:text-rose-300",
  userMessage: "The server encountered an error.",
  suggestion: "This is usually temporary. Try again in a moment.",
}

const JOB_LIMIT_CATEGORY: ErrorCategory = {
  classification: "server",
  retryable: false,
  badgeLabel: "Queue Full \u00b7 Wait",
  badgeColor: "bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300",
  userMessage: "You already have the maximum number of ingest jobs running.",
  suggestion: "Wait for active jobs to finish or cancel one, then try again.",
}

const TIMEOUT_CATEGORY: ErrorCategory = {
  classification: "timeout",
  retryable: true,
  badgeLabel: "Timeout \u00b7 Retryable",
  badgeColor: "bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300",
  userMessage: "The request took too long.",
  suggestion: "Try again — larger files may need more time.",
}

const UNKNOWN_CATEGORY: ErrorCategory = {
  classification: "unknown",
  retryable: true,
  badgeLabel: "Error \u00b7 Retryable",
  badgeColor: "bg-gray-100 text-gray-800 dark:bg-gray-800/40 dark:text-gray-300",
  userMessage: "An unexpected error occurred.",
  suggestion: "Try again or check the server logs.",
}

/**
 * Order matters: timeout patterns are checked before generic network patterns
 * so that "timeout" is not accidentally classified as network.
 */
const PATTERN_TABLE: PatternEntry[] = [
  {
    patterns: /timeout|timed\s*out|deadline/i,
    category: TIMEOUT_CATEGORY,
  },
  {
    patterns: /\b429\b|too many requests|concurrent job limit|max(?:imum)? concurrent/i,
    category: JOB_LIMIT_CATEGORY,
  },
  {
    patterns: /server (?:is )?not configured|server configuration/i,
    category: CONFIG_CATEGORY,
  },
  {
    patterns: /\b40[13]\b|unauthorized|forbidden|auth/i,
    category: AUTH_CATEGORY,
  },
  {
    patterns: /\b400\b|invalid|unsupported|format/i,
    category: VALIDATION_CATEGORY,
  },
  {
    patterns: /\b50[023]\b|internal\s*server/i,
    category: SERVER_CATEGORY,
  },
  {
    patterns: /econnrefused|fetch\s*failed|network|connection/i,
    category: NETWORK_CATEGORY,
  },
]

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Classify an error message string into a structured `ErrorCategory`.
 *
 * Pattern matching is applied in priority order (timeout > auth > validation >
 * server > network) so that overlapping keywords resolve deterministically.
 * If nothing matches, the error is classified as `"unknown"` (retryable).
 */
export function classifyError(error: string | undefined): ErrorCategory {
  if (!error) return UNKNOWN_CATEGORY

  for (const entry of PATTERN_TABLE) {
    if (entry.patterns.test(error)) {
      return entry.category
    }
  }

  return UNKNOWN_CATEGORY
}
