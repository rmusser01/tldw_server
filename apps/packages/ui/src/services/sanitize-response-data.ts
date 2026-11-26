import { sanitizeServerErrorMessage } from "@/utils/server-error-message"

const REDACTED_VALUE = "[REDACTED]"
const SENSITIVE_KEY_FRAGMENTS = [
  "stack",
  "trace",
  "sql",
  "query",
  "password",
  "passwd",
  "token",
  "secret",
  "path",
  "headers",
  "internalid",
  "authorization",
  "cookie",
  "api_key",
  "apikey",
  "access_key",
  "accesskey",
  "private",
  "credential",
  "session",
  "bearer"
]

const isSensitiveKey = (key: string): boolean => {
  const normalized = key.toLowerCase().replace(/[\s-]/g, "_")
  return SENSITIVE_KEY_FRAGMENTS.some((fragment) => normalized.includes(fragment))
}

// Redact known sensitive fields (stack/trace/sql/query/secret/headers/etc.) recursively.
export const sanitizeResponseData = (
  value: unknown,
  seen: WeakSet<object> = new WeakSet(),
  sanitizeErrorStrings = false
): unknown => {
  if (sanitizeErrorStrings && typeof value === "string") {
    return sanitizeServerErrorMessage(value, "")
  }
  if (value == null || typeof value !== "object") return value
  if (seen.has(value as object)) return REDACTED_VALUE
  seen.add(value as object)

  if (Array.isArray(value)) {
    return value.map((entry) => sanitizeResponseData(entry, seen, sanitizeErrorStrings))
  }

  const result: Record<string, unknown> = {}
  Object.entries(value as Record<string, unknown>).forEach(([key, entry]) => {
    if (isSensitiveKey(key)) {
      result[key] = REDACTED_VALUE
      return
    }
    result[key] = sanitizeResponseData(entry, seen, sanitizeErrorStrings)
  })
  return result
}
