import { isPlaceholderApiKey } from "@/utils/api-key"

let runtimeSingleUserApiKey: string | null = null
let cookieSessionConfigInvalidated = false
let runtimeCsrfCookieName = "csrf_token"
const COOKIE_NAME_PATTERN = /^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$/

export const setRuntimeCsrfCookieName = (value: string | null): void => {
  if (value === null) {
    runtimeCsrfCookieName = "csrf_token"
    return
  }
  if (!COOKIE_NAME_PATTERN.test(value) || /^__(?:Host|Http|Secure)-/i.test(value)) {
    throw new TypeError("Invalid runtime CSRF cookie name")
  }
  runtimeCsrfCookieName = value
}

export const getRuntimeCsrfCookieName = (): string => runtimeCsrfCookieName

const normalizeApiKey = (value?: string | null): string | null => {
  const normalized = String(value || "").trim()
  if (!normalized || /\s/.test(normalized)) return null
  if (isPlaceholderApiKey(normalized)) return null
  return normalized
}

export const setRuntimeSingleUserApiKeyOverride = (
  value?: string | null
): void => {
  runtimeSingleUserApiKey = normalizeApiKey(value)
}

export const getRuntimeSingleUserApiKeyOverride = (): string | null =>
  runtimeSingleUserApiKey

export const clearRuntimeAuthOverride = (): void => {
  runtimeSingleUserApiKey = null
}

export const invalidateCookieSessionConfig = (): void => {
  cookieSessionConfigInvalidated = true
}

export const activateCookieSessionConfig = (): void => {
  cookieSessionConfigInvalidated = false
}

export const isCookieSessionConfigInvalidated = (): boolean =>
  cookieSessionConfigInvalidated
