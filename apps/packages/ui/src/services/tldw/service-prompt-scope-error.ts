import { deriveSingleUserApiKeyCredentialScope } from "@/services/chat-surface-scope"
import { deriveScopedUserId } from "@/utils/media-navigation-scope"

const SCOPE_CHANGED_CODE = "request_config_scope_changed"
const SCOPE_CHANGED_MESSAGE =
  "The server or authenticated account changed before the request was sent."

type ServicePromptTarget = Readonly<{
  serverUrl?: unknown
  authMode?: unknown
  authSource?: unknown
  orgId?: unknown
}>

export const servicePromptTargetsMatch = (
  current: ServicePromptTarget,
  checked: ServicePromptTarget
): boolean => ([
  "serverUrl",
  "authMode",
  "authSource",
  "orgId"
] as const).every((key) => (current[key] ?? null) === (checked[key] ?? null))

export const servicePromptPrincipalMatches = (
  current: Readonly<{
    authMode?: unknown
    accessToken?: unknown
  }>,
  expectedUserId: unknown
): boolean => {
  if (expectedUserId === null || expectedUserId === undefined) return true
  if (current.authMode !== "multi-user") return false
  const expected = String(expectedUserId).trim()
  if (!expected) return false
  return deriveScopedUserId({
    userId: null,
    authMode: "multi-user",
    accessToken:
      typeof current.accessToken === "string" ? current.accessToken : null
  }) === deriveScopedUserId({
    userId: expected,
    authMode: "multi-user",
    accessToken: null
  })
}

export const servicePromptRefreshLineageMatches = (
  current: Readonly<{ refreshToken?: unknown }>,
  expectedRefreshToken: unknown
): boolean => {
  if (expectedRefreshToken === undefined) return true
  if (typeof expectedRefreshToken !== "string") return false
  const expected = expectedRefreshToken.trim()
  return Boolean(expected) &&
    String(current.refreshToken || "").trim() === expected
}

export const servicePromptSingleUserApiKeyScopeMatches = (
  current: Readonly<{
    authMode?: unknown
    apiKey?: unknown
  }>,
  expectedScope: unknown
): boolean => {
  if (current.authMode !== "single-user") {
    return expectedScope === undefined
  }
  if (typeof expectedScope !== "string") return false
  return deriveSingleUserApiKeyCredentialScope(
    "single-user",
    typeof current.apiKey === "string" ? current.apiKey : null
  ) === expectedScope
}

const readCanonicalPathname = (path: unknown): string | null => {
  const pathname = String(path || "").split(/[?#]/, 1)[0]
  if (!pathname.startsWith("/") ||
    pathname.includes("\\") ||
    pathname.includes("//") ||
    /%(?:2e|2f|5c)/i.test(pathname) ||
    pathname.split("/").some((segment) => segment === "." || segment === "..")
  ) {
    return null
  }
  try {
    decodeURIComponent(pathname)
  } catch {
    return null
  }
  return pathname
}

export const isServicePromptRequestPath = (
  path: unknown,
  method: unknown
): boolean => {
  const pathname = readCanonicalPathname(path)
  if (!pathname) return false
  const requestMethod = String(method || "GET").toUpperCase()
  if (pathname === "/api/v1/service-prompts") {
    return requestMethod === "GET"
  }
  if (/^\/api\/v1\/service-prompts\/[^/]+$/.test(pathname)) {
    return ["GET", "PUT", "DELETE"].includes(requestMethod)
  }
  if (requestMethod === "GET") {
    return /^\/api\/v1\/writing\/manuscripts\/(?:scenes\/[^/]+|projects\/[^/]+\/(?:characters|world-info))$/.test(pathname)
  }
  if (requestMethod !== "POST") return false
  if (pathname === "/api/v1/chats/") return true
  return /^\/api\/v1\/(?:auth\/refresh|chat\/completions|media\/add|rag\/search|research\/websearch)$/.test(pathname) ||
    /^\/api\/v1\/chats\/[^/]+\/messages$/.test(pathname)
}

export const createServicePromptScopeChangedError = () =>
  Object.assign(new Error(SCOPE_CHANGED_MESSAGE), {
    status: 412,
    details: {
      detail: { code: SCOPE_CHANGED_CODE, message: SCOPE_CHANGED_MESSAGE }
    }
  })

const asRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === "object"
    ? value as Record<string, unknown>
    : null

export const isRequestConfigScopeChangedError = (error: unknown): boolean => {
  const seen = new Set<unknown>()
  let current: unknown = error

  while (current && !seen.has(current)) {
    seen.add(current)
    const candidate = asRecord(current)
    if (!candidate) return false

    const details = asRecord(candidate.details)
    const detail = asRecord(candidate.detail) ?? asRecord(details?.detail)
    const status = Number(candidate.status ?? candidate.statusCode)
    const code = candidate.code ?? details?.code ?? detail?.code
    if (status === 412 && code === SCOPE_CHANGED_CODE) return true

    current = candidate.cause
  }
  return false
}
