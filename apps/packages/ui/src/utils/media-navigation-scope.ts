export const MEDIA_NAVIGATION_FORMAT_VALUES = [
  "auto",
  "plain",
  "markdown",
  "html"
] as const

export type MediaNavigationFormat = (typeof MEDIA_NAVIGATION_FORMAT_VALUES)[number]

export const MEDIA_DISPLAY_MODE_LABEL_TO_FORMAT = {
  Auto: "auto",
  Plain: "plain",
  Markdown: "markdown",
  Rich: "html"
} as const

export type MediaDisplayModeLabel = keyof typeof MEDIA_DISPLAY_MODE_LABEL_TO_FORMAT

export const MEDIA_DISPLAY_MODE_FORMAT_TO_LABEL: Record<
  MediaNavigationFormat,
  MediaDisplayModeLabel
> = {
  auto: "Auto",
  plain: "Plain",
  markdown: "Markdown",
  html: "Rich"
}

const DEFAULT_SERVER_FINGERPRINT = "server:unknown"
const DEFAULT_USER_SCOPE = "user:anonymous"

const fnv1a36 = (value: string): string => {
  let hash = 2166136261
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i)
    hash +=
      (hash << 1) +
      (hash << 4) +
      (hash << 7) +
      (hash << 8) +
      (hash << 24)
  }
  return (hash >>> 0).toString(36)
}

const normalizeServerIdentity = (serverUrl?: string | null): string | null => {
  const raw = String(serverUrl || "").trim()
  if (!raw) return null
  try {
    const parsed = new URL(raw)
    const protocol = parsed.protocol.toLowerCase()
    const hostname = parsed.hostname.toLowerCase()
    const includePort = Boolean(
      parsed.port &&
        !((protocol === "http:" && parsed.port === "80") || (protocol === "https:" && parsed.port === "443"))
    )
    const port = includePort ? `:${parsed.port}` : ""
    const pathname = parsed.pathname.replace(/\/+$/, "")
    return `${protocol}//${hostname}${port}${pathname}`
  } catch {
    return raw.replace(/\/+$/, "").toLowerCase()
  }
}

const decodeJwtPayload = (token: string): Record<string, unknown> | null => {
  const trimmed = String(token || "").trim()
  if (!trimmed) return null
  const parts = trimmed.split(".")
  if (parts.length < 2) return null
  const payloadPart = parts[1]
  if (!payloadPart) return null
  try {
    const normalized = payloadPart.replace(/-/g, "+").replace(/_/g, "/")
    const padding = normalized.length % 4 === 0 ? "" : "=".repeat(4 - (normalized.length % 4))
    const decoded = atob(`${normalized}${padding}`)
    const parsed = JSON.parse(decoded)
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : null
  } catch {
    return null
  }
}

const resolveTokenUserClaim = (token?: string | null): string | null => {
  const payload = decodeJwtPayload(String(token || ""))
  if (!payload) return null
  const candidateKeys = ["sub", "user_id", "uid", "id"]
  for (const key of candidateKeys) {
    const value = payload[key]
    if (typeof value === "string" && value.trim()) return value.trim()
    if (typeof value === "number" && Number.isFinite(value)) return String(value)
  }
  return null
}

/** Read workspace hints from server-issued claims; authorization remains server-side. */
export const deriveTokenOrgId = (token: string): number | undefined => {
  const payload = decodeJwtPayload(token)
  if (!payload) return undefined
  const isOrgId = (value: unknown): value is number =>
    typeof value === "number" && Number.isSafeInteger(value) && value > 0
  const orgIds = Array.isArray(payload.org_ids)
    ? payload.org_ids.filter(isOrgId)
    : []
  if (isOrgId(payload.active_org_id) &&
      (!Array.isArray(payload.org_ids) || orgIds.includes(payload.active_org_id))) {
    return payload.active_org_id
  }
  // The former organization list preferred newest records. Use newest ID when
  // multiple memberships have no active selection in the token.
  return orgIds.reduce<number | undefined>((latest, id) =>
    latest === undefined || id > latest ? id : latest, undefined)
}

export const coerceMediaNavigationFormat = (
  value: unknown,
  fallback: MediaNavigationFormat = "auto"
): MediaNavigationFormat => {
  const normalized = String(value || "").trim().toLowerCase()
  if (!normalized) return fallback
  if ((MEDIA_NAVIGATION_FORMAT_VALUES as readonly string[]).includes(normalized)) {
    return normalized as MediaNavigationFormat
  }
  return fallback
}

export interface MediaNavigationScopeInput {
  serverUrl?: string | null
  userId?: string | number | null
  authMode?: string | null
  accessToken?: string | null
}

export const deriveServerFingerprint = (serverUrl?: string | null): string => {
  const normalized = normalizeServerIdentity(serverUrl)
  if (!normalized) return DEFAULT_SERVER_FINGERPRINT
  return `server:${fnv1a36(normalized)}`
}

export const deriveScopedUserId = ({
  userId,
  authMode,
  accessToken
}: Omit<MediaNavigationScopeInput, "serverUrl">): string => {
  if (userId !== null && userId !== undefined) {
    const normalized = String(userId).trim()
    if (normalized) return `user:${normalized}`
  }

  const normalizedAuthMode = String(authMode || "").toLowerCase().trim()
  if (normalizedAuthMode === "single-user") {
    return "user:single-user"
  }

  const tokenClaim = resolveTokenUserClaim(accessToken)
  if (tokenClaim) return `user:${tokenClaim}`

  return DEFAULT_USER_SCOPE
}

export const buildMediaNavigationScopeKey = (
  input: MediaNavigationScopeInput
): string => {
  const serverFingerprint = deriveServerFingerprint(input.serverUrl)
  const scopedUserId = deriveScopedUserId(input)
  return `${serverFingerprint}:${scopedUserId}`
}

export const buildMediaNavigationMediaKey = (
  input: MediaNavigationScopeInput,
  mediaId: string | number
): string => `${buildMediaNavigationScopeKey(input)}:media:${String(mediaId)}`
