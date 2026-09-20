// Absolute-URL credential guard — the single canonical source for the MV3
// extension's origin-allowlist + cross-origin auth-suppression rules.
//
// Both the normal request path (`services/tldw/request-core.ts`) and the
// background upload/stream proxy (`services/background-proxy.ts`) previously kept
// their own byte-for-byte copies of this logic; they now import these primitives
// so there is exactly one implementation. This module is intentionally
// dependency-light (no imports) so it is safe to import from the background
// entry, request-core, and background-proxy without circular-import or
// heavy-dependency risk.
//
// `request-core.ts` additionally warns (once) about a malformed configured
// serverUrl / allowlist entry; those diagnostics are supplied via the optional
// `AllowlistWarnHooks` so the shared logic stays free of console side effects for
// its other callers (which silently ignore malformed URLs, as before).

// Any config-shaped object may be passed; only `serverUrl` and
// `absoluteUrlAllowlist` are read (defensively, tolerating any value type).
// The structural arm admits interface-typed configs (e.g. TldwConfig), which
// lack the implicit index signature `Record<string, unknown>` requires.
export type AllowlistConfig =
  | Readonly<{ serverUrl?: unknown; absoluteUrlAllowlist?: unknown }>
  | Record<string, unknown>
  | null
  | undefined

// Optional diagnostics hooks. When omitted, malformed URLs are silently ignored
// (the behaviour the background handlers rely on). request-core supplies these
// to preserve its once-per-value console warnings.
export type AllowlistWarnHooks = {
  onMalformedServerUrl?: (raw: string, error: unknown) => void
  onMalformedAllowlistEntry?: (raw: string, error: unknown) => void
}

export const ABSOLUTE_URL_BLOCK_ERROR =
  "Absolute URL requests are blocked unless the request origin is explicitly allowlisted."

export const isAbsoluteHttpUrl = (path: unknown): boolean =>
  typeof path === "string" && /^https?:/i.test(path)

export const parseHttpOrigin = (
  value: unknown,
  onError?: (raw: string, error: unknown) => void
): string | null => {
  const raw = String(value || "").trim()
  if (!raw) return null
  try {
    const parsed = new URL(raw)
    if (!/^https?:$/i.test(parsed.protocol)) return null
    return parsed.origin.toLowerCase()
  } catch (error) {
    onError?.(raw, error)
    return null
  }
}

const toAllowlistEntries = (value: unknown): string[] => {
  if (Array.isArray(value)) {
    return value
      .map((entry) => String(entry || "").trim())
      .filter((entry) => entry.length > 0)
  }
  if (typeof value === "string") {
    const trimmed = value.trim()
    if (!trimmed) return []
    if (!trimmed.includes(",")) return [trimmed]
    return trimmed
      .split(",")
      .map((entry) => entry.trim())
      .filter((entry) => entry.length > 0)
  }
  return []
}

const configuredServerOrigin = (
  cfg: AllowlistConfig,
  onError?: (raw: string, error: unknown) => void
): string | null =>
  parseHttpOrigin((cfg as Record<string, unknown> | null)?.serverUrl, onError)

export const absoluteOriginAllowlistFromConfig = (
  cfg: AllowlistConfig,
  hooks?: AllowlistWarnHooks
): Set<string> => {
  const out = new Set<string>()
  // The configured serverUrl is parsed silently here (matching request-core's
  // allowlist path, which only warns about explicit allowlist entries).
  const serverOrigin = configuredServerOrigin(cfg)
  if (serverOrigin) out.add(serverOrigin)
  for (const entry of toAllowlistEntries(
    (cfg as Record<string, unknown> | null)?.absoluteUrlAllowlist
  )) {
    const parsedOrigin = parseHttpOrigin(entry, hooks?.onMalformedAllowlistEntry)
    if (parsedOrigin) out.add(parsedOrigin)
  }
  return out
}

export const isAbsoluteUrlAllowlisted = (
  absoluteUrl: string,
  cfg: AllowlistConfig,
  hooks?: AllowlistWarnHooks
): boolean => {
  try {
    const target = new URL(absoluteUrl)
    if (!/^https?:$/i.test(target.protocol)) return false
    return absoluteOriginAllowlistFromConfig(cfg, hooks).has(
      target.origin.toLowerCase()
    )
  } catch {
    return false
  }
}

export const isSameOriginAbsoluteUrlForConfiguredServer = (
  absoluteUrl: string,
  cfg: AllowlistConfig,
  hooks?: AllowlistWarnHooks
): boolean => {
  const serverOrigin = configuredServerOrigin(cfg, hooks?.onMalformedServerUrl)
  if (!serverOrigin) return false
  try {
    const target = new URL(absoluteUrl)
    if (!/^https?:$/i.test(target.protocol)) return false
    return target.origin.toLowerCase() === serverOrigin
  } catch {
    return false
  }
}

export type AbsoluteUrlAccess = {
  // Whether the resolved request path is an absolute http(s) URL.
  isAbsolute: boolean
  // Whether the request must be refused before any fetch (cross-origin and not
  // allowlisted). Mirrors the request-path ABSOLUTE_URL_BLOCK_ERROR guard.
  blocked: boolean
  // Whether auth headers (X-API-KEY / Authorization / X-TLDW-Org-Id) must be
  // withheld. Mirrors request-core's `shouldSkipAuth` for absolute URLs.
  skipAuth: boolean
}

// Decide, for a background upload/stream request path, whether the request is
// absolute, must be blocked, and whether credentials may be attached. This is
// the single decision function the background handlers should call.
export const evaluateAbsoluteUrlAccess = (
  path: unknown,
  cfg: AllowlistConfig,
  hooks?: AllowlistWarnHooks
): AbsoluteUrlAccess => {
  if (!isAbsoluteHttpUrl(path)) {
    return { isAbsolute: false, blocked: false, skipAuth: false }
  }
  const absoluteUrl = String(path)
  const sameOrigin = isSameOriginAbsoluteUrlForConfiguredServer(
    absoluteUrl,
    cfg,
    hooks
  )
  const allowlisted = isAbsoluteUrlAllowlisted(absoluteUrl, cfg, hooks)
  return {
    isAbsolute: true,
    blocked: !sameOrigin && !allowlisted,
    skipAuth: !sameOrigin
  }
}
