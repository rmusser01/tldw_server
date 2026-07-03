// Shared guard for rendering/opening untrusted URLs (source/citation metadata,
// web-search results, API JSON). Hand-rolled `<a href>` and `window.open` call
// sites bypass the markdown renderer's `urlTransform`, so a `javascript:` URL
// would otherwise execute on click on the app origin. Allowlist http(s)/mailto.

const SAFE_SCHEMES = new Set(["http:", "https:", "mailto:"])

// Browsers strip C0 control characters (incl. tab/newline/CR) and DEL when they
// resolve a URL, so `java\tscript:` becomes `javascript:` at click time. Remove
// them before scheme detection so a control-char scheme can't slip through.
const stripControlChars = (value: string): string => {
  let out = ""
  for (const ch of value) {
    const code = ch.charCodeAt(0)
    // Drop C0 controls (0x00-0x1F) and DEL (0x7F).
    if (code <= 0x1f || code === 0x7f) continue
    out += ch
  }
  return out
}

const isRelativeUrl = (value: string): boolean =>
  value.startsWith("/") ||
  value.startsWith("#") ||
  value.startsWith("./") ||
  value.startsWith("../")

/**
 * Returns a cleaned copy of `url` when it is safe to navigate to, otherwise
 * `null`. "Safe" means an http/https/mailto absolute URL, or a relative URL
 * (path/anchor). Whitespace and control characters are normalized so that
 * obfuscated schemes such as `java\tscript:` cannot bypass the allowlist.
 */
export const safeExternalUrl = (url: unknown): string | null => {
  if (typeof url !== "string") return null
  const cleaned = stripControlChars(url).trim()
  if (!cleaned) return null
  // Relative URLs never carry a dangerous scheme; keep them as-is.
  if (isRelativeUrl(cleaned)) return cleaned
  let parsed: URL
  try {
    // Resolve against a base so both absolute and bare-relative inputs parse;
    // the resulting protocol is authoritative regardless of casing.
    parsed = new URL(cleaned, "http://localhost/")
  } catch {
    return null
  }
  if (!SAFE_SCHEMES.has(parsed.protocol)) return null
  return cleaned
}

/**
 * `window.open` guarded by {@link safeExternalUrl}. No-ops (returns null) when
 * the URL is unsafe or `window` is unavailable (SSR).
 */
export const openExternalUrl = (
  url: unknown,
  target: string = "_blank",
  features: string = "noopener,noreferrer"
): Window | null => {
  const safe = safeExternalUrl(url)
  if (!safe) return null
  if (typeof window === "undefined") return null
  return window.open(safe, target, features)
}
