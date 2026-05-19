import type { ThemeRgbTokenKey } from "./types"

const TOKEN_TO_CSS_VAR: Record<ThemeRgbTokenKey, string> = {
  bg: "--color-bg",
  surface: "--color-surface",
  surface2: "--color-surface-2",
  elevated: "--color-elevated",
  primary: "--color-primary",
  primaryStrong: "--color-primary-strong",
  accent: "--color-accent",
  success: "--color-success",
  warn: "--color-warn",
  danger: "--color-danger",
  muted: "--color-muted",
  border: "--color-border",
  borderStrong: "--color-border-strong",
  text: "--color-text",
  textMuted: "--color-text-muted",
  textSubtle: "--color-text-subtle",
  focus: "--color-focus",
}

/**
 * Convert an RGB triple string ("47 111 237") to hex ("#2f6fed").
 */
function tripleToHex(triple: string): string {
  const parts = triple.trim().split(/\s+/)
  const r = Math.max(0, Math.min(255, parseInt(parts[0] ?? "0", 10)))
  const g = Math.max(0, Math.min(255, parseInt(parts[1] ?? "0", 10)))
  const b = Math.max(0, Math.min(255, parseInt(parts[2] ?? "0", 10)))
  return `#${r.toString(16).padStart(2, "0")}${g.toString(16).padStart(2, "0")}${b.toString(16).padStart(2, "0")}`
}

/**
 * Read a single computed CSS custom property value and return it as a hex string.
 * Falls back to "#000000" if the property is not found.
 */
export function getComputedToken(key: ThemeRgbTokenKey): string {
  if (typeof document === "undefined") return "#000000"
  const cssVar = TOKEN_TO_CSS_VAR[key]
  const value = getComputedStyle(document.documentElement)
    .getPropertyValue(cssVar)
    .trim()
  if (!value) return "#000000"
  return tripleToHex(value)
}

/**
 * Read all computed CSS custom property values and return them as hex strings.
 * Useful for JS contexts (Cytoscape, Canvas, Chart.js) that need hex colors.
 */
export function getComputedTokens(): Record<ThemeRgbTokenKey, string> {
  if (typeof document === "undefined") {
    const empty = {} as Record<ThemeRgbTokenKey, string>
    for (const key of Object.keys(TOKEN_TO_CSS_VAR) as ThemeRgbTokenKey[]) {
      empty[key] = "#000000"
    }
    return empty
  }

  const style = getComputedStyle(document.documentElement)
  const result = {} as Record<ThemeRgbTokenKey, string>
  for (const [key, cssVar] of Object.entries(TOKEN_TO_CSS_VAR)) {
    const value = style.getPropertyValue(cssVar).trim()
    result[key as ThemeRgbTokenKey] = value ? tripleToHex(value) : "#000000"
  }
  return result
}
