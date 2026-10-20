import type { RGBTriple, ThemeColorTokens } from "./types"

export type ContrastLevel = "AA" | "AAA"

export type ThemeContrastAuditResult = {
  pair: string
  ratio: number
  passesAA: boolean
  passesAAA: boolean
}

const RGB_TOKEN_SPLIT = /\s+/

const clampChannel = (value: number): number => {
  if (!Number.isFinite(value)) return 0
  if (value < 0) return 0
  if (value > 255) return 255
  return value
}

const srgbToLinear = (channel: number): number => {
  const normalized = clampChannel(channel) / 255
  if (normalized <= 0.03928) {
    return normalized / 12.92
  }
  return ((normalized + 0.055) / 1.055) ** 2.4
}

export const parseRgbTriple = (value: RGBTriple): [number, number, number] => {
  const parts = String(value || "")
    .trim()
    .split(RGB_TOKEN_SPLIT)
    .filter(Boolean)
    .slice(0, 3)
    .map((part) => Number.parseInt(part, 10))

  const [r = 0, g = 0, b = 0] = parts
  return [clampChannel(r), clampChannel(g), clampChannel(b)]
}

export const relativeLuminance = (value: RGBTriple): number => {
  const [r, g, b] = parseRgbTriple(value)
  const rLinear = srgbToLinear(r)
  const gLinear = srgbToLinear(g)
  const bLinear = srgbToLinear(b)

  return 0.2126 * rLinear + 0.7152 * gLinear + 0.0722 * bLinear
}

export const contrastRatio = (foreground: RGBTriple, background: RGBTriple): number => {
  const foregroundLum = relativeLuminance(foreground)
  const backgroundLum = relativeLuminance(background)
  const lighter = Math.max(foregroundLum, backgroundLum)
  const darker = Math.min(foregroundLum, backgroundLum)
  return (lighter + 0.05) / (darker + 0.05)
}

export const meetsTextContrast = (
  foreground: RGBTriple,
  background: RGBTriple,
  level: ContrastLevel = "AA"
): boolean => {
  const ratio = contrastRatio(foreground, background)
  return level === "AAA" ? ratio >= 7 : ratio >= 4.5
}

export const meetsNonTextContrast = (
  indicator: RGBTriple,
  adjacentColor: RGBTriple
): boolean => contrastRatio(indicator, adjacentColor) >= 3

const BLACK: RGBTriple = "0 0 0"
const WHITE: RGBTriple = "255 255 255"

/**
 * Black or white, whichever reads better on `background`. Black wins ties.
 * Used for text on solid primary and danger fills.
 */
export const readableForeground = (background: RGBTriple): RGBTriple =>
  contrastRatio(BLACK, background) >= contrastRatio(WHITE, background)
    ? BLACK
    : WHITE

/**
 * The first candidate whose lowest contrast against `backgrounds` reaches
 * `minimum`; otherwise the candidate with the best lowest contrast.
 */
export const firstReadableColor = (
  candidates: RGBTriple[],
  backgrounds: RGBTriple[],
  minimum = 4.5
): RGBTriple => {
  const worst = (color: RGBTriple) =>
    Math.min(...backgrounds.map((background) => contrastRatio(color, background)))
  const readable = candidates.find((color) => worst(color) >= minimum)
  if (readable) return readable
  return candidates.reduce((best, color) => (worst(color) > worst(best) ? color : best))
}

const toAuditResult = (
  pair: string,
  foreground: RGBTriple,
  background: RGBTriple
): ThemeContrastAuditResult => {
  const ratio = contrastRatio(foreground, background)
  return {
    pair,
    ratio,
    passesAA: ratio >= 4.5,
    passesAAA: ratio >= 7,
  }
}

export const auditThemeTextContrast = (
  tokens: ThemeColorTokens
): ThemeContrastAuditResult[] => [
  toAuditResult("text/bg", tokens.text, tokens.bg),
  toAuditResult("text/surface", tokens.text, tokens.surface),
  toAuditResult("text/surface2", tokens.text, tokens.surface2),
  toAuditResult("textMuted/surface", tokens.textMuted, tokens.surface),
  toAuditResult("textMuted/surface2", tokens.textMuted, tokens.surface2),
  toAuditResult("textSubtle/surface", tokens.textSubtle, tokens.surface),
  toAuditResult("textSubtle/surface2", tokens.textSubtle, tokens.surface2),
]
