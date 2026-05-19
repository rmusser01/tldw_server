import type {
  ThemeColorTokens,
  ThemeDefinition,
  ThemeTypography,
  ThemeShape,
  ThemeLayout,
  ThemeComponents,
} from "./types"
import { validateRgbTriple } from "./conversion"

// ---------------------------------------------------------------------------
// Color token validation
// ---------------------------------------------------------------------------

/** The 17 keys that must be valid RGB space-separated triples. */
const RGB_KEYS: (keyof ThemeColorTokens)[] = [
  "bg", "surface", "surface2", "elevated",
  "primary", "primaryStrong", "accent",
  "success", "warn", "danger", "muted",
  "border", "borderStrong",
  "text", "textMuted", "textSubtle", "focus",
]

/**
 * Validate that a value has the shape of ThemeColorTokens.
 * Checks all 17 RGB keys plus shadowSm / shadowMd (non-empty strings).
 */
export function validateThemeColorTokens(tokens: unknown): tokens is ThemeColorTokens {
  if (!tokens || typeof tokens !== "object") return false
  const obj = tokens as Record<string, unknown>

  const rgbOk = RGB_KEYS.every(
    (key) => typeof obj[key] === "string" && validateRgbTriple(obj[key] as string),
  )
  if (!rgbOk) return false

  // shadowSm and shadowMd are full CSS box-shadow values, not RGB triples
  if (typeof obj.shadowSm !== "string" || !(obj.shadowSm as string).trim()) return false
  if (typeof obj.shadowMd !== "string" || !(obj.shadowMd as string).trim()) return false

  return true
}

// ---------------------------------------------------------------------------
// Section validators (private helpers)
// ---------------------------------------------------------------------------

function isNumber(v: unknown): v is number {
  return typeof v === "number" && !Number.isNaN(v)
}

function numberInRange(v: unknown, min: number, max: number): boolean {
  return isNumber(v) && v >= min && v <= max
}

function validateTypography(value: unknown): value is ThemeTypography {
  if (!value || typeof value !== "object") return false
  const obj = value as Record<string, unknown>
  if (typeof obj.fontFamily !== "string" || !obj.fontFamily) return false
  if (typeof obj.fontFamilyMono !== "string" || !obj.fontFamilyMono) return false
  if (!numberInRange(obj.fontSizeBody, 8, 32)) return false
  if (!numberInRange(obj.fontSizeMessage, 8, 32)) return false
  if (!numberInRange(obj.fontSizeCaption, 8, 32)) return false
  if (!numberInRange(obj.fontSizeLabel, 8, 32)) return false
  return true
}

function validateShape(value: unknown): value is ThemeShape {
  if (!value || typeof value !== "object") return false
  const obj = value as Record<string, unknown>
  if (!numberInRange(obj.radiusSm, 0, 100)) return false
  if (!numberInRange(obj.radiusMd, 0, 100)) return false
  if (!numberInRange(obj.radiusLg, 0, 100)) return false
  if (!numberInRange(obj.radiusXl, 0, 100)) return false
  if (!numberInRange(obj.surfaceBlur, 0, 100)) return false
  return true
}

function validateLayout(value: unknown): value is ThemeLayout {
  if (!value || typeof value !== "object") return false
  const obj = value as Record<string, unknown>
  if (!numberInRange(obj.sidebarWidth, 150, 600)) return false
  if (!numberInRange(obj.sidebarCollapsedWidth, 40, 120)) return false
  if (!numberInRange(obj.headerHeight, 40, 80)) return false
  if (!numberInRange(obj.contentMaxWidth, 600, 1400)) return false
  if (obj.density !== "compact" && obj.density !== "default" && obj.density !== "comfortable") {
    return false
  }
  return true
}

const BUTTON_STYLES = new Set(["rounded", "square", "pill"])
const INPUT_STYLES = new Set(["bordered", "underlined", "filled"])
const CARD_STYLES = new Set(["flat", "elevated", "outlined"])
const ANIMATION_SPEEDS = new Set(["none", "subtle", "normal"])

function validateComponents(value: unknown): value is ThemeComponents {
  if (!value || typeof value !== "object") return false
  const obj = value as Record<string, unknown>
  if (!BUTTON_STYLES.has(obj.buttonStyle as string)) return false
  if (!INPUT_STYLES.has(obj.inputStyle as string)) return false
  if (!CARD_STYLES.has(obj.cardStyle as string)) return false
  if (!ANIMATION_SPEEDS.has(obj.animationSpeed as string)) return false
  return true
}

// ---------------------------------------------------------------------------
// Legacy (v0) theme validation — used for migration detection
// ---------------------------------------------------------------------------

/** The 17 original RGB-only keys (no shadows). */
const LEGACY_RGB_KEYS: string[] = [
  "bg", "surface", "surface2", "elevated",
  "primary", "primaryStrong", "accent",
  "success", "warn", "danger", "muted",
  "border", "borderStrong",
  "text", "textMuted", "textSubtle", "focus",
]

function validateLegacyColorTokens(tokens: unknown): boolean {
  if (!tokens || typeof tokens !== "object") return false
  const obj = tokens as Record<string, unknown>
  return LEGACY_RGB_KEYS.every(
    (key) => typeof obj[key] === "string" && validateRgbTriple(obj[key] as string),
  )
}

/**
 * Detect a legacy v0 theme definition (no version field, palette with only
 * 17 RGB tokens and no shadow values). Used by custom-themes.ts to identify
 * themes that need migration to v1.
 */
export function validateLegacyThemeDefinition(value: unknown): boolean {
  if (!value || typeof value !== "object") return false
  const obj = value as Record<string, unknown>
  if (typeof obj.id !== "string" || !obj.id) return false
  if (typeof obj.name !== "string" || !obj.name) return false
  if (typeof obj.builtin !== "boolean") return false
  // Legacy themes do not have a version field
  if ("version" in obj) return false
  if (!obj.palette || typeof obj.palette !== "object") return false
  const palette = obj.palette as Record<string, unknown>
  return validateLegacyColorTokens(palette.light) && validateLegacyColorTokens(palette.dark)
}

// ---------------------------------------------------------------------------
// V1 theme definition validation
// ---------------------------------------------------------------------------

/**
 * Validate that a value has the shape of a complete v1 ThemeDefinition.
 */
export function validateThemeDefinition(value: unknown): value is ThemeDefinition {
  if (!value || typeof value !== "object") return false
  const obj = value as Record<string, unknown>
  if (typeof obj.id !== "string" || !obj.id) return false
  if (typeof obj.name !== "string" || !obj.name) return false
  if (typeof obj.builtin !== "boolean") return false
  if (obj.version !== 1) return false
  if (!obj.palette || typeof obj.palette !== "object") return false
  const palette = obj.palette as Record<string, unknown>
  if (!validateThemeColorTokens(palette.light)) return false
  if (!validateThemeColorTokens(palette.dark)) return false
  if (!validateTypography(obj.typography)) return false
  if (!validateShape(obj.shape)) return false
  if (!validateLayout(obj.layout)) return false
  if (!validateComponents(obj.components)) return false
  return true
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/**
 * Generate a unique theme ID from a name.
 * Format: "custom-{slug}-{timestamp36}"
 */
export function generateThemeId(name: string): string {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 20) || "theme"
  const ts = Date.now().toString(36)
  return `custom-${slug}-${ts}`
}

/** Allowed primary font families for curated theme fonts. */
const CURATED_FONTS = new Set([
  "Inter",
  "Space Grotesk",
  "Arimo",
  "system-ui",
  "Georgia",
  "Courier New",
])

/**
 * Check whether the primary font (the part before the first comma) is in
 * the curated allow-list.
 */
export function isCuratedFont(fontFamily: string): boolean {
  const primary = fontFamily.split(",")[0].trim()
  return CURATED_FONTS.has(primary)
}
