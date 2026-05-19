import type { ThemeDefinition } from "./types"
import { CURRENT_THEME_VERSION } from "./defaults"
import {
  validateThemeDefinition,
  validateLegacyThemeDefinition,
  generateThemeId,
  isCuratedFont,
} from "./validation"
import { migrateTheme } from "./migration"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** Envelope format for exported theme JSON files. */
export interface ThemeFileWrapper {
  tldw_theme: true
  version: number
  exported_at: string
  theme: ThemeDefinition
}

/** Result of attempting to import a theme from JSON. */
export type ImportResult =
  | { valid: true; theme: ThemeDefinition; warnings: string[] }
  | { valid: false; error: string }

// ---------------------------------------------------------------------------
// Export helpers
// ---------------------------------------------------------------------------

/** Wrap a theme in the standard export envelope. */
export function createThemeExport(theme: ThemeDefinition): ThemeFileWrapper {
  return {
    tldw_theme: true,
    version: CURRENT_THEME_VERSION,
    exported_at: new Date().toISOString(),
    theme,
  }
}

/**
 * Trigger a browser download of the theme as a JSON file.
 * Filename: `tldw-theme-{slug}.json` where slug is derived from the theme name.
 */
export function downloadThemeJson(theme: ThemeDefinition): void {
  const wrapper = createThemeExport(theme)
  const json = JSON.stringify(wrapper, null, 2)
  const blob = new Blob([json], { type: "application/json" })
  const url = URL.createObjectURL(blob)

  const slug = theme.name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 40) || "custom"

  const anchor = document.createElement("a")
  anchor.href = url
  anchor.download = `tldw-theme-${slug}.json`
  document.body.appendChild(anchor)
  anchor.click()

  // Clean up
  document.body.removeChild(anchor)
  URL.revokeObjectURL(url)
}

// ---------------------------------------------------------------------------
// Import
// ---------------------------------------------------------------------------

/**
 * Parse and validate a JSON string as an importable theme.
 *
 * Handles three formats:
 * 1. **Wrapper** — object with `tldw_theme: true` wrapping a theme
 * 2. **Raw v1** — a bare ThemeDefinition with `version: 1`
 * 3. **Raw legacy v0** — a bare legacy theme (no version field)
 */
export function parseImportedTheme(jsonString: string): ImportResult {
  // Step 1: Parse JSON
  let parsed: unknown
  try {
    parsed = JSON.parse(jsonString)
  } catch {
    return { valid: false, error: "File contents are not valid JSON." }
  }

  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    return { valid: false, error: "Expected a JSON object." }
  }

  const obj = parsed as Record<string, unknown>
  const warnings: string[] = []

  // Step 2: Determine format and extract the raw theme object
  let rawTheme: unknown

  if (obj.tldw_theme === true) {
    // Wrapper format
    const wrapperVersion = typeof obj.version === "number" ? obj.version : 0
    if (wrapperVersion > CURRENT_THEME_VERSION) {
      return {
        valid: false,
        error:
          `This theme was exported from a newer version of tldw (format v${wrapperVersion}). ` +
          `Please update tldw to import it (current format: v${CURRENT_THEME_VERSION}).`,
      }
    }
    rawTheme = obj.theme
    if (!rawTheme || typeof rawTheme !== "object") {
      return { valid: false, error: "Wrapper is missing the 'theme' property." }
    }
  } else {
    // Bare theme (v1 or legacy v0)
    rawTheme = obj
  }

  // Step 3: Validate as v1, or try legacy v0 + migration
  let theme: ThemeDefinition

  if (validateThemeDefinition(rawTheme)) {
    theme = rawTheme
  } else if (validateLegacyThemeDefinition(rawTheme)) {
    try {
      theme = migrateTheme(rawTheme as Record<string, unknown>)
      warnings.push("Theme was in legacy v0 format and has been migrated to v1.")
    } catch (err) {
      return {
        valid: false,
        error: `Failed to migrate legacy theme: ${err instanceof Error ? err.message : String(err)}`,
      }
    }

    // Validate the migrated result
    if (!validateThemeDefinition(theme)) {
      return { valid: false, error: "Migrated theme failed validation." }
    }
  } else {
    return {
      valid: false,
      error: "The file does not contain a valid tldw theme definition.",
    }
  }

  // Step 4: Check font curation
  if (!isCuratedFont(theme.typography.fontFamily)) {
    warnings.push(
      `Font "${theme.typography.fontFamily.split(",")[0].trim()}" is not in the curated font list. ` +
      "It may not render correctly if the font is not installed.",
    )
  }

  // Step 5: Generate a new ID and mark as non-builtin
  theme = {
    ...theme,
    id: generateThemeId(theme.name),
    builtin: false,
  }

  return { valid: true, theme, warnings }
}
