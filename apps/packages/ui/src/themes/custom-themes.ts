import type { ThemeDefinition } from "./types"
import { validateThemeDefinition, validateLegacyThemeDefinition, generateThemeId } from "./validation"
import { migrateTheme } from "./migration"
import { CUSTOM_THEMES_SETTING } from "@/services/settings/ui-settings"
import { getStorageForSetting } from "@/services/settings/registry"

function readCustomThemes(): ThemeDefinition[] {
  if (typeof window === "undefined") return []
  try {
    const raw = window.localStorage.getItem(CUSTOM_THEMES_SETTING.localStorageKey!)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return []

    let needsPersist = false
    const themes: ThemeDefinition[] = []

    for (const item of parsed) {
      if (validateThemeDefinition(item)) {
        themes.push(item)
      } else if (validateLegacyThemeDefinition(item)) {
        themes.push(migrateTheme(item as Record<string, unknown>))
        needsPersist = true
      }
      // else: invalid entry, skip
    }

    if (needsPersist) {
      writeCustomThemes(themes)
    }

    return themes
  } catch {
    return []
  }
}

function writeCustomThemes(themes: ThemeDefinition[]): void {
  if (typeof window === "undefined") return
  const key = CUSTOM_THEMES_SETTING.localStorageKey!
  window.localStorage.setItem(key, JSON.stringify(themes))
  // Also update the plasmo storage layer so useSetting picks up the change
  const storage = getStorageForSetting(CUSTOM_THEMES_SETTING)
  void storage.set(CUSTOM_THEMES_SETTING.key, themes)
}

/**
 * Get all custom (user-created) themes from storage.
 */
export function getCustomThemes(): ThemeDefinition[] {
  return readCustomThemes()
}

/**
 * Save a custom theme. If a theme with the same id exists, it's replaced.
 * Built-in themes cannot be overwritten.
 */
export function saveCustomTheme(theme: ThemeDefinition): void {
  if (theme.builtin) return
  const existing = readCustomThemes()
  const idx = existing.findIndex((t) => t.id === theme.id)
  if (idx >= 0) {
    existing[idx] = theme
  } else {
    existing.push(theme)
  }
  writeCustomThemes(existing)
}

/**
 * Delete a custom theme by ID. Built-in themes cannot be deleted.
 */
export function deleteCustomTheme(id: string): void {
  const existing = readCustomThemes()
  writeCustomThemes(existing.filter((t) => t.id !== id))
}

/**
 * Duplicate a theme with a new name and generated ID.
 */
export function duplicateTheme(source: ThemeDefinition, newName: string): ThemeDefinition {
  return {
    ...source,
    id: generateThemeId(newName),
    name: newName,
    version: 1,
    builtin: false,
    palette: {
      light: { ...source.palette.light },
      dark: { ...source.palette.dark },
    },
    typography: { ...source.typography },
    shape: { ...source.shape },
    layout: { ...source.layout },
    components: { ...source.components },
    basePresetId: source.id,
  }
}
