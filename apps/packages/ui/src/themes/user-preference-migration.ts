export const THEME_PRESET_KEY = "tldw:themePreset"
export const THEME_MIGRATION_VERSION_KEY = "tldw:themeMigrationVersion"

export const CURRENT_USER_PREFERENCE_MIGRATION = 1

const THEME_MIGRATION_IGNORED_KEYS = new Set([
  THEME_PRESET_KEY,
  THEME_MIGRATION_VERSION_KEY,
])

const hasOtherPersistedLocalState = (): boolean => {
  for (let index = 0; index < window.localStorage.length; index += 1) {
    const key = window.localStorage.key(index)
    if (key && !THEME_MIGRATION_IGNORED_KEYS.has(key)) {
      return true
    }
  }
  return false
}

/**
 * One-shot migration for theme user preferences.
 *
 * Version 1: make `primer` the default preset for fresh installs. Existing
 * users keep their stored preset — even the explicit `"default"` choice — so
 * switching to Primer is never involuntary. Blank/corrupt preset markers are
 * repaired to `primer`, while a totally missing preset is only seeded when the
 * browser has no other persisted tldw state yet.
 *
 * Safe to call on every app startup; is idempotent via the version marker.
 */
export function migrateThemeUserPreferences(): void {
  if (typeof window === "undefined") return

  try {
    const rawVersion = window.localStorage.getItem(THEME_MIGRATION_VERSION_KEY)
    const trimmedVersion = rawVersion?.trim() ?? null
    const storedVersion =
      trimmedVersion !== null && /^\d+$/.test(trimmedVersion)
        ? Number(trimmedVersion)
        : 0
    const normalizedVersion = Number.isSafeInteger(storedVersion)
      ? storedVersion
      : 0
    if (normalizedVersion >= CURRENT_USER_PREFERENCE_MIGRATION) return

    const storedPreset = window.localStorage.getItem(THEME_PRESET_KEY)
    const hasStoredPreset =
      typeof storedPreset === "string" && storedPreset.trim().length > 0
    const shouldSeedPrimer =
      !hasStoredPreset &&
      (storedPreset !== null || !hasOtherPersistedLocalState())

    if (shouldSeedPrimer) {
      window.localStorage.setItem(THEME_PRESET_KEY, "primer")
    }

    window.localStorage.setItem(
      THEME_MIGRATION_VERSION_KEY,
      String(CURRENT_USER_PREFERENCE_MIGRATION)
    )
  } catch {
    // Safari private mode, quota exceeded, or storage access denied.
    // Migration is best-effort; running again on next load is safe.
  }
}
