export const THEME_PRESET_KEY = "tldw:themePreset"
export const THEME_MIGRATION_VERSION_KEY = "tldw:themeMigrationVersion"

export const CURRENT_USER_PREFERENCE_MIGRATION = 1

/**
 * One-shot migration for theme user preferences.
 *
 * Version 1: make `primer` the default preset for fresh installs. Existing
 * users keep their stored preset — even the explicit `"default"` choice — so
 * switching to Primer is never involuntary.
 *
 * Safe to call on every app startup; is idempotent via the version marker.
 */
export function migrateThemeUserPreferences(): void {
  if (typeof window === "undefined") return

  try {
    const rawVersion = window.localStorage.getItem(THEME_MIGRATION_VERSION_KEY)
    // Coerce missing/corrupt values to 0 so migration runs; NaN >= N is always false.
    const storedVersion = Number.parseInt(rawVersion ?? "0", 10)
    const normalizedVersion = Number.isFinite(storedVersion) ? storedVersion : 0
    if (normalizedVersion >= CURRENT_USER_PREFERENCE_MIGRATION) return

    const storedPreset = window.localStorage.getItem(THEME_PRESET_KEY)
    if (storedPreset === null) {
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
