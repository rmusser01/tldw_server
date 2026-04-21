import { useCallback, useEffect, useMemo } from "react"
import type { ThemeConfig } from "antd"
import { useDarkMode } from "~/hooks/useDarkmode"
import { useSetting } from "@/hooks/useSetting"
import { THEME_SETTING, THEME_PRESET_SETTING, CUSTOM_THEMES_SETTING } from "@/services/settings/ui-settings"
import type { ThemeValue } from "@/services/settings/ui-settings"
import type { ThemeColorTokens, ThemeDefinition } from "@/themes/types"
import { getThemeById, getDefaultTheme, getAllPresets } from "@/themes/presets"
import { applyThemeTokens, clearThemeTokens } from "@/themes/apply-theme"
import { buildAntdThemeConfig } from "@/themes/antd-theme"
import { saveCustomTheme as saveCustomThemeFn, deleteCustomTheme as deleteCustomThemeFn } from "@/themes/custom-themes"

export function useTheme() {
  // Existing dark mode hook — handles class toggling, system preference, persistence
  const { mode, toggleDarkMode } = useDarkMode()

  // Theme preset ID from settings
  const [themeId, setThemeId] = useSetting(THEME_PRESET_SETTING)

  // Mode preference (system/dark/light) — we expose the setter for the ThemePicker
  const [modePreference, setModePreference] = useSetting(THEME_SETTING)

  // Custom themes from settings
  const [customThemes, setCustomThemes] = useSetting(CUSTOM_THEMES_SETTING)

  // Resolve the ThemeDefinition from the preset ID (checking builtins + custom)
  const themeDefinition: ThemeDefinition = useMemo(
    () => getThemeById(themeId, customThemes) ?? getDefaultTheme(),
    [themeId, customThemes]
  )

  // Pick light or dark palette based on resolved mode
  const isDark = mode === "dark"
  const tokens: ThemeColorTokens = useMemo(
    () => (isDark ? themeDefinition.palette.dark : themeDefinition.palette.light),
    [isDark, themeDefinition]
  )

  // Apply CSS custom properties when tokens or any theme section changes
  useEffect(() => {
    if (themeDefinition.id === "default") {
      // For the default theme, remove inline overrides so the stylesheet values apply.
      // This ensures backward compatibility.
      clearThemeTokens()
    } else {
      applyThemeTokens(tokens, themeDefinition)
    }
  }, [tokens, themeDefinition])

  // Clean up inline styles if component unmounts (e.g., during HMR)
  useEffect(() => {
    return () => {
      clearThemeTokens()
    }
  }, [])

  // Build Ant Design theme config
  const antdTheme: ThemeConfig = useMemo(
    () => buildAntdThemeConfig(tokens, isDark, themeDefinition.typography, themeDefinition.shape, themeDefinition.layout),
    [tokens, isDark, themeDefinition.typography, themeDefinition.shape, themeDefinition.layout]
  )

  const setThemePresetId = useCallback(
    (id: string) => {
      void setThemeId(id)
    },
    [setThemeId]
  )

  const setModePreferenceWrapped = useCallback(
    (pref: ThemeValue) => {
      void setModePreference(pref)
    },
    [setModePreference]
  )

  // All presets (builtin + custom) for the picker
  const presets = useMemo(
    () => getAllPresets(customThemes),
    [customThemes]
  )

  const saveCustomTheme = useCallback(
    (theme: ThemeDefinition) => {
      saveCustomThemeFn(theme)
      // Update reactive state so useSetting subscribers re-render
      const updated = customThemes.filter((t) => t.id !== theme.id)
      updated.push(theme)
      void setCustomThemes(updated)
    },
    [customThemes, setCustomThemes]
  )

  const deleteCustomTheme = useCallback(
    (id: string) => {
      deleteCustomThemeFn(id)
      void setCustomThemes(customThemes.filter((t) => t.id !== id))
      // If the deleted theme was active, fall back to default
      if (themeId === id) {
        void setThemeId("default")
      }
    },
    [customThemes, setCustomThemes, themeId, setThemeId]
  )

  return {
    /** Resolved mode: "dark" | "light" */
    mode,
    /** User preference: "system" | "dark" | "light" */
    modePreference,
    /** Update mode preference */
    setModePreference: setModePreferenceWrapped,
    /** Current theme preset ID */
    themeId,
    /** Set theme preset by ID */
    setThemeId: setThemePresetId,
    /** Full resolved ThemeDefinition */
    themeDefinition,
    /** Active color tokens (light or dark palette) */
    tokens,
    /** Ant Design ThemeConfig object — pass to ConfigProvider */
    antdTheme,
    /** Toggle dark/light (preserves legacy API) */
    toggleDarkMode,
    /** All available presets (builtin + custom) */
    presets,
    /** User-created custom themes */
    customThemes,
    /** Save or update a custom theme */
    saveCustomTheme,
    /** Delete a custom theme by ID */
    deleteCustomTheme,
  }
}
