import type { ThemeDefinition } from "./types"
import {
  CURRENT_THEME_VERSION,
  defaultTypography,
  defaultShape,
  defaultLayout,
  defaultComponents,
  defaultLightShadows,
  defaultDarkShadows,
} from "./defaults"
import { validateThemeDefinition } from "./validation"

/**
 * Migrate a theme from any older version to the current schema.
 * Throws if the version is newer than supported or if the theme is invalid.
 */
export function migrateTheme(raw: Record<string, unknown>): ThemeDefinition {
  const version = typeof raw.version === "number" ? raw.version : 0

  if (version > CURRENT_THEME_VERSION) {
    throw new Error(
      `Theme version ${version} is newer than supported (${CURRENT_THEME_VERSION}). Please update tldw.`
    )
  }

  const result = version < 1 ? migrateV0ToV1(raw) : raw

  if (!validateThemeDefinition(result)) {
    throw new Error("Theme failed validation after migration.")
  }

  return result
}

function migrateV0ToV1(raw: Record<string, unknown>): ThemeDefinition {
  const palette = raw.palette as { light: Record<string, string>; dark: Record<string, string> } | undefined

  const lightShadows = defaultLightShadows()
  const darkShadows = defaultDarkShadows()

  return {
    id: raw.id as string,
    name: raw.name as string,
    description: (raw.description as string) ?? undefined,
    version: 1,
    builtin: (raw.builtin as boolean) ?? false,
    palette: {
      light: {
        ...palette?.light,
        shadowSm: palette?.light?.shadowSm ?? lightShadows.shadowSm,
        shadowMd: palette?.light?.shadowMd ?? lightShadows.shadowMd,
      } as ThemeDefinition["palette"]["light"],
      dark: {
        ...palette?.dark,
        shadowSm: palette?.dark?.shadowSm ?? darkShadows.shadowSm,
        shadowMd: palette?.dark?.shadowMd ?? darkShadows.shadowMd,
      } as ThemeDefinition["palette"]["dark"],
    },
    typography: defaultTypography(),
    shape: defaultShape(),
    layout: defaultLayout(),
    components: defaultComponents(),
    basePresetId: undefined,
  }
}
