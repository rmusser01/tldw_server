# Adjustable Theme System — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Expand the existing 17-color-token theme system into a full design-system with typography, shape, layout, component variants, dual-mode editor, and JSON import/export.

**Architecture:** Flat token expansion with CSS custom properties + data attributes on `<html>`. Ant Design components themed via ConfigProvider + programmatic props. Quick mode (10 levers with derivation) + Advanced mode (5-tab full token editor). JSON import/export with migration.

**Tech Stack:** TypeScript, React, Tailwind CSS 3, Ant Design 6, CSS custom properties, OKLCH color math, localStorage/Plasmo storage.

**Design doc:** `Docs/Plans/2026-04-05-adjustable-theme-system-design.md`

---

## Task 1: Expand Type Definitions

**Files:**
- Modify: `apps/packages/ui/src/themes/types.ts`

**Step 1: Write the expanded types**

Replace the contents of `types.ts` with the expanded interfaces. Keep the existing `RGBTriple` type and `ThemeColorTokens` (adding `shadowSm` and `shadowMd`), then add the 4 new interfaces and expand `ThemeDefinition`:

```typescript
/** RGB space-separated string, e.g., "47 111 237" */
export type RGBTriple = string

/** The 19 semantic color tokens that drive the entire UI */
export interface ThemeColorTokens {
  bg: RGBTriple
  surface: RGBTriple
  surface2: RGBTriple
  elevated: RGBTriple
  primary: RGBTriple
  primaryStrong: RGBTriple
  accent: RGBTriple
  success: RGBTriple
  warn: RGBTriple
  danger: RGBTriple
  muted: RGBTriple
  border: RGBTriple
  borderStrong: RGBTriple
  text: RGBTriple
  textMuted: RGBTriple
  textSubtle: RGBTriple
  focus: RGBTriple
  /** Full box-shadow string, per light/dark mode */
  shadowSm: string
  /** Full box-shadow string, per light/dark mode */
  shadowMd: string
}

export interface ThemeTypography {
  fontFamily: string
  fontFamilyMono: string
  fontSizeBody: number
  fontSizeMessage: number
  fontSizeCaption: number
  fontSizeLabel: number
}

export interface ThemeShape {
  radiusSm: number
  radiusMd: number
  radiusLg: number
  radiusXl: number
  surfaceBlur: number
}

export interface ThemeLayout {
  sidebarWidth: number
  sidebarCollapsedWidth: number
  headerHeight: number
  contentMaxWidth: number
  density: "compact" | "default" | "comfortable"
}

export interface ThemeComponents {
  buttonStyle: "rounded" | "square" | "pill"
  inputStyle: "bordered" | "underlined" | "filled"
  cardStyle: "flat" | "elevated" | "outlined"
  animationSpeed: "none" | "subtle" | "normal"
}

export interface ThemePalette {
  light: ThemeColorTokens
  dark: ThemeColorTokens
}

export interface ThemeDefinition {
  id: string
  name: string
  description?: string
  version: 1
  builtin: boolean
  palette: ThemePalette
  typography: ThemeTypography
  shape: ThemeShape
  layout: ThemeLayout
  components: ThemeComponents
  basePresetId?: string
}
```

**Step 2: Verify TypeScript compilation**

Run: `cd apps && npx tsc --noEmit -p packages/ui/tsconfig.json 2>&1 | head -40`

Expected: Type errors in files that reference the old `ThemeDefinition` shape (missing `version`, `typography`, `shape`, `layout`, `components` fields). This is expected — we'll fix those in subsequent tasks.

**Step 3: Commit**

```bash
git add apps/packages/ui/src/themes/types.ts
git commit -m "feat(theme): expand ThemeDefinition with typography, shape, layout, component types"
```

---

## Task 2: Add Default Token Values and Migration

**Files:**
- Create: `apps/packages/ui/src/themes/defaults.ts`
- Create: `apps/packages/ui/src/themes/migration.ts`

**Step 1: Create defaults.ts with factory functions**

This file provides the default values for all new token sections, used by migration and preset expansion:

```typescript
import type { ThemeTypography, ThemeShape, ThemeLayout, ThemeComponents } from "./types"

export const CURRENT_THEME_VERSION = 1

export function defaultTypography(): ThemeTypography {
  return {
    fontFamily: '"Inter", system-ui, sans-serif',
    fontFamilyMono: '"Courier New", monospace',
    fontSizeBody: 14,
    fontSizeMessage: 15,
    fontSizeCaption: 12,
    fontSizeLabel: 11,
  }
}

export function defaultShape(): ThemeShape {
  return {
    radiusSm: 2,
    radiusMd: 6,
    radiusLg: 8,
    radiusXl: 12,
    surfaceBlur: 0,
  }
}

export function defaultLayout(): ThemeLayout {
  return {
    sidebarWidth: 260,
    sidebarCollapsedWidth: 64,
    headerHeight: 56,
    contentMaxWidth: 960,
    density: "default",
  }
}

export function defaultComponents(): ThemeComponents {
  return {
    buttonStyle: "rounded",
    inputStyle: "bordered",
    cardStyle: "elevated",
    animationSpeed: "normal",
  }
}

/** Default shadow values for light mode */
export function defaultLightShadows() {
  return {
    shadowSm: "0 1px 3px rgba(0,0,0,0.12)",
    shadowMd: "0 6px 18px rgba(0,0,0,0.08)",
  }
}

/** Default shadow values for dark mode */
export function defaultDarkShadows() {
  return {
    shadowSm: "0 1px 2px rgba(0,0,0,0.3)",
    shadowMd: "0 4px 12px rgba(0,0,0,0.25)",
  }
}
```

**Step 2: Create migration.ts**

```typescript
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

/**
 * Migrate a theme from any older version to the current schema.
 * Returns the migrated theme, or throws if the version is too new.
 */
export function migrateTheme(raw: Record<string, unknown>): ThemeDefinition {
  const version = typeof raw.version === "number" ? raw.version : 0

  if (version > CURRENT_THEME_VERSION) {
    throw new Error(
      `Theme version ${version} is newer than supported (${CURRENT_THEME_VERSION}). Please update tldw.`
    )
  }

  if (version < 1) {
    return migrateV0ToV1(raw)
  }

  // Already at current version — validate before returning
  if (!validateThemeDefinition(raw)) {
    throw new Error("Theme claims current version but fails validation")
  }
  return raw
}

/**
 * v0 → v1: The original 17-token schema had no version, typography, shape, layout, or components.
 * Backfill all new sections with defaults. Add shadowSm/shadowMd to palette.
 */
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
```

**Step 3: Verify compilation**

Run: `cd apps && npx tsc --noEmit -p packages/ui/tsconfig.json 2>&1 | grep -c "error"` (count should decrease or stay same — no NEW errors from these files)

**Step 4: Commit**

```bash
git add apps/packages/ui/src/themes/defaults.ts apps/packages/ui/src/themes/migration.ts
git commit -m "feat(theme): add default token factories and v0→v1 migration"
```

---

## Task 3: Update Presets with New Token Sections

**Files:**
- Modify: `apps/packages/ui/src/themes/presets.ts`

**Step 1: Add shadow values and new sections to all 5 presets**

Add `version: 1`, `shadowSm`/`shadowMd` to each preset's light/dark palette, and add `typography`, `shape`, `layout`, `components` sections. Import defaults from `defaults.ts`.

Each preset gets its own shadow values appropriate to its palette. Typography, shape, layout, and components use the defaults for now (presets primarily differ by color). Example for the Default theme:

```typescript
import type { ThemeDefinition } from "./types"
import {
  defaultTypography,
  defaultShape,
  defaultLayout,
  defaultComponents,
  defaultLightShadows,
  defaultDarkShadows,
} from "./defaults"

const defaultTheme: ThemeDefinition = {
  id: "default",
  name: "Default",
  description: "The original tldw palette",
  version: 1,
  builtin: true,
  palette: {
    light: {
      // ... existing 17 tokens unchanged ...
      ...defaultLightShadows(),
    },
    dark: {
      // ... existing 17 tokens unchanged ...
      ...defaultDarkShadows(),
    },
  },
  typography: defaultTypography(),
  shape: defaultShape(),
  layout: defaultLayout(),
  components: defaultComponents(),
}
```

Apply the same pattern to `solarizedTheme`, `nordTheme`, `highContrastTheme`, and `rosePineTheme`. Each keeps its existing 17 color tokens. Add shadow values tuned per-theme:

- **Solarized**: warmer shadows — light: `rgba(88,66,20,0.10)`, dark: `rgba(0,0,0,0.30)`
- **Nord**: cool shadows — light: `rgba(46,52,64,0.10)`, dark: `rgba(0,0,0,0.28)`
- **High Contrast**: minimal shadows — light: `rgba(0,0,0,0.15)`, dark: `rgba(0,0,0,0.40)`
- **Rose Pine**: soft shadows — light: `rgba(87,82,121,0.08)`, dark: `rgba(0,0,0,0.30)`

**Step 2: Verify compilation**

Run: `cd apps && npx tsc --noEmit -p packages/ui/tsconfig.json 2>&1 | head -20`

**Step 3: Commit**

```bash
git add apps/packages/ui/src/themes/presets.ts
git commit -m "feat(theme): add v1 token sections to all 5 built-in presets"
```

---

## Task 4: Update Validation for V1 Schema

**Files:**
- Modify: `apps/packages/ui/src/themes/validation.ts`

**Step 1: Expand validation to cover new sections**

The existing `validateThemeDefinition` and `validateThemeColorTokens` need to handle the new fields. Add shadow keys to the required token list. Add validation for typography, shape, layout, and components sections:

```typescript
import type { ThemeColorTokens, ThemeDefinition, ThemeTypography, ThemeShape, ThemeLayout, ThemeComponents } from "./types"
import { validateRgbTriple } from "./conversion"

const REQUIRED_COLOR_KEYS: (keyof ThemeColorTokens)[] = [
  "bg", "surface", "surface2", "elevated",
  "primary", "primaryStrong", "accent",
  "success", "warn", "danger", "muted",
  "border", "borderStrong",
  "text", "textMuted", "textSubtle", "focus",
  "shadowSm", "shadowMd",
]

/** Keys that must be valid RGB triples (all except shadows) */
const RGB_KEYS: (keyof ThemeColorTokens)[] = [
  "bg", "surface", "surface2", "elevated",
  "primary", "primaryStrong", "accent",
  "success", "warn", "danger", "muted",
  "border", "borderStrong",
  "text", "textMuted", "textSubtle", "focus",
]

const DENSITY_VALUES = ["compact", "default", "comfortable"] as const
const BUTTON_STYLES = ["rounded", "square", "pill"] as const
const INPUT_STYLES = ["bordered", "underlined", "filled"] as const
const CARD_STYLES = ["flat", "elevated", "outlined"] as const
const ANIMATION_SPEEDS = ["none", "subtle", "normal"] as const

const ALLOWED_FONTS = [
  "Inter", "Space Grotesk", "Arimo", "system-ui", "Georgia", "Courier New",
]

export function validateThemeColorTokens(tokens: unknown): tokens is ThemeColorTokens {
  if (!tokens || typeof tokens !== "object") return false
  const obj = tokens as Record<string, unknown>
  // Validate RGB triple keys
  for (const key of RGB_KEYS) {
    if (typeof obj[key] !== "string" || !validateRgbTriple(obj[key] as string)) return false
  }
  // Validate shadow keys (must be non-empty strings, not RGB triples)
  if (typeof obj.shadowSm !== "string" || !obj.shadowSm) return false
  if (typeof obj.shadowMd !== "string" || !obj.shadowMd) return false
  return true
}

function validateTypography(value: unknown): value is ThemeTypography {
  if (!value || typeof value !== "object") return false
  const obj = value as Record<string, unknown>
  if (typeof obj.fontFamily !== "string" || !obj.fontFamily) return false
  if (typeof obj.fontFamilyMono !== "string" || !obj.fontFamilyMono) return false
  for (const key of ["fontSizeBody", "fontSizeMessage", "fontSizeCaption", "fontSizeLabel"] as const) {
    if (typeof obj[key] !== "number" || obj[key] < 8 || obj[key] > 32) return false
  }
  return true
}

function validateShape(value: unknown): value is ThemeShape {
  if (!value || typeof value !== "object") return false
  const obj = value as Record<string, unknown>
  for (const key of ["radiusSm", "radiusMd", "radiusLg", "radiusXl", "surfaceBlur"] as const) {
    if (typeof obj[key] !== "number" || obj[key] < 0 || obj[key] > 100) return false
  }
  return true
}

function validateLayout(value: unknown): value is ThemeLayout {
  if (!value || typeof value !== "object") return false
  const obj = value as Record<string, unknown>
  if (typeof obj.sidebarWidth !== "number" || obj.sidebarWidth < 150 || obj.sidebarWidth > 600) return false
  if (typeof obj.sidebarCollapsedWidth !== "number" || obj.sidebarCollapsedWidth < 40 || obj.sidebarCollapsedWidth > 120) return false
  if (typeof obj.headerHeight !== "number" || obj.headerHeight < 40 || obj.headerHeight > 80) return false
  if (typeof obj.contentMaxWidth !== "number" || obj.contentMaxWidth < 600 || obj.contentMaxWidth > 1400) return false
  if (!DENSITY_VALUES.includes(obj.density as typeof DENSITY_VALUES[number])) return false
  return true
}

function validateComponents(value: unknown): value is ThemeComponents {
  if (!value || typeof value !== "object") return false
  const obj = value as Record<string, unknown>
  if (!BUTTON_STYLES.includes(obj.buttonStyle as typeof BUTTON_STYLES[number])) return false
  if (!INPUT_STYLES.includes(obj.inputStyle as typeof INPUT_STYLES[number])) return false
  if (!CARD_STYLES.includes(obj.cardStyle as typeof CARD_STYLES[number])) return false
  if (!ANIMATION_SPEEDS.includes(obj.animationSpeed as typeof ANIMATION_SPEEDS[number])) return false
  return true
}

/**
 * Validate a v0 theme (legacy 17-token schema, no version field).
 * Used by custom-themes.ts to load old themes before migration.
 */
export function validateLegacyThemeDefinition(value: unknown): boolean {
  if (!value || typeof value !== "object") return false
  const obj = value as Record<string, unknown>
  if (typeof obj.id !== "string" || !obj.id) return false
  if (typeof obj.name !== "string" || !obj.name) return false
  if (typeof obj.builtin !== "boolean") return false
  if (!obj.palette || typeof obj.palette !== "object") return false
  const palette = obj.palette as Record<string, unknown>
  // Legacy palette validation — only the original 17 RGB keys
  const legacyKeys = RGB_KEYS // same keys minus shadowSm/shadowMd
  for (const mode of ["light", "dark"] as const) {
    if (!palette[mode] || typeof palette[mode] !== "object") return false
    const tokens = palette[mode] as Record<string, unknown>
    for (const key of legacyKeys) {
      if (typeof tokens[key] !== "string" || !validateRgbTriple(tokens[key] as string)) return false
    }
  }
  return true
}

/**
 * Validate a complete v1 ThemeDefinition.
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

/** Generate a unique theme ID from a name. */
export function generateThemeId(name: string): string {
  const slug = name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 20) || "theme"
  const ts = Date.now().toString(36)
  return `custom-${slug}-${ts}`
}

/**
 * Check if a font family is in the curated bundled/system list.
 */
export function isCuratedFont(fontFamily: string): boolean {
  const normalized = fontFamily.replace(/['"]/g, "").split(",")[0]?.trim() ?? ""
  return ALLOWED_FONTS.some((f) => f.toLowerCase() === normalized.toLowerCase())
}
```

**Step 2: Verify compilation**

Run: `cd apps && npx tsc --noEmit -p packages/ui/tsconfig.json 2>&1 | head -20`

**Step 3: Commit**

```bash
git add apps/packages/ui/src/themes/validation.ts
git commit -m "feat(theme): expand validation for v1 schema with typography, shape, layout, components"
```

---

## Task 5: Update custom-themes.ts with Migration

**Files:**
- Modify: `apps/packages/ui/src/themes/custom-themes.ts`

**Step 1: Integrate migration into the read path**

Update `readCustomThemes()` to detect legacy themes (no `version` field) and migrate them on first load. Use `validateLegacyThemeDefinition` for v0 themes and `validateThemeDefinition` for v1 themes. Persist migrated themes back to localStorage:

```typescript
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
        // Migrate v0 theme to v1
        try {
          const migrated = migrateTheme(item as Record<string, unknown>)
          themes.push(migrated)
          needsPersist = true
        } catch {
          // Skip themes that fail migration
        }
      }
      // Invalid entries are silently dropped
    }

    // Persist migrated themes so we don't re-migrate on every load
    if (needsPersist) {
      writeCustomThemes(themes)
    }

    return themes
  } catch {
    return []
  }
}

// ... writeCustomThemes, saveCustomTheme, deleteCustomTheme, duplicateTheme unchanged
// except duplicateTheme must also copy the new sections:
```

Update `duplicateTheme` to deep-copy all new sections:

```typescript
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
```

**Step 2: Verify compilation**

Run: `cd apps && npx tsc --noEmit -p packages/ui/tsconfig.json 2>&1 | head -20`

**Step 3: Commit**

```bash
git add apps/packages/ui/src/themes/custom-themes.ts
git commit -m "feat(theme): integrate v0→v1 migration into custom theme loading"
```

---

## Task 6: Update apply-theme.ts for New Token Categories

**Files:**
- Modify: `apps/packages/ui/src/themes/apply-theme.ts`

**Step 1: Expand token application to cover all CSS variables and data attributes**

```typescript
import type { ThemeColorTokens, ThemeDefinition } from "./types"

const COLOR_TOKEN_TO_CSS_VAR: Record<keyof Omit<ThemeColorTokens, "shadowSm" | "shadowMd">, string> = {
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

const ALL_CSS_VARS = [
  ...Object.values(COLOR_TOKEN_TO_CSS_VAR),
  "--shadow-sm", "--shadow-md",
  "--font-family", "--font-family-mono",
  "--font-size-body", "--font-size-message", "--font-size-caption", "--font-size-label",
  "--radius-sm", "--radius-md", "--radius-lg", "--radius-xl", "--surface-blur",
  "--sidebar-width", "--sidebar-collapsed-width", "--header-height", "--content-max-width",
]

const DATA_ATTRIBUTES = [
  "data-density", "data-button-style", "data-input-style",
  "data-card-style", "data-animation",
]

/**
 * Apply all theme tokens as inline CSS custom properties + data attributes on <html>.
 */
export function applyThemeTokens(tokens: ThemeColorTokens, theme: ThemeDefinition): void {
  if (typeof document === "undefined") return
  const el = document.documentElement
  const style = el.style

  // Colors (RGB triples)
  for (const [key, cssVar] of Object.entries(COLOR_TOKEN_TO_CSS_VAR)) {
    style.setProperty(cssVar, tokens[key as keyof ThemeColorTokens] as string)
  }

  // Shadows (full box-shadow strings)
  style.setProperty("--shadow-sm", tokens.shadowSm)
  style.setProperty("--shadow-md", tokens.shadowMd)

  // Typography
  style.setProperty("--font-family", theme.typography.fontFamily)
  style.setProperty("--font-family-mono", theme.typography.fontFamilyMono)
  style.setProperty("--font-size-body", `${theme.typography.fontSizeBody}px`)
  style.setProperty("--font-size-message", `${theme.typography.fontSizeMessage}px`)
  style.setProperty("--font-size-caption", `${theme.typography.fontSizeCaption}px`)
  style.setProperty("--font-size-label", `${theme.typography.fontSizeLabel}px`)

  // Shape
  style.setProperty("--radius-sm", `${theme.shape.radiusSm}px`)
  style.setProperty("--radius-md", `${theme.shape.radiusMd}px`)
  style.setProperty("--radius-lg", `${theme.shape.radiusLg}px`)
  style.setProperty("--radius-xl", `${theme.shape.radiusXl}px`)
  style.setProperty("--surface-blur", `${theme.shape.surfaceBlur}px`)

  // Layout
  style.setProperty("--sidebar-width", `${theme.layout.sidebarWidth}px`)
  style.setProperty("--sidebar-collapsed-width", `${theme.layout.sidebarCollapsedWidth}px`)
  style.setProperty("--header-height", `${theme.layout.headerHeight}px`)
  style.setProperty("--content-max-width", `${theme.layout.contentMaxWidth}px`)

  // Data attributes (component variants + density)
  el.setAttribute("data-density", theme.layout.density)
  el.setAttribute("data-button-style", theme.components.buttonStyle)
  el.setAttribute("data-input-style", theme.components.inputStyle)
  el.setAttribute("data-card-style", theme.components.cardStyle)
  el.setAttribute("data-animation", theme.components.animationSpeed)
}

/**
 * Remove all inline theme overrides, falling back to the stylesheet defaults.
 */
export function clearThemeTokens(): void {
  if (typeof document === "undefined") return
  const el = document.documentElement
  const style = el.style
  for (const cssVar of ALL_CSS_VARS) {
    style.removeProperty(cssVar)
  }
  for (const attr of DATA_ATTRIBUTES) {
    el.removeAttribute(attr)
  }
}
```

**Note:** The signature of `applyThemeTokens` changes from `(tokens: ThemeColorTokens)` to `(tokens: ThemeColorTokens, theme: ThemeDefinition)`. This requires updating call sites in `useTheme.ts` (Task 8).

**Step 2: Commit**

```bash
git add apps/packages/ui/src/themes/apply-theme.ts
git commit -m "feat(theme): expand applyThemeTokens for all CSS vars and data attributes"
```

---

## Task 7: Expand Ant Design Bridge

**Files:**
- Modify: `apps/packages/ui/src/themes/antd-theme.ts`

**Step 1: Update buildAntdThemeConfig to accept new token sections**

```typescript
import { theme, type ThemeConfig } from "antd"
import type { RGBTriple, ThemeColorTokens, ThemeTypography, ThemeShape, ThemeLayout } from "./types"

export function rgbTripleToHex(triple: RGBTriple): string {
  // ... existing implementation unchanged ...
}

/**
 * Build an Ant Design ThemeConfig from our expanded semantic tokens.
 */
export function buildAntdThemeConfig(
  tokens: ThemeColorTokens,
  isDark: boolean,
  typography?: ThemeTypography,
  shape?: ThemeShape,
  layout?: ThemeLayout,
): ThemeConfig {
  const algorithms = [isDark ? theme.darkAlgorithm : theme.defaultAlgorithm]
  if (layout?.density === "compact") {
    algorithms.push(theme.compactAlgorithm)
  }

  return {
    algorithm: algorithms.length === 1 ? algorithms[0] : algorithms,
    token: {
      // Colors (existing)
      colorPrimary: rgbTripleToHex(tokens.primary),
      colorSuccess: rgbTripleToHex(tokens.success),
      colorWarning: rgbTripleToHex(tokens.warn),
      colorError: rgbTripleToHex(tokens.danger),
      colorTextBase: rgbTripleToHex(tokens.text),
      colorBgBase: rgbTripleToHex(tokens.bg),
      colorBorder: rgbTripleToHex(tokens.border),
      colorBgContainer: rgbTripleToHex(tokens.surface),
      colorBgElevated: rgbTripleToHex(tokens.elevated),
      colorLink: rgbTripleToHex(tokens.primary),
      // Typography (new)
      fontFamily: typography?.fontFamily ?? "Arimo",
      fontSize: typography?.fontSizeBody ?? 14,
      // Shape (new)
      ...(shape && {
        borderRadius: shape.radiusMd,
        borderRadiusLG: shape.radiusLg,
        borderRadiusSM: shape.radiusSm,
      }),
      // Shadows (new)
      boxShadow: tokens.shadowSm,
      boxShadowSecondary: tokens.shadowMd,
    },
  }
}
```

**Step 2: Commit**

```bash
git add apps/packages/ui/src/themes/antd-theme.ts
git commit -m "feat(theme): expand Ant Design bridge with typography, shape, layout tokens"
```

---

## Task 8: Update useTheme Hook

**Files:**
- Modify: `apps/packages/ui/src/hooks/useTheme.ts`

**Step 1: Update hook to pass full theme to applyThemeTokens and antd bridge**

The main changes:
1. `applyThemeTokens(tokens)` → `applyThemeTokens(tokens, themeDefinition)`
2. `buildAntdThemeConfig(tokens, isDark)` → `buildAntdThemeConfig(tokens, isDark, themeDefinition.typography, themeDefinition.shape, themeDefinition.layout)`
3. Expose `themeDefinition` fields more directly for component consumption

```typescript
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
  const { mode, toggleDarkMode } = useDarkMode()
  const [themeId, setThemeId] = useSetting(THEME_PRESET_SETTING)
  const [modePreference, setModePreference] = useSetting(THEME_SETTING)
  const [customThemes, setCustomThemes] = useSetting(CUSTOM_THEMES_SETTING)

  const themeDefinition: ThemeDefinition = useMemo(
    () => getThemeById(themeId, customThemes) ?? getDefaultTheme(),
    [themeId, customThemes]
  )

  const isDark = mode === "dark"
  const tokens: ThemeColorTokens = useMemo(
    () => (isDark ? themeDefinition.palette.dark : themeDefinition.palette.light),
    [isDark, themeDefinition]
  )

  // Apply CSS custom properties + data attributes when tokens or theme change
  useEffect(() => {
    if (themeDefinition.id === "default") {
      clearThemeTokens()
    } else {
      applyThemeTokens(tokens, themeDefinition)
    }
  }, [tokens, themeDefinition])

  useEffect(() => {
    return () => { clearThemeTokens() }
  }, [])

  // Build Ant Design theme config with expanded tokens
  const antdTheme: ThemeConfig = useMemo(
    () => buildAntdThemeConfig(
      tokens,
      isDark,
      themeDefinition.typography,
      themeDefinition.shape,
      themeDefinition.layout,
    ),
    [tokens, isDark, themeDefinition.typography, themeDefinition.shape, themeDefinition.layout]
  )

  // ... rest of hook unchanged (setThemePresetId, setModePreferenceWrapped,
  // presets, saveCustomTheme, deleteCustomTheme) ...

  return {
    mode,
    modePreference,
    setModePreference: setModePreferenceWrapped,
    themeId,
    setThemeId: setThemePresetId,
    themeDefinition,
    tokens,
    antdTheme,
    toggleDarkMode,
    presets,
    customThemes,
    saveCustomTheme,
    deleteCustomTheme,
  }
}
```

**Step 2: Verify compilation**

Run: `cd apps && npx tsc --noEmit -p packages/ui/tsconfig.json 2>&1 | head -30`

At this point, the remaining errors should only be in `ThemeEditorModal.tsx` (which constructs a `ThemeDefinition` without the new fields). We fix that next.

**Step 3: Commit**

```bash
git add apps/packages/ui/src/hooks/useTheme.ts
git commit -m "feat(theme): update useTheme hook for expanded token application and antd bridge"
```

---

## Task 9: Update ThemeEditorModal for V1 Schema

**Files:**
- Modify: `apps/packages/ui/src/components/Common/Settings/ThemeEditorModal.tsx`

**Step 1: Update the editor to construct valid v1 ThemeDefinition objects**

The existing editor only manages color tokens. For now, the new sections use defaults. The full advanced editor (Task 15) will replace this later. Key changes:

1. Import defaults and use them in `handleSave`
2. Update export format to include `tldw_theme` marker and all sections
3. Update import to handle both v0 and v1 formats via migration
4. Add `shadowSm`/`shadowMd` to the token editor

Add `shadowSm` and `shadowMd` to the `TOKEN_KEYS` array. In `handleSave`, construct the full v1 shape:

```typescript
const handleSave = useCallback(() => {
  if (!name.trim()) {
    void message.warning("Please enter a theme name")
    return
  }
  const theme: ThemeDefinition = {
    id: editingTheme?.id ?? generateThemeId(name),
    name: name.trim(),
    description: description.trim() || undefined,
    version: 1,
    builtin: false,
    palette: { light: { ...lightTokens }, dark: { ...darkTokens } },
    typography: editingTheme?.typography ?? defaultTypography(),
    shape: editingTheme?.shape ?? defaultShape(),
    layout: editingTheme?.layout ?? defaultLayout(),
    components: editingTheme?.components ?? defaultComponents(),
    basePresetId: editingTheme?.basePresetId,
  }
  onSave(theme)
  onClose()
}, [name, description, lightTokens, darkTokens, editingTheme, onSave, onClose])
```

Update `handleExport` to use the new wrapper format:

```typescript
const handleExport = useCallback(() => {
  const theme: ThemeDefinition = { /* same as handleSave */ }
  const exportData = {
    tldw_theme: true,
    version: 1,
    exported_at: new Date().toISOString(),
    theme,
  }
  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: "application/json" })
  // ... download logic unchanged ...
}, [/* deps */])
```

Update `handleImport` to accept both the new wrapper format and the old raw format, using migration:

```typescript
const handleImport = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
  const file = event.target.files?.[0]
  if (!file) return
  const reader = new FileReader()
  reader.onload = (e) => {
    try {
      const raw = JSON.parse(e.target?.result as string)
      let theme: ThemeDefinition

      // New wrapper format
      if (raw.tldw_theme === true && raw.theme) {
        if (!validateThemeDefinition(raw.theme)) {
          // Try migration
          try {
            theme = migrateTheme(raw.theme)
          } catch {
            void message.error("Invalid theme file format")
            return
          }
        } else {
          theme = raw.theme
        }
      }
      // Legacy raw format (v0)
      else if (validateLegacyThemeDefinition(raw)) {
        theme = migrateTheme(raw as Record<string, unknown>)
      }
      // v1 raw format (no wrapper)
      else if (validateThemeDefinition(raw)) {
        theme = raw
      }
      else {
        void message.error("Invalid theme file format")
        return
      }

      // Populate editor state from imported theme
      setName(theme.name)
      setDescription(theme.description ?? "")
      setLightTokens({ ...theme.palette.light })
      setDarkTokens({ ...theme.palette.dark })
      void message.success(`Theme "${theme.name}" imported`)
    } catch {
      void message.error("Failed to parse theme file")
    }
  }
  reader.readAsText(file)
  event.target.value = ""
}, [])
```

**Step 2: Verify full compilation passes**

Run: `cd apps && npx tsc --noEmit -p packages/ui/tsconfig.json 2>&1 | head -20`

Expected: Zero errors (or only pre-existing errors unrelated to themes).

**Step 3: Commit**

```bash
git add apps/packages/ui/src/components/Common/Settings/ThemeEditorModal.tsx
git commit -m "feat(theme): update ThemeEditorModal for v1 schema, new export wrapper, migration on import"
```

---

## Task 10: Update barrel exports

**Files:**
- Modify: `apps/packages/ui/src/themes/index.ts`

**Step 1: Add new exports**

Add exports for the new modules:

```typescript
export type { RGBTriple, ThemeColorTokens, ThemePalette, ThemeDefinition, ThemeTypography, ThemeShape, ThemeLayout, ThemeComponents } from "./types"
export { getBuiltinPresets, getAllPresets, getThemeById, getDefaultTheme } from "./presets"
export { applyThemeTokens, clearThemeTokens } from "./apply-theme"
export { buildAntdThemeConfig, rgbTripleToHex } from "./antd-theme"
export { hexToRgbTriple, validateRgbTriple } from "./conversion"
export {
  parseRgbTriple,
  relativeLuminance,
  contrastRatio,
  meetsTextContrast,
  meetsNonTextContrast,
  auditThemeTextContrast,
} from "./contrast"
export type { ContrastLevel, ThemeContrastAuditResult } from "./contrast"
export { validateThemeDefinition, validateThemeColorTokens, validateLegacyThemeDefinition, generateThemeId, isCuratedFont } from "./validation"
export { getCustomThemes, saveCustomTheme, deleteCustomTheme, duplicateTheme } from "./custom-themes"
export { getComputedTokens, getComputedToken } from "./runtime-tokens"
export { CURRENT_THEME_VERSION, defaultTypography, defaultShape, defaultLayout, defaultComponents, defaultLightShadows, defaultDarkShadows } from "./defaults"
export { migrateTheme } from "./migration"
```

**Step 2: Commit**

```bash
git add apps/packages/ui/src/themes/index.ts
git commit -m "feat(theme): update barrel exports for new theme modules"
```

---

## Task 11: Update Tailwind Config for CSS Variable Tokens

**Files:**
- Modify: `apps/tldw-frontend/tailwind.config.js`

**Step 1: Add CSS variable references for typography, shape, and shadow tokens**

```javascript
/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: "class",
  content: [
    "../packages/ui/src/**/*.{ts,tsx,html}",
    "./pages/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}"
  ],
  theme: {
    extend: {
      colors: {
        // ... all 17 existing color mappings unchanged ...
      },
      fontFamily: {
        display: ["Space Grotesk", "Inter", "sans-serif"],
        body: ["var(--font-family)", "Inter", "system-ui", "sans-serif"],
        mono: ["var(--font-family-mono)", "Courier New", "monospace"],
        arimo: ["Arimo", "sans-serif"],
      },
      fontSize: {
        body: ["var(--font-size-body, 14px)", { lineHeight: "1.43" }],
        message: ["var(--font-size-message, 15px)", { lineHeight: "1.47" }],
        caption: ["var(--font-size-caption, 12px)", { lineHeight: "1.33" }],
        label: ["var(--font-size-label, 11px)", { lineHeight: "1.27", letterSpacing: "0.04em" }],
      },
      borderRadius: {
        sm: "var(--radius-sm, 2px)",
        md: "var(--radius-md, 6px)",
        lg: "var(--radius-lg, 8px)",
        xl: "var(--radius-xl, 12px)",
        card: "var(--radius-xl, 12px)",
        pill: "9999px",
      },
      boxShadow: {
        sm: "var(--shadow-sm, 0 1px 3px rgba(0,0,0,0.12))",
        md: "var(--shadow-md, 0 6px 18px rgba(0,0,0,0.08))",
        card: "var(--shadow-md, 0 6px 18px rgba(0,0,0,0.16))",
        modal: "var(--shadow-md, 0 10px 30px rgba(0,0,0,0.28))",
      },
      // ... backgroundImage, maskImage, keyframes, animation unchanged ...
    }
  },
  plugins: [require("@tailwindcss/forms"), require("@tailwindcss/typography")]
}
```

**Key design decisions:**
- CSS variable references use fallback values (e.g., `var(--radius-sm, 2px)`) so the default theme (which clears inline styles) still works correctly from the stylesheet values.
- `fontFamily.body` uses the CSS variable with Inter as fallback.
- Standard Tailwind text sizes (`text-xs`, `text-sm`, `text-base`, `text-lg`) are NOT overridden.
- `rounded-sm/md/lg/xl` ARE overridden to use CSS variables — this is intentional per the design.

**Step 2: Commit**

```bash
git add apps/tldw-frontend/tailwind.config.js
git commit -m "feat(theme): wire Tailwind config to CSS variable tokens with fallbacks"
```

---

## Task 12: Add CSS for Default Shadows, Density, Animation, and Component Variants

**Files:**
- Modify: `apps/packages/ui/src/assets/tailwind-shared.css`

**Step 1: Add default shadow CSS variables to :root and .dark**

After the existing `--color-focus` line in `:root`, add:

```css
  --shadow-sm: 0 1px 3px rgba(0,0,0,0.12);
  --shadow-md: 0 6px 18px rgba(0,0,0,0.08);
  --font-family: "Inter", system-ui, sans-serif;
  --font-family-mono: "Courier New", monospace;
  --font-size-body: 14px;
  --font-size-message: 15px;
  --font-size-caption: 12px;
  --font-size-label: 11px;
  --radius-sm: 2px;
  --radius-md: 6px;
  --radius-lg: 8px;
  --radius-xl: 12px;
  --surface-blur: 0px;
  --sidebar-width: 260px;
  --sidebar-collapsed-width: 64px;
  --header-height: 56px;
  --content-max-width: 960px;
  --spacing-unit: 4px;
  --duration-base: 150ms;
```

Add matching dark-mode shadow values in `.dark`:

```css
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
  --shadow-md: 0 4px 12px rgba(0,0,0,0.25);
```

**Step 2: Add density, animation, and component variant CSS inside `@layer components`**

Add these rules at the end of the existing `@layer components` block:

```css
  /* ===== Density ===== */
  [data-density="compact"]     { --spacing-unit: 3px; }
  [data-density="comfortable"] { --spacing-unit: 5px; }

  /* ===== Animation speed ===== */
  [data-animation="none"]   { --duration-base: 0ms; }
  [data-animation="subtle"] { --duration-base: 75ms; }

  /* ===== Button variants (custom components only) ===== */
  [data-button-style="square"]  .tldw-btn { border-radius: var(--radius-sm); }
  [data-button-style="rounded"] .tldw-btn { border-radius: var(--radius-md); }
  [data-button-style="pill"]    .tldw-btn { border-radius: 9999px; }

  /* ===== Input variants (custom components only) ===== */
  [data-input-style="underlined"] .panel-input {
    border-radius: 0;
    border: none;
    border-bottom: 1px solid rgb(var(--color-border));
  }
  [data-input-style="filled"] .panel-input {
    border: none;
    background: rgb(var(--color-surface-2));
  }

  /* ===== Card variants ===== */
  [data-card-style="flat"]     .panel-card { box-shadow: none; border: none; }
  [data-card-style="outlined"] .panel-card {
    box-shadow: none;
    border: 1px solid rgb(var(--color-border));
  }
```

**Step 3: Update existing `.panel-card`, `.panel-input`, `.panel-icon-button` transitions to use `--duration-base`**

Replace hardcoded `duration-150` with `var(--duration-base)`:

```css
  .panel-card {
    @apply rounded-xl border border-border bg-surface text-text shadow-card transition-shadow ease-out motion-reduce:transition-none;
    transition-duration: var(--duration-base);
  }

  .panel-input {
    @apply w-full rounded-md border border-border bg-surface px-3 py-2 text-body text-text placeholder:text-text-muted outline-none transition-colors ease-out motion-reduce:transition-none focus:ring-2 focus:ring-focus;
    transition-duration: var(--duration-base);
  }

  .panel-icon-button {
    @apply inline-flex h-7 w-7 items-center justify-center rounded-md border border-border bg-surface text-text-muted transition-colors ease-out motion-reduce:transition-none hover:bg-surface2 hover:text-text;
    transition-duration: var(--duration-base);
  }
```

**Step 4: Commit**

```bash
git add apps/packages/ui/src/assets/tailwind-shared.css
git commit -m "feat(theme): add CSS variables, density, animation, and component variant rules"
```

---

## Task 13: Create useAntdVariants Hook

**Files:**
- Create: `apps/packages/ui/src/hooks/useAntdVariants.ts`

**Step 1: Create the hook that maps theme component tokens to Ant Design props**

```typescript
import { useMemo } from "react"
import { useTheme } from "./useTheme"

/**
 * Maps theme component variant tokens to Ant Design component props.
 * Use this in components that render Ant Design <Input>, <Select>, or <Button>.
 */
export function useAntdVariants() {
  const { themeDefinition } = useTheme()
  const { buttonStyle, inputStyle } = themeDefinition.components

  return useMemo(() => ({
    /** Pass to <Button shape={buttonShape}> */
    buttonShape: buttonStyle === "pill" ? ("round" as const) : ("default" as const),

    /** Pass to <Input variant={inputVariant}> or <Select variant={inputVariant}> */
    inputVariant: (
      inputStyle === "underlined" ? "borderless"
      : inputStyle === "filled" ? "filled"
      : "outlined"
    ) as "borderless" | "filled" | "outlined",
  }), [buttonStyle, inputStyle])
}
```

**Step 2: Commit**

```bash
git add apps/packages/ui/src/hooks/useAntdVariants.ts
git commit -m "feat(theme): add useAntdVariants hook for Ant Design component variant mapping"
```

---

## Task 14: Create JSON Import/Export Utilities

**Files:**
- Create: `apps/packages/ui/src/themes/import-export.ts`

**Step 1: Create validation and export utilities**

```typescript
import type { ThemeDefinition } from "./types"
import { CURRENT_THEME_VERSION } from "./defaults"
import { validateThemeDefinition, validateLegacyThemeDefinition, generateThemeId, isCuratedFont } from "./validation"
import { migrateTheme } from "./migration"

export interface ThemeFileWrapper {
  tldw_theme: true
  version: number
  exported_at: string
  theme: ThemeDefinition
}

export type ImportResult =
  | { valid: true; theme: ThemeDefinition; warnings: string[] }
  | { valid: false; error: string }

/**
 * Create a theme export wrapper object.
 */
export function createThemeExport(theme: ThemeDefinition): ThemeFileWrapper {
  return {
    tldw_theme: true,
    version: CURRENT_THEME_VERSION,
    exported_at: new Date().toISOString(),
    theme,
  }
}

/**
 * Download a theme as a JSON file.
 */
export function downloadThemeJson(theme: ThemeDefinition): void {
  const data = createThemeExport(theme)
  const slug = theme.name.toLowerCase().replace(/[^a-z0-9]+/g, "-").slice(0, 30)
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" })
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = `tldw-theme-${slug}.json`
  a.click()
  URL.revokeObjectURL(url)
}

/**
 * Validate and parse an imported theme file.
 * Returns the theme with a new ID (to avoid collisions) or an error.
 */
export function parseImportedTheme(jsonString: string): ImportResult {
  let raw: unknown
  try {
    raw = JSON.parse(jsonString)
  } catch {
    return { valid: false, error: "File is not valid JSON" }
  }

  if (!raw || typeof raw !== "object") {
    return { valid: false, error: "File does not contain a JSON object" }
  }

  const obj = raw as Record<string, unknown>
  const warnings: string[] = []
  let theme: ThemeDefinition

  // New wrapper format
  if (obj.tldw_theme === true && obj.theme && typeof obj.theme === "object") {
    const innerVersion = (obj.theme as Record<string, unknown>).version as number | undefined
    if (typeof innerVersion === "number" && innerVersion > CURRENT_THEME_VERSION) {
      return { valid: false, error: `This theme requires a newer version of tldw (theme v${innerVersion}, supported v${CURRENT_THEME_VERSION})` }
    }

    if (validateThemeDefinition(obj.theme)) {
      theme = obj.theme as ThemeDefinition
    } else {
      try {
        theme = migrateTheme(obj.theme as Record<string, unknown>)
      } catch (e) {
        return { valid: false, error: (e as Error).message }
      }
    }
  }
  // Raw v1 theme (no wrapper)
  else if (validateThemeDefinition(obj)) {
    theme = obj as unknown as ThemeDefinition
  }
  // Raw legacy v0 theme
  else if (validateLegacyThemeDefinition(obj)) {
    try {
      theme = migrateTheme(obj as Record<string, unknown>)
      warnings.push("Theme was in legacy format and has been upgraded to v1")
    } catch (e) {
      return { valid: false, error: (e as Error).message }
    }
  }
  else {
    return { valid: false, error: "Not a valid tldw theme file" }
  }

  // Check font family
  if (!isCuratedFont(theme.typography.fontFamily)) {
    warnings.push(`Font "${theme.typography.fontFamily}" is not bundled and may not render consistently`)
  }

  // Generate new ID to avoid collisions
  theme = {
    ...theme,
    id: generateThemeId(theme.name),
    builtin: false,
  }

  return { valid: true, theme, warnings }
}
```

**Step 2: Add to barrel exports**

In `apps/packages/ui/src/themes/index.ts`, add:
```typescript
export { createThemeExport, downloadThemeJson, parseImportedTheme } from "./import-export"
export type { ThemeFileWrapper, ImportResult } from "./import-export"
```

**Step 3: Commit**

```bash
git add apps/packages/ui/src/themes/import-export.ts apps/packages/ui/src/themes/index.ts
git commit -m "feat(theme): add JSON import/export utilities with validation and migration"
```

---

## Task 15: Create Quick Mode Derivation Utilities

**Files:**
- Create: `apps/packages/ui/src/themes/derivation.ts`

**Step 1: Create OKLCH surface derivation and other lever-to-token mapping functions**

```typescript
import type { RGBTriple } from "./types"
import { parseRgbTriple } from "./contrast"
import { hexToRgbTriple, rgbTripleToHex } from "./conversion"

// ---- OKLCH helpers (simplified, no external dependency) ----

function srgbToLinear(c: number): number {
  c = c / 255
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
}

function linearToSrgb(c: number): number {
  c = Math.max(0, Math.min(1, c))
  return Math.round((c <= 0.0031308 ? c * 12.92 : 1.055 * c ** (1 / 2.4) - 0.055) * 255)
}

function rgbToOklch(r: number, g: number, b: number): [number, number, number] {
  const lr = srgbToLinear(r), lg = srgbToLinear(g), lb = srgbToLinear(b)
  const l_ = 0.4122214708 * lr + 0.5363325363 * lg + 0.0514459929 * lb
  const m_ = 0.2119034982 * lr + 0.6806995451 * lg + 0.1073969566 * lb
  const s_ = 0.0883024619 * lr + 0.2817188376 * lg + 0.6299787005 * lb
  const l = Math.cbrt(l_), m = Math.cbrt(m_), s = Math.cbrt(s_)
  const L = 0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s
  const a = 1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s
  const bOk = 0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s
  const C = Math.sqrt(a * a + bOk * bOk)
  const H = Math.atan2(bOk, a) * (180 / Math.PI)
  return [L, C, H < 0 ? H + 360 : H]
}

function oklchToRgb(L: number, C: number, H: number): [number, number, number] {
  const hRad = (H * Math.PI) / 180
  const a = C * Math.cos(hRad), b = C * Math.sin(hRad)
  const l = L + 0.3963377774 * a + 0.2158037573 * b
  const m = L - 0.1055613458 * a - 0.0638541728 * b
  const s = L - 0.0894841775 * a - 1.2914855480 * b
  const l3 = l * l * l, m3 = m * m * m, s3 = s * s * s
  const r = +4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3
  const g = -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3
  const bOut = -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3
  return [linearToSrgb(r), linearToSrgb(g), linearToSrgb(bOut)]
}

/**
 * Derive 4 surface colors from a single tint hex color.
 * Uses OKLCH lightness offsets to generate coherent bg/surface/surface2/elevated.
 */
export function deriveSurfacePalette(
  tintHex: string,
  isDark: boolean
): { bg: RGBTriple; surface: RGBTriple; surface2: RGBTriple; elevated: RGBTriple } {
  const triple = hexToRgbTriple(tintHex)
  const [r, g, b] = parseRgbTriple(triple)
  const [L, C, H] = rgbToOklch(r, g, b)

  // Use very low chroma for surfaces (tinted neutrals)
  const surfaceC = Math.min(C, 0.02)

  const offsets = isDark
    ? { bg: 0, surface: 0.03, surface2: 0.05, elevated: 0.07 }
    : { bg: 0, surface: 0.03, surface2: 0.04, elevated: 0.05 }

  const toTriple = (lOffset: number): RGBTriple => {
    const [nr, ng, nb] = oklchToRgb(Math.max(0, Math.min(1, L + lOffset)), surfaceC, H)
    return `${nr} ${ng} ${nb}`
  }

  return {
    bg: toTriple(offsets.bg),
    surface: toTriple(offsets.surface),
    surface2: toTriple(offsets.surface2),
    elevated: toTriple(offsets.elevated),
  }
}

/**
 * Derive border-radius values from a 0-100 roundness slider.
 */
export function deriveRadii(roundness: number): {
  radiusSm: number; radiusMd: number; radiusLg: number; radiusXl: number;
  buttonStyle: "square" | "rounded" | "pill"
} {
  const t = Math.max(0, Math.min(100, roundness)) / 100
  return {
    radiusSm: Math.round(t * 6),
    radiusMd: Math.round(t * 12),
    radiusLg: Math.round(2 + t * 16),
    radiusXl: Math.round(4 + t * 20),
    buttonStyle: t < 0.2 ? "square" : t > 0.8 ? "pill" : "rounded",
  }
}

/**
 * Derive shadow strings from a 0-100 intensity slider.
 */
export function deriveShadows(
  intensity: number,
  isDark: boolean
): { shadowSm: string; shadowMd: string } {
  const t = Math.max(0, Math.min(100, intensity)) / 100
  const darkFactor = isDark ? 0.6 : 1
  const smAlpha = (t * 0.2 * darkFactor).toFixed(2)
  const mdAlpha = (t * 0.15 * darkFactor).toFixed(2)
  return {
    shadowSm: t === 0 ? "none" : `0 1px ${Math.round(2 + t * 4)}px rgba(0,0,0,${smAlpha})`,
    shadowMd: t === 0 ? "none" : `0 ${Math.round(2 + t * 8)}px ${Math.round(6 + t * 18)}px rgba(0,0,0,${mdAlpha})`,
  }
}
```

**Step 2: Export from barrel**

Add to `index.ts`:
```typescript
export { deriveSurfacePalette, deriveRadii, deriveShadows } from "./derivation"
```

**Step 3: Commit**

```bash
git add apps/packages/ui/src/themes/derivation.ts apps/packages/ui/src/themes/index.ts
git commit -m "feat(theme): add OKLCH surface derivation and quick-mode lever utilities"
```

---

## Tasks 16-18: UI Components (Quick Editor, Advanced Editor, Updated ThemePicker)

These are large UI tasks that should each be their own implementation session. The plan provides the architecture and key interfaces — the implementing agent should:

### Task 16: Quick Mode Editor Component

**Files:**
- Create: `apps/packages/ui/src/components/Common/Settings/ThemeQuickEditor.tsx`

**Architecture:**
- 10 lever controls (color pickers, sliders, dropdowns, segmented toggles)
- Each lever calls its derivation function and applies tokens via `applyThemeTokens()` for live preview
- "Apply" saves as custom theme, "Cancel" reverts via `clearThemeTokens()` + re-apply original
- "Advanced" button opens advanced editor with current values
- Uses Ant Design `Slider`, `ColorPicker`, `Select`, `Segmented` components
- Accessible: all controls labeled, sliders have min/max announced, contrast warning visible

### Task 17: Advanced Mode Editor Component

**Files:**
- Create: `apps/packages/ui/src/components/Common/Settings/ThemeAdvancedEditor.tsx`

**Architecture:**
- 5 `Tabs` (Ant Design) — Colors, Typography, Shape, Layout, Components
- Colors tab: extends existing `ColorTokenRow` pattern, adds shadow string inputs, contrast audit badges
- Typography tab: font `Select` dropdowns, number `InputNumber` for sizes, preview text
- Shape tab: `Slider` for radii with visual preview, surface blur slider
- Layout tab: `Slider` for dimensions, `Segmented` for density
- Components tab: visual `Radio.Group` with button/input/card previews, animation toggle
- Per-tab "Reset to defaults" button
- Live preview via `applyThemeTokens()` on every change

### Task 18: Update ThemePicker Integration

**Files:**
- Modify: `apps/packages/ui/src/components/Common/Settings/ThemePicker.tsx`

**Architecture:**
- Replace "Show advanced theme tools" toggle with mode selector: "Quick" / "Advanced"
- Quick mode opens `ThemeQuickEditor` modal
- Advanced mode opens `ThemeAdvancedEditor` modal (replaces current `ThemeEditorModal`)
- Import/Export buttons moved to the picker level (not inside editor)
- "Duplicate & Export" button on builtin presets
- Keep existing `ThemeSwatch` component

---

## Tasks 19-20: Layout Refactoring (High Risk)

These are the most invasive changes and should be done in a separate PR with visual regression testing.

### Task 19: Wire Layout CSS Variables into Components

Search for hardcoded sidebar width, header height, and content max-width values in components. Replace with `var(--sidebar-width)`, `var(--header-height)`, `var(--content-max-width)` in inline styles or CSS.

**Key areas to search:**
- Sidebar components: look for `w-64`, `w-[260px]`, or similar
- Header components: look for `h-14`, `h-[56px]`, or similar
- Content wrappers: look for `max-w-*` classes

### Task 20: Density Overrides for Custom Components

Add `data-density` CSS rules for custom components beyond the already-defined `.panel-card` and `.tldw-btn` rules. Add `.tldw-btn` class to the custom `Button` component.

---

## Execution Order & Dependencies

```
Task 1 (types) → Task 2 (defaults/migration) → Task 3 (presets) → Task 4 (validation)
    → Task 5 (custom-themes) → Task 6 (apply-theme) → Task 7 (antd bridge)
    → Task 8 (useTheme) → Task 9 (ThemeEditorModal) → Task 10 (barrel exports)
    → Task 11 (tailwind config) → Task 12 (CSS rules) → Task 13 (useAntdVariants)

Task 14 (import/export) — depends on Tasks 2, 4
Task 15 (derivation) — depends on Task 1

Tasks 16-18 (UI) — depend on Tasks 1-15
Tasks 19-20 (layout) — depend on Tasks 11-12, can parallel with 16-18
```

**Parallelizable groups:**
- Tasks 14 + 15 can run in parallel after Task 4
- Tasks 16 + 17 can run in parallel after Task 15
- Tasks 19 + 20 can run in parallel with Tasks 16-18
