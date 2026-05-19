import type { ThemeColorTokens, ThemeDefinition } from "./types"

// ---------------------------------------------------------------------------
// Color token -> CSS variable mapping (original 17 color tokens)
// ---------------------------------------------------------------------------
const TOKEN_TO_CSS_VAR: Record<keyof Pick<ThemeColorTokens,
  "bg" | "surface" | "surface2" | "elevated" | "primary" | "primaryStrong" |
  "accent" | "success" | "warn" | "danger" | "muted" | "border" |
  "borderStrong" | "text" | "textMuted" | "textSubtle" | "focus"
>, string> = {
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

// ---------------------------------------------------------------------------
// Comprehensive list of every CSS custom property we set, for clean removal.
// ---------------------------------------------------------------------------
const ALL_CSS_VARS: string[] = [
  // Colors
  ...Object.values(TOKEN_TO_CSS_VAR),
  // Shadows
  "--shadow-sm",
  "--shadow-md",
  // Typography
  "--font-family",
  "--font-family-mono",
  "--font-size-body",
  "--font-size-message",
  "--font-size-caption",
  "--font-size-label",
  // Shape
  "--radius-sm",
  "--radius-md",
  "--radius-lg",
  "--radius-xl",
  "--surface-blur",
  // Layout
  "--sidebar-width",
  "--sidebar-collapsed-width",
  "--header-height",
  "--content-max-width",
]

// ---------------------------------------------------------------------------
// Data attributes we set on <html> for component / density / animation styles.
// ---------------------------------------------------------------------------
const DATA_ATTRIBUTES: string[] = [
  "data-density",
  "data-button-style",
  "data-input-style",
  "data-card-style",
  "data-animation",
]

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function px(value: number): string {
  return `${value}px`
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Apply the full theme as inline CSS custom properties and data attributes on
 * `<html>`.  Inline styles beat `:root` / `.dark` stylesheet rules in
 * specificity, so the theme overrides the default palette without touching the
 * stylesheet.
 *
 * @param tokens - The resolved color tokens for the current color-scheme mode.
 * @param theme  - The full theme definition (typography, shape, layout, components).
 */
export function applyThemeTokens(tokens: ThemeColorTokens, theme: ThemeDefinition): void {
  if (typeof document === "undefined") return

  const style = document.documentElement.style
  const html = document.documentElement

  // --- Colors (RGB triples) ---
  for (const [key, cssVar] of Object.entries(TOKEN_TO_CSS_VAR)) {
    style.setProperty(cssVar, tokens[key as keyof ThemeColorTokens] as string)
  }

  // --- Shadows (raw CSS values) ---
  style.setProperty("--shadow-sm", tokens.shadowSm)
  style.setProperty("--shadow-md", tokens.shadowMd)

  // --- Typography ---
  const { typography } = theme
  style.setProperty("--font-family", typography.fontFamily)
  style.setProperty("--font-family-mono", typography.fontFamilyMono)
  style.setProperty("--font-size-body", px(typography.fontSizeBody))
  style.setProperty("--font-size-message", px(typography.fontSizeMessage))
  style.setProperty("--font-size-caption", px(typography.fontSizeCaption))
  style.setProperty("--font-size-label", px(typography.fontSizeLabel))

  // --- Shape ---
  const { shape } = theme
  style.setProperty("--radius-sm", px(shape.radiusSm))
  style.setProperty("--radius-md", px(shape.radiusMd))
  style.setProperty("--radius-lg", px(shape.radiusLg))
  style.setProperty("--radius-xl", px(shape.radiusXl))
  style.setProperty("--surface-blur", px(shape.surfaceBlur))

  // --- Layout ---
  const { layout } = theme
  style.setProperty("--sidebar-width", px(layout.sidebarWidth))
  style.setProperty("--sidebar-collapsed-width", px(layout.sidebarCollapsedWidth))
  style.setProperty("--header-height", px(layout.headerHeight))
  style.setProperty("--content-max-width", px(layout.contentMaxWidth))

  // --- Data attributes (component variants, density, animation) ---
  html.setAttribute("data-density", layout.density)
  const { components } = theme
  html.setAttribute("data-button-style", components.buttonStyle)
  html.setAttribute("data-input-style", components.inputStyle)
  html.setAttribute("data-card-style", components.cardStyle)
  html.setAttribute("data-animation", components.animationSpeed)
}

/**
 * Remove all inline theme overrides and data attributes, falling back to the
 * stylesheet defaults (`:root` / `.dark`).
 * Called when the user selects the "default" theme.
 */
export function clearThemeTokens(): void {
  if (typeof document === "undefined") return

  const style = document.documentElement.style
  const html = document.documentElement

  // Remove all CSS custom properties
  for (const cssVar of ALL_CSS_VARS) {
    style.removeProperty(cssVar)
  }

  // Remove all data attributes
  for (const attr of DATA_ATTRIBUTES) {
    html.removeAttribute(attr)
  }
}
