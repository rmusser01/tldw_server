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

export function defaultLightShadows() {
  return {
    shadowSm: "0 1px 3px rgba(0,0,0,0.12)",
    shadowMd: "0 6px 18px rgba(0,0,0,0.08)",
  }
}

export function defaultDarkShadows() {
  return {
    shadowSm: "0 1px 2px rgba(0,0,0,0.3)",
    shadowMd: "0 4px 12px rgba(0,0,0,0.25)",
  }
}
