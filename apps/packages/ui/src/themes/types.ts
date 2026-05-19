/** RGB space-separated string, e.g., "47 111 237" */
export type RGBTriple = string

/** The semantic color tokens and shadow values that drive the entire UI */
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
  /** Full CSS box-shadow value for small elevation, e.g., "0 1px 2px rgba(0,0,0,0.1)" */
  shadowSm: string
  /** Full CSS box-shadow value for medium elevation, e.g., "0 4px 12px rgba(0,0,0,0.15)" */
  shadowMd: string
}

/** Keys within ThemeColorTokens that hold RGB triple strings (excludes shadow tokens). */
export type ThemeRgbTokenKey = Exclude<keyof ThemeColorTokens, "shadowSm" | "shadowMd">

export interface ThemePalette {
  light: ThemeColorTokens
  dark: ThemeColorTokens
}

/** Font families and size scale for the theme */
export interface ThemeTypography {
  fontFamily: string
  fontFamilyMono: string
  fontSizeBody: number
  fontSizeMessage: number
  fontSizeCaption: number
  fontSizeLabel: number
}

/** Border-radius and blur values (in px) for surfaces and controls */
export interface ThemeShape {
  radiusSm: number
  radiusMd: number
  radiusLg: number
  radiusXl: number
  surfaceBlur: number
}

/** Sidebar, header, and content dimensions (in px) plus density preference */
export interface ThemeLayout {
  sidebarWidth: number
  sidebarCollapsedWidth: number
  headerHeight: number
  contentMaxWidth: number
  density: "compact" | "default" | "comfortable"
}

/** Visual variant selections for common UI components */
export interface ThemeComponents {
  buttonStyle: "rounded" | "square" | "pill"
  inputStyle: "bordered" | "underlined" | "filled"
  cardStyle: "flat" | "elevated" | "outlined"
  animationSpeed: "none" | "subtle" | "normal"
}

/** Complete theme definition including colors, typography, shape, layout, and component variants */
export interface ThemeDefinition {
  id: string
  name: string
  description?: string
  version: 1
  palette: ThemePalette
  typography: ThemeTypography
  shape: ThemeShape
  layout: ThemeLayout
  components: ThemeComponents
  builtin: boolean
  /** Optional reference to a built-in preset this theme was derived from */
  basePresetId?: string
}
