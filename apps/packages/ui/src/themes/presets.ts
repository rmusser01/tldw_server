import type { ThemeDefinition } from "./types"
import {
  defaultTypography,
  defaultShape,
  defaultLayout,
  defaultComponents,
  defaultLightShadows,
  defaultDarkShadows,
} from "./defaults"
import { primerTheme } from "./primer-preset"

/**
 * Default theme — matches the existing CSS custom properties in tailwind-shared.css.
 * Light text-subtle has been darkened for WCAG AA compliance.
 */
const defaultTheme: ThemeDefinition = {
  id: "default",
  name: "Default",
  description: "The original tldw palette",
  version: 1 as const,
  builtin: true,
  palette: {
    light: {
      bg: "244 242 238",
      surface: "255 255 255",
      surface2: "240 237 231",
      elevated: "251 250 247",
      primary: "47 111 237",
      primaryStrong: "36 86 199",
      accent: "31 181 159",
      success: "34 160 123",
      warn: "217 119 6",
      danger: "224 88 109",
      muted: "102 112 133",
      border: "226 221 211",
      borderStrong: "207 200 186",
      text: "31 35 40",
      textMuted: "91 100 114",
      textSubtle: "110 120 135",
      focus: "13 134 119",
      ...defaultLightShadows(),
    },
    dark: {
      bg: "15 17 19",
      surface: "23 26 31",
      surface2: "29 33 40",
      elevated: "35 40 50",
      primary: "92 141 255",
      primaryStrong: "62 106 224",
      accent: "79 209 176",
      success: "59 214 150",
      warn: "247 185 85",
      danger: "255 107 139",
      muted: "154 163 178",
      border: "43 49 59",
      borderStrong: "59 67 80",
      text: "231 233 238",
      textMuted: "179 186 198",
      textSubtle: "144 152 166",
      focus: "79 209 176",
      ...defaultDarkShadows(),
    },
  },
  typography: defaultTypography(),
  shape: defaultShape(),
  layout: defaultLayout(),
  components: defaultComponents(),
}

/**
 * Solarized — warm academic palette inspired by Ethan Schoonover's Solarized.
 */
const solarizedTheme: ThemeDefinition = {
  id: "solarized",
  name: "Solarized",
  description: "Warm academic tones with yellow and orange accents",
  version: 1 as const,
  builtin: true,
  palette: {
    light: {
      bg: "253 246 227",        // base3
      surface: "238 232 213",   // base2
      surface2: "227 221 201",
      elevated: "255 250 235",
      primary: "38 139 210",    // blue
      primaryStrong: "32 117 178",
      accent: "42 161 152",     // cyan
      success: "133 153 0",     // green
      warn: "181 137 0",        // yellow
      danger: "220 50 47",      // red
      muted: "101 123 131",     // base00
      border: "210 203 184",
      borderStrong: "192 184 163",
      text: "7 54 66",          // base02
      textMuted: "78 98 105",   // darkened for AA on surface/surface2
      textSubtle: "101 123 131",// base00
      focus: "34 145 136",      // darkened for >=3:1 focus indicator contrast
      shadowSm: "0 1px 3px rgba(88,66,20,0.10)",
      shadowMd: "0 6px 18px rgba(88,66,20,0.08)",
    },
    dark: {
      bg: "0 43 54",            // base03
      surface: "7 54 66",       // base02
      surface2: "14 64 77",
      elevated: "22 74 88",
      primary: "38 139 210",    // blue
      primaryStrong: "62 157 224",
      accent: "42 161 152",     // cyan
      success: "133 153 0",     // green
      warn: "181 137 0",        // yellow
      danger: "220 50 47",      // red
      muted: "147 161 161",     // base1
      border: "27 72 85",
      borderStrong: "42 88 102",
      text: "253 246 227",      // base3
      textMuted: "238 232 213", // base2
      textSubtle: "147 161 161",// base1
      focus: "42 161 152",
      shadowSm: "0 1px 2px rgba(0,0,0,0.30)",
      shadowMd: "0 4px 12px rgba(0,0,0,0.25)",
    },
  },
  typography: defaultTypography(),
  shape: defaultShape(),
  layout: defaultLayout(),
  components: defaultComponents(),
}

/**
 * Nord — cool arctic palette from the Nord color scheme.
 */
const nordTheme: ThemeDefinition = {
  id: "nord",
  name: "Nord",
  description: "Cool blue-gray arctic tones",
  version: 1 as const,
  builtin: true,
  palette: {
    light: {
      bg: "236 239 244",        // Snow Storm nord6
      surface: "229 233 240",   // nord5
      surface2: "216 222 233",  // nord4
      elevated: "242 244 248",
      primary: "94 129 172",    // Frost nord10
      primaryStrong: "76 108 148",
      accent: "136 192 208",    // nord8
      success: "163 190 140",   // Aurora nord14
      warn: "235 203 139",      // nord13
      danger: "191 97 106",     // nord11
      muted: "104 117 140",
      border: "200 207 221",
      borderStrong: "182 191 209",
      text: "46 52 64",         // Polar Night nord0
      textMuted: "67 76 94",    // nord1
      textSubtle: "104 117 140",
      focus: "94 129 172",      // align with primary for >=3:1 focus contrast
      shadowSm: "0 1px 3px rgba(46,52,64,0.10)",
      shadowMd: "0 6px 18px rgba(46,52,64,0.08)",
    },
    dark: {
      bg: "46 52 64",           // Polar Night nord0
      surface: "59 66 82",      // nord1
      surface2: "67 76 94",     // nord2
      elevated: "76 86 106",    // nord3
      primary: "136 192 208",   // Frost nord8
      primaryStrong: "129 161 193",// nord9
      accent: "94 129 172",     // nord10
      success: "163 190 140",   // nord14
      warn: "235 203 139",      // nord13
      danger: "191 97 106",     // nord11
      muted: "182 191 209",
      border: "67 76 94",
      borderStrong: "76 86 106",
      text: "236 239 244",      // Snow Storm nord6
      textMuted: "229 233 240", // nord5
      textSubtle: "216 222 233",// nord4
      focus: "136 192 208",
      shadowSm: "0 1px 2px rgba(0,0,0,0.28)",
      shadowMd: "0 4px 12px rgba(0,0,0,0.22)",
    },
  },
  typography: defaultTypography(),
  shape: defaultShape(),
  layout: defaultLayout(),
  components: defaultComponents(),
}

/**
 * High Contrast — WCAG AAA compliant, near-black/near-white backgrounds.
 */
const highContrastTheme: ThemeDefinition = {
  id: "high-contrast",
  name: "High Contrast",
  description: "Maximum readability, WCAG AAA compliant",
  version: 1 as const,
  builtin: true,
  palette: {
    light: {
      bg: "255 255 255",
      surface: "245 245 245",
      surface2: "235 235 235",
      elevated: "250 250 250",
      primary: "0 70 180",
      primaryStrong: "0 55 145",
      accent: "0 130 110",
      success: "0 120 60",
      warn: "180 90 0",
      danger: "190 30 45",
      muted: "75 75 75",
      border: "180 180 180",
      borderStrong: "130 130 130",
      text: "0 0 0",
      textMuted: "50 50 50",
      textSubtle: "75 75 75",
      focus: "0 70 180",
      shadowSm: "0 1px 3px rgba(0,0,0,0.15)",
      shadowMd: "0 6px 18px rgba(0,0,0,0.12)",
    },
    dark: {
      bg: "0 0 0",
      surface: "18 18 18",
      surface2: "30 30 30",
      elevated: "40 40 40",
      primary: "100 160 255",
      primaryStrong: "70 130 230",
      accent: "80 220 190",
      success: "70 230 130",
      warn: "255 190 60",
      danger: "255 90 100",
      muted: "180 180 180",
      border: "70 70 70",
      borderStrong: "100 100 100",
      text: "255 255 255",
      textMuted: "210 210 210",
      textSubtle: "180 180 180",
      focus: "100 160 255",
      shadowSm: "0 1px 2px rgba(0,0,0,0.40)",
      shadowMd: "0 4px 12px rgba(0,0,0,0.35)",
    },
  },
  typography: defaultTypography(),
  shape: defaultShape(),
  layout: defaultLayout(),
  components: defaultComponents(),
}

/**
 * Rose Pine — muted purple-pink palette, popular in dev tools.
 */
const rosePineTheme: ThemeDefinition = {
  id: "rose-pine",
  name: "Rose Pine",
  description: "Muted purple-pink tones for a cozy feel",
  version: 1 as const,
  builtin: true,
  palette: {
    light: {
      bg: "250 244 237",        // Dawn base
      surface: "255 250 243",   // Dawn surface
      surface2: "242 233 222",  // Dawn overlay
      elevated: "255 250 243",
      primary: "87 82 121",     // Dawn iris
      primaryStrong: "72 67 106",
      accent: "215 130 126",    // Dawn rose
      success: "40 105 131",    // Dawn pine
      warn: "234 157 52",       // Dawn gold
      danger: "180 99 122",     // Dawn love
      muted: "152 147 165",     // Dawn muted
      border: "223 218 206",    // Dawn highlight low
      borderStrong: "206 202 205",// Dawn highlight med
      text: "87 82 121",        // Dawn text
      textMuted: "107 102 132", // darkened for AA on surface/surface2
      textSubtle: "134 128 152",// darkened for >=3:1 subtle floor
      focus: "168 110 122",     // darkened for >=3:1 focus indicator contrast
      shadowSm: "0 1px 3px rgba(87,82,121,0.08)",
      shadowMd: "0 6px 18px rgba(87,82,121,0.06)",
    },
    dark: {
      bg: "25 23 36",           // Main base
      surface: "30 28 44",      // Main surface
      surface2: "38 35 58",     // Main overlay
      elevated: "44 40 64",
      primary: "196 167 231",   // Main iris
      primaryStrong: "174 140 212",
      accent: "235 188 186",    // Main rose
      success: "49 116 143",    // Main pine -> brightened
      warn: "246 193 119",      // Main gold
      danger: "235 111 146",    // Main love
      muted: "110 106 134",     // Main muted
      border: "42 39 63",       // Main highlight low
      borderStrong: "57 53 82", // Main highlight med
      text: "224 222 244",      // Main text
      textMuted: "144 140 170", // Main subtle
      textSubtle: "112 108 136",// brightened to meet subtle floor on surface2
      focus: "235 188 186",
      shadowSm: "0 1px 2px rgba(0,0,0,0.30)",
      shadowMd: "0 4px 12px rgba(0,0,0,0.25)",
    },
  },
  typography: defaultTypography(),
  shape: defaultShape(),
  layout: defaultLayout(),
  components: defaultComponents(),
}

const PRESETS: ThemeDefinition[] = [
  defaultTheme,
  primerTheme,
  solarizedTheme,
  nordTheme,
  highContrastTheme,
  rosePineTheme,
]

export function getBuiltinPresets(): ThemeDefinition[] {
  return PRESETS
}

export function getAllPresets(customThemes: ThemeDefinition[] = []): ThemeDefinition[] {
  return [...PRESETS, ...customThemes]
}

export function getThemeById(id: string, customThemes: ThemeDefinition[] = []): ThemeDefinition | undefined {
  return PRESETS.find((t) => t.id === id) ?? customThemes.find((t) => t.id === id)
}

export function getDefaultTheme(): ThemeDefinition {
  return defaultTheme
}
