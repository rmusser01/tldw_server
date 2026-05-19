import type { ThemeDefinition } from "./types"
import { defaultShape, defaultLayout, defaultComponents } from "./defaults"

/**
 * Primer — cyberpunk-leaning "Young Lady's Illustrated Primer" palette.
 * Dark mode reads as phosphor cyan on near-black; light mode is a parchment
 * ("Parchment Primer") with deep teal ink.
 * Source: tmp_dir/ui_kits/webui/colors_and_type.css.
 */
export const primerTheme: ThemeDefinition = {
  id: "primer",
  name: "Primer",
  description: "Phosphor cyan on near-black; parchment in light mode",
  version: 1 as const,
  builtin: true,
  palette: {
    light: {
      // Parchment Primer — daylight variant
      bg: "246 241 230",
      surface: "253 250 242",
      surface2: "241 235 221",
      elevated: "255 253 247",
      primary: "16 110 122",
      primaryStrong: "10 78 87",
      accent: "186 120 27",
      success: "22 140 92",
      warn: "186 120 27",
      danger: "186 43 92",
      muted: "116 126 140",
      border: "222 214 194",
      borderStrong: "190 180 158",
      text: "26 32 40",
      textMuted: "84 94 110",
      textSubtle: "116 126 140",
      focus: "16 110 122",
      shadowSm: "0 1px 2px rgb(0 0 0 / 0.08)",
      shadowMd: "0 6px 18px rgb(0 0 0 / 0.08)",
    },
    dark: {
      // Dark Primer — default cyberpunk variant
      bg: "5 7 10",
      surface: "11 15 20",
      surface2: "18 24 33",
      elevated: "26 34 48",
      primary: "92 225 230",
      primaryStrong: "48 183 190",
      accent: "245 181 68",
      success: "87 245 164",
      warn: "245 181 68",
      danger: "255 62 136",
      muted: "95 104 116",
      border: "46 56 72",
      borderStrong: "76 92 116",
      text: "232 234 230",
      textMuted: "154 160 166",
      textSubtle: "118 128 140",
      focus: "92 225 230",
      shadowSm: "0 1px 2px rgb(0 0 0 / 0.5)",
      shadowMd: "0 8px 24px rgb(0 0 0 / 0.45)",
    },
  },
  typography: {
    fontFamily: '"Inter", system-ui, sans-serif',
    fontFamilyMono: '"JetBrains Mono", "Courier New", monospace',
    fontSizeBody: 14,
    fontSizeMessage: 15,
    fontSizeCaption: 12,
    fontSizeLabel: 11,
  },
  shape: defaultShape(),
  layout: defaultLayout(),
  components: defaultComponents(),
}
