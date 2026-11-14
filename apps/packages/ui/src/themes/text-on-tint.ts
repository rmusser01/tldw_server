import { mixRgb, readableShade } from "./contrast"
import type { RGBTriple, ThemeColorTokens } from "./types"

/** Opacity of the tint a status chip puts behind its own colour (`bg-success/10`). */
export const TEXT_TINT_ALPHA = 0.1

/**
 * The CSS variable each `text-<kind>` Tailwind utility paints with. Fills,
 * borders and rings (`bg-success`, `border-primary`) keep the base colour.
 */
export const TEXT_ON_TINT_VARS = {
  primary: "--color-primary-text",
  success: "--color-success-text",
  warn: "--color-warn-text",
  danger: "--color-danger-text",
} as const

export type TextOnTintKind = keyof typeof TEXT_ON_TINT_VARS

/**
 * The text shade of `kind`: its colour moved towards the theme's text colour
 * until it reaches 4.5:1 on the page, card and panel surfaces and on its own
 * 10% tint over each of them (#3124). A colour that already does is kept.
 */
export const textOnTintShade = (
  tokens: ThemeColorTokens,
  kind: TextOnTintKind
): RGBTriple => {
  const color = tokens[kind]
  const surfaces = [tokens.bg, tokens.surface, tokens.surface2]
  const tints = surfaces.map((surface) => mixRgb(surface, color, TEXT_TINT_ALPHA))
  return readableShade(color, tokens.text, [...surfaces, ...tints])
}

/** Every text-on-tint variable and its value for one palette mode. */
export const textOnTintTokens = (
  tokens: ThemeColorTokens
): Array<[cssVar: string, value: RGBTriple]> =>
  (Object.keys(TEXT_ON_TINT_VARS) as TextOnTintKind[]).map((kind) => [
    TEXT_ON_TINT_VARS[kind],
    textOnTintShade(tokens, kind),
  ])
