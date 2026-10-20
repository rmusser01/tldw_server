import { readFileSync } from "node:fs"
import { createRequire } from "node:module"
import postcss from "postcss"
import { theme } from "antd"
import { describe, expect, it } from "vitest"
import { buildAntdThemeConfig, rgbTripleToHex } from "../antd-theme"
import { contrastRatio } from "../contrast"
import { getBuiltinPresets, getDefaultTheme } from "../presets"
import type { RGBTriple, ThemeColorTokens, ThemeDefinition } from "../types"

const require = createRequire(import.meta.url)

type Mode = "light" | "dark"

const MODES: Mode[] = ["light", "dark"]

const cases = getBuiltinPresets().flatMap((preset) =>
  MODES.map((mode) => ({ name: `${preset.id}/${mode}`, preset, mode }))
)

const defaultCases = MODES.map((mode) => ({
  name: `default/${mode}`,
  preset: getDefaultTheme(),
  mode
}))

function build(preset: ThemeDefinition, mode: Mode) {
  const tokens = preset.palette[mode]
  const config = buildAntdThemeConfig(
    tokens,
    mode === "dark",
    preset.typography,
    preset.shape,
    preset.layout
  )
  return { tokens, config, design: theme.getDesignToken(config) }
}

/** Parse an antd colour ("#fff", "#rrggbb", "rgba(r,g,b,a)") composited over `background`. */
function toTriple(color: string, background: RGBTriple): RGBTriple {
  const value = color.trim()
  const hex = value.match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i)
  if (hex) {
    const digits =
      hex[1].length === 3
        ? hex[1].split("").map((d) => d + d).join("")
        : hex[1]
    return [0, 2, 4].map((i) => parseInt(digits.slice(i, i + 2), 16)).join(" ")
  }
  const rgba = value.match(/^rgba?\(([^)]+)\)$/i)
  if (!rgba) throw new Error(`Unparsed colour ${color}`)
  const [r, g, b, a = "1"] = rgba[1].split(",").map((part) => part.trim())
  const alpha = Number(a)
  const bg = background.split(/\s+/).map(Number)
  return [r, g, b]
    .map((channel, index) =>
      Math.round(Number(channel) * alpha + bg[index] * (1 - alpha))
    )
    .join(" ")
}

const ratio = (color: string, background: RGBTriple) =>
  contrastRatio(toTriple(color, background), background)

const surfaces = (tokens: ThemeColorTokens): Array<[string, RGBTriple]> => [
  ["bg", tokens.bg],
  ["surface", tokens.surface],
  ["surface2", tokens.surface2],
  ["elevated", tokens.elevated]
]

// XP-20 (#3124): antd's dark algorithm turned --color-primary #5c8dff into
// #517bdc, so antd and Tailwind primary controls were two different blues.
describe("antd theme uses the app's own colours", () => {
  it.each(cases)("keeps the primary and danger colours of $name", ({ preset, mode }) => {
    const { tokens, design } = build(preset, mode)
    expect(design.colorPrimary.toLowerCase()).toBe(rgbTripleToHex(tokens.primary))
    expect(design.colorError.toLowerCase()).toBe(rgbTripleToHex(tokens.danger))
  })
})

// AX-07 (#3124): white text on antd primary buttons measured 4.02:1 in dark
// mode, placeholders 1.64-2.09:1, and links 3.89:1 on light surfaces.
describe("antd theme text contrast", () => {
  it.each(cases)("puts readable text on solid primary and danger fills in $name", ({ preset, mode }) => {
    const { config, design } = build(preset, mode)
    const button = config.components?.Button
    const radio = config.components?.Radio
    const datePicker = config.components?.DatePicker
    const onPrimary = toTriple(String(button?.primaryColor), "0 0 0")
    const primary = toTriple(design.colorPrimary, "0 0 0")
    const danger = toTriple(design.colorError, "0 0 0")

    expect(contrastRatio(onPrimary, primary)).toBeGreaterThanOrEqual(4.5)
    expect(
      contrastRatio(toTriple(String(button?.dangerColor), "0 0 0"), danger)
    ).toBeGreaterThanOrEqual(4.5)
    expect(radio?.buttonSolidCheckedColor).toBe(button?.primaryColor)
    expect(datePicker?.colorTextLightSolid).toBe(button?.primaryColor)
  })

  it.each(cases)("keeps placeholders readable on input backgrounds in $name", ({ preset, mode }) => {
    const { tokens, design } = build(preset, mode)
    // Inputs and selects are painted --color-surface-2 by tailwind-shared.css.
    expect(ratio(design.colorTextPlaceholder, tokens.surface2)).toBeGreaterThanOrEqual(4.5)
  })

  it.each(cases)("keeps description and tertiary text readable on every surface in $name", ({ preset, mode }) => {
    const { tokens, design } = build(preset, mode)
    const failures = surfaces(tokens).flatMap(([name, background]) =>
      (["colorTextDescription", "colorTextTertiary"] as const)
        .map((token) => [token, ratio(design[token], background)] as const)
        .filter(([, value]) => value < 4.5)
        .map(([token, value]) => `${token}/${name}=${value.toFixed(2)}`)
    )
    expect(failures).toEqual([])
  })

  it.each(defaultCases)("keeps links readable on page and panel backgrounds in $name", ({ preset, mode }) => {
    const { tokens, design } = build(preset, mode)
    for (const [name, background] of surfaces(tokens)) {
      expect(ratio(design.colorLink, background), name).toBeGreaterThanOrEqual(4.5)
    }
  })

  it.each(cases)("draws links in a primary shade in $name", ({ preset, mode }) => {
    const { tokens, design } = build(preset, mode)
    expect([rgbTripleToHex(tokens.primary), rgbTripleToHex(tokens.primaryStrong)]).toContain(
      design.colorLink.toLowerCase()
    )
  })
})

// AX-08 (#3124): antd drew keyboard focus as a 3px outline in a pale primary
// tint (1.35-1.57:1), not in the theme focus colour.
describe("antd theme focus outlines", () => {
  const FOCUS_COMPONENTS = [
    "Button",
    "Checkbox",
    "Dropdown",
    "Menu",
    "Modal",
    "Pagination",
    "Radio",
    "Segmented",
    "Switch",
    "Tabs"
  ] as const

  it.each(cases)("outlines focused controls in the theme focus colour in $name", ({ preset, mode }) => {
    const { tokens, config, design } = build(preset, mode)
    expect(design.lineWidthFocus).toBe(2)
    for (const name of FOCUS_COMPONENTS) {
      const component = config.components?.[name] as { colorPrimaryBorder?: string } | undefined
      expect(component?.colorPrimaryBorder, name).toBe(rgbTripleToHex(tokens.focus))
    }
  })

  it("leaves the slider track out of the focus colour", () => {
    const { config } = build(getDefaultTheme(), "light")
    expect(
      (config.components?.Slider as { colorPrimaryBorder?: string } | undefined)?.colorPrimaryBorder
    ).toBeUndefined()
  })

  it.each(defaultCases)("keeps the default focus colour at 3:1 on every surface in $name", ({ preset, mode }) => {
    const { tokens } = build(preset, mode)
    for (const [name, background] of surfaces(tokens)) {
      expect(contrastRatio(tokens.focus, background), name).toBeGreaterThanOrEqual(3)
    }
  })
})

// AX-07 (#3124): default light --color-text-subtle measured 4.43:1 and the
// success colour, used for status text, 2.94-3.29:1.
describe("default palette text tokens", () => {
  const stylesheetTokens = (selector: string) => {
    const css = postcss.parse(
      readFileSync(require.resolve("../../assets/tailwind-shared.css"), "utf8")
    )
    const values = new Map<string, string>()
    css.walkRules(selector, (rule) => {
      rule.walkDecls((decl) => {
        values.set(decl.prop, decl.value)
      })
    })
    return values
  }

  it.each(defaultCases)("keeps subtle and success text at 4.5:1 in $name", ({ preset, mode }) => {
    const tokens = preset.palette[mode]
    for (const [name, background] of surfaces(tokens)) {
      expect(contrastRatio(tokens.textSubtle, background), `textSubtle/${name}`).toBeGreaterThanOrEqual(4.5)
    }
    for (const [name, background] of [
      ["bg", tokens.bg],
      ["surface", tokens.surface],
      ["elevated", tokens.elevated]
    ] as const) {
      expect(contrastRatio(tokens.success, background), `success/${name}`).toBeGreaterThanOrEqual(4.5)
    }
  })

  it.each([
    [":root", "light"],
    [".dark", "dark"]
  ] as const)("matches the stylesheet defaults in %s", (selector, mode) => {
    const values = stylesheetTokens(selector)
    const tokens = getDefaultTheme().palette[mode]
    expect(values.get("--color-text-subtle")).toBe(tokens.textSubtle)
    expect(values.get("--color-success")).toBe(tokens.success)
    expect(values.get("--color-focus")).toBe(tokens.focus)
  })
})
