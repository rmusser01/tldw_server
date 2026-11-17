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

  // Solarized and Nord links measured 3.30-4.44:1 when neither primary shade
  // reached 4.5:1; hover and pressed links (#7099e0, 2.87:1) were derived.
  it.each(cases)("keeps links readable on page and panel backgrounds in $name", ({ preset, mode }) => {
    const { tokens, design } = build(preset, mode)
    const failures = surfaces(tokens).flatMap(([name, background]) =>
      (["colorLink", "colorLinkHover", "colorLinkActive"] as const)
        .map((token) => [token, ratio(design[token], background)] as const)
        .filter(([, value]) => value < 4.5)
        .map(([token, value]) => `${token}/${name}=${value.toFixed(2)}`)
    )
    expect(failures).toEqual([])
  })

  it.each(cases)("draws links in a primary shade in $name", ({ preset, mode }) => {
    const { tokens, design } = build(preset, mode)
    const link = design.colorLink.toLowerCase()
    const primaryShades = [rgbTripleToHex(tokens.primary), rgbTripleToHex(tokens.primaryStrong)]
    const readablePrimary = [tokens.primary, tokens.primaryStrong].some((shade) =>
      surfaces(tokens).every(([, background]) => contrastRatio(shade, background) >= 4.5)
    )
    if (readablePrimary) {
      expect(primaryShades).toContain(link)
    } else {
      // Only darkened or lightened towards the text colour as far as needed.
      expect(link).not.toBe(rgbTripleToHex(tokens.text))
    }
  })

  // The light theme's pressed primary button (#1e51c7) put its black label at
  // 3.06:1, and pressed danger (#ba3f56) at 3.93:1.
  it.each(cases)("keeps button text readable on hover and pressed fills in $name", ({ preset, mode }) => {
    const { config, design } = build(preset, mode)
    const button = config.components?.Button
    const onPrimary = toTriple(String(button?.primaryColor), "0 0 0")
    const onDanger = toTriple(String(button?.dangerColor), "0 0 0")
    const failures = [
      ...(["colorPrimaryHover", "colorPrimaryActive"] as const).map(
        (token) => [token, contrastRatio(onPrimary, toTriple(design[token], "0 0 0"))] as const
      ),
      ...(["colorErrorHover", "colorErrorActive"] as const).map(
        (token) => [token, contrastRatio(onDanger, toTriple(design[token], "0 0 0"))] as const
      )
    ]
      .filter(([, value]) => value < 4.5)
      .map(([token, value]) => `${token}=${value.toFixed(2)}`)
    expect(failures).toEqual([])
  })
})

// AX-07 (#3124): pinning the theme primary and danger colours left other antd
// components drawing white (colorTextLightSolid) on those fills: in the
// default dark theme the current Steps number measured 3.13:1 and Badge
// counts 2.72:1. Every component that puts text on a solid primary or
// danger fill needs a readable foreground too.
describe("antd theme text on solid fills in other components", () => {
  type Fill = "primary" | "danger"
  const FILL_TEXT: Array<{ component: string; token: string; fills: Fill[] }> = [
    { component: "Steps", token: "colorTextLightSolid", fills: ["primary", "danger"] },
    { component: "Badge", token: "colorTextLightSolid", fills: ["danger", "primary"] },
    { component: "Tour", token: "colorTextLightSolid", fills: ["primary"] },
    { component: "Dropdown", token: "colorTextLightSolid", fills: ["danger"] },
    { component: "Tag", token: "colorTextLightSolid", fills: ["primary"] },
    { component: "Menu", token: "darkItemSelectedColor", fills: ["primary"] },
    { component: "Menu", token: "darkDangerItemSelectedColor", fills: ["danger"] },
    { component: "Tree", token: "directoryNodeSelectedColor", fills: ["primary"] }
  ]

  it.each(cases)("keeps text on primary and danger fills readable in $name", ({ preset, mode }) => {
    const { config, design } = build(preset, mode)
    const fills: Record<Fill, RGBTriple> = {
      primary: toTriple(design.colorPrimary, "0 0 0"),
      danger: toTriple(design.colorError, "0 0 0")
    }
    const failures = FILL_TEXT.flatMap(({ component, token, fills: names }) => {
      const override = (config.components as Record<string, Record<string, unknown>> | undefined)?.[
        component
      ]?.[token]
      // Unset, each of these tokens falls back to the global colorTextLightSolid.
      const value = typeof override === "string" ? override : design.colorTextLightSolid
      const text = toTriple(value, "0 0 0")
      // One text colour on two fills cannot always reach 4.5:1 on both (for
      // example rose-pine light primary and danger), so the bar drops to the
      // best worst case black or white can manage there.
      const bestWorstCase = Math.max(
        ...(["0 0 0", "255 255 255"] as const).map((candidate) =>
          Math.min(...names.map((fill) => contrastRatio(candidate, fills[fill])))
        )
      )
      const bar = Math.min(4.5, bestWorstCase) - 0.005
      return names
        .map((fill) => [fill, contrastRatio(text, fills[fill])] as const)
        .filter(([, value]) => value < bar)
        .map(([fill, value]) => `${component}.${token} on ${fill}=${value.toFixed(2)}`)
    })
    expect(failures).toEqual([])
  })

  it.each(cases)("keeps the fallback avatar icon readable in $name", ({ preset, mode }) => {
    // antd paints the fallback avatar colorTextPlaceholder with a
    // colorTextLightSolid icon; the global placeholder is now a text colour.
    const { tokens, config, design } = build(preset, mode)
    const avatar = config.components?.Avatar as
      | { colorTextPlaceholder?: string; colorTextLightSolid?: string }
      | undefined
    const fill = toTriple(avatar?.colorTextPlaceholder ?? design.colorTextPlaceholder, tokens.surface)
    const icon = toTriple(avatar?.colorTextLightSolid ?? design.colorTextLightSolid, "0 0 0")
    expect(contrastRatio(icon, fill)).toBeGreaterThanOrEqual(4.5)
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

  // AX-07: light danger text (#e0586d, used by Delete menu items and
  // text-danger) measured 3.25-3.48:1, and warning text (#d97706) 2.73-3.19:1.
  it.each(defaultCases)("keeps danger and warning text at 4.5:1 on every surface in $name", ({ preset, mode }) => {
    const tokens = preset.palette[mode]
    const failures = surfaces(tokens).flatMap(([name, background]) =>
      (["danger", "warn"] as const)
        .map((token) => [token, contrastRatio(tokens[token], background)] as const)
        .filter(([, value]) => value < 4.5)
        .map(([token, value]) => `${token}/${name}=${value.toFixed(2)}`)
    )
    expect(failures).toEqual([])
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
    expect(values.get("--color-danger")).toBe(tokens.danger)
    expect(values.get("--color-warn")).toBe(tokens.warn)
  })
})
