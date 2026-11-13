import { readFileSync } from "node:fs"
import { createRequire } from "node:module"
import postcss from "postcss"
import tailwindcss from "tailwindcss"
import { afterEach, describe, expect, it } from "vitest"
import { applyThemeTokens, clearThemeTokens } from "../apply-theme"
import { contrastRatio, mixRgb } from "../contrast"
import { getBuiltinPresets, getDefaultTheme } from "../presets"
import { textOnTintTokens } from "../text-on-tint"

const require = createRequire(import.meta.url)

// #3124 follow-up: status chips put coloured text on a 10% tint of the same
// colour (`text-success bg-success/10`), and primary does the same. In the
// light theme those pairs were 4.24-4.41:1 (status) and 3.60:1 (primary),
// below WCAG AA's 4.5:1, and nothing scanned the light theme to catch it.
const KINDS = ["primary", "success", "warn", "danger"] as const
const TINT = 0.1
/** The page, card and panel surfaces tinted chips sit on. */
const SURFACES = ["--color-bg", "--color-surface", "--color-surface-2"] as const

type Tokens = Map<string, string>

/** The stylesheet's default palette: `:root` (light) and `.dark` over it. */
const stylesheetThemes = (): Record<"light" | "dark", Tokens> => {
  const css = postcss.parse(
    readFileSync(require.resolve("../../assets/tailwind-shared.css"), "utf8")
  )
  const read = (selector: string) => {
    const tokens: Tokens = new Map()
    css.walkRules(selector, (rule) => {
      rule.walkDecls((decl) => {
        if (decl.prop.startsWith("--")) tokens.set(decl.prop, decl.value.trim())
      })
    })
    return tokens
  }
  const light = read(":root")
  return { light, dark: new Map([...light, ...read(".dark")]) }
}

/** The variable each `text-<kind>` and `bg-<kind>/10` utility paints with. */
const utilityVariables = async (client: string) => {
  const config = require(`../../../../../${client}/tailwind.config.js`)
  const classes = KINDS.flatMap((kind) => [`text-${kind}`, `bg-${kind}/10`])
  const result = await postcss([
    tailwindcss({ ...config, content: [{ raw: classes.join(" ") }] })
  ]).process("@tailwind utilities;", { from: undefined })
  const variables = new Map<string, string>()
  result.root.walkRules((rule) => {
    rule.walkDecls(/^(color|background-color)$/, (decl) => {
      const variable = decl.value.match(/var\((--[a-z0-9-]+)/)?.[1]
      if (variable) variables.set(rule.selector.replace(/\\/g, ""), variable)
    })
  })
  return Object.fromEntries(
    KINDS.map((kind) => [
      kind,
      { text: variables.get(`.text-${kind}`), fill: variables.get(`.bg-${kind}/10`) }
    ])
  )
}

const tintFailures = (
  label: string,
  textColor: (kind: (typeof KINDS)[number]) => string | undefined,
  fillColor: (kind: (typeof KINDS)[number]) => string | undefined,
  surfaces: string[]
) =>
  KINDS.flatMap((kind) => {
    const text = textColor(kind)
    const fill = fillColor(kind)
    if (!text || !fill) return [`${label} ${kind}: no colour (text ${text}, fill ${fill})`]
    return surfaces.flatMap((surface) => {
      const tint = mixRgb(surface, fill, TINT)
      const ratio = contrastRatio(text, tint)
      return ratio >= 4.5 ? [] : [`${label} ${kind} on ${kind}/10 over ${surface}: ${ratio.toFixed(2)}`]
    })
  })

afterEach(clearThemeTokens)

describe("text on its own /10 tint", () => {
  it.each(["tldw-frontend", "extension"])(
    "reaches 4.5:1 in both default themes (%s)",
    async (client) => {
      const variables = await utilityVariables(client)
      const failures = Object.entries(stylesheetThemes()).flatMap(([mode, tokens]) =>
        tintFailures(
          mode,
          (kind) => tokens.get(variables[kind].text ?? ""),
          (kind) => tokens.get(variables[kind].fill ?? ""),
          SURFACES.map((surface) => tokens.get(surface)!)
        )
      )
      expect(failures).toEqual([])
    },
    30_000
  )

  it("reaches 4.5:1 for every built-in theme applied at runtime", async () => {
    const variables = await utilityVariables("tldw-frontend")
    const failures: string[] = []
    for (const theme of getBuiltinPresets()) {
      for (const [mode, tokens] of Object.entries(theme.palette)) {
        applyThemeTokens(tokens, theme)
        const style = document.documentElement.style
        const value = (variable: string | undefined) =>
          variable ? style.getPropertyValue(variable).trim() || undefined : undefined
        failures.push(
          ...tintFailures(
            `${theme.id}/${mode}`,
            (kind) => value(variables[kind].text),
            (kind) => value(variables[kind].fill),
            [tokens.bg, tokens.surface, tokens.surface2]
          )
        )
      }
    }
    expect(failures).toEqual([])
  }, 30_000)

  it("keeps the stylesheet's default shades equal to the derived ones", () => {
    const stylesheet = stylesheetThemes()
    const { palette } = getDefaultTheme()
    for (const mode of ["light", "dark"] as const) {
      const derived = textOnTintTokens(palette[mode])
      expect(
        Object.fromEntries(derived.map(([variable]) => [variable, stylesheet[mode].get(variable)])),
        mode
      ).toEqual(Object.fromEntries(derived))
    }
  })

  it("removes the runtime text colours with the rest of a theme", async () => {
    const variables = await utilityVariables("tldw-frontend")
    const [theme] = getBuiltinPresets()
    applyThemeTokens(theme.palette.light, theme)
    clearThemeTokens()
    for (const kind of KINDS) {
      expect(document.documentElement.style.getPropertyValue(variables[kind].text!)).toBe("")
    }
  }, 30_000)
})
