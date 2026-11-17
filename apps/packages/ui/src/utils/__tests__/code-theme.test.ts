import { afterEach, describe, expect, it } from "vitest"
import { contrastRatio } from "@/themes/contrast"
import { getDefaultTheme } from "@/themes/presets"
import type { RGBTriple } from "@/themes/types"
import { normalizeLanguage, resolveTheme } from "../code-theme"

describe("normalizeLanguage", () => {
  it("normalizes text aliases to plaintext", () => {
    expect(normalizeLanguage("text")).toBe("plaintext")
    expect(normalizeLanguage("txt")).toBe("plaintext")
    expect(normalizeLanguage("plain")).toBe("plaintext")
  })

  it("normalizes python shorthand", () => {
    expect(normalizeLanguage("py")).toBe("python")
  })
})

const toTriple = (color: string): RGBTriple => {
  const hex = color.trim().match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i)
  if (hex) {
    const digits =
      hex[1].length === 3 ? hex[1].split("").map((d) => d + d).join("") : hex[1]
    return [0, 2, 4].map((i) => parseInt(digits.slice(i, i + 2), 16)).join(" ")
  }
  const rgb = color.match(/^rgba?\(([^)]+)\)$/i)
  if (!rgb) throw new Error(`Unparsed colour ${color}`)
  return rgb[1].split(",").slice(0, 3).map((part) => part.trim()).join(" ")
}

// AX-07 (#3124): the "auto" code themes (GitHub in light, Dracula in dark)
// had token colours down to 2.43:1. Code renders on the theme background in
// CodeBlock and on the page's own surfaces in Markdown, so every token colour
// must reach 4.5:1 on both.
describe("auto code theme contrast", () => {
  afterEach(() => {
    document.documentElement.classList.remove("dark", "light")
  })

  it.each(["light", "dark"] as const)("keeps every %s token colour at 4.5:1", (mode) => {
    document.documentElement.classList.remove("dark", "light")
    document.documentElement.classList.add(mode)
    const theme = resolveTheme("auto")
    const palette = getDefaultTheme().palette[mode]
    const backgrounds: Array<[string, RGBTriple]> = [
      ["theme", toTriple(String(theme.plain.backgroundColor))],
      ["bg", palette.bg],
      ["surface", palette.surface],
      ["surface2", palette.surface2],
      ["elevated", palette.elevated]
    ]
    const colours = [
      ["plain", theme.plain.color],
      ...theme.styles.map((entry) => [entry.types.join("/"), entry.style.color] as const)
    ].filter((entry): entry is [string, string] => typeof entry[1] === "string")

    const failures = colours.flatMap(([name, colour]) =>
      backgrounds
        .map(([surface, background]) => [surface, contrastRatio(toTriple(colour), background)] as const)
        .filter(([, ratio]) => ratio < 4.5)
        .map(([surface, ratio]) => `${name} on ${surface}: ${ratio.toFixed(2)}`)
    )
    expect(failures).toEqual([])
  })

  it("follows the page theme", () => {
    document.documentElement.classList.add("light")
    const light = resolveTheme("auto")
    document.documentElement.classList.replace("light", "dark")
    const dark = resolveTheme("auto")
    expect(light).not.toBe(dark)
  })
})
