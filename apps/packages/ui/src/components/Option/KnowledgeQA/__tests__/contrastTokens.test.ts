import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

type RgbTuple = [number, number, number]

const parseSection = (css: string, selector: ":root" | ".dark"): Record<string, RgbTuple> => {
  const sectionRegex = new RegExp(`${selector}\\s*\\{([\\s\\S]*?)\\}`, "m")
  const match = css.match(sectionRegex)
  if (!match) {
    throw new Error(`Missing ${selector} token section`)
  }
  const tokens: Record<string, RgbTuple> = {}
  for (const entry of match[1].split("\n")) {
    const tokenMatch = entry
      .trim()
      .match(/^--(color-[a-z0-9-]+):\s*([0-9]{1,3})\s+([0-9]{1,3})\s+([0-9]{1,3});/)
    if (!tokenMatch) continue
    tokens[tokenMatch[1]] = [
      Number.parseInt(tokenMatch[2], 10),
      Number.parseInt(tokenMatch[3], 10),
      Number.parseInt(tokenMatch[4], 10),
    ]
  }
  return tokens
}

const toLinear = (channel: number): number => {
  const normalized = channel / 255
  if (normalized <= 0.03928) return normalized / 12.92
  return ((normalized + 0.055) / 1.055) ** 2.4
}

const contrastRatio = (foreground: RgbTuple, background: RgbTuple): number => {
  const luminance = (rgb: RgbTuple) =>
    0.2126 * toLinear(rgb[0]) +
    0.7152 * toLinear(rgb[1]) +
    0.0722 * toLinear(rgb[2])
  const fg = luminance(foreground)
  const bg = luminance(background)
  const lighter = Math.max(fg, bg)
  const darker = Math.min(fg, bg)
  return (lighter + 0.05) / (darker + 0.05)
}

describe("Knowledge QA contrast token audit", () => {
  const css = readFileSync(
    path.resolve(
      path.dirname(fileURLToPath(import.meta.url)),
      "../../../../assets/tailwind-shared.css"
    ),
    "utf8"
  )
  const lightTokens = parseSection(css, ":root")
  const darkTokens = parseSection(css, ".dark")

  it("keeps citation badge contrast WCAG AA-compliant in light and dark themes", () => {
    const white: RgbTuple = [255, 255, 255]
    const slate900: RgbTuple = [17, 24, 39]
    expect(contrastRatio(white, lightTokens["color-primary"])).toBeGreaterThanOrEqual(4.5)
    expect(contrastRatio(slate900, darkTokens["color-primary"])).toBeGreaterThanOrEqual(4.5)
  })

  it("keeps muted helper text readable against surface-2 backgrounds", () => {
    expect(
      contrastRatio(lightTokens["color-text-muted"], lightTokens["color-surface-2"])
    ).toBeGreaterThanOrEqual(4.5)
    expect(
      contrastRatio(darkTokens["color-text-muted"], darkTokens["color-surface-2"])
    ).toBeGreaterThanOrEqual(4.5)
  })
})
