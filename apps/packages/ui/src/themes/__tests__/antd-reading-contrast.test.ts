import { createRequire } from "node:module"
import { theme } from "antd"
import { describe, expect, it } from "vitest"
import { buildAntdThemeConfig } from "../antd-theme"
import { contrastRatio } from "../contrast"
import { getBuiltinPresets } from "../presets"

const require = createRequire(import.meta.url)
const { FastColor } = createRequire(require.resolve("antd/package.json"))("@ant-design/fast-color")

const ratio = (foreground: string, background: string) => {
  const fg = new FastColor(foreground).onBackground(background).toRgb()
  const bg = new FastColor(background).toRgb()
  return contrastRatio([fg.r, fg.g, fg.b].join(" "), [bg.r, bg.g, bg.b].join(" "))
}

const palettes = getBuiltinPresets().flatMap((preset) =>
  Object.entries(preset.palette).map(([mode, tokens]) => ({ id: preset.id, mode, tokens }))
)

describe.each(palettes)("Ant Design reading contrast: $id/$mode", ({ mode, tokens }) => {
  it("keeps the resolved placeholder readable on input and popup surfaces", () => {
    const resolved = theme.getDesignToken(buildAntdThemeConfig(tokens, mode === "dark"))
    for (const background of [resolved.colorBgContainer, resolved.colorBgElevated]) {
      expect(ratio(resolved.colorTextPlaceholder, background)).toBeGreaterThanOrEqual(4.5)
    }
  })

  it("keeps link button text readable in normal, hover and pressed states", () => {
    const button = buildAntdThemeConfig(tokens, mode === "dark").components?.Button
    for (const foreground of [button?.colorLink, button?.colorLinkHover, button?.colorLinkActive]) {
      expect(typeof foreground).toBe("string")
      for (const background of [tokens.bg, tokens.surface, tokens.surface2, tokens.elevated]) {
        expect(
          ratio(String(foreground), "rgb(" + background.split(" ").join(",") + ")")
        ).toBeGreaterThanOrEqual(4.5)
      }
    }
  })
})
