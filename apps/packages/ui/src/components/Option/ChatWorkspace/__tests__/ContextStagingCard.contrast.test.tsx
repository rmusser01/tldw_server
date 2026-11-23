import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { contrastRatio } from "@/themes/contrast"
import { getBuiltinPresets } from "@/themes/presets"
import type { ThemeDefinition } from "@/themes/types"
import { useDarkModeStore } from "@/hooks/useDarkmode"
import { ContextStagingCard, type ContextStagingCardProps } from "../ContextStagingCard"

const settings = vi.hoisted(() => new Map<string, unknown>())
vi.mock("@/hooks/useSetting", () => ({
  useSetting: (setting: { key: string; defaultValue: unknown }) => [
    settings.get(setting.key) ?? setting.defaultValue
  ]
}))

const props: ContextStagingCardProps = {
  sources: [{ sourceId: "s1", mediaId: 1, title: "Notes", type: "document", scopeLabel: "Workspace", availability: "ready" as const }],
  onClear: vi.fn(), onInsert: vi.fn(), onSend: vi.fn()
}

const card = () => <><style>{".text-white { color: rgb(255, 255, 255); }"}</style><ContextStagingCard {...props} /></>

const assertContrast = (background: string) => {
  const button = screen.getByRole("button", { name: "Send with staged context" })
  const foreground = getComputedStyle(button).color.match(/\d+/g)?.slice(0, 3).join(" ") || "255 255 255"
  expect(button).toBeEnabled()
  expect(contrastRatio(foreground, background)).toBeGreaterThanOrEqual(4.5)
}

describe("Context staging production theme contrast", () => {
  beforeEach(() => settings.clear())

  for (const preset of getBuiltinPresets()) {
    it.each(["light", "dark"] as const)(`meets AA on ${preset.id} %s primaryStrong`, mode => {
      settings.set("tldw:themePreset", preset.id)
      useDarkModeStore.setState({ mode })
      render(card())
      assertContrast(preset.palette[mode].primaryStrong)
    })
  }

  it("updates contrast on preset changes and honors custom palettes", () => {
    useDarkModeStore.setState({ mode: "dark" })
    const view = render(card())
    const custom: ThemeDefinition = {
      ...getBuiltinPresets()[0], id: "custom-probe", builtin: false,
      palette: {
        ...getBuiltinPresets()[0].palette,
        dark: { ...getBuiltinPresets()[0].palette.dark, primaryStrong: "230 230 230" }
      }
    }
    settings.set("tldw:themePreset", custom.id)
    settings.set("tldw:customThemes", [custom])
    view.rerender(card())
    assertContrast("230 230 230")
  })
})
