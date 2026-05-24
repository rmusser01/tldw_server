import { describe, expect, it } from "vitest"
import { getBuiltinPresets } from "@/themes/presets"
import { contrastRatio } from "@/themes/contrast"

describe("ResearchWorkspace stage 11 contrast gates", () => {
  const presets = getBuiltinPresets()

  it("keeps source icon token pairing at WCAG AA (text-muted on surface-2)", () => {
    const failures: string[] = []

    for (const preset of presets) {
      for (const [mode, tokens] of Object.entries(preset.palette)) {
        const ratio = contrastRatio(tokens.textMuted, tokens.surface2)
        if (ratio < 4.5) {
          failures.push(`${preset.id}/${mode}: textMuted/surface2=${ratio.toFixed(2)}`)
        }
      }
    }

    expect(failures, failures.join(" | ")).toEqual([])
  })

  it("keeps mobile tab badge token pairing at WCAG AA (text on surface-2)", () => {
    const failures: string[] = []

    for (const preset of presets) {
      for (const [mode, tokens] of Object.entries(preset.palette)) {
        const ratio = contrastRatio(tokens.text, tokens.surface2)
        if (ratio < 4.5) {
          failures.push(`${preset.id}/${mode}: text/surface2=${ratio.toFixed(2)}`)
        }
      }
    }

    expect(failures, failures.join(" | ")).toEqual([])
  })
})
