import { describe, expect, it } from "vitest"
import { primerTheme } from "../primer-preset"
import { contrastRatio } from "../contrast"

// The app-wide auditThemeTextContrast does not currently check the `elevated`
// surface (popovers, modals) — and a pre-existing Rose Pine finding prevents
// widening it without scope creep. Guard Primer explicitly instead.
describe("primer theme — elevated surface contrast", () => {
  for (const mode of ["dark", "light"] as const) {
    const tokens = primerTheme.palette[mode]

    it(`${mode}: text on elevated meets AA (>= 4.5)`, () => {
      expect(contrastRatio(tokens.text, tokens.elevated)).toBeGreaterThanOrEqual(4.5)
    })

    it(`${mode}: textMuted on elevated meets AA (>= 4.5)`, () => {
      expect(contrastRatio(tokens.textMuted, tokens.elevated)).toBeGreaterThanOrEqual(4.5)
    })

    it(`${mode}: textSubtle on elevated meets 3:1 non-text floor`, () => {
      expect(contrastRatio(tokens.textSubtle, tokens.elevated)).toBeGreaterThanOrEqual(3.0)
    })
  }
})
