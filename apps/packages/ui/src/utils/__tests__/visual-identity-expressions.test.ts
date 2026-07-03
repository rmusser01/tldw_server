import { describe, expect, it } from "vitest"
import {
  normalizeVisualIdentityExpressionKey,
  VISUAL_IDENTITY_EXPRESSION_OPTIONS
} from "../visual-identity-expressions"

describe("visual identity expression utilities", () => {
  it("uses the character mood baseline as canonical expression slots", () => {
    expect(VISUAL_IDENTITY_EXPRESSION_OPTIONS.map((option) => option.key)).toEqual([
      "neutral",
      "happy",
      "excited",
      "sad",
      "angry",
      "thinking",
      "confused",
      "surprised"
    ])
  })

  it("normalizes known expression aliases", () => {
    expect(normalizeVisualIdentityExpressionKey("JOYFUL")).toBe("happy")
    expect(normalizeVisualIdentityExpressionKey("anger")).toBe("angry")
  })

  it("normalizes custom expression labels", () => {
    expect(normalizeVisualIdentityExpressionKey("bashful smile")).toBe(
      "custom:bashful_smile"
    )
  })

  it("preserves already-normalized custom expressions", () => {
    expect(normalizeVisualIdentityExpressionKey("custom:side eye")).toBe(
      "custom:side_eye"
    )
  })
})
