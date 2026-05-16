import { describe, expect, it } from "vitest"
import { buildAnswerTrustSummary, formatSourceList } from "../trustSummary"

describe("buildAnswerTrustSummary", () => {
  it("formats selected source names with readable joining", () => {
    expect(formatSourceList(["media_db", "notes", "prompts"])).toBe(
      "Documents & Media, Notes, and Prompts"
    )
  })

  it("summarizes sources, citations, web fallback, and caveats", () => {
    expect(
      buildAnswerTrustSummary({
        selectedSources: ["media_db", "notes"],
        resultCount: 12,
        citationCount: 5,
        webFallbackEnabled: true,
        webFallbackTriggered: false,
        generationProvider: null,
        generationModel: null,
        sourceHealthCaveatCount: 2,
        trustLabel: "Partial",
      })
    ).toEqual([
      "Searched Documents & Media and Notes. 12 sources returned, 5 cited.",
      "Web fallback enabled, not used.",
      "AI model: Server default.",
      "2 selected sources need attention.",
      "Trust: Partial.",
    ])
  })
})
