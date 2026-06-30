import { describe, expect, it } from "vitest"
import { buildAnswerTrustSummary, formatSourceList } from "../trustSummary"
import type { KnowledgeAnswerTrustState } from "../types"

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
        trustState: "uncited_degraded_answer",
      })
    ).toEqual([
      "Searched Documents & Media and Notes. 12 sources returned, 5 cited.",
      "Web fallback enabled, not used.",
      "AI model: Server default.",
      "2 selected sources need attention.",
      "Trust: Uncited answer.",
    ])
  })

  it.each<[KnowledgeAnswerTrustState, string]>([
    ["cited_answer", "Cited answer"],
    ["uncited_degraded_answer", "Uncited answer"],
    ["no_answer_insufficient_evidence", "Insufficient evidence"],
    ["no_results", "No results"],
    ["failed_search", "Failed search"],
    ["unsynced_local_result", "Unsynced local result"],
    ["unknown_trust", "Trust unknown"],
  ])("summarizes %s as %s", (trustState, expectedLabel) => {
    expect(
      buildAnswerTrustSummary({
        selectedSources: ["media_db"],
        resultCount: 1,
        citationCount: 0,
        webFallbackEnabled: false,
        webFallbackTriggered: false,
        generationProvider: null,
        generationModel: null,
        sourceHealthCaveatCount: 0,
        trustState,
      })
    ).toContain(`Trust: ${expectedLabel}.`)
  })
})
