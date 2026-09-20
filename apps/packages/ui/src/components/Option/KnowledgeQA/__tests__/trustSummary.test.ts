import { beforeEach, describe, expect, it } from "vitest"
import { createInstance } from "i18next"
import knowledgeEn from "@/assets/locale/en/knowledge.json"
import ICUWithInterpolation from "@/i18n/icu-format"
import { buildAnswerTrustSummary, formatSourceList } from "../trustSummary"
import type { KnowledgeAnswerTrustState } from "../types"

const testI18n = createInstance().use(ICUWithInterpolation)
const t = testI18n.getFixedT(null, "knowledge")

describe("buildAnswerTrustSummary", () => {
  beforeEach(async () => {
    await testI18n.init({
      lng: "en", fallbackLng: "en", defaultNS: "knowledge",
      resources: { en: { knowledge: knowledgeEn } },
      interpolation: { escapeValue: false },
    })
  })

  it.each([
    [0, "Aucune recherche. 0 résultats, 0 citations."],
    [1, "Recherche : Discussions. 1 résultat, 1 citations."],
    [3, "Recherche : Discussions. 3 résultats, 2 citations."],
  ])("translates source outcomes with %s retained results", async (count, expected) => {
    testI18n.addResourceBundle("fr", "knowledge", { trustSummary: {
      searched: "Recherche : {{sources}}. {count, plural, one {# résultat} other {# résultats}}, {{citationCount}} citations.",
      unsearched: "Aucune recherche. {{count}} résultats, {{citationCount}} citations.",
      unavailable: "Recherche impossible : {{sources}}.",
      partialFailure: "Échec partiel : {{sources}}.",
    } })
    testI18n.addResourceBundle("fr", "sidepanel", { rag: { sources: { chats: "Discussions" } } })
    await testI18n.changeLanguage("fr")

    const lines = buildAnswerTrustSummary({
      selectedSources: ["chats"], resultCount: count, citationCount: Math.min(count, 2),
      webFallbackEnabled: false, webFallbackTriggered: false,
      generationProvider: null, generationModel: null, sourceHealthCaveatCount: 0,
      sourceStatus: { chats: { status: "error", count } },
    }, t)

    expect(lines[0]).toBe(expected)
    expect(lines).toContain(count === 0 ? "Recherche impossible : Discussions." : "Échec partiel : Discussions.")
  })

  it("translates the joining of multiple source names", async () => {
    testI18n.addResourceBundle("fr", "knowledge", { trustSummary: {
      sourceList: "{{first}} et {{last}}", sourceSeparator: ", ",
    } })
    testI18n.addResourceBundle("fr", "sidepanel", { rag: { sources: {
      media: "Documents", notes: "Notes personnelles", prompts: "Instructions",
    } } })
    await testI18n.changeLanguage("fr")

    expect(formatSourceList(["media_db", "notes", "prompts"], t)).toBe(
      "Documents, Notes personnelles et Instructions"
    )
  })

  it("formats selected source names with readable joining", () => {
    expect(formatSourceList(["media_db", "notes", "prompts"], t)).toBe(
      "Documents & Media, Notes, and Prompts"
    )
  })

  it("reports failed retrieval sources even when preflight health was ready", () => {
    const lines = buildAnswerTrustSummary({
      selectedSources: ["media_db", "characters", "chats"],
      resultCount: 1,
      citationCount: 1,
      webFallbackEnabled: false,
      webFallbackTriggered: false,
      generationProvider: null,
      generationModel: null,
      sourceHealthCaveatCount: 0,
      sourceStatus: {
        media_db: { status: "searched", count: 1 },
        characters: { status: "error", count: 0, reason: "retrieval_failed" },
        chats: { status: "error", count: 0, reason: "retrieval_failed" },
      },
    }, t)

    expect(lines).toContain("Searched Documents & Media. 1 source returned, 1 cited.")
    expect(lines).toContain("Could not search Characters and Chats.")
    expect(lines).not.toContain("Selected sources look ready.")
  })

  it("keeps successful evidence visible when another search of that source fails", () => {
    const lines = buildAnswerTrustSummary({
      selectedSources: ["characters"],
      resultCount: 1,
      citationCount: 1,
      webFallbackEnabled: false,
      webFallbackTriggered: false,
      generationProvider: null,
      generationModel: null,
      sourceHealthCaveatCount: 0,
      sourceStatus: { characters: { status: "error", count: 1, reason: "retrieval_failed" } },
    }, t)

    expect(lines).toContain("Searched Characters. 1 source returned, 1 cited.")
    expect(lines).toContain("Some searches failed for Characters.")
    expect(lines).not.toContain("Selected sources look ready.")
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
      }, t)
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
      }, t)
    ).toContain(`Trust: ${expectedLabel}.`)
  })
})
