import { describe, expect, it } from "vitest"
import {
  buildFlashcardsGenerateRoute,
  parseFlashcardsGenerateIntentFromLocation,
  parseFlashcardsGenerateIntentFromSearch
} from "@/services/tldw/flashcards-generate-handoff"

describe("flashcards generate handoff helpers", () => {
  it("builds a transfer route with generate prefill params", () => {
    const route = buildFlashcardsGenerateRoute({
      text: "Mitochondria is the powerhouse of the cell.",
      sourceType: "media",
      sourceId: "42",
      sourceTitle: "Biology Lecture"
    })

    expect(route.startsWith("/flashcards?")).toBe(true)
    const search = route.slice(route.indexOf("?"))
    const params = new URLSearchParams(search.slice(1))
    expect(params.get("tab")).toBe("importExport")
    expect(params.get("generate")).toBe("1")
    expect(params.get("generate_text")).toBe(
      "Mitochondria is the powerhouse of the cell."
    )
    expect(params.get("generate_source_type")).toBe("media")
    expect(params.get("generate_source_id")).toBe("42")
    expect(params.get("generate_source_title")).toBe("Biology Lecture")
  })

  it("parses generate intent from normal search params", () => {
    const intent = parseFlashcardsGenerateIntentFromSearch(
      "?generate=1&generate_text=ATP%20production&generate_source_type=note&generate_source_id=77"
    )

    expect(intent).toEqual({
      text: "ATP production",
      sourceType: "note",
      sourceId: "77",
      sourceTitle: undefined,
      conversationId: undefined,
      messageId: undefined
    })
  })

  it("keeps manual source details for selected-page extension captures", () => {
    const route = buildFlashcardsGenerateRoute({
      text: "Highlighted page text",
      sourceType: "manual",
      sourceId: "https://example.test/source",
      sourceTitle: "Captured Page"
    })

    const intent = parseFlashcardsGenerateIntentFromSearch(
      route.slice(route.indexOf("?"))
    )

    expect(intent).toEqual({
      text: "Highlighted page text",
      sourceType: "manual",
      sourceId: "https://example.test/source",
      sourceTitle: "Captured Page",
      conversationId: undefined,
      messageId: undefined
    })
  })

  it("parses generate intent from hash-based routes", () => {
    const intent = parseFlashcardsGenerateIntentFromLocation({
      search: "",
      hash: "#/flashcards?generate=1&generate_text=Cell%20cycle&generate_source_type=message&generate_source_id=tab-1"
    })

    expect(intent).toEqual({
      text: "Cell cycle",
      sourceType: "message",
      sourceId: "tab-1",
      sourceTitle: undefined,
      conversationId: undefined,
      messageId: undefined
    })
  })
})
