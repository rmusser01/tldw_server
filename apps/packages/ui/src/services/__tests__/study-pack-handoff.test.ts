import { describe, expect, it } from "vitest"

import {
  buildStudyPackRoute,
  parseStudyPackIntentFromLocation,
  parseStudyPackIntentFromSearch
} from "@/services/tldw/study-pack-handoff"

describe("study-pack handoff helpers", () => {
  it("builds a study-pack route that round-trips source ids and titles", () => {
    const route = buildStudyPackRoute({
      title: "Networks",
      sourceItems: [
        {
          sourceType: "media",
          sourceId: "42",
          sourceTitle: "Lecture 5"
        }
      ]
    })

    expect(route.startsWith("/flashcards?")).toBe(true)
    const params = new URLSearchParams(route.slice(route.indexOf("?") + 1))
    expect(params.get("tab")).toBe("importExport")
    expect(params.get("study_pack")).toBe("1")
    expect(params.get("study_pack_title")).toBe("Networks")

    const intent = parseStudyPackIntentFromSearch(route.slice(route.indexOf("?")))
    expect(intent).toEqual({
      title: "Networks",
      sourceItems: [
        {
          sourceType: "media",
          sourceId: "42",
          sourceTitle: "Lecture 5"
        }
      ]
    })
  })

  it("parses study-pack intent from hash-based routes", () => {
    const intent = parseStudyPackIntentFromLocation({
      search: "",
      hash: "#/flashcards?tab=importExport&study_pack=1&study_pack_title=Biology&study_pack_payload=%7B%22title%22%3A%22Biology%22%2C%22sourceItems%22%3A%5B%7B%22sourceType%22%3A%22note%22%2C%22sourceId%22%3A%22note-1%22%2C%22sourceTitle%22%3A%22Chapter%201%22%7D%5D%7D"
    })

    expect(intent).toEqual({
      title: "Biology",
      sourceItems: [
        {
          sourceType: "note",
          sourceId: "note-1",
          sourceTitle: "Chapter 1"
        }
      ]
    })
  })

  it("returns null when the study-pack signal is missing or payload is invalid", () => {
    expect(parseStudyPackIntentFromSearch("?tab=importExport")).toBeNull()
    expect(
      parseStudyPackIntentFromSearch(
        "?tab=importExport&study_pack=1&study_pack_payload=%7Bnot-json"
      )
    ).toBeNull()
  })
})
