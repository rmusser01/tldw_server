import { describe, expect, it } from "vitest"

import { buildNotesTagSearchParams } from "../notes-tag-match"

/**
 * NL-12 tag-match follow-up (#3101): the any/all toggle applies to the tags
 * the reader picked. Tags a view implies (Inbox's "captured") always match.
 */
const params = (
  picked: string[],
  required: string[],
  match: "all" | "any"
): Record<string, string[]> => {
  const search = new URLSearchParams()
  buildNotesTagSearchParams(search, { picked, required, match })
  return Object.fromEntries(
    Array.from(new Set(search.keys())).map((key) => [key, search.getAll(key)])
  )
}

describe("buildNotesTagSearchParams", () => {
  it("sends picked tags with the reader's match when no view adds a tag", () => {
    expect(params(["rag", "baking"], [], "any")).toEqual({
      tokens: ["rag", "baking"],
      tokens_match: ["any"]
    })
    expect(params(["rag"], [], "any")).toEqual({ tokens: ["rag"] })
  })

  it("keeps the view's tag and one picked tag both required", () => {
    expect(params(["rag"], ["captured"], "any")).toEqual({
      tokens: ["captured", "rag"],
      tokens_match: ["all"]
    })
  })

  it("sends the view's tag as required when the reader asks for any of several tags", () => {
    expect(params(["rag", "baking"], ["captured"], "any")).toEqual({
      tokens: ["rag", "baking"],
      tokens_match: ["any"],
      required_tokens: ["captured"]
    })
  })

  it("needs no required tokens when every tag must match anyway", () => {
    expect(params(["rag", "baking"], ["captured"], "all")).toEqual({
      tokens: ["captured", "rag", "baking"],
      tokens_match: ["all"]
    })
  })

  it("sends a view's tag alone as the only token", () => {
    expect(params([], ["captured"], "any")).toEqual({ tokens: ["captured"] })
  })

  it("does not repeat a picked tag the view already requires", () => {
    expect(params(["captured", "rag", "baking"], ["captured"], "any")).toEqual({
      tokens: ["rag", "baking"],
      tokens_match: ["any"],
      required_tokens: ["captured"]
    })
  })
})
