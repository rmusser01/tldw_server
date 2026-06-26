import type { JSONContent } from "@tiptap/react"
import { describe, expect, it } from "vitest"
import {
  plainTextToTipTapJson,
  resolveTipTapDocument,
  tipTapJsonToPlainText
} from "../writing-tiptap-utils"

const RICH_DOC: JSONContent = {
  type: "doc",
  content: [
    {
      type: "paragraph",
      content: [{ type: "text", text: "Stored rich draft" }]
    }
  ]
}

describe("writing tiptap utils", () => {
  it("prefers stored prompt_rich documents over plain-text reconstruction", () => {
    expect(resolveTipTapDocument("plain fallback", RICH_DOC)).toEqual(RICH_DOC)
  })

  it("falls back to plain-text conversion when prompt_rich is absent", () => {
    expect(resolveTipTapDocument("plain fallback", null)).toEqual(
      plainTextToTipTapJson("plain fallback")
    )
  })

  it("serializes adjacent rich paragraphs with blank-line paragraph delimiters", () => {
    expect(
      tipTapJsonToPlainText({
        type: "doc",
        content: [
          {
            type: "paragraph",
            content: [{ type: "text", text: "First paragraph." }]
          },
          {
            type: "paragraph",
            content: [{ type: "text", text: "Second paragraph." }]
          }
        ]
      })
    ).toBe("First paragraph.\n\nSecond paragraph.")
  })

  it("serializes scene breaks as standalone manuscript blocks", () => {
    expect(
      tipTapJsonToPlainText({
        type: "doc",
        content: [
          {
            type: "paragraph",
            content: [{ type: "text", text: "Before" }]
          },
          { type: "sceneBreak" },
          {
            type: "paragraph",
            content: [{ type: "text", text: "After" }]
          }
        ]
      })
    ).toBe("Before\n\n***\n\nAfter")
  })
})
