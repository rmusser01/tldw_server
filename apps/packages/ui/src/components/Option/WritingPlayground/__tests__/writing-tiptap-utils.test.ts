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

  it("preserves empty rich paragraphs as distinct blank-line blocks", () => {
    expect(
      tipTapJsonToPlainText({
        type: "doc",
        content: [
          {
            type: "paragraph",
            content: [{ type: "text", text: "First paragraph." }]
          },
          {
            type: "paragraph"
          },
          {
            type: "paragraph",
            content: [{ type: "text", text: "Second paragraph." }]
          }
        ]
      })
    ).toBe("First paragraph.\n\n\n\nSecond paragraph.")
  })

  it("serializes nested list items with block delimiters", () => {
    expect(
      tipTapJsonToPlainText({
        type: "doc",
        content: [
          {
            type: "bulletList",
            content: [
              {
                type: "listItem",
                content: [
                  {
                    type: "paragraph",
                    content: [{ type: "text", text: "First item" }]
                  }
                ]
              },
              {
                type: "listItem",
                content: [
                  {
                    type: "paragraph",
                    content: [{ type: "text", text: "Second item" }]
                  }
                ]
              }
            ]
          }
        ]
      })
    ).toBe("First item\n\nSecond item")
  })

  it("serializes nested blockquote paragraphs with block delimiters", () => {
    expect(
      tipTapJsonToPlainText({
        type: "doc",
        content: [
          {
            type: "blockquote",
            content: [
              {
                type: "paragraph",
                content: [{ type: "text", text: "First quote" }]
              },
              {
                type: "paragraph",
                content: [{ type: "text", text: "Second quote" }]
              }
            ]
          }
        ]
      })
    ).toBe("First quote\n\nSecond quote")
  })

  it("round-trips single newlines as hard breaks inside a paragraph", () => {
    const json = plainTextToTipTapJson("First line\nSecond line")

    expect(json).toEqual({
      type: "doc",
      content: [
        {
          type: "paragraph",
          content: [
            { type: "text", text: "First line" },
            { type: "hardBreak" },
            { type: "text", text: "Second line" }
          ]
        }
      ]
    })
    expect(tipTapJsonToPlainText(json)).toBe("First line\nSecond line")
  })

  it("round-trips empty paragraphs without collapsing them", () => {
    const text = "First paragraph.\n\n\n\nSecond paragraph."

    expect(tipTapJsonToPlainText(plainTextToTipTapJson(text))).toBe(text)
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

  it("round-trips scene breaks as standalone manuscript blocks", () => {
    const text = "Before\n\n***\n\nAfter"

    expect(tipTapJsonToPlainText(plainTextToTipTapJson(text))).toBe(text)
  })
})
