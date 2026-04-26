import type { JSONContent } from "@tiptap/react"
import { describe, expect, it } from "vitest"
import {
  plainTextToTipTapJson,
  resolveTipTapDocument
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
})
