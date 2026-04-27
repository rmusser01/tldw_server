import type { JSONContent } from "@tiptap/react"
import { describe, expect, it } from "vitest"
import {
  DEFAULT_SETTINGS,
  getPromptRichFromPayload,
  mergePayloadIntoSession
} from "../hooks/utils"

const RICH_DOC: JSONContent = {
  type: "doc",
  content: [
    {
      type: "paragraph",
      content: [{ type: "text", text: "Rich draft" }]
    }
  ]
}

describe("writing session payload utils", () => {
  it("stores prompt_rich when rich content is supplied", () => {
    const payload = mergePayloadIntoSession({}, "Rich draft", DEFAULT_SETTINGS, null, null, false, {
      promptRich: RICH_DOC
    })

    expect(payload.prompt).toBe("Rich draft")
    expect(payload.prompt_rich).toEqual(RICH_DOC)
  })

  it("clears prompt_rich on plain-text prompt updates", () => {
    const payload = mergePayloadIntoSession(
      { prompt: "old", prompt_rich: RICH_DOC },
      "Plain replacement",
      DEFAULT_SETTINGS,
      null,
      null,
      false,
      { promptRich: null }
    )

    expect(payload.prompt).toBe("Plain replacement")
    expect(payload).not.toHaveProperty("prompt_rich")
  })

  it("returns null for malformed prompt_rich payloads", () => {
    expect(getPromptRichFromPayload({ prompt_rich: "bad" })).toBeNull()
  })
})
