import { describe, expect, it } from "vitest"

import { buildCharacterSelectionPayload } from "../utils"

describe("buildCharacterSelectionPayload", () => {
  it("preserves greeting variants, extensions, and image fields for pre-hydration consumers", () => {
    const payload = buildCharacterSelectionPayload({
      id: "char-123",
      name: "Payload Bot",
      greeting: "Primary greeting",
      alternate_greetings: ["Alt one", "Alt two"],
      extensions: { voice: "calm" },
      image_base64: "AAAA",
      image_mime: "image/png",
      avatar_url: "",
      system_prompt: "System prompt",
      description: "Description",
      version: 7
    })

    expect(payload).toEqual(
      expect.objectContaining({
        id: "char-123",
        name: "Payload Bot",
        greeting: "Primary greeting",
        alternate_greetings: ["Alt one", "Alt two"],
        extensions: { voice: "calm" },
        image_base64: "AAAA",
        image_mime: "image/png",
        version: 7
      })
    )
  })

  it("normalizes legacy serialized alternate greetings and extensions", () => {
    const payload = buildCharacterSelectionPayload({
      id: "char-legacy",
      name: "Legacy Payload Bot",
      greeting: "Primary greeting",
      alternate_greetings: "[\"Alt one\",\"Alt two\"]",
      extensions: "{\"voice\":\"calm\",\"tldw\":{\"favorite\":true}}",
      avatar_url: ""
    })

    expect(payload.alternate_greetings).toEqual(["Alt one", "Alt two"])
    expect(payload.extensions).toEqual({
      voice: "calm",
      tldw: { favorite: true }
    })
  })
})
