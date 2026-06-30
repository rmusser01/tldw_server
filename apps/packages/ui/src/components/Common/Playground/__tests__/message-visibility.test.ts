import { describe, expect, it } from "vitest"

import { hasVisibleAssistantResponse } from "../message-visibility"
import { IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE } from "@/utils/image-generation-chat"

describe("hasVisibleAssistantResponse", () => {
  it("treats whitespace-only assistant text as not visible", () => {
    expect(hasVisibleAssistantResponse({ message: "   " })).toBe(false)
  })

  it("recognizes assistant text, images, tool calls, and image generation events as visible", () => {
    expect(hasVisibleAssistantResponse({ message: "Visible reply" })).toBe(true)
    expect(hasVisibleAssistantResponse({ images: ["https://example.test/image.png"] })).toBe(true)
    expect(hasVisibleAssistantResponse({ toolCalls: [{ id: "call-1" }] })).toBe(true)
    expect(
      hasVisibleAssistantResponse({
        message_type: IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE
      })
    ).toBe(true)
    expect(
      hasVisibleAssistantResponse({
        messageType: IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE
      })
    ).toBe(true)
  })
})
