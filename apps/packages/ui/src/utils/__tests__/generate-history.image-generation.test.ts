import { describe, expect, it } from "vitest"
import { buildAssistantErrorContent, encodeChatErrorPayload } from "@/utils/chat-error-message"
import { generateHistory } from "@/utils/generate-history"
import {
  IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE,
  IMAGE_GENERATION_USER_MESSAGE_TYPE
} from "@/utils/image-generation-chat"

describe("generateHistory image-generation filtering", () => {
  it("excludes image generation no-op messages from prompt history", () => {
    const history = generateHistory(
      [
        {
          role: "user",
          content: "normal user turn"
        },
        {
          role: "assistant",
          content: "normal assistant turn"
        },
        {
          role: "user",
          content: "image prompt",
          messageType: IMAGE_GENERATION_USER_MESSAGE_TYPE
        },
        {
          role: "assistant",
          content: "",
          messageType: IMAGE_GENERATION_ASSISTANT_MESSAGE_TYPE
        }
      ],
      "gpt-4o-mini"
    )

    expect(history).toHaveLength(2)
    expect(history[0]?._getType()).toBe("human")
    expect(history[1]?._getType()).toBe("ai")
  })
})


describe("generateHistory display-error projection", () => {
  const error = encodeChatErrorPayload({ summary: "Provider failed", hint: "Retry", detail: "Upstream failed" })
  it("omits only a recognized assistant display error", () => {
    const history = generateHistory([{ role: "user", content: "Question" }, { role: "assistant", content: error }], "gpt-test")
    expect(history.map(row => row._getType())).toEqual(["human"])
  })
  it("retains a user quotation, malformed marker and genuine partial prose", () => {
    const history = generateHistory([
      { role: "user", content: error },
      { role: "assistant", content: "__tldw_error__:not-json" },
      { role: "assistant", content: buildAssistantErrorContent("Partial answer", new Error("Interrupted")) }
    ], "gpt-test")
    expect(history).toHaveLength(3)
    expect(JSON.stringify(history)).toContain("__tldw_error__:not-json")
    expect(JSON.stringify(history)).toContain("Partial answer")
  })
})
