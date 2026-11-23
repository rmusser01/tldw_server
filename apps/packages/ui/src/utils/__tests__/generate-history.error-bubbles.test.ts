import { describe, expect, it } from "vitest"
import { generateHistory } from "@/utils/generate-history"
import { encodeChatErrorPayload } from "@/utils/chat-error-message"
import type { BaseMessage } from "@/types/messages"

const errorBubble = encodeChatErrorPayload({
  summary: "No response was returned.",
  hint: "Retry, or choose a different model.",
  detail: "No response text was returned."
})

const textOf = (message: BaseMessage) =>
  typeof message.content === "string"
    ? message.content
    : message.content
        .filter((part) => part.type === "text")
        .map((part) => part.text)
        .join("")

describe.each(["gpt-4o-mini", "local_model-abcd-1234-abc-5678"])(
  "generateHistory diagnostic filtering for %s",
  (model) => {
    it("excludes assistant error bubbles from a follow-up request", () => {
      const messages = [
        { role: "user" as const, content: "First request" },
        { role: "assistant" as const, content: errorBubble },
        { role: "user" as const, content: "Follow-up request" }
      ]

      const history = generateHistory(messages, model)

      expect(history.map(textOf)).toEqual(["First request", "Follow-up request"])
      expect(messages[1].content).toBe(errorBubble)
    })

    it("preserves user-authored error payloads", () => {
      const history = generateHistory(
        [{ role: "user", content: errorBubble }],
        model
      )

      expect(history.map(textOf)).toEqual([errorBubble])
    })

    it.each([
      "A partial answer before the connection failed.",
      "__tldw_error__:not-json",
      '__tldw_error__:{"summary":42,"hint":"not a UI error"}',
      `Example payload: ${errorBubble}`
    ])("preserves assistant content that is not a recognized error bubble: %s", (content) => {
      const history = generateHistory([{ role: "assistant", content }], model)

      expect(history.map(textOf)).toEqual([content])
    })
  }
)
