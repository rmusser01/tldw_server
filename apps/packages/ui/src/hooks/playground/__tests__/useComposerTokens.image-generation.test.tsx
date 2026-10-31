import { describe, expect, it } from "vitest"
import { renderHook } from "@testing-library/react"
import {
  useComposerTokens,
  type UseComposerTokensParams
} from "@/hooks/playground/useComposerTokens"
import { IMAGE_GENERATION_USER_MESSAGE_TYPE } from "@/utils/image-generation-chat"

describe("useComposerTokens image-generation filtering", () => {
  const initialProps: UseComposerTokensParams = {
    message: "",
    messages: [
      { isBot: false, message: "A previous conversation with some content" }
    ],
    systemPrompt: "",
    resolvedMaxContext: 4096,
    apiModelLabel: "gpt-4o-mini",
    isSending: false
  }

  it("keeps a cleared conversation at zero when the next send begins", () => {
    const { result, rerender } = renderHook(useComposerTokens, { initialProps })
    expect(result.current.conversationTokenCount).toBeGreaterThan(0)
    rerender({ ...initialProps, messages: [] })
    expect(result.current.conversationTokenCount).toBe(0)

    rerender({ ...initialProps, messages: [], isSending: true })

    expect(result.current.conversationTokenCount).toBe(0)
  })

  it("freezes the prior total while streaming and refreshes when streaming finishes", () => {
    const { result, rerender } = renderHook(useComposerTokens, { initialProps })
    const previousCount = result.current.conversationTokenCount
    const streamedMessages = [
      ...initialProps.messages,
      { isBot: true, message: "A much longer assistant response ".repeat(20) }
    ]
    rerender({ ...initialProps, messages: streamedMessages, isSending: true })
    expect(result.current.conversationTokenCount).toBe(previousCount)

    rerender({ ...initialProps, messages: streamedMessages, isSending: false })

    expect(result.current.conversationTokenCount).toBeGreaterThan(previousCount)
  })

  it("does not include image-generation messages in conversation token totals", () => {
    const baseMessages = [
      { isBot: false, message: "Hello there" },
      { isBot: true, message: "Hi! How can I help?" }
    ]
    const withImageMessage = [
      ...baseMessages,
      {
        isBot: false,
        message: "Generate an image prompt that should not count",
        messageType: IMAGE_GENERATION_USER_MESSAGE_TYPE
      }
    ]

    const { result: baseResult } = renderHook(() =>
      useComposerTokens({
        message: "",
        messages: baseMessages,
        systemPrompt: "",
        resolvedMaxContext: 4096,
        apiModelLabel: "gpt-4o-mini",
        isSending: false
      })
    )

    const { result: withImageResult } = renderHook(() =>
      useComposerTokens({
        message: "",
        messages: withImageMessage,
        systemPrompt: "",
        resolvedMaxContext: 4096,
        apiModelLabel: "gpt-4o-mini",
        isSending: false
      })
    )

    expect(withImageResult.current.conversationTokenCount).toBe(
      baseResult.current.conversationTokenCount
    )
  })
})
