// @vitest-environment jsdom
import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { useConversationContextComposition } from "../useConversationContextComposition"
import type { ConversationContextPrimitiveClient } from "@/services/conversation-context/conversationContextComposer"

const buildPrimitives = (): ConversationContextPrimitiveClient => ({
  processDictionary: vi.fn(async ({ text }) => ({
    original_text: text,
    processed_text: "Echo Vault",
    replacements: 1,
    iterations: 1,
    entries_used: [7]
  })),
  processWorldBookContext: vi.fn(async ({ text }) => ({
    injected_content:
      text === "Echo Vault" ? "Echo Vault lore." : "Unmatched lore.",
    entries_matched: 1,
    tokens_used: 3,
    books_used: 1,
    entry_ids: [11],
    diagnostics: [{ entry_id: 11, world_book_id: 3 }]
  }))
})

describe("useConversationContextComposition", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("reuses one composition object for preview and send", async () => {
    const primitives = buildPrimitives()
    const { result } = renderHook(() =>
      useConversationContextComposition({
        draftMessage: "EV",
        selection: {
          characterId: null,
          worldBookIds: [3],
          dictionaryIds: [7]
        },
        primitives
      })
    )

    await waitFor(() => expect(result.current.status).toBe("ready"))

    const previewComposition = result.current.composition
    const send = await result.current.composeForSend({
      message: "EV",
      history: []
    })

    expect(send.composition).toBe(previewComposition)
    expect(send.requestOverrides.historyForModel).toEqual([
      {
        role: "system",
        content: expect.stringContaining("Echo Vault lore.")
      }
    ])
  })

  it("builds model-only send overrides without transforming the stored user message", async () => {
    const primitives = buildPrimitives()
    const { result } = renderHook(() =>
      useConversationContextComposition({
        draftMessage: "EV",
        selection: {
          characterId: null,
          worldBookIds: [3],
          dictionaryIds: [7]
        },
        primitives
      })
    )

    await waitFor(() => expect(result.current.status).toBe("ready"))

    const send = await result.current.composeForSend({
      message: "EV",
      history: [{ role: "user", content: "Earlier authored text" }]
    })

    expect(send.requestOverrides.messageForModel).toBe("Echo Vault")
    expect(send.authoredMessage).toBe("EV")
    expect(primitives.processDictionary).toHaveBeenCalledTimes(1)
  })

  it("allows blank chat send with no selected optional context", async () => {
    const primitives = buildPrimitives()
    const { result } = renderHook(() =>
      useConversationContextComposition({
        draftMessage: "Hello",
        selection: {
          characterId: null,
          worldBookIds: [],
          dictionaryIds: []
        },
        primitives
      })
    )

    await waitFor(() => expect(result.current.status).toBe("ready"))

    const send = await result.current.composeForSend({
      message: "Hello",
      history: []
    })

    expect(send.composition.readiness).toBe("ready")
    expect(send.requestOverrides.historyForModel).toEqual([])
    expect(send.requestOverrides.messageForModel).toBe("Hello")
    expect(primitives.processDictionary).not.toHaveBeenCalled()
    expect(primitives.processWorldBookContext).not.toHaveBeenCalled()
  })

  it("debounces preview composition while keeping send composition immediate", async () => {
    vi.useFakeTimers()
    const primitives = buildPrimitives()
    const { result, unmount } = renderHook(() =>
      useConversationContextComposition({
        draftMessage: "EV",
        selection: {
          characterId: null,
          worldBookIds: [3],
          dictionaryIds: [7]
        },
        debounceMs: 100,
        primitives
      })
    )

    expect(result.current.status).toBe("loading")
    expect(primitives.processDictionary).not.toHaveBeenCalled()

    let sendMessageForModel: string | undefined
    await act(async () => {
      const send = await result.current.composeForSend({
        message: "EV",
        history: []
      })
      sendMessageForModel = send.requestOverrides.messageForModel
    })

    expect(sendMessageForModel).toBe("Echo Vault")
    expect(primitives.processDictionary).toHaveBeenCalledTimes(1)
    unmount()
  })
})
