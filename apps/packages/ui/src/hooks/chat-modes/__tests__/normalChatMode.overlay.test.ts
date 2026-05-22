import { beforeEach, describe, expect, it, vi } from "vitest"
import { SystemMessage } from "@/types/messages"

const mocks = vi.hoisted(() => ({
  systemPromptForNonRagOption: vi.fn(async () => "Base system prompt"),
  getPromptById: vi.fn(async () => null),
  humanMessageFormatter: vi.fn(async () => ({ role: "user", content: "Hello" })),
  systemPromptFormatter: vi.fn(async ({ content }: { content: string }) =>
    new SystemMessage({ content })
  ),
  maybeInjectActorMessage: vi.fn(async (history: unknown[], actorSettings: unknown) =>
    actorSettings
      ? [...history, new SystemMessage({ content: "Actor scene injection" })]
      : history
  ),
  runChatPipeline: vi.fn()
}))

vi.mock("~/services/tldw-server", () => ({
  systemPromptForNonRagOption: mocks.systemPromptForNonRagOption,
  getWebSearchPrompt: vi.fn(async () => "Search wrapper\n{search_results}")
}))

vi.mock("@/db/dexie/helpers", () => ({
  getPromptById: mocks.getPromptById
}))

vi.mock("@/utils/human-message", () => ({
  humanMessageFormatter: mocks.humanMessageFormatter
}))

vi.mock("@/utils/system-message", () => ({
  systemPromptFormatter: mocks.systemPromptFormatter
}))

vi.mock("@/utils/actor", () => ({
  maybeInjectActorMessage: mocks.maybeInjectActorMessage
}))

vi.mock("@/services/search", () => ({
  getSearchSettings: vi.fn(async () => ({
    searchProvider: "google",
    totalSearchResults: 1
  }))
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(async () => null),
    webSearch: vi.fn(async () => ({
      web_search_results_dict: {
        results: [
          {
            title: "Source title",
            url: "https://example.com/source",
            snippet: "Search snippet"
          }
        ]
      }
    }))
  }
}))

vi.mock("./../chatModePipeline", () => ({
  runChatPipeline: mocks.runChatPipeline
}))

import { normalChatMode } from "../normalChatMode"

describe("normalChatMode overlay prompt ordering", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.runChatPipeline.mockImplementation(
      async (mode, message, image, isRegenerate, messages, history, signal, params) =>
        mode.preparePrompt({
          ...params,
          message,
          image,
          isRegenerate,
          messages,
          history,
          signal,
          createdAt: 0,
          generateMessageId: "assistant-1",
          resolvedAssistantMessageId: "assistant-1",
          resolvedAssistantParentMessageId: "user-1",
          resolvedModelId: params.selectedModel,
          regenerateVariants: [],
          modelInfo: null
        })
    )
  })

  it("keeps base prompt first, overlay second, actor after overlay, and web search last", async () => {
    const prompt = await normalChatMode(
      "Where should I go?",
      "",
      false,
      [],
      [],
      new AbortController().signal,
      {
        selectedModel: "gpt-test",
        useOCR: false,
        selectedSystemPrompt: "",
        currentChatModelSettings: {},
        overlaySystemPrompt: "Overlay snapshot prompt",
        actorSettings: { isEnabled: true } as never,
        webSearch: true,
        setMessages: vi.fn(),
        saveMessageOnSuccess: vi.fn(async () => null),
        saveMessageOnError: vi.fn(async () => null),
        setHistory: vi.fn(),
        setIsProcessing: vi.fn(),
        setStreaming: vi.fn(),
        setAbortController: vi.fn(),
        historyId: null,
        setHistoryId: vi.fn()
      }
    )

    expect(prompt.chatHistory).toHaveLength(4)
    expect((prompt.chatHistory[0] as SystemMessage).content).toBe("Base system prompt")
    expect((prompt.chatHistory[1] as SystemMessage).content).toBe("Overlay snapshot prompt")
    expect((prompt.chatHistory[2] as SystemMessage).content).toBe("Actor scene injection")
    expect((prompt.chatHistory[3] as SystemMessage).content).toContain("Source title")
  })

  it("keeps generic systemPromptAppendix appended to the active base prompt", async () => {
    const prompt = await normalChatMode(
      "Hello",
      "",
      false,
      [],
      [],
      new AbortController().signal,
      {
        selectedModel: "gpt-test",
        useOCR: false,
        selectedSystemPrompt: "",
        currentChatModelSettings: {},
        systemPromptAppendix: "Formatting guide suffix",
        setMessages: vi.fn(),
        saveMessageOnSuccess: vi.fn(async () => null),
        saveMessageOnError: vi.fn(async () => null),
        setHistory: vi.fn(),
        setIsProcessing: vi.fn(),
        setStreaming: vi.fn(),
        setAbortController: vi.fn(),
        historyId: null,
        setHistoryId: vi.fn()
      }
    )

    expect(prompt.chatHistory).toHaveLength(1)
    expect((prompt.chatHistory[0] as SystemMessage).content).toBe(
      "Base system prompt\n\nFormatting guide suffix"
    )
  })
})
