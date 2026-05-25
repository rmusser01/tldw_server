import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  promptForRag: vi.fn(),
  generateHistory: vi.fn(),
  pageAssistModel: vi.fn(),
  humanMessageFormatter: vi.fn(),
  removeReasoning: vi.fn(),
  formatDocs: vi.fn(),
  getNoOfRetrievedDocs: vi.fn(),
  coerceBooleanOrNull: vi.fn(),
  tldwInitialize: vi.fn(),
  ragSearch: vi.fn(),
  maybeInjectActorMessage: vi.fn(),
  resolveApiProviderForModel: vi.fn(),
  runChatPipeline: vi.fn(),
  appendSystemPromptSuffix: vi.fn()
}))

vi.mock("~/services/tldw-server", () => ({
  promptForRag: (...args: unknown[]) => mocks.promptForRag(...args)
}))

vi.mock("@/utils/generate-history", () => ({
  generateHistory: (...args: unknown[]) => mocks.generateHistory(...args)
}))

vi.mock("@/models", () => ({
  pageAssistModel: (...args: unknown[]) => mocks.pageAssistModel(...args)
}))

vi.mock("@/utils/human-message", () => ({
  humanMessageFormatter: (...args: unknown[]) =>
    mocks.humanMessageFormatter(...args)
}))

vi.mock("@/libs/reasoning", () => ({
  removeReasoning: (...args: unknown[]) => mocks.removeReasoning(...args)
}))

vi.mock("@/utils/format-docs", () => ({
  formatDocs: (...args: unknown[]) => mocks.formatDocs(...args)
}))

vi.mock("@/services/app", () => ({
  getNoOfRetrievedDocs: (...args: unknown[]) =>
    mocks.getNoOfRetrievedDocs(...args)
}))

vi.mock("@/services/rag/unified-rag", () => ({
  DEFAULT_RAG_SETTINGS: {
    include_note_ids: [],
    include_media_ids: [],
    top_k: 8,
    search_mode: "hybrid",
    enable_generation: true,
    enable_citations: true,
    enable_intent_routing: true
  }
}))

vi.mock("@/services/settings/registry", () => ({
  coerceBooleanOrNull: (...args: unknown[]) =>
    mocks.coerceBooleanOrNull(...args)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: (...args: unknown[]) => mocks.tldwInitialize(...args),
    ragSearch: (...args: unknown[]) => mocks.ragSearch(...args)
  }
}))

vi.mock("@/utils/actor", () => ({
  maybeInjectActorMessage: (...args: unknown[]) =>
    mocks.maybeInjectActorMessage(...args)
}))

vi.mock("@/utils/resolve-api-provider", () => ({
  resolveApiProviderForModel: (...args: unknown[]) =>
    mocks.resolveApiProviderForModel(...args)
}))

vi.mock("../chatModePipeline", () => ({
  runChatPipeline: (...args: unknown[]) => mocks.runChatPipeline(...args)
}))

vi.mock("@/utils/output-formatting-guide", () => ({
  appendSystemPromptSuffix: (...args: unknown[]) =>
    mocks.appendSystemPromptSuffix(...args)
}))

import { __testing__ } from "../ragMode"

const createRagContext = (overrides: Record<string, unknown> = {}) =>
  ({
    message: "What phrase proves the selected source was used?",
    image: "",
    isRegenerate: false,
    messages: [],
    history: [],
    signal: new AbortController().signal,
    createdAt: 1,
    generateMessageId: "assistant-1",
    resolvedUserMessageId: "user-1",
    resolvedAssistantMessageId: "assistant-1",
    resolvedAssistantParentMessageId: "user-1",
    resolvedModelId: "gemma3:1b",
    selectedModel: "gemma3:1b",
    userModelId: "gemma3:1b",
    modelInfo: null,
    regenerateVariants: [],
    useOCR: false,
    selectedKnowledge: null,
    currentChatModelSettings: { apiProvider: "ollama" },
    toolChoice: "none",
    setMessages: vi.fn(),
    saveMessageOnSuccess: vi.fn(),
    saveMessageOnError: vi.fn(),
    setHistory: vi.fn(),
    setIsProcessing: vi.fn(),
    setStreaming: vi.fn(),
    setAbortController: vi.fn(),
    historyId: null,
    setHistoryId: vi.fn(),
    ragMediaIds: [7, 8],
    ragSearchMode: "hybrid",
    ragTopK: null,
    ragEnableGeneration: true,
    ragEnableCitations: true,
    ragSources: [],
    ragAdvancedOptions: { enable_intent_routing: true },
    ...overrides
  }) as any

describe("ragMode sanitizer", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.promptForRag.mockResolvedValue({
      ragPrompt: "Use context:\n{context}\nQuestion: {question}",
      ragQuestionPrompt: "{question}"
    })
    mocks.generateHistory.mockReturnValue([])
    mocks.humanMessageFormatter.mockImplementation(async (input) => input)
    mocks.removeReasoning.mockImplementation((value) => value)
    mocks.formatDocs.mockImplementation((docs) =>
      docs.map((doc: any) => doc.pageContent).join("\n")
    )
    mocks.getNoOfRetrievedDocs.mockResolvedValue(8)
    mocks.coerceBooleanOrNull.mockImplementation((value) =>
      typeof value === "boolean" ? value : null
    )
    mocks.tldwInitialize.mockResolvedValue(undefined)
    mocks.maybeInjectActorMessage.mockImplementation(async (history) => history)
    mocks.resolveApiProviderForModel.mockResolvedValue("ollama")
    mocks.appendSystemPromptSuffix.mockImplementation(
      (prompt, suffix) => `${prompt}${suffix ?? ""}`
    )
  })

  it("preserves legacy numeric include_note_ids arrays", () => {
    const sanitized = __testing__.sanitizeRagAdvancedOptions({
      include_note_ids: [101, 202]
    })

    expect(sanitized.include_note_ids).toEqual([101, 202])
  })

  it("normalizes mixed include_note_ids arrays to strings", () => {
    const sanitized = __testing__.sanitizeRagAdvancedOptions({
      include_note_ids: [101, "note-2"]
    })

    expect(sanitized.include_note_ids).toEqual(["101", "note-2"])
  })

  it("disables intent routing and reuses retrieval for selected workspace media sources", async () => {
    mocks.ragSearch.mockResolvedValue({
      documents: [
        {
          content: "The Gate C live acceptance phrase is PASTE-EVIDENCE-ORION.",
          metadata: {
            source: "media_db",
            title: "TASK-478.5 Paste Evidence Source",
            type: "text"
          }
        }
      ]
    })
    const context = createRagContext()

    await expect(
      __testing__.ragModeDefinition.preflight?.(context)
    ).resolves.toBeNull()

    expect(mocks.ragSearch).toHaveBeenCalledWith(
      "What phrase proves the selected source was used?",
      expect.objectContaining({
        include_media_ids: [7, 8],
        sources: ["media_db"],
        enable_intent_routing: false,
        enable_pre_retrieval_clarification: false
      })
    )

    const prompt = await __testing__.ragModeDefinition.preparePrompt(context)

    expect(mocks.ragSearch).toHaveBeenCalledTimes(1)
    expect(prompt.sources).toEqual([
      expect.objectContaining({
        name: "TASK-478.5 Paste Evidence Source",
        type: "text"
      })
    ])
    expect(mocks.humanMessageFormatter).toHaveBeenCalledWith(
      expect.objectContaining({
        content: [
          expect.objectContaining({
            text: expect.stringContaining("PASTE-EVIDENCE-ORION")
          })
        ]
      })
    )
  })

  it("handles selected-source RAG with no evidence without continuing as general chat", async () => {
    mocks.ragSearch.mockResolvedValue({
      documents: [],
      generated_answer:
        "Could you clarify what specific item or context you want me to focus on?",
      metadata: {
        clarification: {
          required: true,
          stage: "pre_retrieval",
          reason: "ambiguous_reference_without_context"
        }
      }
    })

    const response = await __testing__.ragModeDefinition.preflight?.(
      createRagContext()
    )

    expect(response).toMatchObject({
      handled: true,
      fullText: expect.stringContaining("Could you clarify")
    })
    expect(response?.fullText).toContain("did not send this as general chat")
  })

  it("does not require an LLM query rewrite before retrieving selected workspace media sources", async () => {
    mocks.ragSearch.mockResolvedValue({
      documents: [
        {
          content: "The Gate C live acceptance phrase is PASTE-EVIDENCE-ORION.",
          metadata: {
            title: "TASK-478.5 Paste Evidence Source",
            type: "text"
          }
        }
      ]
    })
    const context = createRagContext({
      messages: [
        {
          isBot: false,
          name: "You",
          message: "Earlier question",
          sources: [],
          images: []
        },
        {
          isBot: true,
          name: "Assistant",
          message: "Earlier answer",
          sources: [],
          images: []
        }
      ]
    })

    await __testing__.ragModeDefinition.preflight?.(context)

    expect(mocks.pageAssistModel).not.toHaveBeenCalled()
    expect(mocks.ragSearch).toHaveBeenCalledWith(
      "What phrase proves the selected source was used?",
      expect.any(Object)
    )
  })

  it("uses backend generated answers directly for selected workspace media RAG", async () => {
    mocks.ragSearch.mockResolvedValue({
      documents: [
        {
          content: "The Gate C live acceptance phrase is PASTE-EVIDENCE-ORION.",
          metadata: {
            title: "TASK-478.5 Paste Evidence Source",
            type: "text"
          }
        }
      ],
      generated_answer:
        "The exact phrase is PASTE-EVIDENCE-ORION. It proves pasted workspace sources are indexed and used in grounded Research Workspace answers with visible evidence."
    })

    const response = await __testing__.ragModeDefinition.preflight?.(
      createRagContext()
    )

    expect(response).toMatchObject({
      handled: true,
      fullText: expect.stringContaining("PASTE-EVIDENCE-ORION"),
      sources: [
        expect.objectContaining({
          name: "TASK-478.5 Paste Evidence Source",
          mode: "rag"
        })
      ],
      generationInfo: expect.objectContaining({
        grounded: true,
        reason: "selected_source_generated_answer"
      })
    })
  })
})
