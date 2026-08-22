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
    collection_id: null,
    include_note_ids: [],
    include_media_ids: [],
    ground_truth_doc_ids: [],
    top_k: 8,
    search_mode: "hybrid",
    enable_generation: true,
    enable_citations: true,
    enable_intent_routing: true,
    accumulation_time_budget_sec: null,
    subquery_time_budget_sec: null,
    subquery_doc_budget: null,
    grading_model: null,
    grading_provider: null,
    fast_hallucination_provider: null,
    fast_hallucination_model: null,
    utility_grading_provider: null,
    utility_grading_model: null
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
    mocks.resolveApiProviderForModel(...args),
  parseProviderQualifiedModelSelection: (value: unknown) => {
    const raw = String(value || "").trim()
    if (!raw.startsWith("llama.cpp:")) {
      return {
        raw,
        modelId: raw,
        provider: undefined,
        isProviderQualified: false
      }
    }
    return {
      raw,
      modelId: raw.slice("llama.cpp:".length),
      provider: "llama.cpp",
      isProviderQualified: true
    }
  }
}))

vi.mock("../chatModePipeline", () => ({
  runChatPipeline: (...args: unknown[]) => mocks.runChatPipeline(...args),
  getRequiredServicePrompt: (snapshot: any, id: string) => {
    const resolved = snapshot?.definitions?.[id]
    if (!resolved) throw new Error(`Service Prompt snapshot is missing ${id}.`)
    return resolved
  }
}))

vi.mock("@/utils/output-formatting-guide", () => ({
  appendSystemPromptSuffix: (...args: unknown[]) =>
    mocks.appendSystemPromptSuffix(...args)
}))

import { __testing__ } from "../ragMode"

const servicePromptSnapshot = {
  scopeKey: "test-scope",
  requestScope: {
    config: {
      serverUrl: "https://example.test",
      authMode: "single-user" as const
    },
    userId: 1
  },
  capability: "supported" as const,
  scopeSignal: new AbortController().signal,
  scopeInvalidatedSignal: new AbortController().signal,
  release: vi.fn(),
  definitions: {
    "chat.rag.answer": {
      definition: {
        id: "chat.rag.answer",
        parts: [
          {
            key: "template",
            mode: "template" as const,
            required_variables: ["context", "question"]
          }
        ]
      },
      parts: {
        template: "Use context:\n{context}\nQuestion: {question}"
      },
      source: "packaged" as const,
      revision: null
    },
    "chat.rag.question_rewrite": {
      definition: {
        id: "chat.rag.question_rewrite",
        parts: [
          {
            key: "template",
            mode: "template" as const,
            required_variables: ["chat_history", "question"]
          }
        ]
      },
      parts: {
        template: "History: {chat_history}\nQuestion: {question}"
      },
      source: "packaged" as const,
      revision: null
    }
  }
}

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
    servicePromptSnapshot,
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

  it("drops the unsupported generic filters option", () => {
    const sanitized = __testing__.sanitizeRagAdvancedOptions({
      filters: { url: "https://example.com/private" },
      include_media_ids: [321]
    })

    expect(sanitized).toEqual({ include_media_ids: [321] })
  })

  it("preserves validated ground-truth document id arrays", () => {
    const sanitized = __testing__.sanitizeRagAdvancedOptions({
      ground_truth_doc_ids: [" doc-1 ", "doc-2"]
    })

    expect(sanitized.ground_truth_doc_ids).toEqual(["doc-1", "doc-2"])
  })

  it("preserves finite values and nulls for nullable numeric settings", () => {
    const sanitized = __testing__.sanitizeRagAdvancedOptions({
      collection_id: 7,
      accumulation_time_budget_sec: null,
      subquery_time_budget_sec: 1.5,
      subquery_doc_budget: 4
    })
    const cleared = __testing__.sanitizeRagAdvancedOptions({
      collection_id: null,
      accumulation_time_budget_sec: 0,
      subquery_time_budget_sec: null,
      subquery_doc_budget: null
    })

    expect(sanitized).toEqual({
      collection_id: 7,
      accumulation_time_budget_sec: null,
      subquery_time_budget_sec: 1.5,
      subquery_doc_budget: 4
    })
    expect(cleared).toEqual({
      collection_id: null,
      accumulation_time_budget_sec: 0,
      subquery_time_budget_sec: null,
      subquery_doc_budget: null
    })
  })

  it("preserves trimmed values and nulls for nullable string settings", () => {
    const sanitized = __testing__.sanitizeRagAdvancedOptions({
      grading_model: " grader-model ",
      grading_provider: null,
      fast_hallucination_provider: " fast-provider ",
      fast_hallucination_model: null,
      utility_grading_provider: " utility-provider ",
      utility_grading_model: null
    })

    expect(sanitized).toEqual({
      grading_model: "grader-model",
      grading_provider: null,
      fast_hallucination_provider: "fast-provider",
      fast_hallucination_model: null,
      utility_grading_provider: "utility-provider",
      utility_grading_model: null
    })
  })

  it("rejects malformed advanced settings and transport controls", () => {
    const sanitized = __testing__.sanitizeRagAdvancedOptions({
      ground_truth_doc_ids: ["doc-1", 2],
      collection_id: "7",
      accumulation_time_budget_sec: Number.NaN,
      subquery_time_budget_sec: Number.POSITIVE_INFINITY,
      subquery_doc_budget: "4",
      grading_model: 1,
      grading_provider: "  ",
      signal: new AbortController().signal,
      requestScope: { userId: 1 },
      query: "do not override the submitted query"
    })

    expect(sanitized).toEqual({})
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

  it("sends raw generation_model when the selected RAG model is provider-qualified", async () => {
    mocks.ragSearch.mockResolvedValue({
      documents: [
        {
          content: "The llama.cpp provider accepted the raw GGUF model id.",
          metadata: {
            source: "media_db",
            title: "llama.cpp runtime source",
            type: "text"
          }
        }
      ]
    })
    mocks.resolveApiProviderForModel.mockResolvedValue("llama.cpp")

    await expect(
      __testing__.ragModeDefinition.preflight?.(
        createRagContext({
          selectedModel:
            "llama.cpp:gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf",
          currentChatModelSettings: { apiProvider: undefined }
        })
      )
    ).resolves.toBeNull()

    expect(mocks.ragSearch).toHaveBeenCalledWith(
      "What phrase proves the selected source was used?",
      expect.objectContaining({
        generation_model:
          "gemma-4-26B-A4B-it-ultra-uncensored-heretic-Q4_K_M.gguf",
        generation_provider: "llama.cpp"
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

  it("does not convert a selected-source scope rejection into a handled response", async () => {
    const scopeError = Object.assign(new Error("scope changed"), {
      status: 412,
      details: {
        detail: { code: "request_config_scope_changed" }
      }
    })
    mocks.ragSearch.mockRejectedValueOnce(scopeError)

    await expect(
      __testing__.ragModeDefinition.preflight?.(createRagContext())
    ).rejects.toBe(scopeError)
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

  it("continues through chat completion when selected workspace media RAG returns evidence and a generated answer", async () => {
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
    const context = createRagContext()

    await expect(
      __testing__.ragModeDefinition.preflight?.(context)
    ).resolves.toBeNull()

    const prompt = await __testing__.ragModeDefinition.preparePrompt(context)

    expect(mocks.ragSearch).toHaveBeenCalledTimes(1)
    expect(prompt.sources).toEqual([
      expect.objectContaining({
        name: "TASK-478.5 Paste Evidence Source",
        mode: "rag"
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
    expect(mocks.humanMessageFormatter).toHaveBeenCalledWith(
      expect.objectContaining({
        content: [
          expect.objectContaining({
            text: expect.stringContaining(
              "What phrase proves the selected source was used?"
            )
          })
        ]
      })
    )
  })
})
