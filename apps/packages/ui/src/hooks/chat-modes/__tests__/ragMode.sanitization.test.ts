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
  bgRequest: vi.fn(),
  stream: vi.fn(),
  maybeInjectActorMessage: vi.fn(),
  getModels: vi.fn(),
  runChatPipeline: vi.fn(),
  captureNormalHistoryTurn: vi.fn(),
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

vi.mock("@/libs/reasoning", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/libs/reasoning")>(),
  removeReasoning: (...args: unknown[]) => mocks.removeReasoning(...args)
}))

vi.mock("@/db/dexie/nickname", () => ({ getModelNicknameByID: vi.fn().mockResolvedValue(null) }))
vi.mock("@/utils/mcp-disclosure", () => ({ applyMcpModuleDisclosureFromToolCalls: vi.fn() }))
vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.bgRequest(...args),
  bgStream: vi.fn(),
  bgUpload: vi.fn()
}))

vi.mock("@/utils/format-docs", () => ({
  formatDocs: (...args: unknown[]) => mocks.formatDocs(...args)
}))

vi.mock("@/services/app", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/services/app")>(),
  getNoOfRetrievedDocs: (...args: unknown[]) =>
    mocks.getNoOfRetrievedDocs(...args)
}))

vi.mock("@/services/rag/unified-rag", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/services/rag/unified-rag")>()
  return {
  ...actual,
  DEFAULT_RAG_SETTINGS: {
    ...actual.DEFAULT_RAG_SETTINGS,
    collection_id: null,
    include_note_ids: [],
    include_media_ids: [],
    ground_truth_doc_ids: [],
    top_k: 8,
    search_mode: "hybrid",
    enable_generation: true,
    generation_model: null,
    generation_provider: null,
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
  }
})

vi.mock("@/services/settings/registry", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/services/settings/registry")>(),
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

vi.mock("@/services/tldw", () => ({
  tldwModels: {
    getModels: (...args: unknown[]) => mocks.getModels(...args)
  }
}))

vi.mock("../chatModePipeline", async (importOriginal) => ({
  ...await importOriginal<typeof import("../chatModePipeline")>(),
  runChatPipeline: (...args: unknown[]) => mocks.runChatPipeline(...args),
  getRequiredServicePrompt: (snapshot: any, id: string) => {
    const resolved = snapshot?.definitions?.[id]
    if (!resolved) throw new Error(`Service Prompt snapshot is missing ${id}.`)
    return resolved
  }
}))

vi.mock("../normalChatMode", async (importOriginal) => ({
  ...await importOriginal<typeof import("../normalChatMode")>(),
  captureNormalHistoryTurn: (...args: unknown[]) => mocks.captureNormalHistoryTurn(...args)
}))

vi.mock("@/utils/output-formatting-guide", () => ({
  appendSystemPromptSuffix: (...args: unknown[]) =>
    mocks.appendSystemPromptSuffix(...args)
}))

import { __testing__, ragMode } from "../ragMode"

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

describe("ragMode captured-turn lifecycle", () => {
  it.each(["submitted", "failed", "throws"])("releases its captured turn when the pipeline %s", async (status) => {
    const turn = {
      capture: { rows: [], selected_content: [], snapshot: { owner_key: "native-owner", nodes: [] }, view: { conversation_id: "chat-1" } },
      resultId: status === "submitted" ? "result-1" : null,
      followResult: vi.fn().mockResolvedValue(true),
      finish: vi.fn().mockResolvedValue(undefined),
      release: vi.fn()
    }
    mocks.captureNormalHistoryTurn.mockResolvedValue(turn)
    mocks.runChatPipeline.mockReset()
    if (status === "throws") mocks.runChatPipeline.mockRejectedValue(new Error("preparation failed"))
    else mocks.runChatPipeline.mockResolvedValue({ status })
    const context = createRagContext({ historySelection: {}, tldwTurn: { user_message_id: "input-1" } })
    const request = ragMode(context.message, "", false, [], [], context.signal, context)
    if (status === "throws") await expect(request).rejects.toThrow("preparation failed")
    else await expect(request).resolves.toEqual({ status })
    expect(turn.release).toHaveBeenCalledOnce()
    if (status === "submitted") expect(turn.finish).toHaveBeenCalledWith(true)
    else expect(turn.finish).not.toHaveBeenCalled()
  })
})

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
    mocks.getModels.mockReset()
    mocks.getModels.mockResolvedValue([])
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

  it("prepares serialized note evidence without generating or dropping its semantic type", async () => {
    mocks.ragSearch.mockResolvedValue({ documents: [{ content: "Exact serialized note excerpt.",
      metadata: { source: "notes_db", title: "Selected note", type: "note", media_type: "text",
        media_id: 7, chunk_id: 17, note_id: "note-7", record_id: 19, start: 0, end: 35 } }] })
    const context = createRagContext({ historyTurn: { serverOwned: true } })

    await expect(__testing__.ragModeDefinition.preflight?.(context)).resolves.toBeNull()
    const prompt = await __testing__.ragModeDefinition.preparePrompt(context)

    expect(prompt.sources).toEqual([{ name: "Selected note", type: "note", mode: "rag", url: "",
      pageContent: "Exact serialized note excerpt.", metadata: { source: "notes_db", title: "Selected note",
        media_id: "7", chunk_id: "17", source_type: "note" } }])
    expect(mocks.ragSearch).toHaveBeenCalledTimes(1)
    expect(mocks.ragSearch.mock.calls[0][1].enable_generation).toBe(false)
    expect(mocks.stream).not.toHaveBeenCalled()
    expect(mocks.runChatPipeline).not.toHaveBeenCalled()
  })

  it.each([true, false])("retrieves server-owned evidence without generating an unadmitted answer when generation is %s", async (ragEnableGeneration) => {
    mocks.ragSearch.mockResolvedValue({
      documents: [{
        content: "The selected memo's launch date is 18 November 2026.",
        metadata: { source: "media_db", title: "Selected memo", type: "text" }
      }]
    })
    const context = createRagContext({
      historyTurn: { serverOwned: true },
      ragEnableGeneration,
      ragAdvancedOptions: {
        enable_generation: true,
        generation_model: "unadmitted-model",
        generation_provider: "unadmitted-provider",
        generation_prompt: "Do not generate this before admission."
      },
      currentChatModelSettings: { apiProvider: undefined }
    })

    await expect(__testing__.ragModeDefinition.preflight?.(context)).resolves.toBeNull()
    const prompt = await __testing__.ragModeDefinition.preparePrompt(context)

    expect(mocks.ragSearch).toHaveBeenCalledTimes(1)
    const options = mocks.ragSearch.mock.calls[0][1]
    expect(options.enable_generation).toBe(false)
    expect(options).not.toHaveProperty("generation_model")
    expect(options).not.toHaveProperty("generation_provider")
    expect(options).not.toHaveProperty("generation_prompt")
    expect(mocks.getModels).not.toHaveBeenCalled()
    expect(mocks.pageAssistModel).not.toHaveBeenCalled()
    expect(prompt.sources[0].pageContent).toContain("18 November 2026")
  })

  it("retrieves selected evidence without generating a discarded answer for a provider-qualified model", async () => {
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
        enable_generation: false,
        include_media_ids: [7, 8],
        sources: ["media_db"]
      })
    )
    expect(mocks.ragSearch.mock.calls[0][1]).not.toHaveProperty("generation_model")
    expect(mocks.ragSearch.mock.calls[0][1]).not.toHaveProperty("generation_provider")
  })

  it.each(["missing", "unavailable", "stale"])(
    "retrieves without generation or model lookup with %s metadata for a llama-qualified selection",
    async (metadataState) => {
      if (metadataState === "unavailable") {
        mocks.getModels.mockRejectedValue(new Error("Metadata unavailable"))
      } else if (metadataState === "stale") {
        mocks.getModels.mockResolvedValue([
          {
            id: "../../../models/gemma:Q4_K_M/model.gguf",
            name: "../../../models/gemma:Q4_K_M/model.gguf",
            provider: "openai",
            type: "chat"
          }
        ])
      }
      mocks.ragSearch.mockResolvedValue({ documents: [] })

      await __testing__.ragModeDefinition.preflight?.(
        createRagContext({
          selectedModel: "llama:../../../models/gemma:Q4_K_M/model.gguf",
          currentChatModelSettings: { apiProvider: "openai" }
        })
      )

      expect(mocks.ragSearch).toHaveBeenCalledWith(
        "What phrase proves the selected source was used?",
        expect.objectContaining({
          enable_generation: false,
          include_media_ids: [7, 8],
          sources: ["media_db"]
        })
      )
      expect(mocks.ragSearch.mock.calls[0][1]).not.toHaveProperty("generation_model")
      expect(mocks.ragSearch.mock.calls[0][1]).not.toHaveProperty("generation_provider")
      expect(mocks.getModels).not.toHaveBeenCalled()
    }
  )

  it("retrieves without generation or catalog lookup for an unqualified model", async () => {
    mocks.getModels.mockResolvedValue([
      {
        id: "../../../models/gemma:Q4_K_M/model.gguf",
        name: "../../../models/gemma:Q4_K_M/model.gguf",
        provider: "llama",
        type: "chat"
      }
    ])
    mocks.ragSearch.mockResolvedValue({ documents: [] })

    await __testing__.ragModeDefinition.preflight?.(
      createRagContext({
        selectedModel: "../../../models/gemma:Q4_K_M/model.gguf",
        currentChatModelSettings: { apiProvider: undefined }
      })
    )

    expect(mocks.ragSearch).toHaveBeenCalledWith(
      "What phrase proves the selected source was used?",
      expect.objectContaining({
        enable_generation: false,
        include_media_ids: [7, 8],
        sources: ["media_db"]
      })
    )
    expect(mocks.ragSearch.mock.calls[0][1]).not.toHaveProperty("generation_model")
    expect(mocks.ragSearch.mock.calls[0][1]).not.toHaveProperty("generation_provider")
    expect(mocks.getModels).not.toHaveBeenCalled()
  })

  it("sends retrieval-only RAG on the wire then streams one grounded answer with the same selected sources", async () => {
    const { chatRagMethods } = await import("@/services/tldw/domains/chat-rag")
    const { runChatPipeline } = await vi.importActual<typeof import("../chatModePipeline")>("../chatModePipeline")
    const context = createRagContext()
    mocks.ragSearch.mockImplementation((query, options) => chatRagMethods.ragSearch.call(
      { normalizeRagQuery: (value: string) => value } as unknown as ThisParameterType<typeof chatRagMethods.ragSearch>, query, options
    ))
    mocks.bgRequest.mockResolvedValue({ documents: [{
      content: "Larch is the selected source's marker.",
      metadata: { title: "Selected Larch source", type: "text", source: "media_db" }
    }] })
    mocks.stream.mockImplementation(async function* () { yield { content: "The marker is Larch." } })
    mocks.pageAssistModel.mockResolvedValue({ stream: mocks.stream })

    const result = await runChatPipeline(__testing__.ragModeDefinition, context.message, "", false,
      [], [], context.signal, context)

    expect(result).toMatchObject({ status: "submitted" })
    expect(mocks.bgRequest).toHaveBeenCalledTimes(1)
    expect(mocks.bgRequest).toHaveBeenCalledWith(expect.objectContaining({
      path: "/api/v1/rag/search",
      method: "POST",
      body: expect.objectContaining({
        query: context.message,
        enable_generation: false,
        include_media_ids: [7, 8],
        sources: ["media_db"],
        top_k: 8,
        enable_citations: true,
        enable_intent_routing: false
      })
    }))
    expect(mocks.stream).toHaveBeenCalledTimes(1)
    expect(mocks.stream.mock.calls[0][0]).toEqual(expect.arrayContaining([
      expect.objectContaining({ content: expect.arrayContaining([
        expect.objectContaining({ text: expect.stringContaining("Larch is the selected source's marker.") })
      ]) })
    ]))
    expect(context.saveMessageOnSuccess).toHaveBeenCalledWith(expect.objectContaining({
      fullText: "The marker is Larch.",
      source: [expect.objectContaining({ name: "Selected Larch source", mode: "rag" })]
    }))
  })

  it("does not display a stray generated answer when selected-source retrieval has no evidence", async () => {
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
      fullText: expect.stringContaining("couldn't find supporting evidence")
    })
    expect(response?.fullText).toContain("did not send this as general chat")
    expect(response?.fullText).not.toContain("Could you clarify")
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

  it.each([
    ["transport", Object.assign(new Error("Cannot reach server"), { status: 0 })],
    [
      "provider",
      Object.assign(new Error("The selected provider configuration is invalid."), {
        status: 503,
        code: "provider_configuration_invalid"
      })
    ]
  ])("propagates a selected-source %s failure instead of completing an answer", async (_, error) => {
    mocks.ragSearch.mockRejectedValueOnce(error)

    await expect(
      __testing__.ragModeDefinition.preflight?.(createRagContext())
    ).rejects.toBe(error)
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
