import { beforeEach, describe, expect, it, vi } from "vitest"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
import type { ChatHistory, Message } from "~/store/option"

const mocks = vi.hoisted(() => ({
  loadServicePromptSnapshot: vi.fn(),
  runChatPipeline: vi.fn(),
  getAllDefaultModelSettings: vi.fn(),
  getSessionFiles: vi.fn(),
  addFileToSession: vi.fn(),
  getPromptById: vi.fn(),
  humanMessageFormatter: vi.fn(),
  pageAssistModel: vi.fn(),
  promptForRag: vi.fn(),
  getWebSearchPrompt: vi.fn(),
  systemPromptForNonRagOption: vi.fn(),
  generateHistory: vi.fn(),
  removeReasoning: vi.fn(),
  formatDocs: vi.fn(),
  getNoOfRetrievedDocs: vi.fn(),
  getMaxContextSize: vi.fn(),
  getTabContents: vi.fn(),
  maybeInjectActorMessage: vi.fn(),
  systemPromptFormatter: vi.fn(),
  getSearchSettings: vi.fn(),
  tldwInitialize: vi.fn(),
  ragSearch: vi.fn(),
  webSearch: vi.fn(),
  resolveApiProviderForModel: vi.fn()
}))

vi.mock("@/services/service-prompts", async (importOriginal) => {
  const actual = await importOriginal<
    typeof import("@/services/service-prompts")
  >()
  return {
    ...actual,
    loadServicePromptSnapshot: (...args: unknown[]) =>
      mocks.loadServicePromptSnapshot(...args)
  }
})

vi.mock("~/services/tldw-server", async (importOriginal) => {
  const actual = await importOriginal<typeof import("~/services/tldw-server")>()
  return {
    ...actual,
    promptForRag: (...args: unknown[]) => mocks.promptForRag(...args),
    getWebSearchPrompt: (...args: unknown[]) =>
      mocks.getWebSearchPrompt(...args),
    systemPromptForNonRagOption: (...args: unknown[]) =>
      mocks.systemPromptForNonRagOption(...args)
  }
})

vi.mock("../chatModePipeline", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../chatModePipeline")>()
  return {
    ...actual,
    runChatPipeline: (...args: unknown[]) => mocks.runChatPipeline(...args)
  }
})

vi.mock("@/services/model-settings", () => ({
  getAllDefaultModelSettings: (...args: unknown[]) =>
    mocks.getAllDefaultModelSettings(...args)
}))

vi.mock("@/utils/human-message", () => ({
  humanMessageFormatter: (...args: unknown[]) =>
    mocks.humanMessageFormatter(...args)
}))

vi.mock("@/models", () => ({
  pageAssistModel: (...args: unknown[]) => mocks.pageAssistModel(...args)
}))

vi.mock("@/utils/generate-history", () => ({
  generateHistory: (...args: unknown[]) => mocks.generateHistory(...args)
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

vi.mock("@/services/kb", () => ({
  getMaxContextSize: (...args: unknown[]) => mocks.getMaxContextSize(...args)
}))

vi.mock("@/libs/get-tab-contents", () => ({
  getTabContents: (...args: unknown[]) => mocks.getTabContents(...args)
}))

vi.mock("@/utils/actor", () => ({
  maybeInjectActorMessage: (...args: unknown[]) =>
    mocks.maybeInjectActorMessage(...args)
}))

vi.mock("@/utils/system-message", () => ({
  systemPromptFormatter: (...args: unknown[]) =>
    mocks.systemPromptFormatter(...args)
}))

vi.mock("@/services/search", () => ({
  getSearchSettings: (...args: unknown[]) => mocks.getSearchSettings(...args)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: (...args: unknown[]) => mocks.tldwInitialize(...args),
    ragSearch: (...args: unknown[]) => mocks.ragSearch(...args),
    webSearch: (...args: unknown[]) => mocks.webSearch(...args)
  }
}))

vi.mock("@/utils/resolve-api-provider", () => ({
  resolveApiProviderForModel: (...args: unknown[]) =>
    mocks.resolveApiProviderForModel(...args),
  parseProviderQualifiedModelSelection: (value: unknown) => ({
    raw: String(value || ""),
    modelId: String(value || ""),
    provider: undefined,
    isProviderQualified: false
  })
}))

vi.mock("@/db/dexie/helpers", () => {
  return {
    getSessionFiles: (...args: unknown[]) => mocks.getSessionFiles(...args),
    addFileToSession: (...args: unknown[]) => mocks.addFileToSession(...args),
    getPromptById: (...args: unknown[]) => mocks.getPromptById(...args)
  }
})

import { ragMode } from "../ragMode"
import { tabChatMode } from "../tabChatMode"
import { documentChatMode } from "../documentChatMode"
import { normalChatMode } from "../normalChatMode"
import { LEGACY_SERVICE_PROMPT_DEFAULTS } from "~/services/tldw-server"

const requestScope = Object.freeze({
  config: Object.freeze({
    serverUrl: "https://example.test",
    authMode: "single-user" as const,
    authSource: "manual" as const,
    orgId: 7
  }),
  userId: 42
})

const definition = (
  id: "chat.rag.answer" | "chat.rag.question_rewrite" | "chat.web_search.answer",
  requiredVariables: readonly string[],
  template: string
) => ({
  definition: {
    id,
    parts: [
      {
        key: "template",
        mode: "template" as const,
        required_variables: requiredVariables
      }
    ]
  },
  parts: { template },
  source: "user" as const,
  revision: `${id}-revision`
})

const snapshot = (
  ids: readonly string[],
  templates: Partial<
    Record<
      "chat.rag.answer" | "chat.rag.question_rewrite" | "chat.web_search.answer",
      string
    >
  > = {},
  scopeSignal = new AbortController().signal
): ServicePromptSnapshot => ({
  scopeKey: "server:https://example.test|auth:single-user|user:single-user",
  requestScope,
  capability: "supported",
  scopeSignal,
  scopeInvalidatedSignal: new AbortController().signal,
  release: vi.fn(),
  definitions: {
    ...(ids.includes("chat.rag.answer")
      ? {
          "chat.rag.answer": definition(
            "chat.rag.answer",
            ["context", "question"],
            templates["chat.rag.answer"] ??
              "Context: {context}\nQuestion: {question}"
          )
        }
      : {}),
    ...(ids.includes("chat.rag.question_rewrite")
      ? {
          "chat.rag.question_rewrite": definition(
            "chat.rag.question_rewrite",
            ["chat_history", "question"],
            templates["chat.rag.question_rewrite"] ??
              "History: {chat_history}\nQuestion: {question}"
          )
        }
      : {}),
    ...(ids.includes("chat.web_search.answer")
      ? {
          "chat.web_search.answer": definition(
            "chat.web_search.answer",
            ["current_date_time", "search_results"],
            templates["chat.web_search.answer"] ??
              "Now: {current_date_time}\nResults: {search_results}"
          )
        }
      : {})
  }
})

const callbacks = () => ({
  setMessages: vi.fn(),
  saveMessageOnSuccess: vi.fn(async () => null),
  saveMessageOnError: vi.fn(async () => null),
  setHistory: vi.fn(),
  setIsProcessing: vi.fn(),
  setStreaming: vi.fn(),
  setAbortController: vi.fn(),
  historyId: null,
  setHistoryId: vi.fn()
})

const ragParams = (servicePromptSnapshot?: ServicePromptSnapshot) => ({
  ...callbacks(),
  selectedModel: "model-a",
  useOCR: false,
  selectedKnowledge: null,
  currentChatModelSettings: null,
  ragMediaIds: null,
  ragSearchMode: "hybrid" as const,
  ragTopK: null,
  ragEnableGeneration: false,
  ragEnableCitations: false,
  ragSources: [],
  servicePromptSnapshot
})

const tabParams = (servicePromptSnapshot?: ServicePromptSnapshot) => ({
  ...callbacks(),
  selectedModel: "model-a",
  useOCR: false,
  selectedSystemPrompt: "",
  servicePromptSnapshot
})

const documentParams = (servicePromptSnapshot?: ServicePromptSnapshot) => ({
  ...callbacks(),
  selectedModel: "model-a",
  useOCR: false,
  currentChatModelSettings: null,
  fileRetrievalEnabled: true,
  setActionInfo: vi.fn(),
  servicePromptSnapshot
})

const normalParams = (
  webSearch: boolean,
  servicePromptSnapshot?: ServicePromptSnapshot
) => ({
  ...callbacks(),
  selectedModel: "model-a",
  useOCR: false,
  selectedSystemPrompt: "",
  currentChatModelSettings: {},
  webSearch,
  servicePromptSnapshot
})

beforeEach(() => {
    vi.clearAllMocks()
    mocks.runChatPipeline.mockResolvedValue({ status: "submitted" })
    mocks.getAllDefaultModelSettings.mockResolvedValue(undefined)
    mocks.getSessionFiles.mockResolvedValue([])
    mocks.addFileToSession.mockResolvedValue(undefined)
    mocks.getPromptById.mockResolvedValue(null)
    mocks.humanMessageFormatter.mockImplementation(async (input) => input)
    mocks.promptForRag.mockResolvedValue({
      ragPrompt: LEGACY_SERVICE_PROMPT_DEFAULTS["chat.rag.answer"].template,
      ragQuestionPrompt:
        LEGACY_SERVICE_PROMPT_DEFAULTS["chat.rag.question_rewrite"].template
    })
    mocks.getWebSearchPrompt.mockResolvedValue(
      LEGACY_SERVICE_PROMPT_DEFAULTS["chat.web_search.answer"].template
    )
    mocks.systemPromptForNonRagOption.mockResolvedValue(undefined)
    mocks.generateHistory.mockReturnValue([])
    mocks.removeReasoning.mockImplementation((value) => value)
    mocks.formatDocs.mockImplementation((docs) =>
      docs.map((doc: { pageContent?: string }) => doc.pageContent || "").join("\n")
    )
    mocks.getNoOfRetrievedDocs.mockResolvedValue(8)
    mocks.getMaxContextSize.mockResolvedValue(8_000)
    mocks.getTabContents.mockResolvedValue("")
    mocks.maybeInjectActorMessage.mockImplementation(async (history) => history)
    mocks.systemPromptFormatter.mockImplementation(async ({ content }) => ({
      role: "system",
      content
    }))
    mocks.getSearchSettings.mockResolvedValue({
      searchProvider: "google",
      totalSearchResults: 2,
      googleDomain: "google.com"
    })
    mocks.tldwInitialize.mockResolvedValue(undefined)
    mocks.ragSearch.mockResolvedValue({ results: [] })
    mocks.webSearch.mockResolvedValue({
      web_search_results_dict: { results: [] }
    })
  mocks.resolveApiProviderForModel.mockResolvedValue(undefined)
})

describe("Service Prompt mode wrapper ownership", () => {
  it("uses and releases a freshly loaded snapshot lease for the whole pipeline", async () => {
    const scopeController = new AbortController()
    const release = vi.fn()
    const resolved: ServicePromptSnapshot = {
      ...snapshot(["chat.rag.answer", "chat.rag.question_rewrite"]),
      scopeSignal: scopeController.signal,
      release
    }
    mocks.loadServicePromptSnapshot.mockResolvedValue(resolved)
    const ownerSignal = new AbortController().signal
    const releaseAbortControllerIfOwned = vi.fn(() => true)

    await ragMode(
      "question",
      "",
      false,
      [],
      [],
      ownerSignal,
      { ...ragParams(), releaseAbortControllerIfOwned }
    )

    expect(mocks.runChatPipeline.mock.calls[0]?.[6]).toBe(scopeController.signal)
    const pipelineParams = mocks.runChatPipeline.mock.calls[0]?.[7]
    pipelineParams.releaseAbortControllerIfOwned(scopeController.signal)
    expect(releaseAbortControllerIfOwned).toHaveBeenCalledWith(ownerSignal)
    expect(release).toHaveBeenCalledOnce()
  })

  it("uses but does not release a caller-owned snapshot lease", async () => {
    const scopeController = new AbortController()
    const release = vi.fn()
    const supplied: ServicePromptSnapshot = {
      ...snapshot(["chat.rag.answer", "chat.rag.question_rewrite"]),
      scopeSignal: scopeController.signal,
      release
    }

    await ragMode(
      "question",
      "",
      false,
      [],
      [],
      new AbortController().signal,
      ragParams(supplied)
    )

    expect(mocks.runChatPipeline.mock.calls[0]?.[6]).toBe(scopeController.signal)
    expect(release).not.toHaveBeenCalled()
  })

  it("loads the Main RAG answer and rewrite snapshot before entering the pipeline", async () => {
    const resolved = snapshot([
      "chat.rag.answer",
      "chat.rag.question_rewrite"
    ])
    mocks.loadServicePromptSnapshot.mockResolvedValue(resolved)
    const signal = new AbortController().signal

    await ragMode("question", "", false, [], [], signal, ragParams())

    expect(mocks.loadServicePromptSnapshot).toHaveBeenCalledOnce()
    expect(mocks.loadServicePromptSnapshot).toHaveBeenCalledWith(
      ["chat.rag.answer", "chat.rag.question_rewrite"],
      { signal }
    )
    expect(mocks.loadServicePromptSnapshot.mock.invocationCallOrder[0]).toBeLessThan(
      mocks.runChatPipeline.mock.invocationCallOrder[0]
    )
    expect(mocks.runChatPipeline.mock.calls[0]?.[7]).toMatchObject({
      servicePromptSnapshot: resolved
    })
  })

  it("loads only the answer snapshot for Tab Chat", async () => {
    const resolved = snapshot(["chat.rag.answer"])
    mocks.loadServicePromptSnapshot.mockResolvedValue(resolved)
    const signal = new AbortController().signal

    await tabChatMode(
      "question",
      "",
      [],
      false,
      [],
      [],
      signal,
      tabParams()
    )

    expect(mocks.loadServicePromptSnapshot).toHaveBeenCalledWith(
      ["chat.rag.answer"],
      { signal }
    )
    expect(mocks.runChatPipeline.mock.calls[0]?.[7]).toMatchObject({
      servicePromptSnapshot: resolved
    })
  })

  it("loads the Document Chat answer and rewrite snapshot before entering the pipeline", async () => {
    const resolved = snapshot([
      "chat.rag.answer",
      "chat.rag.question_rewrite"
    ])
    mocks.loadServicePromptSnapshot.mockResolvedValue(resolved)
    const signal = new AbortController().signal

    await documentChatMode(
      "question",
      "",
      false,
      [],
      [],
      signal,
      [],
      documentParams()
    )

    expect(mocks.loadServicePromptSnapshot).toHaveBeenCalledOnce()
    expect(mocks.loadServicePromptSnapshot).toHaveBeenCalledWith(
      ["chat.rag.answer", "chat.rag.question_rewrite"],
      { signal }
    )
    expect(mocks.loadServicePromptSnapshot.mock.invocationCallOrder[0]).toBeLessThan(
      mocks.runChatPipeline.mock.invocationCallOrder[0]
    )
    expect(mocks.runChatPipeline.mock.calls[0]?.[7]).toMatchObject({
      servicePromptSnapshot: resolved
    })
  })

  it("forwards new Document session files into the atomic chat save", async () => {
    const resolved = snapshot([
      "chat.rag.answer",
      "chat.rag.question_rewrite"
    ])
    const existingFile = { id: "existing", filename: "existing.txt" }
    const newFile = { id: "new", filename: "new.txt" }
    mocks.getSessionFiles.mockResolvedValueOnce([existingFile])
    const params = {
      ...documentParams(resolved),
      historyId: "history-1"
    }

    await documentChatMode(
      "question",
      "",
      false,
      [],
      [],
      new AbortController().signal,
      [existingFile, newFile] as any,
      params
    )
    const pipelineParams = mocks.runChatPipeline.mock.calls[0]?.[7]
    await pipelineParams.saveMessageOnSuccess({ fullText: "answer" })

    expect(params.saveMessageOnSuccess).toHaveBeenCalledWith(
      expect.objectContaining({
        fullText: "answer",
        sessionFilesToAdd: [newFile]
      })
    )
    expect(mocks.addFileToSession).not.toHaveBeenCalled()
  })

  it("loads the web-search snapshot only when Normal Chat enables web search", async () => {
    const resolved = snapshot(["chat.web_search.answer"])
    mocks.loadServicePromptSnapshot.mockResolvedValue(resolved)
    const signal = new AbortController().signal

    await normalChatMode(
      "question",
      "",
      false,
      [],
      [],
      signal,
      normalParams(true)
    )

    expect(mocks.loadServicePromptSnapshot).toHaveBeenCalledOnce()
    expect(mocks.loadServicePromptSnapshot).toHaveBeenCalledWith(
      ["chat.web_search.answer"],
      { signal }
    )
    expect(mocks.loadServicePromptSnapshot.mock.invocationCallOrder[0]).toBeLessThan(
      mocks.runChatPipeline.mock.invocationCallOrder[0]
    )
    expect(mocks.runChatPipeline.mock.calls[0]?.[7]).toMatchObject({
      servicePromptSnapshot: resolved
    })

    vi.clearAllMocks()
    mocks.runChatPipeline.mockResolvedValue({ status: "submitted" })
    await normalChatMode(
      "ordinary question",
      "",
      false,
      [],
      [],
      signal,
      normalParams(false)
    )

    expect(mocks.loadServicePromptSnapshot).not.toHaveBeenCalled()
  })

  it("reuses a caller-supplied immutable snapshot in every wrapper", async () => {
    const ragSnapshot = snapshot([
      "chat.rag.answer",
      "chat.rag.question_rewrite"
    ])
    const answerSnapshot = snapshot(["chat.rag.answer"])
    const webSnapshot = snapshot(["chat.web_search.answer"])
    const signal = new AbortController().signal

    await ragMode(
      "question",
      "",
      false,
      [],
      [],
      signal,
      ragParams(ragSnapshot)
    )
    await tabChatMode(
      "question",
      "",
      [],
      false,
      [],
      [],
      signal,
      tabParams(answerSnapshot)
    )
    await documentChatMode(
      "question",
      "",
      false,
      [],
      [],
      signal,
      [],
      documentParams(ragSnapshot)
    )
    await normalChatMode(
      "question",
      "",
      false,
      [],
      [],
      signal,
      normalParams(true, webSnapshot)
    )

    expect(mocks.loadServicePromptSnapshot).not.toHaveBeenCalled()
    expect(mocks.runChatPipeline.mock.calls.map((call) => call[7])).toEqual([
      expect.objectContaining({ servicePromptSnapshot: ragSnapshot }),
      expect.objectContaining({ servicePromptSnapshot: answerSnapshot }),
      expect.objectContaining({ servicePromptSnapshot: ragSnapshot }),
      expect.objectContaining({ servicePromptSnapshot: webSnapshot })
    ])
  })

  it("rejects a caller snapshot missing the Main RAG answer before pipeline retrieval", async () => {
    const incomplete = snapshot(["chat.rag.question_rewrite"])
    const params = ragParams(incomplete)
    mocks.ragSearch.mockRejectedValue(new Error("retrieval unavailable"))

    await expect(
      ragMode(
        "question",
        "",
        false,
        [],
        [],
        new AbortController().signal,
        params
      )
    ).rejects.toThrow("chat.rag.answer")

    expect(mocks.runChatPipeline).not.toHaveBeenCalled()
    expect(mocks.ragSearch).not.toHaveBeenCalled()
    expect(params.setMessages).not.toHaveBeenCalled()
  })

  it("rejects a caller snapshot missing the rewrite before selected-source preflight", async () => {
    const incomplete = snapshot(["chat.rag.answer"])
    const params = { ...ragParams(incomplete), ragMediaIds: [42] }

    await expect(
      ragMode(
        "selected-source question",
        "",
        false,
        [],
        [],
        new AbortController().signal,
        params
      )
    ).rejects.toThrow("chat.rag.question_rewrite")

    expect(mocks.runChatPipeline).not.toHaveBeenCalled()
    expect(mocks.ragSearch).not.toHaveBeenCalled()
    expect(params.setMessages).not.toHaveBeenCalled()
  })

  it("validates a freshly loaded Main RAG snapshot before pipeline entry", async () => {
    mocks.loadServicePromptSnapshot.mockResolvedValue(
      snapshot(["chat.rag.answer"])
    )

    await expect(
      ragMode(
        "question",
        "",
        false,
        [],
        [],
        new AbortController().signal,
        ragParams()
      )
    ).rejects.toThrow("chat.rag.question_rewrite")

    expect(mocks.runChatPipeline).not.toHaveBeenCalled()
  })

  it("rejects a caller snapshot missing the Tab answer before pipeline entry", async () => {
    const params = tabParams(snapshot([]))

    await expect(
      tabChatMode(
        "question",
        "",
        [],
        false,
        [],
        [],
        new AbortController().signal,
        params
      )
    ).rejects.toThrow("chat.rag.answer")

    expect(mocks.runChatPipeline).not.toHaveBeenCalled()
    expect(params.setMessages).not.toHaveBeenCalled()
  })

  it.each([
    {
      name: "answer",
      snapshot: snapshot(["chat.rag.question_rewrite"]),
      missingId: "chat.rag.answer"
    },
    {
      name: "rewrite",
      snapshot: snapshot(["chat.rag.answer"]),
      missingId: "chat.rag.question_rewrite"
    }
  ])(
    "rejects a Document snapshot missing $name before model and session work",
    async ({ snapshot: incomplete, missingId }) => {
      const params = {
        ...documentParams(incomplete),
        historyId: "history-1"
      }

      await expect(
        documentChatMode(
          "question",
          "",
          false,
          [],
          [],
          new AbortController().signal,
          [],
          params
        )
      ).rejects.toThrow(missingId)

      expect(mocks.getAllDefaultModelSettings).not.toHaveBeenCalled()
      expect(mocks.getSessionFiles).not.toHaveBeenCalled()
      expect(mocks.runChatPipeline).not.toHaveBeenCalled()
      expect(params.setMessages).not.toHaveBeenCalled()
    }
  )

  it("rejects a web-enabled Normal snapshot missing its answer before image preflight", async () => {
    const params = normalParams(true, snapshot([]))

    await expect(
      normalChatMode(
        "question",
        "",
        false,
        [],
        [],
        new AbortController().signal,
        params
      )
    ).rejects.toThrow("chat.web_search.answer")

    expect(mocks.runChatPipeline).not.toHaveBeenCalled()
    expect(params.setMessages).not.toHaveBeenCalled()
  })

  it("does not read or validate Service Prompts for Normal Chat without web search", async () => {
    const incomplete = snapshot([])

    await normalChatMode(
      "ordinary question",
      "",
      false,
      [],
      [],
      new AbortController().signal,
      normalParams(false, incomplete)
    )

    expect(mocks.loadServicePromptSnapshot).not.toHaveBeenCalled()
    expect(mocks.runChatPipeline).toHaveBeenCalledOnce()
    expect(mocks.runChatPipeline.mock.calls[0]?.[7]).toMatchObject({
      servicePromptSnapshot: incomplete
    })
  })
})

type PreparedPrompt = {
  chatHistory: Array<{ role?: string; content?: unknown }>
  humanMessage?: { content?: Array<{ text?: string }> }
  sources?: unknown[]
}

const previousMessages: Message[] = [
  {
    isBot: false,
    name: "You",
    message: "Earlier {question} $&",
    sources: [],
    images: []
  },
  {
    isBot: true,
    name: "Assistant",
    message: "Earlier answer \\ path",
    sources: [],
    images: []
  }
]

const history: ChatHistory = [
  { role: "user", content: "Earlier {question} $&" },
  { role: "assistant", content: "Earlier answer \\ path" }
]

const useDefinitionPipeline = (runPreflight = false) => {
  mocks.runChatPipeline.mockImplementation(
    async (
      mode,
      message,
      image,
      isRegenerate,
      messages,
      chatHistory,
      signal,
      params
    ) => {
      const context = {
        ...params,
        message,
        image,
        isRegenerate,
        messages,
        history: chatHistory,
        signal,
        createdAt: 1,
        generateMessageId: "assistant-1",
        resolvedUserMessageId: "user-1",
        resolvedAssistantMessageId: "assistant-1",
        resolvedAssistantParentMessageId: "user-1",
        resolvedModelId: params.selectedModel,
        userModelId: params.selectedModel,
        modelInfo: null,
        regenerateVariants: []
      }
      if (runPreflight && mode.preflight) {
        const preflight = await mode.preflight(context)
        if (preflight?.handled) return preflight
      }
      return mode.preparePrompt(context)
    }
  )
}

const promptText = (prepared: PreparedPrompt) =>
  prepared.humanMessage?.content?.[0]?.text

describe("Service Prompt mode definitions", () => {
  it("renders Main RAG once, keeps the original answer question, and locks rewrite tools", async () => {
    useDefinitionPipeline()
    const originalQuestion = "Current {context} $' \\ question"
    const retrievedContext = "Evidence $& {question} \\ context"
    const systemPromptAppendix = "APPENDIX {context} $&"
    const signal = new AbortController().signal
    const resolved = snapshot(
      ["chat.rag.answer", "chat.rag.question_rewrite"],
      {
        "chat.rag.answer": "ANSWER[{context}] QUESTION[{question}]",
        "chat.rag.question_rewrite":
          "REWRITE[{chat_history}] CURRENT[{question}]"
      },
      signal
    )
    const invoke = vi.fn(async () => ({ content: "standalone retrieval query" }))
    mocks.pageAssistModel.mockResolvedValue({ invoke })
    mocks.ragSearch.mockResolvedValue({
      results: [{ content: retrievedContext, metadata: { title: "Source" } }]
    })
    const prepared = (await ragMode(
      originalQuestion,
      "",
      false,
      previousMessages,
      history,
      signal,
      {
        ...ragParams(resolved),
        systemPromptAppendix,
        toolChoice: "auto"
      }
    )) as unknown as PreparedPrompt

    expect(mocks.promptForRag).not.toHaveBeenCalled()
    expect(mocks.loadServicePromptSnapshot).not.toHaveBeenCalled()
    expect(mocks.pageAssistModel).toHaveBeenCalledWith({
      model: "model-a",
      toolChoice: "none",
      tools: [],
      saveToDb: false,
      requestScope
    })
    expect(mocks.humanMessageFormatter.mock.calls[0]?.[0]).toMatchObject({
      content: [
        {
          text:
            "REWRITE[Human: Earlier {question} $&\nAssistant: Earlier answer \\ path] " +
            "CURRENT[Current {context} $' \\ question]"
        }
      ]
    })
    expect(mocks.ragSearch).toHaveBeenCalledWith(
      "standalone retrieval query",
      expect.objectContaining({ signal, requestScope })
    )
    expect(invoke).toHaveBeenCalledWith(
      [expect.anything()],
      { signal }
    )
    expect(promptText(prepared)).toBe(
      "ANSWER[Evidence $& {question} \\ context] " +
        "QUESTION[Current {context} $' \\ question]\n\n" +
        systemPromptAppendix
    )
  })

  it("renders Tab Chat once with the original question and never rewrites", async () => {
    useDefinitionPipeline()
    const originalQuestion = "Tab {context} $' \\ question"
    const tabContext = "Tab evidence $& {question}"
    const appendix = "TAB APPENDIX {question} $&"
    const resolved = snapshot(["chat.rag.answer"], {
      "chat.rag.answer": "TAB[{context}] QUESTION[{question}]"
    })
    mocks.getTabContents.mockResolvedValue(tabContext)

    const prepared = (await tabChatMode(
      originalQuestion,
      "",
      [],
      false,
      previousMessages,
      history,
      new AbortController().signal,
      { ...tabParams(resolved), systemPromptAppendix: appendix }
    )) as unknown as PreparedPrompt

    expect(mocks.promptForRag).not.toHaveBeenCalled()
    expect(mocks.pageAssistModel).not.toHaveBeenCalled()
    expect(promptText(prepared)).toBe(
      `TAB[${tabContext}] QUESTION[${originalQuestion}]\n\n${appendix}`
    )
  })

  it("keeps the packaged Tab image provider message byte-identical", async () => {
    useDefinitionPipeline()
    const question = "Describe this image {question} $&"
    const image = "data:image/png;base64,dGVzdA=="
    const resolved = snapshot(["chat.rag.answer"], {
      "chat.rag.answer":
        LEGACY_SERVICE_PROMPT_DEFAULTS["chat.rag.answer"].template
    })

    await tabChatMode(
      question,
      image,
      [],
      false,
      [],
      [],
      new AbortController().signal,
      tabParams(resolved)
    )

    expect(mocks.humanMessageFormatter).toHaveBeenCalledOnce()
    expect(mocks.humanMessageFormatter).toHaveBeenCalledWith({
      content: [
        { text: question, type: "text" },
        { image_url: image, type: "image_url" }
      ],
      model: "model-a",
      useOCR: false
    })
  })

  it("uses current Document history for a tool-disabled rewrite but the original final question", async () => {
    useDefinitionPipeline()
    const originalQuestion = "Document {context} $' \\ question"
    const retrievedContext = "Document evidence $& {question}"
    const appendix = "DOCUMENT APPENDIX {context}"
    const signal = new AbortController().signal
    const resolved = snapshot(
      ["chat.rag.answer", "chat.rag.question_rewrite"],
      {
        "chat.rag.answer": "DOC[{context}] QUESTION[{question}]",
        "chat.rag.question_rewrite":
          "DOC REWRITE[{chat_history}] CURRENT[{question}]"
      },
      signal
    )
    const invoke = vi.fn(async () => ({ content: "document retrieval query" }))
    mocks.pageAssistModel.mockResolvedValue({ invoke })
    mocks.ragSearch.mockResolvedValue({
      results: [{ content: retrievedContext, metadata: { title: "Document" } }]
    })
    const prepared = (await documentChatMode(
      originalQuestion,
      "",
      false,
      previousMessages,
      history,
      signal,
      [],
      {
        ...documentParams(resolved),
        systemPromptAppendix: appendix,
        toolChoice: "required"
      }
    )) as unknown as PreparedPrompt

    expect(mocks.promptForRag).not.toHaveBeenCalled()
    expect(mocks.pageAssistModel).toHaveBeenCalledWith({
      model: "model-a",
      toolChoice: "none",
      tools: [],
      saveToDb: false,
      requestScope
    })
    expect(mocks.humanMessageFormatter.mock.calls[0]?.[0]).toMatchObject({
      content: [
        {
          text:
            "DOC REWRITE[Human: Earlier {question} $&\n" +
            "Assistant: Earlier answer \\ path] " +
            "CURRENT[Document {context} $' \\ question]"
        }
      ]
    })
    expect(mocks.ragSearch).toHaveBeenCalledWith(
      "document retrieval query",
      expect.objectContaining({
        sources: ["media_db"],
        enable_generation: false,
        signal,
        requestScope
      })
    )
    expect(invoke).toHaveBeenCalledWith(
      [expect.anything()],
      { signal }
    )
    expect(promptText(prepared)).toBe(
      `DOC[${retrievedContext}] QUESTION[${originalQuestion}]\n\n${appendix}`
    )
  })

  it("does not convert selected-source account cancellation into a handled response", async () => {
    useDefinitionPipeline(true)
    const controller = new AbortController()
    const aborted = new DOMException("Aborted", "AbortError")
    mocks.ragSearch.mockImplementation(async () => {
      controller.abort()
      throw aborted
    })

    await expect(ragMode(
      "question",
      "",
      false,
      [],
      [],
      controller.signal,
      {
        ...ragParams(snapshot([
          "chat.rag.answer",
          "chat.rag.question_rewrite"
        ], {}, controller.signal)),
        ragMediaIds: [42]
      }
    )).rejects.toBe(aborted)
  })

  it("renders the web-search ISO timestamp and normalized results in one pass", async () => {
    useDefinitionPipeline()
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-07-16T19:20:21.000Z"))
    const resolved = snapshot(["chat.web_search.answer"], {
      "chat.web_search.answer":
        "WEB NOW<{current_date_time}> RESULTS<{search_results}>"
    })
    mocks.webSearch.mockResolvedValue({
      web_search_results_dict: {
        results: [
          {
            title: "Result title",
            url: "https://example.test/result",
            snippet: "Literal $& {current_date_time} \\ snippet",
            publishedDate: "2026-07-15"
          }
        ]
      }
    })

    try {
      const prepared = (await normalChatMode(
        "search question",
        "",
        false,
        [],
        [],
        new AbortController().signal,
        normalParams(true, resolved)
      )) as unknown as PreparedPrompt

      expect(mocks.getWebSearchPrompt).not.toHaveBeenCalled()
      expect(prepared.chatHistory.at(-1)?.content).toBe(
        "WEB NOW<2026-07-16T19:20:21.000Z> RESULTS<" +
          "Result 1:\n" +
          "Title: Result title\n" +
          "URL: https://example.test/result\n" +
          "Snippet: Literal $& {current_date_time} \\ snippet\n" +
          "Published: 2026-07-15>"
      )
      expect(mocks.webSearch).toHaveBeenCalledWith({
        query: "search question",
        aggregate: false,
        engine: "google",
        result_count: 2,
        google_domain: "google.com",
        signal: expect.any(AbortSignal),
        requestScope
      })
    } finally {
      vi.useRealTimers()
    }
  })

  it("checks web-search prompt availability outside the best-effort search catch", async () => {
    useDefinitionPipeline()

    await expect(
      normalChatMode(
        "search question",
        "",
        false,
        [],
        [],
        new AbortController().signal,
        normalParams(true, snapshot([]))
      )
    ).rejects.toThrow("chat.web_search.answer")

    expect(mocks.webSearch).not.toHaveBeenCalled()
  })

  it("keeps ordinary web-provider failure as a best-effort fallback after prompt resolution", async () => {
    useDefinitionPipeline()
    mocks.webSearch.mockRejectedValue(new Error("provider unavailable"))
    const consoleError = vi.spyOn(console, "error").mockImplementation(() => {})

    const prepared = (await normalChatMode(
      "search question",
      "",
      false,
      [],
      [],
      new AbortController().signal,
      normalParams(true, snapshot(["chat.web_search.answer"]))
    )) as unknown as PreparedPrompt

    expect(prepared.sources).toEqual([])
    expect(consoleError).toHaveBeenCalledWith(
      "Web search failed, continuing without context",
      expect.any(Error)
    )
  })

  it("does not swallow a web-search scope rejection before the signal aborts", async () => {
    useDefinitionPipeline()
    const scopeError = Object.assign(new Error("scope changed"), {
      status: 412,
      details: {
        detail: { code: "request_config_scope_changed" }
      }
    })
    mocks.webSearch.mockRejectedValue(scopeError)

    await expect(normalChatMode(
      "search question",
      "",
      false,
      [],
      [],
      new AbortController().signal,
      normalParams(true, snapshot(["chat.web_search.answer"]))
    )).rejects.toBe(scopeError)
  })

  it("does not replace a document RAG scope rejection with inline context", async () => {
    useDefinitionPipeline()
    const scopeError = Object.assign(new Error("scope changed"), {
      status: 412,
      details: {
        detail: { code: "request_config_scope_changed" }
      }
    })
    mocks.ragSearch.mockRejectedValue(scopeError)

    await expect(documentChatMode(
      "document question",
      "",
      false,
      [],
      [],
      new AbortController().signal,
      [{
        id: "file-1",
        filename: "notes.txt",
        type: "text/plain",
        content: "stale fallback",
        size: 14,
        uploadedAt: 1,
        processed: true
      }],
      documentParams(snapshot([
        "chat.rag.answer",
        "chat.rag.question_rewrite"
      ]))
    )).rejects.toBe(scopeError)
  })

  it("does not convert a selected-source prompt failure into a grounding fallback", async () => {
    useDefinitionPipeline(true)
    mocks.ragSearch.mockResolvedValue({
      results: [{ content: "Selected evidence", metadata: { title: "Source" } }]
    })
    const missingAnswer = snapshot(["chat.rag.question_rewrite"])

    await expect(
      ragMode(
        "selected question",
        "",
        false,
        [],
        [],
        new AbortController().signal,
        { ...ragParams(missingAnswer), ragMediaIds: [42] }
      )
    ).rejects.toThrow("chat.rag.answer")
  })

  it("preserves selected-source retrieval failure as a grounded handled response", async () => {
    useDefinitionPipeline(true)
    mocks.ragSearch.mockRejectedValue(new Error("retrieval unavailable"))

    const handled = (await ragMode(
      "selected question",
      "",
      false,
      [],
      [],
      new AbortController().signal,
      {
        ...ragParams(
          snapshot(["chat.rag.answer", "chat.rag.question_rewrite"])
        ),
        ragMediaIds: [42]
      }
    )) as unknown as { handled: boolean; fullText: string }

    expect(handled).toMatchObject({
      handled: true,
      fullText: expect.stringContaining("couldn't retrieve evidence")
    })
  })

  it("keeps ordinary packaged RAG provider messages byte-identical", async () => {
    useDefinitionPipeline()
    const question = "What is the answer?"
    const context = "Ordinary evidence."
    const ordinaryMessages: Message[] = [
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
    const ordinaryHistory: ChatHistory = [
      { role: "user", content: "Earlier question" },
      { role: "assistant", content: "Earlier answer" }
    ]
    const resolved = snapshot(
      ["chat.rag.answer", "chat.rag.question_rewrite"],
      {
        "chat.rag.answer":
          LEGACY_SERVICE_PROMPT_DEFAULTS["chat.rag.answer"].template,
        "chat.rag.question_rewrite":
          LEGACY_SERVICE_PROMPT_DEFAULTS["chat.rag.question_rewrite"].template
      }
    )
    const invoke = vi.fn(async () => ({ content: "Standalone question" }))
    mocks.pageAssistModel.mockResolvedValue({ invoke })
    mocks.ragSearch.mockResolvedValue({
      results: [{ content: context, metadata: { title: "Source" } }]
    })

    const prepared = (await ragMode(
      question,
      "",
      false,
      ordinaryMessages,
      ordinaryHistory,
      new AbortController().signal,
      ragParams(resolved)
    )) as unknown as PreparedPrompt

    expect(mocks.humanMessageFormatter.mock.calls[0]?.[0]).toMatchObject({
      content: [
        {
          text: LEGACY_SERVICE_PROMPT_DEFAULTS[
            "chat.rag.question_rewrite"
          ].template
            .replace(
              "{chat_history}",
              "Human: Earlier question\nAssistant: Earlier answer"
            )
            .replace("{question}", question)
        }
      ]
    })
    expect(promptText(prepared)).toBe(
      LEGACY_SERVICE_PROMPT_DEFAULTS["chat.rag.answer"].template
        .replace("{context}", context)
        .replace("{question}", question)
    )
  })
})
