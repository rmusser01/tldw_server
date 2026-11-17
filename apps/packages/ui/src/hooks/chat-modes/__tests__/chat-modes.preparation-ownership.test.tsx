// @vitest-environment jsdom
import { act, cleanup, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { ChatTldwOptions } from "@/models/ChatTldw"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
import type { ActorSceneSend } from "@/hooks/chat/actor-scene-source"
import type { Message as StoredMessage } from "@/db/dexie/types"

// Declared table IO only. H1 loading/capture/choice, local user admission,
// settlement, actual mode preparation, formatters and the pipeline stay real.
// This does not qualify native IndexedDB or a browser/provider journey.
const storage = vi.hoisted(() => {
  type Row = Record<string, unknown>
  type Collection = {
    toArray: () => Promise<Row[]>
    and: (predicate: (row: Row) => boolean) => Collection
    delete: () => Promise<void>
  }
  const tables: Record<
    string,
    {
      rows: Map<string, Row>
      get: (id: unknown) => Promise<Row | undefined>
      add: (row: Row) => Promise<void>
      put: (row: Row) => Promise<void>
      bulkPut: (rows: Row[]) => Promise<void>
      update: (id: unknown, patch: Row) => Promise<void>
      delete: (id: unknown) => Promise<void>
      toArray: () => Promise<Row[]>
      where: (field: string) => { equals: (value: unknown) => Collection }
    }
  > = {}
  for (const name of [
    "chatHistories",
    "messages",
    "userSettings",
    "sessionFiles",
    "compareStates",
    "historySelections",
    "historyProjections",
    "forkOperations",
    "modelNickname",
    "prompts",
    "searchTokens"
  ]) {
    const rows = new Map<string, Row>()
    const key = (row: Row) =>
      JSON.stringify(
        name === "historySelections"
          ? [
              row.profile_id,
              row.client_session_id,
              row.owner_key,
              row.conversation_id
            ]
          : name === "searchTokens"
            ? [row.token, row.history_id]
            : row.id
      )
    const collection = (matches: (row: Row) => boolean): Collection => ({
      toArray: async () => structuredClone([...rows.values()].filter(matches)),
      and: (predicate) => collection((row) => matches(row) && predicate(row)),
      delete: async () => {
        for (const [id, row] of rows) if (matches(row)) rows.delete(id)
      }
    })
    tables[name] = {
      rows,
      get: async (id) => structuredClone(rows.get(JSON.stringify(id))),
      add: async (row) => {
        if (rows.has(key(row))) throw new Error("duplicate")
        rows.set(key(row), structuredClone(row))
      },
      put: async (row) => {
        rows.set(key(row), structuredClone(row))
      },
      bulkPut: async (values) => {
        values.forEach((row) => rows.set(key(row), structuredClone(row)))
      },
      update: async (id, patch) => {
        rows.set(JSON.stringify(id), {
          ...rows.get(JSON.stringify(id)),
          ...structuredClone(patch)
        })
      },
      delete: async (id) => {
        rows.delete(JSON.stringify(id))
      },
      toArray: async () => structuredClone([...rows.values()]),
      where: (field) => ({
        equals: (value) => collection((row) => row[field] === value)
      })
    }
  }
  let tail = Promise.resolve()
  return {
    tables,
    db: {
      ...tables,
      transaction: (
        _mode: unknown,
        _tables: unknown,
        operation: (tx: { abort: () => void }) => Promise<unknown>
      ) => {
        const result = tail.then(async () => {
          const copies = Object.values(tables).map(
            (table) => new Map(table.rows)
          )
          try {
            return await operation({
              abort: () => {
                throw new Error("aborted")
              }
            })
          } catch (error) {
            Object.values(tables).forEach((table, index) => {
              table.rows.clear()
              copies[index].forEach((row, key) => table.rows.set(key, row))
            })
            throw error
          }
        })
        tail = result.then(
          () => {},
          () => {}
        )
        return result
      }
    }
  }
})
vi.mock("@/db/dexie/schema", () => ({ db: storage.db }))
const account = vi.hoisted(() => ({
  scope: {
    config: {
      serverUrl: "https://preparation.test",
      authMode: "multi-user" as const
    },
    userId: "alice"
  }
}))
vi.mock("@/services/service-prompts", async (original) => ({
  ...(await original<typeof import("@/services/service-prompts")>()),
  resolveServicePromptScope: async () => structuredClone(account.scope),
  subscribeToServicePromptConfigChanges: () => () => {}
}))
const factory = vi.hoisted(() => ({ before: vi.fn() }))
vi.mock("@/models", async () => {
  const { ChatTldw } = await import("@/models/ChatTldw")
  return {
    pageAssistModel: async (options: ChatTldwOptions) => {
      await factory.before(options)
      return new ChatTldw({
        ...options,
        saveToDb: options.saveToDb ?? false,
        streaming: true,
        apiProvider: "openai",
        supportsMultimodal: true
      })
    }
  }
})

import { normalChatMode } from "../normalChatMode"
import { documentChatMode } from "../documentChatMode"
import { ragMode } from "../ragMode"
import { useHistorySelection } from "@/hooks/chat/useHistorySelection"
import {
  captureChatDraftLoad,
  captureChatDraftSource
} from "@/hooks/chat/chat-draft-transition"
import { captureActorSettingsTarget } from "@/utils/actor-settings-target"
import {
  formatSelectedHistory,
  saveHistory,
  saveMessage
} from "@/db/dexie/helpers"
import { saveMessageOnSuccess, saveMessageOnError } from "@/hooks/chat-helper"
import {
  useStoreMessageOption,
  type ChatHistory,
  type Message
} from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { useStoreChatModelSettings } from "@/store/model"
import { useMcpToolsStore } from "@/store/mcp-tools"
import { createDefaultActorSettings } from "@/types/actor"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { tldwChat } from "@/services/tldw"
import * as systemSettings from "~/services/tldw-server"
import * as searchSettings from "@/services/search"
import * as appSettings from "@/services/app"
import * as modelSettings from "@/services/model-settings"
import * as humanFormatter from "@/utils/human-message"
import { setTitleGenEnabled } from "@/services/title"

const originalInput = "Original auxiliary question\nKeep these exact bytes."
const rewrittenInput = "Rewritten original auxiliary question"
const originalImage =
  "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6nDsAAAAASUVORK5CYII="
const answer = "Answer from the original admitted turn"
const initialCore = useStoreMessageOption.getState()
const initialSession = usePlaygroundSessionStore.getState()
const initialNextData = Object.getOwnPropertyDescriptor(window, "__NEXT_DATA__")
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((accept) => {
    resolve = accept
  })
  return { promise, resolve }
}
const definition = (
  id:
    | "chat.rag.answer"
    | "chat.rag.question_rewrite"
    | "chat.web_search.answer",
  required: string[],
  template: string
) => ({
  definition: {
    id,
    parts: [
      {
        key: "template",
        mode: "template" as const,
        required_variables: required
      }
    ]
  },
  parts: { template },
  source: "user" as const,
  revision: `${id}-revision`
})
const makeSnapshot = (
  execution: AbortSignal,
  invalidated: AbortSignal
): ServicePromptSnapshot => ({
  scopeKey: "original-auxiliary-account",
  requestScope: structuredClone(account.scope),
  capability: "supported",
  scopeSignal: execution,
  scopeInvalidatedSignal: invalidated,
  release: () => {},
  definitions: {
    "chat.rag.answer": definition(
      "chat.rag.answer",
      ["context", "question"],
      "Context:{context}\nQuestion:{question}"
    ),
    "chat.rag.question_rewrite": definition(
      "chat.rag.question_rewrite",
      ["chat_history", "question"],
      "History:{chat_history}\nQuestion:{question}"
    ),
    "chat.web_search.answer": definition(
      "chat.web_search.answer",
      ["current_date_time", "search_results"],
      "Date:{current_date_time}\nResults:{search_results}"
    )
  }
})
type Mode = "web-search" | "document" | "rag" | "selected-rag"
type Stage =
  | "system-settings"
  | "search-settings"
  | "document-settings"
  | "retrieval-settings"
  | "formatter"
  | "rewrite-factory"
  | "initialize"
type Retirement = "New" | "same-ID H1 choice"

beforeEach(async () => {
  vi.restoreAllMocks()
  factory.before.mockReset()
  localStorage.clear()
  sessionStorage.clear()
  Object.values(storage.tables).forEach((table) => table.rows.clear())
  usePlaygroundSessionStore.setState(initialSession)
  useStoreMessageOption.setState({
    ...initialCore,
    messages: [],
    history: [],
    historyId: null,
    serverChatId: null,
    selectedModel: "test",
    toolChoice: "none",
    isProcessing: false,
    streaming: false
  })
  useStoreMessageOption.getState().startNewChatDraft(false)
  useStoreChatModelSettings.getState().reset()
  useMcpToolsStore.setState({ toolsLoading: false, healthState: "unknown" })
  Object.defineProperty(window, "__NEXT_DATA__", {
    configurable: true,
    value: {}
  })
  document.title = "Original auxiliary chat"
  await setTitleGenEnabled(false)
})
afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  useStoreMessageOption.setState(initialCore)
  usePlaygroundSessionStore.setState(initialSession)
  if (initialNextData)
    Object.defineProperty(window, "__NEXT_DATA__", initialNextData)
  else Reflect.deleteProperty(window, "__NEXT_DATA__")
})

const fixture = async (mode: Mode) => {
  const execution = new AbortController()
  const invalidated = new AbortController()
  const stop = new AbortController()
  const snapshot = makeSnapshot(execution.signal, invalidated.signal)
  // H1 qualifies the ordinary source. Switching the current preparation mode
  // does not certify a persisted external-context/RAG history as ordinary H1.
  const history = await saveHistory(
    "Original local H1 chat",
    false,
    "web-ui",
    undefined,
    undefined,
    snapshot.requestScope
  )
  await saveMessage({
    id: "prior-user",
    history_id: history.id,
    role: "user",
    name: "You",
    content: "Prior question",
    images: [],
    parent_message_id: null
  })
  await saveMessage({
    id: "prior-answer",
    history_id: history.id,
    role: "assistant",
    name: "test",
    content: "Prior answer",
    images: [],
    parent_message_id: "prior-user"
  })
  const mounted = renderHook(() => useHistorySelection())
  await act(async () => {
    expect(
      await mounted.result.current.loadConversation({ historyId: history.id })
    ).toBe(true)
  })
  const selected = mounted.result.current.getCurrent()
  if (
    !selected.owner ||
    !selected.view ||
    selected.capture?.status !== "captured"
  )
    throw new Error("fixture requires an actual local H1 capture")
  expect(selected.owner.kind).toBe("local")
  expect(
    captureChatDraftLoad("switch")({
      owner: selected.owner,
      view: selected.view
    })
  ).toBe(true)
  useStoreMessageOption.getState().setHistoryId(history.id)
  const display = formatSelectedHistory(selected.capture)
  useStoreMessageOption.setState({
    history: display.history,
    messages: display.messages
  })
  const selectionFence = mounted.result.current.fence()
  const source = captureChatDraftSource(
    () => selectionFence() && !invalidated.signal.aborted
  )
  const address = useStoreMessageOption.getState().chatDraftAddress
  const target = captureActorSettingsTarget({
    address,
    owner: selected.owner,
    view: selected.view,
    isCurrent: source.isCurrent
  })
  if (!target)
    throw new Error("fixture requires the original local scene address")
  const settings = createDefaultActorSettings()
  // A certified neutral local read is declared here; it grants no native
  // authority. Original core/source and actual H1 epoch guards remain real.
  const actorSceneSend: ActorSceneSend = {
    scene: {
      target,
      originalRecord: null,
      settings,
      origin: "default",
      profileUnavailable: false
    },
    requestScope: snapshot.requestScope,
    isCurrent: source.isCurrent
  }
  const webSearch = vi
    .spyOn(tldwClient, "webSearch")
    .mockResolvedValue({
      web_search_results_dict: {
        results: [
          {
            title: "Original evidence",
            url: "https://evidence.test",
            content: "Evidence for the original input"
          }
        ]
      }
    } as never)
  const ragSearch = vi
    .spyOn(tldwClient, "ragSearch")
    .mockResolvedValue({
      results: [
        {
          content: "Original retrieved evidence",
          metadata: { title: "Original source", type: "text" }
        }
      ]
    } as never)
  const initialize = vi.spyOn(tldwClient, "initialize").mockResolvedValue()
  const rewriteProvider = vi
    .spyOn(tldwChat, "sendMessage")
    .mockResolvedValue(rewrittenInput)
  const mainProvider = vi
    .spyOn(tldwChat, "streamMessage")
    .mockImplementation(async function* () {
      yield answer
    })
  const saveSuccess = vi.fn(saveMessageOnSuccess)
  const saveError = vi.fn(saveMessageOnError)
  let abortOwner: AbortController | null = stop
  const common = {
    selectedModel: "test",
    useOCR: false,
    currentChatModelSettings: {},
    toolChoice: "none" as const,
    historyId: history.id,
    actorSceneSend,
    actorScene: actorSceneSend.scene,
    actorSettings: settings,
    servicePromptSnapshot: snapshot,
    userMessageId: "new-user",
    assistantMessageId: "new-answer",
    setMessages: (next: Message[] | ((previous: Message[]) => Message[])) => {
      if (source.isCurrent()) useStoreMessageOption.getState().setMessages(next)
    },
    setHistory: (next: ChatHistory) => {
      if (source.isCurrent()) useStoreMessageOption.getState().setHistory(next)
    },
    setHistoryId: (id: string) => {
      if (source.isCurrent()) useStoreMessageOption.getState().setHistoryId(id)
    },
    setIsProcessing: (value: boolean) => {
      if (source.isCurrent())
        useStoreMessageOption.getState().setIsProcessing(value)
    },
    setStreaming: (value: boolean) => {
      if (source.isCurrent())
        useStoreMessageOption.getState().setStreaming(value)
    },
    setAbortController: (next: AbortController | null) => {
      if (source.isCurrent()) abortOwner = next
    },
    ownsAbortController: () => abortOwner === stop,
    releaseAbortControllerIfOwned: () => {
      if (abortOwner !== stop) return false
      abortOwner = null
      return true
    },
    saveMessageOnSuccess: saveSuccess,
    saveMessageOnError: saveError
  }
  const send = () => {
    if (mode === "web-search")
      return normalChatMode(
        originalInput,
        originalImage,
        false,
        display.messages,
        display.history,
        stop.signal,
        {
          ...common,
          selectedSystemPrompt: "",
          webSearch: true,
          serverChatId: null,
          historySelection: {
            controller: mounted.result.current,
            originIsCurrent: selectionFence,
            sourceIsCurrent: source.isCurrent
          }
        }
      )
    if (mode === "document")
      return documentChatMode(
        originalInput,
        originalImage,
        false,
        display.messages,
        display.history,
        stop.signal,
        [],
        {
          ...common,
          fileRetrievalEnabled: true,
          setActionInfo: () => {}
        }
      )
    return ragMode(
      originalInput,
      originalImage,
      false,
      display.messages,
      display.history,
      stop.signal,
      {
        ...common,
        selectedKnowledge: null,
        ragMediaIds: mode === "selected-rag" ? [7] : null,
        ragSearchMode: "hybrid",
        ragTopK: null,
        ragEnableGeneration: false,
        ragEnableCitations: true,
        ragSources: ["media_db"]
      }
    )
  }
  const retire = async (boundary: Retirement) => {
    if (boundary === "New") {
      // Retire the original command without revoking its account or aborting
      // provider IO. A turn admitted before this point still owns its result.
      act(() => {
        usePlaygroundSessionStore.getState().markSourceSelectionIntent()
        useStoreMessageOption.getState().startNewChatDraft(false)
      })
      useStoreMessageOption.setState({
        messages: [
          {
            id: "replacement",
            isBot: false,
            name: "You",
            message: "New draft remains",
            sources: []
          }
        ],
        history: [{ role: "user", content: "New draft remains" }]
      })
    } else {
      const original = mounted.result.current.getCurrent().view!
      await act(async () => {
        expect(
          await mounted.result.current.choose({
            kind: "after_message",
            message_id: "prior-answer"
          })
        ).toBe(true)
      })
      const next = mounted.result.current.getCurrent().view!
      expect(next.conversation_id).toBe(original.conversation_id)
      expect(next.selection_revision).toBeGreaterThan(
        original.selection_revision
      )
    }
    expect(source.isCurrent()).toBe(false)
    expect(execution.signal.aborted).toBe(false)
    expect(invalidated.signal.aborted).toBe(false)
    expect(stop.signal.aborted).toBe(false)
  }
  return {
    mode,
    history,
    mounted,
    source,
    snapshot,
    stop,
    display,
    webSearch,
    ragSearch,
    initialize,
    rewriteProvider,
    mainProvider,
    saveSuccess,
    saveError,
    send,
    retire
  }
}
type Fixture = Awaited<ReturnType<typeof fixture>>

const hold = (origin: Fixture, stage: Stage) => {
  const entered = deferred<void>()
  const release = deferred<void>()
  const wait = async () => {
    entered.resolve()
    await release.promise
  }
  if (stage === "system-settings") {
    const actual = systemSettings.systemPromptForNonRagOption
    vi.spyOn(
      systemSettings,
      "systemPromptForNonRagOption"
    ).mockImplementationOnce(async () => {
      await wait()
      return actual()
    })
  } else if (stage === "search-settings") {
    const actual = searchSettings.getSearchSettings
    vi.spyOn(searchSettings, "getSearchSettings").mockImplementationOnce(
      async () => {
        await wait()
        return actual()
      }
    )
  } else if (stage === "document-settings") {
    const actual = modelSettings.getAllDefaultModelSettings
    vi.spyOn(
      modelSettings,
      "getAllDefaultModelSettings"
    ).mockImplementationOnce(async () => {
      await wait()
      return actual()
    })
  } else if (stage === "retrieval-settings") {
    const actual = appSettings.getNoOfRetrievedDocs
    vi.spyOn(appSettings, "getNoOfRetrievedDocs").mockImplementationOnce(
      async () => {
        await wait()
        return actual()
      }
    )
  } else if (stage === "formatter") {
    const actual = humanFormatter.humanMessageFormatter
    vi.spyOn(humanFormatter, "humanMessageFormatter").mockImplementationOnce(
      async (value) => {
        await wait()
        return actual(value)
      }
    )
  } else if (stage === "rewrite-factory") {
    factory.before.mockImplementationOnce(wait)
  } else {
    origin.initialize.mockImplementationOnce(wait)
  }
  return { entered: entered.promise, release: () => release.resolve() }
}
const awaitEntered = async (
  entered: Promise<void>,
  pending: ReturnType<Fixture["send"]>
) => {
  await Promise.race([
    entered,
    pending.then((result) => {
      throw new Error(
        `Preparation returned ${JSON.stringify(result)} before the declared held IO was entered`
      )
    })
  ])
}
const noDispatch = (origin: Fixture) => {
  expect(origin.rewriteProvider).not.toHaveBeenCalled()
  expect(origin.webSearch).not.toHaveBeenCalled()
  expect(origin.ragSearch).not.toHaveBeenCalled()
  expect(origin.mainProvider).not.toHaveBeenCalled()
  expect(origin.saveSuccess).not.toHaveBeenCalled()
  expect(origin.saveError).not.toHaveBeenCalled()
}
const stages = [
  ["web-search", "system-settings"],
  ["web-search", "formatter"],
  ["web-search", "search-settings"],
  ["document", "document-settings"],
  ["document", "retrieval-settings"],
  ["document", "formatter"],
  ["document", "rewrite-factory"],
  ["rag", "rewrite-factory"],
  ["rag", "formatter"],
  ["selected-rag", "initialize"],
  ["selected-rag", "retrieval-settings"]
] as const

describe("actual auxiliary preparation keeps the original command source", () => {
  for (const [mode, stage] of stages) {
    it.each(["New", "same-ID H1 choice"] as const)(
      `${mode} does not dispatch old auxiliary or main input after held ${stage} and %s`,
      async (boundary) => {
        const origin = await fixture(mode)
        const gate = hold(origin, stage)
        const pending = origin.send()
        try {
          await awaitEntered(gate.entered, pending)
          noDispatch(origin)
          await origin.retire(boundary)
          const replacement = useStoreMessageOption.getState()
          gate.release()
          const result = await pending
          noDispatch(origin)
          expect(result.status).toBe("skipped")
          expect(useStoreMessageOption.getState().messages).toEqual(
            replacement.messages
          )
          expect(useStoreMessageOption.getState().history).toEqual(
            replacement.history
          )
          expect(
            (await storage.tables.messages.toArray())
              .map((row) => row.id)
              .sort()
          ).toEqual(["prior-answer", "prior-user"])
        } finally {
          gate.release()
          await pending.catch(() => {})
          origin.mounted.unmount()
        }
      }
    )
  }

  it.each(["web-search", "document", "rag", "selected-rag"] as const)(
    "preserves unchanged-current %s preparation and exact submitted input",
    async (mode) => {
      const origin = await fixture(mode)
      const stage: Stage =
        mode === "web-search"
          ? "search-settings"
          : mode === "document"
            ? "formatter"
            : mode === "rag"
              ? "rewrite-factory"
              : "initialize"
      const gate = hold(origin, stage)
      const pending = origin.send()
      try {
        await awaitEntered(gate.entered, pending)
        expect(origin.source.isCurrent()).toBe(true)
        noDispatch(origin)
        gate.release()
        expect(await pending).toEqual({ status: "submitted" })
        if (mode === "web-search") {
          expect(origin.webSearch).toHaveBeenCalledOnce()
          expect(origin.webSearch.mock.calls[0][0]).toMatchObject({
            query: originalInput,
            requestScope: origin.snapshot.requestScope
          })
          expect(origin.rewriteProvider).not.toHaveBeenCalled()
        } else {
          expect(origin.ragSearch).toHaveBeenCalledOnce()
          expect(origin.ragSearch.mock.calls[0]).toEqual([
            mode === "selected-rag" ? originalInput : rewrittenInput,
            expect.objectContaining({
              enable_generation: false,
              requestScope: origin.snapshot.requestScope
            })
          ])
          if (mode === "selected-rag")
            expect(origin.rewriteProvider).not.toHaveBeenCalled()
          else {
            expect(origin.rewriteProvider).toHaveBeenCalledOnce()
            expect(
              JSON.stringify(origin.rewriteProvider.mock.calls[0][0])
            ).toContain(originalInput.replaceAll("\n", "\\n"))
          }
        }
        expect(origin.mainProvider).toHaveBeenCalledOnce()
        expect(origin.saveSuccess).toHaveBeenCalledWith(
          expect.objectContaining({
            message: originalInput,
            image: originalImage,
            fullText: answer,
            requestScope: origin.snapshot.requestScope
          })
        )
        expect(origin.saveError).not.toHaveBeenCalled()
      } finally {
        gate.release()
        await pending.catch(() => {})
        origin.mounted.unmount()
      }
    }
  )

  it.each(["New", "same-ID H1 choice"] as const)(
    "settles the already admitted original local H1 turn after %s without duplicating its input",
    async (boundary) => {
      const origin = await fixture("web-search")
      const entered = deferred<void>()
      const release = deferred<void>()
      origin.mainProvider.mockImplementationOnce(async function* () {
        entered.resolve()
        await release.promise
        yield answer
      })
      const pending = origin.send()
      try {
        await awaitEntered(entered.promise, pending)
        // This is an actual H1/local-transaction admission, not a fabricated turn.
        expect(await storage.tables.messages.get("new-user")).toMatchObject({
          history_id: origin.history.id,
          role: "user",
          content: originalInput,
          images: [originalImage]
        })
        expect(await storage.tables.messages.get("new-answer")).toBeUndefined()
        expect(origin.mainProvider).toHaveBeenCalledOnce()
        expect(origin.source.isCurrent()).toBe(true)
        await origin.retire(boundary)
        const replacement = useStoreMessageOption.getState()
        release.resolve()
        expect(await pending).toEqual({ status: "submitted" })
        const rows =
          (await storage.tables.messages.toArray()) as StoredMessage[]
        expect(rows.filter((row) => row.id === "new-user")).toHaveLength(1)
        expect(rows.filter((row) => row.id === "new-answer")).toEqual([
          expect.objectContaining({
            history_id: origin.history.id,
            role: "assistant",
            content: answer,
            parent_message_id: "new-user"
          })
        ])
        expect(origin.mainProvider).toHaveBeenCalledOnce()
        expect(origin.saveSuccess).toHaveBeenCalledOnce()
        expect(origin.saveError).not.toHaveBeenCalled()
        expect(useStoreMessageOption.getState().messages).toEqual(
          replacement.messages
        )
        expect(useStoreMessageOption.getState().history).toEqual(
          replacement.history
        )
      } finally {
        release.resolve()
        await pending.catch(() => {})
        origin.mounted.unmount()
      }
    }
  )
})
