import { act, renderHook, waitFor } from "@testing-library/react"
import { createElement, type PropsWithChildren } from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { App } from "antd"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

// Only physical storage/table engines, browser capture, and fetch are replaced.
// The compact consumer, verified account, original Scene save/read, note
// injection/formatter, model factory, API domain, and proxy execute real code.
// No server ACK or raw ID manufactures a saved-history owner or H1 receipt.
const physical = vi.hoisted(() => {
  type Change = { oldValue?: unknown; newValue?: unknown }
  type Watch = Record<string, (change: Change) => void>
  type Read = { area: string; key: string; value: unknown; stack: string }
  type Row = Record<string, unknown>
  type Write = { table: string; row: Row }
  type Collection = {
    toArray: () => Promise<Row[]>
    and: (predicate: (row: Row) => boolean) => Collection
    delete: () => Promise<void>
  }
  const areas = new Map<string, Map<string, unknown>>()
  const watchers = new Map<string, Set<Watch>>()
  const storageListeners = new Set<
    (changes: Record<string, Change>, area: string) => void
  >()
  const control = {
    beforeRead: undefined as ((read: Read) => Promise<unknown>) | undefined,
    afterWrite: undefined as ((write: Write) => Promise<void>) | undefined,
    serverAck: undefined as string | undefined
  }
  const areaFor = (area: string) => {
    if (!areas.has(area)) areas.set(area, new Map())
    return areas.get(area)!
  }
  const publish = (
    area: string,
    key: string,
    oldValue: unknown,
    newValue: unknown
  ) => {
    const change = {
      oldValue: structuredClone(oldValue),
      newValue: structuredClone(newValue)
    }
    for (const subscription of watchers.get(area) ?? [])
      subscription[key]?.(change)
    // Faithful physical onChanged delivery lets real account/epoch observers
    // compare the actual old/new stored configuration, including ABA.
    for (const listener of storageListeners) listener({ [key]: change }, area)
  }
  const storage = (area = "local") => ({
    area,
    hasPersistentBackend: true,
    get: async (key: string) => {
      const read = {
        area,
        key,
        value: structuredClone(areaFor(area).get(key)),
        stack: new Error().stack ?? ""
      }
      return control.beforeRead ? control.beforeRead(read) : read.value
    },
    set: async (key: string, value: unknown) => {
      const rows = areaFor(area),
        oldValue = rows.get(key)
      rows.set(key, structuredClone(value))
      publish(area, key, oldValue, value)
    },
    remove: async (key: string) => {
      const rows = areaFor(area),
        oldValue = rows.get(key)
      rows.delete(key)
      publish(area, key, oldValue, undefined)
    },
    getAll: async () => structuredClone(Object.fromEntries(areaFor(area))),
    watch: (subscription: Watch) => {
      if (!watchers.has(area)) watchers.set(area, new Set())
      watchers.get(area)!.add(subscription)
      return () => {
        watchers.get(area)?.delete(subscription)
      }
    },
    unwatch: (subscription: Watch) => {
      watchers.get(area)?.delete(subscription)
    }
  })
  const tables: Record<string, ReturnType<typeof table>> = {}
  function table(name: string) {
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
          : row.id
      )
    const collection = (matches: (row: Row) => boolean): Collection => ({
      toArray: async () => structuredClone([...rows.values()].filter(matches)),
      and: (predicate) => collection((row) => matches(row) && predicate(row)),
      delete: async () => {
        for (const [id, row] of rows) if (matches(row)) rows.delete(id)
      }
    })
    return {
      rows,
      get: async (id: unknown) => structuredClone(rows.get(JSON.stringify(id))),
      add: async (row: Row) => {
        if (rows.has(key(row))) throw new Error("duplicate")
        rows.set(key(row), structuredClone(row))
        await control.afterWrite?.({ table: name, row: structuredClone(row) })
      },
      put: async (row: Row) => {
        rows.set(key(row), structuredClone(row))
        await control.afterWrite?.({ table: name, row: structuredClone(row) })
      },
      bulkPut: async (values: Row[]) => {
        values.forEach((row) => rows.set(key(row), structuredClone(row)))
      },
      update: async (id: unknown, patch: Row) => {
        const encoded = JSON.stringify(id)
        if (rows.has(encoded))
          rows.set(encoded, { ...rows.get(encoded), ...structuredClone(patch) })
      },
      delete: async (id: unknown) => {
        rows.delete(JSON.stringify(id))
      },
      toArray: async () => structuredClone([...rows.values()]),
      reverse: () => ({
        toArray: async () => structuredClone([...rows.values()].reverse())
      }),
      where: (field: string) => ({
        equals: (value: unknown) => collection((row) => row[field] === value)
      })
    }
  }
  for (const name of [
    "chatHistories",
    "messages",
    "prompts",
    "modelNickname",
    "userSettings",
    "sessionFiles",
    "historySelections",
    "historyProjections",
    "compareStates",
    "forkOperations"
  ])
    tables[name] = table(name)
  let transactionTail = Promise.resolve()
  const db = {
    ...tables,
    transaction: (
      _mode: unknown,
      _tables: unknown,
      operation: (tx: { abort: () => void }) => Promise<unknown>
    ) => {
      const result = transactionTail.then(async () => {
        const copies = Object.values(tables).map((table) => new Map(table.rows))
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
      transactionTail = result.then(
        () => {},
        () => {}
      )
      return result
    }
  }
  return {
    areas,
    areaFor,
    storage,
    storageListeners,
    control,
    tables,
    db,
    fetch: vi.fn(),
    screenshot: vi.fn()
  }
})
vi.mock("wxt/browser", () => ({
  browser: {
    runtime: { id: undefined },
    storage: {
      onChanged: {
        addListener: (
          listener: (changes: Record<string, unknown>, area: string) => void
        ) => physical.storageListeners.add(listener),
        removeListener: (
          listener: (changes: Record<string, unknown>, area: string) => void
        ) => physical.storageListeners.delete(listener)
      }
    }
  }
}))
vi.mock("@/utils/safe-storage", async (original) => ({
  ...(await original<typeof import("@/utils/safe-storage")>()),
  createSafeStorage: (options: { area?: string } = {}) =>
    physical.storage(options.area)
}))
vi.mock("@/db/dexie/schema", () => ({ db: physical.db }))

import { useMessage } from "../useMessage"
import { PageAssistProvider } from "@/components/Common/PageAssistProvider"
import {
  HistorySelectionProvider,
  useHistorySelectionContext
} from "@/hooks/chat/useHistorySelection"
import {
  captureActorSceneSend,
  captureCurrentActorSceneTarget
} from "@/hooks/chat/actor-scene-source"
import {
  getActorSceneForChat,
  saveActorSettingsForChat
} from "@/services/actor-settings"
import { createDefaultActorSettings } from "@/types/actor"
import { useActorStore } from "@/store/actor"
import { useStoreMessageOption } from "@/store/option"
import { useStoreChatModelSettings } from "@/store/model"
import { useMcpToolsStore } from "@/store/mcp-tools"
import { usePageAssist } from "@/context"
import { tldwClient, type TldwConfig } from "@/services/tldw/TldwApiClient"
import { tldwModels } from "@/services/tldw"
import { setTitleGenEnabled } from "@/services/title"
import { deriveSingleUserApiKeyCredentialScope } from "@/services/chat-surface-scope"
import { resolveServicePromptScope } from "@/services/service-prompts"
import {
  normalizeChatSubmitResult,
  type ChatSubmitResult
} from "@/hooks/chat/chat-action-utils"

const originalConfig: TldwConfig = {
  serverUrl: "https://compact-scene-notes.example",
  authMode: "single-user",
  authSource: "manual",
  credentialSource: "manual",
  apiKeyPersistence: "device",
  apiKeyServerOrigin: "https://compact-scene-notes.example",
  apiKey: "original-compact-scene-note-key"
}
const replacementConfig: TldwConfig = {
  ...originalConfig,
  apiKey: "replacement-compact-scene-note-key"
}
const originalNote = "Original frozen compact Scene note"
const laterNote = "Later edited Scene note"
const screenshot = "data:image/png;base64,original-compact-screenshot"
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}
type Wire = {
  url: string
  body: Record<string, unknown>
  headers: Headers
  signal?: AbortSignal | null
}
let wires: Wire[]
let initialCore: ReturnType<typeof useStoreMessageOption.getState>

beforeEach(async () => {
  vi.clearAllMocks()
  physical.control.beforeRead = undefined
  physical.control.afterWrite = undefined
  physical.control.serverAck = undefined
  physical.areas.clear()
  Object.values(physical.tables).forEach((table) => table.rows.clear())
  localStorage.clear()
  sessionStorage.clear()
  physical.areaFor("local").set("tldwConfig", structuredClone(originalConfig))
  for (const area of ["local", "sync"]) {
    physical.areaFor(area).set("selectedModel", "test")
    physical
      .areaFor(area)
      .set("systemPromptForNonRag", "Original Vision default")
    physical
      .areaFor(area)
      .set("copilotSummaryPrompt", "Original summary: {text}")
  }
  wires = []
  physical.fetch.mockImplementation(async (url: string, init: RequestInit) => {
    const path = new URL(url).pathname
    if (path === "/openapi.json")
      return Response.json({
        paths: {
          "/api/v1/characters/{id}": {},
          "/api/v1/chats/": {}
        }
      })
    if (path === "/api/v1/llm/models/metadata")
      return Response.json({
        models: [
          {
            id: "test",
            model: "test",
            name: "Test vision",
            provider: "openai",
            type: "chat",
            capabilities: ["vision"]
          }
        ]
      })
    if (path === "/api/v1/llm/providers")
      return Response.json({
        default_provider: "openai",
        providers: [
          {
            name: "openai",
            default_model: "test",
            is_configured: true,
            enabled: true
          }
        ]
      })
    if (
      physical.control.serverAck &&
      path === "/api/v1/chats/" + physical.control.serverAck
    )
      return Response.json({
        id: physical.control.serverAck,
        title: "Original acknowledged compact chat",
        version: 1,
        state: "in-progress"
      })
    if (path !== "/api/v1/chat/completions")
      throw new Error("Unexpected physical request: " + path)
    wires.push({
      url,
      body: JSON.parse(String(init.body)),
      headers: new Headers(init.headers),
      signal: init.signal
    })
    const receipt = physical.control.serverAck
      ? {
          tldw_conversation_id: physical.control.serverAck,
          tldw_user_message_id: "owned-server-user",
          tldw_message_id: "owned-server-answer"
        }
      : {}
    return new Response(
      "data: " +
        JSON.stringify({
          ...receipt,
          choices: [{ delta: { content: "Original compact answer" } }]
        }) +
        "\n\ndata: [DONE]\n\n",
      { headers: { "Content-Type": "text/event-stream" } }
    )
  })
  vi.stubGlobal("fetch", physical.fetch)
  physical.screenshot.mockImplementation(
    (_window: unknown, _options: unknown, done?: (data: string) => void) => {
      if (done) done(screenshot)
      else return Promise.resolve(screenshot)
    }
  )
  // The real screenshot helper chooses a browser API by build target. Supply
  // both physical callback and Promise shapes rather than replacing the helper.
  vi.stubGlobal("chrome", {
    runtime: {},
    tabs: {
      query: (_query: unknown, done: (tabs: unknown[]) => void) =>
        done([{ id: 19 }]),
      captureVisibleTab: physical.screenshot
    }
  })
  vi.stubGlobal("browser", {
    tabs: {
      query: async () => [{ id: 19 }],
      captureVisibleTab: physical.screenshot
    }
  })
  const tails = new Map<string, Promise<unknown>>()
  Object.defineProperty(navigator, "locks", {
    configurable: true,
    value: {
      request: (key: string, operation: () => unknown) => {
        const next = (tails.get(key) ?? Promise.resolve()).then(operation)
        tails.set(
          key,
          next.catch(() => undefined)
        )
        return next
      }
    }
  })
  initialCore = useStoreMessageOption.getState()
  initialCore.setServerChatId(null)
  useStoreMessageOption.setState({
    historyId: null,
    messages: [],
    history: [],
    selectedModel: "test",
    selectedModelStorageHandled: true,
    chatMode: "normal",
    toolChoice: "none",
    webSearch: false,
    useOCR: false,
    temporaryChat: false,
    streaming: false,
    isProcessing: false,
    isLoading: false,
    isEmbedding: false,
    selectedSystemPrompt: null,
    selectedQuickPrompt: null,
    selectedKnowledge: null,
    fileRetrievalEnabled: false,
    ragMediaIds: null,
    contextFiles: [],
    uploadedFiles: [],
    documentContext: null,
    replyTarget: null,
    compareMode: false,
    compareSelectedModels: [],
    queuedMessages: [],
    messageSteeringMode: "none",
    messageSteeringForceNarrate: false,
    serverChatCharacterId: null,
    serverChatAssistantId: null,
    serverChatAssistantKind: null,
    serverChatPersonaMemoryMode: null,
    serverChatMetaLoaded: false,
    serverChatState: null,
    serverChatVersion: null
  })
  useStoreMessageOption.getState().startNewChatDraft(false)
  useStoreChatModelSettings.getState().reset()
  useStoreChatModelSettings.getState().setApiProvider("openai")
  useMcpToolsStore.setState({ toolsLoading: false, healthState: "unknown" })
  useActorStore.getState().reset()
  await tldwClient.initialize()
  tldwClient.characterCache.clear()
  tldwClient.characterInFlight.clear()
  await tldwModels.clearCache()
  await setTitleGenEnabled(false)
})
afterEach(() => {
  physical.control.beforeRead = undefined
  physical.control.afterWrite = undefined
  useStoreMessageOption.setState(initialCore)
  useActorStore.getState().reset()
  vi.unstubAllGlobals()
})

const mountOriginalScene = async (temporary: boolean, enabled: boolean) => {
  const resolvedAccount = await resolveServicePromptScope()
  expect(resolvedAccount.config).toMatchObject({
    serverUrl: originalConfig.serverUrl,
    authMode: "single-user",
    authSource: "manual",
    expectedSingleUserApiKeyScope: deriveSingleUserApiKeyCredentialScope(
      "single-user",
      originalConfig.apiKey
    )
  })
  expect(resolvedAccount.userId).toBeNull()
  await act(async () => {
    useStoreMessageOption.getState().setTemporaryChat(temporary)
    useStoreMessageOption.getState().startNewChatDraft(temporary)
  })
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  })
  const Wrapper = ({ children }: PropsWithChildren) =>
    createElement(
      QueryClientProvider,
      { client: queryClient },
      createElement(
        App,
        null,
        createElement(
          HistorySelectionProvider,
          null,
          createElement(PageAssistProvider, null, children)
        )
      )
    )
  const mounted = renderHook(
    () => ({
      actions: useMessage(),
      selection: useHistorySelectionContext(),
      context: usePageAssist()
    }),
    { wrapper: Wrapper }
  )
  const unmount = () => {
    mounted.unmount()
    queryClient.clear()
  }
  try {
    await waitFor(() =>
      expect(mounted.result.current.selection?.draftAccount?.isCurrent()).toBe(
        true
      )
    )
    const selection = mounted.result.current.selection!
    expect(selection.draftAccount!.requestScope!.config).toEqual(
      resolvedAccount.config
    )
    expect(selection.getCurrent()).toMatchObject({
      status: "idle",
      owner: null,
      view: null
    })
    const address = useStoreMessageOption.getState().chatDraftAddress
    expect(address).toMatchObject({ kind: "fresh", temporary })
    const target = captureCurrentActorSceneTarget(selection)
    expect(target).not.toBeNull()
    const settings = {
      ...createDefaultActorSettings(),
      isEnabled: enabled,
      notes: originalNote
    }
    await act(async () => {
      expect(
        await saveActorSettingsForChat({
          target: target!,
          settings,
          expectedRecord: null
        })
      ).toBe(true)
    })
    const scene = await getActorSceneForChat({ target: target! })
    expect(scene.originalRecord?.value).toEqual(settings)
    expect(scene.settings).toEqual(settings)
    const captured = captureActorSceneSend(selection)
    expect(captured).not.toBeNull()
    const frozen = await captured!.read()
    expect(frozen.requestScope.config).toEqual(resolvedAccount.config)
    expect(frozen.scene.originalRecord).toEqual(scene.originalRecord)
    expect(frozen.isCurrent()).toBe(true)
    expect(wires).toHaveLength(0)
    return {
      mounted,
      unmount,
      selection,
      address,
      target: target!,
      scene,
      settings,
      frozen
    }
  } catch (error) {
    unmount()
    throw error
  }
}
type Consumer = Awaited<ReturnType<typeof mountOriginalScene>>
type Mode = "vision" | "preset"
const start = (consumer: Consumer, mode: Mode) => {
  let pending!: Promise<ChatSubmitResult>
  act(() => {
    pending = consumer.mounted.result.current.actions.onSubmit(
      mode === "vision"
        ? {
            message: "Original Vision question",
            image: "",
            requestOverrides: { chatMode: "vision" }
          }
        : {
            message: "Original selected passage",
            image: "",
            messageType: "summary"
          }
    )
  })
  return pending
}
const assertOriginalWire = (
  mode: Mode,
  temporary: boolean,
  enabled: boolean
) => {
  expect(wires).toHaveLength(1)
  const wire = wires[0]
  expect(wire.url).toBe(originalConfig.serverUrl + "/api/v1/chat/completions")
  expect(wire.headers.get("X-API-KEY")).toBe(originalConfig.apiKey)
  expect(wire.body.save_to_db).toBe(!temporary)
  expect(wire.body.conversation_id).toBeUndefined()
  const messages = wire.body.messages as { role: string; content: unknown }[]
  const text = JSON.stringify(messages)
  expect(text).toContain(
    mode === "vision"
      ? "Original Vision question"
      : "Original summary: Original selected passage"
  )
  if (mode === "vision") {
    expect(physical.screenshot).toHaveBeenCalledOnce()
    expect(
      messages.find((message) => message.role === "user")?.content
    ).toEqual([
      { type: "text", text: "Original Vision question" },
      { type: "image_url", image_url: { url: screenshot } }
    ])
  } else expect(physical.screenshot).not.toHaveBeenCalled()
  if (enabled) expect(text).toContain("Scene notes: " + originalNote)
  else expect(text).not.toContain(originalNote)
  expect(text).not.toContain(laterNote)
  expect(
    useStoreMessageOption.getState().messages.map((message) => message.message)
  ).toEqual([
    mode === "vision"
      ? "Original Vision question"
      : "Original selected passage",
    "Original compact answer"
  ])
  const saved = [...physical.tables.messages.rows.values()]
  if (temporary) {
    expect(useStoreMessageOption.getState().historyId).toBe("temp")
    expect(saved).toEqual([])
    expect(physical.tables.chatHistories.rows.size).toBe(0)
  } else {
    expect(saved.map((message) => message.content)).toEqual([
      mode === "vision"
        ? "Original Vision question"
        : "Original selected passage",
      "Original compact answer"
    ])
    expect(physical.tables.chatHistories.rows.size).toBe(1)
  }
}
const holdActualNotePreparation = () => {
  const entered = deferred<void>(),
    release = deferred<void>()
  let held = false
  physical.control.beforeRead = async (read) => {
    if (
      !held &&
      read.key === "selectedModel" &&
      /buildActorMessage|maybeInjectActorMessage/.test(read.stack)
    ) {
      held = true
      entered.resolve()
      await release.promise
    }
    return read.value
  }
  return { entered, release }
}
const awaitNoteGate = async (
  entered: Promise<void>,
  pending: Promise<ChatSubmitResult>
) => {
  await Promise.race([
    entered,
    pending.then((result) => {
      throw new Error(
        "The real actor-note formatter was not reached: " +
          JSON.stringify(result)
      )
    })
  ])
}

describe('Compact New preserves the current model/provider pair', () => {
  it('sends a retained bare model through its chosen provider after New', async () => {
    const consumer = await mountOriginalScene(false, false);
    try {
      act(() => {
        useStoreChatModelSettings
          .getState()
          .updateSettings({
            apiProvider: 'custom',
            systemPrompt: 'Old private instructions',
            extraHeaders: 'old headers',
            extraBody: 'old body',
          });
      });
      act(() => expect(consumer.mounted.result.current.actions.clearChat()).toBe(true));
      let result!: ChatSubmitResult;
      await act(async () => {
        result = await consumer.mounted.result.current.actions.onSubmit({
          message: 'Fresh bare-model question',
          image: '',
          messageType: 'summary',
        });
      });
      expect(normalizeChatSubmitResult(result)).toMatchObject({ status: 'submitted' });
      expect(wires).toHaveLength(1);
      expect(wires[0].body).toMatchObject({ model: 'test', api_provider: 'custom' });
      expect(JSON.stringify(wires[0].body.messages)).not.toContain('Old private instructions');
    } finally {
      consumer.unmount();
    }
  });

  it('keeps the execution-time provider after the rendered selection changes', async () => {
    const consumer = await mountOriginalScene(false, false);
    try {
      act(() => {
        useStoreChatModelSettings
          .getState()
          .updateSettings({
            apiProvider: 'custom',
            systemPrompt: 'Prior instructions',
            extraHeaders: 'old headers',
            extraBody: 'old body',
            temperature: 0.2,
          });
      });
      const originalNew = consumer.mounted.result.current.actions.clearChat;
      act(() => {
        useStoreMessageOption.getState().setSelectedModel('anthropic:shared');
        useStoreChatModelSettings.getState().setApiProvider('anthropic');
      });
      act(() => expect(originalNew()).toBe(true));
      expect(useStoreMessageOption.getState().selectedModel).toBe('anthropic:shared');
      expect(useStoreChatModelSettings.getState()).toMatchObject({
        apiProvider: 'anthropic',
        globalSettings: { apiProvider: 'anthropic' },
        systemPrompt: undefined,
        extraHeaders: undefined,
        extraBody: undefined,
        temperature: undefined,
      });
      act(() => {
        useStoreChatModelSettings.getState().setActiveSettingsScope('anthropic:shared');
      });
      expect(useStoreChatModelSettings.getState().apiProvider).toBe('anthropic');
      expect(wires).toHaveLength(0);
    } finally {
      consumer.unmount();
    }
  });
});

describe("compact frozen Scene notes through the actual request", () => {
  for (const temporary of [false, true])
    for (const mode of ["vision", "preset"] as const) {
      const label = (temporary ? "Temporary " : "fresh ") + mode
      for (const enabled of [true, false]) {
        it(
          label +
            " uses the original saved " +
            (enabled ? "enabled" : "disabled") +
            " Scene",
          async () => {
            const consumer = await mountOriginalScene(temporary, enabled)
            try {
              let result!: ChatSubmitResult
              await act(async () => {
                result = await start(consumer, mode)
              })
              expect(normalizeChatSubmitResult(result)).toMatchObject({
                status: "submitted"
              })
              assertOriginalWire(mode, temporary, enabled)
            } finally {
              consumer.unmount()
            }
          }
        )
      }

      it(
        label +
          " keeps the frozen notes when the saved Scene is edited during real note preparation",
        async () => {
          const consumer = await mountOriginalScene(temporary, true)
          const gate = holdActualNotePreparation()
          const pending = start(consumer, mode)
          try {
            await awaitNoteGate(gate.entered.promise, pending)
            expect(wires).toHaveLength(0)
            await act(async () => {
              expect(
                await saveActorSettingsForChat({
                  target: consumer.target,
                  settings: { ...consumer.settings, notes: laterNote },
                  expectedRecord: consumer.scene.originalRecord
                })
              ).toBe(true)
            })
            expect(consumer.frozen.isCurrent()).toBe(true)
            expect(
              (await getActorSceneForChat({ target: consumer.target })).settings
                .notes
            ).toBe(laterNote)
            let result!: ChatSubmitResult
            await act(async () => {
              gate.release.resolve()
              result = await pending
            })
            expect(normalizeChatSubmitResult(result)).toMatchObject({
              status: "submitted"
            })
            assertOriginalWire(mode, temporary, true)
          } finally {
            gate.release.resolve()
            await pending
            consumer.unmount()
          }
        }
      )

      for (const retirement of ["New", "account ABA"] as const) {
        it(
          label +
            " refuses dispatch after " +
            retirement +
            " during real note preparation",
          async () => {
            const consumer = await mountOriginalScene(temporary, true)
            const gate = holdActualNotePreparation()
            const pending = start(consumer, mode)
            try {
              await awaitNoteGate(gate.entered.promise, pending)
              expect(wires).toHaveLength(0)
              if (retirement === "New") {
                await act(async () => {
                  expect(
                    consumer.mounted.result.current.actions.clearChat()
                  ).toBe(true)
                })
                expect(
                  useStoreMessageOption.getState().chatDraftAddress
                ).not.toBe(consumer.address)
              } else {
                await act(async () => {
                  await physical
                    .storage("local")
                    .set("tldwConfig", replacementConfig)
                  await physical
                    .storage("local")
                    .set("tldwConfig", originalConfig)
                })
                expect(physical.areaFor("local").get("tldwConfig")).toEqual(
                  originalConfig
                )
                expect(consumer.target.account!.isCurrent()).toBe(false)
              }
              expect(consumer.frozen.isCurrent()).toBe(false)
              const replacementAddress =
                useStoreMessageOption.getState().chatDraftAddress
              const replacementRevision =
                useStoreMessageOption.getState().chatDraftRevision
              const replacementMessages = structuredClone(
                useStoreMessageOption.getState().messages
              )
              const replacementHistory = structuredClone(
                useStoreMessageOption.getState().history
              )
              if (retirement === "New") expect(replacementMessages).toEqual([])
              let result!: ChatSubmitResult
              await act(async () => {
                gate.release.resolve()
                result = await pending
              })
              expect(normalizeChatSubmitResult(result)).toMatchObject({
                status: "skipped"
              })
              expect(wires).toHaveLength(0)
              expect(physical.tables.messages.rows.size).toBe(0)
              expect(physical.tables.chatHistories.rows.size).toBe(0)
              expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
                replacementAddress
              )
              expect(useStoreMessageOption.getState().chatDraftRevision).toBe(
                replacementRevision
              )
              expect(useStoreMessageOption.getState().messages).toEqual(
                replacementMessages
              )
              expect(useStoreMessageOption.getState().history).toEqual(
                replacementHistory
              )
            } finally {
              gate.release.resolve()
              await pending
              consumer.unmount()
            }
          }
        )
      }
    }
})

describe("compact idle H1 retains exact legacy persistence publication", () => {
  const serverId = "fd6214dc-9b6e-4d65-8d82-2ec757c12001"
  for (const mode of ["vision", "preset"] as const) {
    for (const temporary of [false, true]) {
      it(
        mode +
          " retains the actual " +
          (temporary ? "Temporary" : "local") +
          " history publication without manufacturing a bind",
        async () => {
          const consumer = await mountOriginalScene(temporary, true)
          try {
            let result!: ChatSubmitResult
            await act(async () => {
              result = await start(consumer, mode)
            })
            expect(normalizeChatSubmitResult(result)).toMatchObject({
              status: "submitted"
            })
            assertOriginalWire(mode, temporary, true)
            const rows = [...physical.tables.chatHistories.rows.values()]
            expect(useStoreMessageOption.getState().historyId).toBe(
              temporary ? "temp" : rows[0]?.id
            )
            expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
              consumer.address
            )
            expect(consumer.selection.getCurrent()).toMatchObject({
              status: "idle",
              owner: null,
              view: null
            })
          } finally {
            consumer.unmount()
          }
        }
      )
    }

    for (const delivery of ["local", "server"] as const) {
      const boundaries =
        delivery === "server"
          ? ([
              "current",
              "history-new",
              "history-aba",
              "server-new",
              "server-aba"
            ] as const)
          : (["history-new", "history-aba"] as const)
      for (const boundary of boundaries) {
        it(
          mode +
            " certifies only its " +
            delivery +
            " publication across " +
            boundary,
          async () => {
            const consumer = await mountOriginalScene(false, true)
            const settingsModule = await import("@/services/chat-settings")
            // Observe the real service entry/lease; its original implementation is retained.
            const settingsSync = vi.spyOn(
              settingsModule,
              "syncChatSettingsForServerChat"
            )
            physical.control.serverAck =
              delivery === "server" ? serverId : undefined
            let replacement:
              | ReturnType<typeof useStoreMessageOption.getState>
              | undefined
            let stopSubscriber: (() => void) | undefined
            const source = useStoreMessageOption.getState()
            if (boundary !== "current") {
              const field = boundary.startsWith("history-")
                ? "historyId"
                : "serverChatId"
              stopSubscriber = useStoreMessageOption.subscribe(
                (state, previous) => {
                  const issuedId = state[field]
                  if (replacement || !issuedId || previous[field] === issuedId)
                    return
                  stopSubscriber?.()
                  const ownHistoryId = state.historyId
                  useStoreMessageOption.getState().startNewChatDraft(false)
                  useStoreMessageOption.setState({
                    historyId: boundary.endsWith("aba") ? ownHistoryId : null,
                    serverChatId: boundary === "server-aba" ? issuedId : null,
                    serverChatMetaLoaded: true,
                    serverChatTitle: "Replacement owns its details",
                    messages: [
                      {
                        id: "replacement-row",
                        isBot: false,
                        name: "You",
                        message: "Exact newer draft row",
                        images: [],
                        sources: []
                      }
                    ],
                    history: [
                      { role: "user", content: "Exact newer draft row" }
                    ]
                  })
                  replacement = useStoreMessageOption.getState()
                }
              )
            }
            try {
              let result!: ChatSubmitResult
              await act(async () => {
                result = await start(consumer, mode)
              })
              expect(wires).toHaveLength(1)
              expect(wires[0].body.save_to_db).toBe(true)
              expect(wires[0].body.conversation_id).toBeUndefined()
              expect(JSON.stringify(wires[0].body.messages)).toContain(
                "Scene notes: " + originalNote
              )
              const histories = [...physical.tables.chatHistories.rows.values()]
              const rows = [...physical.tables.messages.rows.values()]
              expect(histories).toHaveLength(1)
              expect(rows.map((row) => row.content)).toEqual([
                mode === "vision"
                  ? "Original Vision question"
                  : "Original selected passage",
                "Original compact answer"
              ])
              if (boundary === "current") {
                expect(normalizeChatSubmitResult(result)).toMatchObject({
                  status: "submitted"
                })
                expect(useStoreMessageOption.getState().historyId).toBe(
                  histories[0].id
                )
                expect(useStoreMessageOption.getState().serverChatId).toBe(
                  serverId
                )
                expect(rows.map((row) => row.serverMessageId)).toEqual([
                  "owned-server-user",
                  "owned-server-answer"
                ])
                expect(settingsSync).toHaveBeenCalledOnce()
                const options = settingsSync.mock.calls[0][0]
                expect(options.sourceDraftAddress).toBe(source.chatDraftAddress)
                expect(options.requestScope?.config).toEqual(
                  consumer.frozen.requestScope.config
                )
                expect(options.validate_lease?.()).toBe(true)
                expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
                  consumer.address
                )
                expect(consumer.selection.getCurrent()).toMatchObject({
                  status: "idle",
                  owner: null,
                  view: null
                })
                act(() => {
                  useStoreMessageOption.getState().startNewChatDraft(false)
                })
                expect(options.validate_lease?.()).toBe(false)
              } else {
                expect(replacement).toBeDefined()
                expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
                  replacement!.chatDraftAddress
                )
                expect(useStoreMessageOption.getState().chatDraftRevision).toBe(
                  replacement!.chatDraftRevision
                )
                expect(useStoreMessageOption.getState().historyId).toBe(
                  replacement!.historyId
                )
                expect(useStoreMessageOption.getState().serverChatId).toBe(
                  replacement!.serverChatId
                )
                expect(
                  useStoreMessageOption.getState().serverChatMetaLoaded
                ).toBe(true)
                expect(useStoreMessageOption.getState().serverChatTitle).toBe(
                  "Replacement owns its details"
                )
                expect(useStoreMessageOption.getState().messages).toEqual(
                  replacement!.messages
                )
                expect(useStoreMessageOption.getState().history).toEqual(
                  replacement!.history
                )
                expect(settingsSync).not.toHaveBeenCalled()
              }
            } finally {
              stopSubscriber?.()
              settingsSync.mockRestore()
              physical.control.serverAck = undefined
              consumer.unmount()
            }
          }
        )
      }
    }
  }
})

// The actual persistence helper and abort-aware transaction wrapper execute.
// The declared table engine serializes/rolls back rows; these controls do not
// certify native IndexedDB transaction scheduling or cross-window behavior.
describe("compact actual post-response persistence retains original account ownership", () => {
  for (const mode of ["vision", "preset"] as const) {
    for (const boundary of [
      "current",
      "New",
      "Stop",
      "account replacement",
      "account ABA"
    ] as const) {
      it(
        mode + " qualifies held actual persistence after " + boundary,
        async () => {
          const consumer = await mountOriginalScene(false, true)
          const entered = deferred<void>(),
            release = deferred<void>()
          let held = false
          physical.control.afterWrite = async (write) => {
            if (
              !held &&
              write.table === "messages" &&
              write.row.role === "assistant" &&
              write.row.content === "Original compact answer"
            ) {
              held = true
              entered.resolve()
              await release.promise
            }
          }
          const pending = start(consumer, mode)
          try {
            await Promise.race([
              entered.promise,
              pending.then((result) => {
                throw new Error(
                  "Actual successful response persistence was not reached: " +
                    JSON.stringify(result)
                )
              })
            ])
            expect(wires).toHaveLength(1)
            expect(wires[0].body.save_to_db).toBe(true)
            expect(JSON.stringify(wires[0].body.messages)).toContain(
              "Scene notes: " + originalNote
            )
            expect(
              [...physical.tables.messages.rows.values()].map(
                (row) => row.content
              )
            ).toEqual([
              mode === "vision"
                ? "Original Vision question"
                : "Original selected passage",
              "Original compact answer"
            ])
            expect(physical.tables.chatHistories.rows.size).toBe(1)
            expect(useStoreMessageOption.getState().historyId).toBeNull()
            expect(consumer.selection.getCurrent()).toMatchObject({
              status: "idle",
              owner: null,
              view: null
            })
            const responseController =
              consumer.mounted.result.current.context.getController?.()
            expect(responseController).toBeInstanceOf(AbortController)
            expect(responseController!.signal.aborted).toBe(false)
            let replacement:
              | ReturnType<typeof useStoreMessageOption.getState>
              | undefined
            if (boundary === "New") {
              await act(async () => {
                expect(
                  consumer.mounted.result.current.actions.clearChat()
                ).toBe(true)
              })
              replacement = useStoreMessageOption.getState()
              expect(replacement.chatDraftAddress).not.toBe(consumer.address)
            } else if (boundary === "Stop") {
              act(() => {
                consumer.mounted.result.current.actions.stopStreamingRequest()
              })
              expect(responseController!.signal.aborted).toBe(true)
              expect(consumer.target.account!.isCurrent()).toBe(true)
            } else if (boundary.startsWith("account")) {
              await act(async () => {
                await physical
                  .storage("local")
                  .set("tldwConfig", replacementConfig)
                if (boundary === "account ABA")
                  await physical
                    .storage("local")
                    .set("tldwConfig", originalConfig)
              })
              expect(consumer.target.account!.isCurrent()).toBe(false)
              if (boundary === "account ABA")
                expect(physical.areaFor("local").get("tldwConfig")).toEqual(
                  originalConfig
                )
            }
            let result!: ChatSubmitResult
            await act(async () => {
              release.resolve()
              result = await pending
            })
            if (boundary.startsWith("account")) {
              expect(physical.tables.messages.rows.size).toBe(0)
              expect(physical.tables.chatHistories.rows.size).toBe(0)
              expect(useStoreMessageOption.getState().historyId).toBeNull()
              expect(useStoreMessageOption.getState().serverChatId).toBeNull()
            } else {
              expect(normalizeChatSubmitResult(result)).toMatchObject({
                status: "submitted"
              })
              const rows = [...physical.tables.messages.rows.values()]
              const histories = [...physical.tables.chatHistories.rows.values()]
              expect(rows.map((row) => row.content)).toEqual([
                mode === "vision"
                  ? "Original Vision question"
                  : "Original selected passage",
                "Original compact answer"
              ])
              expect(histories).toHaveLength(1)
              expect(
                rows.every((row) => row.history_id === histories[0].id)
              ).toBe(true)
              if (replacement) {
                expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
                  replacement.chatDraftAddress
                )
                expect(useStoreMessageOption.getState().chatDraftRevision).toBe(
                  replacement.chatDraftRevision
                )
                expect(useStoreMessageOption.getState().historyId).toBe(
                  replacement.historyId
                )
                expect(useStoreMessageOption.getState().serverChatId).toBe(
                  replacement.serverChatId
                )
                expect(useStoreMessageOption.getState().messages).toEqual(
                  replacement.messages
                )
                expect(useStoreMessageOption.getState().history).toEqual(
                  replacement.history
                )
              } else {
                expect(useStoreMessageOption.getState().historyId).toBe(
                  histories[0].id
                )
                expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
                  consumer.address
                )
              }
            }
            expect(consumer.selection.getCurrent()).toMatchObject({
              status: "idle",
              owner: null,
              view: null
            })
            expect(wires).toHaveLength(1)
          } finally {
            release.resolve()
            await pending
            physical.control.afterWrite = undefined
            consumer.unmount()
          }
        }
      )
    }
  }
})

// This reaches the real first local H1 bind. The subscriber replaces a raw
// display ID during the actual typed-address publication; it does not invoke
// the private identity publisher or grant a different history any authority.
describe("compact real accepted bind retains exact command publication", () => {
  for (const boundary of ["current", "raw history replacement"] as const) {
    it(
      "normal first bind qualifies " +
        boundary +
        " before response publication",
      async () => {
        const consumer = await mountOriginalScene(false, true)
        let replacement:
          | ReturnType<typeof useStoreMessageOption.getState>
          | undefined
        let boundAddress:
          | ReturnType<
              typeof useStoreMessageOption.getState
            >["chatDraftAddress"]
          | undefined
        let unsubscribe: (() => void) | undefined
        unsubscribe = useStoreMessageOption.subscribe((state, previous) => {
          if (
            state.chatDraftAddress?.kind !== "local" ||
            previous.chatDraftAddress?.kind !== "fresh"
          )
            return
          boundAddress = state.chatDraftAddress
          unsubscribe?.()
          if (boundary === "raw history replacement") {
            useStoreMessageOption.setState({
              historyId: "replacement-raw-history",
              serverChatMetaLoaded: true,
              serverChatTitle: "Replacement owns its metadata",
              messages: [
                {
                  id: "replacement-bind-row",
                  isBot: false,
                  name: "You",
                  message: "Exact replacement view row",
                  images: [],
                  sources: []
                }
              ],
              history: [{ role: "user", content: "Exact replacement view row" }]
            })
            replacement = useStoreMessageOption.getState()
          }
        })
        try {
          let result!: ChatSubmitResult
          await act(async () => {
            result = await consumer.mounted.result.current.actions.onSubmit({
              message: "Original normal bind question",
              image: ""
            })
          })
          expect(boundAddress).toMatchObject({ kind: "local" })
          expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
            boundAddress
          )
          const accepted = consumer.selection.getCurrent()
          expect(accepted.status).toBe("ready")
          expect(accepted.owner).toMatchObject({
            kind: "local",
            conversation_id: boundAddress!.id
          })
          expect(accepted.view?.conversation_id).toBe(boundAddress!.id)
          expect(accepted.capture?.status).toBe("captured")
          if (boundary === "current") {
            expect(normalizeChatSubmitResult(result)).toMatchObject({
              status: "submitted"
            })
            expect(useStoreMessageOption.getState().historyId).toBe(
              boundAddress!.id
            )
            expect(wires).toHaveLength(1)
            expect(wires[0].body.save_to_db).toBe(false)
            expect(JSON.stringify(wires[0].body.messages)).toContain(
              "Scene notes: " + originalNote
            )
            expect(
              [...physical.tables.messages.rows.values()].map(
                (row) => row.content
              )
            ).toEqual([
              "Original normal bind question",
              "Original compact answer"
            ])
            const boundTarget = captureCurrentActorSceneTarget(
              consumer.mounted.result.current.selection!
            )
            expect(boundTarget).not.toBeNull()
            const boundScene = await getActorSceneForChat({
              target: boundTarget!
            })
            expect(boundScene.originalRecord?.value).toEqual(
              consumer.scene.originalRecord!.value
            )
            expect(boundScene.settings).toEqual(consumer.settings)
          } else {
            expect(replacement).toBeDefined()
            expect(wires).toHaveLength(0)
            expect(physical.tables.messages.rows.size).toBe(0)
            expect(useStoreMessageOption.getState().historyId).toBe(
              "replacement-raw-history"
            )
            expect(useStoreMessageOption.getState().serverChatId).toBe(
              replacement!.serverChatId
            )
            expect(useStoreMessageOption.getState().serverChatMetaLoaded).toBe(
              true
            )
            expect(useStoreMessageOption.getState().serverChatTitle).toBe(
              "Replacement owns its metadata"
            )
            expect(useStoreMessageOption.getState().messages).toEqual(
              replacement!.messages
            )
            expect(useStoreMessageOption.getState().history).toEqual(
              replacement!.history
            )
          }
        } finally {
          unsubscribe?.()
          consumer.unmount()
        }
      }
    )
  }
})
