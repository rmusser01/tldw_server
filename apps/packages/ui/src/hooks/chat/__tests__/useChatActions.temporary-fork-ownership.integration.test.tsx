import { act, renderHook, waitFor } from "@testing-library/react"
import {
  createElement,
  useCallback,
  useRef,
  useState,
  type PropsWithChildren
} from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { App } from "antd"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

// Only physical storage/table engines and fetch are replaced. The full hook,
// original verified account/Scene, Retry handler, model factory, active API
// domains and transport guards execute production code. Issued SDK IDs remain
// command metadata; they never manufacture a Scene locator or H1 receipt.
const physical = vi.hoisted(() => {
  type Change = { oldValue?: unknown; newValue?: unknown }
  type Watch = Record<string, (change: Change) => void>
  type Read = { area: string; key: string; value: unknown; stack: string }
  type Row = Record<string, unknown>
  type Collection = {
    toArray: () => Promise<Row[]>
    and: (predicate: (row: Row) => boolean) => Collection
    delete: () => Promise<void>
    modify: (patch: Row | ((row: Row) => void)) => Promise<void>
  }
  const areas = new Map<string, Map<string, unknown>>()
  const watchers = new Map<string, Set<Watch>>()
  const storageListeners = new Set<
    (changes: Record<string, Change>, area: string) => void
  >()
  const control = {
    beforeRead: undefined as ((read: Read) => Promise<unknown>) | undefined,
    beforeFetch: undefined as
      | ((url: string, init: RequestInit) => Promise<Response | undefined>)
      | undefined
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
          : name === "sessionFiles"
            ? row.sessionId
            : name === "searchTokens"
              ? [row.token, row.history_id]
              : row.id
      )
    const collection = (matches: (row: Row) => boolean): Collection => ({
      toArray: async () => structuredClone([...rows.values()].filter(matches)),
      and: (predicate) => collection((row) => matches(row) && predicate(row)),
      delete: async () => {
        for (const [id, row] of rows) if (matches(row)) rows.delete(id)
      },
      modify: async (patch) => {
        for (const [id, original] of rows)
          if (matches(original)) {
            const row = structuredClone(original)
            if (typeof patch === "function") patch(row)
            else Object.assign(row, structuredClone(patch))
            rows.set(id, row)
          }
      }
    })
    return {
      rows,
      get: async (id: unknown) => structuredClone(rows.get(JSON.stringify(id))),
      add: async (row: Row) => {
        if (rows.has(key(row))) throw new Error("duplicate")
        rows.set(key(row), structuredClone(row))
      },
      put: async (row: Row) => {
        rows.set(key(row), structuredClone(row))
      },
      bulkPut: async (values: Row[]) => {
        values.forEach((row) => rows.set(key(row), structuredClone(row)))
      },
      bulkGet: async (ids: unknown[]) =>
        ids.map((id) => structuredClone(rows.get(JSON.stringify(id)))),
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
    "forkOperations",
    "searchTokens"
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
    fetch: vi.fn()
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

import { useChatActions } from "../useChatActions"
import { useServerChatHistoryId } from "../useServerChatHistoryId"
import {
  HistorySelectionProvider,
  useHistorySelectionContext
} from "../useHistorySelection"
import {
  captureActorSceneSendForTarget,
  captureCurrentActorSceneTarget,
  type ActorSceneSend
} from "../actor-scene-source"
import { captureChatDraftSource } from "../chat-draft-transition"
import { captureActorSettingsTarget } from "@/utils/actor-settings-target"
import {
  getActorSceneForChat,
  saveActorSettingsForChat
} from "@/services/actor-settings"
import { createDefaultActorSettings } from "@/types/actor"
import { useActorStore } from "@/store/actor"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { useStoreChatModelSettings } from "@/store/model"
import { useMcpToolsStore } from "@/store/mcp-tools"
import { tldwClient, type TldwConfig } from "@/services/tldw/TldwApiClient"
import { tldwModels } from "@/services/tldw"
import { setTitleGenEnabled } from "@/services/title"
import { deriveSingleUserApiKeyCredentialScope } from "@/services/chat-surface-scope"
import { resolveServicePromptScope } from "@/services/service-prompts"
import { serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror"
import {
  CHAT_ROUTE_REPLACEMENT_EVENT,
  type ChatRouteReplacementDetail
} from "@/utils/character-chat-mode-intent"
import {
  normalizeChatSubmitResult,
  type ChatSubmitResult
} from "../chat-action-utils"

const originalConfig: TldwConfig = {
  serverUrl: "https://full-temporary-fork.example",
  authMode: "single-user",
  authSource: "manual",
  credentialSource: "manual",
  apiKeyPersistence: "device",
  apiKeyServerOrigin: "https://full-temporary-fork.example",
  apiKey: "original-full-temporary-fork-key"
}
const replacementConfig: TldwConfig = {
  ...originalConfig,
  serverUrl: "https://replacement-full-temporary-fork.example",
  apiKeyServerOrigin: "https://replacement-full-temporary-fork.example",
  apiKey: "replacement-full-temporary-fork-key"
}
const parentChatId = "original-character-parent"
const childChatId = "original-character-fork"
const originalNote = "Original privately owned full Scene note"
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}
const textResponse = (text: string) =>
  new Response(
    `data: ${JSON.stringify({ choices: [{ delta: { content: text } }] })}\n\ndata: [DONE]\n\n`,
    { headers: { "Content-Type": "text/event-stream" } }
  )
type Wire = {
  url: string
  path: string
  method: string
  body: Record<string, unknown>
  headers: Headers
  signal?: AbortSignal | null
}
let wires: Wire[]
let extraChatModels: string[]
let answer: string
let serverMessageSequence: number
let initialCore: ReturnType<typeof useStoreMessageOption.getState>
let initialSession: ReturnType<typeof usePlaygroundSessionStore.getState>

beforeEach(async () => {
  vi.clearAllMocks()
  physical.control.beforeRead = undefined
  physical.control.beforeFetch = undefined
  physical.areas.clear()
  Object.values(physical.tables).forEach((table) => table.rows.clear())
  localStorage.clear()
  sessionStorage.clear()
  physical.areaFor("local").set("tldwConfig", structuredClone(originalConfig))
  for (const area of ["local", "sync"])
    physical
      .areaFor(area)
      .set("systemPromptForNonRagOption", "Original default prompt")
  wires = []
  extraChatModels = []
  answer = "Original answer"
  serverMessageSequence = 0
  physical.fetch.mockImplementation(
    async (rawUrl: string, init: RequestInit = {}) => {
      const url = String(rawUrl),
        path = new URL(url).pathname,
        method = init.method ?? "GET"
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
              name: "Test chat",
              provider: "openai",
              type: "chat",
              capabilities: []
            },
            ...extraChatModels.map((model) => ({
              id: model,
              model,
              name: model,
              provider: "openai",
              type: "chat",
              capabilities: []
            }))
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
      const wire: Wire = {
        url,
        path,
        method,
        body: init.body ? JSON.parse(String(init.body)) : {},
        headers: new Headers(init.headers),
        signal: init.signal
      }
      wires.push(wire)
      // Record the actual physical handoff before a held response. No semantic
      // helper supplies a dispatch notification, source fence or issued ID.
      const override = await physical.control.beforeFetch?.(url, init)
      if (override) return override
      if (path === "/api/v1/characters/27" || path === "/api/v1/characters/27/")
        return Response.json({
          id: 27,
          name: "Captain Mira",
          system_prompt: "Stay in character.",
          first_message: ""
        })
      if (path === "/api/v1/chats/" && method === "POST")
        return Response.json({
          id: wire.body.parent_conversation_id ? childChatId : parentChatId,
          character_id: 27,
          title: "Captain Mira",
          state: "in-progress",
          version: 1,
          source: "webui-character-chat"
        })
      if (
        path === "/api/v1/chats/" + parentChatId ||
        path === "/api/v1/chats/" + childChatId
      )
        return Response.json({
          id: path.endsWith(childChatId) ? childChatId : parentChatId,
          character_id: 27,
          title: "Captain Mira",
          state: "in-progress",
          version: 1,
          source: "webui-character-chat"
        })
      if (/^\/api\/v1\/chats\/[^/]+\/messages$/.test(path) && method === "POST")
        return Response.json({
          id: "issued-user-or-copy-" + ++serverMessageSequence,
          version: 1
        })
      if (/^\/api\/v1\/chats\/[^/]+\/completions\/persist$/.test(path))
        return Response.json({
          assistant_message_id: "issued-assistant-" + ++serverMessageSequence,
          version: 1
        })
      if (path === "/api/v1/visual-identities/bindings/resolve")
        return Response.json({
          actor_kind: "character",
          actor_id: 27,
          pack_id: null,
          pack_version_id: null,
          expression_key: null,
          requested_expression_key: "neutral",
          asset_id: null,
          storage_relpath: null,
          fallback_reason: "No declared visual identity binding",
          is_animated: false,
          content_type: null,
          asset_url: null,
          preview_url: null
        })
      // The declared legacy server exposes no chat-settings endpoint. Preserve
      // the production best-effort compatibility fallback after the real call.
      if (/^\/api\/v1\/chats\/[^/]+\/settings$/.test(path))
        return Response.json({ detail: "Not Found" }, { status: 404 })
      if (
        path === "/api/v1/chat/completions" ||
        /^\/api\/v1\/chats\/[^/]+\/complete-v2$/.test(path)
      )
        return textResponse(answer)
      throw new Error("Unexpected physical request: " + method + " " + path)
    }
  )
  vi.stubGlobal("fetch", physical.fetch)
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
  initialSession = usePlaygroundSessionStore.getState()
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
    temporaryChat: true,
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
    serverChatTitle: null,
    serverChatCharacterId: null,
    serverChatAssistantId: null,
    serverChatAssistantKind: null,
    serverChatPersonaMemoryMode: null,
    serverChatMetaLoaded: false,
    serverChatState: null,
    serverChatVersion: null,
    serverChatTopic: null,
    serverChatClusterId: null,
    serverChatSource: null,
    serverChatExternalRef: null
  })
  useStoreMessageOption.getState().startNewChatDraft(true)
  usePlaygroundSessionStore.getState().clearSession()
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
  physical.control.beforeFetch = undefined
  useStoreMessageOption.setState(initialCore)
  usePlaygroundSessionStore.setState(initialSession)
  useActorStore.getState().reset()
  vi.unstubAllGlobals()
})

const makeWrapper = (h1: boolean) => {
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
        h1 ? createElement(HistorySelectionProvider, null, children) : children
      )
    )
  return { Wrapper, queryClient }
}
const mountFull = (h1 = true, character = false) => {
  const { Wrapper, queryClient } = makeWrapper(h1)
  const mounted = renderHook(
    () => {
      const core = useStoreMessageOption(),
        settings = useStoreChatModelSettings()
      const selection = useHistorySelectionContext()
      const [abortController, updateAbortController] =
        useState<AbortController | null>(null)
      const controllerRef = useRef<AbortController | null>(null)
      const setAbortController = useCallback((next: AbortController | null) => {
        controllerRef.current = next
        updateAbortController(next)
      }, [])
      type Options = Parameters<typeof useChatActions>[0]
      const t = ((_key: string, fallback?: unknown) =>
        typeof fallback === "string" ? fallback : _key) as Options["t"]
      const { ensureServerChatHistoryId } = useServerChatHistoryId({
        serverChatId: core.serverChatId,
        historyId: core.historyId,
        setHistoryId: core.setHistoryId,
        temporaryChat: core.temporaryChat,
        t
      })
      const actions = useChatActions({
        ...core,
        t,
        notification: {
          error: vi.fn(),
          warning: vi.fn(),
          info: vi.fn(),
          success: vi.fn()
        } as unknown as Options["notification"],
        abortController,
        setAbortController,
        getAbortController: () => controllerRef.current,
        currentChatModelSettings: settings,
        compareModeActive: false,
        compareMaxModels: 3,
        compareFeatureEnabled: true,
        markCompareHistoryCreated: vi.fn(),
        ensureServerChatHistoryId,
        messageSteeringPrompts: null,
        invalidateServerChatHistory: vi.fn(),
        selectedCharacter: null,
        selectedAssistant: character
          ? {
              kind: "character",
              id: "27",
              name: "Captain Mira",
              metadata: { selectionMode: "tracked" }
            }
          : null
      })
      return {
        actions,
        selection,
        getSignal: () => controllerRef.current?.signal
      }
    },
    { wrapper: Wrapper }
  )
  return {
    mounted,
    unmount: () => {
      mounted.unmount()
      queryClient.clear()
    }
  }
}
const mountOriginalAccount = async () => {
  const { Wrapper, queryClient } = makeWrapper(true)
  const mounted = renderHook(() => useHistorySelectionContext(), {
    wrapper: Wrapper
  })
  try {
    await waitFor(() =>
      expect(mounted.result.current?.draftAccount?.isCurrent()).toBe(true)
    )
    return {
      mounted,
      unmount: () => {
        mounted.unmount()
        queryClient.clear()
      }
    }
  } catch (error) {
    mounted.unmount()
    queryClient.clear()
    throw error
  }
}
const prepareScene = async (
  selection: NonNullable<ReturnType<typeof useHistorySelectionContext>>,
  enabled: boolean
) => {
  const resolved = await resolveServicePromptScope()
  const account = selection.draftAccount!
  expect(account.isCurrent()).toBe(true)
  expect(account.requestScope).toEqual(resolved)
  expect(account.requestScope.config.expectedSingleUserApiKeyScope).toBe(
    deriveSingleUserApiKeyCredentialScope("single-user", originalConfig.apiKey)
  )
  const address = useStoreMessageOption.getState().chatDraftAddress
  const revision = useStoreMessageOption.getState().chatDraftRevision
  const session = usePlaygroundSessionStore.getState()
  const viewIsCurrent = selection.fence()
  const originalCommandIsCurrent = () => {
    const core = useStoreMessageOption.getState(),
      currentSession = usePlaygroundSessionStore.getState()
    return (
      account.isCurrent() &&
      viewIsCurrent() &&
      core.chatDraftAddress === address &&
      core.chatDraftRevision === revision &&
      currentSession.restoreRevision === session.restoreRevision &&
      currentSession.sourceSelectionRevision === session.sourceSelectionRevision
    )
  }
  const target = captureCurrentActorSceneTarget(selection)
  expect(target).not.toBeNull()
  const settings = {
    ...createDefaultActorSettings(),
    isEnabled: enabled,
    notes: originalNote
  }
  expect(
    await saveActorSettingsForChat({
      target: target!,
      settings,
      historyId: null,
      serverChatId: null,
      expectedRecord: null
    })
  ).toBe(true)
  const original = await getActorSceneForChat({ target: target! })
  expect(original.originalRecord?.value).toEqual(settings)
  const capture = async (): Promise<ActorSceneSend> => {
    // Keep the same qualified private draft locator through its actual SDK
    // seed. The issued parent ID is metadata, never the Scene target or owner.
    const source = captureChatDraftSource(originalCommandIsCurrent)
    const commandTarget = captureActorSettingsTarget({
      address,
      requestScope: account.requestScope,
      accountIsCurrent: account.isCurrent,
      isCurrent: source.isCurrent
    })
    const captured = captureActorSceneSendForTarget(
      commandTarget,
      account,
      source.isCurrent,
      originalCommandIsCurrent
    )
    expect(captured?.isCurrent()).toBe(true)
    const actorSceneSend = await captured!.read()
    expect(actorSceneSend.scene.target.key).toBe(original.target.key)
    expect(actorSceneSend.scene.originalRecord).toEqual(original.originalRecord)
    expect(actorSceneSend.requestScope).toEqual(account.requestScope)
    return actorSceneSend
  }
  return {
    account,
    address,
    revision,
    original,
    capture,
    originalCommandIsCurrent
  }
}
const mountTemporary = async () => {
  const consumer = mountFull()
  try {
    await waitFor(() =>
      expect(
        consumer.mounted.result.current.selection?.draftAccount?.isCurrent()
      ).toBe(true)
    )
    const scene = await prepareScene(
      consumer.mounted.result.current.selection!,
      true
    )
    expect(useStoreMessageOption.getState()).toMatchObject({
      historyId: null,
      serverChatId: null,
      temporaryChat: true
    })
    return { ...consumer, scene }
  } catch (error) {
    consumer.unmount()
    throw error
  }
}
const steeringTuple = () => {
  const state = useStoreMessageOption.getState()
  return {
    mode: state.messageSteeringMode,
    forceNarrate: state.messageSteeringForceNarrate,
    revision: state.messageSteeringRevision
  }
}
const armSteering = async () => {
  await act(async () => {
    const core = useStoreMessageOption.getState()
    core.setMessageSteeringMode("continue_as_user")
    core.setMessageSteeringForceNarrate(true)
  })
  return steeringTuple()
}
const presentation = () => {
  const core = useStoreMessageOption.getState()
  return {
    address: core.chatDraftAddress,
    revision: core.chatDraftRevision,
    historyId: core.historyId,
    serverChatId: core.serverChatId,
    title: core.serverChatTitle,
    state: core.serverChatState,
    version: core.serverChatVersion,
    character: core.serverChatCharacterId,
    history: structuredClone(core.history),
    messages: structuredClone(core.messages)
  }
}
const replacePresentation = (
  historyId: string | null = "temp",
  serverChatId: string | null = null
) => {
  const core = useStoreMessageOption.getState()
  core.startNewChatDraft(serverChatId === null)
  core.setServerChatId(serverChatId)
  core.setHistoryId(historyId, { preserveServerChatId: true })
  core.setServerChatTitle("Later selected title")
  core.setServerChatState("completed")
  core.setServerChatVersion(93)
  core.setServerChatCharacterId(null)
  core.setHistory([
    { role: "user", content: "Later selected question", image: "" },
    { role: "assistant", content: "Later selected answer" }
  ])
  core.setMessages([
    {
      id: "later-user",
      isBot: false,
      role: "user",
      name: "You",
      message: "Later selected question"
    },
    {
      id: "later-answer",
      isBot: true,
      role: "assistant",
      name: "test",
      message: "Later selected answer"
    }
  ])
  return presentation()
}
const complete = async (pending: Promise<ChatSubmitResult | void>) => {
  let result!: ChatSubmitResult | void
  await act(async () => {
    result = await pending
  })
  return normalizeChatSubmitResult(result)
}
const mainWires = () =>
  wires.filter(
    (wire) =>
      wire.path === "/api/v1/chat/completions" ||
      wire.path.endsWith("/complete-v2")
  )
const assertOriginalWire = (wire: Wire) => {
  expect(new URL(wire.url).origin).toBe(originalConfig.serverUrl)
  expect(wire.headers.get("X-API-KEY")).toBe(originalConfig.apiKey)
  expect(wire.signal?.aborted).toBe(false)
}
const beginTemporarySend = (
  consumer: Awaited<ReturnType<typeof mountTemporary>>
) => {
  let pending!: Promise<ChatSubmitResult>
  act(() => {
    pending = consumer.mounted.result.current.actions.onSubmit({
      message: "Original Temporary question",
      image: ""
    })
  })
  return pending
}
const awaitGate = async (
  entered: Promise<unknown>,
  pending: Promise<ChatSubmitResult | void>
) => {
  await Promise.race([
    entered,
    pending.then((result) => {
      throw new Error(
        "The declared physical gate was not reached: " + JSON.stringify(result)
      )
    })
  ])
}

describe("full ordinary Temporary Send retains its original intent through its own sentinel", () => {
  it("consumes original one-use steering after its first successful physical response and own temp publication", async () => {
    const consumer = await mountTemporary()
    const originalSteering = await armSteering()
    const entered = deferred<Wire>(),
      held = deferred<Response>()
    let pending: Promise<ChatSubmitResult> | undefined
    physical.control.beforeFetch = async (url, _init) => {
      if (new URL(url).pathname !== "/api/v1/chat/completions") return undefined
      entered.resolve(wires.at(-1)!)
      return held.promise
    }
    try {
      pending = beginTemporarySend(consumer)
      await awaitGate(entered.promise, pending)
      const wire = await entered.promise
      assertOriginalWire(wire)
      expect(wire.body.save_to_db).toBe(false)
      expect(JSON.stringify(wire.body.messages)).toContain(originalNote)
      expect(JSON.stringify(wire.body.messages)).toContain(
        "Continue the user's current thought"
      )
      expect(JSON.stringify(wire.body.messages)).toContain(
        "Use narrative prose style"
      )
      expect(steeringTuple()).toEqual(originalSteering)
      held.resolve(textResponse("Original answer"))
      expect(await complete(pending)).toMatchObject({ status: "submitted" })
      expect(useStoreMessageOption.getState()).toMatchObject({
        historyId: "temp",
        serverChatId: null,
        chatDraftAddress: consumer.scene.address,
        chatDraftRevision: consumer.scene.revision
      })
      expect(
        useStoreMessageOption
          .getState()
          .messages.map((message) => message.message)
      ).toEqual(["Original Temporary question", "Original answer"])
      expect(steeringTuple()).toEqual({
        mode: "none",
        forceNarrate: false,
        revision: originalSteering.revision + 1
      })
      expect((await consumer.scene.capture()).scene.originalRecord).toEqual(
        consumer.scene.original.originalRecord
      )
      expect(
        consumer.mounted.result.current.selection!.getCurrent()
      ).toMatchObject({ status: "idle", owner: null })
      expect(physical.tables.chatHistories.rows.size).toBe(0)
      expect(physical.tables.messages.rows.size).toBe(0)
    } finally {
      held.resolve(textResponse("Original answer"))
      if (pending) await complete(pending)
      consumer.unmount()
    }
  })

  it.each([null, "temp"] as const)(
    "an old first response cannot consume or publish into New with later history ID %s",
    async (laterHistoryId) => {
      const consumer = await mountTemporary()
      const originalSteering = await armSteering()
      const entered = deferred<Wire>(),
        held = deferred<Response>()
      let pending: Promise<ChatSubmitResult> | undefined
      physical.control.beforeFetch = async (url, _init) => {
        if (new URL(url).pathname !== "/api/v1/chat/completions")
          return undefined
        entered.resolve(wires.at(-1)!)
        return held.promise
      }
      try {
        pending = beginTemporarySend(consumer)
        await awaitGate(entered.promise, pending)
        assertOriginalWire(await entered.promise)
        expect(steeringTuple()).toEqual(originalSteering)
        let later!: ReturnType<typeof presentation>
        act(() => {
          later = replacePresentation(laterHistoryId)
        })
        const laterSteering = await armSteering()
        expect(later.address).not.toBe(consumer.scene.address)
        expect(later.revision).toBeGreaterThan(consumer.scene.revision)
        held.resolve(textResponse("Old completed Temporary answer"))
        await complete(pending)
        expect(presentation()).toEqual(later)
        expect(steeringTuple()).toEqual(laterSteering)
        expect(mainWires()).toHaveLength(1)
        expect(physical.tables.chatHistories.rows.size).toBe(0)
      } finally {
        held.resolve(textResponse("Old completed Temporary answer"))
        if (pending) await complete(pending)
        consumer.unmount()
      }
    }
  )
})

const seedTemporary = async (
  consumer: Awaited<ReturnType<typeof mountTemporary>>
) => {
  let result!: ChatSubmitResult
  await act(async () => {
    result = await consumer.mounted.result.current.actions.onSubmit({
      message: "Original Retry question",
      image: ""
    })
  })
  expect(normalizeChatSubmitResult(result)).toMatchObject({
    status: "submitted"
  })
  expect(mainWires()).toHaveLength(1)
  assertOriginalWire(mainWires()[0])
  expect(
    useStoreMessageOption.getState().messages.map((message) => message.message)
  ).toEqual(["Original Retry question", "Original answer"])
  expect(useStoreMessageOption.getState()).toMatchObject({
    historyId: "temp",
    serverChatId: null
  })
  expect(consumer.mounted.result.current.selection!.getCurrent()).toMatchObject(
    { status: "idle", owner: null }
  )
  wires.length = 0
}

describe("the public no-dispatch full Retry captures its original supported Temporary source", () => {
  it("regenerates the unchanged original Temporary question using the actual Scene/factory/transport", async () => {
    const consumer = await mountTemporary()
    try {
      await seedTemporary(consumer)
      answer = "Original retry answer"
      let pending!: Promise<ChatSubmitResult | void>
      act(() => {
        pending =
          consumer.mounted.result.current.actions.regenerateLastMessage()
      })
      expect(await complete(pending)).toMatchObject({ status: "submitted" })
      expect(mainWires()).toHaveLength(1)
      assertOriginalWire(mainWires()[0])
      expect(JSON.stringify(mainWires()[0].body.messages)).toContain(
        "Original Retry question"
      )
      expect(JSON.stringify(mainWires()[0].body.messages)).toContain(
        originalNote
      )
      expect(
        useStoreMessageOption
          .getState()
          .messages.filter((message) => !message.isBot)
      ).toHaveLength(1)
      expect(useStoreMessageOption.getState().messages.at(-1)?.message).toBe(
        "Original retry answer"
      )
      expect((await consumer.scene.capture()).scene.originalRecord).toEqual(
        consumer.scene.original.originalRecord
      )
      expect(physical.tables.chatHistories.rows.size).toBe(0)
    } finally {
      consumer.unmount()
    }
  })

  it("New in the real beforeSubmit microtask prevents old public Retry from mutating or recapturing the later Temporary Scene", async () => {
    const consumer = await mountTemporary()
    try {
      await seedTemporary(consumer)
      let pending!: Promise<ChatSubmitResult | void>
      let later!: ReturnType<typeof presentation>
      act(() => {
        pending =
          consumer.mounted.result.current.actions.regenerateLastMessage()
      })
      await act(async () => {
        // The public wrapper's real native-retry await resumes first. Its
        // ordinary handler then reaches the production async beforeSubmit
        // no-fork return. New runs before that second continuation; neither
        // helper nor original Scene/source capability is replaced by a double.
        await Promise.resolve()
        later = replacePresentation("temp")
      })
      const result = await complete(pending)
      expect(result.status).toBe("skipped")
      expect(mainWires()).toEqual([])
      expect(presentation()).toEqual(later)
      expect(later.address).not.toBe(consumer.scene.address)
      expect(physical.tables.chatHistories.rows.size).toBe(0)
    } finally {
      consumer.unmount()
    }
  })
})

// Hold the actual private Scene bucket read in the public no-dispatch Retry.
// Catalog replies only declare a second usable model; no model factory, Scene
// helper, dispatch marker, source guard or owner capability is substituted.
const startPublicRetryAtSceneRead = async () => {
  extraChatModels = ["replacement-test"]
  const consumer = await mountTemporary()
  let pending: Promise<ChatSubmitResult | void> | undefined
  const held = deferred<unknown>()
  try {
    await act(async () => {
      useStoreChatModelSettings.getState().setTemperature(0.24)
    })
    await seedTemporary(consumer)
    const originalSteering = await armSteering()
    const originalPresentation = presentation()
    const originalSettings = structuredClone(
      useStoreChatModelSettings.getState().getEffectiveSettings()
    )
    const originalModel = useStoreMessageOption.getState().selectedModel
    const originalToolChoice = useStoreMessageOption.getState().toolChoice
    const entered =
      deferred<Parameters<NonNullable<typeof physical.control.beforeRead>>[0]>()
    const storageKey =
      "actorSettings:owned:" + consumer.scene.original.target.key
    physical.control.beforeRead = async (read) => {
      if (
        read.area !== "local" ||
        read.key !== storageKey ||
        !read.stack.includes("readOwnedRecord")
      )
        return read.value
      physical.control.beforeRead = undefined
      entered.resolve(read)
      return held.promise
    }
    answer = "Original Scene-held retry answer"
    act(() => {
      pending = consumer.mounted.result.current.actions.regenerateLastMessage()
    })
    await awaitGate(entered.promise, pending!)
    const read = await entered.promise
    expect(read.stack).toContain("getActorSceneForChat")
    expect(read.value).toEqual(consumer.scene.original.originalRecord)
    expect(presentation()).toEqual(originalPresentation)
    expect(mainWires()).toEqual([])
    expect(consumer.scene.account.isCurrent()).toBe(true)
    return {
      consumer,
      pending: pending!,
      held,
      read,
      originalPresentation,
      originalSteering,
      originalSettings,
      originalModel,
      originalToolChoice
    }
  } catch (error) {
    physical.control.beforeRead = undefined
    held.resolve(consumer.scene.original.originalRecord)
    if (pending) await complete(pending)
    consumer.unmount()
    throw error
  }
}

describe("public full Temporary Retry retains entry model, tool and settings through the actual Scene read", () => {
  it("dispatches the unchanged original intent after the real held Scene read", async () => {
    const turn = await startPublicRetryAtSceneRead()
    try {
      expect(useStoreMessageOption.getState().selectedModel).toBe(
        turn.originalModel
      )
      expect(useStoreMessageOption.getState().toolChoice).toBe(
        turn.originalToolChoice
      )
      expect(
        useStoreChatModelSettings.getState().getEffectiveSettings()
      ).toEqual(turn.originalSettings)
      turn.held.resolve(turn.read.value)
      expect(await complete(turn.pending)).toMatchObject({
        status: "submitted"
      })
      expect(mainWires()).toHaveLength(1)
      assertOriginalWire(mainWires()[0])
      expect(mainWires()[0].body.model).toBe("test")
      expect(mainWires()[0].body.temperature).toBe(0.24)
      expect(mainWires()[0].body.save_to_db).toBe(false)
      expect(JSON.stringify(mainWires()[0].body.messages)).toContain(
        "Original Retry question"
      )
      expect(JSON.stringify(mainWires()[0].body.messages)).toContain(
        originalNote
      )
      expect(
        useStoreMessageOption
          .getState()
          .messages.filter((message) => !message.isBot)
      ).toHaveLength(1)
      expect(useStoreMessageOption.getState().messages.at(-1)?.message).toBe(
        "Original Scene-held retry answer"
      )
      expect(steeringTuple()).toEqual({
        mode: "none",
        forceNarrate: false,
        revision: turn.originalSteering.revision + 1
      })
      expect(
        (await turn.consumer.scene.capture()).scene.originalRecord
      ).toEqual(turn.consumer.scene.original.originalRecord)
      expect(physical.tables.chatHistories.rows.size).toBe(0)
      expect(physical.tables.messages.rows.size).toBe(0)
    } finally {
      turn.held.resolve(turn.read.value)
      await complete(turn.pending)
      turn.consumer.unmount()
    }
  })

  it.each(["model", "tool", "effective-settings"] as const)(
    "%s replacement while the actual original Scene read is held prevents response dispatch and old presentation writes",
    async (replacement) => {
      const turn = await startPublicRetryAtSceneRead()
      try {
        await act(async () => {
          if (replacement === "model")
            useStoreMessageOption
              .getState()
              .setSelectedModel("replacement-test")
          else if (replacement === "tool")
            useStoreMessageOption.getState().setToolChoice("auto")
          else useStoreChatModelSettings.getState().setTemperature(0.81)
        })
        if (replacement === "model")
          expect(useStoreMessageOption.getState().selectedModel).toBe(
            "replacement-test"
          )
        else if (replacement === "tool")
          expect(useStoreMessageOption.getState().toolChoice).toBe("auto")
        else
          expect(
            useStoreChatModelSettings.getState().getEffectiveSettings()
              .temperature
          ).toBe(0.81)
        // The same qualified Scene/account/draft survives this settings edit;
        // retirement is the original command's dispatch intent, not New or an
        // unavailable replacement model/catalog masking a response attempt.
        expect(turn.consumer.scene.account.isCurrent()).toBe(true)
        expect(presentation()).toEqual(turn.originalPresentation)
        expect(steeringTuple()).toEqual(turn.originalSteering)
        turn.held.resolve(turn.read.value)
        expect(await complete(turn.pending)).toMatchObject({
          status: "skipped"
        })
        expect(mainWires()).toEqual([])
        expect(presentation()).toEqual(turn.originalPresentation)
        expect(steeringTuple()).toEqual(turn.originalSteering)
        expect(
          (await turn.consumer.scene.capture()).scene.originalRecord
        ).toEqual(turn.consumer.scene.original.originalRecord)
        expect(physical.tables.chatHistories.rows.size).toBe(0)
        expect(physical.tables.messages.rows.size).toBe(0)
      } finally {
        turn.held.resolve(turn.read.value)
        await complete(turn.pending)
        turn.consumer.unmount()
      }
    }
  )
})

const mountSeededLegacyCharacter = async () => {
  await act(async () => {
    const core = useStoreMessageOption.getState()
    core.setTemporaryChat(false)
    core.startNewChatDraft(false)
  })
  const originalAccount = await mountOriginalAccount()
  const consumer = mountFull(false, true)
  try {
    const scene = await prepareScene(
      originalAccount.mounted.result.current!,
      false
    )
    expect(consumer.mounted.result.current.selection).toBeNull()
    const actorSceneSend = await scene.capture()
    let seeded!: ChatSubmitResult
    await act(async () => {
      seeded = await consumer.mounted.result.current.actions.onSubmit({
        message: "Original Character question",
        image: "",
        actorSceneSend
      })
    })
    expect(normalizeChatSubmitResult(seeded)).toMatchObject({
      status: "submitted"
    })
    const create = wires.find(
      (wire) => wire.path === "/api/v1/chats/" && wire.method === "POST"
    )!
    expect(create).toBeDefined()
    assertOriginalWire(create)
    expect(String(create.body.character_id)).toBe("27")
    const response = mainWires()[0]
    expect(response.path).toBe("/api/v1/chats/" + parentChatId + "/complete-v2")
    assertOriginalWire(response)
    expect(response.body).toMatchObject({
      model: "test",
      provider: "openai",
      save_to_db: true
    })
    const state = useStoreMessageOption.getState()
    expect(state.serverChatId).toBe(parentChatId)
    expect(String(state.serverChatCharacterId)).toBe("27")
    expect(state.historyId).toBeTruthy()
    expect(state.historyId).not.toBe("temp")
    expect(state.chatDraftAddress).toBe(scene.address)
    expect(state.chatDraftRevision).toBe(scene.revision)
    expect(state.messages.map((message) => message.message)).toEqual([
      "Original Character question",
      "Original answer"
    ])
    expect(
      state.messages.every((message) => Boolean(message.serverMessageId))
    ).toBe(true)
    const originalHistoryId = state.historyId!
    expect(
      await physical.tables.chatHistories.get(originalHistoryId)
    ).toMatchObject({
      id: originalHistoryId,
      server_scope_key: serverChatMirrorOwnerKey({
        requestScope: scene.account.requestScope
      })
    })
    expect(
      (
        await physical.tables.messages
          .where("history_id")
          .equals(originalHistoryId)
          .toArray()
      ).map((row) => row.content)
    ).toEqual(["Original Character question", "Original answer"])
    expect(originalAccount.mounted.result.current!.getCurrent()).toMatchObject({
      status: "idle",
      owner: null
    })
    expect((await scene.capture()).scene.originalRecord).toEqual(
      scene.original.originalRecord
    )
    wires.length = 0
    return {
      ...consumer,
      scene,
      originalAccount,
      originalHistoryId,
      unmount: () => {
        consumer.unmount()
        originalAccount.unmount()
      }
    }
  } catch (error) {
    consumer.unmount()
    originalAccount.unmount()
    throw error
  }
}
type LegacyConsumer = Awaited<ReturnType<typeof mountSeededLegacyCharacter>>
const startExplicitRetry = async (consumer: LegacyConsumer) => {
  const actorSceneSend = await consumer.scene.capture()
  const controller = new AbortController()
  let pending!: Promise<ChatSubmitResult | void>
  act(() => {
    pending = consumer.mounted.result.current.actions.regenerateLastMessage({
      actorSceneSend,
      controller
    })
  })
  return { pending, actorSceneSend, controller }
}
const settleRetry = async (pending: Promise<ChatSubmitResult | void>) => {
  let settled!: { result?: ChatSubmitResult; error?: unknown }
  await act(async () => {
    try {
      settled = { result: normalizeChatSubmitResult(await pending) }
    } catch (error) {
      settled = { error }
    }
  })
  return settled
}
const installOwnedRouteListener = (consumer: LegacyConsumer) => {
  const accepted: ChatRouteReplacementDetail[] = []
  const handle = (event: Event) => {
    const detail = (event as CustomEvent<ChatRouteReplacementDetail>).detail
    const core = useStoreMessageOption.getState(),
      session = usePlaygroundSessionStore.getState()
    // Declared UI event boundary mirrors Playground's synchronous href/parent/
    // history/restore checks, and invokes the actual store cancellation. The
    // full router/container is not mounted; the receipt follows the real
    // cancellation and uses the resulting live revision.
    if (
      !detail ||
      detail.href !== window.location.href ||
      detail.serverChatId !== core.serverChatId ||
      detail.historyId !== core.historyId ||
      detail.restoreRevision !== session.restoreRevision ||
      detail.serverChatId !== parentChatId ||
      detail.historyId !== consumer.originalHistoryId ||
      detail.characterId !== "27"
    )
      return
    accepted.push(detail)
    session.cancelPendingRestore()
    if (
      detail.onRestoreCancelled &&
      !detail.onRestoreCancelled(
        usePlaygroundSessionStore.getState().restoreRevision
      )
    )
      return
  }
  window.addEventListener(CHAT_ROUTE_REPLACEMENT_EVENT, handle)
  return {
    accepted,
    remove: () =>
      window.removeEventListener(CHAT_ROUTE_REPLACEMENT_EVENT, handle)
  }
}

const readRetainedPrivateScene = async (consumer: LegacyConsumer) => {
  const source = captureChatDraftSource(
    () =>
      consumer.scene.account.isCurrent() &&
      useStoreMessageOption.getState().chatDraftAddress ===
        consumer.scene.address &&
      useStoreMessageOption.getState().chatDraftRevision ===
        consumer.scene.revision
  )
  const target = captureActorSettingsTarget({
    address: consumer.scene.address,
    requestScope: consumer.scene.account.requestScope,
    accountIsCurrent: consumer.scene.account.isCurrent,
    isCurrent: source.isCurrent
  })
  expect(target).not.toBeNull()
  return getActorSceneForChat({ target: target! })
}

describe("full legacy Character Retry owns original preparation and exact fork publications", () => {
  it("uses the original scope for an unchanged explicit Retry, including its actual own route cancellation and issued fork IDs", async () => {
    const consumer = await mountSeededLegacyCharacter()
    const route = installOwnedRouteListener(consumer)
    const restoreRevision = usePlaygroundSessionStore.getState().restoreRevision
    try {
      answer = "Original fork answer"
      const { pending, actorSceneSend } = await startExplicitRetry(consumer)
      const settled = await settleRetry(pending)
      expect(settled.error).toBeUndefined()
      expect(settled.result).toMatchObject({ status: "submitted" })
      expect(route.accepted).toHaveLength(1)
      expect(route.accepted[0]).toMatchObject({
        serverChatId: parentChatId,
        historyId: consumer.originalHistoryId,
        restoreRevision,
        nextChatId: childChatId,
        characterId: "27"
      })
      expect(usePlaygroundSessionStore.getState().restoreRevision).toBe(
        restoreRevision + 1
      )
      const originalRead = wires.find(
        (wire) =>
          wire.path === "/api/v1/chats/" + parentChatId && wire.method === "GET"
      )!
      expect(originalRead).toBeDefined()
      assertOriginalWire(originalRead)
      const create = wires.find(
        (wire) => wire.path === "/api/v1/chats/" && wire.method === "POST"
      )!
      assertOriginalWire(create)
      expect(create.body.parent_conversation_id).toBe(parentChatId)
      expect(String(create.body.character_id)).toBe("27")
      const copies = wires.filter(
        (wire) => wire.path === "/api/v1/chats/" + childChatId + "/messages"
      )
      expect(copies.map((wire) => wire.body.content)).toEqual([
        "Original Character question"
      ])
      expect(mainWires()).toHaveLength(1)
      assertOriginalWire(mainWires()[0])
      expect(mainWires()[0].path).toBe(
        "/api/v1/chats/" + childChatId + "/complete-v2"
      )
      expect(mainWires()[0].body.save_to_db).toBe(true)
      expect(actorSceneSend.requestScope).toEqual(
        consumer.scene.account.requestScope
      )
      const state = useStoreMessageOption.getState()
      expect(state.serverChatId).toBe(childChatId)
      expect(state.historyId).not.toBe(consumer.originalHistoryId)
      expect(state.messages.map((message) => message.message)).toEqual([
        "Original Character question",
        "Original fork answer"
      ])
      expect(
        state.messages.every((message) => Boolean(message.serverMessageId))
      ).toBe(true)
      expect(
        await physical.tables.chatHistories.get(state.historyId)
      ).toMatchObject({
        server_chat_id: childChatId,
        server_scope_key: serverChatMirrorOwnerKey({
          requestScope: consumer.scene.account.requestScope
        })
      })
      expect(
        (
          await physical.tables.messages
            .where("history_id")
            .equals(state.historyId)
            .toArray()
        ).map((row) => row.content)
      ).toEqual(["Original Character question", "Original fork answer"])
      expect((await readRetainedPrivateScene(consumer)).originalRecord).toEqual(
        consumer.scene.original.originalRecord
      )
      expect(
        consumer.originalAccount.mounted.result.current!.getCurrent()
      ).toMatchObject({ status: "idle", owner: null })
    } finally {
      route.remove()
      consumer.unmount()
    }
  })

  it.each(["New", "account replacement", "account ABA", "Stop"] as const)(
    "%s during the actual fork scope-preparation read prevents all later fork IO and publication",
    async (retirement) => {
      const consumer = await mountSeededLegacyCharacter()
      const route = installOwnedRouteListener(consumer)
      const entered = deferred<{ stack: string; value: unknown }>(),
        held = deferred<void>()
      let pending: Promise<ChatSubmitResult | void> | undefined
      let originalController: AbortController | undefined
      physical.control.beforeRead = async (read) => {
        if (
          read.area !== "local" ||
          read.key !== "tldwConfig" ||
          !read.stack.includes("loadServicePromptSnapshot")
        )
          return read.value
        physical.control.beforeRead = undefined
        entered.resolve({ stack: read.stack, value: read.value })
        await held.promise
        return read.value
      }
      try {
        const dispatch = await startExplicitRetry(consumer)
        pending = dispatch.pending
        originalController = dispatch.controller
        await awaitGate(entered.promise, pending)
        expect((await entered.promise).stack).toContain(
          "loadServicePromptSnapshot"
        )
        expect((await entered.promise).value).toEqual(originalConfig)
        expect(wires).toEqual([])
        let later = presentation()
        if (retirement === "New")
          act(() => {
            later = replacePresentation("later-history", "later-chat")
          })
        else if (retirement === "Stop") act(() => originalController!.abort())
        else {
          await act(async () => {
            await physical.storage("local").set("tldwConfig", replacementConfig)
            if (retirement === "account ABA")
              await physical.storage("local").set("tldwConfig", originalConfig)
          })
          await waitFor(() =>
            expect(consumer.scene.account.isCurrent()).toBe(false)
          )
          later = presentation()
        }
        held.resolve(undefined)
        const settled = await settleRetry(pending)
        expect(settled.result?.status).not.toBe("submitted")
        expect(wires).toEqual([])
        expect(route.accepted).toEqual([])
        expect(presentation()).toEqual(later)
        expect(
          [...physical.tables.chatHistories.rows.values()].some(
            (row) => row.server_chat_id === childChatId
          )
        ).toBe(false)
        if (retirement === "Stop")
          expect(originalController.signal.aborted).toBe(true)
      } finally {
        held.resolve(undefined)
        if (pending) await settleRetry(pending)
        route.remove()
        consumer.unmount()
      }
    }
  )

  it.each(["New", "account replacement", "account ABA"] as const)(
    "%s while the original scoped parent read is physically held prevents child writes and presentation",
    async (retirement) => {
      const consumer = await mountSeededLegacyCharacter()
      const route = installOwnedRouteListener(consumer)
      const entered = deferred<Wire>(),
        held = deferred<Response>()
      let pending: Promise<ChatSubmitResult | void> | undefined
      physical.control.beforeFetch = async (url, _init) => {
        if (new URL(url).pathname !== "/api/v1/chats/" + parentChatId)
          return undefined
        physical.control.beforeFetch = undefined
        entered.resolve(wires.at(-1)!)
        return held.promise
      }
      try {
        pending = (await startExplicitRetry(consumer)).pending
        await awaitGate(entered.promise, pending)
        const originalRead = await entered.promise
        assertOriginalWire(originalRead)
        expect(wires).toEqual([originalRead])
        let later = presentation()
        if (retirement === "New")
          act(() => {
            later = replacePresentation("later-history", "later-chat")
          })
        else {
          await act(async () => {
            await physical.storage("local").set("tldwConfig", replacementConfig)
            if (retirement === "account ABA")
              await physical.storage("local").set("tldwConfig", originalConfig)
          })
          await waitFor(() =>
            expect(consumer.scene.account.isCurrent()).toBe(false)
          )
          await waitFor(() => expect(originalRead.signal?.aborted).toBe(true))
          later = presentation()
        }
        held.resolve(
          Response.json({
            id: parentChatId,
            character_id: 27,
            title: "Captain Mira",
            state: "in-progress",
            version: 1
          })
        )
        const settled = await settleRetry(pending)
        expect(settled.result?.status).not.toBe("submitted")
        expect(wires).toEqual([originalRead])
        expect(route.accepted).toEqual([])
        expect(presentation()).toEqual(later)
        expect(
          [...physical.tables.chatHistories.rows.values()].some(
            (row) => row.server_chat_id === childChatId
          )
        ).toBe(false)
      } finally {
        held.resolve(
          Response.json({
            id: parentChatId,
            character_id: 27,
            title: "Captain Mira",
            state: "in-progress",
            version: 1
          })
        )
        if (pending) await settleRetry(pending)
        route.remove()
        consumer.unmount()
      }
    }
  )

  it.each([
    ["server", "New"],
    ["server", "ABA"],
    ["history", "New"],
    ["history", "ABA"]
  ] as const)(
    "a synchronous own %s ID subscriber performing %s retires every later fork mutation",
    async (boundary, replacement) => {
      const consumer = await mountSeededLegacyCharacter()
      const route = installOwnedRouteListener(consumer)
      let later: ReturnType<typeof presentation> | undefined
      let observedOwnHistory: string | null = null
      let triggered = false
      const unsubscribe = useStoreMessageOption.subscribe((state, previous) => {
        const ownServer =
          boundary === "server" &&
          state.serverChatId === childChatId &&
          previous.serverChatId !== childChatId
        const ownHistory =
          boundary === "history" &&
          state.serverChatId === childChatId &&
          state.historyId !== previous.historyId &&
          Boolean(state.historyId) &&
          state.historyId !== consumer.originalHistoryId
        if (triggered || (!ownServer && !ownHistory)) return
        triggered = true
        observedOwnHistory = state.historyId
        // The callback runs inside the real setter notification. For ABA, return
        // to the exact issued IDs only after a real New address/revision; those
        // IDs must never renew the original command or Scene lease.
        later = replacePresentation(
          replacement === "ABA" ? state.historyId : "later-history",
          replacement === "ABA" ? childChatId : "later-chat"
        )
      })
      try {
        const { pending } = await startExplicitRetry(consumer)
        const settled = await settleRetry(pending)
        expect(triggered).toBe(true)
        expect(route.accepted).toHaveLength(1)
        expect(later?.address).not.toBe(consumer.scene.address)
        expect(later?.revision).toBeGreaterThan(consumer.scene.revision)
        if (boundary === "history")
          expect(observedOwnHistory).not.toBe(consumer.originalHistoryId)
        expect(settled.result?.status).not.toBe("submitted")
        expect(mainWires()).toEqual([])
        expect(presentation()).toEqual(later)
        const copies = wires.filter(
          (wire) => wire.path === "/api/v1/chats/" + childChatId + "/messages"
        )
        expect(copies.map((wire) => wire.body.content)).toEqual([
          "Original Character question"
        ])
        // Already admitted copy rows belong to the original account; New does
        // not roll them back or lend authority to the later presentation.
        expect(
          [...physical.tables.chatHistories.rows.values()].filter(
            (row) => row.server_chat_id === childChatId
          )
        ).toHaveLength(1)
      } finally {
        unsubscribe()
        route.remove()
        consumer.unmount()
      }
    }
  )

  it("the ordinary no-dispatch legacy Character call refuses before fork IO when no qualified Scene capture is available", async () => {
    const consumer = await mountSeededLegacyCharacter()
    const before = presentation()
    try {
      let pending!: Promise<ChatSubmitResult | void>
      act(() => {
        pending =
          consumer.mounted.result.current.actions.regenerateLastMessage()
      })
      const settled = await settleRetry(pending)
      expect(settled.result?.status).not.toBe("submitted")
      expect(wires).toEqual([])
      expect(presentation()).toEqual(before)
      expect(
        [...physical.tables.chatHistories.rows.values()].some(
          (row) => row.server_chat_id === childChatId
        )
      ).toBe(false)
      expect(
        consumer.originalAccount.mounted.result.current!.getCurrent()
      ).toMatchObject({ status: "idle", owner: null })
    } finally {
      consumer.unmount()
    }
  })
})
