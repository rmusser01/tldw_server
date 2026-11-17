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

// Physical storage and fetch are the only replaced boundaries. The full
// consumer, verified draft owner, Scene reads, factories, API domains and proxy
// dispatch/abort guards all execute their production implementations.
const physical = vi.hoisted(() => {
  type Read = { area: string; key: string; value: unknown; stack: string }
  type Watch = Record<
    string,
    (change: { oldValue?: unknown; newValue?: unknown }) => void
  >
  const areas = new Map<string, Map<string, unknown>>()
  const watchers = new Map<string, Set<Watch>>()
  const control = {
    beforeRead: undefined as ((read: Read) => Promise<unknown>) | undefined,
    beforeFetch: undefined as
      | ((url: string, init: RequestInit) => Promise<Response | undefined>)
      | undefined
  }
  const areaFor = (name: string) => {
    if (!areas.has(name)) areas.set(name, new Map())
    return areas.get(name)!
  }
  const publish = (
    area: string,
    key: string,
    oldValue: unknown,
    newValue: unknown
  ) => {
    for (const subscription of watchers.get(area) ?? [])
      subscription[key]?.({
        oldValue: structuredClone(oldValue),
        newValue: structuredClone(newValue)
      })
  }
  // Match the direct WebUI Storage shim, whose default backend area is local.
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
      // The WebUI useStorage hook calls the returned unsubscribe on a key or
      // instance change; service callers also use unwatch(map) directly.
      return () => {
        watchers.get(area)?.delete(subscription)
      }
    },
    unwatch: (subscription: Watch) => {
      watchers.get(area)?.delete(subscription)
    }
  })
  type Row = Record<string, unknown>
  type Collection = {
    toArray: () => Promise<Row[]>
    and: (predicate: (row: Row) => boolean) => Collection
  }
  const tables: Record<string, ReturnType<typeof table>> = {}
  function table() {
    const rows = new Map<string, Row>()
    const collection = (matches: (row: Row) => boolean): Collection => ({
      toArray: async () => structuredClone([...rows.values()].filter(matches)),
      and: (predicate: (row: Row) => boolean) =>
        collection((row) => matches(row) && predicate(row))
    })
    return {
      rows,
      get: async (id: unknown) => structuredClone(rows.get(JSON.stringify(id))),
      put: async (row: Row) => {
        rows.set(JSON.stringify(row.id), structuredClone(row))
      },
      update: async (id: unknown, value: Row) => {
        rows.set(JSON.stringify(id), {
          ...rows.get(JSON.stringify(id)),
          ...structuredClone(value)
        })
      },
      toArray: async () => structuredClone([...rows.values()]),
      where: (key: string) => ({
        equals: (value: unknown) => collection((row) => row[key] === value)
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
    tables[name] = table()
  return { areas, areaFor, control, storage, tables, fetch: vi.fn() }
})
vi.mock("wxt/browser", () => ({
  browser: {
    runtime: { id: undefined },
    storage: { onChanged: { addListener: vi.fn(), removeListener: vi.fn() } }
  }
}))
vi.mock("@/utils/safe-storage", async (original) => ({
  ...(await original<typeof import("@/utils/safe-storage")>()),
  createSafeStorage: (options: { area?: string } = {}) =>
    physical.storage(options.area)
}))
vi.mock("@/db/dexie/schema", () => ({
  db: {
    ...physical.tables,
    transaction: () => {
      throw new Error(
        "This Temporary consumer must not persist a durable history"
      )
    }
  }
}))

import { useChatActions } from "../useChatActions"
import { useServerChatHistoryId } from "../useServerChatHistoryId"
import {
  HistorySelectionProvider,
  useHistorySelectionContext
} from "../useHistorySelection"
import {
  captureActorSceneSend,
  captureCurrentActorSceneTarget
} from "../actor-scene-source"
import {
  getActorSceneForChat,
  saveActorSettingsForChat
} from "@/services/actor-settings"
import { createDefaultActorSettings } from "@/types/actor"
import { useActorStore } from "@/store/actor"
import { useStoreMessageOption } from "@/store/option"
import { useStoreChatModelSettings } from "@/store/model"
import { useMcpToolsStore } from "@/store/mcp-tools"
import { tldwClient, type TldwConfig } from "@/services/tldw/TldwApiClient"
import { tldwModels } from "@/services/tldw"
import { setTitleGenEnabled } from "@/services/title"
import { __clearSettingCachesForTests } from "@/services/settings/registry"
import { deriveSingleUserApiKeyCredentialScope } from "@/services/chat-surface-scope"
import { resolveServicePromptScope } from "@/services/service-prompts"
import {
  normalizeChatSubmitResult,
  type ChatSubmitResult
} from "../chat-action-utils"

const originalConfig: TldwConfig = {
  serverUrl: "https://response-dispatch.example",
  authMode: "single-user",
  authSource: "manual",
  credentialSource: "manual",
  apiKeyPersistence: "device",
  apiKeyServerOrigin: "https://response-dispatch.example",
  apiKey: "original-response-dispatch-fixture-key"
}
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
const imageResponse = () =>
  Response.json({
    artifact: {
      file_id: 81,
      export: {
        content_b64: "original-generated-image",
        content_type: "image/png",
        format: "png",
        bytes: 24
      }
    }
  })
type Wire = {
  url: string
  body: Record<string, unknown>
  headers: Headers
  signal?: AbortSignal | null
}
let wires: Wire[]
let auxiliaryWires: Wire[]
let responseGate: ReturnType<typeof deferred<Response>> | undefined
let initialCore: ReturnType<typeof useStoreMessageOption.getState>

beforeEach(async () => {
  vi.clearAllMocks()
  // The physical areas below are cleared out-of-band (no storage events), so
  // registry-cached settings (imageBackendConfigs) would survive from the
  // previous test and skip the physical reads these cases gate on.
  __clearSettingCachesForTests()
  physical.control.beforeRead = undefined
  physical.control.beforeFetch = undefined
  physical.areas.clear()
  Object.values(physical.tables).forEach((table) => table.rows.clear())
  localStorage.clear()
  sessionStorage.clear()
  physical.areaFor("local").set("tldwConfig", structuredClone(originalConfig))
  for (const area of ["local", "sync"]) {
    physical
      .areaFor(area)
      .set("systemPromptForNonRagOption", "Original default prompt")
    physical.areaFor(area).set("imageBackendConfigs", {
      comfyui: {
        width: 640,
        height: 480,
        steps: 19,
        cfgScale: 6,
        seed: 17,
        format: "png"
      }
    })
  }
  wires = []
  auxiliaryWires = []
  responseGate = undefined
  physical.fetch.mockImplementation(async (url: string, init: RequestInit) => {
    const path = new URL(url).pathname
    const overridden = await physical.control.beforeFetch?.(url, init)
    if (overridden) return overridden
    if (path === "/openapi.json")
      return Response.json({
        paths: {
          "/api/v1/characters/{id}": {},
          "/api/v1/chats/": {}
        }
      })
    if (path === "/api/v1/characters/27" || path === "/api/v1/characters/27/") {
      auxiliaryWires.push({
        url,
        body: {},
        headers: new Headers(init.headers),
        signal: init.signal
      })
      return Response.json({
        id: 27,
        name: "Captain Mira",
        system_prompt: "Stay in character."
      })
    }
    if (path === "/api/v1/chats/") {
      auxiliaryWires.push({
        url,
        body: JSON.parse(String(init.body)),
        headers: new Headers(init.headers),
        signal: init.signal
      })
      return Response.json({
        id: "legacy-character-chat",
        character_id: 27,
        title: "Captain Mira",
        version: 1
      })
    }
    if (
      path === "/api/v1/chats/legacy-character-chat/messages" ||
      path === "/api/v1/chats/legacy-character-chat/completions/persist"
    ) {
      auxiliaryWires.push({
        url,
        body: JSON.parse(String(init.body)),
        headers: new Headers(init.headers),
        signal: init.signal
      })
      return Response.json({
        id: "legacy-user-message",
        assistant_message_id: "legacy-answer-message",
        version: 1
      })
    }
    if (path === "/api/v1/visual-identities/bindings/resolve") {
      auxiliaryWires.push({
        url,
        body: {},
        headers: new Headers(init.headers),
        signal: init.signal
      })
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
    }
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
      path !== "/api/v1/chat/completions" &&
      path !== "/api/v1/files/create" &&
      path !== "/api/v1/chats/legacy-character-chat/complete-v2"
    )
      throw new Error("Unexpected physical request: " + path)
    wires.push({
      url,
      body: JSON.parse(String(init.body)),
      headers: new Headers(init.headers),
      signal: init.signal
    })
    if (responseGate) return responseGate.promise
    return path === "/api/v1/files/create"
      ? imageResponse()
      : textResponse("Seed answer")
  })
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
  useStoreMessageOption.getState().startNewChatDraft(true)
  useStoreChatModelSettings.getState().reset()
  useStoreChatModelSettings.getState().setApiProvider("openai")
  useMcpToolsStore.setState({ toolsLoading: false, healthState: "unknown" })
  useActorStore.getState().reset()
  await tldwClient.initialize()
  // Each case owns its physical hydration read. A prior aborted caller can
  // still finish the shared unscoped catalog read and populate these caches.
  tldwClient.characterCache.clear()
  tldwClient.characterInFlight.clear()
  await tldwModels.clearCache()
  await setTitleGenEnabled(false)
})
afterEach(() => {
  physical.control.beforeRead = undefined
  physical.control.beforeFetch = undefined
  responseGate = undefined
  useStoreMessageOption.setState(initialCore)
  useActorStore.getState().reset()
  vi.unstubAllGlobals()
})

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
    useStoreMessageOption.getState().setMessageSteeringMode("continue_as_user")
    useStoreMessageOption.getState().setMessageSteeringForceNarrate(true)
  })
  return steeringTuple()
}
const mountSeededTemporaryConsumer = async () => {
  // Qualify the actual configuration/auth IO independently before the provider
  // mounts. This is an assertion, never a supplied draft-account capability.
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
        createElement(HistorySelectionProvider, null, children)
      )
    )
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
        selectedAssistant: null
      })
      return {
        actions,
        selection,
        getSignal: () => controllerRef.current?.signal
      }
    },
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
    const verified =
      mounted.result.current.selection!.draftAccount!.requestScope!
    expect(verified.config).toEqual(resolvedAccount.config)
    expect(verified.userId).toEqual(resolvedAccount.userId)
    expect(verified.config.expectedSingleUserApiKeyScope).toBe(
      deriveSingleUserApiKeyCredentialScope(
        "single-user",
        originalConfig.apiKey
      )
    )
    let seeded!: ChatSubmitResult
    await act(async () => {
      seeded = await mounted.result.current.actions.onSubmit({
        message: "Original seed question",
        image: ""
      })
    })
    expect(normalizeChatSubmitResult(seeded)).toMatchObject({
      status: "submitted"
    })
    expect(wires).toHaveLength(1)
    expect(wires[0].url).toBe(
      originalConfig.serverUrl + "/api/v1/chat/completions"
    )
    expect(wires[0].headers.get("X-API-KEY")).toBe(originalConfig.apiKey)
    expect(wires[0].body.save_to_db).toBe(false)
    expect(useStoreMessageOption.getState().historyId).toBe("temp")
    expect(
      useStoreMessageOption
        .getState()
        .messages.map((message) => message.message)
    ).toEqual(["Original seed question", "Seed answer"])
    expect(mounted.result.current.selection!.getCurrent()).toMatchObject({
      status: "idle",
      owner: null
    })
    expect(physical.tables.chatHistories.rows.size).toBe(0)
    // Capture and save only after the real first Send acknowledged its own
    // Temporary sentinel. No raw ID creates a Scene owner or load receipt.
    const target = captureCurrentActorSceneTarget(
      mounted.result.current.selection
    )
    expect(target).not.toBeNull()
    const scene = {
      ...createDefaultActorSettings(),
      isEnabled: true,
      notes: "Original owned Scene note"
    }
    expect(
      await saveActorSettingsForChat({
        target: target!,
        settings: scene,
        historyId: "temp",
        serverChatId: null,
        expectedRecord: null
      })
    ).toBe(true)
    const original = await getActorSceneForChat({ target: target! })
    expect(original.originalRecord?.value).toEqual(scene)
    const originalAddress = useStoreMessageOption.getState().chatDraftAddress
    const assertOriginal = async () => {
      expect(useStoreMessageOption.getState().chatDraftAddress).toBe(
        originalAddress
      )
      expect(mounted.result.current.selection!.draftAccount!.isCurrent()).toBe(
        true
      )
      expect(
        (await getActorSceneForChat({ target: target! })).originalRecord
      ).toEqual(original.originalRecord)
    }
    wires.length = 0
    return { mounted, unmount, assertOriginal }
  } catch (error) {
    unmount()
    throw error
  }
}
type Consumer = Awaited<ReturnType<typeof mountSeededTemporaryConsumer>>
const start = (consumer: Consumer, image: boolean) => {
  let pending!: Promise<ChatSubmitResult>
  act(() => {
    pending = consumer.mounted.result.current.actions.onSubmit(
      image
        ? {
            message: "Original image prompt",
            image: "",
            imageBackendOverride: "comfyui",
            userMessageType: "IMAGEGEN_USER",
            assistantMessageType: "IMAGEGEN_ASSISTANT"
          }
        : { message: "", image: "", isContinue: true }
    )
  })
  return pending
}
const awaitGate = async (
  entered: Promise<void>,
  pending: Promise<ChatSubmitResult>
) => {
  await Promise.race([
    entered,
    pending.then((result) => {
      throw new Error(
        "The actual physical preparation gate was not entered: " +
          JSON.stringify(result)
      )
    })
  ])
}

// The retained SDK hook has no H1 context. Its command receives only the
// original private Scene captured from the real still-mounted draft provider;
// no issued server ID supplies a history owner, view, load receipt or new Scene.
const mountLegacyCharacterConsumer = async () => {
  const original = await mountSeededTemporaryConsumer()
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  })
  const Wrapper = ({ children }: PropsWithChildren) =>
    createElement(
      QueryClientProvider,
      { client: queryClient },
      createElement(App, null, children)
    )
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
        selectedAssistant: {
          kind: "character",
          id: "27",
          name: "Assistant",
          metadata: { selectionMode: "tracked" }
        }
      })
      return {
        actions,
        selection,
        getSignal: () => controllerRef.current?.signal
      }
    },
    { wrapper: Wrapper }
  )
  expect(mounted.result.current.selection).toBeNull()
  return {
    mounted,
    original,
    start: async () => {
      // Command T0 capture precedes its read and every hydration/setup await.
      const capture = captureActorSceneSend(
        original.mounted.result.current.selection
      )
      expect(capture?.isCurrent()).toBe(true)
      const actorSceneSend = await capture!.read()
      let pending!: Promise<ChatSubmitResult>
      act(() => {
        pending = mounted.result.current.actions.onSubmit({
          message: "Original character question",
          image: "",
          actorSceneSend
        })
      })
      return { pending, actorSceneSend }
    },
    unmount: () => {
      mounted.unmount()
      queryClient.clear()
      original.unmount()
    }
  }
}

describe("full response steering reaches an actual dispatch boundary", () => {
  it.each(["Stop", "prompt storage failure"] as const)(
    "Continue retains original steering after %s before the actual main transport",
    async (failure) => {
      const consumer = await mountSeededTemporaryConsumer()
      const original = await armSteering()
      const entered = deferred<void>(),
        held = deferred<unknown>()
      let originalSignal: AbortSignal | undefined
      let pending: Promise<ChatSubmitResult> | undefined
      // This physical default-prompt read is the real Continue preparation
      // await. A failure here has no response handoff to consume steering.
      physical.control.beforeRead = async (read) => {
        if (read.key !== "systemPromptForNonRagOption") return read.value
        physical.control.beforeRead = undefined
        originalSignal = consumer.mounted.result.current.getSignal()
        entered.resolve(undefined)
        await held.promise
        if (failure === "prompt storage failure")
          throw new Error("Declared default prompt storage failure")
        return read.value
      }
      try {
        pending = start(consumer, false)
        await awaitGate(entered.promise, pending)
        expect(steeringTuple()).toEqual(original)
        expect(wires).toEqual([])
        if (failure === "Stop")
          act(() =>
            consumer.mounted.result.current.actions.stopStreamingRequest()
          )
        if (failure === "Stop") expect(originalSignal?.aborted).toBe(true)
        held.resolve(undefined)
        let outcome!: ChatSubmitResult
        await act(async () => {
          outcome = await pending!
        })
        expect(normalizeChatSubmitResult(outcome)).toMatchObject({
          status: failure === "Stop" ? "skipped" : "failed"
        })
        expect(wires).toEqual([])
        expect(steeringTuple()).toEqual(original)
        await consumer.assertOriginal()
      } finally {
        held.resolve(undefined)
        if (pending)
          await act(async () => {
            await pending
          })
        consumer.unmount()
      }
    }
  )

  it.each(["Stop", "credential refusal"] as const)(
    "image generation retains original steering after %s during actual bgRequest configuration",
    async (failure) => {
      const consumer = await mountSeededTemporaryConsumer()
      const original = await armSteering()
      const entered = deferred<void>(),
        held = deferred<unknown>()
      let imagePreparationRead = false,
        imageConfigReads = 0,
        heldStack = ""
      let originalSignal: AbortSignal | undefined
      let pending: Promise<ChatSubmitResult> | undefined
      physical.control.beforeRead = async (read) => {
        if (read.key === "imageBackendConfigs") imagePreparationRead = true
        if (
          !imagePreparationRead ||
          read.area !== "local" ||
          read.key !== "tldwConfig"
        )
          return read.value
        imageConfigReads++
        // The first read initializes the real image preflight client. The
        // second is the scoped bgRequest guard before request-core/fetch.
        if (imageConfigReads !== 2) return read.value
        physical.control.beforeRead = undefined
        heldStack = read.stack
        originalSignal = consumer.mounted.result.current.getSignal()
        entered.resolve(undefined)
        await held.promise
        return failure === "credential refusal"
          ? { ...originalConfig, apiKey: "later-response-dispatch-fixture-key" }
          : read.value
      }
      try {
        pending = start(consumer, true)
        await awaitGate(entered.promise, pending)
        expect(heldStack).toContain("resolveCurrentServicePromptConfig")
        expect(wires).toEqual([])
        expect(steeringTuple()).toEqual(original)
        if (failure === "Stop")
          act(() =>
            consumer.mounted.result.current.actions.stopStreamingRequest()
          )
        if (failure === "Stop") expect(originalSignal?.aborted).toBe(true)
        held.resolve(undefined)
        let outcome!: ChatSubmitResult
        await act(async () => {
          outcome = await pending!
        })
        expect(normalizeChatSubmitResult(outcome)).toMatchObject(
          failure === "credential refusal"
            ? { status: "skipped", reason: "Request scope changed" }
            : { status: "skipped" }
        )
        expect(wires).toEqual([])
        expect(steeringTuple()).toEqual(original)
        await consumer.assertOriginal()
      } finally {
        held.resolve(undefined)
        if (pending)
          await act(async () => {
            await pending
          })
        consumer.unmount()
      }
    }
  )

  it.each([false, true])(
    "consumes exactly the original intent after a successful actual %s transport",
    async (image) => {
      const consumer = await mountSeededTemporaryConsumer()
      const original = await armSteering()
      const heldResponse = deferred<Response>()
      responseGate = heldResponse
      let pending: Promise<ChatSubmitResult> | undefined
      try {
        pending = start(consumer, image)
        await waitFor(() => expect(wires).toHaveLength(1))
        expect(steeringTuple()).toEqual(original)
        expect(wires[0].headers.get("X-API-KEY")).toBe(originalConfig.apiKey)
        expect(wires[0].signal?.aborted).toBe(false)
        if (image) {
          expect(wires[0].url).toBe(
            originalConfig.serverUrl + "/api/v1/files/create"
          )
          expect(wires[0].body).toMatchObject({
            file_type: "image",
            payload: {
              backend: "comfyui",
              prompt: "Original image prompt",
              width: 640,
              height: 480,
              steps: 19,
              cfg_scale: 6,
              seed: 17
            },
            export: { format: "png", mode: "inline", async_mode: "sync" }
          })
        } else {
          expect(wires[0].url).toBe(
            originalConfig.serverUrl + "/api/v1/chat/completions"
          )
          const wire = JSON.stringify(wires[0].body.messages)
          expect(wire).toContain("Original seed question")
          expect(wire).toContain("Seed answer")
          expect(wire).toContain("Original owned Scene note")
          expect(wire).toContain("Continue the user's current thought")
          expect(wire).toContain("Use narrative prose style")
          expect(wires[0].body.save_to_db).toBe(false)
        }
        heldResponse.resolve(
          image ? imageResponse() : textResponse(" continued answer")
        )
        let outcome!: ChatSubmitResult
        await act(async () => {
          outcome = await pending!
        })
        expect(normalizeChatSubmitResult(outcome)).toMatchObject({
          status: "submitted"
        })
        expect(steeringTuple()).toEqual({
          mode: "none",
          forceNarrate: false,
          revision: original.revision + 1
        })
        expect(wires).toHaveLength(1)
        await consumer.assertOriginal()
        if (image)
          expect(
            useStoreMessageOption.getState().messages.at(-1)?.images
          ).toEqual(["data:image/png;base64,original-generated-image"])
        else
          expect(
            useStoreMessageOption
              .getState()
              .messages.map((message) => message.message)
          ).toEqual(["Original seed question", "Seed answer continued answer"])
      } finally {
        heldResponse.resolve(
          image ? imageResponse() : textResponse(" continued answer")
        )
        if (pending)
          await act(async () => {
            await pending
          })
        consumer.unmount()
      }
    }
  )
})

describe("retained tracked-character response dispatch", () => {
  it.each(["Stop", "chat setup failure"] as const)(
    "retains original steering after held real character hydration and %s before complete-v2",
    async (failure) => {
      const consumer = await mountLegacyCharacterConsumer()
      const original = await armSteering()
      const entered = deferred<void>(),
        held = deferred<void>()
      let originalSignal: AbortSignal | undefined
      let pending: Promise<ChatSubmitResult> | undefined
      let creationFailed = false
      physical.control.beforeFetch = async (url, init) => {
        const path = new URL(url).pathname
        if (
          path === "/api/v1/characters/27" ||
          path === "/api/v1/characters/27/"
        ) {
          expect(new Headers(init.headers).get("X-API-KEY")).toBe(
            originalConfig.apiKey
          )
          // The caller's actual Stop controller is the contract under review.
          // The catalog may finish through its shared unscoped cache request.
          originalSignal = consumer.mounted.result.current.getSignal()
          entered.resolve(undefined)
          await held.promise
        }
        if (path === "/api/v1/chats/" && failure === "chat setup failure") {
          creationFailed = true
          return Response.json(
            { detail: "Declared character chat setup failure" },
            { status: 503 }
          )
        }
        return undefined
      }
      try {
        const started = await consumer.start()
        pending = started.pending
        await awaitGate(entered.promise, pending)
        expect(steeringTuple()).toEqual(original)
        expect(wires).toEqual([])
        if (failure === "Stop")
          act(() =>
            consumer.mounted.result.current.actions.stopStreamingRequest()
          )
        if (failure === "Stop") expect(originalSignal?.aborted).toBe(true)
        held.resolve(undefined)
        let outcome!: ChatSubmitResult
        await act(async () => {
          outcome = await pending!
        })
        expect(normalizeChatSubmitResult(outcome)).toMatchObject({
          status: failure === "Stop" ? "skipped" : "failed"
        })
        expect(creationFailed).toBe(failure === "chat setup failure")
        expect(wires).toEqual([])
        expect(steeringTuple()).toEqual(original)
        await consumer.original.assertOriginal()
        expect(useActorStore.getState().applied).toBeNull()
      } finally {
        held.resolve(undefined)
        if (pending)
          await act(async () => {
            await pending
          })
        consumer.unmount()
      }
    }
  )

  it("preserves the intentional hydration fallback and sends original steering on the actual complete-v2 wire", async () => {
    const consumer = await mountLegacyCharacterConsumer()
    const original = await armSteering()
    const heldResponse = deferred<Response>()
    responseGate = heldResponse
    let hydrationFailed = false
    let pending: Promise<ChatSubmitResult> | undefined
    physical.control.beforeFetch = async (url) => {
      if (new URL(url).pathname === "/api/v1/characters/27") {
        hydrationFailed = true
        return Response.json(
          { detail: "Declared catalog unavailable" },
          { status: 503 }
        )
      }
      return undefined
    }
    try {
      pending = (await consumer.start()).pending
      await waitFor(() => expect(wires).toHaveLength(1))
      expect(hydrationFailed).toBe(true)
      expect(steeringTuple()).toEqual(original)
      expect(
        auxiliaryWires.some(
          (wire) => wire.url === originalConfig.serverUrl + "/api/v1/chats/"
        )
      ).toBe(true)
      expect(wires[0].url).toBe(
        originalConfig.serverUrl +
          "/api/v1/chats/legacy-character-chat/complete-v2?scope_type=global"
      )
      expect(wires[0].headers.get("X-API-KEY")).toBe(originalConfig.apiKey)
      expect(wires[0].body).toMatchObject({
        include_character_context: true,
        model: "test",
        continue_as_user: true,
        impersonate_user: false,
        force_narrate: true,
        stream: true,
        save_to_db: false
      })
      expect(wires[0].body).not.toHaveProperty("onDispatch")
      expect(wires[0].body).not.toHaveProperty("onResponseDispatch")
      heldResponse.resolve(textResponse("Original character answer"))
      let outcome!: ChatSubmitResult
      await act(async () => {
        outcome = await pending!
      })
      expect(normalizeChatSubmitResult(outcome)).toMatchObject({
        status: "submitted"
      })
      expect(wires).toHaveLength(1)
      expect(useStoreMessageOption.getState().messages.at(-1)?.message).toBe(
        "Original character answer"
      )
    } finally {
      heldResponse.resolve(textResponse("Original character answer"))
      if (pending)
        await act(async () => {
          await pending
        })
      consumer.unmount()
    }
  })

  it("forwards a local dispatch callback exactly once at the real character stream handoff", async () => {
    const original = await mountSeededTemporaryConsumer()
    const capture = captureActorSceneSend(
      original.mounted.result.current.selection
    )
    const actorSceneSend = await capture!.read()
    // An actual API creation ACK supplies this SDK resource ID. It is never
    // installed as an H1 owner/view or used to rekey the original private Scene.
    const created = await tldwClient.createChat(
      { character_id: 27 },
      { requestScope: actorSceneSend.requestScope }
    )
    expect(created.id).toBe("legacy-character-chat")
    const heldResponse = deferred<Response>()
    responseGate = heldResponse
    const dispatched = vi.fn()
    const signal = new AbortController().signal
    const options = {
      signal,
      requestScope: actorSceneSend.requestScope,
      onDispatch: dispatched
    }
    const receiving = (async () => {
      const chunks: unknown[] = []
      for await (const chunk of tldwClient.streamCharacterChatCompletion(
        created.id,
        {
          model: "test",
          save_to_db: false,
          include_character_context: true
        },
        options
      ))
        chunks.push(chunk)
      return chunks
    })()
    try {
      await waitFor(() => expect(wires).toHaveLength(1))
      expect(dispatched).toHaveBeenCalledOnce()
      expect(wires[0].headers.get("X-API-KEY")).toBe(originalConfig.apiKey)
      expect(wires[0].body).not.toHaveProperty("onDispatch")
      heldResponse.resolve(textResponse("Original SDK character answer"))
      expect(await receiving).toContainEqual({
        choices: [{ delta: { content: "Original SDK character answer" } }]
      })
      expect(dispatched).toHaveBeenCalledOnce()
      await original.assertOriginal()
    } finally {
      heldResponse.resolve(textResponse("Original SDK character answer"))
      await receiving
      original.unmount()
    }
  })
})
