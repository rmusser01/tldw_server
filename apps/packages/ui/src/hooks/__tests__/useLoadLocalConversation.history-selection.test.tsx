import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
const mocks = vi.hoisted(() => ({
  history: vi.fn(),
  details: vi.fn(),
  files: vi.fn(),
  prompt: vi.fn(),
  loadSnapshot: vi.fn(async () => ({ requestScope: {}, scopeSignal: new AbortController().signal, scopeInvalidatedSignal: new AbortController().signal, release: vi.fn() }))
}))
vi.mock("@/db/dexie/chat", () => ({
  PageAssistDatabase: class {
    getChatHistory = mocks.history
    getHistoryInfo = mocks.details
  }
}))
vi.mock("@/db/dexie/helpers", () => ({
  formatToMessage: (rows: any[]) => rows,
  formatToChatHistory: (rows: any[]) => rows,
  getPromptById: mocks.prompt,
  getSessionFiles: mocks.files
}))
vi.mock("@/services/model-settings", () => ({
  lastUsedChatModelEnabled: async () => false
}))
vi.mock("@/utils/update-page-title", () => ({ updatePageTitle: vi.fn() }))
vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: mocks.loadSnapshot,
  resolveServicePromptScope: async () => ({}),
  subscribeToServicePromptConfigChanges: () => () => {}
}))
vi.mock("@/db/dexie/server-chat-mirror", () => ({ serverChatMirrorOwnerKey: () => "scope" }))
vi.mock("@/hooks/chat/useHistorySelection", () => ({
  useHistorySelectionContext: () => null
}))
import { useLoadLocalConversation } from "../useLoadLocalConversation"
const deferred = () => {
  let resolve!: (value: any) => void
  const promise = new Promise((r) => {
    resolve = r
  })
  return { promise, resolve }
}
beforeEach(() => {
  vi.clearAllMocks()
  mocks.loadSnapshot.mockReset().mockResolvedValue({ requestScope: {}, scopeSignal: new AbortController().signal, scopeInvalidatedSignal: new AbortController().signal, release: vi.fn() })
  mocks.details.mockResolvedValue({ server_scope_key: "scope" })
  mocks.files.mockResolvedValue([])
})
it("does not install an older conversation after a new navigation finishes", async () => {
  const old = deferred()
  mocks.history.mockImplementation((id: string) =>
    id === "old" ? old.promise : Promise.resolve([{ id }])
  )
  const { result } = renderHook(() => {
    const [rows, setRows] = React.useState<any[]>([])
    const load = useLoadLocalConversation(
      {
        setServerChatId: vi.fn(),
        setHistoryId: vi.fn(),
        setHistory: vi.fn(),
        setMessages: setRows,
        setSelectedModel: vi.fn(),
        setSelectedSystemPrompt: vi.fn(),
        setSystemPrompt: vi.fn(),
        setContextFiles: vi.fn()
      },
      { t: (key) => key, errorLogPrefix: "load", errorDefaultMessage: "failed" }
    )
    return { rows, load }
  })
  let pending!: Promise<void>
  act(() => {
    pending = result.current.load("old")
  })
  await act(async () => {
    await result.current.load("new")
  })
  await act(async () => {
    old.resolve([{ id: "old" }])
    await pending
  })
  expect(result.current.rows).toEqual([{ id: "new" }])
})

it("does not restart an accepted effect load when its setters render the loaded messages", async () => {
  mocks.history.mockImplementation(async (id: string) => [{ id }])
  const attempts = vi.fn()
  const { result, unmount } = renderHook(() => {
    const [rows, setRows] = React.useState<unknown[]>([])
    const load = useLoadLocalConversation(
      {
        setServerChatId: () => {},
        setHistoryId: () => {},
        setHistory: () => {},
        setMessages: setRows,
        setSelectedModel: () => {},
        setSelectedSystemPrompt: () => {},
        setSystemPrompt: () => {},
        setContextFiles: () => {}
      },
      { t: (key) => key, errorLogPrefix: "load", errorDefaultMessage: "failed" }
    )
    React.useEffect(() => {
      // Bound the broken implementation so this regression cannot exhaust a worker.
      if (attempts.mock.calls.length >= 2) return
      attempts()
      void load("owned")
    }, [load])
    return rows
  })
  try {
    await waitFor(() => expect(result.current).toEqual([{ id: "owned" }]))
    await act(async () => { await Promise.resolve() })
    expect(attempts).toHaveBeenCalledTimes(1)
  } finally {
    unmount()
  }
})

it("uses the current message setter for a deliberate load after rerender", async () => {
  mocks.history.mockResolvedValue([{ id: "owned" }])
  const first = vi.fn<(rows: unknown[]) => void>()
  const second = vi.fn<(rows: unknown[]) => void>()
  const { result, rerender, unmount } = renderHook(({ sink }) =>
    useLoadLocalConversation(
      {
        setServerChatId: () => {},
        setHistoryId: () => {},
        setHistory: () => {},
        setMessages: sink,
        setSelectedModel: () => {},
        setSelectedSystemPrompt: () => {},
        setSystemPrompt: () => {},
        setContextFiles: () => {}
      },
      { t: (key) => key, errorLogPrefix: "load", errorDefaultMessage: "failed" }
    ),
    { initialProps: { sink: first } }
  )
  try {
    rerender({ sink: second })
    await act(async () => { await result.current("owned") })
    expect(second).toHaveBeenCalledWith([{ id: "owned" }])
    expect(first).not.toHaveBeenCalled()
  } finally {
    unmount()
  }
})

// Delay actual dynamic-module imports while mounting the production controller
// together with the navigation hook. Capture enforces the native lease and signal.
const mountNativeImportRace = async () => {
  vi.resetModules()
  let currentController: any
  const captures = vi.fn(
    async (owner: any, view: any, purpose: any, signal: AbortSignal) => {
      if (
        signal.aborted ||
        (owner.kind === "native" && !owner.validate_lease())
      ) {
        throw new Error("request_config_scope_changed")
      }
      const bound = {
        ...view,
        owner_key: owner.owner_key || `native-${owner.conversation_id}`
      }
      return {
        status: "captured",
        view: bound,
        snapshot: {
          version: 1,
          owner_key: bound.owner_key,
          conversation_id: owner.conversation_id,
          nodes: [],
          fences: { conversation: "1", history: "1", settings: "1" },
          source_digest: "source",
          storage_context_digest: "storage",
          interpretation_status: { kind: "parent_graph_v1" }
        },
        rows: [],
        selected_content: [],
        purpose,
        storage_context_digest: "storage"
      }
    }
  )
  const admission = vi.fn()
  const provider = vi.fn()
  const onCapture = vi.fn()
  vi.doMock("@/db/dexie/history-selection", () => ({
    ensureLocalProfileId: async () => "profile",
    loadHistoryBookmark: async () => null,
    saveHistoryBookmark: async () => {},
    loadHistoryTurnRecoveries: async () => [],
    dismissHistoryTurnRecovery: async () => {},
    getLocalHistoryOwner: async (id: string) => ({
      kind: "local",
      profile_id: "profile",
      owner_key: `local-${id}`,
      conversation_id: id
    })
  }))
  vi.doMock("@/services/chat-history-selection", () => ({
    captureHistorySnapshot: captures,
    confirmLegacyHistoryProjection: vi.fn(),
    appendSelectedUser: admission
  }))
  vi.doMock("@/services/tldw", () => ({
    tldwChat: { streamMessage: provider }
  }))
  vi.doMock("@/services/service-prompts", () => ({
    loadServicePromptSnapshot: mocks.loadSnapshot,
    resolveServicePromptScope: async ({ signal }: any) => {
      if (signal.aborted) throw new Error("aborted")
      return {}
    },
    subscribeToServicePromptConfigChanges: () => () => {}
  }))
  vi.doMock("@/db/dexie/server-chat-mirror", () => ({
    serverChatMirrorOwnerKey: () => "scope",
    linkServerChatMirror: async () => {}
  }))
  vi.doMock("@/hooks/chat/useHistorySelection", async () => ({
    ...(await vi.importActual<any>("@/hooks/chat/useHistorySelection")),
    useHistorySelectionContext: () => currentController
  }))
  const { useHistorySelection } =
    await import("@/hooks/chat/useHistorySelection")
  const { useLoadLocalConversation: useActualLoader } =
    await import("../useLoadLocalConversation")
  const deps = {
    setServerChatId: vi.fn(),
    setHistoryId: vi.fn(),
    setHistory: vi.fn(),
    setMessages: vi.fn(),
    setSelectedModel: vi.fn(),
    setSelectedSystemPrompt: vi.fn(),
    setSystemPrompt: vi.fn(),
    setContextFiles: vi.fn()
  }
  const mounted = renderHook(() => {
    const selection = useHistorySelection({ onCapture })
    currentController = selection
    const load = useActualLoader(deps, {
      t: (key) => key,
      errorLogPrefix: "load",
      errorDefaultMessage: "failed"
    })
    return { selection, load }
  })
  return { ...mounted, deps, captures, admission, provider, onCapture }
}

it("reopens an intentional profile-only fork when account scope is unavailable", async () => {
  const mounted = await mountNativeImportRace()
  mocks.loadSnapshot.mockRejectedValue(new Error("account_scope_unavailable"))
  mocks.details.mockResolvedValue({ id: "A", local_owner_key: "local-A", last_used_prompt: { prompt_content: "Local fork prompt" } })
  mocks.history.mockResolvedValue([])
  mocks.files.mockResolvedValue([{ id: "local-file" }])
  await act(async () => { expect(await mounted.result.current.load("A")).toBe(true) })
  expect(mocks.loadSnapshot).not.toHaveBeenCalled()
  expect(mounted.result.current.selection.getCurrent()).toMatchObject({
    status: "ready", owner: { kind: "local", conversation_id: "A" }
  })
  expect(mounted.onCapture).toHaveBeenCalledOnce()
  expect(mounted.deps.setHistoryId).toHaveBeenCalledWith("A")
  expect(mounted.deps.setSystemPrompt).toHaveBeenCalledWith("Local fork prompt")
  expect(mounted.deps.setContextFiles).toHaveBeenCalledWith([{ id: "local-file" }])
  expect(mounted.admission).not.toHaveBeenCalled()
  expect(mounted.provider).not.toHaveBeenCalled()
})

it.each([
  { id: "A", server_scope_key: "scope", local_owner_key: "local-A" },
  { id: "A", server_scope_key: "foreign", local_owner_key: "local-A" },
  { id: "A", server_chat_id: "A", local_owner_key: "local-A" },
  { id: "A", message_source: "server", local_owner_key: "local-A" },
  { id: "A" },
  null
])("keeps non-profile caches closed while account scope is unavailable: %j", async (details) => {
  const mounted = await mountNativeImportRace()
  mocks.loadSnapshot.mockRejectedValue(new Error("account_scope_unavailable"))
  mocks.details.mockResolvedValue(details && { ...details, last_used_prompt: { prompt_content: "Private cache prompt" } })
  mocks.history.mockResolvedValue([{ content: "Private cache transcript" }])
  await act(async () => { expect(await mounted.result.current.load("A")).toBe(false) })
  expect(mounted.captures).not.toHaveBeenCalled()
  expect(mounted.onCapture).not.toHaveBeenCalled()
  expect(mounted.deps.setHistoryId).not.toHaveBeenCalled()
  expect(mounted.deps.setSystemPrompt).not.toHaveBeenCalled()
  expect(mounted.deps.setMessages).not.toHaveBeenCalled()
})

it.each(["principal", "navigation"])("fences profile-only metadata completion after %s changes", async (boundary) => {
  const mounted = await mountNativeImportRace()
  const started = deferred(), finish = deferred()
  mocks.details.mockImplementationOnce(() => { started.resolve(undefined); return finish.promise })
  mocks.history.mockResolvedValue([])
  await act(async () => {
    const pending = mounted.result.current.load("A")
    await started.promise
    if (boundary === "principal") window.dispatchEvent(new Event("tldw:auth-principal-changed"))
    else {
      mocks.details.mockResolvedValue({ id: "B", local_owner_key: "local-B" })
      expect(await mounted.result.current.load("B")).toBe(true)
    }
    finish.resolve({ id: "A", local_owner_key: "local-A", last_used_prompt: { prompt_content: "Stale fork prompt" } })
    expect(await pending).toBe(false)
  })
  expect(mounted.captures.mock.calls.some(([owner]) => owner.conversation_id === "A")).toBe(false)
  expect(mounted.deps.setHistoryId).not.toHaveBeenCalledWith("A")
  expect(mounted.deps.setSystemPrompt).not.toHaveBeenCalledWith("Stale fork prompt")
})

it.each(["account-local", "profile-fork", "native"])(
  "does not publish a held %s navigation capture after restore cancellation",
  async (kind) => {
    const mounted = await mountNativeImportRace()
    mocks.details.mockResolvedValue({ id: "A", ...(kind === "profile-fork"
      ? { local_owner_key: "local-A" }
      : { server_scope_key: "scope", ...(kind === "native" ? { server_chat_id: "A" } : {}) }) })
    mocks.history.mockResolvedValue([])
    const started = deferred()
    const finish = deferred()
    const capture = mounted.captures.getMockImplementation()!
    mounted.captures.mockImplementationOnce(async (...args) => {
      started.resolve(undefined)
      await finish.promise
      return capture(...args)
    })
    const { usePlaygroundSessionStore } = await import("@/store/playground-session")
    await act(async () => {
      const pending = mounted.result.current.load("A")
      await started.promise
      usePlaygroundSessionStore.getState().clearSession()
      finish.resolve(undefined)
      expect(await pending).toBe(false)
    })
    expect(mounted.onCapture).not.toHaveBeenCalled()
    expect(mounted.result.current.selection.getCurrent().capture).toBeNull()
    expect(mounted.deps.setHistoryId).not.toHaveBeenCalled()
    expect(mounted.deps.setSystemPrompt).not.toHaveBeenCalled()
  }
)

it.each(["beginLoad", "ready-local-B"])(
  "rejects stale native import completion after %s navigation",
  async (navigation) => {
    const mounted = await mountNativeImportRace()
    const moduleGate = deferred(),
      entered = deferred(),
      destinationRows = deferred()
    vi.doMock("@/db/dexie/server-chat-mirror", async () => {
      entered.resolve(undefined)
      await moduleGate.promise
      return {
        serverChatMirrorOwnerKey: () => "scope",
        linkServerChatMirror: async () => {}
      }
    })
    mocks.details.mockImplementation(async (id: string) =>
      id === "A"
        ? { id, server_chat_id: "A", server_scope_key: "scope" }
        : { id, local_owner_key: `local-${id}` }
    )
    mocks.history.mockImplementation((id: string) =>
      id === "B" && navigation === "beginLoad"
        ? destinationRows.promise
        : Promise.resolve([])
    )
    const receipt = vi.fn()
    await act(async () => {
      const old = mounted.result.current.selection.loadConversation(
        { historyId: "A" },
        null,
        receipt
      )
      await entered.promise
      const destination = mounted.result.current.load("B")
      if (navigation === "ready-local-B") await destination
      const before = mounted.result.current.selection.getCurrent()
      moduleGate.resolve(undefined)
      expect(await old).toBe(false)
      expect(receipt).not.toHaveBeenCalled()
      expect(mounted.result.current.selection.getCurrent()).toBe(before)
      if (navigation === "beginLoad") {
        destinationRows.resolve([])
        await destination
      }
    })
    expect(mounted.result.current.selection.getCurrent().owner).toMatchObject({
      kind: "local",
      conversation_id: "B"
    })
    expect(mounted.deps.setHistoryId).toHaveBeenLastCalledWith("B")
    expect(
      mounted.captures.mock.calls.some(
        ([owner]) => owner.conversation_id === "A"
      )
    ).toBe(false)
    expect(mounted.admission).not.toHaveBeenCalled()
    expect(mounted.provider).not.toHaveBeenCalled()
  }
)

it("a superseded service-module import cannot install a subscription or resolve a scope", async () => {
  const mounted = await mountNativeImportRace()
  const moduleGate = deferred(),
    entered = deferred()
  const staleSubscribe = vi.fn(() => vi.fn())
  const resolveScope = vi.fn(async ({ signal }: any) => {
    if (signal.aborted) throw new Error("aborted")
    return {}
  })
  vi.doMock("@/services/service-prompts", async () => {
    entered.resolve(undefined)
    await moduleGate.promise
    return {
      loadServicePromptSnapshot: mocks.loadSnapshot,
      resolveServicePromptScope: resolveScope,
      subscribeToServicePromptConfigChanges: staleSubscribe
    }
  })
  mocks.details.mockImplementation(async (id: string) =>
    id === "A" ? { id, server_chat_id: "A", server_scope_key: "scope" } : { id, local_owner_key: `local-${id}` }
  )
  mocks.history.mockResolvedValue([])
  const receipt = vi.fn()
  await act(async () => {
    const old = mounted.result.current.selection.loadConversation(
      { historyId: "A" },
      null,
      receipt
    )
    await entered.promise
    await mounted.result.current.load("B")
    const destination = mounted.result.current.selection.getCurrent()
    expect(destination).toMatchObject({ status: "ready", error: null })
    moduleGate.resolve(undefined)
    expect(await old).toBe(false)
    expect(receipt).not.toHaveBeenCalled()
    expect(mounted.result.current.selection.getCurrent()).toBe(destination)
    expect(staleSubscribe).not.toHaveBeenCalled()
    expect(resolveScope).not.toHaveBeenCalled()
  })
  expect(mounted.admission).not.toHaveBeenCalled()
  expect(mounted.provider).not.toHaveBeenCalled()
})
