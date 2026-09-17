import React from "react"
import { act, renderHook } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
const mocks = vi.hoisted(() => ({
  history: vi.fn(),
  details: vi.fn(),
  files: vi.fn(),
  prompt: vi.fn()
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
  mocks.details.mockResolvedValue({})
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
    const selection = useHistorySelection()
    currentController = selection
    const load = useActualLoader(deps, {
      t: (key) => key,
      errorLogPrefix: "load",
      errorDefaultMessage: "failed"
    })
    return { selection, load }
  })
  return { ...mounted, deps, captures, admission, provider }
}

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
        : { id }
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
      resolveServicePromptScope: resolveScope,
      subscribeToServicePromptConfigChanges: staleSubscribe
    }
  })
  mocks.details.mockImplementation(async (id: string) =>
    id === "A" ? { id, server_chat_id: "A", server_scope_key: "scope" } : { id }
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
