/**
 * XS-01 (#3105): opening a past chat from side-panel search must open it in
 * its own tab and leave every other tab's conversation alone.
 *
 * The history selection publishes a loaded chat into the shared chat store
 * (onCapture) before the side panel finishes opening it, so the test drives a
 * fake selection that captures immediately and holds the message list read.
 */
import { useStoreChatModelSettings } from "@/store/model"
import { useStoreMessageOption } from "@/store/option"
import {
  type SidepanelChatSnapshot,
  type SidepanelChatTab,
  useSidepanelChatTabsStore
} from "@/store/sidepanel-chat-tabs"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import SidepanelChat from "../sidepanel-chat"
import { chatDraftOwnerKey } from "@/hooks/useChatDraftOwner"

const io = vi.hoisted(() => ({
  noop: () => {},
  t: (key: string, fallback?: string) =>
    typeof fallback === "string" ? fallback : key,
  sequence: 0,
  data: new Map<string, unknown>(),
  server: vi.fn(),
  updateChat: vi.fn(),
  getChat: vi.fn(),
  updateHistory: vi.fn(),
  loadedTitle: undefined as string | null | undefined,
  failCapture: new Set<string>(),
  manifest: vi.fn(),
  load: vi.fn()
}))

/** The question and answer each fake server chat holds. */
const transcript = (chatId: string) => [`${chatId} question`, `${chatId} answer`]
const displayFor = (chatId: string): Pick<SidepanelChatSnapshot, "history" | "messages"> => {
  const [question, answer] = transcript(chatId)
  return {
    history: [
      { role: "user", content: question },
      { role: "assistant", content: answer }
    ],
    messages: [
      { id: `${chatId}-q`, role: "user", isBot: false, name: "You", message: question, sources: [] },
      { id: `${chatId}-a`, role: "assistant", isBot: true, name: "Assistant", message: answer, sources: [] }
    ]
  }
}


/** Declared native reader fixture; the real H1 controller is covered by ownership.test. */
const captureFor = (chatId: string) => {
  const display = displayFor(chatId)
  const rows = display.messages.map((message, index) => ({ id: message.id, revision: "1", parent_id: index ? display.messages[index - 1].id : null, role: message.role, settled: true }))
  return {
    status: "captured" as const, chatId,
    view: { owner_key: "owner", conversation_id: chatId, view_session_id: "view-" + chatId, interpretation: { kind: "parent_graph_v1" as const }, cursor: { kind: "after_message" as const, message_id: rows[1].id }, selection_revision: 0 },
    snapshot: { owner_key: "owner", conversation_id: chatId, nodes: rows }, rows,
    selected_content: display.messages.map(message => ({ id: message.id, revision: "1", message: message.message, images: [] }))
  }
}

vi.mock("@/hooks/chat/useHistorySelection", async () => {
  const ReactModule = await import("react")
  const Context = ReactModule.createContext<unknown>(null)
  type Capture = ReturnType<typeof captureFor>
  type Live = {
    owner: { kind: "native"; conversation_id: string; request_scope: unknown; owner_key: string; validate_lease: () => boolean } | null
    capture: Capture | null
    view: Capture["view"] | null
    status: "idle" | "ready"
  }
  const empty = (): Live => ({ owner: null, capture: null, view: null, status: "idle" })
  return {
    HistorySelectionContext: Context,
    useHistorySelectionContext: () => ReactModule.useContext(Context),
    useHistorySelection: (options: { onCapture?: (capture: Capture) => void } = {}) => {
      const onCapture = ReactModule.useRef(options.onCapture)
      onCapture.current = options.onCapture
      const [controller] = ReactModule.useState(() => {
        let live = empty()
        let epoch = 0
        const views = new Map<string, Live>()
        let active = "default"
        return {
          status: "idle",
          activate: (key: string) => {
            if (key === active) return
            views.set(active, live)
            epoch++
            active = key
            live = views.get(key) ?? empty()
          },
          reset: () => {
            epoch++
            live = empty()
          },
          beginLoad: () => {
            epoch++
          },
          fence: () => {
            const token = epoch
            return () => token === epoch
          },
          getSignal: () => new AbortController().signal,
          getReference: () =>
            live.capture ? { owner_key: "owner", conversation_id: live.capture.chatId } : null,
          getCurrent: () => live,
          prepareExpansionPath: async () => null,
          loadConversation: async (target: { serverChatId?: string | null; isCurrent?: () => boolean }, _reference?: unknown, onLoaded?: (receipt: { owner: NonNullable<Live["owner"]>; view: NonNullable<Live["view"]> }) => void) => {
            const chatId = target.serverChatId
            const token = ++epoch
            // The real hook reads its owner and history before it captures.
            await io.load(chatId)
            if (token !== epoch || target.isCurrent?.() === false) return false
            if (!chatId || io.failCapture.has(chatId)) {
              live = empty()
              return false
            }
            const capture = captureFor(chatId)
            live = {
              owner: { kind: "native", conversation_id: chatId, owner_key: "owner", request_scope: { config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "alice" }, validate_lease: () => true },
              capture, view: capture.view, status: "ready"
            }
            // Like the real hook, publish the capture before the load resolves.
            onCapture.current?.(capture)
            onLoaded?.({ owner: live.owner!, view: live.view! })
            return true
          }
        }
      })
      return controller
    }
  }
})
vi.mock("@/services/chat-history-selection", async original => ({
  ...await original<typeof import("@/services/chat-history-selection")>(),
  captureHistorySnapshot: (...args: unknown[]) => io.manifest(...args)
}))
vi.mock("@/hooks/useLoadLocalConversation", () => ({
  restoreReadableLocalComparison: async () => {}
}))
vi.mock("@/components/Common/Playground/HistorySelectionReview", () => ({
  HistorySelectionReview: () => null
}))
vi.mock("@/db/dexie/server-chat-mirror", () => ({
  serverChatMirrorOwnerKey: () =>
    `["http://chat.test","multi-user","manual",null,"alice",null]`,
  linkServerChatMirror: async ({ chatId }: { chatId: string }) => `local-${chatId}`,
  reconcileServerChatMirror: async () => ({ localIds: new Map() })
}))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: async (key: string) => io.data.get(key),
    set: async (key: string, value: unknown) => {
      io.data.set(key, value)
    },
    remove: async (key: string) => {
      io.data.delete(key)
    }
  }),
  safeStorageSerde: {
    deserializer: (value: unknown) =>
      typeof value === "string" ? JSON.parse(value) : value
  }
}))
vi.mock("@/hooks/useMessage", async () => {
  const { useSelectedModel } = await import("@/hooks/chat/useSelectedModel")
  const { useHistorySelectionContext } = await import("@/hooks/chat/useHistorySelection")
  return {
    useMessage: () => {
      const state = useStoreMessageOption()
      const selection = useSelectedModel()
      const historySelection = useHistorySelectionContext()
      return new Proxy(state, {
        get: (target, key) =>
          key in selection
            ? Reflect.get(selection, key)
            : key in target
              ? Reflect.get(target, key)
              : key === "clearChat"
                ? (beforeReset?: () => void) => {
                    beforeReset?.()
                    useStoreMessageOption.getState().startNewChatDraft()
                    historySelection?.reset()
                    useStoreMessageOption.setState({ messages: [], history: [], historyId: null, serverChatId: null, queuedMessages: [] })
                    return true
                  }
                : String(key).startsWith("set") ||
                    ["stopStreamingRequest", "onSubmit"].includes(String(key))
                  ? io.noop
                  : null
      })
    }
  }
})
vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: async () => ({
    requestScope: {
      config: { serverUrl: "http://chat.test", authMode: "multi-user" },
      userId: "alice"
    },
    scopeSignal: new AbortController().signal,
    scopeInvalidatedSignal: new AbortController().signal,
    release: () => {}
  })
}))
vi.mock("@/hooks/useBackgroundMessage", () => ({ default: () => null }))
vi.mock("@/hooks/useMigration", () => ({ useMigration: () => {} }))
vi.mock("@/hooks/useSmartScroll", () => ({
  useSmartScroll: () => ({
    containerRef: { current: null },
    autoScrollToBottom: io.noop
  })
}))
vi.mock("@/hooks/keyboard/useKeyboardShortcuts", () => ({
  useChatShortcuts: io.noop,
  useSidebarShortcuts: io.noop,
  useChatModeShortcuts: io.noop,
  useWebSearchShortcuts: io.noop
}))
vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionActions: () => ({ checkOnce: io.noop })
}))
vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: io.noop }))
vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({ warning: io.noop, error: io.noop })
}))
vi.mock("@/hooks/useCharacterGreeting", () => ({
  useCharacterGreeting: io.noop
}))
vi.mock("@/hooks/useTTS", () => ({ useTTS: () => ({ cancel: io.noop }) }))
vi.mock("@/hooks/useSelectedCharacter", () => ({
  useSelectedCharacter: () => [null]
}))
vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [null]
}))
vi.mock("@/hooks/useSetting", () => ({ useSetting: () => [100] }))
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: unknown, fallback: unknown) => [
    fallback,
    async () => {},
    { isLoading: false }
  ]
}))
vi.mock("@/store/ui-mode", () => ({
  useUiModeStore: (select: (state: unknown) => unknown) =>
    select({ mode: "pro" })
}))
vi.mock("@/store/artifacts", () => ({
  useArtifactsStore: (select: (state: unknown) => unknown) =>
    select({ isOpen: false, closeArtifact: io.noop })
}))
vi.mock("@/db/dexie/helpers", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/db/dexie/helpers")>()),
  generateID: () => `new-${++io.sequence}`,
  getTitleById: async () => "",
  getRecentChatFromCopilot: async () => null,
  getFullChatData: async () => null,
  formatSelectedHistory: (capture: { chatId: string }) => displayFor(capture.chatId),
  updateHistory: (...args: unknown[]) => io.updateHistory(...args)
}))
vi.mock("@/services/app", () => ({ copilotResumeLastChat: async () => false }))
vi.mock("@/services/web-clipper/enrichment", () => ({
  readPendingWebClipAnalyzeRequest: () => null,
  clearPendingWebClipAnalyzeRequest: io.noop
}))
vi.mock("@/components/Sidepanel/Chat/body", () => ({
  SidePanelBody: () => (
    <div>
      {useStoreMessageOption((state) => state.messages).map((item, index) => (
        <p key={index}>{item.message}</p>
      ))}
    </div>
  )
}))
vi.mock("@/components/Sidepanel/Chat/form", () => ({
  SidepanelForm: () => null
}))
vi.mock("@/components/Sidepanel/Chat/SidepanelHeaderSimple", () => ({
  SidepanelHeaderSimple: (props: {
    onRenameTitle?: (title: string) => void
    loadEditableTitle?: () => Promise<string | null>
  }) => {
    const [editing, setEditing] = React.useState(false)
    return (
    <>
      <button onClick={() => setEditing(true)}>Edit header title</button>
      {editing && <input aria-label="Header title draft" defaultValue="Uncommitted name" />}
      <button onClick={() => props.onRenameTitle?.("Renamed in header")}>Rename from header</button>
      <button
        onClick={() => {
          void props.loadEditableTitle?.().then((title) => {
            io.loadedTitle = title
          })
        }}>
        Load header title
      </button>
    </>
  )
  }
}))
vi.mock("@/components/Sidepanel/Chat/ConnectionBanner", () => ({
  ConnectionBanner: () => null
}))
vi.mock("@/components/Sidepanel/Chat/Sidebar", () => ({
  SidepanelChatSidebar: (props: {
    tabs: { id: string; label: string }[]
    onSelectTab: (id: string) => void
    onOpenServerChat: (chat: { id: string; title: string }) => void
  }) => (
    <nav>
      {props.tabs.map((tab) => (
        <button key={tab.id} onClick={() => props.onSelectTab(tab.id)}>
          {tab.label}
        </button>
      ))}
      {["chat-1", "chat-2"].map((chatId) => (
        <button
          key={chatId}
          onClick={() => props.onOpenServerChat({ id: chatId, title: `Title ${chatId}` })}>
          Open {chatId}
        </button>
      ))}
    </nav>
  )
}))
vi.mock("@/components/Common/CommandPaletteHost", () => ({
  CommandPaletteHost: () => null
}))
vi.mock("@/components/Common/CommandPalette", () => ({
  CommandPalette: () => null
}))
vi.mock("@/components/Timeline", () => ({ TimelineModal: () => null }))
vi.mock("@/components/Sidepanel/Notes/NoteQuickSaveModal", () => ({
  default: () => null
}))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: io.t }) }))
vi.mock(
  "@plasmohq/storage",
  async () =>
    import("../../../../../tldw-frontend/extension/shims/plasmo-storage")
)
vi.mock("@/services/tldw/deployment-mode", () => ({
  isHostedTldwDeployment: () => false
}))
vi.mock("@/utils/browser-runtime", () => ({ isExtensionRuntime: () => false }))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: async () => {},
    listChatMessages: (...args: unknown[]) => io.server(...args),
    updateChat: (...args: unknown[]) => io.updateChat(...args),
    getChat: (...args: unknown[]) => io.getChat(...args),
    ensureConfigForRequest: async () => ({
      serverUrl: "http://chat.test",
      authMode: "multi-user"
    })
  }
}))
vi.mock("wxt/browser", () => ({
  browser: { runtime: { sendMessage: async () => ({ tabId: 7 }) } }
}))

const ownerKey = `["http://chat.test","multi-user","manual",null,"alice",null]`
const storedKey = `sidepanelChatTabsState:v2:${encodeURIComponent(ownerKey)}:tab-7`

const snapshotFor = (chatId: string | null): SidepanelChatSnapshot => ({
  ...(chatId ? displayFor(chatId) : { history: [], messages: [] }),
  chatMode: "normal",
  historyId: chatId ? `local-${chatId}` : null,
  webSearch: false,
  toolChoice: "none",
  selectedModel: "llama:one",
  selectedSystemPrompt: null,
  selectedQuickPrompt: null,
  temporaryChat: false,
  useOCR: false,
  serverChatId: chatId,
  serverChatState: null,
  serverChatTopic: null,
  serverChatClusterId: null,
  serverChatSource: null,
  serverChatExternalRef: null,
  queuedMessages: [],
  modelSettings: {}
})
const tabFor = (id: string, chatId: string | null): SidepanelChatTab => ({
  id,
  label: chatId ? `Title ${chatId}` : "Scratch",
  labelSource: "manual",
  historyId: chatId ? `local-${chatId}` : null,
  serverChatId: chatId,
  serverChatTopic: null,
  updatedAt: 1
})
/** Two saved tabs: an unbound scratch tab and tab A, active, holding chat 1. */
const seedTabs = () =>
  io.data.set(storedKey, {
    version: 2,
    ownerKey,
    tabs: [tabFor("scratch", null), tabFor("tab-a", "chat-1")],
    activeTabId: "tab-a",
    snapshotsById: { scratch: snapshotFor(null), "tab-a": snapshotFor("chat-1") }
  })

const serverMessages = (chatId: string) => {
  const [question, answer] = transcript(chatId)
  return [
    { id: `${chatId}-q`, role: "user", content: question },
    { id: `${chatId}-a`, role: "assistant", content: answer, parent_message_id: `${chatId}-q` }
  ]
}
const held = () => {
  let release!: () => void
  const gate = new Promise<void>((resolve) => {
    release = resolve
  })
  return { gate, release }
}

/** Each tab's id, binding and the message texts in its saved snapshot. */
const savedTabs = () => {
  const { tabs, snapshotsById, activeTabId } = useSidepanelChatTabsStore.getState()
  return {
    activeTabId,
    tabs: tabs.map((tab) => ({
      id: tab.id,
      serverChatId: tab.serverChatId,
      messages: (snapshotsById[tab.id]?.messages ?? []).map((message) => message.message)
    }))
  }
}
const tabHolding = (chatId: string) =>
  savedTabs().tabs.filter((tab) => tab.serverChatId === chatId)

describe("side-panel tabs when opening a past chat (XS-01)", () => {
  beforeEach(() => {
    io.sequence = 0
    io.data.clear()
    io.failCapture.clear()
    io.load.mockReset().mockResolvedValue(undefined)
    vi.restoreAllMocks()
    io.manifest.mockReset().mockImplementation(async (owner: { conversation_id: string }) => captureFor(owner.conversation_id))
    io.server.mockReset().mockImplementation(async (chatId: string) => serverMessages(chatId))
    window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
    useSidepanelChatTabsStore.getState().clear()
    useStoreMessageOption.setState({ messages: [], history: [], historyId: null, serverChatId: null })
    useStoreChatModelSettings.getState().reset()
    seedTabs()
  })

  it("loads a legacy saved tab without replaying its one-time insertion command", async () => {
    const stored = io.data.get(storedKey) as { snapshotsById: Record<string, SidepanelChatSnapshot> }
    stored.snapshotsById["tab-a"].selectedQuickPrompt = "old tab command"
    render(<SidepanelChat />)
    await screen.findByText("chat-1 answer")
    expect(useStoreMessageOption.getState().selectedQuickPrompt).toBeNull()
  })

  it("keeps the current tab's conversation while the opened chat loads, then opens it in its own tab", async () => {
    render(<SidepanelChat />)
    await screen.findByText("chat-1 answer")
    const { gate, release } = held()
    io.server.mockImplementationOnce(async (chatId: string) => {
      await gate
      return serverMessages(chatId)
    })

    fireEvent.click(await screen.findByRole("button", { name: "Open chat-2" }))
    await waitFor(() => expect(io.server).toHaveBeenCalledWith("chat-2", expect.anything(), expect.anything()))
    await act(async () => {})

    // While chat 2 loads, tab A still holds chat 1 and nothing else.
    expect(tabHolding("chat-1")).toEqual([
      { id: "tab-a", serverChatId: "chat-1", messages: transcript("chat-1") }
    ])

    await act(async () => {
      release()
    })
    await waitFor(() => expect(tabHolding("chat-2")).toHaveLength(1))
    await act(async () => {})

    const opened = tabHolding("chat-2")[0]
    expect(opened.messages).toEqual(transcript("chat-2"))
    expect(opened.id).not.toBe("tab-a")
    expect(savedTabs().activeTabId).toBe(opened.id)
    expect(tabHolding("chat-1")).toEqual([
      { id: "tab-a", serverChatId: "chat-1", messages: transcript("chat-1") }
    ])
    expect(savedTabs().tabs.find((tab) => tab.id === "scratch")).toEqual({
      id: "scratch",
      serverChatId: null,
      messages: []
    })
    expect(savedTabs().tabs).toHaveLength(3)
    expect(screen.getByText("chat-2 answer")).toBeInTheDocument()
  })

  it("persists tab A with chat 1 after opening chat 2", async () => {
    render(<SidepanelChat />)
    await screen.findByText("chat-1 answer")
    fireEvent.click(await screen.findByRole("button", { name: "Open chat-2" }))
    await waitFor(() => expect(tabHolding("chat-2")).toHaveLength(1))
    act(() => {
      window.dispatchEvent(new Event("beforeunload"))
    })
    const persisted = io.data.get(storedKey) as {
      tabs: SidepanelChatTab[]
      snapshotsById: Record<string, SidepanelChatSnapshot>
    }
    const persistedTabA = persisted.snapshotsById["tab-a"]
    expect(persistedTabA.serverChatId).toBe("chat-1")
    expect(persistedTabA.messages.map((message) => message.message)).toEqual(transcript("chat-1"))
    const chat2Tabs = persisted.tabs.filter((tab) => tab.serverChatId === "chat-2")
    expect(chat2Tabs).toHaveLength(1)
    expect(persisted.snapshotsById[chat2Tabs[0].id].messages.map((message) => message.message)).toEqual(
      transcript("chat-2")
    )
  })

  it("switches to the tab already bound to a chat instead of opening another", async () => {
    render(<SidepanelChat />)
    await screen.findByText("chat-1 answer")
    fireEvent.click(await screen.findByRole("button", { name: "Scratch" }))
    await waitFor(() => expect(savedTabs().activeTabId).toBe("scratch"))
    fireEvent.click(await screen.findByRole("button", { name: "Open chat-1" }))
    await waitFor(() => expect(savedTabs().activeTabId).toBe("tab-a"))
    expect(savedTabs().tabs).toHaveLength(2)
    expect(io.server).not.toHaveBeenCalled()
  })

  it("returns to the previous tab and drops the new one when the chat cannot be opened", async () => {
    io.failCapture.add("chat-2")
    render(<SidepanelChat />)
    await screen.findByText("chat-1 answer")
    fireEvent.click(await screen.findByRole("button", { name: "Open chat-2" }))
    await waitFor(() => expect(savedTabs().activeTabId).toBe("tab-a"))
    await act(async () => {})
    expect(savedTabs().tabs.map((tab) => tab.id)).toEqual(["scratch", "tab-a"])
    expect(tabHolding("chat-1")[0].messages).toEqual(transcript("chat-1"))
    expect(screen.getByText("chat-1 answer")).toBeInTheDocument()
  })

  it("drops the half-opened tab when the user switches tabs before the chat loads", async () => {
    render(<SidepanelChat />)
    await screen.findByText("chat-1 answer")
    const { gate, release } = held()
    io.server.mockImplementationOnce(async (chatId: string) => {
      await gate
      return serverMessages(chatId)
    })
    fireEvent.click(await screen.findByRole("button", { name: "Open chat-2" }))
    await waitFor(() => expect(io.server).toHaveBeenCalled())
    fireEvent.click(screen.getByRole("button", { name: "Scratch" }))
    await act(async () => {
      release()
    })
    await act(async () => {})
    expect(savedTabs().activeTabId).toBe("scratch")
    expect(tabHolding("chat-2")).toEqual([])
    expect(tabHolding("chat-1")).toEqual([
      { id: "tab-a", serverChatId: "chat-1", messages: transcript("chat-1") }
    ])
    expect(savedTabs().tabs.find((tab) => tab.id === "scratch")?.messages).toEqual([])
  })

  it("reopens a cancelled preparation once without adopting its late result", async () => {
    render(<SidepanelChat />)
    await screen.findByText("chat-1 answer")
    const { gate, release } = held()
    io.server.mockImplementationOnce(async (chatId: string) => { await gate; return serverMessages(chatId) })
    fireEvent.click(await screen.findByRole("button", { name: "Open chat-2" }))
    await waitFor(() => expect(io.server).toHaveBeenCalled())
    expect(tabHolding("chat-2")).toEqual([])
    fireEvent.click(screen.getByRole("button", { name: "Scratch" }))
    await waitFor(() => expect(savedTabs().activeTabId).toBe("scratch"))
    fireEvent.click(await screen.findByRole("button", { name: "Open chat-2" }))
    await waitFor(() => expect(tabHolding("chat-2")).toHaveLength(1))
    const openedId = tabHolding("chat-2")[0].id
    await act(async () => { release() })
    expect(savedTabs().activeTabId).toBe(openedId)
    expect(tabHolding("chat-2")).toEqual([{ id: openedId, serverChatId: "chat-2", messages: transcript("chat-2") }])
    expect(tabHolding("chat-1")[0].messages).toEqual(transcript("chat-1"))
  })

  it.each(["source", "failed replacement"])("releases a retired server opening's loading state after selecting %s", async target => {
    if (target === "failed replacement") {
      const stored = io.data.get(storedKey) as { tabs: SidepanelChatTab[]; snapshotsById: Record<string, SidepanelChatSnapshot> }
      const address = { kind: "native" as const, id: "chat-3", ownerKey: "owner", scope: { type: "global" as const } }
      stored.tabs.push(tabFor("tab-c", "chat-3"))
      stored.snapshotsById["tab-c"] = { ...snapshotFor("chat-3"), chatDraftAddress: address, chatDraftHandle: { address, ownerHash: chatDraftOwnerKey({ config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "alice" }) } }
      io.failCapture.add("chat-3")
    }
    render(<SidepanelChat />)
    await screen.findByText("chat-1 answer")
    const { gate, release } = held()
    io.server.mockImplementationOnce(async (chatId: string) => { await gate; return serverMessages(chatId) })
    fireEvent.click(await screen.findByRole("button", { name: "Open chat-2" }))
    await waitFor(() => expect(io.server).toHaveBeenCalled())
    expect(useStoreMessageOption.getState().isLoading).toBe(true)
    fireEvent.click(screen.getByRole("button", { name: target === "source" ? "Title chat-1" : "Title chat-3" }))
    await act(async () => { release() })
    await waitFor(() => expect(useStoreMessageOption.getState().isLoading).toBe(false))
    expect(savedTabs().activeTabId).toBe("tab-a")
    expect(tabHolding("chat-2")).toEqual([])
    vi.spyOn(Date, "now").mockReturnValue(Date.now() + 60_000)
    await act(async () => { window.dispatchEvent(new Event("focus")); await new Promise(resolve => setTimeout(resolve, 420)) })
    expect(io.manifest).toHaveBeenCalledOnce()
  })

  it("suspends freshness through qualification of a saved target tab", async () => {
    const stored = io.data.get(storedKey) as { tabs: SidepanelChatTab[]; snapshotsById: Record<string, SidepanelChatSnapshot> }
    const address = { kind: "native" as const, id: "chat-2", ownerKey: "owner", scope: { type: "global" as const } }
    stored.tabs.push(tabFor("tab-b", "chat-2"))
    stored.snapshotsById["tab-b"] = { ...snapshotFor("chat-2"), chatDraftAddress: address, chatDraftHandle: { address, ownerHash: chatDraftOwnerKey({ config: { serverUrl: "http://chat.test", authMode: "multi-user" }, userId: "alice" }) } }
    render(<SidepanelChat />)
    await screen.findByText("chat-1 answer")
    const { gate, release } = held()
    io.load.mockImplementationOnce(async () => { await gate })
    fireEvent.click(await screen.findByRole("button", { name: "Title chat-2" }))
    await waitFor(() => expect(io.load).toHaveBeenCalledWith("chat-2"))
    expect(savedTabs().activeTabId).toBe("tab-a")
    vi.spyOn(Date, "now").mockReturnValue(Date.now() + 60_000)
    await act(async () => { window.dispatchEvent(new Event("focus")); await new Promise(resolve => setTimeout(resolve, 420)) })
    expect(io.manifest).not.toHaveBeenCalled()
    await act(async () => { release() })
    await waitFor(() => expect(savedTabs().activeTabId).toBe("tab-b"))
  })

  it("suspends source freshness while an owned target prepares", async () => {
    render(<SidepanelChat />)
    await screen.findByText("chat-1 answer")
    const { gate, release } = held()
    io.server.mockImplementationOnce(async (chatId: string) => { await gate; return serverMessages(chatId) })
    fireEvent.click(await screen.findByRole("button", { name: "Open chat-2" }))
    await waitFor(() => expect(io.server).toHaveBeenCalled())
    vi.spyOn(Date, "now").mockReturnValue(Date.now() + 60_000)
    await act(async () => { window.dispatchEvent(new Event("focus")); await new Promise(resolve => setTimeout(resolve, 420)) })
    expect(io.manifest).not.toHaveBeenCalled()
    await act(async () => { release() })
    await waitFor(() => expect(tabHolding("chat-2")).toHaveLength(1))
  })
})

describe("side-panel header rename (XS-07)", () => {
  beforeEach(() => {
    io.sequence = 0
    io.data.clear()
    io.failCapture.clear()
    io.load.mockReset().mockResolvedValue(undefined)
    vi.restoreAllMocks()
    io.manifest.mockReset().mockImplementation(async (owner: { conversation_id: string }) => captureFor(owner.conversation_id))
    io.server.mockReset().mockImplementation(async (chatId: string) => serverMessages(chatId))
    io.updateChat.mockReset().mockImplementation(async (_id: string, data: { title: string }) => ({ title: data.title }))
    io.updateHistory.mockReset().mockResolvedValue(undefined)
    io.getChat.mockReset()
    io.loadedTitle = undefined
    window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
    useSidepanelChatTabsStore.getState().clear()
    useStoreMessageOption.setState({ messages: [], history: [], historyId: null, serverChatId: null })
    seedTabs()
  })

  it("renames the active tab's server chat and local copy, not just the tab", async () => {
    render(<SidepanelChat />)
    await screen.findByText("chat-1 answer")
    fireEvent.click(screen.getByRole("button", { name: "Rename from header" }))
    await waitFor(() =>
      expect(useSidepanelChatTabsStore.getState().tabs.find((tab) => tab.id === "tab-a")?.label).toBe(
        "Renamed in header"
      )
    )
    expect(io.updateChat).toHaveBeenCalledWith(
      "chat-1",
      { title: "Renamed in header" },
      { requestScope: expect.objectContaining({ userId: "alice" }) }
    )
    expect(io.updateHistory).toHaveBeenCalledWith("local-chat-1", "Renamed in header")
  })

  it("keeps the tab's name when the server rename fails", async () => {
    io.updateChat.mockRejectedValue(new Error("HTTP 409"))
    render(<SidepanelChat />)
    await screen.findByText("chat-1 answer")
    fireEvent.click(screen.getByRole("button", { name: "Rename from header" }))
    await waitFor(() => expect(io.updateChat).toHaveBeenCalled())
    await act(async () => {})
    expect(useSidepanelChatTabsStore.getState().tabs.find((tab) => tab.id === "tab-a")?.label).toBe("Title chat-1")
    expect(io.updateHistory).not.toHaveBeenCalled()
  })

  it("renames the still-visible source while another chat prepares", async () => {
    render(<SidepanelChat />)
    await screen.findByText("chat-1 answer")
    const { gate, release } = held()
    io.server.mockImplementationOnce(async (chatId: string) => { await gate; return serverMessages(chatId) })
    fireEvent.click(await screen.findByRole("button", { name: "Open chat-2" }))
    await waitFor(() => expect(io.server).toHaveBeenCalled())
    fireEvent.click(screen.getByRole("button", { name: "Rename from header" }))
    await waitFor(() => expect(io.updateChat).toHaveBeenCalledWith("chat-1", { title: "Renamed in header" }, expect.anything()))
    expect(io.updateHistory).toHaveBeenCalledWith("local-chat-1", "Renamed in header")
    await act(async () => { release() })
    await waitFor(() => expect(tabHolding("chat-2")).toHaveLength(1))
    expect(useSidepanelChatTabsStore.getState().tabs.find(tab => tab.serverChatId === "chat-2")?.label).not.toBe("Renamed in header")
  })

  it("retires title editing when switching to a different tab with the same label", async () => {
    const stored = io.data.get(storedKey) as { tabs: SidepanelChatTab[] }
    stored.tabs[0].label = stored.tabs[1].label
    render(<SidepanelChat />)
    await screen.findByText("chat-1 answer")
    fireEvent.click(screen.getByRole("button", { name: "Edit header title" }))
    expect(screen.getByRole("textbox", { name: "Header title draft" })).toBeInTheDocument()
    fireEvent.click(screen.getAllByRole("button", { name: "Title chat-1" })[0])
    await waitFor(() => expect(savedTabs().activeTabId).toBe("scratch"))
    expect(screen.queryByRole("textbox", { name: "Header title draft" })).not.toBeInTheDocument()
    expect(io.updateChat).not.toHaveBeenCalled()
  })

  it("refuses a source title read that finishes after switching tabs", async () => {
    const { gate, release } = held()
    io.getChat.mockImplementation(async () => { await gate; return { id: "chat-1", title: "Departed full title" } })
    render(<SidepanelChat />)
    await screen.findByText("chat-1 answer")
    fireEvent.click(screen.getByRole("button", { name: "Load header title" }))
    await waitFor(() => expect(io.getChat).toHaveBeenCalled())
    fireEvent.click(screen.getByRole("button", { name: "Scratch" }))
    await waitFor(() => expect(savedTabs().activeTabId).toBe("scratch"))
    await act(async () => { release() })
    await waitFor(() => expect(io.loadedTitle).toBeNull())
    expect(io.updateChat).not.toHaveBeenCalled()
  })

  it("gives the header the active chat's full title to edit", async () => {
    const fullTitle = "Quarterly planning review for the northern region sales team"
    io.getChat.mockResolvedValue({ id: "chat-1", title: fullTitle })
    render(<SidepanelChat />)
    await screen.findByText("chat-1 answer")
    fireEvent.click(screen.getByRole("button", { name: "Load header title" }))
    await waitFor(() => expect(io.loadedTitle).toBe(fullTitle))
    expect(io.getChat).toHaveBeenCalledWith("chat-1", {
      requestScope: expect.objectContaining({ userId: "alice" })
    })
  })
})
