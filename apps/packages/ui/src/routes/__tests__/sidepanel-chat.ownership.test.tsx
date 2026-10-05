import { buildQueuedRequest } from "@/utils/chat-request-queue"
import { useSelectedModel } from "@/hooks/chat/useSelectedModel"
import { useStoreChatModelSettings } from "@/store/model"
import { useStoreMessageOption } from "@/store/option"
import {
  type SidepanelChatSnapshot,
  useSidepanelChatTabsStore
} from "@/store/sidepanel-chat-tabs"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import React from "react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import LegacySidepanelChat from "../../../../../tldw-frontend/extension/routes/sidepanel-chat"
import SidepanelChat from "../sidepanel-chat"
import { useHistorySelectionContext, type HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import { requestSettingsNavigation, SETTINGS_NAVIGATION_REQUEST_EVENT } from "@/utils/settings-return"
import type { HistoryBookmarkScope } from "@/db/dexie/types"
import type { HistoryOwnerV1 } from "@/services/chat-history-selection"
import type { HistoryCaptureRequestV1, HistoryViewSelectionV1 } from "@/types/history-selection"

const io = vi.hoisted(() => ({
  noop: () => {},
  t: (key: string, fallback?: string) =>
    typeof fallback === "string" ? fallback : key,
  user: "alice",
  sequence: 0,
  localId: "foreign",
  serverId: "server-chat",
  captureNative: false,
  controller: null as HistorySelectionController | null,
  data: new Map<string, unknown>(),
  read: vi.fn(),
  local: vi.fn(),
  scope: vi.fn(),
  server: vi.fn()
}))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: (key: string) => io.read(key),
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
vi.mock("@/hooks/useMessage", () => ({
  useMessage: () => {
    const state = useStoreMessageOption()
    const historySelection = useHistorySelectionContext()
    const selection = useSelectedModel()
    return new Proxy(state, {
      get: (target, key) =>
        key in selection
          ? Reflect.get(selection, key)
          : key in target
            ? Reflect.get(target, key)
            : key === "clearChat"
              ? (beforeReset?: () => void) => {
                  if (!requestSettingsNavigation("/")) return false
                  beforeReset?.()
                  useStoreMessageOption.getState().startNewChatDraft()
                  historySelection?.reset()
                  useStoreMessageOption.setState({
                    messages: [],
                    history: [],
                    historyId: null,
                    serverChatId: null,
                    queuedMessages: []
                  })
                  return true
                }
              : String(key).startsWith("set") ||
                  ["stopStreamingRequest", "onSubmit"].includes(String(key))
                ? io.noop
                : null
    })
  }
}))
vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: () => io.scope(),
  resolveServicePromptScope: async () => (await io.scope()).requestScope,
  subscribeToServicePromptConfigChanges: () => io.noop
}))
// Storage/capture I/O only; the mounted logical-view controller and route remain real.
vi.mock("@/db/dexie/chat", async original => {
  const actual = await original<typeof import("@/db/dexie/chat")>()
  return { ...actual, PageAssistDatabase: class extends actual.PageAssistDatabase {
    async getHistoryInfo(id: string) { return (await io.local(id))?.historyInfo }
  } }
})
vi.mock("@/db/dexie/history-selection", async original => ({
  ...await original<typeof import("@/db/dexie/history-selection")>(),
  ensureLocalProfileId: async () => "profile",
  getLocalHistoryOwner: async (id: string) => ({ kind: "local", profile_id: "profile", owner_key: "local-profile", conversation_id: id }),
  loadHistoryBookmark: async (scope: HistoryBookmarkScope, owner: { owner_key: string; conversation_id: string }) => io.data.get(`bookmark:${JSON.stringify([scope.profile_id, scope.client_session_id, owner.owner_key, owner.conversation_id])}`) ?? null,
  saveHistoryBookmark: async (scope: HistoryBookmarkScope, view: HistoryViewSelectionV1) => {
    io.data.set(`bookmark:${JSON.stringify([scope.profile_id, scope.client_session_id, view.owner_key, view.conversation_id])}`, { ...scope, view })
  },
  loadHistoryTurnRecoveries: async () => []
}))
vi.mock("@/services/chat-history-selection", async original => ({
  ...await original<typeof import("@/services/chat-history-selection")>(),
  captureHistorySnapshot: async (owner: HistoryOwnerV1, incoming: HistoryCaptureRequestV1["view"], purpose: string) => {
    if (owner.kind === "unavailable" || (owner.kind === "native" && (!io.captureNative || owner.conversation_id === "failed"))) throw new Error("unsupported_history_capability")
    const cached = owner.kind === "native" ? { historyInfo: { id: owner.conversation_id }, messages: [] } : await io.local(owner.conversation_id)
    if (!cached || cached.historyInfo.id !== owner.conversation_id) throw new Error("missing_conversation")
    const view = { ...incoming, owner_key: owner.owner_key || `native:${owner.conversation_id}` }
    const rows = cached.messages.map((row: { id: string; role: string }) => ({ id: row.id, revision: "1", role: row.role, parent_id: null, settled: true }))
    return { status: "captured", view, purpose, rows,
      selected_content: cached.messages.map((row: { id: string; content: string }) => ({ id: row.id, revision: "1", message: row.content, images: [] })),
      storage_context_digest: "storage",
      snapshot: { version: 1, owner_key: view.owner_key, conversation_id: owner.conversation_id, nodes: rows,
        fences: { conversation: "1", history: "1", settings: "1" }, source_digest: "source", storage_context_digest: "storage", interpretation_status: { kind: "parent_graph_v1" } }
    }
  }
}))
vi.mock("@/hooks/useBackgroundMessage", () => ({ default: () => null }))
vi.mock("@/db/dexie/fork-operations", async original => ({
  ...await original<typeof import("@/db/dexie/fork-operations")>(),
  findForkCandidate: async () => null,
  loadForkOperations: async () => []
}))
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
const storageWrite = vi.hoisted(() => async () => {})
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: unknown, fallback: unknown) => [
    fallback,
    storageWrite,
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
  getFullChatData: (id: string) => io.local(id)
}))
vi.mock("@/services/app", () => ({ copilotResumeLastChat: async () => false }))
vi.mock("@/services/web-clipper/enrichment", () => ({
  readPendingWebClipAnalyzeRequest: () => null,
  clearPendingWebClipAnalyzeRequest: io.noop
}))
vi.mock("@/components/Sidepanel/Chat/body", () => ({
  SidePanelBody: () => {
    io.controller = useHistorySelectionContext()
    return <div>
      {useStoreMessageOption((state) => state.messages).map((item, index) => (
        <p key={index}>{item.message}</p>
      ))}
    </div>
  }
}))
vi.mock("@/components/Sidepanel/Chat/form", () => ({
  SidepanelForm: () => null
}))
vi.mock("@/components/Sidepanel/Chat/SidepanelHeaderSimple", () => ({
  SidepanelHeaderSimple: () => null
}))
vi.mock("@/components/Sidepanel/Chat/ConnectionBanner", () => ({
  ConnectionBanner: () => null
}))
vi.mock("@/components/Sidepanel/Chat/Sidebar", () => ({
  SidepanelChatSidebar: (props: {
    tabs: { id: string; label: string }[]
    onSelectTab: (id: string) => void
    onCloseTab: (id: string) => void
    onNewTab: () => void
    onOpenLocalHistory: (id: string) => void
    onOpenServerChat: (chat: { id: string; title: string }) => void
  }) => (
    <nav>
      <button onClick={props.onNewTab}>New conversation</button>
      {props.tabs.map((tab) => (
        <React.Fragment key={tab.id}><button onClick={() => props.onSelectTab(tab.id)}>
          {tab.label}
        </button><button aria-label={`Close ${tab.label}`} onClick={() => props.onCloseTab(tab.id)}>Close</button></React.Fragment>
      ))}
      <button onClick={() => props.onOpenLocalHistory(io.localId)}>
        Open cached history
      </button>
      <button
        onClick={() =>
          props.onOpenServerChat({ id: io.serverId, title: "Server chat" })
        }>
        Open server history
      </button>
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
    ensureConfigForRequest: async () => ({
      serverUrl: "http://chat.test",
      authMode: "multi-user"
    })
  }
}))
vi.mock("wxt/browser", () => ({
  browser: { runtime: { sendMessage: async () => ({ tabId: 7 }) } }
}))

const ownerKey = (user: string) =>
  `["http://chat.test","multi-user","manual",null,"${user}",null]`
const storedKey = (user: string) =>
  `sidepanelChatTabsState:v2:${encodeURIComponent(ownerKey(user))}:tab-7`
const snapshot = (text = "ALICE TRANSCRIPT"): SidepanelChatSnapshot => ({
  history: [{ role: "user", content: text }],
  messages: [
    { role: "user", isBot: false, name: "User", message: text, sources: [] }
  ],
  chatMode: "normal",
  historyId: null,
  webSearch: false,
  toolChoice: "none",
  selectedModel: "llama:one",
  selectedSystemPrompt: null,
  selectedQuickPrompt: null,
  temporaryChat: false,
  useOCR: false,
  serverChatId: null,
  serverChatState: null,
  serverChatTopic: null,
  serverChatClusterId: null,
  serverChatSource: null,
  serverChatExternalRef: null,
  queuedMessages: [],
  modelSettings: { systemPrompt: "ALICE PRIVATE PROMPT" }
})
const tabsState = (user = "alice") => ({
  version: 2,
  ownerKey: ownerKey(user),
  tabs: [
    {
      id: `${user}-one`,
      label: `${user} one`,
      labelSource: "manual",
      historyId: null,
      serverChatId: null,
      serverChatTopic: null,
      updatedAt: 1
    },
    {
      id: `${user}-two`,
      label: `${user} two`,
      labelSource: "manual",
      historyId: null,
      serverChatId: null,
      serverChatTopic: null,
      updatedAt: 1
    }
  ],
  activeTabId: `${user}-one`,
  snapshotsById: {
    [`${user}-one`]: snapshot(`${user} first`),
    [`${user}-two`]: snapshot(`${user} inactive`)
  }
})
const switchAccount = (user: string) =>
  act(() => {
    io.user = user
    window.dispatchEvent(
      new CustomEvent("tldw:config-updated", {
        detail: { authorityChanged: true }
      })
    )
  })

const completedReplyFixture = () => {
  const saved = tabsState()
  const stale = saved.snapshotsById["alice-one"]
  stale.historyId = "owned-history"
  stale.serverChatId = "owned-chat"
  stale.messages = [{ id: "reply-id", isBot: true, role: "assistant", name: "Character",
    message: "1\n2\n▋", sources: [], parentMessageId: "user-id" }]
  stale.queuedMessages = [buildQueuedRequest({ id: "queue-id", clientRequestId: "client-id",
    conversationId: "owned-chat", promptText: "Retained queued request" })]
  const mirror = {
    historyInfo: { id: "owned-history", server_chat_id: "owned-chat", server_scope_key: ownerKey("alice") },
    messages: [{ id: "reply-id", history_id: "owned-history", role: "assistant", name: "Character",
      content: "1\n2\n3\n4", parent_message_id: "user-id", createdAt: 1 }]
  }
  return { saved, stale, mirror }
}

describe.each([
  ["shared", SidepanelChat],
  ["legacy", LegacySidepanelChat]
] as const)("%s sidepanel snapshot ownership", (_name, Route) => {
  beforeEach(() => {
    io.user = "alice"
    io.sequence = 0
    io.localId = "foreign"
    io.serverId = "server-chat"
    io.captureNative = false
    io.controller = null
    io.data.clear()
    io.read.mockReset().mockImplementation(async (key) => io.data.get(key))
    io.local.mockReset()
    io.server.mockReset()
    io.scope
      .mockReset()
      .mockImplementation(async () => ({
        requestScope: {
          config: { serverUrl: "http://chat.test", authMode: "multi-user" },
          userId: io.user
        },
        scopeSignal: new AbortController().signal,
        scopeInvalidatedSignal: new AbortController().signal,
        release: io.noop
      }))
    window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
    useSidepanelChatTabsStore.getState().clear()
    useStoreChatModelSettings.getState().reset()
  })

  it("offers restore Retry without overwriting unread owned tabs", async () => {
    const saved = tabsState()
    io.data.set(storedKey("alice"), saved)
    io.read.mockImplementation(async key => {
      if (key === storedKey("alice")) throw new Error("Disk temporarily unavailable")
      return io.data.get(key)
    })
    render(<Route />)
    const alert = await screen.findByRole("alert")
    expect(alert).toHaveTextContent("Could not restore your saved chat. Retry to load it.")
    expect(screen.queryByLabelText("Restoring previous chat")).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "New conversation" }))
    expect(io.data.get(storedKey("alice"))).toEqual(saved)
    expect(useSidepanelChatTabsStore.getState().tabs).toEqual([])
    io.read.mockImplementation(async key => io.data.get(key))
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    await screen.findByRole("button", { name: "alice one" })
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    expect(useStoreMessageOption.getState().messages[0]?.message).toBe("alice first")
    expect(useSidepanelChatTabsStore.getState().tabs).toHaveLength(2)
  })

  it.each(["account", "unmount"])("ignores a restore rejection after %s replacement", async boundary => {
    let rejectRead!: (error: Error) => void
    io.read.mockImplementation(key => key === storedKey("alice")
      ? new Promise((_resolve, reject) => { rejectRead = reject })
      : Promise.resolve(io.data.get(key)))
    const view = render(<Route />)
    await waitFor(() => expect(rejectRead).toBeTypeOf("function"))
    if (boundary === "account") switchAccount("bob")
    else view.unmount()
    await act(async () => { rejectRead(new Error("Late Alice read failure")) })
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it("recovers a completed owned reply from its stale streaming snapshot", async () => {
    const { saved, stale, mirror } = completedReplyFixture()
    io.data.set(storedKey("alice"), saved)
    io.local.mockResolvedValue(mirror)
    render(<Route />)
    await waitFor(() => expect(useStoreMessageOption.getState().messages[0]?.message).toBe("1\n2\n3\n4"))
    expect(useStoreMessageOption.getState().messages[0]?.id).toBe("reply-id")
    expect(useStoreMessageOption.getState().queuedMessages).toEqual(stale.queuedMessages)
    expect(useStoreMessageOption.getState().history).toEqual(stale.history)
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe("ALICE PRIVATE PROMPT")
  })

  it.each(["foreign", "unowned", "history", "conversation", "row-history", "id", "parent", "prefix", "unfinished", "missing", "acknowledged", "offline"])("retains the original snapshot for %s local recovery", async reason => {
    const { saved, stale, mirror } = completedReplyFixture()
    if (reason === "foreign") mirror.historyInfo.server_scope_key = ownerKey("bob")
    if (reason === "unowned") mirror.historyInfo.server_scope_key = ""
    if (reason === "history") mirror.historyInfo.id = "another-history"
    if (reason === "conversation") mirror.historyInfo.server_chat_id = "another-chat"
    if (reason === "row-history") mirror.messages[0].history_id = "another-history"
    if (reason === "id") mirror.messages[0].id = "another-reply"
    if (reason === "parent") mirror.messages[0].parent_message_id = "another-user"
    if (reason === "prefix") mirror.messages[0].content = "Changed saved content"
    if (reason === "unfinished") mirror.messages[0].content = "1\n2\n3\n▋"
    if (reason === "missing") mirror.messages = []
    if (reason === "acknowledged") stale.messages[0].serverMessageId = "already-saved"
    io.data.set(storedKey("alice"), saved)
    if (reason === "offline") io.local.mockRejectedValue(new Error("Mirror unavailable"))
    else io.local.mockResolvedValue(mirror)
    render(<Route />)
    await screen.findByRole("button", { name: "alice one" })
    await act(async () => {})
    expect(useStoreMessageOption.getState().messages).toEqual(stale.messages)
    expect(useStoreMessageOption.getState().queuedMessages).toEqual(stale.queuedMessages)
  })

  it.each(["account", "unmount"])("rejects a completed local reply arriving after %s", async boundary => {
    const { saved, mirror } = completedReplyFixture()
    io.data.set(storedKey("alice"), saved)
    let finish!: (value: typeof mirror) => void
    io.local.mockImplementation(() => new Promise(resolve => { finish = resolve }))
    const mounted = render(<Route />)
    await waitFor(() => expect(finish).toBeTypeOf("function"))
    if (boundary === "account") switchAccount("bob")
    else mounted.unmount()
    const before = useStoreMessageOption.getState().messages
    await act(async () => { finish(mirror) })
    expect(useStoreMessageOption.getState().messages).toEqual(before)
    expect(useStoreMessageOption.getState().messages.some(message => message.message === "1\n2\n3\n4")).toBe(false)
  })

  it("does not publish an unowned legacy snapshot or overwrite it", async () => {
    const legacy = tabsState()
    io.data.set("sidepanelChatTabsState:tab-7", legacy)
    render(<Route />)
    await screen.findByRole("button", { name: "Open cached history" })
    await act(async () => {})
    expect(useStoreMessageOption.getState().messages).toEqual([])
    expect(useStoreChatModelSettings.getState().systemPrompt).not.toBe(
      "ALICE PRIVATE PROMPT"
    )
    expect(io.data.get("sidepanelChatTabsState:tab-7")).toBe(legacy)
  })

  it("rejects foreign local history before publishing its content", async () => {
    io.local.mockResolvedValue({
      historyInfo: {
        id: "foreign",
        title: "Foreign private",
        server_scope_key: ownerKey("bob")
      },
      messages: [
        {
          id: "foreign-message",
          role: "user",
          content: "FOREIGN LOCAL TRANSCRIPT"
        }
      ]
    })
    render(<Route />)
    fireEvent.click(
      await screen.findByRole("button", { name: "Open cached history" })
    )
    await act(async () => {})
    expect(useStoreMessageOption.getState().messages).toEqual([])
  })

  it("restores and selects same-owner tabs, clears them for Bob and recovers Alice on return", async () => {
    io.data.set(storedKey("alice"), tabsState())
    render(<Route />)
    fireEvent.click(await screen.findByRole("button", { name: "alice two" }))
    await waitFor(() =>
      expect(useStoreMessageOption.getState().messages[0]?.message).toBe(
        "alice inactive"
      )
    )
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe(
      "ALICE PRIVATE PROMPT"
    )
    switchAccount("bob")
    await waitFor(() =>
      expect(
        useSidepanelChatTabsStore
          .getState()
          .tabs.some((tab) => tab.id.startsWith("alice"))
      ).toBe(false)
    )
    expect(useStoreMessageOption.getState().messages).toEqual([])
    expect(useStoreChatModelSettings.getState().systemPrompt).not.toBe(
      "ALICE PRIVATE PROMPT"
    )
    switchAccount("alice")
    await screen.findByRole("button", { name: "alice two" })
    expect(useStoreMessageOption.getState().messages[0]?.message).toBe(
      "alice inactive"
    )
  })

  it("restores owned tabs during StrictMode effect replay", async () => {
    io.data.set(storedKey("alice"), tabsState())
    render(
      <React.StrictMode>
        <Route />
      </React.StrictMode>
    )
    await screen.findByText("alice first")
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe(
      "ALICE PRIVATE PROMPT"
    )
    expect(useSidepanelChatTabsStore.getState().tabs).toHaveLength(2)
  })

  it("does not restart durable restore when the real model-selection setter changes", async () => {
    const saved = tabsState()
    io.read.mockImplementation(async (key) =>
      key === storedKey("alice") ? saved : undefined
    )
    render(<Route />)
    await screen.findByText("alice first")
    act(() => {
      useStoreMessageOption
        .getState()
        .setMessages(snapshot("LIVE UNSAVED CONTENT").messages)
      useStoreMessageOption.getState().setSelectedModel("llama:two")
    })
    await act(async () => {})
    expect(useStoreMessageOption.getState().selectedModel).toBe("llama:two")
    expect(useStoreMessageOption.getState().messages[0]?.message).toBe(
      "LIVE UNSAVED CONTENT"
    )
  })

  it("finishes a held owned restore when the real model-selection setter changes", async () => {
    let finish!: (value: unknown) => void
    io.read.mockImplementation((key) =>
      key === storedKey("alice")
        ? new Promise((resolve) => {
            finish = resolve
          })
        : Promise.resolve(undefined)
    )
    render(<Route />)
    await waitFor(() => expect(finish).toBeTypeOf("function"))
    act(() => useStoreMessageOption.getState().setSelectedModel("llama:two"))
    await act(async () => {
      finish(tabsState())
    })
    await screen.findByText("alice first")
  })

  it("restores the last selected owned tab after a fresh mount", async () => {
    io.data.set(storedKey("alice"), tabsState())
    const view = render(<Route />)
    fireEvent.click(await screen.findByRole("button", { name: "alice two" }))
    await screen.findByText("alice inactive")
    view.unmount()
    useSidepanelChatTabsStore.getState().clear()
    useStoreMessageOption.setState({
      history: [],
      messages: [],
      historyId: null
    })
    useStoreChatModelSettings.getState().reset()
    render(<Route />)
    await screen.findByText("alice inactive")
    expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-two")
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe(
      "ALICE PRIVATE PROMPT"
    )
  })

  it.each(["local", "server"])(
    "ignores a held %s response after selecting another owned tab",
    async (source) => {
      io.data.set(storedKey("alice"), tabsState())
      let finish!: (value: unknown) => void
      const pending = new Promise((resolve) => {
        finish = resolve
      })
      if (source === "local") io.local.mockReturnValueOnce(pending)
      else io.server.mockReturnValueOnce(pending)
      render(<Route />)
      await screen.findByText("alice first")
      fireEvent.click(
        screen.getByRole("button", {
          name:
            source === "local" ? "Open cached history" : "Open server history"
        })
      )
      fireEvent.click(screen.getByRole("button", { name: "alice two" }))
      await act(async () => {
        finish(
          source === "local"
            ? {
                historyInfo: {
                  id: "alice-old",
                  title: "Alice old",
                  server_scope_key: ownerKey("alice")
                },
                messages: [{ id: "old", role: "user", content: "ALICE LATE" }]
              }
            : [{ id: "old", role: "user", content: "ALICE LATE" }]
        )
      })
      expect(useStoreMessageOption.getState().messages[0]?.message).toBe(
        "alice inactive"
      )
      expect(useStoreMessageOption.getState().isLoading).toBe(false)
    }
  )

  it.each(["account", "unmount"])(
    "ignores a held owned storage read after %s",
    async (ending) => {
      let finish!: (value: unknown) => void
      io.read.mockImplementation((key) =>
        key === storedKey("alice")
          ? new Promise((resolve) => {
              finish = resolve
            })
          : Promise.resolve(io.data.get(key))
      )
      const view = render(<Route />)
      await waitFor(() => expect(finish).toBeTypeOf("function"))
      if (ending === "account") {
        switchAccount("bob")
        await waitFor(() =>
          expect(useSidepanelChatTabsStore.getState().ownerKey).toBe(
            ownerKey("bob")
          )
        )
      } else view.unmount()
      await act(async () => {
        finish(tabsState())
      })
      expect(useStoreMessageOption.getState().messages).toEqual([])
      expect(
        useSidepanelChatTabsStore
          .getState()
          .tabs.some((tab) => tab.id.startsWith("alice"))
      ).toBe(false)
      expect(useStoreChatModelSettings.getState().systemPrompt).not.toBe(
        "ALICE PRIVATE PROMPT"
      )
    }
  )

  it("rejects a late Alice identity result after switching to Bob during preflight", async () => {
    const alice = await io.scope()
    let finish!: (value: unknown) => void
    io.scope.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          finish = resolve
        })
    )
    io.data.set(storedKey("alice"), tabsState())
    render(<Route />)
    await waitFor(() => expect(finish).toBeTypeOf("function"))
    switchAccount("bob")
    await screen.findByRole("button", { name: "Open cached history" })
    await act(async () => {
      finish(alice)
    })
    expect(useSidepanelChatTabsStore.getState().ownerKey).toBe(ownerKey("bob"))
    expect(useStoreMessageOption.getState().messages).toEqual([])
    expect(io.data.get(storedKey("alice"))).toMatchObject({
      activeTabId: "alice-one"
    })
  })

  it.each(["identity", "storage"])(
    "preserves saved owner data when %s is temporarily unavailable",
    async (failure) => {
      const saved = tabsState()
      io.data.set(storedKey("alice"), saved)
      if (failure === "identity")
        io.scope.mockRejectedValueOnce(new Error("temporarily unavailable"))
      else io.read.mockRejectedValueOnce(new Error("temporarily unavailable"))
      render(<Route />)
      await act(async () => {})
      act(() => window.dispatchEvent(new Event("beforeunload")))
      expect(io.data.get(storedKey("alice"))).toBe(saved)
      expect(useStoreMessageOption.getState().messages).toEqual([])
    }
  )

  it("retains the active tab on same-owner rotation and a failed focus recheck", async () => {
    io.data.set(storedKey("alice"), tabsState())
    render(<Route />)
    fireEvent.click(await screen.findByRole("button", { name: "alice two" }))
    io.scope.mockRejectedValueOnce(new Error("offline"))
    act(() => {
      window.dispatchEvent(
        new CustomEvent("tldw:config-updated", {
          detail: { authorityChanged: false }
        })
      )
      window.dispatchEvent(new Event("focus"))
    })
    await act(async () => {})
    expect(useStoreMessageOption.getState().messages[0]?.message).toBe(
      "alice inactive"
    )
    expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-two")
  })

  it.each(["local", "server"])(
    "rejects a held %s history response after account replacement",
    async (source) => {
      let finish!: (value: unknown) => void
      const response = new Promise((resolve) => {
        finish = resolve
      })
      if (source === "local") io.local.mockReturnValueOnce(response)
      else io.server.mockReturnValueOnce(response)
      render(<Route />)
      fireEvent.click(
        await screen.findByRole("button", {
          name:
            source === "local" ? "Open cached history" : "Open server history"
        })
      )
      switchAccount("bob")
      await waitFor(() =>
        expect(useSidepanelChatTabsStore.getState().ownerKey).toBe(
          ownerKey("bob")
        )
      )
      await act(async () => {
        finish(
          source === "local"
            ? {
                historyInfo: {
                  id: "alice-old",
                  title: "Alice old",
                  server_scope_key: ownerKey("alice")
                },
                messages: [{ id: "old", role: "user", content: "ALICE LATE" }]
              }
            : [{ id: "old", role: "user", content: "ALICE LATE" }]
        )
      })
      expect(useStoreMessageOption.getState().messages).toEqual([])
      expect(
        useSidepanelChatTabsStore
          .getState()
          .tabs.some(
            (tab) =>
              tab.historyId === "alice-old" ||
              tab.serverChatId === "server-chat"
          )
      ).toBe(false)
    }
  )

  it("restores an owned local history and its title", async () => {
    io.localId = "owned"
    io.local.mockResolvedValue({
      historyInfo: {
        id: "owned",
        title: "Owned cached title",
        server_scope_key: ownerKey("alice")
      },
      messages: [
        {
          id: "owned-message",
          role: "user",
          content: "OWNED CACHED TRANSCRIPT"
        }
      ]
    })
    render(<Route />)
    fireEvent.click(
      await screen.findByRole("button", { name: "Open cached history" })
    )
    await screen.findByText("OWNED CACHED TRANSCRIPT")
    expect(useStoreMessageOption.getState().historyId).toBe("owned")
  })

  if (_name === "shared") it("loads a destination in its own logical view without overwriting the source tab", async () => {
    io.data.set(storedKey("alice"), tabsState())
    io.localId = "B"
    const data = (id: string) => ({ historyInfo: { id, title: id, server_scope_key: ownerKey("alice") },
      messages: [{ id: `${id}-message`, role: "user", content: `${id} TRANSCRIPT` }] })
    let release!: () => void
    const held = new Promise<void>(resolve => { release = resolve })
    let opening = false
    io.local.mockImplementation(async id => { if (id === "B" && opening) await held; return data(id) })
    render(<Route />)
    await screen.findByRole("button", { name: "alice one" })
    await act(async () => {
      await io.controller!.loadConversation({ historyId: "A" }, null)
      useStoreMessageOption.getState().setHistoryId("A")
      useStoreMessageOption.getState().setChatDraftAddress({ kind: "local", id: "A", profileId: "profile" })
    })
    await screen.findByText("A TRANSCRIPT")
    opening = true
    fireEvent.click(screen.getByRole("button", { name: "Open cached history" }))
    await waitFor(() => expect(io.local).toHaveBeenCalledWith("B"))
    expect(io.controller!.getCurrent().view?.conversation_id).toBe("A")
    expect(io.controller!.getCurrent().status).toBe("ready")
    expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-one")
    expect(useStoreMessageOption.getState().messages[0]?.message).toBe("A TRANSCRIPT")
    act(() => useStoreChatModelSettings.getState().setSystemPrompt("EDITED A SETUP"))
    await waitFor(() => expect(useSidepanelChatTabsStore.getState().snapshotsById["alice-one"].modelSettings?.systemPrompt).toBe("EDITED A SETUP"))
    await act(async () => { release(); await held })
    await screen.findByText("B TRANSCRIPT")
    fireEvent.click(screen.getByRole("button", { name: "alice one" }))
    await screen.findByText("A TRANSCRIPT")
    expect(useStoreChatModelSettings.getState().systemPrompt).toBe("EDITED A SETUP")
    expect(useStoreMessageOption.getState().chatDraftAddress).toMatchObject({ kind: "local", id: "A" })
    await waitFor(() => expect(io.controller!.getCurrent().view?.conversation_id).toBe("A"))
  })

  if (_name === "shared") it("keeps the live source owner when a second opening fails after native preparation", async () => {
    io.data.set(storedKey("alice"), tabsState())
    io.captureNative = true
    io.local.mockImplementation(async id => ({ historyInfo: { id, title: id, server_scope_key: ownerKey("alice") },
      messages: [{ id: `${id}-message`, role: "user", content: `${id} TRANSCRIPT` }] }))
    let release!: () => void
    const held = new Promise<void>(resolve => { release = resolve })
    io.server.mockImplementation(async () => { await held; return [] })
    render(<Route />)
    await screen.findByRole("button", { name: "alice one" })
    await act(async () => {
      await io.controller!.loadConversation({ historyId: "A" }, null)
      useStoreMessageOption.getState().setHistoryId("A")
      useStoreMessageOption.getState().setChatDraftAddress({ kind: "local", id: "A", profileId: "profile" })
    })
    await screen.findByText("A TRANSCRIPT")
    const source = io.controller!.getCurrent()
    const sourceCurrent = io.controller!.fence()
    fireEvent.click(screen.getByRole("button", { name: "Open server history" }))
    // The route lists messages only after B's native H1 capture is ready.
    await waitFor(() => expect(io.server).toHaveBeenCalledWith("server-chat", expect.anything(), expect.anything()))
    expect(io.controller!.getCurrent()).toBe(source)
    expect(sourceCurrent()).toBe(true)
    io.serverId = "failed"
    fireEvent.click(screen.getByRole("button", { name: "Open server history" }))
    await waitFor(() => expect(useStoreMessageOption.getState().isLoading).toBe(false))
    await act(async () => { release(); await held })
    expect(io.controller!.getCurrent()).toBe(source)
    expect(sourceCurrent()).toBe(true)
    expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-one")
    expect(useStoreMessageOption.getState().chatDraftAddress).toMatchObject({ kind: "local", id: "A" })
  })

  if (_name === "shared") it("resets the new logical tab after preserving the departing selection", async () => {
    io.data.set(storedKey("alice"), tabsState())
    io.local.mockResolvedValue({ historyInfo: { id: "A", title: "A" },
      messages: [{ id: "A-message", role: "user", content: "A TRANSCRIPT" }] })
    render(<Route />)
    await screen.findByRole("button", { name: "alice one" })
    await act(async () => {
      await io.controller!.open({ kind: "local", profile_id: "profile", owner_key: "local-profile", conversation_id: "A" }, null)
      useStoreMessageOption.getState().setHistoryId("A")
      useStoreMessageOption.getState().setChatDraftAddress({ kind: "local", id: "A", profileId: "profile" })
    })
    await screen.findByText("A TRANSCRIPT")
    fireEvent.click(screen.getByRole("button", { name: "New conversation" }))
    expect(useStoreMessageOption.getState().chatDraftAddress?.kind).toBe("fresh")
    expect(io.controller!.getCurrent().status).toBe("idle")
    act(() => { io.controller!.activate("alice-one") })
    expect(io.controller!.getCurrent().view?.conversation_id).toBe("A")
    expect(io.controller!.getCurrent().view?.cursor).toEqual({ kind: "after_message", message_id: "A-message" })
  })

  if (_name === "shared") it.each([1, 2])("keeps the active tab when closing %s tab(s) requires declined navigation", async count => {
    const saved = tabsState()
    saved.tabs = saved.tabs.slice(0, count)
    io.data.set(storedKey("alice"), saved)
    render(<Route />)
    await screen.findByRole("button", { name: "alice one" })
    const source = useSidepanelChatTabsStore.getState().getSnapshot("alice-one")
    const decline = (event: Event) => event.preventDefault()
    window.addEventListener(SETTINGS_NAVIGATION_REQUEST_EVENT, decline, { once: true })
    fireEvent.click(screen.getByRole("button", { name: "Close alice one" }))
    expect(useSidepanelChatTabsStore.getState().tabs).toHaveLength(count)
    expect(useSidepanelChatTabsStore.getState().activeTabId).toBe("alice-one")
    expect(useSidepanelChatTabsStore.getState().getSnapshot("alice-one")).toEqual(source)
  })

  it("does not persist an invalidated render into the replacement owner's durable key", async () => {
    io.data.set(storedKey("alice"), tabsState())
    render(<Route />)
    await screen.findByRole("button", { name: "alice two" })
    act(() => {
      useStoreMessageOption
        .getState()
        .setMessages(snapshot("ALICE PENDING CLOSURE").messages)
      io.user = "bob"
      window.dispatchEvent(
        new CustomEvent("tldw:config-updated", {
          detail: { authorityChanged: true }
        })
      )
      window.dispatchEvent(new Event("beforeunload"))
    })
    await screen.findByRole("button", { name: "Open cached history" })
    await act(async () => {})
    expect(JSON.stringify(io.data.get(storedKey("bob")))).not.toContain("ALICE")
    expect(JSON.stringify(io.data.get(storedKey("alice")))).not.toContain(
      "ALICE PENDING CLOSURE"
    )
  })

  it.each(["local", "server", "new"])(
    "keeps the initial owned restore intact when %s is requested during a held read",
    async (action) => {
      let finish!: (value: unknown) => void
      io.read.mockImplementation((key) =>
        key === storedKey("alice")
          ? new Promise((resolve) => {
              finish = resolve
            })
          : Promise.resolve(undefined)
      )
      render(<Route />)
      await waitFor(() => expect(finish).toBeTypeOf("function"))
      fireEvent.click(
        screen.getByRole("button", {
          name:
            action === "local"
              ? "Open cached history"
              : action === "server"
                ? "Open server history"
                : "New conversation"
        })
      )
      await act(async () => {
        finish(tabsState())
      })
      await screen.findByText("alice first")
      expect(io.local).not.toHaveBeenCalled()
      expect(io.server).not.toHaveBeenCalled()
    }
  )

  it("clears live Alice state when focus discovers Bob without a local auth event", async () => {
    io.data.set(storedKey("alice"), tabsState())
    render(<Route />)
    await screen.findByText("alice first")
    act(() => {
      io.user = "bob"
      window.dispatchEvent(new Event("focus"))
    })
    await waitFor(() =>
      expect(useSidepanelChatTabsStore.getState().ownerKey).toBe(
        ownerKey("bob")
      )
    )
    await act(async () => {})
    expect(useStoreMessageOption.getState().messages).toEqual([])
    expect(useStoreChatModelSettings.getState().systemPrompt).not.toBe(
      "ALICE PRIVATE PROMPT"
    )
    expect(JSON.stringify(io.data.get(storedKey("bob")))).not.toContain(
      "alice first"
    )
    expect(JSON.stringify(io.data.get(storedKey("bob")))).not.toContain(
      "ALICE PRIVATE PROMPT"
    )
  })
})
