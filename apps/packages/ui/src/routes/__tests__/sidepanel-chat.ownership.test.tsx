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

const io = vi.hoisted(() => ({
  noop: () => {},
  t: (key: string, fallback?: string) =>
    typeof fallback === "string" ? fallback : key,
  user: "alice",
  sequence: 0,
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
    const selection = useSelectedModel()
    return new Proxy(state, {
      get: (target, key) =>
        key in selection
          ? Reflect.get(selection, key)
          : key in target
            ? Reflect.get(target, key)
            : key === "clearChat"
              ? () =>
                  useStoreMessageOption.setState({
                    messages: [],
                    history: [],
                    historyId: null,
                    serverChatId: null,
                    queuedMessages: []
                  })
              : String(key).startsWith("set") ||
                  ["stopStreamingRequest", "onSubmit"].includes(String(key))
                ? io.noop
                : null
    })
  }
}))
vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: () => io.scope()
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
  SidepanelHeaderSimple: () => null
}))
vi.mock("@/components/Sidepanel/Chat/ConnectionBanner", () => ({
  ConnectionBanner: () => null
}))
vi.mock("@/components/Sidepanel/Chat/Sidebar", () => ({
  SidepanelChatSidebar: (props: {
    tabs: { id: string; label: string }[]
    onSelectTab: (id: string) => void
    onNewTab: () => void
    onOpenLocalHistory: (id: string) => void
    onOpenServerChat: (chat: { id: string; title: string }) => void
  }) => (
    <nav>
      <button onClick={props.onNewTab}>New conversation</button>
      {props.tabs.map((tab) => (
        <button key={tab.id} onClick={() => props.onSelectTab(tab.id)}>
          {tab.label}
        </button>
      ))}
      <button onClick={() => props.onOpenLocalHistory("foreign")}>
        Open cached history
      </button>
      <button
        onClick={() =>
          props.onOpenServerChat({ id: "server-chat", title: "Server chat" })
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
