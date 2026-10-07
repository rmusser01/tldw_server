/**
 * XP-08 (#3105): a side-panel tab bound to a server chat must notice turns
 * added to that chat elsewhere (the full page, another window) instead of
 * showing a stale transcript and forking the chat on the next send.
 *
 * A fake server holds each chat's message graph. The fake history selection
 * keeps a per-chat bookmark like the real one, so a reopened panel restores the
 * old leaf unless the panel refreshes it.
 */
import { useStoreChatModelSettings } from "@/store/model"
import { useStoreMessageOption } from "@/store/option"
import {
  type SidepanelChatSnapshot,
  type SidepanelChatTab,
  useSidepanelChatTabsStore
} from "@/store/sidepanel-chat-tabs"
import type { SidepanelSendGate } from "@/hooks/chat/sidepanel-send-gate"
import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import React from "react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import SidepanelChat from "../sidepanel-chat"

type Turn = { id: string; parent_id: string | null; role: "user" | "assistant"; text: string }

const io = vi.hoisted(() => ({
  noop: () => {},
  sequence: 0,
  data: new Map<string, unknown>(),
  /** Each server chat's messages, oldest first. */
  chats: new Map<string, Turn[]>(),
  /** The leaf each chat's history selection last showed (the real hook keeps this in Dexie). */
  bookmarks: new Map<string, string | null>(),
  /** Every manifest read the panel made to check a chat for new turns. */
  manifestReads: [] as string[],
  chooseFails: false,
  gate: null as React.MutableRefObject<SidepanelSendGate | null> | null,
  composerMessages: [] as string[]
}))

/** The server's text for a message. */
const textOf = (chatId: string, id: string) =>
  io.chats.get(chatId)?.find((turn) => turn.id === id)?.text ?? id

/** What the history selection captures for `chatId` with its view ending at `cursorId`. */
const captureFor = (chatId: string, cursorId: string | null, revision = 0) => {
  const turns = io.chats.get(chatId) ?? []
  const byId = new Map(turns.map((turn) => [turn.id, turn]))
  const rows: Turn[] = []
  for (let turn = cursorId ? byId.get(cursorId) : undefined; turn; turn = turn.parent_id ? byId.get(turn.parent_id) : undefined) {
    rows.unshift(turn)
  }
  const node = (turn: Turn) => ({
    id: turn.id,
    revision: "1",
    parent_id: turn.parent_id,
    role: turn.role,
    settled: true
  })
  return {
    status: "captured" as const,
    chatId,
    snapshot: { nodes: turns.map(node), owner_key: "owner", conversation_id: chatId },
    rows: rows.map(node),
    selected_content: rows.map((turn) => ({ id: turn.id, revision: "1", message: turn.text, images: [] })),
    view: {
      view_session_id: `view-${chatId}`,
      owner_key: "owner",
      conversation_id: chatId,
      interpretation: { kind: "parent_graph_v1" as const },
      cursor: cursorId
        ? { kind: "after_message" as const, message_id: cursorId }
        : { kind: "empty" as const },
      selection_revision: revision
    }
  }
}
type Capture = ReturnType<typeof captureFor>

/** The latest leaf: the last message nothing replies to. */
const tipOf = (chatId: string) => {
  const turns = io.chats.get(chatId) ?? []
  const parents = new Set(turns.map((turn) => turn.parent_id))
  return [...turns].reverse().find((turn) => !parents.has(turn.id))?.id ?? null
}

vi.mock("@/hooks/chat/useHistorySelection", async () => {
  const ReactModule = await import("react")
  const Context = ReactModule.createContext<unknown>(null)
  type Live = {
    owner: { kind: "native"; conversation_id: string; request_scope: unknown; validate_lease: () => boolean } | null
    capture: Capture | null
    view: Capture["view"] | null
    status: "idle" | "ready"
  }
  const empty = (): Live => ({ owner: null, capture: null, view: null, status: "idle" })
  return {
    HistorySelectionContext: Context,
    useHistorySelectionContext: () => ReactModule.useContext(Context),
    useHistorySelection: (options: { onCapture?: (capture: Capture) => void }) => {
      const onCapture = ReactModule.useRef(options.onCapture)
      onCapture.current = options.onCapture
      const [controller] = ReactModule.useState(() => {
        let live = empty()
        let epoch = 0
        let revision = 0
        const views = new Map<string, Live>()
        let active = "default"
        const install = (chatId: string, cursorId: string | null) => {
          const capture = captureFor(chatId, cursorId, ++revision)
          io.bookmarks.set(chatId, cursorId)
          live = {
            owner: { kind: "native", conversation_id: chatId, request_scope: {}, validate_lease: () => true },
            capture,
            view: capture.view,
            status: "ready"
          }
          onCapture.current?.(capture)
        }
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
          loadConversation: async (target: { serverChatId?: string | null }) => {
            const chatId = target.serverChatId
            const token = ++epoch
            await Promise.resolve()
            if (token !== epoch) return false
            if (!chatId) {
              live = empty()
              return false
            }
            // Like the real hook, restore the bookmarked leaf, else start at the latest one.
            install(chatId, io.bookmarks.has(chatId) ? io.bookmarks.get(chatId)! : tipOf(chatId))
            return true
          },
          choose: async (cursor: { kind: string; message_id?: string }) => {
            const chatId = live.capture?.chatId
            const token = ++epoch
            await Promise.resolve()
            if (token !== epoch || !chatId || io.chooseFails) return false
            install(chatId, cursor.kind === "after_message" ? cursor.message_id! : null)
            return true
          }
        }
      })
      return controller
    }
  }
})
vi.mock("@/services/chat-history-selection", () => ({
  // The panel reads the chat's message manifest (no message content) to check it.
  captureHistorySnapshot: async (owner: { conversation_id: string }, view: { cursor: { kind: string } }) => {
    io.manifestReads.push(owner.conversation_id)
    expect(view.cursor).toEqual({ kind: "empty" })
    return captureFor(owner.conversation_id, null)
  }
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
  return {
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
  useAntdNotification: () => ({ warning: io.noop, error: io.noop, success: io.noop, info: io.noop })
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
  formatSelectedHistory: (capture: Capture) => ({
    history: capture.rows.map((row) => ({ role: row.role, content: textOf(capture.chatId, row.id) })),
    messages: capture.rows.map((row) => ({
      id: row.id,
      role: row.role,
      isBot: row.role === "assistant",
      name: row.role === "assistant" ? "Assistant" : "You",
      message: textOf(capture.chatId, row.id),
      sources: []
    }))
  })
}))
vi.mock("@/services/app", () => ({ copilotResumeLastChat: async () => false }))
vi.mock("@/services/web-clipper/enrichment", () => ({
  readPendingWebClipAnalyzeRequest: () => null,
  clearPendingWebClipAnalyzeRequest: io.noop
}))
vi.mock("@/components/Sidepanel/Chat/body", () => ({
  SidePanelBody: () => (
    <div data-testid="transcript">
      {useStoreMessageOption((state) => state.messages).map((item, index) => (
        <p key={index}>{item.message}</p>
      ))}
    </div>
  )
}))
vi.mock("@/components/Sidepanel/Chat/form", async () => {
  const ReactModule = await import("react")
  const { SidepanelSendGateContext: GateContext } = await import("@/hooks/chat/sidepanel-send-gate")
  return {
    // The composer sends through useMessage, which asks the panel's send gate first.
    SidepanelForm: () => {
      io.gate = ReactModule.useContext(GateContext)
      return null
    }
  }
})
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
  }) => (
    <nav>
      {props.tabs.map((tab) => (
        <button key={tab.id} onClick={() => props.onSelectTab(tab.id)}>
          {tab.label}
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
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string | { defaultValue?: string }, options?: Record<string, unknown>) => {
      const template =
        typeof fallback === "string" ? fallback : fallback?.defaultValue ?? key
      const values = { ...(typeof fallback === "object" ? fallback : {}), ...options }
      return template.replace(/\{\{(\w+)\}\}/g, (_match, name) => String(values[name] ?? ""))
    }
  })
}))
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
    listChatMessages: async () => [],
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

/** Notice copy the e2e reproduction (sidepanel-p0.spec.ts, XP-08) accepts. */
const STALE_NOTICE = /updated in another (window|tab)|newer messages|out of date/i

/** A server chat with one question and its answer. */
const seedServerChat = (chatId: string) =>
  io.chats.set(chatId, [
    { id: `${chatId}-q1`, parent_id: null, role: "user", text: `${chatId} question` },
    { id: `${chatId}-a1`, parent_id: `${chatId}-q1`, role: "assistant", text: `${chatId} answer` }
  ])

/** Another client continues the chat from its latest message. */
const continueElsewhere = (chatId: string) => {
  const turns = io.chats.get(chatId)!
  const leaf = tipOf(chatId)
  turns.push(
    { id: `${chatId}-q2`, parent_id: leaf, role: "user", text: `${chatId} follow-up from the full page` },
    { id: `${chatId}-a2`, parent_id: `${chatId}-q2`, role: "assistant", text: `${chatId} reply from the full page` }
  )
}

const snapshotFor = (chatId: string | null, latestSeen?: string | null): SidepanelChatSnapshot => ({
  history: [],
  messages: chatId
    ? [
        { id: `${chatId}-q1`, isBot: false, name: "You", message: `${chatId} question`, sources: [] },
        { id: `${chatId}-a1`, isBot: true, name: "Assistant", message: `${chatId} answer`, sources: [] }
      ]
    : [],
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
  modelSettings: {},
  ...(latestSeen !== undefined ? { serverChatLatestSeenId: latestSeen } : {})
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

/**
 * A saved panel whose active tab shows chat-1 as it was when the panel closed:
 * the question and answer, with the history selection's bookmark on the answer.
 */
const seedSavedPanel = ({ active = "tab-a" }: { active?: string } = {}) => {
  io.bookmarks.set("chat-1", "chat-1-a1")
  io.data.set(storedKey, {
    version: 2,
    ownerKey,
    tabs: [tabFor("scratch", null), tabFor("tab-a", "chat-1")],
    activeTabId: active,
    snapshotsById: { scratch: snapshotFor(null), "tab-a": snapshotFor("chat-1", "chat-1-a1") }
  })
}

const transcript = () => within().map((node) => node.textContent)
const within = () => Array.from(screen.getByTestId("transcript").querySelectorAll("p"))

/** Let queued promise callbacks and React updates run. */
const settle = async () => {
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 0))
  })
}

/** Return focus to the panel, past the focus refresh's debounce and rate limit. */
const refocusPanel = async () => {
  const now = Date.now() + 60_000
  vi.spyOn(Date, "now").mockReturnValue(now)
  act(() => {
    window.dispatchEvent(new Event("focus"))
  })
  await act(async () => {
    await new Promise((resolve) => setTimeout(resolve, 600))
  })
}

describe("side-panel tabs that fall behind their server chat (XP-08)", () => {
  beforeEach(() => {
    io.sequence = 0
    io.data.clear()
    io.chats.clear()
    io.bookmarks.clear()
    io.manifestReads = []
    io.chooseFails = false
    io.gate = null
    io.composerMessages = []
    window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
    useSidepanelChatTabsStore.getState().clear()
    useStoreMessageOption.setState({ messages: [], history: [], historyId: null, serverChatId: null, streaming: false })
    useStoreChatModelSettings.getState().reset()
    seedServerChat("chat-1")
  })
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
  })

  it("shows the turns added elsewhere when the panel reopens", async () => {
    seedSavedPanel()
    continueElsewhere("chat-1")

    render(<SidepanelChat />)

    expect(await screen.findByText("chat-1 reply from the full page")).toBeInTheDocument()
    expect(transcript()).toEqual([
      "chat-1 question",
      "chat-1 answer",
      "chat-1 follow-up from the full page",
      "chat-1 reply from the full page"
    ])
    // The panel says why its transcript changed.
    expect(screen.getByText(STALE_NOTICE)).toBeInTheDocument()
    // The next send continues from the server's latest message.
    expect(io.bookmarks.get("chat-1")).toBe("chat-1-a2")
  })

  it("leaves a reopened tab alone when nothing changed on the server", async () => {
    seedSavedPanel()

    render(<SidepanelChat />)

    await screen.findByText("chat-1 answer")
    await settle()
    expect(transcript()).toEqual(["chat-1 question", "chat-1 answer"])
    expect(screen.queryByText(STALE_NOTICE)).not.toBeInTheDocument()
    expect(io.bookmarks.get("chat-1")).toBe("chat-1-a1")
  })

  it("pulls new server turns when the panel regains focus", async () => {
    seedSavedPanel()
    render(<SidepanelChat />)
    await screen.findByText("chat-1 answer")
    await settle()

    continueElsewhere("chat-1")
    await refocusPanel()

    expect(await screen.findByText("chat-1 reply from the full page")).toBeInTheDocument()
    expect(io.manifestReads).toEqual(["chat-1"])
    expect(io.bookmarks.get("chat-1")).toBe("chat-1-a2")
  })

  it("checks the server at most once per burst of focus events", async () => {
    seedSavedPanel()
    render(<SidepanelChat />)
    await screen.findByText("chat-1 answer")
    await settle()

    vi.spyOn(Date, "now").mockReturnValue(Date.now() + 60_000)
    act(() => {
      window.dispatchEvent(new Event("focus"))
      document.dispatchEvent(new Event("visibilitychange"))
      window.dispatchEvent(new Event("focus"))
    })
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 600))
    })
    act(() => {
      window.dispatchEvent(new Event("focus"))
    })
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 600))
    })

    expect(io.manifestReads).toEqual(["chat-1"])
  })

  it("keeps unsent local messages and shows an updated-elsewhere notice instead", async () => {
    seedSavedPanel()
    render(<SidepanelChat />)
    await screen.findByText("chat-1 answer")
    await settle()
    // A message that exists only in this tab (say, a failed send's error).
    act(() => {
      useStoreMessageOption.getState().setMessages((previous) => [
        ...previous,
        { id: "local-only", isBot: true, name: "Assistant", message: "Unsent local reply", sources: [] }
      ])
    })

    continueElsewhere("chat-1")
    await refocusPanel()

    const notice = await screen.findByRole("status", { name: /chat update/i })
    expect(notice).toHaveTextContent(STALE_NOTICE)
    expect(transcript()).toEqual(["chat-1 question", "chat-1 answer", "Unsent local reply"])
    expect(io.bookmarks.get("chat-1")).toBe("chat-1-a1")

    // Refreshing is the user's choice.
    fireEvent.click(screen.getByRole("button", { name: /refresh/i }))
    expect(await screen.findByText("chat-1 reply from the full page")).toBeInTheDocument()
    expect(io.bookmarks.get("chat-1")).toBe("chat-1-a2")
  })

  it("does not check tabs that have no server chat", async () => {
    seedSavedPanel({ active: "scratch" })
    render(<SidepanelChat />)
    await settle()

    await refocusPanel()
    const gate = io.gate?.current
    expect(gate).toBeTypeOf("function")
    await act(async () => {
      expect(await gate!({ message: "hello" })).toEqual({ proceed: true, refreshed: false })
    })

    expect(io.manifestReads).toEqual([])
    expect(screen.queryByText(STALE_NOTICE)).not.toBeInTheDocument()
  })

  it("does not keep a deliberately chosen earlier message from sending there", async () => {
    // The tab last synced at the latest answer and then chose to continue from
    // the question; nothing new arrived, so the choice stands.
    io.bookmarks.set("chat-1", "chat-1-q1")
    io.data.set(storedKey, {
      version: 2,
      ownerKey,
      tabs: [tabFor("tab-a", "chat-1")],
      activeTabId: "tab-a",
      snapshotsById: { "tab-a": snapshotFor("chat-1", "chat-1-a1") }
    })

    render(<SidepanelChat />)
    await screen.findByText("chat-1 question")
    await settle()
    await refocusPanel()

    expect(io.bookmarks.get("chat-1")).toBe("chat-1-q1")
    expect(screen.queryByText(STALE_NOTICE)).not.toBeInTheDocument()
  })

  describe("before a send", () => {
    const sendGate = async () => {
      await waitFor(() => expect(io.gate?.current).toBeTypeOf("function"))
      return io.gate!.current!
    }

    it("re-parents a stale tab's send onto the server's latest message", async () => {
      seedSavedPanel()
      render(<SidepanelChat />)
      await screen.findByText("chat-1 answer")
      await settle()
      continueElsewhere("chat-1")

      const gate = await sendGate()
      let result: Awaited<ReturnType<SidepanelSendGate>> | undefined
      await act(async () => {
        result = await gate({ message: "next question" })
      })

      expect(result).toEqual({ proceed: true, refreshed: true })
      // The send's history selection now ends at the latest server message, so
      // the new turn replies to it instead of forking from the old answer.
      expect(io.bookmarks.get("chat-1")).toBe("chat-1-a2")
      expect(await screen.findByText("chat-1 reply from the full page")).toBeInTheDocument()
      expect(screen.getByText(STALE_NOTICE)).toBeInTheDocument()
    })

    it("lets a current tab's send through without changing its view", async () => {
      seedSavedPanel()
      render(<SidepanelChat />)
      await screen.findByText("chat-1 answer")
      await settle()

      const gate = await sendGate()
      let result: Awaited<ReturnType<SidepanelSendGate>> | undefined
      await act(async () => {
        result = await gate({ message: "next question" })
      })

      expect(result).toEqual({ proceed: true, refreshed: false })
      expect(io.manifestReads).toEqual(["chat-1"])
      expect(io.bookmarks.get("chat-1")).toBe("chat-1-a1")
    })

    it("holds the send, keeps the message and offers Refresh when the tab cannot be refreshed", async () => {
      seedSavedPanel()
      render(<SidepanelChat />)
      await screen.findByText("chat-1 answer")
      await settle()
      continueElsewhere("chat-1")
      io.chooseFails = true
      const restored = vi.fn()
      window.addEventListener("tldw:set-composer-message", restored as EventListener)

      const gate = await sendGate()
      let result: Awaited<ReturnType<SidepanelSendGate>> | undefined
      await act(async () => {
        result = await gate({ message: "next question" })
      })
      window.removeEventListener("tldw:set-composer-message", restored as EventListener)

      expect(result).toEqual({ proceed: false, refreshed: false })
      expect(io.bookmarks.get("chat-1")).toBe("chat-1-a1")
      const notice = await screen.findByRole("alert")
      expect(notice).toHaveTextContent(STALE_NOTICE)
      expect(notice).toHaveTextContent(/wasn't sent/i)
      expect(screen.getByRole("button", { name: /refresh/i })).toBeInTheDocument()
      expect((restored.mock.calls[0]?.[0] as CustomEvent).detail).toMatchObject({ message: "next question" })
    })
  })
})
