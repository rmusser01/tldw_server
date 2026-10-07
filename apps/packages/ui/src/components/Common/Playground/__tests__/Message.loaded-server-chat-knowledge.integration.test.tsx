// @vitest-environment jsdom
/**
 * XP-02 (#3109): a server chat loaded through the H1 history selection keeps its
 * server message ids, so the server-backed message actions are offered and act
 * on the right server rows.
 *
 * The messages come from the real formatSelectedHistory, the projection that
 * Playground.tsx, Layout.tsx, WebLayout.tsx and routes/sidepanel-chat.tsx publish
 * for a loaded chat. Each chat surface renders them with the real
 * PlaygroundMessage and MessageActionsBar.
 */
import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type {
  HistoryNodeV1,
  HistorySelectionCaptureV1
} from "@/types/history-selection"
import { formatSelectedHistory } from "@/db/dexie/helpers"
import { PlaygroundChat } from "@/components/Option/Playground/PlaygroundChat"
import { SidePanelBody } from "@/components/Sidepanel/Chat/body"

const saveChatKnowledgeMock = vi.hoisted(() => vi.fn(async () => undefined))
const messageApiMock = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
  warning: vi.fn(),
  info: vi.fn()
}))
const chatState = vi.hoisted(() => ({
  messages: [] as unknown[],
  toggleMessagePinned: vi.fn()
}))
// A stable t: a new function per render re-runs effects that depend on it.
const translate = vi.hoisted(
  () =>
    (
      key: string,
      fallback?: string | ({ defaultValue?: string } & Record<string, unknown>),
      options?: Record<string, unknown>
    ) => {
      const values = typeof fallback === "object" && fallback ? fallback : options
      const template =
        typeof fallback === "string" ? fallback : fallback?.defaultValue ?? key
      return String(template).replace(/\{\{(\w+)\}\}/g, (_match, token) =>
        values?.[token] == null ? "" : String(values[token])
      )
    }
)

vi.mock("@/db/dexie/chat", () => ({ PageAssistDatabase: class {} }))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: translate })
}))

vi.mock("antd", async (importOriginal) => ({
  ...(await importOriginal<typeof import("antd")>()),
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  // Controlled popover: the trigger toggles it and the content shows only while open.
  Popover: ({
    children,
    content,
    open,
    onOpenChange
  }: {
    children: React.ReactNode
    content?: React.ReactNode
    open?: boolean
    onOpenChange?: (open: boolean) => void
  }) => (
    <>
      <span onClick={() => onOpenChange?.(!open)}>{children}</span>
      {open ? content : null}
    </>
  ),
  App: Object.assign(() => null, {
    useApp: () => ({ message: messageApiMock })
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, defaultValue: unknown) => [defaultValue, vi.fn()]
}))

vi.mock("@/components/Common/Markdown", () => ({
  default: ({ message }: { message: string }) => <div>{message}</div>
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: {
      hasChatKnowledgeSave: true,
      hasNotes: true,
      hasFlashcards: true,
      hasFeedbackExplicit: true,
      hasFeedbackImplicit: false
    }
  })
}))

vi.mock("@/hooks/useFeedback", () => ({
  useFeedback: () => ({
    thumb: null,
    detail: "",
    sourceFeedback: {},
    canSubmit: true,
    isSubmitting: false,
    showThanks: false,
    submitThumb: vi.fn(),
    submitDetail: vi.fn(),
    submitSourceThumb: vi.fn()
  })
}))

vi.mock("@/hooks/useImplicitFeedback", () => ({
  useImplicitFeedback: () => ({
    trackCopy: vi.fn(),
    trackSourcesExpanded: vi.fn(),
    trackSourceClick: vi.fn(),
    trackCitationUsed: vi.fn(),
    trackDwellTime: vi.fn()
  })
}))

vi.mock("@/hooks/useTTS", () => ({
  useTTS: () => ({ cancel: vi.fn(), isSpeaking: false, speak: vi.fn() })
}))

vi.mock("@/hooks/useTldwAudioStatus", () => ({
  useTldwAudioStatus: () => ({ healthState: "ready", voicesAvailable: true })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(async () => undefined),
    saveChatKnowledge: saveChatKnowledgeMock
  }
}))

vi.mock("@/store/ui-mode", () => ({
  useUiModeStore: (selector: (state: { mode: string }) => unknown) =>
    selector({ mode: "pro" })
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: () => ({ data: [], isFetched: true, refetch: vi.fn() })
}))

vi.mock("@tanstack/react-virtual", () => ({
  useVirtualizer: ({ count }: { count: number }) => ({
    getTotalSize: () => count * 120,
    getVirtualItems: () =>
      Array.from({ length: count }, (_unused, index) => ({
        index,
        key: `row-${index}`,
        start: index * 120
      })),
    measureElement: vi.fn(),
    scrollToIndex: vi.fn()
  })
}))

vi.mock("@/hooks/useConnectionState", () => ({ useIsConnected: () => true }))
vi.mock("@/hooks/useSelectedCharacter", () => ({ useSelectedCharacter: () => [null] }))
vi.mock("@/store/webui", () => ({ useWebUI: () => ({ ttsEnabled: false }) }))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn()
  })
}))

vi.mock("@/hooks/chat/useChatSettingsRecord", () => ({
  useChatSettingsRecord: () => ({ settings: null, updateSettings: vi.fn() })
}))

vi.mock("@/components/Common/ChatGreetingPicker", () => ({
  ChatGreetingPicker: () => null
}))

vi.mock("@/components/Option/Playground/PlaygroundEmpty", () => ({
  PlaygroundEmpty: () => null
}))

vi.mock("@/components/Sidepanel/Chat/empty", () => ({
  EmptySidePanel: () => null
}))

// Shape of native_history_owner_key() in tldw_Server_API/app/core/Chat/persistence_service.py.
const NATIVE_OWNER_KEY = `native-history-v1:sha256:${"b".repeat(64)}`
const SERVER_CHAT_ID = "server-chat-7"
const USER_ID = "srv-msg-user-7"
const ASSISTANT_ID = "srv-msg-assistant-7"
const ANSWER = "Use git reflog to find the old head."

const node = (
  id: string,
  parent_id: string | null,
  role: string
): HistoryNodeV1 => ({
  id,
  revision: `digest-${id}`,
  parent_id,
  role,
  settled: true,
  metadata: [],
  assets: []
})

/** A server chat opened from the sidebar, as captured by POST /api/v1/chat/conversations/{id}/history/selection. */
const loadedServerChatCapture = (): HistorySelectionCaptureV1 => {
  const nodes = [
    node(USER_ID, null, "user"),
    node(ASSISTANT_ID, USER_ID, "assistant")
  ]
  return {
    status: "captured",
    snapshot: {
      version: 1,
      owner_key: NATIVE_OWNER_KEY,
      conversation_id: SERVER_CHAT_ID,
      fences: { conversation: "c", history: "h", settings: "s" },
      nodes,
      source_digest: "source",
      interpretation_status: { kind: "parent_graph_v1" },
      storage_context_digest: "storage"
    },
    rows: nodes,
    selected_content: [
      { id: USER_ID, revision: `digest-${USER_ID}`, message: "How do I undo a rebase?", images: [], tool_calls: null, extra_metadata: null },
      { id: ASSISTANT_ID, revision: `digest-${ASSISTANT_ID}`, message: ANSWER, images: [], tool_calls: null, extra_metadata: null }
    ],
    view: {
      view_session_id: "view-7",
      owner_key: NATIVE_OWNER_KEY,
      conversation_id: SERVER_CHAT_ID,
      interpretation: { kind: "parent_graph_v1" },
      cursor: { kind: "after_message", message_id: ASSISTANT_ID },
      selection_revision: 1
    },
    purpose: "send",
    storage_context_digest: "storage"
  }
}

// The fields both chat surfaces read from their message hook to render a loaded server chat.
const loadedChatHookState = () => ({
  messages: chatState.messages,
  setMessages: vi.fn(),
  setHistory: vi.fn(),
  streaming: false,
  isProcessing: false,
  isSearchingInternet: false,
  isEmbedding: false,
  regenerateLastMessage: vi.fn(),
  editMessage: vi.fn(),
  deleteMessage: vi.fn(),
  toggleMessagePinned: chatState.toggleMessagePinned,
  createChatBranch: vi.fn(),
  stopStreamingRequest: vi.fn(),
  ttsEnabled: false,
  actionInfo: null,
  temporaryChat: false,
  historyId: "history-7",
  serverChatId: SERVER_CHAT_ID,
  serverChatCharacterId: null,
  serverChatLoadState: "loaded",
  serverChatLoadError: null,
  messageSteeringMode: "none",
  setMessageSteeringMode: vi.fn(),
  messageSteeringForceNarrate: false,
  setMessageSteeringForceNarrate: vi.fn(),
  clearMessageSteering: vi.fn(),
  createCompareBranch: vi.fn(),
  compareMode: false,
  compareFeatureEnabled: false,
  compareSelectionByCluster: {},
  setCompareSelectionForCluster: vi.fn(),
  compareActiveModelsByCluster: {},
  setCompareActiveModelsForCluster: vi.fn(),
  setCompareSelectedModels: vi.fn(),
  setSelectedModel: vi.fn(),
  setCompareMode: vi.fn(),
  sendPerModelReply: vi.fn(),
  compareCanonicalByCluster: {},
  setCompareCanonicalForCluster: vi.fn(),
  compareContinuationModeByCluster: {},
  setCompareContinuationModeForCluster: vi.fn(),
  setCompareParentForHistory: vi.fn(),
  compareSplitChats: {},
  setCompareSplitChat: vi.fn(),
  compareMaxModels: 3
})

vi.mock("@/hooks/useMessageOption", () => ({ useMessageOption: () => loadedChatHookState() }))
vi.mock("@/hooks/useMessage", () => ({ useMessage: () => loadedChatHookState() }))

const surfaces = [
  {
    surface: "Playground chat",
    renderChat: () =>
      render(
        <MemoryRouter>
          <PlaygroundChat />
        </MemoryRouter>
      ),
    pinsMessages: true
  },
  {
    surface: "sidepanel chat",
    renderChat: () => render(<SidePanelBody />),
    pinsMessages: false
  }
]

const messageCard = (role: "user" | "assistant") => {
  const card = screen
    .getAllByTestId("chat-message")
    .find((element) => element.dataset.role === role)
  if (!card) throw new Error(`no ${role} message rendered`)
  return card
}

const openMoreActions = (card: HTMLElement) => {
  const trigger = within(card)
    .getAllByRole("button", { name: "More actions" })
    .find((button) => button.dataset.testid !== "message-actions-overflow-chip")
  if (!trigger) throw new Error("no More actions menu")
  fireEvent.click(trigger)
}

// Each surface renders the full message tree; leave headroom over the 5s default under a loaded run.
describe.each(surfaces)("$surface on a loaded server chat (XP-02, #3109)", { timeout: 20_000 }, ({ renderChat, pinsMessages }) => {
  beforeEach(() => {
    chatState.messages = formatSelectedHistory(loadedServerChatCapture()).messages
    chatState.toggleMessagePinned.mockClear()
    saveChatKnowledgeMock.mockClear()
    messageApiMock.success.mockClear()
    messageApiMock.error.mockClear()
  })

  it("offers Save to Notes on the assistant reply and saves it against its server message", async () => {
    renderChat()
    const assistant = messageCard("assistant")

    openMoreActions(assistant)
    expect(within(assistant).getByRole("button", { name: "Save to Flashcards" })).toBeInTheDocument()
    fireEvent.click(within(assistant).getByRole("button", { name: "Save to Notes" }))

    // conversation_id + message_id are what the note's back-link points at.
    await waitFor(() =>
      expect(saveChatKnowledgeMock).toHaveBeenCalledWith(
        {
          conversation_id: SERVER_CHAT_ID,
          message_id: ASSISTANT_ID,
          snippet: ANSWER,
          make_flashcard: false
        },
        undefined
      )
    )
    expect(messageApiMock.success).toHaveBeenCalledWith("Saved to Notes")
    expect(messageApiMock.error).not.toHaveBeenCalled()
  })

  it("shows feedback on the assistant reply", () => {
    renderChat()

    expect(
      within(messageCard("assistant")).getByRole("button", { name: "Helpful" })
    ).toBeInTheDocument()
  })

  it("keeps the user turn's server message id", () => {
    renderChat()
    const user = messageCard("user")

    if (pinsMessages) {
      openMoreActions(user)
      fireEvent.click(within(user).getByRole("button", { name: "Pin" }))
      expect(chatState.toggleMessagePinned).toHaveBeenCalledWith(0)
    }
    expect(user).toHaveAttribute("data-server-message-id", USER_ID)
  })
})
