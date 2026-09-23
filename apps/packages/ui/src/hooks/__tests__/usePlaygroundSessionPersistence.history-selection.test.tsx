// @vitest-environment jsdom
import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
import { resolveHistorySelection } from "@/utils/history-selection"

const mocks = vi.hoisted(() => ({
  bookmarks: new Map<string, any>(),
  details: vi.fn(),
  chatData: vi.fn(),
  files: vi.fn(),
  capture: vi.fn(),
  confirm: vi.fn(),
  profile: vi.fn(),
  link: vi.fn(),
  recent: vi.fn(),
  setPrompt: vi.fn(),
  setAssistant: vi.fn(),
  currentScope: "current"
}))
const bookmarkKey = (scope: any, owner: any) =>
  JSON.stringify([
    scope.profile_id,
    scope.client_session_id,
    owner.owner_key,
    owner.conversation_id
  ])
vi.mock("@/db/dexie/history-selection", () => ({
  ensureLocalProfileId: () => mocks.profile(),
  getLocalHistoryOwner: async (id: string) => ({
    kind: "local",
    profile_id: "profile",
    owner_key: "local-owner",
    conversation_id: id
  }),
  loadHistoryBookmark: async (scope: any, owner: any) =>
    mocks.bookmarks.get(bookmarkKey(scope, owner)) || null,
  saveHistoryBookmark: async (scope: any, view: any) =>
    mocks.bookmarks.set(bookmarkKey(scope, view), { ...scope, view })
}))
vi.mock("@/db/dexie/fork-operations", () => ({
  findForkCandidate: async () => null,
  loadForkOperations: async () => []
}))
vi.mock("@/db/dexie/chat", () => ({
  PageAssistDatabase: class {
    getHistoryInfo = mocks.details
  }
}))
vi.mock("@/db/dexie/helpers", () => ({
  getFullChatData: (...args: any[]) => mocks.chatData(...args),
  getSessionFiles: () => mocks.files(),
  getPromptById: async () => null,
  formatToMessage: (rows: any[]) =>
    rows.map((row) => ({ id: row.id, message: row.content, isBot: true })),
  formatToChatHistory: (rows: any[]) =>
    rows.map((row) => ({ role: row.role, content: row.content }))
}))
vi.mock("@/services/chat-history-selection", () => ({
  captureHistorySnapshot: (...args: any[]) => mocks.capture(...args),
  confirmLegacyHistoryProjection: (...args: any[]) => mocks.confirm(...args)
}))
vi.mock("@/services/service-prompts", () => ({
  resolveServicePromptScope: async () => ({ scopeKey: "current" }),
  subscribeToServicePromptConfigChanges: () => () => {}
}))
vi.mock("@/db/dexie/server-chat-mirror", () => ({
  serverChatMirrorOwnerKey: () => "verified-current",
  linkServerChatMirror: (...args: any[]) => mocks.link(...args)
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { getConfig: async () => ({}) }
}))
vi.mock("@/services/chat-surface-scope", () => ({
  buildChatSurfaceScopeKeyFromConfig: () => mocks.currentScope
}))
vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionState: () => ({
    serverUrl: "http://current",
    lastConfigUpdatedAt: 0
  })
}))
vi.mock("@/hooks/useSelectedAssistant", () => ({
  useSelectedAssistant: () => [null, mocks.setAssistant]
}))
vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: () => ({ setSystemPrompt: mocks.setPrompt })
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (_key: string, fallback: string) => fallback })
}))

import {
  HistorySelectionProvider,
  useHistorySelectionContext,
  type HistorySelectionController
} from "@/hooks/chat/useHistorySelection"
import { HistorySelectionReview } from "@/components/Common/Playground/HistorySelectionReview"
import { usePlaygroundSessionPersistence } from "../usePlaygroundSessionPersistence"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"

let controller: HistorySelectionController
let outcome: string | undefined
const reference = {
  profile_id: "profile",
  client_session_id: "saved-tab",
  owner_key: "local-owner",
  conversation_id: "local-A",
  owner_kind: "local"
}
function ColdSession() {
  const selection = useHistorySelectionContext()!
  React.useLayoutEffect(() => { controller = selection }, [selection])
  const session = usePlaygroundSessionPersistence()
  const started = React.useRef(false)
  const { historyId, serverChatId, messages } = useStoreMessageOption()
  React.useEffect(() => {
    if (!session.sessionScopeReady || started.current) return
    started.current = true
    void session.restoreSession().then((result) => {
      outcome = result
      if (result === "not-restored") mocks.recent()
    })
  }, [session.sessionScopeReady, session.restoreSession])
  return (
    <>
      <output data-testid="transcript">
        {messages.map((row) => row.message).join(" / ")}
      </output>
      <HistorySelectionReview
        selection={selection}
        onBind={() =>
          void selection.loadConversation({
            historyId,
            serverChatId,
            bindUnbound: true
          })
        }
      />
    </>
  )
}
const mount = () =>
  render(
    <HistorySelectionProvider
      storageKey="cold-h1"
      onCapture={(capture) => {
        useStoreMessageOption.getState().setMessages(
          capture.selected_content.map((row) => ({
            id: row.id,
            message: row.message,
            isBot: true,
            name: "Assistant",
            sources: []
          }))
        )
      }}
    >
      <ColdSession />
    </HistorySelectionProvider>
  )
function seedBookmark(ref = reference) {
  const view = {
    owner_key: ref.owner_key,
    conversation_id: ref.conversation_id,
    view_session_id: "old-view",
    selection_revision: 4,
    interpretation: { kind: "parent_graph_v1" },
    cursor: { kind: "after_message", message_id: "chosen" }
  }
  mocks.bookmarks.set(bookmarkKey(ref, view), { ...ref, view })
  sessionStorage.setItem("cold-h1", JSON.stringify(ref))
  return view
}
beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  sessionStorage.clear()
  mocks.bookmarks.clear()
  outcome = undefined
  mocks.profile.mockResolvedValue("profile")
  mocks.details.mockResolvedValue(null)
  mocks.files.mockResolvedValue([])
  mocks.link.mockResolvedValue("mirror")
  mocks.chatData.mockResolvedValue({
    historyInfo: {
      id: "local-A",
      last_used_prompt: { prompt_content: "A prompt" }
    },
    messages: []
  })
  mocks.capture.mockImplementation(async (owner, view) => {
    const ownerKey = owner.kind === "local" ? "local-owner" : "native-current"
    if (view.owner_key && view.owner_key !== ownerKey)
      throw new Error("owner_conversation_mismatch")
    const bound = { ...view, owner_key: ownerKey }
    const snapshot: any = {
      version: 1,
      owner_key: ownerKey,
      conversation_id: owner.conversation_id,
      source_digest: "source",
      storage_context_digest: "storage",
      fences: {},
      interpretation_status: { kind: "parent_graph_v1" },
      nodes: [
        {
          id: "chosen",
          revision: "1",
          role: "assistant",
          parent_id: null,
          settled: true,
          preview: "A selected answer"
        }
      ]
    }
    const result = resolveHistorySelection(snapshot, bound, "send", "")
    if (result.status !== "ready") return { ...result, view: bound, snapshot }
    return {
      status: "captured",
      view: bound,
      snapshot,
      rows: result.rows,
      selected_content: result.rows.map((row) => ({
        id: row.id,
        revision: row.revision,
        message: row.preview,
        images: []
      })),
      storage_context_digest: "storage",
      purpose: "send"
    }
  })
  useStoreMessageOption.setState({
    historyId: null,
    serverChatId: null,
    history: [],
    messages: [],
    queuedMessages: [],
    temporaryChat: false,
    compareSelectedModels: [],
    serverChatAssistantId: null,
    serverChatAssistantKind: null,
    serverChatCharacterId: null,
    serverChatPersonaMemoryMode: null,
    serverChatMetaLoaded: false,
    contextFiles: []
  })
  usePlaygroundSessionStore.getState().clearSession()
})
it.each(["remote scope changed", "shared session expired"])(
  "restores the matching local tab before checking whether %s",
  async (scenario) => {
    seedBookmark()
    usePlaygroundSessionStore.getState().saveSession({
      historyId: "local-A",
      historySelectionReference: reference,
      scopeKey:
        scenario === "remote scope changed" ? "previous-account" : "current",
      queuedMessages: [{ promptText: "unsafe queue" } as any],
      trackedAssistantId: "unsafe-persona",
      trackedAssistantKind: "persona"
    })
    if (scenario === "shared session expired")
      usePlaygroundSessionStore.setState({
        lastUpdated: Date.now() - 25 * 60 * 60 * 1000
      })
    mount()
    await waitFor(() => expect(outcome).toBeDefined())
    expect(screen.getByTestId("transcript")).toHaveTextContent(
      "A selected answer"
    )
    expect(controller.view?.cursor).toEqual({
      kind: "after_message",
      message_id: "chosen"
    })
    expect(mocks.recent).not.toHaveBeenCalled()
    expect(useStoreMessageOption.getState().queuedMessages).toEqual([])
    expect(useStoreMessageOption.getState().serverChatAssistantId).toBeNull()
    expect(mocks.setPrompt).toHaveBeenLastCalledWith("A prompt")
  }
)
it("validates a matching native bookmark even when the shared remote scope changed, with no recent fallback", async () => {
  const native = {
    ...reference,
    owner_key: "native-previous",
    owner_kind: "native",
    conversation_id: "native-A"
  }
  seedBookmark(native)
  usePlaygroundSessionStore.getState().saveSession({
    serverChatId: "native-A",
    historySelectionReference: native,
    scopeKey: "previous-account"
  })
  mount()
  await waitFor(() => expect(outcome).toBeDefined())
  expect(controller.error).toBe("owner_conversation_mismatch")
  expect(mocks.recent).not.toHaveBeenCalled()
  expect(useStoreMessageOption.getState().serverChatId).toBeNull()
})
it("restores a cold unbound mirror read-only and binds it with its retained local locator", async () => {
  mocks.details.mockResolvedValue({
    id: "mirror",
    server_chat_id: "native-A",
    title: "Old mirror"
  })
  mocks.chatData.mockResolvedValue({
    historyInfo: { id: "mirror", server_chat_id: "native-A" },
    messages: [
      {
        id: "legacy-row",
        role: "assistant",
        content: "Readable legacy transcript"
      }
    ]
  })
  usePlaygroundSessionStore.getState().saveSession({
    historyId: "mirror",
    serverChatId: "native-A",
    scopeKey: "current"
  })
  mount()
  await screen.findByRole("button", { name: "Connect this conversation" })
  expect(screen.getByTestId("transcript")).toHaveTextContent(
    "Readable legacy transcript"
  )
  expect(useStoreMessageOption.getState().historyId).toBe("mirror")
  expect(useStoreMessageOption.getState().serverChatId).toBeNull()
  expect(mocks.capture).not.toHaveBeenCalled()
  expect(mocks.profile).not.toHaveBeenCalled()
  fireEvent.click(
    screen.getByRole("button", { name: "Connect this conversation" })
  )
  await waitFor(() => expect(mocks.link).toHaveBeenCalled())
  expect(mocks.link.mock.calls[0][0]).toMatchObject({
    chatId: "native-A",
    legacyHistoryId: "mirror",
    currentHistoryId: "mirror",
    ownerKey: "verified-current"
  })
  expect(mocks.link.mock.calls[0][0]).not.toHaveProperty("messages")
  expect(controller.status).toBe("ready")
})

it("does not turn a verified mirror owner mismatch into the unbound read-only fallback", async () => {
  mocks.details.mockResolvedValue({
    id: "mirror",
    server_chat_id: "native-A",
    server_scope_key: "other-owner",
    title: "Bound elsewhere"
  })
  mocks.chatData.mockResolvedValue({
    historyInfo: { id: "mirror" },
    messages: [{ id: "private", content: "Other owner transcript" }]
  })
  usePlaygroundSessionStore.getState().saveSession({
    historyId: "mirror",
    serverChatId: "native-A",
    scopeKey: "current"
  })
  mount()
  await waitFor(() => expect(outcome).toBe("restored"))
  expect(controller.error).toBe("owner_conversation_mismatch")
  expect(screen.getByTestId("transcript")).toBeEmptyDOMElement()
  expect(useStoreMessageOption.getState().historyId).toBeNull()
  expect(
    screen.queryByRole("button", { name: "Connect this conversation" })
  ).toBeNull()
  expect(mocks.link).not.toHaveBeenCalled()
  expect(mocks.recent).not.toHaveBeenCalled()
})

it.each([true, false])("does not replay stale shared-session comparison over a selected local owner state %s", async ownerMode => {
  seedBookmark()
  usePlaygroundSessionStore.getState().saveSession({ historyId: "local-A", historySelectionReference: reference, scopeKey: "current", compareMode: !ownerMode, compareSelectedModels: ["stale"] })
  useStoreMessageOption.setState({ compareMode: ownerMode, compareSelectedModels: ownerMode ? ["A", "B"] : [] })
  mount()
  await waitFor(() => expect(outcome).toBe("restored"))
  expect(useStoreMessageOption.getState().compareMode).toBe(ownerMode)
  expect(useStoreMessageOption.getState().compareSelectedModels).toEqual(ownerMode ? ["A", "B"] : [])
})

it("cancels held selected-local session restore after live owner navigation", async () => {
  seedBookmark()
  usePlaygroundSessionStore.getState().saveSession({ historyId: "local-A", historySelectionReference: reference, scopeKey: "current", compareMode: false, compareSelectedModels: ["stale"] })
  let release!: (value: any) => void
  mocks.chatData.mockImplementationOnce(() => new Promise(resolve => { release = resolve }))
  mount()
  await waitFor(() => expect(mocks.chatData).toHaveBeenCalled())
  await act(async () => {
    await controller.loadConversation({ historyId: "local-B" })
    useStoreMessageOption.setState({ historyId: "local-B", compareMode: true, compareSelectedModels: ["B model"] })
    release({ historyInfo: { id: "local-A", last_used_prompt: { prompt_content: "stale prompt" } }, messages: [] })
  })
  await waitFor(() => expect(outcome).toBe("cancelled"))
  expect(controller.getCurrent().owner).toMatchObject({ kind: "local", conversation_id: "local-B" })
  expect(useStoreMessageOption.getState()).toMatchObject({ historyId: "local-B", compareMode: true, compareSelectedModels: ["B model"] })
  expect(mocks.setPrompt).not.toHaveBeenCalledWith("stale prompt")
})
