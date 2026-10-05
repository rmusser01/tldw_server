// @vitest-environment jsdom
import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
import { resolveHistorySelection } from "@/utils/history-selection"
import { sha256 } from "@noble/hashes/sha2.js"
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js"

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
  currentScope: "current",
  assistantLoading: false,
  configChanged: null as (() => void) | null,
  ownerScope: vi.fn(),
  draftRecords: new Map<string, unknown>()
}))
vi.mock("@/utils/safe-storage", async importOriginal => ({
  ...await importOriginal<typeof import("@/utils/safe-storage")>(),
  createSafeStorage: () => ({
    get: async (key: string) => mocks.draftRecords.get(key),
    set: async (key: string, value: unknown) => { mocks.draftRecords.set(key, value) },
    remove: async (key: string) => { mocks.draftRecords.delete(key) },
    getAll: async () => Object.fromEntries(mocks.draftRecords)
  })
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
  loadHistoryTurnRecoveries: async () => [],
  dismissHistoryTurnRecovery: async () => {},
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
vi.mock("@/services/service-prompts", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/services/service-prompts")>(),
  loadServicePromptSnapshot: async () => ({ requestScope: verifiedScope, scopeSignal: new AbortController().signal, scopeInvalidatedSignal: new AbortController().signal, release: vi.fn() }),
  resolveServicePromptScope: () => mocks.ownerScope(),
  subscribeToServicePromptConfigChanges: (listener: () => void) => { mocks.configChanged = listener; return () => {} }
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
  useSelectedAssistant: () => [null, mocks.setAssistant, { isLoading: mocks.assistantLoading }]
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
import { captureChatDraftLoad } from "@/hooks/chat/chat-draft-transition"
import { useComposerText } from "@/components/Chat/composer/hooks/useComposerText"
import { conversationDraftKey } from "@/utils/chat-draft-address"
import { buildQueuedRequest } from "@/utils/chat-request-queue"

let controller: HistorySelectionController
let outcome: string | undefined
let restoreAgain: () => Promise<string>
let sessionReady: boolean
const verifiedScope = { config: { serverUrl: "http://current", authMode: "multi-user" }, userId: "alice", scopeKey: "current", clientPrincipalVerified: true }
const ownerHash = bytesToHex(sha256(utf8ToBytes(JSON.stringify(["http://current", "multi-user", "manual", null, "alice", null]))))
const reference = {
  profile_id: "profile",
  client_session_id: "saved-tab",
  owner_key: "local-owner",
  conversation_id: "local-A",
  owner_kind: "local"
}
function ColdDraft() {
  const address = useStoreMessageOption(state => state.chatDraftAddress)
  const textareaRef = React.useRef<HTMLTextAreaElement>(null)
  React.useLayoutEffect(() => {
    if (!useStoreMessageOption.getState().chatDraftAddress) useStoreMessageOption.getState().startNewChatDraft()
  }, [])
  const composer = useComposerText({
    draftKey: address ? conversationDraftKey("cold-composer", address) : "cold-composer",
    draftEnabled: address !== null,
    tabScopedDraft: true,
    textareaRef
  })
  return <textarea ref={textareaRef} aria-label="Restored draft" value={composer.form.values.message} readOnly />
}
function ColdSession({ automaticRestore = true, withDraft = false }: { automaticRestore?: boolean; withDraft?: boolean }) {
  const selection = useHistorySelectionContext()!
  React.useLayoutEffect(() => { controller = selection }, [selection])
  const session = usePlaygroundSessionPersistence()
  React.useLayoutEffect(() => {
    restoreAgain = session.restoreSession
    sessionReady = session.sessionScopeReady
  }, [session.restoreSession, session.sessionScopeReady])
  const started = React.useRef(false)
  const { historyId, serverChatId, messages } = useStoreMessageOption()
  React.useEffect(() => {
    if (!automaticRestore || !session.sessionScopeReady || started.current) return
    started.current = true
    void session.restoreSession().then((result) => {
      outcome = result
      if (result === "not-restored") mocks.recent()
    })
  }, [automaticRestore, session.sessionScopeReady, session.restoreSession])
  return (
    <>
      {withDraft && <ColdDraft />}
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
const mount = (automaticRestore = true, withDraft = false) =>
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
      <ColdSession automaticRestore={automaticRestore} withDraft={withDraft} />
    </HistorySelectionProvider>
  )

it.each(["unowned-local", "profile-fork", "other-native"])(
  "does not restore private cache settings from a mixed %s/native locator",
  async (kind) => {
    const details = { id: "cache", ...(kind === "profile-fork"
      ? { local_owner_key: "local-owner" }
      : kind === "other-native" ? { server_chat_id: "native-other" } : {}) }
    mocks.details.mockResolvedValue(details)
    mocks.chatData.mockResolvedValue({
      historyInfo: { ...details, last_used_prompt: { prompt_content: "private cache prompt" } },
      messages: [{ id: "private", role: "assistant", content: "Private cache row" }]
    })
    usePlaygroundSessionStore.getState().saveSession({ historyId: "cache", serverChatId: "native-A", scopeKey: "current" })
    mount()
    await waitFor(() => expect(outcome).toBe("restored"))
    expect(controller.error).toBe("owner_conversation_mismatch")
    expect(mocks.capture).not.toHaveBeenCalled()
    expect(mocks.setPrompt).not.toHaveBeenCalledWith("private cache prompt")
    expect(screen.getByTestId("transcript")).toBeEmptyDOMElement()
  }
)

it.each(["account-local", "profile-fork", "native"])(
  "does not publish a held %s tab restore capture after restore cancellation",
  async (kind) => {
    const ref = kind === "native"
      ? { ...reference, owner_key: "native-current", owner_kind: "native", conversation_id: "native-A" }
      : reference
    seedBookmark(ref)
    if (kind === "account-local") mocks.details.mockResolvedValue({ id: "local-A", server_scope_key: "verified-current" })
    usePlaygroundSessionStore.getState().saveSession({
      ...(kind === "native" ? { serverChatId: "native-A" } : { historyId: "local-A" }),
      historySelectionReference: ref,
      scopeKey: "current"
    })
    const capture = mocks.capture.getMockImplementation()!
    let finish!: () => void
    const gate = new Promise<void>((resolve) => { finish = resolve })
    mocks.capture.mockImplementationOnce(async (...args) => { await gate; return capture(...args) })
    mount()
    await waitFor(() => expect(mocks.capture).toHaveBeenCalledTimes(1))
    await act(async () => { usePlaygroundSessionStore.getState().clearSession(); finish() })
    await waitFor(() => expect(outcome).toBe("cancelled"))
    expect(controller.getCurrent().capture).toBeNull()
    expect(screen.getByTestId("transcript")).toBeEmptyDOMElement()
    expect(mocks.setPrompt).not.toHaveBeenCalledWith("A prompt")
  }
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
  mocks.assistantLoading = false
  mocks.configChanged = null
  mocks.ownerScope.mockReset().mockResolvedValue(verifiedScope)
  localStorage.clear()
  sessionStorage.clear()
  mocks.bookmarks.clear()
  mocks.draftRecords.clear()
  outcome = undefined
  mocks.profile.mockResolvedValue("profile")
  mocks.details.mockImplementation(async (id: string) => ({ id, local_owner_key: "local-owner" }))
  mocks.files.mockResolvedValue([])
  mocks.link.mockResolvedValue("mirror")
  mocks.chatData.mockResolvedValue({
    historyInfo: {
      id: "local-A",
      local_owner_key: "local-owner",
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
    chatDraftAddress: null,
    chatDraftRevision: 0,
    chatDraftTransition: null,
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
  usePlaygroundSessionStore.setState({ chatDraftHandle: null })
})

it.each([false, true])("restores a verified fresh draft before an old H1 bookmark (temporary %s)", async temporary => {
  seedBookmark()
  const address = { kind: "fresh" as const, id: "retained-new", scope: { type: "global" as const }, temporary }
  usePlaygroundSessionStore.getState().saveSession({
    chatDraftHandle: { address, ownerHash }, scopeKey: "current", historyId: null, serverChatId: null
  })
  mount()
  await waitFor(() => expect(outcome).toBe("restored"))
  expect(useStoreMessageOption.getState()).toMatchObject({ chatDraftAddress: address, temporaryChat: temporary, historyId: null, serverChatId: null, messages: [] })
  expect(controller.getCurrent().owner).toBeNull()
  expect(mocks.capture).not.toHaveBeenCalled()
  expect(mocks.recent).not.toHaveBeenCalled()
})

it.each([false, true])("cold restores the retained unsent text through the actual composer and fresh handle (temporary %s)", async temporary => {
  seedBookmark()
  const address = { kind: "fresh" as const, id: "unsent-next", scope: { type: "global" as const }, temporary }
  mocks.draftRecords.set(`registry:draft:${conversationDraftKey("cold-composer", address)}:owner:${ownerHash}`, {
    value: { content: "UNSENT NEXT TURN" }, updatedAt: Date.now()
  })
  usePlaygroundSessionStore.getState().saveSession({ chatDraftHandle: { address, ownerHash }, scopeKey: "current" })
  mount(true, true)
  await waitFor(() => expect(screen.getByRole("textbox", { name: "Restored draft" })).toHaveValue("UNSENT NEXT TURN"))
  expect(outcome).toBe("restored")
  expect(useStoreMessageOption.getState()).toMatchObject({ chatDraftAddress: address, temporaryChat: temporary, historyId: null, serverChatId: null, messages: [], history: [] })
  expect(controller.getCurrent().owner).toBeNull()
  expect(mocks.capture).not.toHaveBeenCalled()
  expect(mocks.recent).not.toHaveBeenCalled()
})

it("defers a fresh handle while owner verification is pending, then rejects another account", async () => {
  let finish!: (value: typeof verifiedScope) => void
  mocks.ownerScope.mockImplementation(() => new Promise(resolve => { finish = resolve }))
  usePlaygroundSessionStore.getState().saveSession({
    chatDraftHandle: { address: { kind: "fresh", id: "private", scope: { type: "global" }, temporary: false }, ownerHash }, scopeKey: "current"
  })
  mount()
  await waitFor(() => expect(mocks.ownerScope).toHaveBeenCalled())
  expect(outcome).toBeUndefined()
  await act(async () => finish({ ...verifiedScope, userId: "bob" }))
  await waitFor(() => expect(outcome).toBe("cancelled"))
  expect(useStoreMessageOption.getState().chatDraftAddress).toBeNull()
  expect(mocks.recent).not.toHaveBeenCalled()
})

it.each([false, true])("immediately retains the new draft handle without saving Temporary queued content (%s)", async temporary => {
  const view = mount()
  await waitFor(() => expect(outcome).toBe("not-restored"))
  act(() => {
    useStoreMessageOption.getState().setTemporaryChat(temporary)
    useStoreMessageOption.getState().startNewChatDraft()
    if (temporary) useStoreMessageOption.getState().setQueuedMessages([buildQueuedRequest({ id: "private-queue", clientRequestId: "private-request", promptText: "private temporary queue" })])
  })
  const address = useStoreMessageOption.getState().chatDraftAddress
  await waitFor(() => expect(usePlaygroundSessionStore.getState()).toMatchObject({ chatDraftHandle: { address, ownerHash }, historySelectionReference: null, historyId: null, serverChatId: null, queuedMessages: [] }))
  view.unmount()
})

it("does not persist a target ID while the draft still addresses its source", async () => {
  const view = mount()
  await waitFor(() => expect(outcome).toBe("not-restored"))
  act(() => useStoreMessageOption.getState().startNewChatDraft())
  const address = useStoreMessageOption.getState().chatDraftAddress
  await waitFor(() => expect(usePlaygroundSessionStore.getState()).toMatchObject({ chatDraftHandle: { address, ownerHash } }))
  act(() => useStoreMessageOption.getState().setServerChatId("held-target"))
  view.unmount()
  await act(async () => { await Promise.resolve(); await Promise.resolve() })
  expect(usePlaygroundSessionStore.getState()).toMatchObject({ chatDraftHandle: { address, ownerHash }, serverChatId: null })
})

it("does not publish the automatic fresh allocation before a retained handle is verified", async () => {
  let finish!: (value: typeof verifiedScope) => void
  mocks.ownerScope.mockImplementation(() => new Promise(resolve => { finish = resolve }))
  const address = { kind: "fresh" as const, id: "retained", scope: { type: "global" as const }, temporary: true }
  usePlaygroundSessionStore.getState().saveSession({ chatDraftHandle: { address, ownerHash }, scopeKey: "current" })
  useStoreMessageOption.getState().startNewChatDraft()
  mount()
  await waitFor(() => expect(mocks.ownerScope).toHaveBeenCalled())
  expect(usePlaygroundSessionStore.getState().chatDraftHandle).toEqual({ address, ownerHash })
  await act(async () => finish(verifiedScope))
  await waitFor(() => expect(outcome).toBe("restored"))
  expect(useStoreMessageOption.getState().chatDraftAddress).toEqual(address)
})

it.each(["local", "native-global", "native-workspace"])("installs a saved %s draft address only after its matching H1 receipt", async kind => {
  const address = kind === "local"
    ? { kind: "local" as const, id: "local-A", profileId: "profile" }
    : { kind: "native" as const, id: "native-A", ownerKey: "native-current", scope: kind === "native-workspace" ? { type: "workspace" as const, workspaceId: "workspace-A" } : { type: "global" as const } }
  usePlaygroundSessionStore.getState().saveSession({ chatDraftHandle: { address, ownerHash }, scopeKey: "current", historyId: kind === "local" ? address.id : null, serverChatId: kind === "local" ? null : address.id })
  mount()
  await waitFor(() => expect(outcome).toBe("restored"))
  expect(useStoreMessageOption.getState().chatDraftAddress).toEqual(address)
  expect(controller.getCurrent().owner?.conversation_id).toBe(address.id)
})

it.each(["local-profile", "native-owner"])("rejects a mismatched saved %s address after H1 qualification", async kind => {
  const address = kind === "local-profile"
    ? { kind: "local" as const, id: "local-A", profileId: "other-profile" }
    : { kind: "native" as const, id: "native-A", ownerKey: "other-owner", scope: { type: "global" as const } }
  usePlaygroundSessionStore.getState().saveSession({ chatDraftHandle: { address, ownerHash }, scopeKey: "current", historyId: kind === "local-profile" ? address.id : null, serverChatId: kind === "local-profile" ? null : address.id })
  mount()
  await waitFor(() => expect(outcome).toBe("cancelled"))
  expect(useStoreMessageOption.getState().chatDraftAddress).toBeNull()
  expect(mocks.recent).not.toHaveBeenCalled()
})

it("honours a deliberate external selection after it cancels the initial handle restore", async () => {
  usePlaygroundSessionStore.getState().saveSession({ chatDraftHandle: { address: { kind: "fresh", id: "old", scope: { type: "global" }, temporary: false }, ownerHash }, scopeKey: "current" })
  useStoreMessageOption.getState().startNewChatDraft()
  const view = mount(false)
  await waitFor(() => expect(sessionReady).toBe(true))
  act(() => {
    usePlaygroundSessionStore.getState().cancelPendingRestore()
    useStoreMessageOption.getState().setServerChatId("explicit-B")
  })
  // A target-ID/source-address pair cannot overwrite the retained pointer while B loads.
  expect(usePlaygroundSessionStore.getState().chatDraftHandle?.address.id).toBe("old")
  const accept = captureChatDraftLoad("switch")
  await act(async () => { expect(await controller.loadConversation({ serverChatId: "explicit-B", isCurrent: accept.isCurrent }, null, accept)).toBe(true) })
  await waitFor(() => expect(usePlaygroundSessionStore.getState()).toMatchObject({ chatDraftHandle: { address: { kind: "native", id: "explicit-B" }, ownerHash }, serverChatId: "explicit-B" }))
  view.unmount()
})

it("lets a current sidebar selection made before mount supersede the retained fresh handle", async () => {
  useStoreMessageOption.getState().startNewChatDraft()
  const source = useStoreMessageOption.getState().chatDraftAddress!
  usePlaygroundSessionStore.getState().saveSession({ chatDraftHandle: { address: source, ownerHash }, scopeKey: "current" })
  usePlaygroundSessionStore.getState().cancelPendingRestore()
  const accept = captureChatDraftLoad("switch")
  usePlaygroundSessionStore.getState().requestServerChatSelection("sidebar-B", accept)
  useStoreMessageOption.getState().setServerChatId("sidebar-B")
  const view = mount()
  await waitFor(() => expect(outcome).toBe("cancelled"))
  expect(useStoreMessageOption.getState()).toMatchObject({ serverChatId: "sidebar-B", chatDraftAddress: source })
  await act(async () => { expect(await controller.loadConversation({ serverChatId: "sidebar-B", isCurrent: accept.isCurrent }, null, accept)).toBe(true) })
  await waitFor(() => expect(usePlaygroundSessionStore.getState()).toMatchObject({ serverChatId: "sidebar-B", chatDraftHandle: { address: { kind: "native", id: "sidebar-B" }, ownerHash } }))
  view.unmount()
})

it.each(["legacy-debounce", "legacy-departure", "bookmark-debounce", "bookmark-departure"])("does not replace a pending %s restore with the composer's automatic fresh address", async path => {
  seedBookmark()
  if (path.startsWith("legacy")) usePlaygroundSessionStore.getState().saveSession({ historyId: "local-A", scopeKey: "current" })
  useStoreMessageOption.getState().startNewChatDraft()
  const capture = mocks.capture.getMockImplementation()!
  let finish!: () => void
  mocks.capture.mockImplementationOnce((...args) => new Promise(resolve => { finish = () => resolve(capture(...args)) }))
  const view = mount()
  await waitFor(() => expect(mocks.capture).toHaveBeenCalled())
  if (path.endsWith("departure")) view.unmount()
  else await act(async () => { await new Promise(resolve => setTimeout(resolve, 1100)) })
  expect(usePlaygroundSessionStore.getState().chatDraftHandle).toBeNull()
  if (path.startsWith("legacy")) expect(usePlaygroundSessionStore.getState().historyId).toBe("local-A")
  await act(async () => finish())
  if (!path.endsWith("departure")) {
    await waitFor(() => expect(outcome).toBe("restored"))
    expect(useStoreMessageOption.getState().chatDraftAddress).toMatchObject({ kind: "local", id: "local-A" })
    view.unmount()
  }
})
it.each(["mounted", "departure"])("retains accepted New before a cancelled held restore settles (%s)", async path => {
  seedBookmark()
  usePlaygroundSessionStore.getState().saveSession({ historyId: "local-A", scopeKey: "current" })
  useStoreMessageOption.getState().startNewChatDraft()
  const capture = mocks.capture.getMockImplementation()!
  let finish!: () => void
  mocks.capture.mockImplementationOnce((...args) => new Promise(resolve => { finish = () => resolve(capture(...args)) }))
  const view = mount()
  await waitFor(() => expect(mocks.capture).toHaveBeenCalled())
  act(() => {
    controller.reset()
    usePlaygroundSessionStore.getState().clearSession()
    useStoreMessageOption.getState().setHistoryId(null)
    useStoreMessageOption.getState().setServerChatId(null)
    useStoreMessageOption.getState().startNewChatDraft()
  })
  const address = useStoreMessageOption.getState().chatDraftAddress!
  await waitFor(() => expect(usePlaygroundSessionStore.getState().chatDraftHandle).toEqual({ address, ownerHash }))
  if (path === "departure") view.unmount()
  await act(async () => finish())
  await waitFor(() => expect(outcome).toBe("cancelled"))
  expect(usePlaygroundSessionStore.getState().chatDraftHandle).toEqual({ address, ownerHash })
  expect(useStoreMessageOption.getState()).toMatchObject({ chatDraftAddress: address, messages: [], historyId: null, serverChatId: null })
  if (path !== "departure") view.unmount()
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
it("retains a cold unbound mirror locator without publishing unverified rows and binds only explicitly", async () => {
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
  expect(screen.getByTestId("transcript")).toBeEmptyDOMElement()
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


it.each(["success", "error"])("automatic restore after readiness cycling retains the invalidated lease and fences held %s", async (late) => {
  const native = { ...reference, owner_key: "native-current", owner_kind: "native", conversation_id: "native-A" }
  seedBookmark(native)
  usePlaygroundSessionStore.getState().saveSession({ serverChatId: "native-A", historySelectionReference: native, scopeKey: "current" })
  const captureNormally = mocks.capture.getMockImplementation()!
  let finish!: () => void
  mocks.capture.mockImplementationOnce((owner, view) => new Promise((resolve, reject) => {
    finish = () => late === "success" ? resolve(captureNormally(owner, view)) : reject(new Error("late old-owner failure"))
  }))
  const mounted = mount()
  await waitFor(() => expect(mocks.capture).toHaveBeenCalledTimes(1))
  act(() => { mocks.currentScope = "other-account"; mocks.configChanged!() })
  expect(controller.error).toBe("request_config_scope_changed")
  mocks.assistantLoading = true
  mounted.rerender(<HistorySelectionProvider storageKey="cold-h1"><ColdSession /></HistorySelectionProvider>)
  await waitFor(() => expect(sessionReady).toBe(false))
  mocks.assistantLoading = false
  mounted.rerender(<HistorySelectionProvider storageKey="cold-h1"><ColdSession /></HistorySelectionProvider>)
  await waitFor(() => expect(sessionReady).toBe(true))
  let restored: string | undefined
  await act(async () => { restored = await restoreAgain() })
  expect(restored).toBe("cancelled")
  expect(mocks.capture).toHaveBeenCalledTimes(1)
  await act(async () => { finish(); await new Promise(resolve => setTimeout(resolve, 0)) })
  expect(controller.error).toBe("request_config_scope_changed")
  expect(controller.getCurrent().capture).toBeNull()
  await act(async () => { await controller.loadConversation({ serverChatId: "native-A" }) })
  expect(controller.getCurrent().capture).not.toBeNull()
  expect(controller.error).toBeNull()
})

it("requires an explicit reopen of an invalidated stamped-local account lease after readiness cycling", async () => {
  seedBookmark()
  mocks.details.mockResolvedValue({ id: "local-A", server_scope_key: "verified-current" })
  usePlaygroundSessionStore.getState().saveSession({ historyId: "local-A", historySelectionReference: reference, scopeKey: "current" })
  const mounted = mount()
  await waitFor(() => expect(outcome).toBe("restored"))
  expect(controller.capture).not.toBeNull()
  act(() => { mocks.configChanged!() })
  expect(controller.error).toBe("request_config_scope_changed")
  expect(controller.getCurrent().capture).toBeNull()
  const captures = mocks.capture.mock.calls.length
  mocks.assistantLoading = true
  mounted.rerender(<HistorySelectionProvider storageKey="cold-h1"><ColdSession /></HistorySelectionProvider>)
  await waitFor(() => expect(sessionReady).toBe(false))
  mocks.assistantLoading = false
  mounted.rerender(<HistorySelectionProvider storageKey="cold-h1"><ColdSession /></HistorySelectionProvider>)
  await waitFor(() => expect(sessionReady).toBe(true))
  await act(async () => { expect(await restoreAgain()).toBe("cancelled") })
  expect(mocks.capture).toHaveBeenCalledTimes(captures)
  await act(async () => { await controller.loadConversation({ historyId: "local-A" }) })
  expect(controller.capture).not.toBeNull()
  expect(controller.error).toBeNull()
})
