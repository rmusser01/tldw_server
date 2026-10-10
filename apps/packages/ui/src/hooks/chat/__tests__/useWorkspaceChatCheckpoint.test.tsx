import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { BrowserRouter, MemoryRouter, useLocation, useNavigate } from "react-router-dom"
import { Modal } from "antd"
import { ConnectionPhase } from "@/types/connection"
import { resolveHistorySelection } from "@/utils/history-selection"
import type { HistorySelectionSnapshotV1, HistoryViewSelectionV1 } from "@/types/history-selection"
import { RouterContext as NextRouterContext } from "../../../../../../tldw-frontend/node_modules/next/dist/shared/lib/router-context.shared-runtime"

const mocks = vi.hoisted(() => ({
  bookmarks: new Map<string, unknown>(),
  listeners: new Set<() => void>(),
  scope: vi.fn(),
  capture: vi.fn(),
  profile: vi.fn(),
  saveBookmark: vi.fn(),
  stop: vi.fn(),
  submit: vi.fn()
}))
const nextNavigation = vi.hoisted(() => ({
  enabled: false,
  router: { asPath: "", replace: vi.fn(async () => true), push: vi.fn(async () => true), back: vi.fn() }
}))
vi.mock("react-router-dom", async (original) => {
  const actual = await original<typeof import("react-router-dom")>()
  const next = await import("../../../../../../tldw-frontend/extension/shims/react-router-dom")
  return { ...actual, useNavigate: () => (nextNavigation.enabled ? next.useNavigate : actual.useNavigate)() }
})
const bookmarkKey = (scope: { profile_id: string; client_session_id: string }, owner: { owner_key: string; conversation_id: string }) =>
  JSON.stringify([scope.profile_id, scope.client_session_id, owner.owner_key, owner.conversation_id])
vi.mock("@/db/dexie/history-selection", () => ({
  ensureLocalProfileId: (...args: unknown[]) => mocks.profile(...args),
  loadHistoryBookmark: async (scope: Parameters<typeof bookmarkKey>[0], owner: Parameters<typeof bookmarkKey>[1]) =>
    mocks.bookmarks.get(bookmarkKey(scope, owner)) ?? null,
  saveHistoryBookmark: async (scope: Parameters<typeof bookmarkKey>[0], view: Parameters<typeof bookmarkKey>[1]) =>
    mocks.saveBookmark(scope, view),
  loadHistoryTurnRecoveries: async () => [],
  dismissHistoryTurnRecovery: async () => {}
}))
vi.mock("@/db/dexie/fork-operations", () => ({
  findForkCandidate: async () => null,
  loadForkOperations: async () => [],
  allowNewForkOperation: async () => {}
}))
vi.mock("@/db/dexie/chat", () => ({
  PageAssistDatabase: class { getHistoryInfo = async () => null }
}))
vi.mock("@/services/chat-history-selection", () => ({
  captureHistorySnapshot: (...args: unknown[]) => mocks.capture(...args),
  readNativeForkSettings: async () => null,
  updateNativeForkSettings: async () => null,
  confirmLegacyHistoryProjection: async () => {}
}))
vi.mock("@/services/service-prompts", () => ({
  resolveServicePromptScope: (...args: unknown[]) => mocks.scope(...args),
  subscribeToServicePromptConfigChanges: (listener: () => void) => {
    mocks.listeners.add(listener)
    return () => { mocks.listeners.delete(listener) }
  }
}))
vi.mock("@/hooks/useMessageOption", async () => {
  const { useStoreMessageOption } = await import("@/store/option")
  return { useMessageOption: () => ({ ...useStoreMessageOption(), onSubmit: mocks.submit, stopStreamingRequest: mocks.stop }) }
})
vi.mock("@/hooks/useSmartScroll", () => ({
  useSmartScroll: () => ({ containerRef: { current: null }, isAutoScrollToBottom: true, autoScrollToBottom: vi.fn() })
}))
vi.mock("@/hooks/useMediaQuery", () => ({ useMobile: () => false }))
vi.mock("@/store/connection", () => ({
  useConnectionStore: (select: (value: unknown) => unknown) => select({
    state: { phase: ConnectionPhase.CONNECTED, isChecking: false, lastError: null }, checkOnce: vi.fn()
  })
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { getChatLorebookDiagnostics: vi.fn(async () => ({ turns: [], total_turns_with_diagnostics: 0 })) }
}))
vi.mock("@/services/tldw-server", () => ({ fetchChatModels: async () => [{ id: "test-model", name: "Test Model", provider: "test" }] }))
vi.mock("@/components/Option/Playground/ChatModelSelectorDropdown", () => ({ ChatModelSelectorDropdown: () => null }))
vi.mock("@/components/Common/Playground/Message", () => ({ PlaygroundMessage: () => null }))

import { HistorySelectionProvider, useHistorySelectionContext, type HistorySelectionController } from "../useHistorySelection"
import { useWorkspaceChatCheckpoint, WorkspaceChatRouteSearchContext } from "../useWorkspaceChatCheckpoint"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { useWorkspaceStore, type WorkspaceChatSession } from "@/store/workspace"
import { serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror"
import { ChatPane } from "@/components/Option/ResearchWorkspace/ChatPane"
import { WorkspaceChatPanel } from "@/components/Option/ChatWorkspace/WorkspaceChatPanel"
import { clearWorkspaceUndoActionsForTests, undoLatestWorkspaceAction } from "@/components/Option/ResearchWorkspace/undo-manager"

const scope = (userId = 1, serverUrl = "http://owner.test") => ({
  config: { serverUrl, authMode: "multi-user" as const, authSource: "manual" as const },
  userId, scopeKey: `account-${userId}`, clientPrincipalVerified: true
})
const reference = (name = "A") => ({
  profile_id: "profile", client_session_id: `saved-${name}`,
  owner_key: `native-${name}`, conversation_id: `chat-${name}`
})
const saved = (name = "A", draft = `Draft ${name}`, owner = scope()): WorkspaceChatSession => ({
  messages: [{ id: `cached-${name}`, name: "Assistant", isBot: true, message: "UNTRUSTED CACHE", sources: [] }],
  history: [{ role: "assistant", content: "UNTRUSTED CACHE" }],
  historyId: null, serverChatId: `chat-${name}`,
  checkpoint: {
    version: 1, ownerKey: serverChatMirrorOwnerKey({ requestScope: owner }),
    workspaceId: `workspace-${name}`, referenceId: `reference-${name}`,
    historySelectionReference: reference(name), draft
  }
})
function seed(name = "A", draft = `Draft ${name}`, owner = scope()) {
  const session = saved(name, draft, owner)
  useWorkspaceStore.getState().saveWorkspaceChatSession(`workspace-${name}::reference-${name}`, session)
  const ref = reference(name)
  mocks.bookmarks.set(bookmarkKey(ref, ref), {
    ...ref,
    view: {
      ...ref, view_session_id: `old-${name}`, selection_revision: 1,
      interpretation: { kind: "parent_graph_v1" }, cursor: { kind: "after_message", message_id: `verified-${name}` }
    }
  })
}
function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((done, fail) => { resolve = done; reject = fail })
  return { promise, resolve, reject }
}
const capture = (owner: { conversation_id: string }, view: HistoryViewSelectionV1) => {
  const name = owner.conversation_id.replace("chat-", "")
  const snapshot: HistorySelectionSnapshotV1 = {
    version: 1, owner_key: `native-${name}`, conversation_id: owner.conversation_id,
    source_digest: "source", storage_context_digest: "storage",
    fences: { conversation: "1", history: "1", settings: "1" },
    interpretation_status: { kind: "parent_graph_v1" },
    nodes: [{ id: `verified-${name}`, revision: "1", parent_id: null, role: "assistant", settled: true, preview: `Verified ${name}` }]
  }
  if (view.owner_key && view.owner_key !== snapshot.owner_key) throw new Error("owner_conversation_mismatch")
  const bound = { ...view, owner_key: snapshot.owner_key }
  const selection = resolveHistorySelection(snapshot, bound, "send", "")
  if (selection.status !== "ready") return { ...selection, snapshot, view: bound }
  return {
    status: "captured", snapshot, view: bound, rows: selection.rows,
    selected_content: selection.rows.map((row) => ({ id: row.id, revision: row.revision, message: row.preview, images: [] })),
    purpose: "send", storage_context_digest: "storage"
  }
}
let controller: HistorySelectionController | null
let checkpointFence: (() => () => boolean) | undefined
let checkpointClear: ReturnType<typeof useWorkspaceChatCheckpoint>["clearChat"] | undefined
function SelectionObserver() {
  const selection = useHistorySelectionContext()
  React.useLayoutEffect(() => { controller = selection }, [selection])
  return null
}
function ResearchRouteSurface() {
  const location = useLocation()
  return <WorkspaceChatRouteSearchContext.Provider value={location.search}>
    <SelectionObserver /><ChatPane />
  </WorkspaceChatRouteSearchContext.Provider>
}
const mountResearchBrowserRouted = () => {
  const content = <BrowserRouter>
  <HistorySelectionProvider onCapture={result => {
    const state = useStoreMessageOption.getState()
    state.setMessages(result.selected_content.map(row => ({ id: row.id, name: "Assistant", isBot: true, message: row.message, sources: [] })))
    state.setHistory(result.selected_content.map(row => ({ role: "assistant", content: row.message })))
  }}><ResearchRouteSurface /></HistorySelectionProvider>
</BrowserRouter>
  return render(nextNavigation.enabled ? <NextRouterContext.Provider value={nextNavigation.router as React.ComponentProps<typeof NextRouterContext.Provider>["value"]}>
    {content}
  </NextRouterContext.Provider> : content)
}
function Surface({ legacySessionKey, routeSearch, replaceRouteSearch, watchSessions = false }: { legacySessionKey?: string; routeSearch?: string; replaceRouteSearch?: (search: string) => void; watchSessions?: boolean }) {
  useWorkspaceStore(state => watchSessions ? state.workspaceChatSessions : null)
  const selection = useHistorySelectionContext()
  React.useLayoutEffect(() => { controller = selection }, [selection])
  const workspaceId = useWorkspaceStore(state => state.workspaceId)
  const hydrated = useWorkspaceStore(state => state.storeHydrated)
  const chat = useStoreMessageOption()
  const [draft, setDraft] = React.useState("")
  const checkpoint = useWorkspaceChatCheckpoint({
    workspaceId, workspaceReady: hydrated, draft, setDraft,
    chat: { ...chat, stopStreamingRequest: mocks.stop }, legacySessionKey, routeSearch, replaceRouteSearch
  })
  React.useLayoutEffect(() => { checkpointFence = checkpoint.fence }, [checkpoint.fence])
  React.useLayoutEffect(() => { checkpointClear = checkpoint.clearChat }, [checkpoint.clearChat])
  return <>
    <input aria-label="Draft" value={draft} onChange={event => checkpoint.setDraft(event.target.value)} />
    <output aria-label="Transcript">{chat.messages.map(row => row.message).join(" / ")}</output>
    <output aria-label="Restoring">{String(checkpoint.restoring)}</output>
    <output aria-label="Restoration error">{checkpoint.restoreError}</output>
  </>
}
function RoutedSurface({ viaContext = false }: { viaContext?: boolean }) {
  const location = useLocation()
  const navigate = useNavigate()
  const content = <>
    <Surface routeSearch={viaContext ? undefined : location.search} />
    {["A", "B", "C"].map(name => <button key={name} onClick={() => navigate(`${location.pathname}?chatId=chat-${name}`)}>Route {name}</button>)}
  </>
  return viaContext ? <WorkspaceChatRouteSearchContext.Provider value={location.search}>{content}</WorkspaceChatRouteSearchContext.Provider> : content
}
const mount = (provider = true, legacySessionKey?: string, research = false, watchSessions = false) => render(provider ?
  <HistorySelectionProvider onCapture={result => {
    const state = useStoreMessageOption.getState()
    state.setMessages(result.selected_content.map(row => ({ id: row.id, name: "Assistant", isBot: true, message: row.message, sources: [] })))
    state.setHistory(result.selected_content.map(row => ({ role: "assistant", content: row.message })))
  }}>{research ? <MemoryRouter><SelectionObserver /><ChatPane /></MemoryRouter> :
    <Surface legacySessionKey={legacySessionKey} watchSessions={watchSessions} />}</HistorySelectionProvider> : <Surface legacySessionKey={legacySessionKey} watchSessions={watchSessions} />)
const changeWorkspace = (name: string, referenceId = `reference-${name}`) => act(() => {
  useWorkspaceStore.setState({ workspaceId: `workspace-${name}`, workspaceChatReferenceId: referenceId })
})
const draft = () => screen.getByRole("textbox", { name: "Draft" }) as HTMLInputElement
const transcript = () => screen.getByLabelText("Transcript").textContent
const checkpoint = (name = "A", ref = `reference-${name}`) =>
  useWorkspaceStore.getState().workspaceChatSessions[`workspace-${name}::${ref}`]
const mountRouted = (search = "?chatId=chat-B", viaContext = false, path = "/chat-workspace") => render(
  <MemoryRouter initialEntries={[`${path}${search}`]}>
    <HistorySelectionProvider onCapture={result => {
      const state = useStoreMessageOption.getState()
      state.setMessages(result.selected_content.map(row => ({ id: row.id, name: "Assistant", isBot: true, message: row.message, sources: [] })))
      state.setHistory(result.selected_content.map(row => ({ role: "assistant", content: row.message })))
    }}><RoutedSurface viaContext={viaContext} /></HistorySelectionProvider>
  </MemoryRouter>)
const mountBrowserRouted = () => render(
  <BrowserRouter>
    <HistorySelectionProvider onCapture={result => {
      const state = useStoreMessageOption.getState()
      state.setMessages(result.selected_content.map(row => ({ id: row.id, name: "Assistant", isBot: true, message: row.message, sources: [] })))
      state.setHistory(result.selected_content.map(row => ({ role: "assistant", content: row.message })))
    }}><RoutedSurface viaContext /></HistorySelectionProvider>
  </BrowserRouter>)
const traverseHistory = async (direction: "back" | "forward") => {
  await act(async () => {
    await new Promise<void>(resolve => {
      window.addEventListener("popstate", () => resolve(), { once: true })
      window.history[direction]()
    })
  })
}
const handoffSearch = (name = "B") => `?historySelection=${encodeURIComponent(JSON.stringify({ ...reference(name), owner_kind: "native" }))}`

beforeEach(() => {
  nextNavigation.enabled = false
  nextNavigation.router.replace.mockClear()
  nextNavigation.router.push.mockClear()
  clearWorkspaceUndoActionsForTests()
  localStorage.setItem("tldw:feature-rollout:workspace_indexeddb_offload_v1:enabled", "0")
  mocks.bookmarks.clear()
  mocks.listeners.clear()
  mocks.scope.mockReset().mockResolvedValue(scope())
  mocks.capture.mockReset().mockImplementation(capture)
  mocks.profile.mockReset().mockResolvedValue("profile")
  mocks.saveBookmark.mockReset().mockImplementation((scope, view) => mocks.bookmarks.set(bookmarkKey(scope, view), { ...scope, view }))
  mocks.stop.mockReset()
  mocks.submit.mockReset().mockResolvedValue({ status: "submitted" })
  window.history.replaceState({}, "", "/chat-workspace")
  useWorkspaceStore.setState({ workspaceId: "workspace-A", workspaceChatReferenceId: "reference-A", storeHydrated: true, workspaceChatSessions: {} })
  useStoreMessageOption.setState({ messages: [], history: [], historyId: null, serverChatId: null, temporaryChat: false, streaming: false, isProcessing: false, isLoading: false })
  usePlaygroundSessionStore.getState().clearSession()
  controller = null
  checkpointFence = undefined
  checkpointClear = undefined
})
afterEach(() => { clearWorkspaceUndoActionsForTests(); localStorage.clear(); vi.restoreAllMocks() })

describe("qualified workspace checkpoint handoff", () => {
  const clearResearchChat = () => {
    vi.spyOn(Modal, "confirm").mockImplementation(config => {
      config.onOk?.()
      return { destroy: vi.fn(), update: vi.fn() } as ReturnType<typeof Modal.confirm>
    })
    fireEvent.click(screen.getByRole("button", { name: "Clear chat" }))
  }

  it("MR3 native clear retires H1 and persists an empty qualified checkpoint", async () => {
    seed()
    mount(true, undefined, true)
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Chat message" })).toHaveValue("Draft A"))
    clearResearchChat()
    expect(controller!.getReference()).toBeNull()
    expect(checkpoint()).toMatchObject({
      messages: [], history: [], historyId: null, serverChatId: null,
      checkpoint: { historySelectionReference: null, draft: "Draft A" }
    })
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it("MR3 native clear does not resurrect the old checkpoint on reopen", async () => {
    seed()
    const surface = mount(true, undefined, true)
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Chat message" })).toHaveValue("Draft A"))
    clearResearchChat()
    surface.unmount()
    mount(true, undefined, true)
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Chat message" })).toHaveValue("Draft A"))
    expect(useStoreMessageOption.getState().messages).toEqual([])
    expect(useStoreMessageOption.getState().serverChatId).toBeNull()
    expect(controller!.getReference()).toBeNull()
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it("MR3 native undo reacquires H1 instead of reinstalling cached rows", async () => {
    seed()
    mount(true, undefined, true)
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Chat message" })).toHaveValue("Draft A"))
    clearResearchChat()
    mocks.capture.mockClear()
    mocks.capture.mockImplementation((owner, view) => {
      const result = capture(owner, view)
      return { ...result, selected_content: [{ id: "verified-A", revision: "2", message: "Fresh native A", images: [] }] }
    })
    await act(async () => { expect(undoLatestWorkspaceAction()).toBe(true) })
    await waitFor(() => expect(mocks.capture).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(useStoreMessageOption.getState().messages[0]?.message).toBe("Fresh native A"))
    expect(controller!.getCurrent().status).toBe("ready")
    expect(checkpoint().checkpoint?.historySelectionReference).toEqual(controller!.getReference())
    expect(checkpoint().messages[0].message).toBe("Fresh native A")
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it("MR3 failed undo retains the cleared checkpoint without cached rows", async () => {
    seed()
    mount(true, undefined, true)
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Chat message" })).toHaveValue("Draft A"))
    clearResearchChat()
    const empty = structuredClone(checkpoint())
    mocks.capture.mockClear()
    mocks.capture.mockRejectedValueOnce(new Error("capture_unavailable"))
    await act(async () => { expect(undoLatestWorkspaceAction()).toBe(true) })
    await waitFor(() => expect(mocks.capture).toHaveBeenCalledTimes(1))
    expect(controller!.getReference()).toBeNull()
    expect(controller!.getCurrent().capture?.status).not.toBe("captured")
    expect(checkpoint()).toEqual(empty)
    expect(useStoreMessageOption.getState().messages).toEqual([])
    expect(useStoreMessageOption.getState().serverChatId).toBeNull()
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it("MR3 undo preserves typing while the actual H1 capture is pending", async () => {
    seed()
    mount(true, undefined, true)
    const composer = () => screen.getByRole("textbox", { name: "Chat message" })
    await waitFor(() => expect(composer()).toHaveValue("Draft A"))
    clearResearchChat()
    const pending = deferred<ReturnType<typeof capture>>()
    mocks.capture.mockReturnValueOnce(pending.promise)
    mocks.capture.mockClear()
    await act(async () => { expect(undoLatestWorkspaceAction()).toBe(true) })
    await waitFor(() => expect(mocks.capture).toHaveBeenCalledTimes(1))
    fireEvent.change(composer(), { target: { value: "Typed during undo" } })
    const [owner, view] = mocks.capture.mock.calls[0]
    await act(async () => { pending.resolve(capture(owner, view)) })
    await waitFor(() => expect(useStoreMessageOption.getState().serverChatId).toBe("chat-A"))
    expect(composer()).toHaveValue("Typed during undo")
    expect(checkpoint().checkpoint?.draft).toBe("Typed during undo")
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it.each(["workspace", "account", "selection"])("MR3 undo cannot overwrite a newer %s", async (change) => {
    seed()
    seed("B")
    mount(true, undefined, true)
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Chat message" })).toHaveValue("Draft A"))
    clearResearchChat()
    if (change === "workspace") changeWorkspace("B")
    if (change === "account") {
      mocks.scope.mockResolvedValue(scope(2))
      act(() => window.dispatchEvent(new Event("tldw:auth-principal-changed")))
    }
    if (change === "selection") await act(async () => {
      await controller!.loadConversation({ serverChatId: "chat-B", scope: { type: "workspace", workspaceId: "workspace-A" } })
      useStoreMessageOption.getState().setServerChatId("chat-B")
    })
    await waitFor(() => expect(controller!.getCurrent().status).not.toBe("loading"))
    const before = structuredClone(useStoreMessageOption.getState().messages)
    const beforeCheckpoint = structuredClone(checkpoint())
    mocks.capture.mockClear()
    await act(async () => { expect(undoLatestWorkspaceAction()).toBe(true) })
    expect(mocks.capture).not.toHaveBeenCalled()
    expect(useStoreMessageOption.getState().messages).toEqual(before)
    expect(checkpoint()).toEqual(beforeCheckpoint)
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it("MR3 confirmation cannot clear a newer H1 selection", async () => {
    seed()
    mount(true, undefined, true)
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Chat message" })).toHaveValue("Draft A"))
    let confirm: (() => void) | undefined
    vi.spyOn(Modal, "confirm").mockImplementation(config => {
      confirm = () => { config.onOk?.() }
      return { destroy: vi.fn(), update: vi.fn() } as ReturnType<typeof Modal.confirm>
    })
    fireEvent.click(screen.getByRole("button", { name: "Clear chat" }))
    await act(async () => {
      await controller!.loadConversation({ serverChatId: "chat-B", scope: { type: "workspace", workspaceId: "workspace-A" } })
      useStoreMessageOption.getState().setServerChatId("chat-B")
    })
    act(() => confirm!())
    expect(useStoreMessageOption.getState().messages[0]?.message).toBe("Verified B")
    expect(controller!.getReference()?.conversation_id).toBe("chat-B")
    expect(undoLatestWorkspaceAction()).toBe(false)
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it.each(["workspace", "selection"])("MR3 a late undo capture cannot overwrite a newer %s", async (change) => {
    seed()
    seed("B")
    mount(true, undefined, true)
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Chat message" })).toHaveValue("Draft A"))
    clearResearchChat()
    const pending = deferred<ReturnType<typeof capture>>()
    mocks.capture.mockReturnValueOnce(pending.promise)
    mocks.capture.mockClear()
    await act(async () => { expect(undoLatestWorkspaceAction()).toBe(true) })
    await waitFor(() => expect(mocks.capture).toHaveBeenCalledTimes(1))
    const [owner, view] = mocks.capture.mock.calls[0]
    if (change === "workspace") changeWorkspace("B")
    else await act(async () => {
      await controller!.loadConversation({ serverChatId: "chat-B", scope: { type: "workspace", workspaceId: "workspace-A" } })
      useStoreMessageOption.getState().setServerChatId("chat-B")
    })
    await waitFor(() => expect(useStoreMessageOption.getState().serverChatId).toBe("chat-B"))
    const before = structuredClone(checkpoint(change === "workspace" ? "B" : "A"))
    await act(async () => { pending.resolve(capture(owner, view)) })
    expect(useStoreMessageOption.getState().messages[0]?.message).toBe("Verified B")
    expect(controller!.getReference()?.conversation_id).toBe("chat-B")
    expect(checkpoint(change === "workspace" ? "B" : "A")).toEqual(before)
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it("MR3 an unexpected foreign scope reply cannot capture or save the old chat", async () => {
    seed()
    mount(true, undefined, true)
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Chat message" })).toHaveValue("Draft A"))
    clearResearchChat()
    const empty = structuredClone(checkpoint())
    mocks.capture.mockClear()
    mocks.scope.mockResolvedValueOnce(scope(2))
    await act(async () => { expect(undoLatestWorkspaceAction()).toBe(true) })
    await waitFor(() => {
      const owner = controller!.getCurrent().owner
      expect(owner?.kind).toBe("native")
      if (owner?.kind === "native") expect(owner.request_scope.userId).toBe(2)
    })
    expect(mocks.capture).not.toHaveBeenCalled()
    expect(controller!.getCurrent().owner?.kind === "native" && controller!.getCurrent().owner.validate_lease()).toBe(false)
    expect(useStoreMessageOption.getState().messages).toEqual([])
    expect(checkpoint()).toEqual(empty)
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it("MR3 a failed bookmark write cannot grant undo checkpoint authority", async () => {
    seed()
    mount(true, undefined, true)
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Chat message" })).toHaveValue("Draft A"))
    clearResearchChat()
    const empty = structuredClone(checkpoint())
    mocks.saveBookmark.mockRejectedValueOnce(new Error("bookmark_write_failed"))
    await act(async () => { expect(undoLatestWorkspaceAction()).toBe(true) })
    await waitFor(() => expect(mocks.saveBookmark).toHaveBeenCalledTimes(2))
    expect(controller!.getReference()).toBeNull()
    expect(controller!.getCurrent().status).not.toBe("ready")
    expect(useStoreMessageOption.getState().serverChatId).toBeNull()
    expect(checkpoint()).toEqual(empty)
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it("MR3-review failed bookmark undo must preserve subsequent typing on reopen", async () => {
    seed()
    const surface = mount(true, undefined, true)
    const composer = () => screen.getByRole("textbox", { name: "Chat message" })
    await waitFor(() => expect(composer()).toHaveValue("Draft A"))
    clearResearchChat()
    mocks.saveBookmark.mockRejectedValueOnce(new Error("bookmark_write_failed"))
    await act(async () => { expect(undoLatestWorkspaceAction()).toBe(true) })
    await waitFor(() => expect(controller!.getCurrent().status).not.toBe("ready"))
    fireEvent.change(composer(), { target: { value: "Typed after failed undo" } })
    expect(checkpoint().checkpoint?.draft).toBe("Typed after failed undo")
    surface.unmount()
    mount(true, undefined, true)
    await waitFor(() => expect(composer()).toHaveValue("Typed after failed undo"))
    expect(useStoreMessageOption.getState().messages).toEqual([])
    expect(controller!.getReference()).toBeNull()
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it.each(["chatId", "handoff"])("MR3-review clear consumes the original %s route without resurrection", async (kind) => {
    seed()
    const search = kind === "chatId" ? "?chatId=chat-A" : handoffSearch("A")
    window.history.replaceState({}, "", `/research-workspace${search}&tab=chat#composer`)
    const surface = mountResearchBrowserRouted()
    const composer = () => screen.getByRole("textbox", { name: "Chat message" })
    await waitFor(() => expect(composer()).toHaveValue("Draft A"))
    clearResearchChat()
    expect(window.location.search).toBe("?tab=chat")
    expect(window.location.hash).toBe("#composer")
    act(() => useStoreMessageOption.setState({ temporaryChat: true }))
    act(() => useStoreMessageOption.setState({ temporaryChat: false }))
    await waitFor(() => expect(composer()).toHaveValue("Draft A"))
    expect(useStoreMessageOption.getState().messages).toEqual([])
    surface.unmount()
    mountResearchBrowserRouted()
    await waitFor(() => expect(composer()).toHaveValue("Draft A"))
    expect(useStoreMessageOption.getState().messages).toEqual([])
    expect(controller!.getReference()).toBeNull()
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it.each(["chatId", "handoff"])("MR3 undo remains qualified after its own %s route is consumed", async (kind) => {
    seed()
    window.history.replaceState({}, "", `/research-workspace${kind === "chatId" ? "?chatId=chat-A" : handoffSearch("A")}`)
    mountResearchBrowserRouted()
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Chat message" })).toHaveValue("Draft A"))
    clearResearchChat()
    expect(mocks.scope).toHaveBeenCalledTimes(2)
    mocks.capture.mockClear()
    await act(async () => { expect(undoLatestWorkspaceAction()).toBe(true) })
    await waitFor(() => expect(mocks.capture).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(useStoreMessageOption.getState().serverChatId).toBe("chat-A"))
    expect(controller!.getCurrent().status).toBe("ready")
    expect(checkpoint().checkpoint?.historySelectionReference).toEqual(controller!.getReference())
    expect(window.location.search).toBe("")
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it("MR3-review2 clear keeps typing durable without awaiting a redundant route scope read", async () => {
    seed()
    window.history.replaceState({}, "", "/research-workspace?chatId=chat-A")
    const pending = deferred<ReturnType<typeof scope>>()
    mocks.scope.mockResolvedValueOnce(scope()).mockResolvedValueOnce(scope()).mockReturnValueOnce(pending.promise)
    const surface = mountResearchBrowserRouted()
    const composer = () => screen.getByRole("textbox", { name: "Chat message" })
    await waitFor(() => expect(composer()).toHaveValue("Draft A"))
    clearResearchChat()
    fireEvent.change(composer(), { target: { value: "Typed while route settles" } })
    surface.unmount()
    await act(async () => { pending.resolve(scope()) })
    expect(checkpoint().checkpoint?.draft).toBe("Typed while route settles")
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it("MR3-review2 clear does not consume an early undo behind a redundant scope read", async () => {
    seed()
    window.history.replaceState({}, "", "/research-workspace?chatId=chat-A")
    const pending = deferred<ReturnType<typeof scope>>()
    mocks.scope.mockResolvedValueOnce(scope()).mockResolvedValueOnce(scope()).mockReturnValueOnce(pending.promise)
    mountResearchBrowserRouted()
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Chat message" })).toHaveValue("Draft A"))
    clearResearchChat()
    mocks.capture.mockClear()
    await act(async () => { expect(undoLatestWorkspaceAction()).toBe(true) })
    await act(async () => { pending.resolve(scope()) })
    await waitFor(() => expect(mocks.capture).toHaveBeenCalledTimes(1))
    await waitFor(() => expect(useStoreMessageOption.getState().serverChatId).toBe("chat-A"))
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it("MR3-review2 current undo failure remains visible after partial authority is retired", async () => {
    seed()
    mount(true, undefined, true)
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Chat message" })).toHaveValue("Draft A"))
    clearResearchChat()
    mocks.saveBookmark.mockRejectedValueOnce(new Error("bookmark_write_failed"))
    await act(async () => { expect(undoLatestWorkspaceAction()).toBe(true) })
    await waitFor(() => expect(screen.getByText("Chat could not be restored.")).toBeVisible())
    expect(screen.queryByText("Chat restored")).not.toBeInTheDocument()
    expect(controller!.getReference()).toBeNull()
    expect(checkpoint().checkpoint?.historySelectionReference).toBeNull()
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it("MR3 delayed route acknowledgment keeps an actual pending undo gated", async () => {
    seed()
    const replace = vi.fn()
    const content = (search: string) => <HistorySelectionProvider onCapture={result => {
      const state = useStoreMessageOption.getState()
      state.setMessages(result.selected_content.map(row => ({ id: row.id, name: "Assistant", isBot: true, message: row.message, sources: [] })))
      state.setHistory(result.selected_content.map(row => ({ role: "assistant", content: row.message })))
    }}><Surface routeSearch={search} replaceRouteSearch={replace} /></HistorySelectionProvider>
    const surface = render(content("?chatId=chat-A"))
    await waitFor(() => expect(draft().value).toBe("Draft A"))
    let undo: ReturnType<NonNullable<typeof checkpointClear>>
    act(() => { undo = checkpointClear!() })
    expect(replace).toHaveBeenCalledWith("")
    const pending = deferred<ReturnType<typeof scope>>()
    mocks.scope.mockReturnValueOnce(pending.promise)
    let completion: Promise<boolean | string> | undefined
    act(() => { completion = undo!() })
    await waitFor(() => expect(mocks.scope).toHaveBeenCalledTimes(3))
    try {
      surface.rerender(content(""))
      expect(screen.getByLabelText("Restoring")).toHaveTextContent("true")
      fireEvent.change(draft(), { target: { value: "Typed during delayed navigation" } })
    } finally {
      await act(async () => { pending.resolve(scope()); await completion })
    }
    expect(screen.getByLabelText("Restoring")).toHaveTextContent("false")
    expect(useStoreMessageOption.getState().serverChatId).toBe("chat-A")
    expect(checkpoint().checkpoint?.draft).toBe("Typed during delayed navigation")
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it("MR3-review3 actual Next navigation clears a sole history query before reopen", async () => {
    seed()
    nextNavigation.enabled = true
    nextNavigation.router.asPath = "/research-workspace?chatId=chat-A"
    window.history.replaceState({}, "", nextNavigation.router.asPath)
    const surface = mountResearchBrowserRouted()
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Chat message" })).toHaveValue("Draft A"))
    clearResearchChat()
    expect(nextNavigation.router.replace).toHaveBeenCalledWith("/research-workspace")
    act(() => {
      window.history.replaceState({}, "", "/research-workspace")
      window.dispatchEvent(new PopStateEvent("popstate"))
    })
    surface.unmount()
    mountResearchBrowserRouted()
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Chat message" })).toHaveValue("Draft A"))
    expect(controller!.getReference()).toBeNull()
    expect(useStoreMessageOption.getState().messages).toEqual([])
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it("MR3-review3 a failed undo keeps the qualified empty draft through delayed route acknowledgment", async () => {
    seed()
    const content = (search: string) => <HistorySelectionProvider onCapture={result => {
      const state = useStoreMessageOption.getState()
      state.setMessages(result.selected_content.map(row => ({ id: row.id, name: "Assistant", isBot: true, message: row.message, sources: [] })))
      state.setHistory(result.selected_content.map(row => ({ role: "assistant", content: row.message })))
    }}><Surface routeSearch={search} replaceRouteSearch={vi.fn()} /></HistorySelectionProvider>
    const surface = render(content("?chatId=chat-A"))
    await waitFor(() => expect(draft()).toHaveValue("Draft A"))
    let undo: ReturnType<NonNullable<typeof checkpointClear>>
    act(() => { undo = checkpointClear!() })
    mocks.saveBookmark.mockRejectedValueOnce(new Error("bookmark_write_failed"))
    await act(async () => { expect(await undo!()).toBe("failed") })
    const reads = mocks.scope.mock.calls.length
    surface.rerender(content(""))
    expect(mocks.scope).toHaveBeenCalledTimes(reads)
    fireEvent.change(draft(), { target: { value: "Typed after failure and delayed route" } })
    surface.unmount()
    expect(checkpoint()).toMatchObject({ messages: [], history: [], historyId: null, serverChatId: null,
      checkpoint: { historySelectionReference: null, draft: "Typed after failure and delayed route" } })
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it.each(["success", "failure"])("MR3-review3 typing after capture remains an empty checkpoint before bookmark %s", async outcome => {
    seed()
    const surface = mount()
    await waitFor(() => expect(draft()).toHaveValue("Draft A"))
    let undo: ReturnType<NonNullable<typeof checkpointClear>>
    act(() => { undo = checkpointClear!() })
    const pending = deferred<void>()
    mocks.saveBookmark.mockReturnValueOnce(pending.promise)
    let completion: Promise<boolean | string> | undefined
    act(() => { completion = undo!() })
    await waitFor(() => expect(mocks.saveBookmark).toHaveBeenCalledTimes(2))
    expect(controller!.getCurrent().capture?.status).toBe("captured")
    expect(useStoreMessageOption.getState().serverChatId).toBeNull()
    fireEvent.change(draft(), { target: { value: "Typed after capture before bookmark" } })
    surface.unmount()
    await act(async () => {
      if (outcome === "failure") pending.reject(new Error("bookmark_write_failed"))
      else pending.resolve()
      await completion
    })
    expect(checkpoint()).toMatchObject({ messages: [], history: [], historyId: null, serverChatId: null,
      checkpoint: { historySelectionReference: null, draft: "Typed after capture before bookmark" } })
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it.each(["workspace", "account", "selection", "temporary"])("MR3-review3 a pending bookmark cannot save into a newer %s", async change => {
    seed()
    seed("B")
    mount()
    await waitFor(() => expect(draft()).toHaveValue("Draft A"))
    let undo: ReturnType<NonNullable<typeof checkpointClear>>
    act(() => { undo = checkpointClear!() })
    const pending = deferred<void>()
    mocks.saveBookmark.mockReturnValueOnce(pending.promise)
    let completion: Promise<boolean | string> | undefined
    act(() => { completion = undo!() })
    await waitFor(() => expect(mocks.saveBookmark).toHaveBeenCalledTimes(2))
    if (change === "workspace") changeWorkspace("B")
    if (change === "account") {
      mocks.scope.mockResolvedValue(scope(2))
      act(() => window.dispatchEvent(new Event("tldw:auth-principal-changed")))
    }
    if (change === "temporary") act(() => useStoreMessageOption.setState({ temporaryChat: true }))
    if (change === "selection") act(() => { controller!.beginLoad() })
    if (change === "workspace") await waitFor(() => expect(controller!.getCurrent().owner).toMatchObject({ conversation_id: "chat-B" }))
    if (change === "account" || change === "temporary") await waitFor(() => expect(screen.getByLabelText("Restoring")).toHaveTextContent("false"))
    fireEvent.change(draft(), { target: { value: "Typed in replacement context" } })
    const before = structuredClone(checkpoint())
    await act(async () => { pending.resolve(); expect(await completion).toBe(false) })
    expect(checkpoint()).toEqual(before)
    expect(checkpoint().checkpoint?.draft).toBe("Draft A")
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it.each(["profile", "scope failure"] as const)("MR3 own route acknowledgment preserves Undo's %s epoch handoff", async boundary => {
    seed()
    const content = (search: string) => <HistorySelectionProvider onCapture={result => {
      const state = useStoreMessageOption.getState()
      state.setMessages(result.selected_content.map(row => ({ id: row.id, name: "Assistant", isBot: true, message: row.message, sources: [] })))
      state.setHistory(result.selected_content.map(row => ({ role: "assistant", content: row.message })))
    }}><Surface routeSearch={search} replaceRouteSearch={vi.fn()} /></HistorySelectionProvider>
    const surface = render(content("?chatId=chat-A"))
    await waitFor(() => expect(draft()).toHaveValue("Draft A"))
    let undo: ReturnType<NonNullable<typeof checkpointClear>>
    act(() => { undo = checkpointClear!() })
    const pendingProfile = deferred<string>()
    if (boundary === "profile") mocks.profile.mockReturnValueOnce(pendingProfile.promise)
    else mocks.scope.mockRejectedValueOnce(new Error("scope_unavailable"))
    let completion: Promise<boolean | string> | undefined
    act(() => { completion = undo!() })
    if (boundary === "profile") await waitFor(() => expect(mocks.profile).toHaveBeenCalledTimes(2))
    else await act(async () => { expect(await completion).toBe("failed") })
    const reads = mocks.scope.mock.calls.length
    const unexpectedRead = deferred<ReturnType<typeof scope>>()
    mocks.scope.mockReturnValueOnce(unexpectedRead.promise)
    try {
      surface.rerender(content(""))
      expect(mocks.scope).toHaveBeenCalledTimes(reads)
      fireEvent.change(draft(), { target: { value: "Typed through own epoch handoff" } })
      surface.unmount()
      expect(checkpoint()).toMatchObject({ messages: [], history: [], historyId: null, serverChatId: null,
        checkpoint: { historySelectionReference: null, draft: "Typed through own epoch handoff" } })
    } finally {
      await act(async () => {
        pendingProfile.resolve("profile")
        unexpectedRead.resolve(scope())
        await completion
      })
    }
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it.each(["native", "local"] as const)("MR3 pre-owner Undo cannot checkpoint a newer %s capture", async kind => {
    seed()
    mount()
    await waitFor(() => expect(draft()).toHaveValue("Draft A"))
    let undo: ReturnType<NonNullable<typeof checkpointClear>>
    act(() => { undo = checkpointClear!() })
    const pendingScope = deferred<ReturnType<typeof scope>>()
    mocks.scope.mockReturnValueOnce(pendingScope.promise)
    let completion: Promise<boolean | string> | undefined
    act(() => { completion = undo!() })
    await waitFor(() => expect(mocks.scope).toHaveBeenCalledTimes(3))
    const pendingBookmark = deferred<void>()
    mocks.saveBookmark.mockReturnValueOnce(pendingBookmark.promise)
    let replacement: Promise<boolean> | undefined
    act(() => {
      replacement = controller!.open(kind === "native" ? {
        kind: "native", conversation_id: "chat-A", request_scope: scope(),
        scope: { type: "workspace", workspaceId: "workspace-A" }, validate_lease: () => true
      } : { kind: "local", profile_id: "profile", owner_key: "native-local-new", conversation_id: "local-new" })
    })
    try {
      await waitFor(() => expect(mocks.saveBookmark).toHaveBeenCalledTimes(2))
      expect(controller!.getCurrent().capture?.status).toBe("captured")
      expect(useStoreMessageOption.getState().serverChatId).toBeNull()
      fireEvent.change(draft(), { target: { value: "Typed in newer pre-owner selection" } })
      expect(checkpoint().checkpoint?.draft).toBe("Draft A")
    } finally {
      await act(async () => {
        pendingScope.resolve(scope())
        pendingBookmark.resolve()
        await replacement
        expect(await completion).toBe(false)
      })
    }
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it.each(["stored", "chat route", "handoff route"])("blocks temporary %s restoration before H1 profile/bookmark writes", async (target) => {
    seed()
    const original = structuredClone(checkpoint())
    const bookmarks = structuredClone([...mocks.bookmarks])
    useStoreMessageOption.setState({ temporaryChat: true })
    const surface = target === "stored" ? mount() : mountRouted(target === "chat route" ? "?chatId=chat-A" : handoffSearch("A"))
    await waitFor(() => expect(screen.getByLabelText("Restoring")).toHaveTextContent("false"))
    expect(mocks.profile).not.toHaveBeenCalled()
    expect(mocks.saveBookmark).not.toHaveBeenCalled()
    expect(mocks.capture).not.toHaveBeenCalled()
    expect(controller!.getCurrent().error).toBe("temporary_history_unavailable")
    expect(controller!.getReference()).toBeNull()
    expect(checkpointFence!()()).toBe(false)
    expect(transcript()).toBe("")
    expect(useStoreMessageOption.getState().serverChatId).toBeNull()
    expect(checkpoint()).toEqual(original)
    surface.unmount()
    expect([...mocks.bookmarks]).toEqual(bookmarks)
    expect(checkpoint()).toEqual(original)
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it("retires a pending normal restore when temporary mode changes", async () => {
    seed()
    const original = structuredClone(checkpoint())
    const bookmarks = structuredClone([...mocks.bookmarks])
    const pending = deferred<ReturnType<typeof scope>>()
    mocks.scope.mockReturnValueOnce(pending.promise)
    mount()
    act(() => useStoreMessageOption.setState({ temporaryChat: true }))
    await act(async () => { pending.resolve(scope()) })
    await waitFor(() => expect(screen.getByLabelText("Restoring")).toHaveTextContent("false"))
    expect(mocks.profile).not.toHaveBeenCalled()
    expect(mocks.saveBookmark).not.toHaveBeenCalled()
    expect(mocks.capture).not.toHaveBeenCalled()
    expect([...mocks.bookmarks]).toEqual(bookmarks)
    expect(checkpoint()).toEqual(original)
    expect(checkpointFence!()()).toBe(false)
    expect(transcript()).toBe("")
    act(() => useStoreMessageOption.setState({ temporaryChat: false }))
    await waitFor(() => expect(transcript()).toBe("Verified A"))
    expect(draft().value).toBe("Draft A")
    expect(checkpointFence!()()).toBe(true)
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it("retires a ready normal capture in temporary mode and recovers its qualified draft", async () => {
    seed()
    mount()
    await waitFor(() => expect(draft().value).toBe("Draft A"))
    fireEvent.change(draft(), { target: { value: "Unsent qualified draft" } })
    const original = structuredClone(checkpoint())
    const bookmarks = structuredClone([...mocks.bookmarks])
    mocks.profile.mockClear()
    mocks.saveBookmark.mockClear()
    mocks.capture.mockClear()
    const oldFence = checkpointFence!()
    act(() => useStoreMessageOption.setState({ temporaryChat: true }))
    await waitFor(() => expect(screen.getByLabelText("Restoring")).toHaveTextContent("false"))
    expect(oldFence()).toBe(false)
    expect(checkpointFence!()()).toBe(false)
    expect(controller!.getCurrent().error).toBe("temporary_history_unavailable")
    expect(controller!.getReference()).toBeNull()
    expect(transcript()).toBe("")
    expect(mocks.profile).not.toHaveBeenCalled()
    expect(mocks.saveBookmark).not.toHaveBeenCalled()
    expect(mocks.capture).not.toHaveBeenCalled()
    expect([...mocks.bookmarks]).toEqual(bookmarks)
    expect(checkpoint()).toEqual(original)
    act(() => useStoreMessageOption.setState({ temporaryChat: false }))
    await waitFor(() => expect(draft().value).toBe("Unsent qualified draft"))
    expect(transcript()).toBe("Verified A")
    expect(checkpointFence!()()).toBe(true)
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it("rejects a late normal capture after entering temporary mode", async () => {
    seed()
    const original = structuredClone(checkpoint())
    const bookmarks = structuredClone([...mocks.bookmarks])
    const pending = deferred<ReturnType<typeof capture>>()
    mocks.capture.mockReturnValueOnce(pending.promise)
    mount()
    await waitFor(() => expect(mocks.capture).toHaveBeenCalledTimes(1))
    const [owner, view] = mocks.capture.mock.calls[0] as Parameters<typeof capture>
    act(() => useStoreMessageOption.setState({ temporaryChat: true }))
    await waitFor(() => expect(controller!.getCurrent().error).toBe("temporary_history_unavailable"))
    mocks.profile.mockClear()
    mocks.capture.mockClear()
    await act(async () => { pending.resolve(capture(owner, view)) })
    expect(controller!.getCurrent().error).toBe("temporary_history_unavailable")
    expect(controller!.getReference()).toBeNull()
    expect(transcript()).toBe("")
    expect(checkpointFence!()()).toBe(false)
    expect(mocks.profile).not.toHaveBeenCalled()
    expect(mocks.saveBookmark).not.toHaveBeenCalled()
    expect(mocks.capture).not.toHaveBeenCalled()
    expect([...mocks.bookmarks]).toEqual(bookmarks)
    expect(checkpoint()).toEqual(original)
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it.each(["scope", "capture"])("retains newer in-memory typing across temporary mode retirement during %s", async (boundary) => {
    seed()
    const original = structuredClone(checkpoint())
    const pendingScope = deferred<ReturnType<typeof scope>>()
    const pendingCapture = deferred<ReturnType<typeof capture>>()
    if (boundary === "scope") mocks.scope.mockReturnValueOnce(pendingScope.promise)
    else mocks.capture.mockReturnValueOnce(pendingCapture.promise)
    mount()
    if (boundary === "capture") await waitFor(() => expect(mocks.capture).toHaveBeenCalledTimes(1))
    const firstCapture = mocks.capture.mock.calls[0] as Parameters<typeof capture> | undefined
    fireEvent.change(draft(), { target: { value: "Newer uncheckpointed draft" } })
    act(() => useStoreMessageOption.setState({ temporaryChat: true }))
    await waitFor(() => expect(controller!.getCurrent().error).toBe("temporary_history_unavailable"))
    expect(draft().value).toBe("Newer uncheckpointed draft")
    expect(checkpoint()).toEqual(original)
    expect(mocks.saveBookmark).not.toHaveBeenCalled()
    await act(async () => {
      if (boundary === "scope") pendingScope.resolve(scope())
      else pendingCapture.resolve(capture(...firstCapture!))
    })
    expect(draft().value).toBe("Newer uncheckpointed draft")
    expect(checkpointFence!()()).toBe(false)
    expect(checkpoint()).toEqual(original)
    expect(mocks.saveBookmark).not.toHaveBeenCalled()
    act(() => useStoreMessageOption.setState({ temporaryChat: false }))
    await waitFor(() => expect(transcript()).toBe("Verified A"))
    expect(draft().value).toBe("Newer uncheckpointed draft")
    expect(checkpointFence!()()).toBe(true)
    expect(checkpoint()!.checkpoint!.draft).toBe("Newer uncheckpointed draft")
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it("reports an unexpected scope failure without exposing details or changing its saved checkpoint", async () => {
    seed()
    const original = structuredClone(checkpoint())
    const pending = deferred<ReturnType<typeof scope>>()
    mocks.scope.mockReturnValueOnce(pending.promise)
    mount()
    fireEvent.change(draft(), { target: { value: "Newer unsent draft" } })
    await act(async () => { pending.reject(new Error("token=private-secret database failed")) })
    await waitFor(() => expect(screen.getByLabelText("Restoration error")).toHaveTextContent("Workspace chat restoration failed"))
    expect(screen.getByLabelText("Restoration error")).not.toHaveTextContent("private-secret")
    expect(screen.getByLabelText("Restoring")).toHaveTextContent("false")
    expect(draft().value).toBe("Newer unsent draft")
    expect(checkpoint()).toEqual(original)
    expect(checkpointFence!()()).toBe(false)
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it("reports an unexpected checkpoint read failure without granting save authority", async () => {
    seed()
    const original = structuredClone(checkpoint())
    vi.spyOn(useWorkspaceStore.getState(), "getWorkspaceChatSession").mockImplementation(() => { throw new Error("storage failed") })
    const surface = mount()
    await waitFor(() => expect(screen.getByLabelText("Restoration error")).toHaveTextContent("Workspace chat restoration failed"))
    expect(checkpointFence!()()).toBe(false)
    surface.unmount()
    expect(checkpoint()).toEqual(original)
    expect(mocks.capture).not.toHaveBeenCalled()
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it("keeps expected scope cancellation silent and preserves its saved checkpoint", async () => {
    seed()
    const original = structuredClone(checkpoint())
    mocks.scope.mockRejectedValueOnce(new DOMException("Cancelled", "AbortError"))
    mount()
    await waitFor(() => expect(screen.getByLabelText("Restoring")).toHaveTextContent("false"))
    expect(screen.getByLabelText("Restoration error")).toBeEmptyDOMElement()
    expect(checkpoint()).toEqual(original)
    expect(checkpointFence!()()).toBe(false)
  })

  it("does not publish a superseded workspace rejection into the current restored workspace", async () => {
    seed(); seed("B")
    const original = structuredClone(checkpoint())
    const pending = deferred<ReturnType<typeof scope>>()
    mocks.scope.mockReturnValueOnce(pending.promise)
    mount()
    changeWorkspace("B")
    await waitFor(() => expect(draft().value).toBe("Draft B"))
    await act(async () => { pending.reject(new Error("Late workspace A failure")) })
    expect(screen.getByLabelText("Restoration error")).toBeEmptyDOMElement()
    expect(draft().value).toBe("Draft B")
    expect(checkpoint()).toEqual(original)
    expect(checkpointFence!()()).toBe(true)
  })

  it("clears the restoration error after a fresh qualified restore without sending", async () => {
    seed()
    mocks.scope.mockRejectedValueOnce(new Error("Unavailable"))
    mount()
    await waitFor(() => expect(screen.getByLabelText("Restoration error")).toHaveTextContent("Workspace chat restoration failed"))
    act(() => { mocks.listeners.forEach(listener => listener()) })
    await waitFor(() => expect(draft().value).toBe("Draft A"))
    expect(screen.getByLabelText("Restoration error")).toBeEmptyDOMElement()
    expect(checkpointFence!()()).toBe(true)
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it.each([true, false])("does not autosave-loop when Research observes persisted sessions (H1 %s)", async active => {
    if (active) seed()
    else {
      const legacy = saved()
      delete legacy.checkpoint
      useWorkspaceStore.setState({ workspaceChatSessions: { "workspace-A::reference-A": legacy } })
    }
    mount(active, "workspace-A::reference-A", false, true)
    await waitFor(() => expect(transcript()).toBe(active ? "Verified A" : "UNTRUSTED CACHE"))
    if (active) fireEvent.change(draft(), { target: { value: "Research persisted edit" } })
    else act(() => useStoreMessageOption.getState().setMessages([{ id: "edited", name: "You", isBot: false, message: "Research persisted edit", sources: [] }]))
    await waitFor(() => expect(active ? checkpoint().checkpoint?.draft : checkpoint().messages[0].message).toBe("Research persisted edit"))
    const unchanged = checkpoint()
    act(() => useWorkspaceStore.getState().saveWorkspaceChatSession("workspace-B::reference-B", saved("B")))
    expect(checkpoint()).toBe(unchanged)
    expect(mocks.submit).not.toHaveBeenCalled()
  })
  it.each(["chatId", "chat_id", "serverChatId", "server_chat_id"])("route %s explicitly opens native workspace B instead of stored A on a fresh document", async alias => {
    seed()
    window.history.replaceState({}, "", `/chat-workspace?${alias}=%20chat-B%20`)
    mount()
    await waitFor(() => expect(transcript()).toBe("Verified B"))
    expect(useStoreMessageOption.getState().serverChatId).toBe("chat-B")
    expect(draft().value).toBe("")
    expect(controller!.getCurrent().owner).toMatchObject({ kind: "native", scope: { type: "workspace", workspaceId: "workspace-A" } })
    expect(checkpoint().checkpoint?.historySelectionReference?.conversation_id).toBe("chat-B")
    expect(checkpointFence!()()).toBe(true)
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it("route H1 handoff uses its owned bookmark and never adopts stored A rows or draft", async () => {
    seed(); seed("B")
    window.history.replaceState({}, "", `/chat-workspace${handoffSearch()}`)
    mount()
    await waitFor(() => expect(transcript()).toBe("Verified B"))
    expect(draft().value).toBe("")
    expect(checkpoint().messages.map(row => row.message)).toEqual(["Verified B"])
    expect(controller!.getReference()?.client_session_id).not.toBe("saved-B")
    expect(useStoreMessageOption.getState().historyId).toBeNull()
  })

  it.each(["chatId", "historySelection"])("route matching %s reload restores the qualified draft only after the native read", async kind => {
    seed()
    const pending = deferred<ReturnType<typeof capture>>()
    mocks.capture.mockReturnValueOnce(pending.promise)
    window.history.replaceState({}, "", `/chat-workspace${kind === "chatId" ? "?chatId=chat-A" : handoffSearch("A")}`)
    mount()
    await waitFor(() => expect(mocks.capture).toHaveBeenCalled())
    expect(draft().value).toBe("")
    expect(useStoreMessageOption.getState().serverChatId).toBeNull()
    const [owner, view] = mocks.capture.mock.calls[0]
    await act(async () => { pending.resolve(capture(owner, view)) })
    await waitFor(() => expect(transcript()).toBe("Verified A"))
    expect(draft().value).toBe("Draft A")
    expect(checkpoint().checkpoint?.draft).toBe("Draft A")
    expect(checkpointFence!()()).toBe(true)
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it("route matching reload cannot replace newer typing with its qualified draft", async () => {
    seed()
    const pending = deferred<ReturnType<typeof capture>>()
    mocks.capture.mockReturnValueOnce(pending.promise)
    window.history.replaceState({}, "", `/chat-workspace${handoffSearch("A")}`)
    mount()
    await waitFor(() => expect(mocks.capture).toHaveBeenCalled())
    const [owner, view] = mocks.capture.mock.calls[0]
    fireEvent.change(draft(), { target: { value: "Newer A" } })
    await act(async () => { pending.resolve(capture(owner, view)) })
    await waitFor(() => expect(transcript()).toBe("Verified A"))
    expect(draft().value).toBe("Newer A")
    expect(checkpoint().checkpoint?.draft).toBe("Newer A")
  })

  it.each(["owner", "profile"])("route matching conversation does not adopt a draft from a different H1 %s", async mismatch => {
    const session = saved()
    session.checkpoint!.historySelectionReference = {
      ...reference(), ...(mismatch === "owner" ? { owner_key: "foreign-owner" } : { profile_id: "foreign-profile" })
    }
    useWorkspaceStore.getState().saveWorkspaceChatSession("workspace-A::reference-A", session)
    window.history.replaceState({}, "", "/chat-workspace?chatId=chat-A")
    mount()
    await waitFor(() => expect(transcript()).toBe("Verified A"))
    expect(draft().value).toBe("")
  })

  it("route same-path react-router query changes open the next scoped H1 selection", async () => {
    seed()
    mountRouted()
    await waitFor(() => expect(transcript()).toBe("Verified B"))
    fireEvent.change(draft(), { target: { value: "Outgoing B" } })
    fireEvent.click(screen.getByRole("button", { name: "Route C" }))
    await waitFor(() => expect(transcript()).toBe("Verified C"))
    expect(useStoreMessageOption.getState().serverChatId).toBe("chat-C")
    expect(draft().value).toBe("")
    expect(checkpoint().checkpoint?.historySelectionReference?.conversation_id).toBe("chat-C")
  })

  it.each(["/chat-workspace", "/research-workspace"])("route root search context drives the shared hook on %s without a direct router hook", async path => {
    seed()
    mountRouted("?chatId=chat-B", true, path)
    await waitFor(() => expect(transcript()).toBe("Verified B"))
    fireEvent.click(screen.getByRole("button", { name: "Route C" }))
    await waitFor(() => expect(transcript()).toBe("Verified C"))
    expect(useStoreMessageOption.getState().serverChatId).toBe("chat-C")
    expect(checkpointFence!()()).toBe(true)
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  describe.each(["/chat-workspace", "/research-workspace"])("ROUTE-F1 root context on %s", path => {
    it.each(["back", "forward"] as const)("same-query %s replaces the ready lease and persists newer drafts", async direction => {
      seed()
      window.history.replaceState({}, "", `${path}?chatId=chat-A#before`)
      window.history.pushState({}, "", `${path}?chatId=chat-A#after`)
      if (direction === "forward") await traverseHistory("back")
      mountBrowserRouted()
      await waitFor(() => expect(checkpointFence!()()).toBe(true))
      expect(transcript()).toBe("Verified A")
      fireEvent.change(draft(), { target: { value: "Before history navigation" } })
      const oldFence = checkpointFence!()
      const oldOwner = controller!.getCurrent().owner

      await traverseHistory(direction)

      expect(window.location.search).toBe("?chatId=chat-A")
      expect(window.location.hash).toBe(direction === "back" ? "#before" : "#after")
      expect(oldFence()).toBe(false)
      expect(oldOwner?.kind === "native" && oldOwner.validate_lease()).toBe(false)
      await waitFor(() => expect(checkpointFence!()()).toBe(true))
      const owner = controller!.getCurrent().owner
      expect(owner).not.toBe(oldOwner)
      expect(owner?.kind === "native" && owner.validate_lease()).toBe(true)
      expect(screen.getByLabelText("Restoring").textContent).toBe("false")
      expect(transcript()).toBe("Verified A")
      expect(useStoreMessageOption.getState().serverChatId).toBe("chat-A")
      expect(draft().value).toBe("Before history navigation")
      fireEvent.change(draft(), { target: { value: "After history navigation" } })
      expect(checkpoint().checkpoint?.draft).toBe("After history navigation")
      expect(checkpoint().checkpoint?.historySelectionReference).toEqual(controller!.getReference())
      expect(mocks.submit).not.toHaveBeenCalled()
    })

    it.each(["back", "forward"] as const)("same-query %s fences a pending capture and restores a replacement lease", async direction => {
      seed()
      const before = structuredClone(checkpoint())
      const pending = deferred<ReturnType<typeof capture>>()
      const replacement = deferred<ReturnType<typeof capture>>()
      mocks.capture.mockReturnValueOnce(pending.promise).mockReturnValueOnce(replacement.promise)
      window.history.replaceState({}, "", `${path}?chatId=chat-A#before`)
      window.history.pushState({}, "", `${path}?chatId=chat-A#after`)
      if (direction === "forward") await traverseHistory("back")
      mountBrowserRouted()
      await waitFor(() => expect(mocks.capture).toHaveBeenCalledTimes(1))
      const [oldOwner, oldView] = mocks.capture.mock.calls[0]
      const oldFence = checkpointFence!()

      await traverseHistory(direction)

      expect(window.location.search).toBe("?chatId=chat-A")
      await waitFor(() => expect(mocks.capture).toHaveBeenCalledTimes(2))
      const [owner, view] = mocks.capture.mock.calls[1]
      await act(async () => { pending.resolve(capture(oldOwner, oldView)) })
      expect(oldOwner.validate_lease()).toBe(false)
      expect(oldFence()).toBe(false)
      expect(checkpointFence!()()).toBe(false)
      expect(transcript()).toBe("")
      expect(useStoreMessageOption.getState().serverChatId).toBeNull()
      expect(checkpoint()).toEqual(before)
      fireEvent.change(draft(), { target: { value: "Typed during replacement" } })
      await act(async () => { replacement.resolve(capture(owner, view)) })
      await waitFor(() => expect(checkpointFence!()()).toBe(true))
      expect(owner.validate_lease()).toBe(true)
      expect(transcript()).toBe("Verified A")
      expect(useStoreMessageOption.getState().serverChatId).toBe("chat-A")
      expect(screen.getByLabelText("Restoring").textContent).toBe("false")
      expect(draft().value).toBe("Typed during replacement")
      expect(checkpoint().checkpoint?.draft).toBe("Typed during replacement")
      fireEvent.change(draft(), { target: { value: "New draft after replacement" } })
      expect(checkpoint().checkpoint?.draft).toBe("New draft after replacement")
      expect(checkpoint().checkpoint?.historySelectionReference).toEqual(controller!.getReference())
      expect(mocks.submit).not.toHaveBeenCalled()
    })
  })

  it.each(["?historySelection=broken", "?historySelection=", "?chatId=../foreign", "?chatId=", "?chatId=chat-B&serverChatId=chat-C", "?historyId=legacy"])("route invalid address %s fails closed without checkpoint overwrite", async search => {
    seed()
    const before = structuredClone(checkpoint())
    window.history.replaceState({}, "", `/chat-workspace${search}`)
    mount()
    await waitFor(() => expect(screen.getByLabelText("Restoring").textContent).toBe("false"))
    fireEvent.change(draft(), { target: { value: "Not adopted" } })
    expect(mocks.capture).not.toHaveBeenCalled()
    expect(transcript()).toBe("")
    expect(useStoreMessageOption.getState().serverChatId).toBeNull()
    expect(checkpoint()).toEqual(before)
    expect(checkpointFence!()()).toBe(false)
  })

  it.each(["local", "foreign profile", "foreign owner", "foreign workspace"])("route %s handoff cannot gain checkpoint authority", async mismatch => {
    seed(); seed("B")
    const before = structuredClone(checkpoint())
    const address = { ...reference("B"), owner_kind: mismatch === "local" ? "local" : "native" }
    if (mismatch === "foreign profile") address.profile_id = "another-profile"
    if (mismatch === "foreign owner") address.owner_key = "foreign-owner"
    if (mismatch === "foreign workspace") mocks.capture.mockRejectedValue(new Error("owner_conversation_mismatch"))
    window.history.replaceState({}, "", `/chat-workspace?historySelection=${encodeURIComponent(JSON.stringify(address))}`)
    mount()
    await waitFor(() => expect(screen.getByLabelText("Restoring").textContent).toBe("false"))
    expect(transcript()).toBe("")
    expect(draft().value).toBe("")
    expect(useStoreMessageOption.getState().serverChatId).toBeNull()
    expect(checkpoint()).toEqual(before)
    expect(checkpointFence!()()).toBe(false)
  })

  it("route query replacement fences the old capture before H1 can publish it", async () => {
    seed()
    const pending = deferred<ReturnType<typeof capture>>()
    mocks.capture.mockReturnValueOnce(pending.promise)
    mountRouted()
    await waitFor(() => expect(mocks.capture).toHaveBeenCalled())
    const [owner, view] = mocks.capture.mock.calls[0]
    fireEvent.click(screen.getByRole("button", { name: "Route C" }))
    await waitFor(() => expect(transcript()).toBe("Verified C"))
    fireEvent.change(draft(), { target: { value: "Current C" } })
    await act(async () => { pending.resolve(capture(owner, view)) })
    expect(transcript()).toBe("Verified C")
    expect(draft().value).toBe("Current C")
    expect(useStoreMessageOption.getState().serverChatId).toBe("chat-C")
    expect(checkpoint().checkpoint?.draft).toBe("Current C")
  })

  it("route query replacement during scope resolution cannot open the superseded address", async () => {
    seed()
    const pending = deferred<ReturnType<typeof scope>>()
    mocks.scope.mockReturnValueOnce(pending.promise)
    mountRouted()
    fireEvent.click(screen.getByRole("button", { name: "Route C" }))
    await waitFor(() => expect(transcript()).toBe("Verified C"))
    await act(async () => { pending.resolve(scope()) })
    expect(mocks.capture.mock.calls.every(([owner]) => owner.conversation_id === "chat-C")).toBe(true)
    expect(useStoreMessageOption.getState().serverChatId).toBe("chat-C")
  })

  it("route query A/B/A cannot publish the first A read into the current A lease", async () => {
    seed()
    const pending = deferred<ReturnType<typeof capture>>()
    mocks.capture.mockReturnValueOnce(pending.promise)
    mountRouted("?chatId=chat-A", true)
    await waitFor(() => expect(mocks.capture).toHaveBeenCalled())
    const [owner, view] = mocks.capture.mock.calls[0]
    fireEvent.click(screen.getByRole("button", { name: "Route B" }))
    await waitFor(() => expect(transcript()).toBe("Verified B"))
    fireEvent.click(screen.getByRole("button", { name: "Route A" }))
    await waitFor(() => expect(transcript()).toBe("Verified A"))
    const current = controller!.getReference()
    fireEvent.change(draft(), { target: { value: "Current A query" } })
    await act(async () => { pending.resolve(capture(owner, view)) })
    expect(controller!.getReference()).toEqual(current)
    expect(draft().value).toBe("Current A query")
    expect(checkpoint().checkpoint?.draft).toBe("Current A query")
  })

  it("route replacement plus New Chat reference rotation retires the old pending query", async () => {
    seed()
    const pending = deferred<ReturnType<typeof capture>>()
    mocks.capture.mockReturnValueOnce(pending.promise)
    mountRouted("?chatId=chat-B", true)
    await waitFor(() => expect(mocks.capture).toHaveBeenCalled())
    const [owner, view] = mocks.capture.mock.calls[0]
    const before = structuredClone(checkpoint())
    act(() => {
      window.dispatchEvent(new Event("tldw:chat-route-replacement"))
      useStoreMessageOption.setState({ messages: [], history: [], historyId: null, serverChatId: null })
      useWorkspaceStore.setState({ workspaceChatReferenceId: "new-chat" })
    })
    await act(async () => { pending.resolve(capture(owner, view)) })
    expect(transcript()).toBe("")
    expect(useStoreMessageOption.getState().serverChatId).toBeNull()
    expect(checkpoint()).toEqual(before)
    expect(checkpoint("A", "new-chat")).toBeUndefined()
  })

  it("route newer typing wins over the explicit address read without restoring A's draft", async () => {
    seed(); seed("B")
    const pending = deferred<ReturnType<typeof capture>>()
    mocks.capture.mockReturnValueOnce(pending.promise)
    window.history.replaceState({}, "", `/chat-workspace${handoffSearch()}`)
    mount()
    await waitFor(() => expect(mocks.capture).toHaveBeenCalled())
    const [owner, view] = mocks.capture.mock.calls[0]
    fireEvent.change(draft(), { target: { value: "Typed for B" } })
    await act(async () => { pending.resolve(capture(owner, view)) })
    await waitFor(() => expect(transcript()).toBe("Verified B"))
    expect(draft().value).toBe("Typed for B")
    expect(checkpoint().checkpoint?.draft).toBe("Typed for B")
  })

  it.each(["account", "workspace"])("route %s change during the explicit capture cannot publish stale sensitive rows", async change => {
    seed(); seed("B")
    const before = structuredClone(checkpoint())
    const pending = deferred<ReturnType<typeof capture>>()
    mocks.capture.mockReturnValueOnce(pending.promise)
    window.history.replaceState({}, "", `/chat-workspace${handoffSearch()}`)
    mount()
    await waitFor(() => expect(mocks.capture).toHaveBeenCalled())
    const [owner, view] = mocks.capture.mock.calls[0]
    mocks.capture.mockRejectedValue(new Error("owner_conversation_mismatch"))
    if (change === "account") {
      mocks.scope.mockResolvedValue(scope(2))
      act(() => { mocks.listeners.forEach(listener => listener()) })
    } else changeWorkspace("B")
    await act(async () => { pending.resolve(capture(owner, view)) })
    await waitFor(() => expect(screen.getByLabelText("Restoring").textContent).toBe("false"))
    expect(transcript()).toBe("")
    expect(draft().value).toBe("")
    expect(useStoreMessageOption.getState().serverChatId).toBeNull()
    expect(checkpoint()).toEqual(before)
  })
  it("stays inactive without H1 context and preserves qualified storage", async () => {
    seed()
    const before = checkpoint()
    const surface = mount(false)
    fireEvent.change(draft(), { target: { value: "Legacy draft" } })
    surface.unmount()
    expect(mocks.scope).not.toHaveBeenCalled()
    expect(checkpoint()).toEqual(before)
  })

  it("waits for completed hydration and verified scope before reading/restoring", async () => {
    seed()
    const pending = deferred<ReturnType<typeof scope>>()
    mocks.scope.mockReturnValue(pending.promise)
    useWorkspaceStore.setState({ storeHydrated: false })
    mount()
    expect(mocks.scope).not.toHaveBeenCalled()
    act(() => { useWorkspaceStore.setState({ storeHydrated: true }) })
    await waitFor(() => expect(mocks.scope).toHaveBeenCalled())
    expect(transcript()).toBe("")
    expect(checkpoint().checkpoint?.draft).toBe("Draft A")
    await act(async () => { pending.resolve(scope()) })
    await waitFor(() => expect(draft().value).toBe("Draft A"))
    expect(transcript()).toBe("Verified A")
  })

  it("restores only H1 captured rows and keeps its new bookmark reference", async () => {
    seed()
    mount()
    await waitFor(() => expect(draft().value).toBe("Draft A"))
    expect(transcript()).toBe("Verified A")
    expect(useStoreMessageOption.getState().serverChatId).toBe("chat-A")
    expect(checkpoint().checkpoint?.historySelectionReference).toEqual(controller!.getReference())
    expect(checkpoint().messages[0].id).toBe("verified-A")
  })

  it("typing during a deferred history capture wins over the saved draft", async () => {
    seed()
    const pending = deferred<ReturnType<typeof capture>>()
    mocks.capture.mockReturnValueOnce(pending.promise)
    mount()
    await waitFor(() => expect(mocks.capture).toHaveBeenCalled())
    fireEvent.change(draft(), { target: { value: "New typing" } })
    const [owner, view] = mocks.capture.mock.calls[0]
    await act(async () => { pending.resolve(capture(owner, view)) })
    await waitFor(() => expect(transcript()).toBe("Verified A"))
    expect(draft().value).toBe("New typing")
    expect(checkpoint().checkpoint?.draft).toBe("New typing")
  })

  it.each(["historySelection", "chatId"])("explicit %s route intent does not restore a checkpoint", async key => {
    seed()
    window.history.replaceState({}, "", `/chat-workspace?${key}=explicit`)
    mount()
    await waitFor(() => expect(screen.getByLabelText("Restoring").textContent).toBe("false"))
    expect(draft().value).toBe("")
    if (key === "historySelection") {
      expect(mocks.capture).not.toHaveBeenCalled()
      expect(checkpoint().checkpoint?.draft).toBe("Draft A")
    } else {
      expect(transcript()).toBe("Verified explicit")
      expect(useStoreMessageOption.getState().serverChatId).toBe("explicit")
      expect(checkpoint().checkpoint?.historySelectionReference?.conversation_id).toBe("explicit")
    }
  })

  it("route replacement during scope resolution fences the deferred restore", async () => {
    seed()
    const pending = deferred<ReturnType<typeof scope>>()
    mocks.scope.mockReturnValueOnce(pending.promise)
    mount()
    act(() => window.dispatchEvent(new Event("tldw:chat-route-replacement")))
    fireEvent.change(draft(), { target: { value: "Explicit new draft" } })
    await act(async () => { pending.resolve(scope()) })
    expect(mocks.capture).not.toHaveBeenCalled()
    expect(draft().value).toBe("Explicit new draft")
  })

  it("New Chat reference change prevents the pending old capture from resurrecting", async () => {
    seed()
    const pending = deferred<ReturnType<typeof capture>>()
    mocks.capture.mockReturnValueOnce(pending.promise)
    mount()
    await waitFor(() => expect(mocks.capture).toHaveBeenCalled())
    const [owner, view] = mocks.capture.mock.calls[0]
    changeWorkspace("A", "new-chat")
    fireEvent.change(draft(), { target: { value: "New chat draft" } })
    await act(async () => { pending.resolve(capture(owner, view)) })
    expect(transcript()).toBe("")
    expect(useStoreMessageOption.getState().serverChatId).toBeNull()
    expect(draft().value).toBe("New chat draft")
    expect(checkpoint("A", "new-chat").checkpoint?.historySelectionReference).toBeNull()
  })

  it("workspace A/B/A does not accept the first A deferred capture", async () => {
    seed("A"); seed("B")
    const pending = deferred<ReturnType<typeof capture>>()
    mocks.capture.mockReturnValueOnce(pending.promise)
    mount()
    await waitFor(() => expect(mocks.capture).toHaveBeenCalled())
    const [owner, view] = mocks.capture.mock.calls[0]
    changeWorkspace("B")
    await waitFor(() => expect(draft().value).toBe("Draft B"))
    changeWorkspace("A")
    await waitFor(() => expect(draft().value).toBe("Draft A"))
    fireEvent.change(draft(), { target: { value: "Current A" } })
    await act(async () => { pending.resolve(capture(owner, view)) })
    expect(draft().value).toBe("Current A")
    expect(transcript()).toBe("Verified A")
    expect(checkpoint("B").checkpoint?.draft).toBe("Draft B")
  })

  it("account/target invalidation clears mounted sensitive state and fences old scope ABA", async () => {
    seed()
    mount()
    await waitFor(() => expect(draft().value).toBe("Draft A"))
    const before = structuredClone(checkpoint())
    const pending = deferred<ReturnType<typeof scope>>()
    mocks.scope.mockReturnValueOnce(pending.promise).mockResolvedValue(scope(2, "http://other.test"))
    act(() => { for (const listener of [...mocks.listeners]) listener() })
    expect(draft().value).toBe("")
    expect(transcript()).toBe("")
    expect(useStoreMessageOption.getState().historyId).toBeNull()
    expect(useStoreMessageOption.getState().serverChatId).toBeNull()
    act(() => { for (const listener of [...mocks.listeners]) listener() })
    await act(async () => { pending.resolve(scope()) })
    expect(transcript()).toBe("")
    expect(checkpoint()).toEqual(before)
    expect(mocks.stop).toHaveBeenCalled()
  })

  it("rejects a failed H1 owner capture without using cached rows, IDs or draft", async () => {
    seed()
    mocks.capture.mockRejectedValue(new Error("owner_conversation_mismatch"))
    const before = structuredClone(checkpoint())
    mount()
    await waitFor(() => expect(controller?.status).toBe("unsupported_history_capability"))
    expect(transcript()).toBe("")
    expect(draft().value).toBe("")
    expect(useStoreMessageOption.getState().serverChatId).toBeNull()
    expect(checkpoint()).toEqual(before)
  })

  it("saves the outgoing qualified snapshot before switch and unmount, never incoming rows", async () => {
    seed("A"); seed("B")
    const surface = mount()
    await waitFor(() => expect(draft().value).toBe("Draft A"))
    fireEvent.change(draft(), { target: { value: "Outgoing A" } })
    changeWorkspace("B")
    await waitFor(() => expect(draft().value).toBe("Draft B"))
    expect(checkpoint("A").checkpoint?.draft).toBe("Outgoing A")
    expect(checkpoint("A").messages.map(row => row.message)).toEqual(["Verified A"])
    fireEvent.change(draft(), { target: { value: "Outgoing B" } })
    surface.unmount()
    expect(checkpoint("B").checkpoint?.draft).toBe("Outgoing B")
    expect(checkpoint("B").messages.map(row => row.message)).toEqual(["Verified B"])
  })

  it("never adopts a legacy base record when the exact qualified reference is missing", async () => {
    seed()
    const legacy = saved()
    delete legacy.checkpoint
    useWorkspaceStore.setState({ workspaceChatSessions: { "workspace-A": legacy } })
    mount()
    await waitFor(() => expect(screen.getByLabelText("Restoring").textContent).toBe("false"))
    expect(mocks.capture).not.toHaveBeenCalled()
    expect(useWorkspaceStore.getState().workspaceChatSessions["workspace-A"]).toEqual(legacy)
    expect(transcript()).toBe("")
  })

  it("restores an empty draft-only record without history authority or IDs", async () => {
    seed()
    const empty = saved()
    Object.assign(empty, { messages: [], history: [], historyId: null, serverChatId: null })
    empty.checkpoint!.historySelectionReference = null
    useWorkspaceStore.setState({ workspaceChatSessions: { "workspace-A::reference-A": empty } })
    mount()
    await waitFor(() => expect(draft().value).toBe("Draft A"))
    expect(mocks.capture).not.toHaveBeenCalled()
    expect(controller?.getReference()).toBeNull()
    expect(checkpoint().serverChatId).toBeNull()
  })

  it("a new live H1 selection during deferred restoration wins over the checkpoint", async () => {
    seed()
    const pending = deferred<ReturnType<typeof capture>>()
    mocks.capture.mockReturnValueOnce(pending.promise)
    mount()
    await waitFor(() => expect(mocks.capture).toHaveBeenCalled())
    const [owner, view] = mocks.capture.mock.calls[0]
    await act(async () => { await controller!.loadConversation({ serverChatId: "chat-B", scope: { type: "workspace", workspaceId: "workspace-A" } }) })
    const selected = controller!.getReference()
    await act(async () => { pending.resolve(capture(owner, view)) })
    expect(transcript()).toBe("Verified B")
    expect(controller!.getReference()).toEqual(selected)
    expect(draft().value).toBe("")
  })

  it("CP-F1 a winning authorized H1 selection settles its lease and persists the outgoing draft", async () => {
    seed()
    const pending = deferred<ReturnType<typeof capture>>()
    mocks.capture.mockReturnValueOnce(pending.promise)
    const surface = mount()
    await waitFor(() => expect(mocks.capture).toHaveBeenCalled())
    const [owner, view] = mocks.capture.mock.calls[0]
    await act(async () => {
      await controller!.loadConversation({ serverChatId: "chat-B", scope: { type: "workspace", workspaceId: "workspace-A" } })
      useStoreMessageOption.getState().setServerChatId("chat-B")
    })
    await act(async () => { pending.resolve(capture(owner, view)) })
    await waitFor(() => expect(screen.getByLabelText("Restoring").textContent).toBe("false"))
    expect(transcript()).toBe("Verified B")
    expect(draft().value).toBe("")
    fireEvent.change(draft(), { target: { value: "Winning B draft" } })
    expect.soft(checkpointFence!()()).toBe(true)
    expect.soft(checkpoint().checkpoint?.draft).toBe("Winning B draft")
    expect.soft(checkpoint().checkpoint?.historySelectionReference?.conversation_id).toBe("chat-B")
    changeWorkspace("C")
    surface.unmount()
    expect(checkpoint().checkpoint?.draft).toBe("Winning B draft")
    expect(checkpoint().messages.map(row => row.message)).toEqual(["Verified B"])
  })

  it("CP-F1 a no-race restoration retains a usable lease and persists edits", async () => {
    seed()
    mount()
    await waitFor(() => expect(draft().value).toBe("Draft A"))
    expect(checkpointFence!()()).toBe(true)
    fireEvent.change(draft(), { target: { value: "Control edit" } })
    expect(checkpoint().checkpoint?.draft).toBe("Control edit")
  })

  it("CP-F1 a winning capture is usable before the superseded checkpoint read finishes", async () => {
    seed()
    const pending = deferred<ReturnType<typeof capture>>()
    mocks.capture.mockReturnValueOnce(pending.promise)
    mount()
    await waitFor(() => expect(mocks.capture).toHaveBeenCalled())
    const [owner, view] = mocks.capture.mock.calls[0]
    await act(async () => {
      await controller!.loadConversation({ serverChatId: "chat-B", scope: { type: "workspace", workspaceId: "workspace-A" } })
      useStoreMessageOption.getState().setServerChatId("chat-B")
    })
    expect(checkpointFence!()()).toBe(true)
    expect(screen.getByLabelText("Restoring").textContent).toBe("false")
    fireEvent.change(draft(), { target: { value: "B while A pending" } })
    await act(async () => { pending.resolve(capture(owner, view)) })
    expect(draft().value).toBe("B while A pending")
    expect(checkpoint().checkpoint?.draft).toBe("B while A pending")
    expect(checkpoint().checkpoint?.historySelectionReference?.conversation_id).toBe("chat-B")
  })

  it.each(["account", "workspace", "mounted ID", "expired lease"])("CP-F1 a winning %s mismatch cannot settle or overwrite the checkpoint", async mismatch => {
    seed()
    const before = structuredClone(checkpoint())
    const pending = deferred<ReturnType<typeof capture>>()
    mocks.capture.mockReturnValueOnce(pending.promise)
    mount()
    await waitFor(() => expect(mocks.capture).toHaveBeenCalled())
    const [owner, view] = mocks.capture.mock.calls[0]
    if (mismatch === "account") mocks.scope.mockResolvedValue(scope(2))
    let winningLease = true
    await act(async () => {
      await controller!.loadConversation({ serverChatId: "chat-B",
        scope: { type: "workspace", workspaceId: mismatch === "workspace" ? "workspace-B" : "workspace-A" },
        isCurrent: () => winningLease })
      if (mismatch === "expired lease") winningLease = false
      if (mismatch !== "mounted ID") useStoreMessageOption.getState().setServerChatId("chat-B")
    })
    await act(async () => { pending.resolve(capture(owner, view)) })
    fireEvent.change(draft(), { target: { value: "Not qualified" } })
    expect(checkpointFence!()()).toBe(false)
    expect(checkpoint()).toEqual(before)
  })

  it("CP-F1 Research dispatch remains eligible after an authorized selection supersedes restoration", async () => {
    seed()
    useWorkspaceStore.setState({ sources: [], selectedSourceIds: [], selectedSourceFolderIds: [], sourceFolders: [], sourceFolderMemberships: [] })
    useStoreMessageOption.setState({ selectedModel: "test-model", chatMode: "normal" })
    const pending = deferred<ReturnType<typeof capture>>()
    mocks.capture.mockReturnValueOnce(pending.promise)
    mount(true, undefined, true)
    await waitFor(() => expect(mocks.capture).toHaveBeenCalled())
    const [owner, view] = mocks.capture.mock.calls[0]
    await act(async () => {
      await controller!.loadConversation({ serverChatId: "chat-B", scope: { type: "workspace", workspaceId: "workspace-A" } })
      useStoreMessageOption.getState().setServerChatId("chat-B")
    })
    await act(async () => { pending.resolve(capture(owner, view)) })
    const input = screen.getByRole("textbox", { name: "Chat message" })
    fireEvent.change(input, { target: { value: "Explicit question for B" } })
    fireEvent.click(screen.getByRole("button", { name: "Send" }))
    await waitFor(() => expect(mocks.submit).toHaveBeenCalledTimes(1))
    expect(mocks.submit.mock.calls[0][0].message).toBe("Explicit question for B")
    expect(mocks.submit.mock.calls[0][0].requestOverrides.tldwTurn.user_message_id)
      .toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i)
    expect(useStoreMessageOption.getState().serverChatId).toBe("chat-B")
  })

  it("typing during scope resolution, including draft ABA, cannot be replaced", async () => {
    seed()
    const pending = deferred<ReturnType<typeof scope>>()
    mocks.scope.mockReturnValueOnce(pending.promise)
    mount()
    fireEvent.change(draft(), { target: { value: "Typed A" } })
    fireEvent.change(draft(), { target: { value: "Typed B" } })
    fireEvent.change(draft(), { target: { value: "Typed A" } })
    await act(async () => { pending.resolve(scope()) })
    await waitFor(() => expect(transcript()).toBe("Verified A"))
    expect(draft().value).toBe("Typed A")
  })

  it.each(["owner", "workspace", "reference"])("retains a rejected %s checkpoint without adoption or overwrite", async field => {
    seed()
    const invalid = saved()
    if (field === "owner") invalid.checkpoint!.ownerKey = "other-owner"
    if (field === "workspace") invalid.checkpoint!.workspaceId = "other-workspace"
    if (field === "reference") invalid.checkpoint!.referenceId = "other-reference"
    useWorkspaceStore.setState({ workspaceChatSessions: { "workspace-A::reference-A": invalid } })
    const surface = mount()
    await waitFor(() => expect(screen.getByLabelText("Restoring").textContent).toBe("false"))
    fireEvent.change(draft(), { target: { value: "Do not adopt" } })
    surface.unmount()
    expect(checkpoint()).toEqual(invalid)
    expect(mocks.capture).not.toHaveBeenCalled()
  })

  it("does not use an unresolved client principal as scope authority", async () => {
    seed()
    mocks.scope.mockResolvedValue({ ...scope(), clientPrincipalVerified: false })
    mount()
    await waitFor(() => expect(screen.getByLabelText("Restoring").textContent).toBe("false"))
    expect(mocks.capture).not.toHaveBeenCalled()
    expect(checkpoint().checkpoint?.draft).toBe("Draft A")
  })

  it("unmount during scope resolution cancels without changing the checkpoint", async () => {
    seed()
    const pending = deferred<ReturnType<typeof scope>>()
    mocks.scope.mockReturnValueOnce(pending.promise)
    const surface = mount()
    surface.unmount()
    await act(async () => { pending.resolve(scope()) })
    expect(mocks.capture).not.toHaveBeenCalled()
    expect(checkpoint().checkpoint?.draft).toBe("Draft A")
  })

  it("a current captured selection during scope resolution wins and is checkpointed", async () => {
    seed()
    const pending = deferred<ReturnType<typeof scope>>()
    mocks.scope.mockReturnValueOnce(pending.promise).mockResolvedValue(scope())
    mount()
    await act(async () => {
      await controller!.loadConversation({ serverChatId: "chat-B", scope: { type: "workspace", workspaceId: "workspace-A" } })
      useStoreMessageOption.getState().setServerChatId("chat-B")
    })
    fireEvent.change(draft(), { target: { value: "Current capture draft" } })
    await act(async () => { pending.resolve(scope()) })
    expect(transcript()).toBe("Verified B")
    expect(draft().value).toBe("Current capture draft")
    expect(checkpoint().checkpoint?.historySelectionReference?.conversation_id).toBe("chat-B")
  })

  it("does not publish a rejected scope lookup over a newer captured selection in the same workspace", async () => {
    seed()
    const original = structuredClone(checkpoint())
    const pending = deferred<ReturnType<typeof scope>>()
    mocks.scope.mockReturnValueOnce(pending.promise).mockResolvedValue(scope())
    mount()
    await act(async () => {
      await controller!.loadConversation({ serverChatId: "chat-B", scope: { type: "workspace", workspaceId: "workspace-A" } })
      useStoreMessageOption.getState().setServerChatId("chat-B")
    })
    fireEvent.change(draft(), { target: { value: "Current capture draft" } })
    await act(async () => { pending.reject(new Error("Superseded scope failure")) })
    expect(screen.getByLabelText("Restoration error")).toBeEmptyDOMElement()
    expect(transcript()).toBe("Verified B")
    expect(draft().value).toBe("Current capture draft")
    expect(checkpoint()).toEqual(original)
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it.each(["scope", "read"])("retires a %s failure after a newer capture without granting checkpoint save authority", async failure => {
    seed()
    const original = structuredClone(checkpoint())
    if (failure === "scope") mocks.scope.mockRejectedValueOnce(new Error("Unavailable"))
    else vi.spyOn(useWorkspaceStore.getState(), "getWorkspaceChatSession").mockImplementationOnce(() => { throw new Error("storage failed") })
    const surface = mount()
    await waitFor(() => expect(screen.getByLabelText("Restoration error")).toHaveTextContent("Workspace chat restoration failed"))
    fireEvent.change(draft(), { target: { value: "Retained unsent draft" } })
    await act(async () => {
      await controller!.loadConversation({ serverChatId: "chat-B", scope: { type: "workspace", workspaceId: "workspace-A" } })
      useStoreMessageOption.getState().setServerChatId("chat-B")
    })
    expect(screen.getByLabelText("Restoration error")).toBeEmptyDOMElement()
    expect(transcript()).toBe("Verified B")
    expect(draft().value).toBe("Retained unsent draft")
    expect(checkpointFence!()()).toBe(false)
    surface.unmount()
    expect(checkpoint()).toEqual(original)
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it.each(["before", "after"])("mounted workspace panel remains usable when capture occurs %s checkpoint rejection", async ordering => {
    seed()
    const original = structuredClone(checkpoint())
    const pending = deferred<ReturnType<typeof scope>>()
    mocks.scope.mockReturnValueOnce(pending.promise).mockResolvedValue(scope())
    useStoreMessageOption.setState({ selectedModel: "test-model", serverChatLoadState: "idle" })
    const runtime = vi.fn()
    const surface = render(<HistorySelectionProvider onCapture={result => {
      useStoreMessageOption.getState().setMessages(result.selected_content.map(row => ({ id: row.id, name: "Assistant", isBot: true, message: row.message, sources: [] })))
      useStoreMessageOption.getState().setHistory(result.selected_content.map(row => ({ role: "assistant", content: row.message })))
    }}><SelectionObserver /><WorkspaceChatPanel workspaceId="workspace-A" workspaceReady backendAvailable
      stagedSources={[]} onClearStagedSources={() => {}} onRuntimeStateChange={runtime} /></HistorySelectionProvider>)
    const composer = screen.getByRole("textbox", { name: "Chat workspace message" })
    fireEvent.change(composer, { target: { value: "Retained unsent draft" } })
    if (ordering === "after") {
      await act(async () => { pending.reject(new Error("Unavailable")) })
      expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled()
      expect(runtime.mock.lastCall?.[0].historyLoadError).toBe("Workspace chat restoration failed")
    }
    await act(async () => {
      await controller!.loadConversation({ serverChatId: "chat-B", scope: { type: "workspace", workspaceId: "workspace-A" } })
      useStoreMessageOption.getState().setServerChatId("chat-B")
      useStoreMessageOption.getState().setServerChatLoadState("loaded")
    })
    if (ordering === "before") await act(async () => { pending.reject(new Error("Unavailable")) })
    expect(screen.getByRole("button", { name: "Send message" })).toBeEnabled()
    expect(runtime.mock.lastCall?.[0].historyLoadError).toBeNull()
    expect(composer).toHaveValue("Retained unsent draft")
    expect(useStoreMessageOption.getState().messages.map(row => row.message)).toEqual(["Verified B"])
    surface.unmount()
    expect(checkpoint()).toEqual(original)
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it.each(["scope", "read"])("keeps a %s restoration failure blocking Send until replacement capture succeeds", async failure => {
    seed()
    const original = structuredClone(checkpoint())
    useStoreMessageOption.setState({ selectedModel: "test-model", serverChatLoadState: "idle" })
    const onCapture: NonNullable<React.ComponentProps<typeof HistorySelectionProvider>["onCapture"]> = result => {
      useStoreMessageOption.getState().setMessages(result.selected_content.map(row => ({ id: row.id, name: "Assistant", isBot: true, message: row.message, sources: [] })))
      useStoreMessageOption.getState().setHistory(result.selected_content.map(row => ({ role: "assistant", content: row.message })))
    }
    const surface = render(<HistorySelectionProvider onCapture={onCapture}><SelectionObserver /></HistorySelectionProvider>)
    await act(async () => {
      await controller!.loadConversation({ serverChatId: "chat-B", scope: { type: "workspace", workspaceId: "workspace-A" } })
      useStoreMessageOption.getState().setServerChatId("chat-B")
      useStoreMessageOption.getState().setServerChatLoadState("loaded")
    })
    if (failure === "scope") mocks.scope.mockRejectedValueOnce(new Error("Unavailable"))
    else vi.spyOn(useWorkspaceStore.getState(), "getWorkspaceChatSession").mockImplementationOnce(() => { throw new Error("storage failed") })
    const runtime = vi.fn()
    surface.rerender(<HistorySelectionProvider onCapture={onCapture}><SelectionObserver />
      <WorkspaceChatPanel workspaceId="workspace-A" workspaceReady backendAvailable stagedSources={[]}
        onClearStagedSources={() => {}} onRuntimeStateChange={runtime} /></HistorySelectionProvider>)
    await waitFor(() => expect(runtime.mock.lastCall?.[0].historyLoadError).toBe("Workspace chat restoration failed"))
    const pending = deferred<ReturnType<typeof scope>>()
    mocks.scope.mockReturnValueOnce(pending.promise)
    let loading!: Promise<boolean>
    act(() => { loading = controller!.loadConversation({ serverChatId: "chat-C", scope: { type: "workspace", workspaceId: "workspace-A" } }) })
    await waitFor(() => expect(mocks.scope.mock.results.at(-1)?.value).toBe(pending.promise))
    fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), { target: { value: "Retained while loading" } })
    expect(screen.getByRole("button", { name: "Send message" })).toBeDisabled()
    expect(runtime.mock.lastCall?.[0].historyLoadError).toBe("Workspace chat restoration failed")
    await act(async () => {
      pending.resolve(scope())
      await loading
      useStoreMessageOption.getState().setServerChatId("chat-C")
      useStoreMessageOption.getState().setServerChatLoadState("loaded")
    })
    expect(screen.getByRole("button", { name: "Send message" })).toBeEnabled()
    expect(runtime.mock.lastCall?.[0].historyLoadError).toBeNull()
    expect(screen.getByRole("textbox", { name: "Chat workspace message" })).toHaveValue("Retained while loading")
    surface.unmount()
    expect(checkpoint()).toEqual(original)
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it("keeps the legacy Research session path only when no controller is mounted", async () => {
    const legacy = saved()
    delete legacy.checkpoint
    useWorkspaceStore.setState({ workspaceChatSessions: { "workspace-A::reference-A": legacy } })
    const surface = mount(false, "workspace-A::reference-A")
    await waitFor(() => expect(transcript()).toBe("UNTRUSTED CACHE"))
    act(() => { useStoreMessageOption.getState().setMessages([{ id: "legacy-new", name: "You", isBot: false, message: "Legacy edit", sources: [] }]) })
    surface.unmount()
    expect(checkpoint().checkpoint).toBeUndefined()
    expect(checkpoint().messages[0].message).toBe("Legacy edit")
  })

  it("the legacy Research path never reads or overwrites a qualified record", async () => {
    seed()
    const before = structuredClone(checkpoint())
    const surface = mount(false, "workspace-A::reference-A")
    act(() => { useStoreMessageOption.getState().setMessages([{ id: "legacy", name: "You", isBot: false, message: "Unowned edit", sources: [] }]) })
    expect(transcript()).not.toBe("UNTRUSTED CACHE")
    surface.unmount()
    expect(checkpoint()).toEqual(before)
  })

  it("H1 scope drift cannot publish rows before the hook checks its load receipt", async () => {
    seed()
    mocks.scope.mockResolvedValueOnce(scope()).mockResolvedValue(scope(2, "http://other.test"))
    mount()
    await waitFor(() => expect(screen.getByLabelText("Restoring").textContent).toBe("false"))
    expect(transcript()).toBe("")
    expect(draft().value).toBe("")
    expect(useStoreMessageOption.getState().serverChatId).toBeNull()
    expect(checkpoint().checkpoint?.draft).toBe("Draft A")
  })

  it("legacy clear and undo are persisted by the real reactive handoff without explicit writes", async () => {
    const legacy = saved()
    delete legacy.checkpoint
    useWorkspaceStore.setState({ workspaceChatSessions: { "workspace-A::reference-A": legacy } })
    mount(false, "workspace-A::reference-A")
    await waitFor(() => expect(transcript()).toBe("UNTRUSTED CACHE"))
    act(() => { useStoreMessageOption.setState({ messages: [], history: [], historyId: null, serverChatId: null }) })
    expect(checkpoint()).toEqual({ messages: [], history: [], historyId: null, serverChatId: null })
    act(() => { useStoreMessageOption.setState(legacy) })
    expect(checkpoint()).toEqual(legacy)
  })

  it("legacy workspace switching saves the outgoing capture, not incoming rows", async () => {
    const legacyA = saved("A")
    const legacyB = saved("B")
    delete legacyA.checkpoint
    delete legacyB.checkpoint
    useWorkspaceStore.setState({ workspaceChatSessions: { "workspace-A::reference-A": legacyA, "workspace-B::reference-B": legacyB } })
    const surface = mount(false, "workspace-A::reference-A")
    await waitFor(() => expect(transcript()).toBe("UNTRUSTED CACHE"))
    act(() => { useStoreMessageOption.getState().setMessages([{ id: "legacy-A", name: "You", isBot: false, message: "Outgoing legacy A", sources: [] }]) })
    changeWorkspace("B")
    surface.rerender(<Surface legacySessionKey="workspace-B::reference-B" />)
    expect(checkpoint("A").messages[0].message).toBe("Outgoing legacy A")
    expect(checkpoint("B")).toEqual(legacyB)
  })

  it("a handoff fence survives its own fresh H1 load, but not a workspace switch", async () => {
    mount()
    await waitFor(() => expect(screen.getByLabelText("Restoring").textContent).toBe("false"))
    const current = checkpointFence!()
    expect(current()).toBe(true)
    await act(async () => {
      await controller!.loadConversation({ serverChatId: "chat-A", scope: { type: "workspace", workspaceId: "workspace-A" } })
      useStoreMessageOption.getState().setServerChatId("chat-A")
    })
    expect(current()).toBe(true)
    changeWorkspace("B")
    expect(current()).toBe(false)
  })

  it("an account change retires a handoff fence even when the account returns", async () => {
    mount()
    await waitFor(() => expect(screen.getByLabelText("Restoring").textContent).toBe("false"))
    const current = checkpointFence!()
    act(() => { mocks.listeners.forEach(listener => listener()) })
    await waitFor(() => expect(screen.getByLabelText("Restoring").textContent).toBe("false"))
    expect(current()).toBe(false)
  })
})
