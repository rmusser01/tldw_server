// @vitest-environment jsdom
import React from "react"
import { act, fireEvent, render, renderHook, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter, Route, Routes, useLocation, useNavigate } from "react-router-dom"

import { useClearChat } from "../useClearChat"
import { HistorySelectionProvider } from "../useHistorySelection"
import { useWorkspaceChatCheckpoint } from "../useWorkspaceChatCheckpoint"
import { useStoreMessageOption } from "@/store/option"
import { useWorkspaceStore } from "@/store/workspace"
import { serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror"
import { resolveHistorySelection } from "@/utils/history-selection"
import type { HistorySelectionSnapshotV1, HistoryViewSelectionV1 } from "@/types/history-selection"
import {
  SETTINGS_NAVIGATION_REQUEST_EVENT,
  type SettingsNavigationRequestDetail
} from "@/utils/settings-return"

const navigateMock = vi.hoisted(() => vi.fn())
const destroyAllMock = vi.hoisted(() => vi.fn())
const cleanupOverlaysMock = vi.hoisted(() => vi.fn())
const updatePageTitleMock = vi.hoisted(() => vi.fn())
const focusTextAreaMock = vi.hoisted(() => vi.fn())
const resetModelSettingsMock = vi.hoisted(() => vi.fn())
const clearSessionMock = vi.hoisted(() => vi.fn())
const optionStoreSetStateMock = vi.hoisted(() => vi.fn())
const integration = vi.hoisted(() => ({ active: false, bookmarks: new Map<string, unknown>(), capture: vi.fn() }))
const historySelectionResetMock = vi.hoisted(() => vi.fn())
const historySelectionContextMock = vi.hoisted(() => vi.fn())

const baseState = vi.hoisted(() => ({
  setMessages: vi.fn(),
  setHistory: vi.fn(),
  setHistoryId: vi.fn(),
  setIsFirstMessage: vi.fn(),
  setIsLoading: vi.fn(),
  setIsProcessing: vi.fn(),
  setStreaming: vi.fn()
}))

const optionState = vi.hoisted(() => ({
  setServerChatId: vi.fn(),
  setServerChatVersion: vi.fn(),
  setContextFiles: vi.fn(),
  setDocumentContext: vi.fn(),
  setUploadedFiles: vi.fn(),
  setFileRetrievalEnabled: vi.fn(),
  setActionInfo: vi.fn(),
  setRagMediaIds: vi.fn(),
  setRagSearchMode: vi.fn(),
  setRagTopK: vi.fn(),
  setRagEnableGeneration: vi.fn(),
  setRagEnableCitations: vi.fn(),
  setRagSources: vi.fn(),
  clearQueuedMessages: vi.fn(),
  setCompareMode: vi.fn(),
  setCompareSelectedModels: vi.fn(),
  clearReplyTarget: vi.fn(),
  setWebSearch: vi.fn()
}))

vi.mock("react-router-dom", async (importActual) => {
  const actual = await importActual<typeof import("react-router-dom")>()
  return { ...actual, useNavigate: () => {
    const navigate = actual.useNavigate()
    return (...args: Parameters<typeof navigate>) => {
      navigateMock(...args)
      return navigate(...args)
    }
  } }
})

vi.mock("antd", () => ({
  Modal: { destroyAll: destroyAllMock }
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: () => [false]
}))

vi.mock("@/hooks/chat/useChatBaseState", () => ({
  useChatBaseState: (store: { getState?: () => unknown }) =>
    integration.active && store.getState ? store.getState() : baseState
}))

vi.mock("@/hooks/chat/useHistorySelection", async (importActual) => {
  const actual = await importActual<typeof import("@/hooks/chat/useHistorySelection")>()
  return { ...actual, useHistorySelectionContext: () => {
    const controller = actual.useHistorySelectionContext()
    return integration.active ? controller : historySelectionContextMock()
  } }
})

vi.mock("@/hooks/utils/messageHelpers", () => ({
  focusTextArea: focusTextAreaMock
}))

vi.mock("@/store/option", async (importActual) => {
  const actual = await importActual<typeof import("@/store/option")>()
  const useStoreMessageOption = Object.assign(
    (selector?: (state: ReturnType<typeof actual.useStoreMessageOption.getState>) => unknown, equality?: (a: unknown, b: unknown) => boolean) => {
      const selected = actual.useStoreMessageOption(selector, equality)
      return integration.active ? selected : selector?.(optionState as unknown as ReturnType<typeof actual.useStoreMessageOption.getState>)
    },
    {
      setState: (...args: Parameters<typeof actual.useStoreMessageOption.setState>) => integration.active
        ? actual.useStoreMessageOption.setState(...args) : optionStoreSetStateMock(...args),
      getState: () => integration.active ? actual.useStoreMessageOption.getState() : { serverChatId: "saved", historyId: "local" }
    }
  )
  return { useStoreMessageOption }
})

vi.mock("@/store", () => ({
  useStoreMessage: vi.fn()
}))

vi.mock("@/store/playground-session", async (importActual) => {
  const actual = await importActual<typeof import("@/store/playground-session")>()
  return { usePlaygroundSessionStore: {
    getState: () => integration.active ? actual.usePlaygroundSessionStore.getState() : { clearSession: clearSessionMock, restoreRevision: 0 }
  } }
})

vi.mock("@/store/model", () => ({
  useStoreChatModelSettings: () => ({ reset: resetModelSettingsMock })
}))

vi.mock("@/utils/cleanup-ant-overlays", () => ({
  cleanupAntOverlays: cleanupOverlaysMock
}))

vi.mock("@/utils/update-page-title", () => ({
  updatePageTitle: updatePageTitleMock
}))

const bookmarkKey = (scope: { profile_id: string; client_session_id: string }, owner: { owner_key: string; conversation_id: string }) =>
  JSON.stringify([scope.profile_id, scope.client_session_id, owner.owner_key, owner.conversation_id])
vi.mock("@/db/dexie/history-selection", () => ({
  ensureLocalProfileId: async () => "profile",
  loadHistoryBookmark: async (scope: Parameters<typeof bookmarkKey>[0], owner: Parameters<typeof bookmarkKey>[1]) => integration.bookmarks.get(bookmarkKey(scope, owner)) ?? null,
  saveHistoryBookmark: async (scope: Parameters<typeof bookmarkKey>[0], view: Parameters<typeof bookmarkKey>[1]) => integration.bookmarks.set(bookmarkKey(scope, view), { ...scope, view }),
  loadHistoryTurnRecoveries: async () => [], dismissHistoryTurnRecovery: async () => {}, saveHistoryTurnRecovery: async () => {}
}))
vi.mock("@/db/dexie/fork-operations", () => ({ findForkCandidate: async () => null, loadForkOperations: async () => [], allowNewForkOperation: async () => {} }))
vi.mock("@/db/dexie/chat", () => ({ PageAssistDatabase: class { getHistoryInfo = async () => null } }))
vi.mock("@/services/chat-history-selection", () => ({
  captureHistorySnapshot: (...args: unknown[]) => integration.capture(...args),
  readNativeForkSettings: async () => null, updateNativeForkSettings: async () => null, confirmLegacyHistoryProjection: async () => {}
}))
vi.mock("@/services/service-prompts", () => ({
  resolveServicePromptScope: async () => scope(), subscribeToServicePromptConfigChanges: () => () => {}
}))
const scope = () => ({
  config: { serverUrl: "http://owner.test", authMode: "multi-user" as const, authSource: "manual" as const },
  userId: 1, scopeKey: "account-1", clientPrincipalVerified: true
})
const oldKey = "workspace-A::reference-A"
function seedExistingCheckpoint() {
  const reference = { profile_id: "profile", client_session_id: "saved-view", owner_key: "native-owner", conversation_id: "saved" }
  useWorkspaceStore.getState().saveWorkspaceChatSession(oldKey, {
    messages: [{ id: "untrusted-cache", name: "Assistant", isBot: true, message: "UNTRUSTED CACHE", sources: [] }],
    history: [{ role: "assistant", content: "UNTRUSTED CACHE" }], historyId: null, serverChatId: "saved",
    checkpoint: { version: 1, ownerKey: serverChatMirrorOwnerKey({ requestScope: scope() }),
      workspaceId: "workspace-A", referenceId: "reference-A", historySelectionReference: reference, draft: "Old workspace draft" }
  })
  integration.bookmarks.set(bookmarkKey(reference, reference), { ...reference, view: {
    ...reference, view_session_id: "saved-view", selection_revision: 1,
    interpretation: { kind: "parent_graph_v1" }, cursor: { kind: "after_message", message_id: "verified-native" }
  } })
}
function RouterWrapper({ children }: { children: React.ReactNode }) {
  return <MemoryRouter initialEntries={[window.location.pathname]}>{children}</MemoryRouter>
}
function CheckpointSurface() {
  const clear = useClearChat()
  const workspaceId = useWorkspaceStore(state => state.workspaceId)
  const hydrated = useWorkspaceStore(state => state.storeHydrated)
  const chat = useStoreMessageOption()
  const [draft, setDraft] = React.useState("")
  const checkpoint = useWorkspaceChatCheckpoint({ workspaceId, workspaceReady: hydrated, draft, setDraft,
    chat: { ...chat, stopStreamingRequest: () => {} } })
  return <>
    <button onClick={() => clear()}>New Chat command</button>
    <input aria-label="Workspace draft" value={draft} onChange={event => checkpoint.setDraft(event.target.value)} />
    <output aria-label="Workspace transcript">{chat.messages.map(row => row.message).join(" / ")}</output>
    <output aria-label="Restoring checkpoint">{String(checkpoint.restoring)}</output>
  </>
}
function RouteHarness({ origin }: { origin: string }) {
  const navigate = useNavigate()
  const location = useLocation()
  React.useLayoutEffect(() => { window.history.replaceState({}, "", location.pathname) }, [location.pathname])
  return <Routes>
    <Route path={origin} element={<HistorySelectionProvider key={origin} onCapture={result => {
      const chat = useStoreMessageOption.getState()
      chat.setMessages(result.selected_content.map(row => ({ id: row.id, name: "Assistant", isBot: true, message: row.message, sources: [] })))
      chat.setHistory(result.selected_content.map(row => ({ role: "assistant", content: row.message })))
    }}><CheckpointSurface /></HistorySelectionProvider>} />
    <Route path="/chat" element={<button onClick={() => navigate(origin)}>Return to workspace</button>} />
  </Routes>
}

describe("useClearChat settings navigation", () => {
  beforeEach(async () => {
    vi.clearAllMocks()
    navigateMock.mockReset()
    integration.active = true
    useStoreMessageOption.setState({ messages: [], history: [], historyId: null, serverChatId: null,
      temporaryChat: false, streaming: false, isProcessing: false, isLoading: false })
    const { usePlaygroundSessionStore } = await vi.importActual<typeof import("@/store/playground-session")>("@/store/playground-session")
    usePlaygroundSessionStore.getState().clearSession()
    integration.active = false
    integration.bookmarks.clear()
    integration.capture.mockReset().mockImplementation((owner: { conversation_id: string }, view: HistoryViewSelectionV1) => {
      const snapshot: HistorySelectionSnapshotV1 = {
        version: 1, owner_key: "native-owner", conversation_id: owner.conversation_id, source_digest: "source", storage_context_digest: "storage",
        fences: { conversation: "1", history: "1", settings: "1" }, interpretation_status: { kind: "parent_graph_v1" },
        nodes: [{ id: "verified-native", revision: "1", parent_id: null, role: "assistant", settled: true, preview: "Verified native conversation" }]
      }
      const bound = { ...view, owner_key: snapshot.owner_key }
      const selected = resolveHistorySelection(snapshot, bound, "send", "")
      return { status: "captured", snapshot, view: bound, rows: selected.status === "ready" ? selected.rows : [],
        selected_content: [{ id: "verified-native", revision: "1", message: "Verified native conversation", images: [] }], purpose: "send", storage_context_digest: "storage" }
    })
    localStorage.setItem("tldw:feature-rollout:workspace_indexeddb_offload_v1:enabled", "0")
    useWorkspaceStore.setState({ workspaceId: "workspace-A", workspaceChatReferenceId: "reference-A", storeHydrated: true, workspaceChatSessions: {} })
    historySelectionContextMock.mockReturnValue({ reset: historySelectionResetMock })
    window.history.replaceState({}, "", "/settings/prompt")
  })

  it("does nothing when the mounted settings editor declines navigation", () => {
    const declineNavigation = vi.fn((event: Event) => event.preventDefault())
    window.addEventListener(
      SETTINGS_NAVIGATION_REQUEST_EVENT,
      declineNavigation,
      { once: true }
    )
    const { result } = renderHook(() => useClearChat(), { wrapper: RouterWrapper })

    let accepted: boolean | undefined
    act(() => { accepted = result.current() })
    expect(accepted).toBe(false)

    expect(declineNavigation).toHaveBeenCalledOnce()
    expect((declineNavigation.mock.calls[0][0] as CustomEvent<
      SettingsNavigationRequestDetail
    >).detail).toEqual({ destination: "/chat" })
    expect(navigateMock).not.toHaveBeenCalled()
    expect(destroyAllMock).not.toHaveBeenCalled()
    expect(cleanupOverlaysMock).not.toHaveBeenCalled()
    expect(baseState.setMessages).not.toHaveBeenCalled()
    expect(optionState.setServerChatId).not.toHaveBeenCalled()
    expect(resetModelSettingsMock).not.toHaveBeenCalled()
    expect(updatePageTitleMock).not.toHaveBeenCalled()
    expect(focusTextAreaMock).not.toHaveBeenCalled()
    expect(clearSessionMock).not.toHaveBeenCalled()
    expect(historySelectionResetMock).not.toHaveBeenCalled()
  })

  it("navigates and resets once when navigation is allowed", () => {
    const { result } = renderHook(() => useClearChat(), { wrapper: RouterWrapper })

    act(() => expect(result.current()).toBe(true))

    expect(navigateMock).toHaveBeenCalledOnce()
    expect(navigateMock).toHaveBeenCalledWith("/chat")
    expect(destroyAllMock).toHaveBeenCalledOnce()
    expect(cleanupOverlaysMock).toHaveBeenCalledOnce()
    expect(optionState.setServerChatId).toHaveBeenCalledOnce()
    expect(resetModelSettingsMock).toHaveBeenCalledOnce()
    expect(updatePageTitleMock).toHaveBeenCalledOnce()
    expect(focusTextAreaMock).toHaveBeenCalledOnce()
    expect(clearSessionMock).toHaveBeenCalledOnce()
    expect(useWorkspaceStore.getState().workspaceChatReferenceId).toBe("reference-A")
    expect(historySelectionResetMock).toHaveBeenCalledOnce()
    expect(historySelectionResetMock.mock.invocationCallOrder[0]).toBeLessThan(
      navigateMock.mock.invocationCallOrder[0]
    )
  })

  it.each(["/chat-workspace", "/research-workspace"])("real New Chat from %s cannot resurrect the old qualified checkpoint on return", async origin => {
    integration.active = true
    seedExistingCheckpoint()
    render(<MemoryRouter initialEntries={[origin]}><RouteHarness origin={origin} /></MemoryRouter>)
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Workspace draft" })).toHaveValue("Old workspace draft"))
    expect(screen.getByLabelText("Workspace transcript")).toHaveTextContent("Verified native conversation")
    fireEvent.change(screen.getByRole("textbox", { name: "Workspace draft" }), { target: { value: "Outgoing edited draft" } })
    const outgoing = structuredClone(useWorkspaceStore.getState().workspaceChatSessions[oldKey])
    const captureCount = integration.capture.mock.calls.length
    fireEvent.click(screen.getByRole("button", { name: "New Chat command" }))
    expect(window.location.pathname).toBe("/chat")
    expect(useStoreMessageOption.getState().serverChatId).toBeNull()
    fireEvent.click(screen.getByRole("button", { name: "Return to workspace" }))
    await waitFor(() => expect(screen.getByLabelText("Restoring checkpoint")).toHaveTextContent("false"))
    expect(screen.getByRole("textbox", { name: "Workspace draft" })).toHaveValue("")
    expect(screen.getByLabelText("Workspace transcript")).toHaveTextContent(/^$/)
    const workspace = useWorkspaceStore.getState()
    expect(workspace.workspaceChatReferenceId).not.toBe("reference-A")
    expect(workspace.workspaceChatSessions[oldKey]).toEqual(outgoing)
    expect(useStoreMessageOption.getState().historyId).toBeNull()
    expect(useStoreMessageOption.getState().serverChatId).toBeNull()
    expect(integration.capture).toHaveBeenCalledTimes(captureCount)
  })

  it.each(["/chat-workspace", "/research-workspace"])("declined New Chat from %s preserves its reference and session", origin => {
    seedExistingCheckpoint()
    const before = structuredClone(useWorkspaceStore.getState().workspaceChatSessions[oldKey])
    window.history.replaceState({}, "", origin)
    window.addEventListener(SETTINGS_NAVIGATION_REQUEST_EVENT, event => event.preventDefault(), { once: true })
    const { result } = renderHook(() => useClearChat(), { wrapper: RouterWrapper })
    act(() => expect(result.current()).toBe(false))
    expect(useWorkspaceStore.getState().workspaceChatReferenceId).toBe("reference-A")
    expect(useWorkspaceStore.getState().workspaceChatSessions[oldKey]).toEqual(before)
    expect(navigateMock).not.toHaveBeenCalled()
    expect(baseState.setMessages).not.toHaveBeenCalled()
  })

  it.each(["/chat", "/knowledge", "/chat-workspace-other", "/research-workspace/shared"])("New Chat from %s retains the unrelated workspace reference and generic destination", origin => {
    window.history.replaceState({}, "", origin)
    const { result } = renderHook(() => useClearChat(), { wrapper: RouterWrapper })
    act(() => expect(result.current()).toBe(true))
    expect(navigateMock).toHaveBeenCalledWith("/chat")
    expect(useWorkspaceStore.getState().workspaceChatReferenceId).toBe("reference-A")
  })

  it("keeps the existing sidepanel New Chat destination", () => {
    window.history.replaceState({}, "", "/sidepanel.html")
    const { result } = renderHook(() => useClearChat(), { wrapper: RouterWrapper })
    act(() => expect(result.current()).toBe(true))
    expect(navigateMock).toHaveBeenCalledWith("/")
    expect(useWorkspaceStore.getState().workspaceChatReferenceId).toBe("reference-A")
  })

  it.each(["workspace", "reference"])("an accepted navigation that replaces the %s does not rotate the incoming reference", changed => {
    window.history.replaceState({}, "", "/chat-workspace")
    navigateMock.mockImplementationOnce(() => useWorkspaceStore.setState({
      workspaceId: changed === "workspace" ? "workspace-B" : "workspace-A", workspaceChatReferenceId: "reference-B"
    }))
    const { result } = renderHook(() => useClearChat(), { wrapper: RouterWrapper })
    act(() => expect(result.current()).toBe(true))
    expect(useWorkspaceStore.getState().workspaceChatReferenceId).toBe("reference-B")
  })

  it.each(["workspace", "reference"])("does not invent an existing %s for an uninitialized workspace", missing => {
    window.history.replaceState({}, "", "/research-workspace")
    useWorkspaceStore.setState({ workspaceId: missing === "workspace" ? "" : "workspace-A", workspaceChatReferenceId: missing === "reference" ? "" : "reference-A" })
    const { result } = renderHook(() => useClearChat(), { wrapper: RouterWrapper })
    act(() => expect(result.current()).toBe(true))
    expect(useWorkspaceStore.getState().workspaceChatReferenceId).toBe(missing === "reference" ? "" : "reference-A")
  })

  it("still clears standalone surfaces without a history selection provider", () => {
    historySelectionContextMock.mockReturnValue(null)
    const { result } = renderHook(() => useClearChat(), { wrapper: RouterWrapper })

    act(() => expect(result.current()).toBe(true))

    expect(navigateMock).toHaveBeenCalledWith("/chat")
    expect(clearSessionMock).toHaveBeenCalledOnce()
    expect(historySelectionResetMock).not.toHaveBeenCalled()
  })
})
