import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, expect, it, vi } from "vitest"
import { ConnectionPhase } from "@/types/connection"
import type { HistoryTurnRecovery } from "@/db/dexie/types"
import { buildUnknownResearchWorkspaceCapabilities, type ResearchWorkspaceCapabilitiesResponse } from "../research-workspace-capabilities"

const uuid = "00000000-0000-4000-8000-000000000001"
const recovery: HistoryTurnRecovery = {
  persistence: "server", logical_user_message_id: uuid, operation_id: "old-operation",
  owner_key: "native-A", conversation_id: "chat-A", created_at: 1,
  origin_view: { owner_key: "native-A", conversation_id: "chat-A", view_session_id: "old-view",
    selection_revision: 1, interpretation: { kind: "parent_graph_v1" }, cursor: { kind: "empty" } },
  selection_digest: "a".repeat(64), request_context_digest: "b".repeat(64),
  input_text: "Original recovered question", input_images: [], result_text: "", state: "unknown"
}
const state = vi.hoisted(() => ({ active: true, restoring: false, epoch: 0, referenceId: "reference-A", ownerKey: "native-A", onSubmit: vi.fn(), getMediaDetails: vi.fn(), captureHead: vi.fn(), captureScope: vi.fn(), regenerate: vi.fn() }))
vi.mock("@/utils/research-web-capture", async () => ({
  ...(await vi.importActual<typeof import("@/utils/research-web-capture")>("@/utils/research-web-capture")),
  assertWebCaptureHeadCurrent: (...args: unknown[]) => state.captureHead(...args)
}))
vi.mock("@/services/service-prompts", async () => ({
  ...(await vi.importActual<typeof import("@/services/service-prompts")>("@/services/service-prompts")),
  loadServicePromptSnapshot: (...args: unknown[]) => state.captureScope(...args)
}))
vi.mock("@/hooks/chat/useWorkspaceChatCheckpoint", () => ({
  useWorkspaceChatCheckpoint: (options: { setDraft: (value: string) => void }) => ({
    active: state.active, restoring: state.restoring, setDraft: options.setDraft, referenceId: state.referenceId,
    fence: () => {
      const token = state.epoch
      return () => token === state.epoch
    },
    controller: state.active ? {
      recoveries: [{ turn: recovery }],
      getReference: () => ({ owner_key: state.ownerKey, conversation_id: "chat-A" }),
      getCurrent: () => ({ status: "ready", owner: { kind: "native", validate_lease: () => true } }),
      fence: () => {
      const token = state.epoch
      return () => token === state.epoch
    } } : null
  })
}))
vi.mock("@/components/Common/Playground/HistorySelectionReview", () => ({
  HistorySelectionReview: ({ onReprepareRecovery }: { onReprepareRecovery?: (turn: HistoryTurnRecovery) => void }) =>
    <button onClick={() => onReprepareRecovery?.(recovery)}>Explicit reprepare</button>
}))
vi.mock("@/hooks/useMessageOption", async () => {
  const { useStoreMessageOption } = await import("@/store/option")
  return { useMessageOption: () => ({ ...useStoreMessageOption(), onSubmit: state.onSubmit, regenerateLastMessage: state.regenerate, stopStreamingRequest: vi.fn() }) }
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
  tldwClient: { getMediaDetails: state.getMediaDetails, getChatLorebookDiagnostics: vi.fn(async () => ({ turns: [], total_turns_with_diagnostics: 0 })) }
}))
vi.mock("@/services/tldw-server", () => ({ fetchChatModels: async () => [{ id: "test-model", name: "Test Model", provider: "test" }] }))
vi.mock("@/components/Option/Playground/ChatModelSelectorDropdown", () => ({ ChatModelSelectorDropdown: () => null }))
vi.mock("@/components/Common/Playground/Message", () => ({
  PlaygroundMessage: ({ onRegenerate, onSaveToWorkspaceNotes }: { onRegenerate?: () => void, onSaveToWorkspaceNotes?: () => void }) => {
    savedReplyCapture = onSaveToWorkspaceNotes
    return <><button onClick={onRegenerate}>Retry interrupted reply</button>
      {onSaveToWorkspaceNotes && <button onClick={onSaveToWorkspaceNotes}>Save reply to workspace notes</button>}</>
  }
}))

import { useStoreMessageOption } from "@/store/option"
import { useWorkspaceStore } from "@/store/workspace"
import { serverWorkspaceMetadata } from "@/store/__tests__/workspace-activation.fixtures"
import { ChatPane } from "../ChatPane"

let savedReplyCapture: (() => void) | undefined

beforeEach(() => {
  savedReplyCapture = undefined
  state.active = true
  state.restoring = false
  state.epoch = 0
  state.referenceId = "reference-A"
  state.ownerKey = "native-A"
  state.onSubmit.mockReset().mockResolvedValue({ status: "submitted" })
  state.getMediaDetails.mockReset().mockResolvedValue({ content: { text: "Operator source text" } })
  state.captureHead.mockReset()
  state.captureScope.mockReset()
  state.regenerate.mockReset()
  recovery.logical_user_message_id = uuid
  recovery.input_text = "Original recovered question"
  useWorkspaceStore.setState({ workspaceId: "workspace-A", workspaceChatReferenceId: "reference-A", storeHydrated: true,
    sources: [], selectedSourceIds: [], selectedSourceFolderIds: [], sourceFolders: [], sourceFolderMemberships: [] })
  useStoreMessageOption.setState({ selectedModel: "test-model", messages: [], history: [], historyId: null,
    serverChatId: null, temporaryChat: false, streaming: false, isProcessing: false, chatMode: "normal" })
})
const submit = () => {
  fireEvent.change(screen.getByRole("textbox", { name: "Chat message" }), { target: { value: "New question" } })
  fireEvent.click(screen.getByRole("button", { name: "Send" }))
}

it.each(["local", "workspace", "note"])("checks the live %s destination before capturing a reply", async destination => {
  useWorkspaceStore.setState({ serverWorkspace: null })
  useWorkspaceStore.getState().setCurrentNote({ ...useWorkspaceStore.getState().currentNote,
    id: null, serverWorkspaceId: undefined, serverScopeKey: undefined,
    title: "Private draft", content: "Unsent body", isDirty: true })
  useStoreMessageOption.setState({ messages: [{ id: "reply", isBot: true, name: "Assistant", message: "Captured reply", sources: [] }] })
  render(<MemoryRouter><ChatPane /></MemoryRouter>)
  const capture = savedReplyCapture
  expect(capture).toBeTypeOf("function")
  const before = useWorkspaceStore.getState().currentNote
  if (destination === "workspace") act(() => {
    useWorkspaceStore.setState({ serverWorkspace: { scopeKey: "owner-a", sourceSignature: "", selectedSourceSignature: "",
      metadata: { ...serverWorkspaceMetadata, id: "workspace-A" }, notes: [] } })
  })
  if (destination === "note") act(() => {
    useWorkspaceStore.getState().setCurrentNote({ ...useWorkspaceStore.getState().currentNote,
      serverWorkspaceId: "workspace-A", serverScopeKey: "owner-a" })
  })
  const destinationNote = useWorkspaceStore.getState().currentNote
  act(() => capture?.())
  if (destination === "local") {
    expect(useWorkspaceStore.getState().currentNote.content).toContain("Unsent body")
    expect(useWorkspaceStore.getState().currentNote.content).toContain("Captured reply")
  } else {
    expect(useWorkspaceStore.getState().currentNote).toBe(destinationNote)
    expect(await screen.findByText("Server notes are view-only")).toBeVisible()
    expect(screen.queryByRole("button", { name: "Save reply to workspace notes" })).toBeNull()
    expect(useWorkspaceStore.getState().currentNote.content).toBe(before.content)
  }
  act(() => { useWorkspaceStore.setState({ serverWorkspace: null }); useWorkspaceStore.getState().setCurrentNote({
    ...useWorkspaceStore.getState().currentNote, serverWorkspaceId: undefined, serverScopeKey: undefined }) })
})

it.each(["current", "new selection", "selection ABA"])(
  "capture preflight keeps Retry's click-time history fence for %s",
  async transition => {
    let releaseHead!: () => void
    const releaseScope = vi.fn()
    state.captureScope.mockResolvedValue({
      requestScope: {}, scopeSignal: new AbortController().signal,
      scopeInvalidatedSignal: new AbortController().signal, release: releaseScope
    })
    state.captureHead.mockImplementation(() => new Promise<void>(resolve => { releaseHead = resolve }))
    state.regenerate.mockImplementation(async (dispatch: { assertCurrent?: () => void }) => {
      state.epoch += 1
      dispatch.assertCurrent?.()
    })
    useWorkspaceStore.setState({
      sources: [{ id: "capture", mediaId: 1, title: "Captured source", type: "website", status: "ready", addedAt: new Date(0),
        webCapture: { clipId: "clip", requestedUrl: "https://example.org", capturedAt: "2026-10-08T00:00:00Z",
          contentSha256: "a".repeat(64), refreshOf: null, mediaId: 1, versionNumber: 1, versionUuid: "version" } }],
      selectedSourceIds: ["capture"]
    })
    useStoreMessageOption.setState({ messages: [{ id: "reply", isBot: true, name: "Assistant", message: "Interrupted reply", sources: [] }] })
    render(<MemoryRouter><ChatPane /></MemoryRouter>)
    fireEvent.click(screen.getByRole("button", { name: "Retry interrupted reply" }))
    await waitFor(() => expect(state.captureHead).toHaveBeenCalledTimes(1))
    if (transition !== "current") state.epoch += 1
    if (transition === "selection ABA") state.epoch += 1
    await act(async () => { releaseHead() })
    await waitFor(() => expect(releaseScope).toHaveBeenCalledTimes(1))
    expect(state.regenerate).toHaveBeenCalledTimes(transition === "current" ? 1 : 0)
    expect(state.onSubmit).not.toHaveBeenCalled()
  }
)

it("keeps retained recovery controls in the transcript scroller outside the composer", () => {
  render(<MemoryRouter><ChatPane /></MemoryRouter>)
  const transcript = screen.getByRole("log", { name: "Chat messages" })
  expect(transcript).toContainElement(screen.getByRole("button", { name: "Explicit reprepare" }))
  expect(transcript).not.toContainElement(screen.getByRole("textbox", { name: "Chat message" }))
})

it.each(["normal", "rag"])("allocates a durable UUID for an explicit new saved %s send", async mode => {
  if (mode === "rag") useWorkspaceStore.setState({
    sources: [{ id: "source", mediaId: 1, title: "Selected source", type: "document", status: "ready", addedAt: new Date(0) }],
    selectedSourceIds: ["source"]
  })
  render(<MemoryRouter><ChatPane /></MemoryRouter>)
  submit()
  await waitFor(() => expect(state.onSubmit).toHaveBeenCalledTimes(1))
  expect(state.onSubmit.mock.calls[0][0].requestOverrides?.tldwTurn?.user_message_id)
    .toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i)
  expect(useStoreMessageOption.getState().chatMode).toBe(mode)
})

it.each(["temporary", "no provider"])("does not add a durable turn to %s legacy sends", async kind => {
  state.active = kind !== "no provider"
  useStoreMessageOption.setState({ temporaryChat: kind === "temporary" })
  render(<MemoryRouter><ChatPane /></MemoryRouter>)
  submit()
  await waitFor(() => expect(state.onSubmit).toHaveBeenCalledTimes(1))
  expect(state.onSubmit.mock.calls[0][0].requestOverrides).toBeUndefined()
})

it("does not dispatch a delayed capability check after the owner lease changes", async () => {
  let resolve!: (value: ResearchWorkspaceCapabilitiesResponse) => void
  const refresh = vi.fn(() => new Promise<ResearchWorkspaceCapabilitiesResponse>(done => { resolve = done }))
  render(<MemoryRouter><ChatPane researchWorkspaceCapabilitiesStale onRefreshResearchWorkspaceCapabilities={refresh} /></MemoryRouter>)
  submit()
  await waitFor(() => expect(refresh).toHaveBeenCalled())
  state.epoch += 1
  await act(async () => { resolve(buildUnknownResearchWorkspaceCapabilities()) })
  expect(state.onSubmit).not.toHaveBeenCalled()
  expect(screen.getByRole("textbox", { name: "Chat message" })).toHaveValue("")
})

it("explicit reprepare retains the durable logical UUID until explicit Send", async () => {
  useStoreMessageOption.setState({ serverChatId: "chat-A" })
  render(<MemoryRouter><ChatPane /></MemoryRouter>)
  fireEvent.click(screen.getByRole("button", { name: "Explicit reprepare" }))
  expect(screen.getByRole("textbox", { name: "Chat message" })).toHaveValue(recovery.input_text)
  expect(state.onSubmit).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole("button", { name: "Send" }))
  await waitFor(() => expect(state.onSubmit).toHaveBeenCalledTimes(1))
  expect(state.onSubmit.mock.calls[0][0].requestOverrides.tldwTurn).toEqual({ user_message_id: uuid })
})

it.each([
  {
    kind: "response preset", preset: true, fullSource: false,
    prepared: "Response preference: When answering, lead with what the selected sources support and call out uncertainty.\n\nUser question: New question"
  },
  {
    kind: "full source", preset: false, fullSource: true,
    prepared: "Use the complete source contents below when answering the user question.\nWhen relevant, cite source titles directly.\n\n<<SOURCE START: Source 1: Selected source>>\nOperator source text\n<<SOURCE END: Source 1: Selected source>>\n\nUser question: New question"
  },
  {
    kind: "preset and full source", preset: true, fullSource: true,
    prepared: "Use the complete source contents below when answering the user question.\nWhen relevant, cite source titles directly.\nResponse preference: When answering, lead with what the selected sources support and call out uncertainty.\n\n<<SOURCE START: Source 1: Selected source>>\nOperator source text\n<<SOURCE END: Source 1: Selected source>>\n\nUser question: New question"
  }
])("reprepares an actual $kind send without preparing its input twice", async ({ preset, fullSource, prepared }) => {
  if (fullSource) useWorkspaceStore.setState({
    sources: [{ id: "source", mediaId: 1, title: "Selected source", type: "document", status: "ready", addedAt: new Date(0) }],
    selectedSourceIds: ["source"]
  })
  useStoreMessageOption.setState({ serverChatId: "chat-A" })
  state.onSubmit.mockResolvedValueOnce({ status: "blocked" })
  render(<MemoryRouter><ChatPane /></MemoryRouter>)
  if (preset) fireEvent.change(screen.getByLabelText("Response style"), { target: { value: "source_first" } })
  if (fullSource) fireEvent.click(screen.getByRole("switch", { name: "Include full source contents" }))
  submit()
  await waitFor(() => expect(screen.getByRole("textbox", { name: "Chat message" })).toHaveValue("New question"))
  const firstSend = state.onSubmit.mock.calls[0][0]
  expect(firstSend.message).toBe(prepared)
  recovery.input_text = firstSend.message
  recovery.logical_user_message_id = firstSend.requestOverrides.tldwTurn.user_message_id
  const prior = structuredClone(recovery)
  fireEvent.click(screen.getByRole("button", { name: "Explicit reprepare" }))
  expect(screen.getByRole("textbox", { name: "Chat message" })).toHaveValue(prepared)
  expect(state.onSubmit).toHaveBeenCalledTimes(1)
  fireEvent.click(screen.getByRole("button", { name: "Send" }))
  await waitFor(() => expect(state.onSubmit).toHaveBeenCalledTimes(2))
  expect(state.onSubmit.mock.calls[1][0].message).toBe(prepared)
  expect(state.onSubmit.mock.calls[1][0].requestOverrides.tldwTurn).toEqual(firstSend.requestOverrides.tldwTurn)
  expect(state.getMediaDetails).toHaveBeenCalledTimes(fullSource ? 1 : 0)
  expect(recovery).toEqual(prior)
})

it("does not reprepare a foreign owner's recovery", () => {
  state.ownerKey = "foreign-owner"
  useStoreMessageOption.setState({ serverChatId: "chat-A" })
  render(<MemoryRouter><ChatPane /></MemoryRouter>)
  fireEvent.click(screen.getByRole("button", { name: "Explicit reprepare" }))
  expect(screen.getByRole("textbox", { name: "Chat message" })).toHaveValue("")
})

it("typing during capability preparation cannot replace the submitted recovery UUID", async () => {
  let resolve!: (value: ResearchWorkspaceCapabilitiesResponse) => void
  const refresh = vi.fn(() => new Promise<ResearchWorkspaceCapabilitiesResponse>(done => { resolve = done }))
  useStoreMessageOption.setState({ serverChatId: "chat-A" })
  render(<MemoryRouter><ChatPane researchWorkspaceCapabilitiesStale onRefreshResearchWorkspaceCapabilities={refresh} /></MemoryRouter>)
  fireEvent.click(screen.getByRole("button", { name: "Explicit reprepare" }))
  fireEvent.click(screen.getByRole("button", { name: "Send" }))
  await waitFor(() => expect(refresh).toHaveBeenCalled())
  fireEvent.change(screen.getByRole("textbox", { name: "Chat message" }), { target: { value: "Next question" } })
  await act(async () => { resolve(buildUnknownResearchWorkspaceCapabilities()) })
  expect(state.onSubmit.mock.calls[0][0].requestOverrides.tldwTurn).toEqual({ user_message_id: uuid })
  expect(screen.getByRole("textbox", { name: "Chat message" })).toHaveValue("Next question")
})

it.each(["trim", "response preset", "full source"])("CP-F2 Research %s text construction retires the recovery UUID", async rewrite => {
  if (rewrite === "trim") recovery.input_text = "  Original recovered question  "
  if (rewrite === "full source") useWorkspaceStore.setState({
    sources: [{ id: "source", mediaId: 1, title: "Selected source", type: "document", status: "ready", addedAt: new Date(0) }],
    selectedSourceIds: ["source"]
  })
  const prior = structuredClone(recovery)
  useStoreMessageOption.setState({ serverChatId: "chat-A" })
  render(<MemoryRouter><ChatPane /></MemoryRouter>)
  fireEvent.click(screen.getByRole("button", { name: "Explicit reprepare" }))
  if (rewrite === "response preset") fireEvent.change(screen.getByLabelText("Response style"), { target: { value: "source_first" } })
  if (rewrite === "full source") fireEvent.click(screen.getByRole("switch", { name: "Include full source contents" }))
  fireEvent.click(screen.getByRole("button", { name: "Send" }))
  await waitFor(() => expect(state.onSubmit).toHaveBeenCalledTimes(1))
  expect(state.onSubmit.mock.calls[0][0].message).not.toBe(recovery.input_text)
  expect(state.onSubmit.mock.calls[0][0].requestOverrides.tldwTurn.user_message_id).not.toBe(uuid)
  expect(recovery).toEqual(prior)
})

it.each(["typing", "model", "source selection", "source title", "mode", "answer length"])("retires the recovered intent after a genuine %s change", async change => {
  useWorkspaceStore.setState({
    sources: [
      { id: "source", mediaId: 1, title: "Selected source", type: "document", status: "ready", addedAt: new Date(0) },
      { id: "other", mediaId: 2, title: "Other source", type: "document", status: "ready", addedAt: new Date(0) }
    ],
    selectedSourceIds: ["source"]
  })
  useStoreMessageOption.setState({ serverChatId: "chat-A" })
  const prior = structuredClone(recovery)
  render(<MemoryRouter><ChatPane /></MemoryRouter>)
  fireEvent.click(screen.getByRole("button", { name: "Explicit reprepare" }))
  if (change === "typing") fireEvent.change(screen.getByRole("textbox", { name: "Chat message" }), { target: { value: "Edited question" } })
  if (change === "model") await act(async () => { useStoreMessageOption.setState({ selectedModel: "other-model" }) })
  if (change === "source selection") await act(async () => { useWorkspaceStore.setState({ selectedSourceIds: ["other"] }) })
  if (change === "source title") await act(async () => {
    useWorkspaceStore.setState({
      sources: useWorkspaceStore.getState().sources.map(source => source.id === "source" ? { ...source, title: "Updated title" } : source)
    })
  })
  if (change === "mode") fireEvent.click(screen.getByRole("button", { name: "General chat", exact: true }))
  if (change === "answer length") fireEvent.change(screen.getByLabelText("Answer length"), { target: { value: "brief" } })
  fireEvent.click(screen.getByRole("button", { name: "Send" }))
  await waitFor(() => expect(state.onSubmit).toHaveBeenCalledTimes(1))
  expect(state.onSubmit.mock.calls[0][0].requestOverrides.tldwTurn.user_message_id).not.toBe(uuid)
  expect(recovery).toEqual(prior)
})

it("keeps the recovered intent when an unselected source changes", async () => {
  useWorkspaceStore.setState({
    sources: [
      { id: "source", mediaId: 1, title: "Selected source", type: "document", status: "ready", addedAt: new Date(0) },
      { id: "other", mediaId: 2, title: "Other source", type: "document", status: "ready", addedAt: new Date(0) }
    ],
    selectedSourceIds: ["source"]
  })
  useStoreMessageOption.setState({ serverChatId: "chat-A" })
  render(<MemoryRouter><ChatPane /></MemoryRouter>)
  fireEvent.click(screen.getByRole("button", { name: "Explicit reprepare" }))
  await act(async () => {
    useWorkspaceStore.setState({
      sources: useWorkspaceStore.getState().sources.map(source => source.id === "other" ? { ...source, title: "Updated title" } : source)
    })
  })
  fireEvent.click(screen.getByRole("button", { name: "Send" }))
  await waitFor(() => expect(state.onSubmit).toHaveBeenCalledTimes(1))
  expect(state.onSubmit.mock.calls[0][0].requestOverrides.tldwTurn).toEqual({ user_message_id: uuid })
})
