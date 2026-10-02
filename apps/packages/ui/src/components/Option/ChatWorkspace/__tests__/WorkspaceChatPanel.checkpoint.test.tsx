import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
import type { HistoryTurnRecovery } from "@/db/dexie/types"
import type { useMessageOption } from "@/hooks/useMessageOption"
import type { Message } from "@/store/option"
import type { MessageRecoveryAction } from "@/components/Common/Playground/Message"

type SubmitPayload = Parameters<ReturnType<typeof useMessageOption>["onSubmit"]>[0]

const uuid = "00000000-0000-4000-8000-000000000001"
const state = vi.hoisted(() => ({
  onSubmit: vi.fn(), inspectRecovery: vi.fn(), referenceId: "reference-A",
  ownerKey: "native-A", conversationId: "chat-A", messages: [] as Message[],
  recoveries: [] as Array<{ turn: HistoryTurnRecovery }>, temporaryChat: false
}))
const turn: HistoryTurnRecovery = {
  persistence: "server", logical_user_message_id: uuid,
  operation_id: "old-operation", owner_key: "native-A", conversation_id: "chat-A",
  origin_view: {
    owner_key: "native-A", conversation_id: "chat-A", view_session_id: "expired-view",
    selection_revision: 1, interpretation: { kind: "parent_graph_v1" }, cursor: { kind: "empty" }
  },
  selection_digest: "a".repeat(64), request_context_digest: "b".repeat(64),
  created_at: 1, input_text: "Original recovered question", input_images: [], result_text: "", state: "unknown"
}
const admission = {
  version: 1 as const, owner_key: "native-A", conversation_id: "chat-A",
  input_message_id: "saved-user", input_message_revision: "1", selection_digest: "a".repeat(64)
}
const finalizedSelection = {
  version: 1 as const, owner_key: "native-A", conversation_id: "chat-A",
  interpretation: { kind: "parent_graph_v1" as const }, cursor: { kind: "empty" as const },
  selection_revision: 1, purpose: "send" as const, messages: [],
  fences: { conversation: "1", history: "1", settings: "1" },
  storage_context_digest: "c".repeat(64), request_context_digest: "b".repeat(64),
  selection_digest: "a".repeat(64)
}
vi.mock("@/hooks/chat/useWorkspaceChatCheckpoint", () => ({
  useWorkspaceChatCheckpoint: (options: { setDraft: (value: string) => void }) => ({
    active: true, restoring: false, setDraft: options.setDraft, referenceId: state.referenceId,
    controller: {
      status: "ready", settingsQualified: true, recoveries: state.recoveries,
      inspectRecovery: state.inspectRecovery,
      owner: { kind: "native", owner_key: state.ownerKey, conversation_id: state.conversationId, validate_lease: () => true },
      getCurrent: () => ({ status: "ready", owner: { kind: "native", validate_lease: () => true } }),
      getReference: () => ({ owner_key: state.ownerKey, conversation_id: state.conversationId })
    }
  })
}))
vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => ({
    messages: state.messages, history: [], historyId: null, serverChatId: "chat-A", temporaryChat: state.temporaryChat,
    onSubmit: state.onSubmit, streaming: false, isLoading: false, isProcessing: false,
    selectedModel: "test-model", selectedAssistant: null, selectedAssistantSource: "explicit",
    serverChatAssistantKind: null, serverChatAssistantId: null,
    serverChatLoadState: "ready", serverChatLoadError: null,
    setMessages: vi.fn(), stopStreamingRequest: vi.fn()
  })
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (_key: string, fallback: string) => fallback })
}))
vi.mock("@/services/service-prompts", () => ({
  resolveServicePromptScope: async () => ({ config: { serverUrl: "http://owner.test", authMode: "multi-user", authSource: "manual" }, userId: 1 })
}))
vi.mock("@/services/tldw-server", () => ({ fetchChatModels: async () => [] }))
vi.mock("@/services/chat-macros", () => ({ cancelChatMacroRun: vi.fn() }))
vi.mock("@/store/model", () => ({ useStoreChatModelSettings: () => ({ apiProvider: "openai" }) }))
vi.mock("@/hooks/useSetting", () => ({ useSetting: () => [undefined] }))
vi.mock("@/hooks/useDarkmode", () => ({ useDarkModeStore: (select: (value: { mode: string }) => unknown) => select({ mode: "light" }) }))
vi.mock("../MacroRunDetailDrawer", () => ({ MacroRunDetailDrawer: () => null }))
vi.mock("@/components/Common/Playground/Message", () => ({
  PlaygroundMessage: ({ message, recoveryActions }: { message: string; recoveryActions: MessageRecoveryAction[] }) => (
    <article>
      {message}
      {recoveryActions.map((action) => (
        <button key={action.id} onClick={action.onClick} disabled={action.disabled}>{action.label}</button>
      ))}
    </article>
  )
}))

import { WorkspaceChatPanel } from "../WorkspaceChatPanel"

beforeEach(() => {
  state.onSubmit.mockReset().mockResolvedValue({ status: "submitted" })
  state.referenceId = "reference-A"
  state.ownerKey = "native-A"
  state.conversationId = "chat-A"
  state.messages = []
  state.temporaryChat = false
  state.recoveries = [{ turn }]
  state.inspectRecovery.mockReset().mockResolvedValue(undefined)
  turn.input_text = "Original recovered question"
  turn.logical_user_message_id = uuid
  turn.admission = admission
  turn.finalized_selection = finalizedSelection
  turn.state = "accepted_unsent"
})
const props = { workspaceId: "workspace-A", workspaceReady: true, stagedSources: [], backendAvailable: true, onClearStagedSources: () => {} }

it("keeps retained recovery controls in the transcript scroller outside the composer", () => {
  render(<WorkspaceChatPanel {...props} />)
  const scroller = screen.getByRole("button", { name: "Reprepare input" }).closest(".overflow-y-auto")
  expect(scroller).toHaveClass("min-h-0", "flex-1")
  expect(scroller).not.toContainElement(screen.getByRole("textbox", { name: "Chat workspace message" }))
})

it("explicit reprepare fills only the draft and reuses logical UUID on the next explicit Send", async () => {
  render(<WorkspaceChatPanel {...props} />)
  fireEvent.click(screen.getByRole("button", { name: "Reprepare input" }))
  expect(screen.getByRole("textbox", { name: "Chat workspace message" })).toHaveValue("Original recovered question")
  expect(state.onSubmit).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole("button", { name: "Send message" }))
  await waitFor(() => expect(state.onSubmit).toHaveBeenCalled())
  expect(state.onSubmit.mock.calls[0][0].requestOverrides.tldwTurn).toEqual({ user_message_id: uuid })
})

it("reprepare cannot copy another owner's recovery draft or UUID", () => {
  state.ownerKey = "other-owner"
  render(<WorkspaceChatPanel {...props} />)
  fireEvent.click(screen.getByRole("button", { name: "Reprepare input" }))
  expect(screen.getByRole("textbox", { name: "Chat workspace message" })).toHaveValue("")
  expect(state.onSubmit).not.toHaveBeenCalled()
})

it("New Chat retires the pending reprepare UUID", async () => {
  const surface = render(<WorkspaceChatPanel {...props} />)
  fireEvent.click(screen.getByRole("button", { name: "Reprepare input" }))
  state.referenceId = "new-chat"
  surface.rerender(<WorkspaceChatPanel {...props} />)
  fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), { target: { value: "New question" } })
  fireEvent.click(screen.getByRole("button", { name: "Send message" }))
  await waitFor(() => expect(state.onSubmit).toHaveBeenCalled())
  expect(state.onSubmit.mock.calls[0][0].requestOverrides.tldwTurn.user_message_id).not.toBe(uuid)
})

it("CP-F2 staged insertion retires the recovery UUID without changing its prior recovery", async () => {
  const prior = structuredClone(turn)
  render(<WorkspaceChatPanel {...props} stagedSources={[{
    sourceId: "source-1", mediaId: 101, title: "Operator Notes", type: "document",
    scopeLabel: "Default workspace", availability: "ready"
  }]} />)
  fireEvent.click(screen.getByRole("button", { name: "Reprepare input" }))
  fireEvent.click(screen.getByRole("button", { name: "Insert context summary" }))
  fireEvent.click(screen.getByRole("button", { name: "Send message" }))
  await waitFor(() => expect(state.onSubmit).toHaveBeenCalledTimes(1))
  expect(state.onSubmit.mock.calls[0][0].message).not.toBe(turn.input_text)
  expect(state.onSubmit.mock.calls[0][0].requestOverrides.tldwTurn.user_message_id).not.toBe(uuid)
  expect(turn).toEqual(prior)
})

it("CP-F2 direct typing retires the recovery UUID", async () => {
  render(<WorkspaceChatPanel {...props} />)
  fireEvent.click(screen.getByRole("button", { name: "Reprepare input" }))
  fireEvent.change(screen.getByRole("textbox", { name: "Chat workspace message" }), { target: { value: "Changed input" } })
  fireEvent.click(screen.getByRole("button", { name: "Send message" }))
  await waitFor(() => expect(state.onSubmit).toHaveBeenCalledTimes(1))
  expect(state.onSubmit.mock.calls[0][0].requestOverrides.tldwTurn.user_message_id).not.toBe(uuid)
})

it("CP-F2 direct fallback context changes the logical input identity", async () => {
  const prior = structuredClone(turn)
  render(<WorkspaceChatPanel {...props} stagedSources={[{
    sourceId: "source-1", mediaId: null, title: "Operator Notes", type: "document",
    scopeLabel: "Default workspace", availability: "unavailable"
  }]} />)
  fireEvent.click(screen.getByRole("button", { name: "Reprepare input" }))
  fireEvent.click(screen.getByRole("button", { name: "Send message" }))
  await waitFor(() => expect(state.onSubmit).toHaveBeenCalledTimes(1))
  expect(state.onSubmit.mock.calls[0][0].message).toContain("Context sources:")
  expect(state.onSubmit.mock.calls[0][0].requestOverrides.tldwTurn.user_message_id).not.toBe(uuid)
  expect(turn).toEqual(prior)
})

it("CP-F2 final user-text trimming cannot reuse a different recovery input", async () => {
  turn.input_text = "  Original recovered question  "
  render(<WorkspaceChatPanel {...props} />)
  fireEvent.click(screen.getByRole("button", { name: "Reprepare input" }))
  fireEvent.click(screen.getByRole("button", { name: "Send message" }))
  await waitFor(() => expect(state.onSubmit).toHaveBeenCalledTimes(1))
  expect(state.onSubmit.mock.calls[0][0].message).toBe("Original recovered question")
  expect(state.onSubmit.mock.calls[0][0].requestOverrides.tldwTurn.user_message_id).not.toBe(uuid)
})

it("CP-F2 ready request-local evidence preserves unchanged recovery input and identity", async () => {
  const prior = structuredClone(turn)
  render(<WorkspaceChatPanel {...props} stagedSources={[{
    sourceId: "source-1", mediaId: 101, title: "Operator Notes", type: "document",
    scopeLabel: "Default workspace", availability: "ready"
  }]} />)
  fireEvent.click(screen.getByRole("button", { name: "Reprepare input" }))
  fireEvent.click(screen.getByRole("button", { name: "Send message" }))
  await waitFor(() => expect(state.onSubmit).toHaveBeenCalledTimes(1))
  expect(state.onSubmit.mock.calls[0][0].message).toBe(turn.input_text)
  expect(state.onSubmit.mock.calls[0][0].requestOverrides.tldwTurn.user_message_id).toBe(uuid)
  expect(turn).toEqual(prior)
})

it.each([
  { admitted: true, assistantRow: false },
  { admitted: true, assistantRow: true },
  { admitted: false, assistantRow: false }
])("uses protected recovery after a selected durable failure (%j)", async ({ admitted, assistantRow }) => {
  state.recoveries = []
  state.onSubmit.mockImplementationOnce(async (request: SubmitPayload) => {
    const durableTurn = request.requestOverrides?.tldwTurn
    if (!durableTurn || typeof durableTurn !== "object" || !("user_message_id" in durableTurn) ||
      typeof durableTurn.user_message_id !== "string") throw new Error("Expected selected durable turn ID")
    turn.input_text = request.message
    turn.logical_user_message_id = durableTurn.user_message_id
    turn.admission = admitted ? admission : undefined
    turn.state = admitted ? "accepted_unsent" : "unknown"
    state.recoveries = [{ turn }]
    state.messages = admitted ? [{ id: "saved-user", isBot: false, message: request.message, name: "You", sources: [] }] : []
    if (assistantRow) state.messages.push({
      id: "partial-assistant", isBot: true, message: "Partial answer", parentMessageId: "saved-user", name: "Assistant", sources: []
    })
    return { status: "failed", errorMessage: "Response outcome unknown" }
  })
  render(<WorkspaceChatPanel {...props} />)
  fireEvent.change(screen.getByRole("textbox"), { target: { value: "Original question" } })
  fireEvent.click(screen.getByRole("button", { name: "Send message" }))
  await screen.findByRole("alert")

  expect(screen.queryByRole("button", { name: "Retry same model" })).not.toBeInTheDocument()
  expect(screen.queryByRole("button", { name: "Switch model" })).not.toBeInTheDocument()
  expect(screen.queryByRole("combobox", { name: "Retry model" })).not.toBeInTheDocument()
  expect(screen.getByText("Original question", { selector: "pre" })).toBeInTheDocument()
  fireEvent.click(screen.getByRole("button", { name: "Verify saved outcome" }))
  expect(state.inspectRecovery).toHaveBeenCalledWith(state.recoveries[0])
  expect(state.onSubmit).toHaveBeenCalledTimes(1)

  if (admitted) {
    fireEvent.click(screen.getByRole("button", { name: "Reprepare input" }))
    expect(screen.getByRole("textbox")).toHaveValue("Original question")
    expect(state.onSubmit).toHaveBeenCalledTimes(1)
    fireEvent.click(screen.getByRole("button", { name: "Send message" }))
    await waitFor(() => expect(state.onSubmit).toHaveBeenCalledTimes(2))
    expect(state.onSubmit.mock.calls[1][0]).toMatchObject({
      message: "Original question", requestOverrides: { tldwTurn: state.onSubmit.mock.calls[0][0].requestOverrides.tldwTurn }
    })
    expect(state.onSubmit.mock.calls[1][0].isRegenerate).not.toBe(true)
  } else {
    expect(screen.queryByRole("button", { name: "Reprepare input" })).not.toBeInTheDocument()
    expect(turn.state).toBe("unknown")
    expect(turn.admission).toBeUndefined()
  }
})

it("keeps legacy recovery available for a temporary chat without a durable turn", async () => {
  state.temporaryChat = true
  state.recoveries = []
  state.onSubmit.mockResolvedValueOnce({ status: "failed", errorMessage: "Provider unavailable" })
  render(<WorkspaceChatPanel {...props} />)
  fireEvent.change(screen.getByRole("textbox"), { target: { value: "Temporary question" } })
  fireEvent.click(screen.getByRole("button", { name: "Send message" }))
  await screen.findByRole("alert")
  expect(state.onSubmit.mock.calls[0][0].requestOverrides.tldwTurn).toBeUndefined()
  expect(screen.getByRole("button", { name: "Switch model" })).toBeEnabled()
  fireEvent.click(screen.getByRole("button", { name: "Retry same model" }))
  await waitFor(() => expect(state.onSubmit).toHaveBeenCalledTimes(2))
  expect(state.onSubmit.mock.calls[1][0].requestOverrides.tldwTurn).toBeUndefined()
})
