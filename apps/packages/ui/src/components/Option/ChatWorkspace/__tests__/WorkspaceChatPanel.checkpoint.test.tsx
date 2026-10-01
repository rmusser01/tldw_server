import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
import type { HistoryTurnRecovery } from "@/db/dexie/types"

const uuid = "00000000-0000-4000-8000-000000000001"
const state = vi.hoisted(() => ({
  onSubmit: vi.fn(), referenceId: "reference-A",
  ownerKey: "native-A", conversationId: "chat-A"
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
vi.mock("@/hooks/chat/useWorkspaceChatCheckpoint", () => ({
  useWorkspaceChatCheckpoint: (options: { setDraft: (value: string) => void }) => ({
    active: true, restoring: false, setDraft: options.setDraft, referenceId: state.referenceId,
    controller: {
      recoveries: [{ turn }],
      owner: { kind: "native", owner_key: state.ownerKey, conversation_id: state.conversationId, validate_lease: () => true },
      getCurrent: () => ({ status: "ready", owner: { kind: "native", validate_lease: () => true } }),
      getReference: () => ({ owner_key: state.ownerKey, conversation_id: state.conversationId })
    }
  })
}))
vi.mock("@/components/Common/Playground/HistorySelectionReview", () => ({
  HistorySelectionReview: ({ onReprepareRecovery }: { onReprepareRecovery?: (turn: HistoryTurnRecovery) => void }) =>
    <button onClick={() => onReprepareRecovery?.(turn)}>Explicit reprepare</button>
}))
vi.mock("@/hooks/useMessageOption", () => ({
  useMessageOption: () => ({
    messages: [], history: [], historyId: null, serverChatId: "chat-A", temporaryChat: false,
    onSubmit: state.onSubmit, streaming: false, isLoading: false, isProcessing: false,
    selectedModel: "test-model", selectedAssistant: null, selectedAssistantSource: "explicit",
    serverChatAssistantKind: null, serverChatAssistantId: null,
    serverChatLoadState: "ready", serverChatLoadError: null,
    setMessages: vi.fn(), stopStreamingRequest: vi.fn()
  })
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
vi.mock("@/components/Common/Playground/Message", () => ({ PlaygroundMessage: () => null }))

import { WorkspaceChatPanel } from "../WorkspaceChatPanel"

beforeEach(() => {
  state.onSubmit.mockReset().mockResolvedValue({ status: "submitted" })
  state.referenceId = "reference-A"
  state.ownerKey = "native-A"
  state.conversationId = "chat-A"
  turn.input_text = "Original recovered question"
})
const props = { workspaceId: "workspace-A", workspaceReady: true, stagedSources: [], backendAvailable: true, onClearStagedSources: () => {} }

it("keeps retained recovery controls in the transcript scroller outside the composer", () => {
  render(<WorkspaceChatPanel {...props} />)
  const scroller = screen.getByRole("button", { name: "Explicit reprepare" }).closest(".overflow-y-auto")
  expect(scroller).toHaveClass("min-h-0", "flex-1")
  expect(scroller).not.toContainElement(screen.getByRole("textbox", { name: "Chat workspace message" }))
})

it("explicit reprepare fills only the draft and reuses logical UUID on the next explicit Send", async () => {
  render(<WorkspaceChatPanel {...props} />)
  fireEvent.click(screen.getByRole("button", { name: "Explicit reprepare" }))
  expect(screen.getByRole("textbox", { name: "Chat workspace message" })).toHaveValue("Original recovered question")
  expect(state.onSubmit).not.toHaveBeenCalled()
  fireEvent.click(screen.getByRole("button", { name: "Send message" }))
  await waitFor(() => expect(state.onSubmit).toHaveBeenCalled())
  expect(state.onSubmit.mock.calls[0][0].requestOverrides.tldwTurn).toEqual({ user_message_id: uuid })
})

it("reprepare cannot copy another owner's recovery draft or UUID", () => {
  state.ownerKey = "other-owner"
  render(<WorkspaceChatPanel {...props} />)
  fireEvent.click(screen.getByRole("button", { name: "Explicit reprepare" }))
  expect(screen.getByRole("textbox", { name: "Chat workspace message" })).toHaveValue("")
  expect(state.onSubmit).not.toHaveBeenCalled()
})

it("New Chat retires the pending reprepare UUID", async () => {
  const surface = render(<WorkspaceChatPanel {...props} />)
  fireEvent.click(screen.getByRole("button", { name: "Explicit reprepare" }))
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
  fireEvent.click(screen.getByRole("button", { name: "Explicit reprepare" }))
  fireEvent.click(screen.getByRole("button", { name: "Insert context summary" }))
  fireEvent.click(screen.getByRole("button", { name: "Send message" }))
  await waitFor(() => expect(state.onSubmit).toHaveBeenCalledTimes(1))
  expect(state.onSubmit.mock.calls[0][0].message).not.toBe(turn.input_text)
  expect(state.onSubmit.mock.calls[0][0].requestOverrides.tldwTurn.user_message_id).not.toBe(uuid)
  expect(turn).toEqual(prior)
})

it("CP-F2 direct typing retires the recovery UUID", async () => {
  render(<WorkspaceChatPanel {...props} />)
  fireEvent.click(screen.getByRole("button", { name: "Explicit reprepare" }))
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
  fireEvent.click(screen.getByRole("button", { name: "Explicit reprepare" }))
  fireEvent.click(screen.getByRole("button", { name: "Send message" }))
  await waitFor(() => expect(state.onSubmit).toHaveBeenCalledTimes(1))
  expect(state.onSubmit.mock.calls[0][0].message).toContain("Context sources:")
  expect(state.onSubmit.mock.calls[0][0].requestOverrides.tldwTurn.user_message_id).not.toBe(uuid)
  expect(turn).toEqual(prior)
})

it("CP-F2 final user-text trimming cannot reuse a different recovery input", async () => {
  turn.input_text = "  Original recovered question  "
  render(<WorkspaceChatPanel {...props} />)
  fireEvent.click(screen.getByRole("button", { name: "Explicit reprepare" }))
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
  fireEvent.click(screen.getByRole("button", { name: "Explicit reprepare" }))
  fireEvent.click(screen.getByRole("button", { name: "Send message" }))
  await waitFor(() => expect(state.onSubmit).toHaveBeenCalledTimes(1))
  expect(state.onSubmit.mock.calls[0][0].message).toBe(turn.input_text)
  expect(state.onSubmit.mock.calls[0][0].requestOverrides.tldwTurn.user_message_id).toBe(uuid)
  expect(turn).toEqual(prior)
})
