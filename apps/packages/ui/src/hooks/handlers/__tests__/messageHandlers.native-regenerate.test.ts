import type { HistorySelectionController } from "@/hooks/chat/useHistorySelection"
import type { Message } from "@/store/option"
import { useStoreMessageOption } from "@/store/option"
import type { HistorySelectionCaptureV1 } from "@/types/history-selection"
import { afterEach, describe, expect, it, vi } from "vitest"

import { createRegenerateLastMessage } from "../messageHandlers"

const fixture = () => {
  const rows = [
    {
      id: "u",
      revision: "opaque-u",
      parent_id: null,
      role: "user",
      settled: true
    },
    {
      id: "a",
      revision: "opaque-a",
      parent_id: "u",
      role: "assistant",
      settled: true
    }
  ]
  const view = {
    owner_key: "owner",
    conversation_id: "chat",
    view_session_id: "view",
    selection_revision: 7,
    interpretation: { kind: "parent_graph_v1" as const },
    cursor: { kind: "after_message" as const, message_id: "a" }
  }
  const capture: HistorySelectionCaptureV1 = {
    status: "captured",
    view,
    purpose: "send",
    storage_context_digest: "storage",
    snapshot: {
      version: 1,
      owner_key: "owner",
      conversation_id: "chat",
      nodes: rows,
      source_digest: "source",
      storage_context_digest: "storage",
      fences: { conversation: "1", history: "1", settings: "1" },
      interpretation_status: { kind: "parent_graph_v1" }
    },
    rows,
    selected_content: rows.map((row) => ({
      id: row.id,
      revision: row.revision,
      message:
        row.role === "user"
          ? "  Exact {{char}} question  \n"
          : "Original reply",
      images: []
    }))
  }
  const state = {
    status: "ready",
    owner: {
      kind: "native",
      conversation_id: "chat",
      owner_key: "owner",
      validate_lease: () => true
    },
    view,
    capture
  }
  const messages: Message[] = capture.selected_content.map((row, index) => ({
    id: row.id,
    name: index ? "Assistant" : "You",
    isBot: Boolean(index),
    message: row.message,
    images: [],
    sources: [],
    parentMessageId: index ? "u" : null
  }))
  let viewCurrent = true,
    sourceCurrent = true
  const onSubmit = vi.fn(async () => ({ status: "submitted" as const }))
  const options = {
    historySelection: {
      getCurrent: () => state
    } as unknown as HistorySelectionController,
    messages,
    getMessages: () => messages,
    history: [{ role: "user" as const, content: "UNRELATED ARRAY QUESTION" }],
    setMessages: vi.fn(),
    setHistory: vi.fn(),
    beforeSubmit: vi.fn(),
    onSubmit,
    validateBeforeSubmitFn: vi.fn(() => true),
    captureViewFence: () => () => viewCurrent,
    captureSourceFence: () => () => sourceCurrent
  }
  return {
    state,
    messages,
    options,
    regenerate: createRegenerateLastMessage(options),
    retire: (kind: string) => {
      if (kind === "view") viewCurrent = false
      else sourceCurrent = false
    }
  }
}
afterEach(() =>
  useStoreMessageOption.setState({ streaming: false, isProcessing: false })
)

describe("native Regenerate action intent", () => {
  it("uses canonical stable identities without truncating originals or consulting legacy history", async () => {
    const f = fixture(),
      before = structuredClone(f.messages),
      origin = f.state.view
    expect(await f.regenerate()).toEqual({ status: "submitted" })
    expect(f.options.onSubmit).toHaveBeenCalledWith(
      expect.objectContaining({
        message: "  Exact {{char}} question  \n",
        image: "",
        isRegenerate: true,
        historyRegenerate: {
          owner_key: "owner",
          conversation_id: "chat",
          message_id: "a",
          message_revision: "opaque-a",
          input_message_id: "u",
          input_message_revision: "opaque-u",
          isCurrent: expect.any(Function),
          isOriginCurrent: expect.any(Function)
        }
      })
    )
    expect(f.messages).toEqual(before)
    expect(f.state.view).toBe(origin)
    expect(f.options.setMessages).not.toHaveBeenCalled()
    expect(f.options.setHistory).not.toHaveBeenCalled()
    expect(f.options.beforeSubmit).not.toHaveBeenCalled()
  })
  it.each(["view", "source"])(
    "a retained action refuses a retired %s before dispatch",
    async (boundary) => {
      const f = fixture()
      f.retire(boundary)
      expect(await f.regenerate()).toEqual({
        status: "skipped",
        reason: "stale_selection"
      })
      expect(f.options.onSubmit).not.toHaveBeenCalled()
    }
  )
  it.each(["streaming", "isProcessing"])(
    "reads live %s before replacing an active response",
    async (flag) => {
      const f = fixture()
      useStoreMessageOption.setState({ [flag]: true })
      expect(await f.regenerate()).toEqual({
        status: "skipped",
        reason: "response_in_progress"
      })
      expect(f.options.onSubmit).not.toHaveBeenCalled()
    }
  )
  it.each(["reply", "input", "append"])(
    "refuses a changed live %s without setters or dispatch",
    async (boundary) => {
      const f = fixture()
      if (boundary === "append")
        f.messages.push({
          id: "new",
          name: "You",
          isBot: false,
          message: "New turn"
        })
      else f.messages[boundary === "reply" ? 1 : 0].message = "Changed row"
      expect(await f.regenerate()).toMatchObject({
        status: "skipped",
        reason: "message_changed"
      })
      expect(f.options.onSubmit).not.toHaveBeenCalled()
      expect(f.options.setMessages).not.toHaveBeenCalled()
    }
  )
  it("keeps the captured identity fence for result following after the mounted H1 fence retires", async () => {
    const f = fixture()
    await f.regenerate()
    const action = f.options.onSubmit.mock.calls[0][0] as unknown as {
      historyRegenerate: {
        isCurrent: () => boolean
        isOriginCurrent: () => boolean
      }
    }
    f.retire("view")
    expect(action.historyRegenerate.isOriginCurrent()).toBe(false)
    expect(action.historyRegenerate.isCurrent()).toBe(true)
    f.retire("source")
    expect(action.historyRegenerate.isCurrent()).toBe(false)
  })
})
