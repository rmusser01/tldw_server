import { describe, expect, it, vi } from "vitest"
vi.mock("@/db/dexie/chat", () => ({ PageAssistDatabase: class {} }))
import { formatToChatHistory, formatToMessage } from "../helpers"
import type { Message } from "../types"
import type { HistorySelectionCaptureV1 } from "@/types/history-selection"
const rows: Message[] = [
  {
    id: "a",
    history_id: "h",
    role: "assistant",
    name: "Assistant",
    content: "Same",
    createdAt: 3,
    parent_message_id: "u"
  },
  {
    id: "b",
    history_id: "h",
    role: "assistant",
    name: "Assistant",
    content: "Same",
    createdAt: 2,
    parent_message_id: "u"
  },
  {
    id: "u",
    history_id: "h",
    role: "user",
    name: "You",
    content: "Question",
    createdAt: 9
  }
]
describe("explicit selected formatting", () => {
  it("preserves supplied IDs/order without timestamp collapse or text deduplication", () => {
    const formatted = formatToMessage(rows, ["u", "b", "a"])
    expect(formatted.map((row) => row.id)).toEqual(["u", "b", "a"])
    expect(
      formatToChatHistory(rows, ["u", "b", "a"]).map((row) => row.content)
    ).toEqual(["Question", "Same", "Same"])
  })
  it("preserves an explicit empty path and rejects missing or duplicate IDs", () => {
    expect(formatToMessage(rows, [])).toEqual([])
    expect(() => formatToMessage(rows, ["missing"])).toThrow("stale_selection")
    expect(() => formatToChatHistory(rows, ["u", "u"])).toThrow(
      "duplicate_message_id"
    )
  })
})

import { formatSelectedHistory } from "../helpers"
it("renders selected content while retaining unloaded stable alternatives and canonical roles", () => {
  const capture: any = {
    status: "captured",
    view: { conversation_id: "h" },
    snapshot: {
      nodes: [
        { id: "u", parent_id: null, role: "user", preview: "Question" },
        { id: "a", parent_id: "u", role: "assistant", preview: "First" },
        { id: "b", parent_id: "u", role: "assistant", preview: "Other" },
        { id: "t", parent_id: "a", role: "tool", preview: "Tool output" }
      ]
    },
    rows: [
      { id: "u", parent_id: null, role: "user" },
      { id: "a", parent_id: "u", role: "assistant" },
      { id: "t", parent_id: "a", role: "tool" }
    ],
    selected_content: [
      { id: "u", message: "Question", images: [] },
      { id: "a", message: "Full first", images: [] },
      {
        id: "t",
        message: "Tool output",
        images: [],
        tool_calls: [{ id: "call" }]
      }
    ]
  }
  const display = formatSelectedHistory(capture)
  expect(display.messages.map((row) => row.id)).toEqual(["u", "a", "t"])
  expect(display.messages[1].variants?.map((row) => row.id)).toEqual(["a", "b"])
  expect(display.messages[1].message).toBe("Full first")
  expect(capture.rows[2].role).toBe("tool")
})

const ownedCapture = (ownerKey: string): HistorySelectionCaptureV1 => {
  const rows = [
    { id: "u", revision: "digest-u", parent_id: null, role: "user", settled: true },
    { id: "a", revision: "digest-a", parent_id: "u", role: "assistant", settled: true }
  ]
  return {
    status: "captured",
    snapshot: {
      version: 1,
      owner_key: ownerKey,
      conversation_id: "h",
      fences: { conversation: "c", history: "h", settings: "s" },
      nodes: rows,
      source_digest: "source",
      interpretation_status: { kind: "parent_graph_v1" },
      storage_context_digest: "storage"
    },
    rows,
    selected_content: [
      { id: "u", revision: "digest-u", message: "Question", images: [], extra_metadata: { local_history: {} } },
      { id: "a", revision: "digest-a", message: "Answer", images: [], extra_metadata: { local_history: { modelId: "m" } } }
    ],
    view: {
      view_session_id: "view",
      owner_key: ownerKey,
      conversation_id: "h",
      interpretation: { kind: "parent_graph_v1" },
      cursor: { kind: "after_message", message_id: "a" },
      selection_revision: 1
    },
    purpose: "send",
    storage_context_digest: "storage"
  }
}

it("leaves serverMessageId unset on a locally owned capture, whose ids are local rows", () => {
  const display = formatSelectedHistory(ownedCapture("local-history-v1:profile"))

  expect(display.messages.map((row) => row.serverMessageId)).toEqual([undefined, undefined])
})

it("does not read a server message version from the node revision digest", () => {
  const display = formatSelectedHistory(ownedCapture(`native-history-v1:sha256:${"c".repeat(64)}`))

  expect(display.messages.map((row) => row.serverMessageId)).toEqual(["u", "a"])
  expect(display.messages.map((row) => row.serverMessageVersion)).toEqual([undefined, undefined])
})
