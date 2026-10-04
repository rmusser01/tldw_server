import { describe, expect, it, vi } from "vitest"
vi.mock("@/db/dexie/chat", () => ({ PageAssistDatabase: class {} }))
import { formatToChatHistory, formatToMessage } from "../helpers"
import type { Message } from "../types"
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

it("restores ordered bounded source presentation without treating public metadata as a receipt", () => {
  const source = { name: "Memo", type: "text", mode: "rag", url: "memo.md", pageContent: "Evidence", metadata: { chunk_id: "c1" } }
  const capture: any = {
    status: "captured", view: { conversation_id: "h" },
    snapshot: { nodes: [{ id: "a", parent_id: "u", role: "assistant" }] },
    rows: [{ id: "a", parent_id: "u", role: "assistant" }],
    selected_content: [{ id: "a", message: "Answer [0]", images: [], extra_metadata: {
      sender_role: "assistant", history_result_v1: { version: 1, request_context_digest: "a".repeat(64), sources: [source] }
    } }]
  }
  expect(formatSelectedHistory(capture).messages[0].sources).toEqual([source])
  expect(formatSelectedHistory(capture).messages[0].metadataExtra).not.toHaveProperty("tldw_history_recovery_v1")
  capture.selected_content[0].extra_metadata.history_result_v1.sources[0].metadata.headers = { authorization: "not-a-display-field" }
  expect(() => formatSelectedHistory(capture)).toThrow("invalid_history_durable_result")
})
