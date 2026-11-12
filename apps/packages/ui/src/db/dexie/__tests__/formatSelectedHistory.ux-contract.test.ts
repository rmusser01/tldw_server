/**
 * UX review 2026-10 contract reproduction XP-02 (#3109).
 *
 * The test asserts the CORRECT behaviour. It started as an `it.fails` marker
 * while the defect existed and is now a plain `it(...)` regression test.
 */
import { describe, expect, it, vi } from "vitest"
import type {
  HistoryNodeV1,
  HistorySelectionCaptureV1
} from "@/types/history-selection"

vi.mock("@/db/dexie/chat", () => ({ PageAssistDatabase: class {} }))

import { formatSelectedHistory } from "../helpers"

// Shape of native_history_owner_key() in tldw_Server_API/app/core/Chat/persistence_service.py.
const NATIVE_OWNER_KEY = `native-history-v1:sha256:${"a".repeat(64)}`
const SERVER_CHAT_ID = "server-chat-1"

const node = (
  id: string,
  parent_id: string | null,
  role: string,
  revision: string
): HistoryNodeV1 => ({
  id,
  revision,
  parent_id,
  role,
  settled: true,
  metadata: [],
  assets: []
})

/** A capture as returned by POST /api/v1/chat/conversations/{id}/history/selection (endpoints/chat.py:8249). */
const buildNativeServerCapture = (): HistorySelectionCaptureV1 => {
  const nodes = [
    node("srv-msg-user-1", null, "user", "1"),
    node("srv-msg-assistant-1", "srv-msg-user-1", "assistant", "1")
  ]
  return {
    status: "captured",
    snapshot: {
      version: 1,
      owner_key: NATIVE_OWNER_KEY,
      conversation_id: SERVER_CHAT_ID,
      fences: { conversation: "c1", history: "h1", settings: "s1" },
      nodes,
      source_digest: "source-digest",
      interpretation_status: { kind: "parent_graph_v1" },
      storage_context_digest: "storage-digest"
    },
    rows: nodes,
    // Server selected_content carries the stored message extra (often null), never local_history.
    selected_content: [
      { id: "srv-msg-user-1", revision: "1", message: "What is RAG?", images: [], tool_calls: null, extra_metadata: null },
      { id: "srv-msg-assistant-1", revision: "1", message: "Retrieval-augmented generation.", images: [], tool_calls: null, extra_metadata: null }
    ],
    view: {
      view_session_id: "view-1",
      owner_key: NATIVE_OWNER_KEY,
      conversation_id: SERVER_CHAT_ID,
      interpretation: { kind: "parent_graph_v1" },
      cursor: { kind: "after_message", message_id: "srv-msg-assistant-1" },
      selection_revision: 1
    },
    purpose: "send",
    storage_context_digest: "storage-digest"
  }
}

describe("formatSelectedHistory UX contract reproductions (#3109)", () => {
  // XP-02 (#3109): formatSelectedHistory built rows only from metadata.local_history (never has serverMessageId), so formatToMessage dropped it.
  it("XP-02 (#3109): messages loaded from a server chat keep their serverMessageId", () => {
    const display = formatSelectedHistory(buildNativeServerCapture())

    expect(
      display.messages.map((message) => [message.id, message.serverMessageId])
    ).toEqual([
      ["srv-msg-user-1", "srv-msg-user-1"],
      ["srv-msg-assistant-1", "srv-msg-assistant-1"]
    ])
  })
})


describe("B02 canonical partial reply display", () => {
  it.each(["stopped", "interrupted"])("keeps %s native assistant status after selected-history reload", (status) => {
    const capture = buildNativeServerCapture()
    capture.selected_content[1].extra_metadata = { generation_status: status, sender_name: "TestBot", provider: "llama.cpp", local_history: { generationInfo: { custom: "retained", stopped: true } } }
    const before = JSON.stringify(capture)
    const display = formatSelectedHistory(capture)

    expect(display.messages[1].generationInfo).toEqual({ interrupted: true, stopped: status === "stopped" })
    expect(display.messages[1].serverMessageId).toBe("srv-msg-assistant-1")
    expect(display.messages[1].metadataExtra).toEqual(capture.selected_content[1].extra_metadata)
    expect(JSON.stringify(capture)).toBe(before)
    expect(display.history[1]).toMatchObject({ role: "assistant", content: "Retrieval-augmented generation." })
  })

  it.each(["complete", "length", "unknown", undefined])("does not invent an interruption for native status %s", (status) => {
    const capture = buildNativeServerCapture()
    capture.selected_content[1].extra_metadata = status ? { generation_status: status } : null
    expect(formatSelectedHistory(capture).messages[1].generationInfo).toBeUndefined()
  })

  it.each(["user", "system"])("does not interpret %s metadata as an assistant interruption", (role) => {
    const capture = buildNativeServerCapture()
    capture.rows[0].role = role
    capture.selected_content[0].extra_metadata = { generation_status: "stopped" }
    expect(formatSelectedHistory(capture).messages[0].generationInfo).toBeUndefined()
  })

  it("retains local generation info without treating server-shaped metadata as authority", () => {
    const capture = buildNativeServerCapture()
    capture.snapshot.owner_key = "local-history-v1:profile"
    capture.selected_content[1].extra_metadata = { generation_status: "stopped", local_history: { generationInfo: { custom: "retained" } } }
    expect(formatSelectedHistory(capture).messages[1].generationInfo).toEqual({ custom: "retained" })
  })
})

const namedSenderCapture = (
  role: string,
  extra: Record<string, unknown> | null = null,
  alternatives: HistoryNodeV1[] = [],
  ownerKey = NATIVE_OWNER_KEY
): HistorySelectionCaptureV1 => {
  const capture = buildNativeServerCapture()
  const rows = capture.rows.map((row, index) => index === 1 ? { ...row, role } : row)
  return {
    ...capture,
    rows,
    snapshot: { ...capture.snapshot, owner_key: ownerKey, nodes: [...rows, ...alternatives] },
    selected_content: capture.selected_content.map((row, index) => index === 1 ? { ...row, extra_metadata: extra } : row)
  }
}

describe("X01 named sender display after upgrade", () => {
  it("renders a stored Character name as an assistant without rewriting capture or provider history", () => {
    const capture = namedSenderCapture("X01 Rowan Archivist", { generation_status: "stopped" })
    const before = JSON.stringify(capture)
    const display = formatSelectedHistory(capture)

    expect(display.messages[1]).toMatchObject({ isBot: true, role: "assistant", name: "Assistant", generationInfo: { interrupted: true, stopped: true } })
    expect(display.history[1]).toMatchObject({ role: "user", content: "Retrieval-augmented generation." })
    expect(JSON.stringify(capture)).toBe(before)
  })

  it.each(["user", "system", "assistant"])("honors explicit %s metadata on a named sender", role => {
    const capture = namedSenderCapture("Saved display name", { sender_role: role, sender_name: "Saved display name" })
    const before = JSON.stringify(capture)

    expect(formatSelectedHistory(capture).messages[1]).toMatchObject({ role, isBot: role === "assistant", name: "Saved display name" })
    expect(JSON.stringify(capture)).toBe(before)
  })

  it("keeps named assistant alternatives and excludes tool output from the alternative list", () => {
    const capture = namedSenderCapture("X01 Rowan Archivist", null, [
      node("alternate", "srv-msg-user-1", "Old Character name", "2"),
      node("tool-result", "srv-msg-user-1", "tool", "3")
    ])
    const before = JSON.stringify(capture)

    expect(formatSelectedHistory(capture).messages[1].variants?.map(item => item.id)).toEqual(["srv-msg-assistant-1", "alternate"])
    expect(JSON.stringify(capture)).toBe(before)
  })

  it.each(["tool", "function"])("keeps an explicitly tagged %s out of assistant alternatives and partial status", role => {
    const capture = namedSenderCapture("assistant", { sender_role: role, generation_status: "stopped" }, [node("alternate", "srv-msg-user-1", "assistant", "2")])
    const display = formatSelectedHistory(capture)
    expect(display.messages[1].generationInfo).toBeUndefined()
    expect(display.messages[1].variants).toBeUndefined()
    expect(display.history[1].role).toBe("assistant")
  })

  it.each(["tool", "function"])("does not group a selected %s sibling with the selected assistant", role => {
    const base = buildNativeServerCapture()
    const tool = node("selected-tool", "srv-msg-user-1", "assistant", "2")
    const capture: HistorySelectionCaptureV1 = {
      ...base,
      rows: [...base.rows, tool],
      snapshot: { ...base.snapshot, nodes: [...base.snapshot.nodes, tool] },
      selected_content: [...base.selected_content, { id: tool.id, revision: tool.revision, message: "Tool output", images: [], extra_metadata: { sender_role: role } }]
    }
    const before = JSON.stringify(capture)
    const display = formatSelectedHistory(capture)
    expect(display.messages.slice(1).map(message => [message.variants, message.activeVariantIndex])).toEqual([[undefined, undefined], [undefined, undefined]])
    expect(JSON.stringify(capture)).toBe(before)
  })

  it.each([
    ["user", "user"], ["human", "user"], ["system", "system"],
    ["tool", "assistant"], ["function", "assistant"], ["tool:read", "user"],
    ["", "user"], ["   ", "user"]
  ])("retains the existing %s display convention", (role, expected) => {
    expect(formatSelectedHistory(namedSenderCapture(role)).messages[1]).toMatchObject({ role: expected, isBot: expected === "assistant" })
  })

  it("keeps an unknown local sender on the existing user fallback", () => {
    const capture = namedSenderCapture("Local display name", null, [], "local-history-v1:profile")
    expect(formatSelectedHistory(capture).messages[1]).toMatchObject({ role: "user", isBot: false })
  })
})
