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
