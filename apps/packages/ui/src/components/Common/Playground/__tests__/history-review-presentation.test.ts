import { describe, expect, it } from "vitest"

import {
  chronologicalHistoryNodes,
  hasReviewableHistory,
  partialHistoryCoverage,
  plainHistoryPreview
} from "../history-review-presentation"
import type {
  HistoryCursorV1,
  HistoryNodeV1,
  HistorySelectionCaptureV1,
  HistoryViewSelectionV1
} from "@/types/history-selection"
import { resolveParentPath } from "@/utils/history-selection"

const node = (
  id: string,
  role: "user" | "assistant",
  parent_id: string | null,
  preview = id
): HistoryNodeV1 => ({ id, revision: "1", role, parent_id, settled: true, preview })

const linear = [
  node("u1", "user", null),
  node("a1", "assistant", "u1"),
  node("u2", "user", "a1"),
  node("a2", "assistant", "u2")
]

const captureAt = (nodes: HistoryNodeV1[], cursor: HistoryCursorV1) => {
  const view: HistoryViewSelectionV1 = {
    view_session_id: "view",
    owner_key: "owner",
    conversation_id: "chat",
    interpretation: { kind: "parent_graph_v1" },
    cursor,
    selection_revision: 1
  }
  const capture = {
    status: "captured",
    view,
    purpose: "send",
    storage_context_digest: "storage",
    rows: resolveParentPath(nodes, cursor),
    selected_content: [],
    snapshot: {
      version: 1,
      owner_key: "owner",
      conversation_id: "chat",
      fences: { conversation: "1", history: "1", settings: "1" },
      nodes,
      source_digest: "source",
      storage_context_digest: "storage",
      interpretation_status: { kind: "parent_graph_v1" }
    }
  } as unknown as HistorySelectionCaptureV1
  return { capture, view }
}

describe("history review presentation (CC-04)", () => {
  it("lists a reply after the question it answers, keeping owner order otherwise", () => {
    const shuffled = [linear[1], linear[0], linear[3], linear[2]]
    expect(chronologicalHistoryNodes(shuffled).map((row) => row.id)).toEqual([
      "u1",
      "a1",
      "u2",
      "a2"
    ])
    const roots = [node("r2", "user", null), node("r1", "user", null)]
    expect(chronologicalHistoryNodes(roots).map((row) => row.id)).toEqual(["r2", "r1"])
  })

  it("orders a very long chain listed newest first without overflowing", () => {
    const chain = Array.from({ length: 50_000 }, (_, index) =>
      node(`n${index}`, index % 2 ? "assistant" : "user", index ? `n${index - 1}` : null))
    const ordered = chronologicalHistoryNodes([...chain].reverse())
    expect(ordered[0].id).toBe("n0")
    expect(ordered.at(-1)?.id).toBe("n49999")
  })

  it("survives a parent cycle without dropping messages", () => {
    const cyclic = [node("x", "user", "y"), node("y", "assistant", "x")]
    expect(chronologicalHistoryNodes(cyclic).map((row) => row.id).sort()).toEqual(["x", "y"])
  })

  it("turns markdown previews into plain text", () => {
    expect(plainHistoryPreview("## Title\n\n**Bold** and _soft_ `code` [link](https://x.test)")).toBe(
      "Title Bold and soft code link"
    )
    expect(plainHistoryPreview("- item one\n> quoted\n```js\nconst a = 1\n```")).toBe(
      "item one quoted const a = 1"
    )
    expect(plainHistoryPreview("snake_case_name stays")).toBe("snake_case_name stays")
    expect(plainHistoryPreview("Use List<String> here<br/>next")).toBe("Use List<String> here next")
    expect(plainHistoryPreview(null)).toBe("")
  })

  it("needs two turns or alternative replies before history is worth choosing", () => {
    expect(hasReviewableHistory(linear.slice(0, 2))).toBe(false)
    expect(hasReviewableHistory(linear)).toBe(true)
    expect(
      hasReviewableHistory([...linear.slice(0, 2), node("a1b", "assistant", "u1")])
    ).toBe(true)
  })

  it("reports how much the AI sees when only the start of the conversation is chosen", () => {
    const empty = captureAt(linear, { kind: "empty" })
    expect(partialHistoryCoverage(empty.capture, empty.view)).toEqual({
      seen: 0,
      total: 4,
      restore: { kind: "after_message", message_id: "a2" },
      hidden: linear
    })
    const prefix = captureAt(linear, { kind: "after_message", message_id: "a1" })
    expect(partialHistoryCoverage(prefix.capture, prefix.view)).toMatchObject({
      seen: 2,
      total: 4
    })
  })

  it("reports nothing when the AI sees everything or follows another branch", () => {
    const full = captureAt(linear, { kind: "after_message", message_id: "a2" })
    expect(partialHistoryCoverage(full.capture, full.view)).toBeNull()
    const branched = [...linear, node("a1b", "assistant", "u1")]
    // The latest message is the alternative reply; the view follows the original branch.
    const other = captureAt(branched, { kind: "after_message", message_id: "a2" })
    expect(partialHistoryCoverage(other.capture, other.view)).toBeNull()
    // A capture of an older view is never described as the current choice.
    expect(partialHistoryCoverage(full.capture, { ...full.view })).toBeNull()
  })
})
