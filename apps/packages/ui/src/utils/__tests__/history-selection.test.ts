import { describe, expect, it } from "vitest"
import { resolveLegacyProjection, bindSelectedHistoryContent, resolveParentPath, resolveHistorySelection, selectionDigest, canonicalSelectionJson, resolveComparisonProjection, canonicalComparisonJson, comparisonDigest } from "../history-selection"
import type { HistoryNodeV1, HistorySelectedContentV1, HistorySelectionSnapshotV1, HistoryViewSelectionV1 } from "@/types/history-selection"
import vectors from "../../../../../../tldw_Server_API/tests/Chat/fixtures/history_selection_v1.json"

const rows: HistoryNodeV1[] = [
  { id: "u1", revision: "1", parent_id: null, role: "user", settled: true },
  { id: "a1", revision: "1", parent_id: "u1", role: "assistant", settled: true },
  { id: "a2", revision: "1", parent_id: "u1", role: "assistant", settled: true },
  { id: "u2", revision: "1", parent_id: "a2", role: "user", settled: true }
]

describe("parent graph resolution", () => {
  it("retains the chosen stable branch and an empty before-root boundary", () => {
    expect(resolveParentPath(rows, { kind: "after_message", message_id: "a1" }).map(r => r.id)).toEqual(["u1", "a1"])
    expect(resolveParentPath(rows, { kind: "before_message", message_id: "u1" })).toEqual([])
  })
  it("does not splice descendants from a sibling even when answer text is equal", () => {
    expect(resolveParentPath(rows, { kind: "after_message", message_id: "u2" }).map(r => r.id)).toEqual(["u1", "a2", "u2"])
    expect(resolveParentPath(rows, { kind: "empty" })).toEqual([])
  })
  it.each([
    [[...rows, rows[0]], "duplicate_message_id"],
    [[{ ...rows[0], parent_id: "missing" }, ...rows.slice(1)], "missing_parent"],
    [[{ ...rows[0], parent_id: "a1" }, ...rows.slice(1)], "cyclic_ancestry"]
  ] as const)("rejects invalid graph with %s", (source, code) => {
    expect(() => resolveParentPath(source, { kind: "after_message", message_id: "a1" })).toThrow(expect.objectContaining({ code }))
  })
  it("rejects a parent in another conversation", () => {
    const source = [
      { ...rows[0], conversation_id: "c1" },
      { ...rows[1], conversation_id: "c2" }
    ]
    expect(() => resolveParentPath(source, { kind: "after_message", message_id: "a1" })).toThrow(expect.objectContaining({ code: "cross_conversation_parent" }))
  })
  it("rejects an orphan outside the chosen path before admitting a branch", () => {
    const source = [...rows, { ...rows[1], id: "orphan", parent_id: "missing" }]
    expect(() => resolveParentPath(source, { kind: "after_message", message_id: "a1" })).toThrow(expect.objectContaining({ code: "missing_parent" }))
  })
})

it("returns a structured stale result for a missing cursor, without an admission", () => {
  const snapshot: HistorySelectionSnapshotV1 = {
    version: 1, owner_key: "local:p", conversation_id: "c", fences: { conversation: "1", history: "1", settings: "1" },
    source_digest: "source", nodes: rows, interpretation_status: { kind: "parent_graph_v1" }, storage_context_digest: "storage"
  }
  const view: HistoryViewSelectionV1 = {
    view_session_id: "v", owner_key: "local:p", conversation_id: "c", interpretation: { kind: "parent_graph_v1" },
    cursor: { kind: "after_message", message_id: "gone" }, selection_revision: 3
  }
  expect(resolveHistorySelection(snapshot, view, "send", "request")).toEqual({ status: "stale_selection", code: "missing_cursor" })
})

it("binds captured text and multiple images to exact selected IDs, order and revisions", () => {
  const path = resolveParentPath(rows, { kind: "after_message", message_id: "a1" })
  const content = [
    { id: "u1", revision: "1", message: "see both", images: ["image://1", "image://2"] },
    { id: "a1", revision: "1", message: "response", images: [] }
  ] satisfies HistorySelectedContentV1[]
  const bound = bindSelectedHistoryContent(path, content)
  expect(bound).toEqual(content)
  content[0].images.push("image://later")
  content[0].message = "later"
  expect(bound[0]).toEqual({ id: "u1", revision: "1", message: "see both", images: ["image://1", "image://2"] })
  for (const drift of [
    [...content].reverse(),
    [{ ...content[0], revision: "2" }, content[1]],
    [{ ...content[0], id: "a2" }, content[1]],
    content.slice(0, 1)
  ]) {
    expect(() => bindSelectedHistoryContent(path, drift)).toThrow(expect.objectContaining({ code: "selected_content_mismatch" }))
  }
  expect(() => bindSelectedHistoryContent([rows[0], rows[0]], [content[0], content[0]])).toThrow(expect.objectContaining({ code: "selected_content_mismatch" }))
})

it("encodes the fixed Unicode selection tuple without normalization", () => {
  expect(selectionDigest({
    version: 1, owner_key: "local:é", conversation_id: "会話", interpretation: { kind: "parent_graph_v1" },
    cursor: { kind: "after_message", message_id: "a🌿" }, selection_revision: 7, purpose: "send",
    messages: [{ id: "a🌿", revision: "rev:画像#1" }], fences: { conversation: "c", history: "h", settings: "s" },
    storage_context_digest: "storage:ref-42", request_context_digest: "request:overlay-9", selection_digest: ""
  })).toBe("4ab831170088e3c5018d9257a7181c44db775d6fee1e64ed94d038360e11abad")
})

it("matches all shared normal digest vectors", () => {
  for (const vector of vectors.selection_vectors) {
    const selection = vector.selection as Parameters<typeof selectionDigest>[0]
    expect(canonicalSelectionJson(selection)).toBe(vector.canonical_json)
    expect(selectionDigest(selection)).toBe(vector.selection_digest)
  }
})

it("retains A's semantic comparison order despite stored cross-model edges", () => {
  const comparisonRows = [
    { ...rows[0], id: "u1", parent_id: null, comparison: { cluster_id: "r1", model_id: null, common: true } },
    { ...rows[1], id: "a1", parent_id: "u1", comparison: { cluster_id: "r1", model_id: "A", common: false } },
    { ...rows[2], id: "b1", parent_id: "u1", comparison: { cluster_id: "r1", model_id: "B", common: false } },
    { ...rows[0], id: "u2", parent_id: "b1", comparison: { cluster_id: "r2", model_id: null, common: true } },
    { ...rows[1], id: "a2", parent_id: "u2", comparison: { cluster_id: "r2", model_id: "A", common: false } },
    { ...rows[2], id: "b2", parent_id: "u2", comparison: { cluster_id: "r2", model_id: "B", common: false } },
    { ...rows[0], id: "uA", parent_id: "b2", comparison: { cluster_id: "r3", model_id: "A", common: false } }
  ]
  expect(resolveComparisonProjection(comparisonRows, "A", "uA").map(row => row.id)).toEqual(["u1", "a1", "u2", "a2", "uA"])
  const vector = vectors.comparison_vector
  expect(resolveComparisonProjection(vector.source_rows as HistoryNodeV1[], vector.model_id, vector.boundary_id).map(row => row.id)).toEqual(vector.expected_path_ids)
  expect(canonicalComparisonJson(vector.selection as Parameters<typeof comparisonDigest>[0])).toBe(vector.canonical_json)
  expect(comparisonDigest(vector.selection as Parameters<typeof comparisonDigest>[0])).toBe(vector.selection_digest)
})


it("binds new legacy descendants to only their protected base, including null branches", () => {
  const node = (id: string, parent_id: string | null, legacy_projection_id?: string): HistoryNodeV1 =>
    ({ id, parent_id, legacy_projection_id, revision: "1", role: "user", settled: true })
  const source = [node("a", null), node("b", null), node("shared", null),
    node("x", "shared", "first"), node("y", "shared", "second"), node("fresh", null, "first")]
  const path = (base: string[], target: string, projection: string) =>
    resolveLegacyProjection(source, base, { kind: "after_message", message_id: target }, projection).map(row => row.id)
  expect(path(["a", "shared"], "x", "first")).toEqual(["a", "shared", "x"])
  expect(path(["b", "shared"], "y", "second")).toEqual(["b", "shared", "y"])
  expect(path(["a", "shared"], "fresh", "first")).toEqual(["fresh"])
  expect(path(["a", "shared"], "shared", "first")).toEqual(["a", "shared"])
  expect(() => path(["a", "shared"], "y", "first")).toThrow(expect.objectContaining({code: "interpretation_mismatch"}))
})
