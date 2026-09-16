import { sha256 } from "@noble/hashes/sha2.js"
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js"
import type {
  CompareHistorySelectionV1, HistoryCursorV1, HistoryNodeV1, HistoryResolutionV1,
  HistorySelectedContentV1, HistorySelectionSnapshotV1, HistorySelectionV1, HistoryViewSelectionV1
} from "@/types/history-selection"

export class HistorySelectionError extends Error {
  constructor(readonly code: string) {
    super(code)
    this.name = "HistorySelectionError"
  }
}

const indexNodes = (nodes: readonly HistoryNodeV1[], validateGraph = false): Map<string, HistoryNodeV1> => {
  const byId = new Map<string, HistoryNodeV1>()
  let conversationId: string | undefined
  for (const node of nodes) {
    if (!node.id || byId.has(node.id)) throw new HistorySelectionError("duplicate_message_id")
    if (node.conversation_id) {
      if (conversationId && conversationId !== node.conversation_id) throw new HistorySelectionError("cross_conversation_parent")
      conversationId = node.conversation_id
    }
    byId.set(node.id, node)
  }
  if (validateGraph) {
    const resolved = new Set<string>()
    for (const node of nodes) {
      const walking = new Set<string>()
      let current: HistoryNodeV1 | undefined = node
      while (current && !resolved.has(current.id)) {
        if (walking.has(current.id)) throw new HistorySelectionError("cyclic_ancestry")
        walking.add(current.id)
        if (current.parent_id === null) break
        const parent: HistoryNodeV1 | undefined = byId.get(current.parent_id)
        if (!parent) throw new HistorySelectionError("missing_parent")
        if (current.conversation_id && parent.conversation_id !== current.conversation_id) {
          throw new HistorySelectionError("cross_conversation_parent")
        }
        current = parent
      }
      for (const id of walking) resolved.add(id)
    }
  }
  return byId
}

/** Resolve only explicit parent IDs. Source order, content and timestamps have no authority. */
export const resolveParentPath = (
  nodes: readonly HistoryNodeV1[], cursor: HistoryCursorV1
): readonly HistoryNodeV1[] => {
  const byId = indexNodes(nodes, true)
  if (cursor.kind === "empty") return []
  const selected = byId.get(cursor.message_id)
  if (!selected) throw new HistorySelectionError("missing_cursor")
  const path: HistoryNodeV1[] = []
  const visited = new Set<string>()
  let current: HistoryNodeV1 | undefined = selected
  while (current) {
    if (visited.has(current.id)) throw new HistorySelectionError("cyclic_ancestry")
    visited.add(current.id)
    path.push(current)
    if (current.parent_id === null) break
    const parent: HistoryNodeV1 | undefined = byId.get(current.parent_id)
    if (!parent) throw new HistorySelectionError("missing_parent")
    if (current.conversation_id && parent.conversation_id !== current.conversation_id) {
      throw new HistorySelectionError("cross_conversation_parent")
    }
    current = parent
  }
  path.reverse()
  return cursor.kind === "before_message" ? path.slice(0, -1) : path
}

/** A reviewed legacy order is immutable; original rows and parent edges remain intact. */
export const resolveLegacyProjection = (
  nodes: readonly HistoryNodeV1[], orderedPathIds: readonly string[], cursor: HistoryCursorV1
): readonly HistoryNodeV1[] => {
  const byId = indexNodes(nodes)
  const seen = new Set<string>()
  const path = orderedPathIds.map(id => {
    if (seen.has(id)) throw new HistorySelectionError("duplicate_projection_member")
    seen.add(id)
    const row = byId.get(id)
    if (!row) throw new HistorySelectionError("missing_projection_member")
    return row
  })
  if (cursor.kind === "empty") return []
  const at = orderedPathIds.indexOf(cursor.message_id)
  if (at < 0) throw new HistorySelectionError("missing_cursor")
  return path.slice(0, at + (cursor.kind === "after_message" ? 1 : 0))
}

/** Bind a separately loaded content payload to the exact captured path. */
export const bindSelectedHistoryContent = (
  rows: readonly HistoryNodeV1[], content: readonly HistorySelectedContentV1[]
): readonly HistorySelectedContentV1[] => {
  if (rows.length !== content.length || new Set(rows.map(row => row.id)).size !== rows.length ||
      rows.some((row, index) => row.id !== content[index]?.id || row.revision !== content[index]?.revision)) {
    throw new HistorySelectionError("selected_content_mismatch")
  }
  return Object.freeze(content.map(item => Object.freeze({
    id: item.id,
    revision: item.revision,
    message: item.message,
    images: Object.freeze([...item.images])
  })))
}

/** The nine-member tuple is the only normal-selection digest input. */
export const canonicalSelectionTuple = (selection: HistorySelectionV1): readonly unknown[] => [
  1,
  selection.owner_key,
  selection.conversation_id,
  [selection.interpretation.kind, selection.interpretation.kind === "legacy_linear_v1" ? selection.interpretation.projection_id : null],
  [selection.cursor.kind, selection.cursor.kind === "empty" ? null : selection.cursor.message_id],
  selection.purpose,
  selection.messages.map(({ id, revision }) => [id, revision]),
  selection.storage_context_digest,
  selection.request_context_digest
]

export const canonicalSelectionJson = (selection: HistorySelectionV1): string => JSON.stringify(canonicalSelectionTuple(selection))
export const selectionDigest = (selection: HistorySelectionV1): string => bytesToHex(sha256(utf8ToBytes(canonicalSelectionJson(selection))))

/** Comparison uses semantic source order, independent of stored cross-model parent edges. */
export const resolveComparisonProjection = (
  nodes: readonly HistoryNodeV1[], modelId: string, boundaryId: string
): readonly HistoryNodeV1[] => {
  indexNodes(nodes)
  const boundary = nodes.find(node => node.id === boundaryId)
  if (!boundary || !boundary.comparison || (!boundary.comparison.common && boundary.comparison.model_id !== modelId)) {
    throw new HistorySelectionError("invalid_comparison_boundary")
  }
  const end = nodes.indexOf(boundary)
  return nodes.slice(0, end + 1).filter(node => node.comparison?.common || node.comparison?.model_id === modelId)
}

export const canonicalComparisonTuple = (selection: CompareHistorySelectionV1): readonly unknown[] => [
  1, "comparison", selection.owner_key, selection.conversation_id, selection.model_id, selection.cluster_id,
  [selection.cursor.kind, selection.cursor.kind === "empty" ? null : selection.cursor.message_id],
  selection.messages.map(({ id, revision }) => [id, revision]),
  selection.storage_context_digest, selection.request_context_digest
]
export const canonicalComparisonJson = (selection: CompareHistorySelectionV1): string => JSON.stringify(canonicalComparisonTuple(selection))
export const comparisonDigest = (selection: CompareHistorySelectionV1): string => bytesToHex(sha256(utf8ToBytes(canonicalComparisonJson(selection))))

export const resolveHistorySelection = (
  snapshot: HistorySelectionSnapshotV1, view: HistoryViewSelectionV1, purpose: "send" | "fork", requestContextDigest: string
): HistoryResolutionV1 => {
  if (snapshot.version !== 1) return { status: "unsupported_history_capability", code: "unsupported_version" }
  if (snapshot.owner_key !== view.owner_key || snapshot.conversation_id !== view.conversation_id) {
    return { status: "invalid_history", code: "owner_conversation_mismatch" }
  }
  const status = snapshot.interpretation_status
  if (status.kind === "legacy_review_required") return { status: "legacy_review_required", code: "legacy_review_required" }
  if (status.kind !== view.interpretation.kind ||
      (status.kind === "legacy_linear_v1" && view.interpretation.kind === "legacy_linear_v1" && status.projection_id !== view.interpretation.projection_id)) {
    return { status: "stale_selection", code: "interpretation_mismatch" }
  }
  try {
    if (snapshot.nodes.some(node => node.conversation_id && node.conversation_id !== snapshot.conversation_id)) {
      throw new HistorySelectionError("cross_conversation_parent")
    }
    const rows = status.kind === "parent_graph_v1"
      ? resolveParentPath(snapshot.nodes, view.cursor)
      : resolveLegacyProjection(snapshot.nodes, status.ordered_path_ids, view.cursor)
    if (rows.some(row => !row.settled)) throw new HistorySelectionError("unsettled_message")
    const selection: HistorySelectionV1 = {
      version: 1, owner_key: snapshot.owner_key, conversation_id: snapshot.conversation_id,
      interpretation: view.interpretation, cursor: view.cursor, selection_revision: view.selection_revision,
      purpose, messages: rows.map(({ id, revision }) => ({ id, revision })), fences: snapshot.fences,
      storage_context_digest: snapshot.storage_context_digest, request_context_digest: requestContextDigest, selection_digest: ""
    }
    return { status: "ready", rows, selection: { ...selection, selection_digest: selectionDigest(selection) } }
  } catch (error) {
    if (!(error instanceof HistorySelectionError)) throw error
    return { status: error.code === "missing_cursor" ? "stale_selection" : "invalid_history", code: error.code }
  }
}
