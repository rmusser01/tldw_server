/**
 * Presentation helpers for the "Review conversation history" controls
 * (HistorySelectionReview, CC-04 #3106). They only read a captured history
 * snapshot; choosing and confirming stay with the history-selection controller.
 */
import type {
  HistoryCaptureResultV1,
  HistoryCursorV1,
  HistoryNodeV1,
  HistoryViewSelectionV1
} from "@/types/history-selection"
import { latestHistoryTipId, resolveHistorySelection } from "@/utils/history-selection"

type LinkedNode = Pick<HistoryNodeV1, "id" | "parent_id">

/**
 * Owner order, except that a message never comes before the message it
 * answers: a reply listed ahead of its question is moved after it.
 */
export const chronologicalHistoryNodes = <T extends LinkedNode>(
  nodes: readonly T[]
): T[] => {
  const byId = new Map(nodes.map((node) => [node.id, node]))
  const ordered: T[] = []
  const placed = new Set<string>()
  for (const node of nodes) {
    // Walk up to the first placed ancestor without recursion, so a long
    // child-first chain can't overflow the stack; a cycle stops the walk.
    const chain: T[] = []
    const inChain = new Set<string>()
    let current: T | undefined = node
    while (current && !placed.has(current.id) && !inChain.has(current.id)) {
      chain.push(current)
      inChain.add(current.id)
      current = current.parent_id === null ? undefined : byId.get(current.parent_id)
    }
    for (let index = chain.length - 1; index >= 0; index -= 1) {
      placed.add(chain[index].id)
      ordered.push(chain[index])
    }
  }
  return ordered
}

/** A one-line, plain-text preview of message text that may contain markdown. */
export const plainHistoryPreview = (value: string | null | undefined): string =>
  (value ?? "")
    .replace(/```[^\n]*/g, " ")
    .replace(/!\[([^\]]*)\]\([^)]*\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
    // HTML tags only, so "List<String>" keeps its text.
    .replace(/<\/?(?:a|b|blockquote|br|code|div|em|h[1-6]|hr|i|img|li|ol|p|pre|s|span|strong|sub|sup|table|td|th|tr|u|ul)\b[^>]*>/gi, " ")
    .replace(/^[ \t]{0,3}(?:#{1,6}|>+|[-*+]|\d+[.)])[ \t]+/gm, "")
    .replace(/\*\*|__|~~|`/g, "")
    .replace(/(^|[\s(])[*_](?=\S)([^*_\n]*?\S)[*_](?=$|[\s).,!?:;])/g, "$1$2")
    .replace(/\s+/g, " ")
    .trim()

/** How many earlier user turns or alternative replies make the controls worth showing. */
export const hasReviewableHistory = (nodes: readonly HistoryNodeV1[]): boolean => {
  const replies = new Map<string | null, number>()
  let questions = 0
  for (const node of nodes) {
    if (node.role === "user") questions += 1
    const siblings = (replies.get(node.parent_id) ?? 0) + 1
    // Two messages answering the same message are alternatives to choose between.
    if (siblings > 1) return true
    replies.set(node.parent_id, siblings)
  }
  return questions >= 2
}

export type HistoryCoverage = {
  /** Messages the AI sees with the current choice. */
  seen: number
  /** Messages it would see through the latest message. */
  total: number
  /** The choice that lets the AI see every message again. */
  restore: HistoryCursorV1
  /** Messages hidden from the AI, in conversation order. */
  hidden: readonly HistoryNodeV1[]
}

/**
 * When the current choice shows the AI only the start of the conversation
 * (or nothing), how much it sees and how to restore the rest. Null when it
 * sees everything, or when the choice follows a different branch.
 */
export const partialHistoryCoverage = (
  capture: HistoryCaptureResultV1 | null | undefined,
  view: HistoryViewSelectionV1 | null | undefined,
  /** The choice "Start fresh" replaced, restored in preference to the latest message. */
  previous?: HistoryCursorV1 | null
): HistoryCoverage | null => {
  if (!capture || capture.status !== "captured" || !view || capture.view !== view)
    return null
  const { snapshot } = capture
  const status = snapshot?.interpretation_status
  if (!status || !Array.isArray(capture.rows)) return null
  const candidates = [
    previous?.kind === "after_message" ? previous.message_id : undefined,
    latestHistoryTipId(snapshot.nodes),
    status.kind === "legacy_linear_v1" ? status.ordered_path_ids.at(-1) : undefined
  ].filter((id): id is string => typeof id === "string")
  for (const id of candidates) {
    const restore: HistoryCursorV1 = { kind: "after_message", message_id: id }
    if (view.cursor.kind === "after_message" && view.cursor.message_id === id) return null
    let full: ReturnType<typeof resolveHistorySelection>
    try {
      full = resolveHistorySelection(snapshot, { ...view, cursor: restore }, "send", "")
    } catch {
      continue
    }
    if (full.status !== "ready") continue
    const seen = capture.rows
    if (
      seen.length >= full.rows.length ||
      seen.some((row, index) => full.rows[index]?.id !== row.id)
    )
      return null
    return {
      seen: seen.length,
      total: full.rows.length,
      restore,
      hidden: full.rows.slice(seen.length)
    }
  }
  return null
}
