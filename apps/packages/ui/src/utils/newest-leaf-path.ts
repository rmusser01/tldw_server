/**
 * The path a chat shows by default when its saved messages branch (CM-N2).
 *
 * A regenerated reply is saved as another reply to the same question, and an
 * edited question as another message after the same answer. Listing every
 * saved message by time puts those alternatives in the middle of the thread,
 * where they read as answers to the wrong question and are sent to the model
 * as context. The default view instead walks parents back from the newest
 * leaf, as versioned chats already do, and leaves the alternatives out.
 */
import type { ServerChatMessage } from "@/services/tldw/TldwApiClient"

type Link = { id: string; parentId: string | null | undefined }
type LinkedNode = { id: string; parent_id: string | null }

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

/**
 * Walk parents back from the newest leaf of `listed`, in conversation order
 * (owner order, with each reply after the message it answers).
 *
 * Legacy rows often have no parent. A row whose parent is missing, unknown or
 * listed after it continues the row before it, so an unlinked legacy chat
 * stays one linear thread and only explicit links create branches.
 */
export const newestLeafPath = <T>(
  listed: readonly T[],
  link: (row: T) => Link
): { path: T[]; hasAlternatives: boolean } => {
  // A reply listed before its question (for example with equal timestamps)
  // is moved after it first, so its explicit link is kept.
  const rows = chronologicalHistoryNodes(
    listed.map((row) => {
      const entry = link(row)
      return { id: entry.id, parent_id: entry.parentId ?? null, row }
    })
  ).map((entry) => entry.row)
  const links = rows.map(link)
  const position = new Map<string, number>()
  links.forEach((entry, index) => {
    if (!position.has(entry.id)) position.set(entry.id, index)
  })
  // Each parent is strictly earlier, so the walk below always ends.
  const parentOf = links.map((entry, index) => {
    const explicit = entry.parentId == null ? undefined : position.get(entry.parentId)
    return explicit !== undefined && explicit < index ? explicit : index - 1
  })
  const answered = new Set(parentOf)
  let tip = rows.length - 1
  while (tip >= 0 && answered.has(tip)) tip -= 1
  const path: T[] = []
  for (let index = tip; index >= 0; index = parentOf[index]) path.push(rows[index])
  path.reverse()
  return { path, hasAlternatives: path.length < rows.length }
}

/** The default path of a saved chat's messages, as the server lists them by time. */
export const serverChatActivePath = (
  messages: readonly ServerChatMessage[]
): { messages: ServerChatMessage[]; hasAlternatives: boolean } => {
  const { path, hasAlternatives } = newestLeafPath(messages, (message) => ({
    id: String(message.id),
    parentId: message.parent_message_id ?? null
  }))
  return { messages: path, hasAlternatives }
}

/**
 * The newest message on the branch that starts at `messageId`: the last node,
 * in owner order, that descends from it (or is it) and that nothing answers.
 * Choosing an alternative reply then shows the rest of its branch instead of
 * stopping at the reply. `messageId` itself when it isn't among `nodes`.
 */
export const newestLeafUnder = (
  nodes: readonly { id: string; parent_id: string | null }[],
  messageId: string
): string => {
  const children = new Map<string, string[]>()
  for (const node of nodes) {
    if (node.parent_id === null) continue
    const siblings = children.get(node.parent_id) ?? []
    siblings.push(node.id)
    children.set(node.parent_id, siblings)
  }
  if (!nodes.some((node) => node.id === messageId)) return messageId
  const branch = new Set<string>()
  const pending = [messageId]
  while (pending.length) {
    const id = pending.pop()!
    if (branch.has(id)) continue
    branch.add(id)
    pending.push(...(children.get(id) ?? []))
  }
  for (let index = nodes.length - 1; index >= 0; index -= 1) {
    const { id } = nodes[index]
    if (branch.has(id) && !children.get(id)?.length) return id
  }
  return messageId
}

/**
 * Where choosing an alternative reply moves a chat: the newest message on the
 * alternative's branch, so later turns under it stay visible. A reviewed
 * legacy order, or a chat without a captured history, keeps the reply itself.
 */
export const variantBranchCursor = (
  selection: {
    view?: { interpretation: { kind: string } } | null
    capture?: { snapshot: { nodes: readonly { id: string; parent_id: string | null }[] } } | null
  } | null | undefined,
  messageId: string
): { kind: "after_message"; message_id: string } => {
  const nodes = selection?.capture?.snapshot.nodes
  return {
    kind: "after_message",
    message_id:
      selection?.view?.interpretation.kind === "parent_graph_v1" && nodes
        ? newestLeafUnder(nodes, messageId)
        : messageId
  }
}
