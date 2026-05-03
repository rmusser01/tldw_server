import type { Deck } from "@/services/flashcards"

type DeckLabelInput = Pick<Deck, "name" | "workspace_id"> | null | undefined
type DeckHierarchyInput = Pick<Deck, "id" | "name" | "workspace_id" | "parent_deck_id"> | null | undefined

export const getDeckWorkspaceId = (deck: DeckLabelInput): string | null => {
  const workspaceId = deck?.workspace_id?.trim()
  return workspaceId ? workspaceId : null
}

export const formatDeckDisplayName = (
  deck: DeckLabelInput,
  fallbackName = "Untitled deck"
): string => {
  const baseName = deck?.name?.trim() || fallbackName
  const workspaceId = getDeckWorkspaceId(deck)
  return workspaceId ? `${baseName} · ${workspaceId}` : baseName
}

export const formatDeckHierarchyLabel = (
  deck: DeckHierarchyInput,
  deckMap: Map<number, DeckHierarchyInput>,
  fallbackName = "Untitled deck"
): string => {
  if (!deck) {
    return fallbackName
  }

  const names: string[] = []
  const seenDeckIds = new Set<number>()
  let current: DeckHierarchyInput = deck

  while (current && !seenDeckIds.has(current.id)) {
    seenDeckIds.add(current.id)
    names.unshift(current.name?.trim() || `Deck ${current.id}`)
    const parentDeckId = current.parent_deck_id
    current = parentDeckId == null ? null : deckMap.get(parentDeckId)
  }

  const workspaceId = getDeckWorkspaceId(deck)
  const baseName = names.length > 0 ? names.join(" / ") : fallbackName
  return workspaceId ? `${baseName} · ${workspaceId}` : baseName
}

export const getDeckDescendantIds = (
  decks: DeckHierarchyInput[],
  rootDeckId: number
): Set<number> => {
  const childrenByParentId = new Map<number, number[]>()
  decks.forEach((deck) => {
    if (!deck || deck.parent_deck_id == null) {
      return
    }
    const children = childrenByParentId.get(deck.parent_deck_id) ?? []
    children.push(deck.id)
    childrenByParentId.set(deck.parent_deck_id, children)
  })

  const descendants = new Set<number>()
  const queue = [...(childrenByParentId.get(rootDeckId) ?? [])]
  while (queue.length > 0) {
    const deckId = queue.shift()
    if (deckId == null || descendants.has(deckId) || deckId === rootDeckId) {
      continue
    }
    descendants.add(deckId)
    queue.push(...(childrenByParentId.get(deckId) ?? []))
  }

  return descendants
}
