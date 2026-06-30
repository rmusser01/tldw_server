import { describe, expect, it } from "vitest"

import type { Deck } from "@/services/flashcards"
import {
  formatDeckHierarchyLabel,
  getDeckDescendantIds
} from "../deck-display"
import { DEFAULT_SCHEDULER_SETTINGS_ENVELOPE } from "../scheduler-settings"

const makeDeck = (overrides: Partial<Deck> & Pick<Deck, "id" | "name">): Deck => ({
  description: null,
  workspace_id: null,
  parent_deck_id: null,
  review_prompt_side: "front",
  deleted: false,
  client_id: "test",
  version: 1,
  created_at: null,
  last_modified: null,
  scheduler_type: "sm2_plus",
  scheduler_settings_json: null,
  scheduler_settings: DEFAULT_SCHEDULER_SETTINGS_ENVELOPE,
  ...overrides
})

describe("deck-display hierarchy helpers", () => {
  it("formats deck labels with ancestor names and workspace scope", () => {
    const parent = makeDeck({ id: 1, name: "Languages" })
    const child = makeDeck({
      id: 2,
      name: "Spanish",
      parent_deck_id: 1,
      workspace_id: "ws-study"
    })
    const deckMap = new Map([
      [parent.id, parent],
      [child.id, child]
    ])

    expect(formatDeckHierarchyLabel(child, deckMap)).toBe("Languages / Spanish · ws-study")
  })

  it("falls back safely when deck hierarchy data contains a cycle", () => {
    const first = makeDeck({ id: 1, name: "First", parent_deck_id: 2 })
    const second = makeDeck({ id: 2, name: "Second", parent_deck_id: 1 })
    const deckMap = new Map([
      [first.id, first],
      [second.id, second]
    ])

    expect(formatDeckHierarchyLabel(first, deckMap)).toBe("Second / First")
  })

  it("returns all descendant ids for filtering parent deck options", () => {
    const decks = [
      makeDeck({ id: 1, name: "Root" }),
      makeDeck({ id: 2, name: "Child", parent_deck_id: 1 }),
      makeDeck({ id: 3, name: "Grandchild", parent_deck_id: 2 }),
      makeDeck({ id: 4, name: "Sibling", parent_deck_id: 1 }),
      makeDeck({ id: 5, name: "Other root" })
    ]

    expect(getDeckDescendantIds(decks, 1)).toEqual(new Set([2, 3, 4]))
  })
})
