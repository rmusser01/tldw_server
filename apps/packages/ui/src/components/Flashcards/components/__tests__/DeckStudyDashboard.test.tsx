// @vitest-environment jsdom
import { fireEvent, render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { DeckStudyDashboard } from "../DeckStudyDashboard"
import { DEFAULT_SCHEDULER_SETTINGS_ENVELOPE } from "../../utils/scheduler-settings"
import type { Deck, FlashcardDeckProgress } from "@/services/flashcards"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
            [key: string]: unknown
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) {
        return defaultValueOrOptions.defaultValue.replace(
          /\{\{(\w+)\}\}/g,
          (_match, token: string) =>
            String((defaultValueOrOptions as Record<string, unknown>)[token] ?? `{{${token}}}`)
        )
      }
      return key
    }
  })
}))

const buildDeck = (id: number, name: string): Deck => ({
  id,
  name,
  description: null,
  workspace_id: null,
  parent_deck_id: null,
  review_prompt_side: "front",
  deleted: false,
  client_id: "test",
  version: 1,
  scheduler_type: "sm2_plus",
  scheduler_settings: DEFAULT_SCHEDULER_SETTINGS_ENVELOPE
})

const progressRows: FlashcardDeckProgress[] = [
  {
    deck_id: 1,
    deck_name: "Fresh deck",
    total: 6,
    new: 2,
    learning: 0,
    due: 0,
    mature: 4
  },
  {
    deck_id: 2,
    deck_name: "Due deck",
    total: 10,
    new: 0,
    learning: 1,
    due: 3,
    mature: 6
  },
  {
    deck_id: 3,
    deck_name: "Caught up deck",
    total: 8,
    new: 0,
    learning: 0,
    due: 0,
    mature: 8
  }
]

describe("DeckStudyDashboard", () => {
  it("sorts ready decks first and shows deck queue counts", () => {
    render(
      <DeckStudyDashboard
        decks={[
          buildDeck(1, "Fresh deck"),
          buildDeck(2, "Due deck"),
          buildDeck(3, "Caught up deck")
        ]}
        deckProgress={progressRows}
        onReviewDeck={() => {}}
        onCramDeck={() => {}}
        onManageDeck={() => {}}
      />
    )

    const rows = screen.getAllByTestId(/^flashcards-deck-study-dashboard-row-/)
    expect(rows).toHaveLength(3)
    expect(within(rows[0]).getByText("Due deck")).toBeInTheDocument()
    expect(within(rows[1]).getByText("Fresh deck")).toBeInTheDocument()
    expect(within(rows[2]).getByText("Caught up deck")).toBeInTheDocument()

    expect(within(rows[0]).getByText("Due: 3")).toBeInTheDocument()
    expect(within(rows[0]).getByText("Learning: 1")).toBeInTheDocument()
    expect(within(rows[0]).getByText("Mature: 6")).toBeInTheDocument()
  })

  it("offers direct review, cram, edit, scheduler, and export actions", () => {
    const onReviewDeck = vi.fn()
    const onCramDeck = vi.fn()
    const onManageDeck = vi.fn()
    const onOpenScheduler = vi.fn()
    const onExportDeck = vi.fn()

    render(
      <DeckStudyDashboard
        decks={[buildDeck(2, "Due deck")]}
        deckProgress={[progressRows[1]]}
        onReviewDeck={onReviewDeck}
        onCramDeck={onCramDeck}
        onManageDeck={onManageDeck}
        onOpenScheduler={onOpenScheduler}
        onExportDeck={onExportDeck}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Review 4 ready" }))
    fireEvent.click(screen.getByRole("button", { name: "Cram" }))
    fireEvent.click(screen.getByRole("button", { name: "Edit" }))
    fireEvent.click(screen.getByRole("button", { name: "Scheduler" }))
    fireEvent.click(screen.getByRole("button", { name: "Export" }))

    expect(onReviewDeck).toHaveBeenCalledWith(2)
    expect(onCramDeck).toHaveBeenCalledWith(2)
    expect(onManageDeck).toHaveBeenCalledWith(2)
    expect(onOpenScheduler).toHaveBeenCalledWith(2)
    expect(onExportDeck).toHaveBeenCalledWith(2)
  })

  it("keeps cram available when a deck has no due work", () => {
    const onReviewDeck = vi.fn()
    const onCramDeck = vi.fn()

    render(
      <DeckStudyDashboard
        decks={[buildDeck(3, "Caught up deck")]}
        deckProgress={[progressRows[2]]}
        onReviewDeck={onReviewDeck}
        onCramDeck={onCramDeck}
        onManageDeck={() => {}}
      />
    )

    expect(screen.getByRole("button", { name: "Caught up" })).toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "Cram" }))

    expect(onReviewDeck).not.toHaveBeenCalled()
    expect(onCramDeck).toHaveBeenCalledWith(3)
  })

  it("renders nothing when there are no deck rows so existing onboarding can own empty state", () => {
    const { container } = render(
      <DeckStudyDashboard
        decks={[]}
        deckProgress={[]}
        onReviewDeck={() => {}}
        onCramDeck={() => {}}
        onManageDeck={() => {}}
      />
    )

    expect(container).toBeEmptyDOMElement()
  })

  it("handles missing deck lists and deck names without crashing", () => {
    render(
      <DeckStudyDashboard
        decks={null as unknown as Deck[]}
        deckProgress={[
          {
            deck_id: 99,
            deck_name: undefined,
            total: 4,
            new: 1,
            learning: 1,
            due: 0,
            mature: 2
          } as unknown as FlashcardDeckProgress
        ]}
        onReviewDeck={() => {}}
        onCramDeck={() => {}}
        onManageDeck={() => {}}
      />
    )

    expect(screen.getByText("Deck 99")).toBeInTheDocument()
    expect(screen.getByText("Review 2 ready")).toBeInTheDocument()
  })
})
