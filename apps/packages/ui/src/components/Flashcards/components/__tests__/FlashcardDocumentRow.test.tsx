import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi, beforeEach } from "vitest"

import { FlashcardDocumentRow } from "../FlashcardDocumentRow"
import type { Deck, Flashcard, FlashcardBulkUpdateResponse } from "@/services/flashcards"
import { DEFAULT_SCHEDULER_SETTINGS_ENVELOPE } from "../../utils/scheduler-settings"

const queryClientSpies = {
  setQueryData: vi.fn(),
  invalidateQueries: vi.fn().mockResolvedValue(undefined)
}

vi.mock("@tanstack/react-query", async () => {
  const actual = await vi.importActual<typeof import("@tanstack/react-query")>("@tanstack/react-query")
  return {
    ...actual,
    useQueryClient: () => queryClientSpies
  }
})

const makeFlashcard = (overrides: Partial<Flashcard> = {}): Flashcard => ({
  uuid: "row-1",
  deck_id: 1,
  front: "Original front",
  back: "Original back",
  notes: "Original note",
  extra: null,
  is_cloze: false,
  tags: ["bio"],
  ef: 2.5,
  interval_days: 3,
  repetitions: 1,
  lapses: 0,
  due_at: null,
  created_at: null,
  last_reviewed_at: null,
  queue_state: "new",
  last_modified: null,
  deleted: false,
  client_id: "test",
  version: 1,
  model_type: "basic",
  reverse: false,
  ...overrides
})

const decks: Deck[] = [
  {
    id: 1,
    name: "Biology",
    deleted: false,
    client_id: "test",
    version: 1,
    review_prompt_side: "front",
    scheduler_type: "sm2_plus",
    scheduler_settings: DEFAULT_SCHEDULER_SETTINGS_ENVELOPE
  }
]

describe("FlashcardDocumentRow", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("renders row metadata chips with design-system badges", () => {
    render(
      <FlashcardDocumentRow
        card={makeFlashcard({
          reverse: true,
          tags: ["bio", "exam"],
          source_ref_type: "media",
          source_ref_id: "42"
        })}
        decks={decks}
        selected={false}
        selectAllAcross={false}
        filterContext={{
          deckId: 1,
          tags: ["bio"],
          sortBy: "due",
          dueStatus: "all"
        }}
        queryKey={["flashcards:document", 1]}
        onToggleSelect={() => {}}
        bulkUpdate={vi.fn()}
      />
    )

    expect(
      screen.getByText("Reversible").closest('[data-ds-component="Badge"]')
    ).toBeInTheDocument()
    expect(
      screen.getByText("New").closest('[data-ds-component="Badge"]')
    ).toBeInTheDocument()
    expect(
      screen
        .getAllByText("Biology")
        .some((node) => node.closest('[data-ds-component="Badge"]'))
    ).toBe(true)
    expect(
      screen
        .getAllByText("bio")
        .some((node) => node.closest('[data-ds-component="Badge"]'))
    ).toBe(true)
    expect(
      screen.getByText("exam").closest('[data-ds-component="Badge"]')
    ).toBeInTheDocument()
    expect(
      screen.getByText("Media #42").closest('[data-ds-component="Badge"]')
    ).toBeInTheDocument()
  })

  it("queues overlapping row edits and sends the second patch after the first succeeds", async () => {
    let resolveFirst: ((value: FlashcardBulkUpdateResponse) => void) | null = null
    const firstPromise = new Promise<FlashcardBulkUpdateResponse>((resolve) => {
      resolveFirst = resolve
    })
    const bulkUpdate = vi
      .fn()
      .mockImplementationOnce(() => firstPromise)
      .mockResolvedValueOnce({
        results: [
          {
            uuid: "row-1",
            status: "updated",
            flashcard: makeFlashcard({
              uuid: "row-1",
              version: 3,
              front: "two"
            })
          }
        ]
      })

    render(
      <FlashcardDocumentRow
        card={makeFlashcard({ uuid: "row-1", version: 1 })}
        decks={decks}
        selected={false}
        selectAllAcross={false}
        filterContext={{
          deckId: 1,
          tags: ["bio"],
          sortBy: "due",
          dueStatus: "all"
        }}
        queryKey={["flashcards:document", 1]}
        onToggleSelect={() => {}}
        bulkUpdate={bulkUpdate}
      />
    )

    fireEvent.click(screen.getByTestId("flashcards-document-row-front-display-row-1"))

    const frontInput = await screen.findByTestId("flashcards-document-row-front-input-row-1")
    fireEvent.change(frontInput, { target: { value: "one" } })
    fireEvent.blur(frontInput)

    fireEvent.change(frontInput, { target: { value: "two" } })
    fireEvent.blur(frontInput)

    expect(bulkUpdate).toHaveBeenCalledTimes(1)
    expect(bulkUpdate).toHaveBeenNthCalledWith(
      1,
      expect.arrayContaining([
        expect.objectContaining({
          uuid: "row-1",
          front: "one",
          expected_version: 1
        })
      ])
    )

    resolveFirst?.({
      results: [
        {
          uuid: "row-1",
          status: "updated",
          flashcard: makeFlashcard({
            uuid: "row-1",
            version: 2,
            front: "one"
          })
        }
      ]
    })

    await waitFor(() => {
      expect(bulkUpdate).toHaveBeenCalledTimes(2)
    })

    expect(bulkUpdate.mock.calls[1][0][0]).toEqual(
      expect.objectContaining({
        uuid: "row-1",
        front: "two",
        expected_version: 2
      })
    )
  })

  it("reapplies nullable field clears after resolving a row conflict", async () => {
    const bulkUpdate = vi
      .fn()
      .mockResolvedValueOnce({
        results: [
          {
            uuid: "row-1",
            status: "conflict",
            error: {
              code: "conflict",
              message: "Version changed elsewhere"
            }
          }
        ]
      })
      .mockResolvedValueOnce({
        results: [
          {
            uuid: "row-1",
            status: "updated",
            flashcard: makeFlashcard({
              uuid: "row-1",
              version: 3,
              notes: null
            })
          }
        ]
      })
    const loadLatestCard = vi.fn().mockResolvedValue(
      makeFlashcard({
        uuid: "row-1",
        version: 2,
        notes: "Server note"
      })
    )

    render(
      <FlashcardDocumentRow
        card={makeFlashcard({ uuid: "row-1", version: 1 })}
        decks={decks}
        selected={false}
        selectAllAcross={false}
        filterContext={{
          deckId: 1,
          tags: ["bio"],
          sortBy: "due",
          dueStatus: "all"
        }}
        queryKey={["flashcards:document", 1]}
        onToggleSelect={() => {}}
        bulkUpdate={bulkUpdate}
        loadLatestCard={loadLatestCard}
      />
    )

    fireEvent.click(screen.getByTestId("flashcards-document-row-row-1"))
    const notesInput = await screen.findByTestId("flashcards-document-row-notes-input-row-1")
    fireEvent.change(notesInput, { target: { value: "" } })
    fireEvent.blur(notesInput)

    const conflictAlert = await screen.findByTestId("flashcards-document-row-conflict-row-1")
    expect(conflictAlert).toHaveAttribute("data-ds-component", "Alert")
    fireEvent.click(screen.getByRole("button", { name: /reapply my edit/i }))

    await waitFor(() => {
      expect(bulkUpdate).toHaveBeenCalledTimes(2)
    })

    expect(loadLatestCard).toHaveBeenCalledWith("row-1")
    expect(bulkUpdate.mock.calls[1][0][0]).toEqual(
      expect.objectContaining({
        uuid: "row-1",
        notes: null,
        expected_version: 2
      })
    )
  })

  it("invalidates the document query when reload fetches a row that no longer matches filters", async () => {
    const bulkUpdate = vi.fn().mockResolvedValueOnce({
      results: [
        {
          uuid: "row-1",
          status: "conflict",
          error: {
            code: "conflict",
            message: "Version changed elsewhere"
          }
        }
      ]
    })
    const loadLatestCard = vi.fn().mockResolvedValue(
      makeFlashcard({
        uuid: "row-1",
        version: 2,
        deck_id: 7,
        tags: ["chem"]
      })
    )

    render(
      <FlashcardDocumentRow
        card={makeFlashcard({ uuid: "row-1", version: 1 })}
        decks={decks}
        selected={false}
        selectAllAcross={false}
        filterContext={{
          deckId: 1,
          tags: ["bio"],
          sortBy: "due",
          dueStatus: "all"
        }}
        queryKey={["flashcards:document", 1]}
        onToggleSelect={() => {}}
        bulkUpdate={bulkUpdate}
        loadLatestCard={loadLatestCard}
      />
    )

    fireEvent.click(screen.getByTestId("flashcards-document-row-front-display-row-1"))
    const frontInput = await screen.findByTestId("flashcards-document-row-front-input-row-1")
    fireEvent.change(frontInput, { target: { value: "Edited front" } })
    fireEvent.blur(frontInput)

    const conflictAlert = await screen.findByTestId("flashcards-document-row-conflict-row-1")
    expect(conflictAlert).toHaveAttribute("data-ds-component", "Alert")
    fireEvent.click(screen.getByRole("button", { name: /reload row/i }))

    await waitFor(() => {
      expect(queryClientSpies.invalidateQueries).toHaveBeenCalledWith({
        queryKey: ["flashcards:document", 1]
      })
    })

    expect(queryClientSpies.setQueryData).not.toHaveBeenCalled()
  })
})
