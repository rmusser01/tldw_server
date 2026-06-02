import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { FlashcardDocumentRow } from "../FlashcardDocumentRow"
import type { Deck, Flashcard, FlashcardBulkUpdateResponse } from "@/services/flashcards"
import { DEFAULT_SCHEDULER_SETTINGS_ENVELOPE } from "../../utils/scheduler-settings"

const uploadFlashcardAsset = vi.hoisted(() => vi.fn())

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

vi.mock("@/services/flashcard-assets", () => ({
  uploadFlashcardAsset
}))

vi.mock("../MarkdownWithBoundary", () => ({
  MarkdownWithBoundary: ({ content }: { content: string }) => <div>{content}</div>
}))

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
    review_prompt_side: "front",
    scheduler_settings: DEFAULT_SCHEDULER_SETTINGS_ENVELOPE
  }
]

describe("FlashcardDocumentRow image insertion", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("inserts an uploaded image snippet into the active row field", async () => {
    uploadFlashcardAsset.mockResolvedValue({
      asset_uuid: "asset-1",
      reference: "flashcard-asset://asset-1",
      markdown_snippet: "![Scan](flashcard-asset://asset-1)"
    })

    const bulkUpdate = vi.fn().mockResolvedValue({
      results: []
    } satisfies FlashcardBulkUpdateResponse)

    render(
      <FlashcardDocumentRow
        card={makeFlashcard({ front: "Alpha Omega" })}
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
    fireEvent.change(frontInput, { target: { value: "Alpha Omega" } })
    ;(frontInput as HTMLInputElement).focus()
    ;(frontInput as HTMLInputElement).setSelectionRange(6, 6)
    fireEvent.select(frontInput)

    const uploadInput = screen.getByLabelText("Upload image for Question row-1")
    fireEvent.change(uploadInput, {
      target: {
        files: [new File(["binary"], "scan.png", { type: "image/png" })]
      }
    })

    await waitFor(() => {
      expect((frontInput as HTMLInputElement).value).toBe(
        "Alpha ![Scan](flashcard-asset://asset-1)Omega"
      )
    })
  })

  it("renders upload failures with a design-system alert", async () => {
    uploadFlashcardAsset.mockRejectedValue(new Error("Upload unavailable"))

    render(
      <FlashcardDocumentRow
        card={makeFlashcard()}
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

    fireEvent.click(screen.getByTestId("flashcards-document-row-front-display-row-1"))
    await screen.findByTestId("flashcards-document-row-front-input-row-1")

    const uploadInput = screen.getByLabelText("Upload image for Question row-1")
    fireEvent.change(uploadInput, {
      target: {
        files: [new File(["binary"], "scan.png", { type: "image/png" })]
      }
    })

    const alert = await screen.findByTestId("flashcards-document-row-upload-error-row-1")
    expect(alert).toHaveAttribute("data-ds-component", "Alert")
    expect(alert).toHaveTextContent("Upload unavailable")
  })
})
