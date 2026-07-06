// @vitest-environment jsdom
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import type { Flashcard } from "@/services/flashcards"
import { FlashcardActionsMenu } from "../FlashcardActionsMenu"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, options?: { defaultValue?: string }) => options?.defaultValue ?? _key
  })
}))

const card: Flashcard = {
  uuid: "card-1",
  front: "Front",
  back: "Back",
  is_cloze: false,
  ef: 2.5,
  interval_days: 0,
  repetitions: 0,
  lapses: 0,
  queue_state: "new",
  deleted: false,
  client_id: "test",
  version: 1,
  model_type: "basic",
  reverse: false
}

describe("FlashcardActionsMenu", () => {
  it("does not fire edit twice for one pointer click", () => {
    const onEdit = vi.fn()
    render(
      <FlashcardActionsMenu
        card={card}
        onEdit={onEdit}
        onReview={vi.fn()}
        onDuplicate={vi.fn()}
        onMove={vi.fn()}
      />
    )

    const editButton = screen.getByRole("button", { name: "Edit" })
    fireEvent.pointerDown(editButton, { button: 0 })
    fireEvent.click(editButton)

    expect(onEdit).toHaveBeenCalledTimes(1)
  })
})
