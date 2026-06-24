import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ReviewProgress } from "../ReviewProgress"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      options?: string | {
        defaultValue?: string
        count?: number
        remaining?: number
        reviewed?: number
      }
    ) => {
      if (typeof options === "string") return options
      return (options?.defaultValue ?? "").replace(/\{\{(\w+)\}\}/g, (_match, token: string) =>
        String((options as Record<string, unknown>)?.[token] ?? `{{${token}}}`)
      )
    }
  })
}))

describe("ReviewProgress responsive layout", () => {
  it("wraps review metrics and keeps long deck names contained", () => {
    render(
      <ReviewProgress
        dueCount={12}
        reviewedCount={3}
        availableNowCount={9}
        scheduledDueCount={4}
        deckName="Long biology deck name that should stay inside the progress row"
      />
    )

    const progress = screen.getByTestId("flashcards-review-progress")
    expect(progress).toHaveClass("flex-wrap", "max-w-full")
    expect(screen.getByTestId("flashcards-review-progress-deck-name")).toHaveClass(
      "max-w-full",
      "min-w-0",
      "truncate"
    )
  })
})
