import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ReviewProgress } from "../ReviewProgress"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
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

describe("ReviewProgress queue language", () => {
  it("labels the available study queue and shows bucket counts separately", () => {
    render(
      <ReviewProgress
        dueCount={6}
        reviewedCount={2}
        deckName="Biology"
        availableNowCount={6}
        scheduledDueCount={3}
        newCount={2}
        learningCount={1}
      />
    )

    const progress = screen.getByTestId("flashcards-review-progress")

    expect(progress).toHaveTextContent("Study queue")
    expect(progress).toHaveTextContent("Available now: 6")
    expect(progress).toHaveTextContent("new: 2")
    expect(progress).toHaveTextContent("learning: 1")
    expect(progress).toHaveTextContent("due: 3")
    expect(progress).not.toHaveTextContent("Scheduled due")
  })
})
