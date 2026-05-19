import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { DeckStudyDefaultsFields } from "../DeckStudyDefaultsFields"

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
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

describe("DeckStudyDefaultsFields", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("renders the review prompt side select", () => {
    render(
      <DeckStudyDefaultsFields
        reviewPromptSide="front"
        onReviewPromptSideChange={vi.fn()}
      />
    )

    expect(screen.getByText("Review prompt side")).toBeInTheDocument()
    expect(screen.getByText("Front first")).toBeInTheDocument()
  })

  it("notifies when the review prompt side changes", () => {
    const onChange = vi.fn()
    render(
      <DeckStudyDefaultsFields
        reviewPromptSide="front"
        onReviewPromptSideChange={onChange}
      />
    )

    fireEvent.mouseDown(screen.getByLabelText("Review prompt side"))
    fireEvent.click(screen.getByText("Back first"))

    expect(onChange).toHaveBeenCalledWith("back", expect.anything())
  })
})
