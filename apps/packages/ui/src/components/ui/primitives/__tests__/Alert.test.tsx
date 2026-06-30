import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
import { Alert } from "../Alert"

describe("Alert", () => {
  it("does not render a content wrapper for whitespace-only children", () => {
    render(
      <Alert title="Review mode" data-testid="review-alert">
        {"\n  "}
        {null}
      </Alert>
    )

    const title = screen.getByText("Review mode")
    expect(screen.getByTestId("review-alert")).toHaveAttribute(
      "data-ds-component",
      "Alert"
    )
    expect(title.parentElement?.children).toHaveLength(1)
  })
})
