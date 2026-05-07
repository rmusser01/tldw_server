// @vitest-environment jsdom

import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { SearchEmptyState } from "../SearchEmptyState"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: unknown) =>
      typeof fallback === "string" ? fallback : key
  })
}))

describe("SearchEmptyState", () => {
  it("renders the initial state with canonical EmptyState semantics and dismissible hint", async () => {
    const user = userEvent.setup()
    const onDismissHint = vi.fn()
    const { container } = render(
      <SearchEmptyState
        variant="initial"
        showHint
        onDismissHint={onDismissHint}
      />
    )

    expect(container.querySelector('[data-ds-component="EmptyState"]')).toBeInTheDocument()
    expect(screen.getByText("No results yet")).toBeInTheDocument()
    expect(
      screen.getByText("Search your knowledge base and insert results into your message.")
    ).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Dismiss" }))
    expect(onDismissHint).toHaveBeenCalledTimes(1)
  })

  it("renders no-results guidance with canonical EmptyState semantics", () => {
    const { container } = render(<SearchEmptyState variant="no-results" />)

    expect(container.querySelector('[data-ds-component="EmptyState"]')).toBeInTheDocument()
    expect(screen.getByText("No results found")).toBeInTheDocument()
    expect(
      screen.getByText("Try a different search query or adjust your filters.")
    ).toBeInTheDocument()
  })

  it("renders timeout recovery with retry action", async () => {
    const user = userEvent.setup()
    const onRetry = vi.fn()
    const { container } = render(
      <SearchEmptyState variant="timeout" onRetry={onRetry} />
    )

    expect(container.querySelector('[data-ds-component="EmptyState"]')).toBeInTheDocument()
    expect(screen.getByText("Request timed out.")).toBeInTheDocument()

    await user.click(screen.getByRole("button", { name: "Retry" }))
    expect(onRetry).toHaveBeenCalledTimes(1)
  })

  it("omits timeout retry action when no retry callback is provided", () => {
    const { container } = render(<SearchEmptyState variant="timeout" />)

    expect(container.querySelector('[data-ds-component="EmptyState"]')).toBeInTheDocument()
    expect(screen.getByText("Request timed out.")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Retry" })).not.toBeInTheDocument()
  })

  it("renders disconnected guidance with canonical EmptyState semantics", () => {
    const { container } = render(<SearchEmptyState variant="disconnected" />)

    expect(container.querySelector('[data-ds-component="EmptyState"]')).toBeInTheDocument()
    expect(
      screen.getByText("Connect to server to search knowledge base")
    ).toBeInTheDocument()
  })
})
