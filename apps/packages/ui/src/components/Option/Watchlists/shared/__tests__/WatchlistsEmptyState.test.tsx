// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { WatchlistsEmptyState } from "../WatchlistsEmptyState"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: unknown) =>
      typeof fallback === "string" ? fallback : key
  })
}))

describe("WatchlistsEmptyState", () => {
  it("adapts feed empty states to the canonical EmptyState primitive", () => {
    const onPrimaryAction = vi.fn()
    const onSecondaryAction = vi.fn()

    render(
      <WatchlistsEmptyState
        entity="feeds"
        primaryLabel="Add RSS feed"
        secondaryLabel="Upload OPML"
        contextHint="Import existing subscriptions to get started faster."
        onPrimaryAction={onPrimaryAction}
        onSecondaryAction={onSecondaryAction}
      />
    )

    const root = screen.getByTestId("watchlists-empty-state-feeds")
    expect(root).toHaveAttribute("data-ds-component", "EmptyState")
    expect(
      screen.getByText("Feeds are the sources your monitors check for new content.")
    ).toBeInTheDocument()
    expect(
      screen.getByText("Import existing subscriptions to get started faster.")
    ).toBeInTheDocument()

    const primary = screen.getByTestId("watchlists-empty-state-feeds-primary")
    const secondary = screen.getByTestId("watchlists-empty-state-feeds-secondary")
    expect(primary).toHaveTextContent("Add RSS feed")
    expect(secondary).toHaveTextContent("Upload OPML")

    fireEvent.click(primary)
    fireEvent.click(secondary)

    expect(onPrimaryAction).toHaveBeenCalledTimes(1)
    expect(onSecondaryAction).toHaveBeenCalledTimes(1)
  })

  it("omits secondary actions for entities without configured secondary copy", () => {
    const onPrimaryAction = vi.fn()
    const onSecondaryAction = vi.fn()

    render(
      <WatchlistsEmptyState
        entity="monitors"
        onPrimaryAction={onPrimaryAction}
        onSecondaryAction={onSecondaryAction}
      />
    )

    expect(screen.getByTestId("watchlists-empty-state-monitors")).toHaveAttribute(
      "data-ds-component",
      "EmptyState"
    )
    expect(
      screen.getByText(
        "Monitors run on a schedule to fetch and process content from your feeds."
      )
    ).toBeInTheDocument()
    expect(
      screen.getByTestId("watchlists-empty-state-monitors-primary")
    ).toHaveTextContent("Create your first monitor")
    expect(
      screen.queryByTestId("watchlists-empty-state-monitors-secondary")
    ).not.toBeInTheDocument()
  })
})
