import React from "react"
import { describe, expect, it, vi } from "vitest"
import { render, screen, within, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        return fallbackOrOptions.defaultValue || key
      }
      return key
    }
  })
}))

import { WorldBookListPanel } from "../WorldBookListPanel"

const mockBooks = [
  {
    id: 1,
    name: "Fantasy Lore",
    description: "Magic systems and creatures",
    entry_count: 42,
    enabled: true,
    last_modified: Date.now() - 3600_000
  },
  {
    id: 2,
    name: "Sci-Fi Tech",
    description: "Spaceships and gadgets",
    entry_count: 17,
    enabled: false,
    last_modified: Date.now() - 86400_000
  },
  {
    id: 3,
    name: "History Notes",
    description: "Important dates and events",
    entry_count: 5,
    enabled: true,
    last_modified: Date.now() - 7200_000
  }
]

const defaultProps = {
  worldBooks: mockBooks,
  selectedWorldBookId: null,
  onSelectWorldBook: vi.fn(),
  selectedRowKeys: [] as React.Key[],
  onSelectedRowKeysChange: vi.fn(),
  pendingDeleteIds: [] as number[],
  onEditWorldBook: vi.fn(),
  onRowAction: vi.fn(),
  tableSort: {} as { field?: string; order?: "ascend" | "descend" | null },
  onTableSortChange: vi.fn(),
  loading: false
}

describe("WorldBookListPanel", () => {
  it("renders Name+Description merged column (both visible)", () => {
    render(<WorldBookListPanel {...defaultProps} />)

    expect(screen.getByText("Fantasy Lore")).toBeInTheDocument()
    expect(screen.getByText("Magic systems and creatures")).toBeInTheDocument()

    expect(screen.getByText("Sci-Fi Tech")).toBeInTheDocument()
    expect(screen.getByText("Spaceships and gadgets")).toBeInTheDocument()

    expect(screen.getByText("History Notes")).toBeInTheDocument()
    expect(screen.getByText("Important dates and events")).toBeInTheDocument()
  })

  it("does NOT render BookOpen icon column, 'Attached To', or 'Budget' headers", () => {
    render(<WorldBookListPanel {...defaultProps} />)

    const columnHeaders = screen.getAllByRole("columnheader")
    const headerTexts = columnHeaders.map((h) => h.textContent?.trim() ?? "")

    expect(headerTexts).not.toContain("Attached To")
    expect(headerTexts).not.toContain("Budget")

    // BookOpen icon column rendered as empty header — there should be no
    // column whose sole content is an SVG icon with no text
    expect(screen.queryByTestId("book-open-icon")).not.toBeInTheDocument()
  })

  it("renders only Edit and overflow menu action buttons per row (3 each for 3 books)", () => {
    render(<WorldBookListPanel {...defaultProps} />)

    const editButtons = screen.getAllByRole("button", { name: /^edit /i })
    expect(editButtons).toHaveLength(3)

    const overflowButtons = screen.getAllByRole("button", { name: /^more actions for /i })
    expect(overflowButtons).toHaveLength(3)
  })

  it("calls onSelectWorldBook when a row is clicked", async () => {
    const user = userEvent.setup()
    const onSelectWorldBook = vi.fn()

    render(
      <WorldBookListPanel
        {...defaultProps}
        onSelectWorldBook={onSelectWorldBook}
      />
    )

    const row = screen.getByText("Fantasy Lore").closest("tr")!
    await user.click(row)

    expect(onSelectWorldBook).toHaveBeenCalledWith(1)
  })

  it("highlights the selected row", () => {
    render(
      <WorldBookListPanel
        {...defaultProps}
        selectedWorldBookId={2}
      />
    )

    const selectedRow = screen.getByText("Sci-Fi Tech").closest("tr")!
    expect(selectedRow.className).toMatch(/bg-primary/)
    expect(selectedRow.className).toMatch(/ring/)
  })

  it("shows disabled status with icon alongside color tag", () => {
    render(<WorldBookListPanel {...defaultProps} />)

    // Sci-Fi Tech is disabled
    const disabledTag = screen.getByText("Disabled")
    expect(disabledTag).toBeInTheDocument()
    // The tag should contain an SVG icon (CirclePause)
    const tagContainer = disabledTag.closest(".ant-tag")!
    const icon = tagContainer.querySelector("svg")
    expect(icon).toBeInTheDocument()
  })

  it("renders overflow menu with correct actions", async () => {
    const user = userEvent.setup()
    const onRowAction = vi.fn()

    render(
      <WorldBookListPanel
        {...defaultProps}
        onRowAction={onRowAction}
      />
    )

    const overflowButtons = screen.getAllByRole("button", {
      name: /^more actions for /i
    })
    await user.click(overflowButtons[0])

    await waitFor(() => {
      expect(screen.getByText("Manage Entries")).toBeInTheDocument()
      expect(screen.getByText("Duplicate")).toBeInTheDocument()
      expect(screen.getByText("Quick Attach Characters")).toBeInTheDocument()
      expect(screen.getByText("Export JSON")).toBeInTheDocument()
      expect(screen.getByText("Statistics")).toBeInTheDocument()
      expect(screen.getByText("Delete")).toBeInTheDocument()
    })

    // Click one of the menu items to verify onRowAction is called
    await user.click(screen.getByText("Manage Entries"))
    expect(onRowAction).toHaveBeenCalledWith("entries", mockBooks[0])
  })
})
