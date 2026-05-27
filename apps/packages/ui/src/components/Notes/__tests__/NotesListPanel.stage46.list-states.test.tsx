import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import NotesListPanel from "../NotesListPanel"

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

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionActions: () => ({
    checkOnce: vi.fn()
  })
}))

const renderPanel = (
  overrides: Partial<React.ComponentProps<typeof NotesListPanel>> = {}
) => {
  const props: React.ComponentProps<typeof NotesListPanel> = {
    listMode: "active",
    searchQuery: "",
    isOnline: true,
    isFetching: false,
    demoEnabled: false,
    capsLoading: false,
    capabilities: { hasNotes: true } as any,
    notes: [],
    total: 0,
    page: 1,
    pageSize: 20,
    selectedId: null,
    onSelectNote: vi.fn(),
    onChangePage: vi.fn(),
    onResetEditor: vi.fn(),
    onOpenSettings: vi.fn(),
    onOpenHealth: vi.fn(),
    onRestoreNote: vi.fn(),
    onExportAllMd: vi.fn(),
    onExportAllCsv: vi.fn(),
    onExportAllJson: vi.fn(),
    ...overrides
  }

  render(<NotesListPanel {...props} />)
}

describe("NotesListPanel stage 46 list states", () => {
  it("renders loading state with an accessible status label", () => {
    renderPanel({ isFetching: true })

    expect(screen.getByTestId("notes-list-loading")).toHaveAttribute("role", "status")
    expect(screen.getByText("Loading notes...")).toBeInTheDocument()
  })

  it("renders first-time empty state separately from filtered no-results state", async () => {
    renderPanel({ hasActiveFilters: false })

    expect(await screen.findByText("No notes yet")).toBeInTheDocument()
    expect(screen.queryByText("No notes match your filters")).not.toBeInTheDocument()
  })

  it("renders no-results state for active search and exposes clear filters", async () => {
    const onClearFilters = vi.fn()
    renderPanel({
      searchQuery: "alpha",
      hasActiveFilters: true,
      onClearFilters
    })

    expect(await screen.findByText("No notes match your filters")).toBeInTheDocument()
    expect(screen.getByText('No notes match "alpha".')).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Clear search & filters" }))
    expect(onClearFilters).toHaveBeenCalledTimes(1)
  })

  it("renders list load errors with retry action instead of an empty state", async () => {
    const onRetry = vi.fn()
    renderPanel({
      hasError: true,
      errorMessage: "Server unavailable",
      onRetry
    })

    expect(await screen.findByTestId("notes-list-error-state")).toHaveTextContent(
      "Could not load notes"
    )
    expect(screen.getByText("Server unavailable")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    expect(onRetry).toHaveBeenCalledTimes(1)
  })
})
