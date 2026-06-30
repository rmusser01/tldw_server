import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import NotesListPanel from "../NotesListPanel"
import type { NoteListItem } from "../types"

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

const baseProps = {
  listMode: "active" as const,
  searchQuery: "",
  isOnline: true,
  isFetching: false,
  demoEnabled: false,
  capsLoading: false,
  capabilities: { hasNotes: true } as any,
  notes: [] as NoteListItem[],
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
  onExportAllJson: vi.fn()
}

describe("NotesListPanel empty and error states", () => {
  it("shows no-results guidance and clear-filters action for active filters", async () => {
    const onClearFilters = vi.fn()

    render(
      <NotesListPanel
        {...baseProps}
        searchQuery="alpha"
        hasActiveFilters
        onClearFilters={onClearFilters}
      />
    )

    expect(await screen.findByText("No notes match these filters")).toBeInTheDocument()
    expect(screen.queryByText("No notes yet")).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Clear filters" }))
    expect(onClearFilters).toHaveBeenCalledTimes(1)
  })

  it("keeps cached notes visible and flags them as stale after refresh failure", () => {
    const note = {
      id: "n-1",
      title: "Cached note",
      content: "Cached content",
      updated_at: "2026-02-18T12:00:00.000Z",
      deleted: false,
      keywords: []
    } satisfies NoteListItem

    render(
      <NotesListPanel
        {...baseProps}
        notes={[note]}
        total={1}
        listError={new Error("Backend unavailable")}
        isStaleResults
        onRetry={vi.fn()}
      />
    )

    expect(screen.getByText("Cached note")).toBeInTheDocument()
    expect(screen.getByText("Showing saved results")).toBeInTheDocument()
    expect(screen.getByText("Refresh failed. Retry to load the latest notes.")).toBeInTheDocument()
  })
})
