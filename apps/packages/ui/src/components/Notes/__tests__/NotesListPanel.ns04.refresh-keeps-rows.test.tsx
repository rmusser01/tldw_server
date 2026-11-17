import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import NotesListPanel from "../NotesListPanel"
import type { NoteListItem } from "../types"

// NS-04: a background refresh keeps the rows on screen (no spinner swap, so
// the scroll position survives); only the first load shows a loading state.

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?: string | { defaultValue?: string; [key: string]: unknown }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

const makeNotes = (count: number): NoteListItem[] =>
  Array.from({ length: count }, (_, index) => ({
    id: `n${index + 1}`,
    title: `Note ${index + 1} title`,
    content: `Body of note ${index + 1}`,
    updated_at: "2026-10-02T15:00:00.000Z",
    deleted: false,
    keywords: index === 0 ? ["research"] : []
  }))

type PanelOverrides = Partial<React.ComponentProps<typeof NotesListPanel>>

const renderPanel = (overrides: PanelOverrides = {}) => {
  const props: React.ComponentProps<typeof NotesListPanel> = {
    listMode: "active",
    searchQuery: "",
    bulkSelectedIds: [],
    isOnline: true,
    isFetching: false,
    demoEnabled: false,
    capsLoading: false,
    capabilities: { hasNotes: true } as any,
    notes: makeNotes(3),
    total: 3,
    page: 1,
    pageSize: 20,
    selectedId: null,
    onSelectNote: vi.fn(),
    onToggleBulkSelection: vi.fn(),
    onTogglePinned: vi.fn(),
    onChangePage: vi.fn(),
    onCreateNote: vi.fn(),
    onResetEditor: vi.fn(),
    onOpenSettings: vi.fn(),
    onOpenHealth: vi.fn(),
    onRestoreNote: vi.fn(),
    onExportAllMd: vi.fn(),
    onExportAllCsv: vi.fn(),
    onExportAllJson: vi.fn(),
    ...overrides
  }
  const utils = render(<NotesListPanel {...props} />)
  return { ...utils, props }
}

describe("NotesListPanel NS-04 background refresh", () => {
  it("keeps rows visible and marks the list busy while refetching", () => {
    renderPanel({ isFetching: true })

    expect(screen.queryByTestId("notes-list-loading")).not.toBeInTheDocument()
    expect(screen.getByTestId("notes-open-button-n1")).toBeInTheDocument()
    expect(screen.getByRole("list", { name: "Notes" })).toHaveAttribute("aria-busy", "true")
    expect(screen.getByTestId("notes-list-refreshing")).toBeInTheDocument()
  })

  it("does not mark the list busy once the refresh settles", () => {
    renderPanel({ isFetching: false })

    expect(screen.getByRole("list", { name: "Notes" })).not.toHaveAttribute("aria-busy", "true")
    expect(screen.queryByTestId("notes-list-refreshing")).not.toBeInTheDocument()
  })

  it("shows the loading state only when there are no rows yet", () => {
    renderPanel({ isFetching: true, notes: undefined, total: 0 })

    expect(screen.getByTestId("notes-list-loading")).toBeInTheDocument()
  })

  it("hides rows left over from another page or search while the new one loads", () => {
    renderPanel({ isFetching: true, isPlaceholderData: true })

    expect(screen.getByTestId("notes-list-loading")).toBeInTheDocument()
    expect(screen.queryByTestId("notes-open-button-n1")).not.toBeInTheDocument()
    expect(screen.queryByTestId("notes-list-refreshing")).not.toBeInTheDocument()
  })
})
