import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import NotesListPanel from "../NotesListPanel"
import type { NoteListItem } from "../types"

// AX-03: the notes list is a single Tab stop with keyboard row actions.

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

const rowButton = (index: number) =>
  screen.getByRole("button", { name: `Open note Note ${index} title` })

describe("NotesListPanel AX-03 keyboard access", () => {
  it("exposes the results as one Tab stop with row controls out of the Tab order", () => {
    renderPanel({ selectedId: "n2" })

    expect(rowButton(1)).toHaveAttribute("tabindex", "-1")
    expect(rowButton(2)).toHaveAttribute("tabindex", "0")
    expect(rowButton(3)).toHaveAttribute("tabindex", "-1")
    for (const checkbox of screen.getAllByRole("checkbox")) {
      expect(checkbox).toHaveAttribute("tabindex", "-1")
    }
    expect(screen.getByTestId("notes-pin-toggle-n1")).toHaveAttribute("tabindex", "-1")
  })

  it("makes the first row the Tab stop when no note is open and follows focus", () => {
    renderPanel()

    expect(rowButton(1)).toHaveAttribute("tabindex", "0")
    rowButton(1).focus()
    fireEvent.keyDown(rowButton(1), { key: "ArrowDown" })
    expect(rowButton(2)).toHaveFocus()
    expect(rowButton(2)).toHaveAttribute("tabindex", "0")
    expect(rowButton(1)).toHaveAttribute("tabindex", "-1")
  })

  it("supports Home, End, PageUp and PageDown", () => {
    renderPanel({ notes: makeNotes(25), total: 25 })

    rowButton(1).focus()
    fireEvent.keyDown(rowButton(1), { key: "End" })
    expect(rowButton(25)).toHaveFocus()

    fireEvent.keyDown(rowButton(25), { key: "Home" })
    expect(rowButton(1)).toHaveFocus()

    fireEvent.keyDown(rowButton(1), { key: "PageDown" })
    expect(rowButton(11)).toHaveFocus()

    fireEvent.keyDown(rowButton(11), { key: "PageUp" })
    expect(rowButton(1)).toHaveFocus()
  })

  it("selects with X or Space and pins with P on the focused row", () => {
    const onToggleBulkSelection = vi.fn()
    const onTogglePinned = vi.fn()
    renderPanel({ onToggleBulkSelection, onTogglePinned, bulkSelectedIds: ["n2"] })

    fireEvent.keyDown(rowButton(1), { key: "x" })
    expect(onToggleBulkSelection).toHaveBeenLastCalledWith("n1", true, false)

    fireEvent.keyDown(rowButton(2), { key: " " })
    expect(onToggleBulkSelection).toHaveBeenLastCalledWith("n2", false, false)

    fireEvent.keyDown(rowButton(3), { key: "p" })
    expect(onTogglePinned).toHaveBeenCalledWith("n3")
  })

  it("opens through the keyboard handler on Enter so the editor can take focus", () => {
    const onSelectNote = vi.fn()
    const onOpenNoteFromKeyboard = vi.fn()
    renderPanel({ onSelectNote, onOpenNoteFromKeyboard })

    fireEvent.keyDown(rowButton(2), { key: "Enter" })
    expect(onOpenNoteFromKeyboard).toHaveBeenCalledWith("n2")
    expect(onSelectNote).not.toHaveBeenCalled()

    fireEvent.click(rowButton(3))
    expect(onSelectNote).toHaveBeenCalledWith("n3")
  })

  it("keeps the accessible name to the title and describes the row details", () => {
    renderPanel()

    const button = rowButton(1)
    const describedBy = String(button.getAttribute("aria-describedby") || "")
    const description = describedBy
      .split(/\s+/)
      .filter(Boolean)
      .map((id) => document.getElementById(id)?.textContent || "")
      .join(" ")
    expect(description).toContain("Body of note 1")
    expect(description).toContain("research")
    expect(description).toMatch(/Enter/)
  })
})
