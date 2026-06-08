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

const notes: NoteListItem[] = [
  {
    id: "n1",
    title: "Alpha note",
    content: "alpha",
    updated_at: new Date().toISOString(),
    deleted: false,
    keywords: []
  },
  {
    id: "n2",
    title: "Beta note",
    content: "beta",
    updated_at: new Date().toISOString(),
    deleted: false,
    keywords: []
  }
]

describe("NotesListPanel stage 18 selected-state accessibility", () => {
  it("keeps note rows exposed as buttons while marking the current note", () => {
    const onSelectNote = vi.fn()
    const { rerender } = render(
      <NotesListPanel
        listMode="active"
        searchQuery=""
        bulkSelectedIds={[]}
        isOnline
        isFetching={false}
        demoEnabled={false}
        capsLoading={false}
        capabilities={{ hasNotes: true } as any}
        notes={notes}
        total={2}
        page={1}
        pageSize={20}
        selectedId="n1"
        onSelectNote={onSelectNote}
        onToggleBulkSelection={vi.fn()}
        onChangePage={vi.fn()}
        onCreateNote={vi.fn()}
        onResetEditor={vi.fn()}
        onOpenSettings={vi.fn()}
        onOpenHealth={vi.fn()}
        onRestoreNote={vi.fn()}
        onExportAllMd={vi.fn()}
        onExportAllCsv={vi.fn()}
        onExportAllJson={vi.fn()}
      />
    )

    expect(screen.getByRole("list", { name: "Notes" })).toBeInTheDocument()
    expect(screen.queryByRole("option")).not.toBeInTheDocument()

    const noteOneButton = screen.getByRole("button", { name: "Open note Alpha note" })
    const noteTwoButton = screen.getByRole("button", { name: "Open note Beta note" })
    expect(noteOneButton).toHaveAttribute("aria-current", "true")
    expect(noteOneButton).not.toHaveAttribute("aria-selected")
    expect(noteTwoButton).not.toHaveAttribute("aria-current")
    expect(noteTwoButton).not.toHaveAttribute("aria-selected")

    fireEvent.click(noteTwoButton)
    expect(onSelectNote).toHaveBeenCalledWith("n2")

    rerender(
      <NotesListPanel
        listMode="active"
        searchQuery=""
        bulkSelectedIds={[]}
        isOnline
        isFetching={false}
        demoEnabled={false}
        capsLoading={false}
        capabilities={{ hasNotes: true } as any}
        notes={notes}
        total={2}
        page={1}
        pageSize={20}
        selectedId="n2"
        onSelectNote={onSelectNote}
        onToggleBulkSelection={vi.fn()}
        onChangePage={vi.fn()}
        onCreateNote={vi.fn()}
        onResetEditor={vi.fn()}
        onOpenSettings={vi.fn()}
        onOpenHealth={vi.fn()}
        onRestoreNote={vi.fn()}
        onExportAllMd={vi.fn()}
        onExportAllCsv={vi.fn()}
        onExportAllJson={vi.fn()}
      />
    )

    expect(screen.getByRole("button", { name: "Open note Alpha note" })).not.toHaveAttribute("aria-current")
    expect(screen.getByRole("button", { name: "Open note Beta note" })).toHaveAttribute("aria-current", "true")
  })

  it("moves focus between note buttons with arrow keys", () => {
    render(
      <NotesListPanel
        listMode="active"
        searchQuery=""
        bulkSelectedIds={[]}
        isOnline
        isFetching={false}
        demoEnabled={false}
        capsLoading={false}
        capabilities={{ hasNotes: true } as any}
        notes={notes}
        total={2}
        page={1}
        pageSize={20}
        selectedId="n1"
        onSelectNote={vi.fn()}
        onToggleBulkSelection={vi.fn()}
        onChangePage={vi.fn()}
        onResetEditor={vi.fn()}
        onOpenSettings={vi.fn()}
        onOpenHealth={vi.fn()}
        onRestoreNote={vi.fn()}
        onExportAllMd={vi.fn()}
        onExportAllCsv={vi.fn()}
        onExportAllJson={vi.fn()}
      />
    )

    const noteOneButton = screen.getByRole("button", { name: "Open note Alpha note" })
    const noteTwoButton = screen.getByRole("button", { name: "Open note Beta note" })
    noteOneButton.focus()

    fireEvent.keyDown(noteOneButton, { key: "ArrowDown" })
    expect(noteTwoButton).toHaveFocus()

    fireEvent.keyDown(noteTwoButton, { key: "ArrowUp" })
    expect(noteOneButton).toHaveFocus()
  })

  it("keeps literal replacement tokens in note titles inside open-note labels", () => {
    render(
      <NotesListPanel
        listMode="active"
        searchQuery=""
        bulkSelectedIds={[]}
        isOnline
        isFetching={false}
        demoEnabled={false}
        capsLoading={false}
        capabilities={{ hasNotes: true } as any}
        notes={[
          {
            id: "n-token",
            title: "Budget $& Review",
            content: "alpha",
            updated_at: new Date().toISOString(),
            deleted: false,
            keywords: []
          }
        ]}
        total={1}
        page={1}
        pageSize={20}
        selectedId="n-token"
        onSelectNote={vi.fn()}
        onToggleBulkSelection={vi.fn()}
        onChangePage={vi.fn()}
        onResetEditor={vi.fn()}
        onOpenSettings={vi.fn()}
        onOpenHealth={vi.fn()}
        onRestoreNote={vi.fn()}
        onExportAllMd={vi.fn()}
        onExportAllCsv={vi.fn()}
        onExportAllJson={vi.fn()}
      />
    )

    expect(screen.getByRole("button", { name: "Open note Budget $& Review" })).toBeInTheDocument()
  })
})
