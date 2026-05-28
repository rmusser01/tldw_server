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
  it("toggles aria-selected and aria-current between selected notes", () => {
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

    const noteOneButton = screen.getByTestId("notes-open-button-n1")
    const noteTwoButton = screen.getByTestId("notes-open-button-n2")
    expect(noteOneButton).toHaveAttribute("aria-selected", "true")
    expect(noteOneButton).toHaveAttribute("aria-current", "true")
    expect(noteTwoButton).toHaveAttribute("aria-selected", "false")
    expect(noteTwoButton).not.toHaveAttribute("aria-current")

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

    expect(screen.getByTestId("notes-open-button-n1")).toHaveAttribute("aria-selected", "false")
    expect(screen.getByTestId("notes-open-button-n2")).toHaveAttribute("aria-selected", "true")
    expect(screen.getByTestId("notes-open-button-n2")).toHaveAttribute("aria-current", "true")
  })
})
