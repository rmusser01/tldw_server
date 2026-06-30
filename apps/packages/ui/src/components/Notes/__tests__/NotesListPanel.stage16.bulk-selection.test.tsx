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
    title: "One",
    content: "one",
    updated_at: new Date().toISOString(),
    deleted: false,
    keywords: []
  },
  {
    id: "n2",
    title: "Two",
    content: "two",
    updated_at: new Date().toISOString(),
    deleted: false,
    keywords: []
  },
  {
    id: "n3",
    title: "Three",
    content: "three",
    updated_at: new Date().toISOString(),
    deleted: false,
    keywords: []
  }
]

describe("NotesListPanel stage 16 bulk selection", () => {
  it("emits checkbox selection toggles including shift-click intent", () => {
    const onToggleBulkSelection = vi.fn()
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
        total={3}
        page={1}
        pageSize={20}
        selectedId={null}
        onSelectNote={vi.fn()}
        onToggleBulkSelection={onToggleBulkSelection}
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

    fireEvent.click(screen.getByTestId("notes-select-checkbox-n1"))
    expect(onToggleBulkSelection).toHaveBeenLastCalledWith("n1", true, false)

    rerender(
      <NotesListPanel
        listMode="active"
        searchQuery=""
        bulkSelectedIds={["n1"]}
        isOnline
        isFetching={false}
        demoEnabled={false}
        capsLoading={false}
        capabilities={{ hasNotes: true } as any}
        notes={notes}
        total={3}
        page={1}
        pageSize={20}
        selectedId={null}
        onSelectNote={vi.fn()}
        onToggleBulkSelection={onToggleBulkSelection}
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

    fireEvent.click(screen.getByTestId("notes-select-checkbox-n3"), { shiftKey: true })
    expect(onToggleBulkSelection).toHaveBeenLastCalledWith("n3", true, true)
  })
})
