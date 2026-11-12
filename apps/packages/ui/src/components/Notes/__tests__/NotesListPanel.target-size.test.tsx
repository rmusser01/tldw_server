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
      if (defaultValueOrOptions?.defaultValue) {
        return String(defaultValueOrOptions.defaultValue).replace(
          /\{\{(\w+)\}\}/g,
          (_match, token) => String(defaultValueOrOptions[token] ?? "")
        )
      }
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
  }
]

const renderPanel = (onToggleBulkSelection = vi.fn()) =>
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
      total={1}
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

// AX-16 (#3124): the bulk-select checkbox was a bare 16x16 px input. Its hit
// area must be at least 24x24 px (44x44 on touch screens) and must not shrink.
describe("NotesListPanel row target sizes", () => {
  it("gives the bulk-select checkbox a 24px hit area that selects the note", () => {
    const onToggleBulkSelection = vi.fn()
    renderPanel(onToggleBulkSelection)

    const checkbox = screen.getByRole("checkbox", {
      name: "Select note Alpha note for bulk actions"
    })
    const hitArea = checkbox.closest("label")
    expect(hitArea).not.toBeNull()
    expect(hitArea).toHaveClass(
      "min-h-6",
      "min-w-6",
      "shrink-0",
      "[@media(pointer:coarse)]:min-h-11",
      "[@media(pointer:coarse)]:min-w-11"
    )

    fireEvent.click(hitArea!)
    expect(onToggleBulkSelection).toHaveBeenCalledWith("n1", true, false)
  })

  it("keeps the pin button at 24px or more and stops it shrinking", () => {
    renderPanel()

    const pin = screen.getByRole("button", { name: "Pin note Alpha note" })
    expect(pin).toHaveClass(
      "min-h-6",
      "min-w-6",
      "shrink-0",
      "[@media(pointer:coarse)]:min-h-11",
      "[@media(pointer:coarse)]:min-w-11"
    )
  })
})
