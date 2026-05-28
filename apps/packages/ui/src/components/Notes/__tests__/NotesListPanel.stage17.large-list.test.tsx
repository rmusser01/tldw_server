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

const largeNotes: NoteListItem[] = Array.from({ length: 550 }, (_, index) => ({
  id: `n-${index + 1}`,
  title: `Note ${index + 1}`,
  content: `Line ${index + 1}`,
  updated_at: new Date().toISOString(),
  deleted: false,
  keywords: index % 2 === 0 ? ["tag"] : []
}))

describe("NotesListPanel stage 17 large list fallback", () => {
  it("renders large note sets with pagination fallback and preserved row selection controls", () => {
    const onToggleBulkSelection = vi.fn()
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
        notes={largeNotes}
        total={550}
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

    expect(screen.getByText("Note 1")).toBeInTheDocument()
    expect(screen.getByText("Note 550")).toBeInTheDocument()
    fireEvent.click(screen.getByTestId("notes-select-checkbox-n-550"))
    expect(onToggleBulkSelection).toHaveBeenCalledWith("n-550", true, false)
  })
})
