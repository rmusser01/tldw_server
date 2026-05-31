import React from "react"
import { render, screen } from "@testing-library/react"
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
      if (key === "option:notesSearch.exportProgressFailedBatches") {
        return "Localized {{count}} failed batch warning"
      }
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
    id: "n-export",
    title: "Exportable note",
    content: "content",
    updated_at: "2026-02-18T00:00:00.000Z",
    deleted: false,
    keywords: []
  }
]

describe("NotesListPanel stage 46 export progress copy", () => {
  it("names failed export batches and warns that the export may be partial", () => {
    render(
      <NotesListPanel
        listMode="active"
        searchQuery=""
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
        onCreateNote={vi.fn()}
        onSelectNote={vi.fn()}
        onChangePage={vi.fn()}
        onResetEditor={vi.fn()}
        onOpenSettings={vi.fn()}
        onOpenHealth={vi.fn()}
        onRestoreNote={vi.fn()}
        onExportAllMd={vi.fn()}
        onExportAllCsv={vi.fn()}
        onExportAllJson={vi.fn()}
        exportProgress={{
          format: "md",
          fetchedNotes: 100,
          fetchedPages: 1,
          failedBatches: 1
        }}
      />
    )

    expect(screen.getByTestId("notes-export-progress")).toHaveTextContent(
      "Localized 1 failed batch warning"
    )
  })
})
