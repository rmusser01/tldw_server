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
    title: "Research plan",
    content: "# Research plan\n\nFocus on dataset coverage and error analysis.",
    updated_at: new Date().toISOString(),
    deleted: false,
    keywords: []
  }
]

describe("NotesListPanel stage 14 preview strategy", () => {
  it("uses the first non-title content line as preview when heading matches title", () => {
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
        onSelectNote={vi.fn()}
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

    expect(
      screen.getByText("Focus on dataset coverage and error analysis.")
    ).toBeInTheDocument()
  })
})
