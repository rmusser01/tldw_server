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
    id: "n-recent",
    title: "Linked and tagged note",
    content: "content",
    updated_at: new Date().toISOString(),
    deleted: false,
    keywords: ["research"],
    conversation_id: "conv-1"
  },
  {
    id: "n-plain",
    title: "Plain note",
    content: "content",
    updated_at: "2020-01-01T00:00:00.000Z",
    deleted: false,
    keywords: []
  }
]

describe("NotesListPanel stage 15 metadata badges", () => {
  it("shows keyword, backlink, and recent-edit badges for qualifying notes", () => {
    const { container } = render(
      <NotesListPanel
        listMode="active"
        searchQuery=""
        isOnline
        isFetching={false}
        demoEnabled={false}
        capsLoading={false}
        capabilities={{ hasNotes: true } as any}
        notes={notes}
        total={2}
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

    const badgeRow = screen.getByTestId("notes-item-badges-n-recent")
    expect(badgeRow.querySelectorAll("svg").length).toBe(3)
    expect(screen.queryByTestId("notes-item-badges-n-plain")).not.toBeInTheDocument()
    expect(container).toHaveTextContent("Linked to conversation")
  })
})
