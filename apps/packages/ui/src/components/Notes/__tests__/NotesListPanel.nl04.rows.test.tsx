import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import NotesListPanel from "../NotesListPanel"
import type { ServerCapabilities } from "@/services/tldw/server-capabilities"
import type { NoteListItem } from "../types"

// NL-04 (#3101): list rows read as plain text and stay dense.

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValueOrOptions?: string | { defaultValue?: string; [key: string]: unknown }) => {
      const template =
        typeof defaultValueOrOptions === "string" ? defaultValueOrOptions : defaultValueOrOptions?.defaultValue
      if (!template) return key
      if (typeof defaultValueOrOptions !== "object") return template
      return template.replace(/\{\{(\w+)\}\}/g, (_, name: string) => String(defaultValueOrOptions[name] ?? ""))
    }
  })
}))

const baseNote = (overrides: Partial<NoteListItem> = {}): NoteListItem => ({
  id: "n1",
  title: "Daily log",
  content: "plain body",
  updated_at: new Date().toISOString(),
  deleted: false,
  keywords: [],
  ...overrides
})

const renderList = (
  notes: NoteListItem[],
  props: Partial<React.ComponentProps<typeof NotesListPanel>> = {}
) =>
  render(
    <NotesListPanel
      listMode="active"
      searchQuery=""
      isOnline
      isFetching={false}
      demoEnabled={false}
      capsLoading={false}
      capabilities={{ hasNotes: true } as ServerCapabilities}
      notes={notes}
      total={notes.length}
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
      {...props}
    />
  )

describe("NotesListPanel NL-04 rows", () => {
  it("previews one line of plain text instead of raw Markdown", () => {
    renderList([
      baseNote({ content: "# Daily log\n\n```sql\n**Mood:** calm, see [[Project Alpha]]\n```" })
    ])
    const preview = screen.getByText("Mood: calm, see Project Alpha")
    expect(preview).toBeInTheDocument()
    expect(document.body.textContent).not.toContain("**")
    expect(document.body.textContent).not.toContain("[[")
    expect(document.body.textContent).not.toContain("```")
  })

  it("previews plain text in Trash rows too", () => {
    renderList([baseNote({ content: "## _Removed_ idea", deleted: true })], { listMode: "trash" })
    expect(screen.getByText("Removed idea")).toBeInTheDocument()
  })
})
