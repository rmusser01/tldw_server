/**
 * NL-07: a search row shows where the note matched.
 *
 * The server returns a body excerpt as text runs, with `match` marking each
 * matched term. The row renders it in place of the first line, highlights the
 * matches, and never treats the text as markup.
 */
import React from "react"
import { render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import NotesListPanel from "../NotesListPanel"
import { NoteSearchSnippet, normalizeNoteSearchSnippet } from "../NoteSearchSnippet"
import type { NoteListItem } from "../types"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValueOrOptions?: string | { defaultValue?: string }) => {
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

const renderPanel = (notes: NoteListItem[], searchQuery = "faiss") =>
  render(
    <NotesListPanel
      listMode="active"
      searchQuery={searchQuery}
      isOnline
      isFetching={false}
      demoEnabled={false}
      capsLoading={false}
      capabilities={{ hasNotes: true } as any}
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
    />
  )

describe("NotesListPanel search snippet (NL-07)", () => {
  it("shows the match context in place of the first line", () => {
    renderPanel([
      {
        id: "n1",
        title: "Vector stores",
        content: "Intro line about databases.\nLater we compare FAISS and others.",
        snippet: [
          { text: "…Later we compare ", match: false },
          { text: "FAISS", match: true },
          { text: " and others.", match: false }
        ]
      }
    ])

    const snippet = screen.getByTestId("notes-search-snippet")
    expect(snippet).toHaveTextContent("…Later we compare FAISS and others.")
    expect(within(snippet).getByText("FAISS", { selector: "mark" })).toBeInTheDocument()
    expect(screen.queryByText(/Intro line about databases/)).not.toBeInTheDocument()
    // The row's description points at the snippet.
    const row = screen.getByTestId("notes-open-button-n1")
    expect(row.getAttribute("aria-describedby")).toContain(snippet.id)
  })

  it("keeps the first-line preview when there is no snippet", () => {
    renderPanel([{ id: "n2", title: "FAISS notes", content: "Intro line about databases." }])

    expect(screen.queryByTestId("notes-search-snippet")).not.toBeInTheDocument()
    expect(screen.getByText("Intro line about databases.")).toBeInTheDocument()
  })

  it("renders snippet text as text, never as markup", () => {
    const { container } = render(
      <NoteSearchSnippet
        id="s"
        segments={[
          { text: "<img src=x onerror=alert(1)> ", match: false },
          { text: "<b>FAISS</b>", match: true }
        ]}
      />
    )

    expect(container.querySelector("img")).toBeNull()
    expect(container.querySelector("b")).toBeNull()
    expect(screen.getByText("<b>FAISS</b>", { selector: "mark" })).toBeInTheDocument()
  })
})

describe("normalizeNoteSearchSnippet", () => {
  it("keeps well-formed segments with a match", () => {
    expect(normalizeNoteSearchSnippet([{ text: "a", match: false }, { text: "b", match: true }])).toEqual([
      { text: "a", match: false },
      { text: "b", match: true }
    ])
  })

  it("drops anything else", () => {
    expect(normalizeNoteSearchSnippet(null)).toBeNull()
    expect(normalizeNoteSearchSnippet("<mark>a</mark>")).toBeNull()
    expect(normalizeNoteSearchSnippet([{ text: 1, match: true }])).toBeNull()
    expect(normalizeNoteSearchSnippet([{ text: "a", match: false }])).toBeNull()
  })
})
