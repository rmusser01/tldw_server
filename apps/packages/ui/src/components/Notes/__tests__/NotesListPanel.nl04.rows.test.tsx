import React from "react"
import { fireEvent, render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import NotesListPanel from "../NotesListPanel"
import type { ServerCapabilities } from "@/services/tldw/server-capabilities"
import type { NoteListItem } from "../types"

// NL-04 (#3101): list rows read as plain text and stay dense.

const APP_LANGUAGE = "de-DE"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    i18n: { language: "de-DE" },
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

  it("replaces the conversation line with a 'From chat' link and no ids", () => {
    const conversationId = "acd86f3a-492e-4a8b-90fb-ff282b9721fb"
    const onOpenLinkedChat = vi.fn()
    const onSelectNote = vi.fn()
    const { container } = renderList(
      [baseNote({ conversation_id: conversationId, message_id: "9f1c2d3e-0000-4b8b-90fb-0123456789ab" })],
      { onOpenLinkedChat, onSelectNote }
    )
    const link = screen.getByTestId("notes-from-chat-n1")
    expect(link).toHaveTextContent("From chat")
    expect(container).not.toHaveTextContent(conversationId)
    expect(container).not.toHaveTextContent("9f1c2d3e")
    expect(container).not.toHaveTextContent("msg ")
    expect(container).not.toHaveTextContent("Linked to conversation")
    // The link is not nested inside the row's open-note button.
    expect(screen.getByTestId("notes-open-button-n1")).not.toContainElement(link)

    fireEvent.click(link)
    expect(onOpenLinkedChat).toHaveBeenCalledWith(conversationId)
    expect(onSelectNote).not.toHaveBeenCalled()
  })

  it("names the chat in the 'From chat' link once its title resolves", () => {
    renderList([baseNote({ conversation_id: "conv-7" })], {
      onOpenLinkedChat: vi.fn(),
      conversationLabelById: { "conv-7": "Research session" }
    })
    const link = screen.getByTestId("notes-from-chat-n1")
    expect(link).toHaveTextContent("From chat: Research session")
    expect(within(link).getByText("Research session")).toBeInTheDocument()
  })

  it("shows a relative date on the title line, with the full date in a tooltip", async () => {
    const updatedAt = new Date(Date.now() - 3 * 60 * 60 * 1000)
    const fullDate = updatedAt.toLocaleString(APP_LANGUAGE)
    renderList([baseNote({ updated_at: updatedAt.toISOString() })])

    const date = screen.getByTestId("notes-row-date-n1")
    expect(date).toHaveTextContent("3h ago")
    expect(document.body.textContent).not.toContain(fullDate)
    expect(date).toHaveAttribute("dateTime", updatedAt.toISOString())
    // Denser rows: the date shares the title's line instead of taking its own.
    expect(date.parentElement).toContainElement(screen.getByText("Daily log"))

    fireEvent.mouseEnter(date)
    expect(await screen.findByText(fullDate)).toBeInTheDocument()
  })

  it("shows when a Trash note was deleted as a relative date", () => {
    const deletedAt = new Date(Date.now() - 2 * 24 * 60 * 60 * 1000)
    renderList([baseNote({ updated_at: deletedAt.toISOString(), deleted: true })], { listMode: "trash" })
    expect(screen.getByTestId("notes-row-date-n1")).toHaveTextContent("2d ago")
    expect(document.body.textContent).not.toContain(deletedAt.toLocaleString(APP_LANGUAGE))
  })

  it("falls back to a date in the app's language after a week", () => {
    const updatedAt = new Date(Date.now() - 30 * 24 * 60 * 60 * 1000)
    renderList([baseNote({ updated_at: updatedAt.toISOString() })])
    expect(screen.getByTestId("notes-row-date-n1")).toHaveTextContent(updatedAt.toLocaleDateString(APP_LANGUAGE))
  })

  it("gives the 'From chat' link the row controls' touch target", () => {
    renderList([baseNote({ conversation_id: "conv-7" })], { onOpenLinkedChat: vi.fn() })
    expect(screen.getByTestId("notes-from-chat-n1")).toHaveClass(
      "min-h-6",
      "[@media(pointer:coarse)]:min-h-11"
    )
  })

  it("describes the row with its 'From chat' link", () => {
    renderList([baseNote({ conversation_id: "conv-7" })], {
      onOpenLinkedChat: vi.fn(),
      conversationLabelById: { "conv-7": "Research session" }
    })
    const link = screen.getByTestId("notes-from-chat-n1")
    const describedBy = screen.getByTestId("notes-open-button-n1").getAttribute("aria-describedby") ?? ""
    expect(link.id).not.toBe("")
    expect(describedBy.split(" ")).toContain(link.id)
  })

  it("opens a row's chat with C, and documents it in the row hint", () => {
    const onOpenLinkedChat = vi.fn()
    renderList(
      [baseNote({ conversation_id: "conv-7" }), baseNote({ id: "n2", title: "Plain note" })],
      { onOpenLinkedChat }
    )
    fireEvent.keyDown(screen.getByTestId("notes-open-button-n2"), { key: "c" })
    expect(onOpenLinkedChat).not.toHaveBeenCalled()
    fireEvent.keyDown(screen.getByTestId("notes-open-button-n1"), { key: "c" })
    expect(onOpenLinkedChat).toHaveBeenCalledWith("conv-7")
    expect(document.body.textContent).toContain("C to open the chat a note came from")
  })

  it("shows no 'From chat' link when the list cannot open chats", () => {
    renderList([baseNote({ conversation_id: "conv-7" })])
    expect(screen.queryByTestId("notes-from-chat-n1")).not.toBeInTheDocument()
  })
})
