import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import NotesEditorHeader from "../NotesEditorHeader"
import NotesListPanel from "../NotesListPanel"
import type { NoteListItem } from "../types"

// XP-14 (#3115): a note's chat backlink must not show raw ids. The
// conversation and message ids move into the tooltip, and a chat whose title
// has not resolved reads "Linked chat" instead of its UUID.

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValueOrOptions?: string | { defaultValue?: string }) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))
vi.mock("@/hooks/useMediaQuery", () => ({ useMobile: () => false }))
vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionActions: () => ({ checkOnce: vi.fn() })
}))

const CONVERSATION_ID = "acd86f3a-492e-4a8b-90fb-ff282b9721fb"
const MESSAGE_ID = "9f1c2d3e-0000-4b8b-90fb-0123456789ab"

const renderHeader = (label: string | null) =>
  render(
    <NotesEditorHeader
      title="Linked note"
      selectedId="note-1"
      backlinkConversationId={CONVERSATION_ID}
      backlinkConversationLabel={label}
      backlinkMessageId={MESSAGE_ID}
      sourceLinks={[]}
      editorDisabled={false}
      openingLinkedChat={false}
      editorMode="edit"
      hasContent
      canSave
      canGenerateFlashcards
      canExport
      isSaving={false}
      canDelete
      isDirty={false}
      onOpenLinkedConversation={() => undefined}
      onOpenSourceLink={() => undefined}
      onChangeEditorMode={() => undefined}
      onCopy={() => undefined}
      onGenerateFlashcards={() => undefined}
      onExport={() => undefined}
      onSave={() => undefined}
      onSaveAndNew={() => undefined}
      onDelete={() => undefined}
    />
  )

describe("note chat backlinks without raw ids (XP-14)", () => {
  it("shows 'Linked chat' in the header while the chat's title is unknown, with the ids in the tooltip", async () => {
    const { container } = renderHeader(null)

    const link = screen.getByText("Linked chat")
    expect(container).not.toHaveTextContent(CONVERSATION_ID)
    expect(container).not.toHaveTextContent(MESSAGE_ID)
    expect(container).not.toHaveTextContent("msg ")

    fireEvent.mouseEnter(link)
    expect(
      await screen.findByText(`Conversation ID: ${CONVERSATION_ID} · Message ID: ${MESSAGE_ID}`)
    ).toBeInTheDocument()
  })

  it("names the chat by its title once it resolves", () => {
    const { container } = renderHeader("Research session")
    expect(container).toHaveTextContent("Linked to conversation: Research session")
    expect(container).not.toHaveTextContent(MESSAGE_ID)
  })

  it("shows 'Linked chat' in the note list instead of the conversation and message ids", () => {
    const notes: NoteListItem[] = [
      {
        id: "n-linked",
        title: "Saved answer",
        content: "content",
        updated_at: new Date().toISOString(),
        deleted: false,
        keywords: [],
        conversation_id: CONVERSATION_ID,
        message_id: MESSAGE_ID
      }
    ]
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

    expect(screen.getByText("Linked chat")).toBeInTheDocument()
    expect(container).not.toHaveTextContent(CONVERSATION_ID)
    expect(container).not.toHaveTextContent(MESSAGE_ID)
  })
})
