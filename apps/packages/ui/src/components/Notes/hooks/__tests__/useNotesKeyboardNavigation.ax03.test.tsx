import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { useNotesKeyboardNavigation } from "../useNotesKeyboardNavigation"

// AX-03 follow-ups: Escape that closes an IME candidate window stays in the
// editor; Alt+Arrow note switching does not run in Trash or behind a dialog;
// Enter waits for the rich editor to become editable before focusing it.

const NOTES = [{ id: "n1" }, { id: "n2" }, { id: "n3" }]

type HarnessProps = {
  listMode?: "active" | "trash"
  selectNote: (id: string | number) => Promise<boolean | void>
  onReady?: (api: ReturnType<typeof useNotesKeyboardNavigation>) => void
}

const Harness = ({ listMode = "active", selectNote, onReady }: HarnessProps) => {
  const richEditorRef = React.useRef<HTMLDivElement>(null)
  const contentTextareaRef = React.useRef<HTMLTextAreaElement>(null)
  const api = useNotesKeyboardNavigation({
    visibleNotes: NOTES,
    selectedId: "n1",
    selectNote,
    richEditorRef,
    contentTextareaRef,
    editorRegionId: "editor-region",
    listMode
  })
  onReady?.(api)
  return (
    <div>
      <div data-notes-list role="list">
        {NOTES.map((note) => (
          <button key={note.id} type="button" data-notes-list-item data-note-row-id={note.id} tabIndex={note.id === "n1" ? 0 : -1}>
            {note.id}
          </button>
        ))}
      </div>
      <section id="editor-region" tabIndex={-1}>
        <input aria-label="Title" />
        <div ref={richEditorRef} role="textbox" aria-label="Note content" data-testid="rich-editor" />
      </section>
      <div role="dialog" aria-modal="true">
        <button type="button">Dialog button</button>
      </div>
    </div>
  )
}

const nextFrames = async (count = 3) => {
  for (let index = 0; index < count; index += 1) {
    await act(async () => {
      await new Promise((resolve) => window.requestAnimationFrame(() => resolve(undefined)))
    })
  }
}

describe("useNotesKeyboardNavigation AX-03 follow-ups", () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("keeps focus in the editor when Escape closes an IME candidate window", async () => {
    render(<Harness selectNote={vi.fn(async () => true)} />)
    const title = screen.getByLabelText("Title")
    title.focus()

    fireEvent.keyDown(title, { key: "Escape", isComposing: true })
    expect(title).toHaveFocus()

    fireEvent.keyDown(title, { key: "Escape", keyCode: 229 })
    expect(title).toHaveFocus()

    fireEvent.keyDown(title, { key: "Escape" })
    expect(screen.getByRole("button", { name: "n1" })).toHaveFocus()
  })

  it("does not switch notes with Alt+Arrow in Trash", () => {
    const selectNote = vi.fn(async () => true)
    render(<Harness listMode="trash" selectNote={selectNote} />)

    fireEvent.keyDown(screen.getByRole("button", { name: "n1" }), { key: "ArrowDown", altKey: true })
    expect(selectNote).not.toHaveBeenCalled()
  })

  it("does not switch notes with Alt+Arrow behind an open dialog", () => {
    const selectNote = vi.fn(async () => true)
    render(<Harness selectNote={selectNote} />)

    fireEvent.keyDown(screen.getByRole("button", { name: "Dialog button" }), { key: "ArrowDown", altKey: true })
    expect(selectNote).not.toHaveBeenCalled()
  })

  it("still switches to the next note with Alt+ArrowDown from the list", async () => {
    const selectNote = vi.fn(async () => true)
    render(<Harness selectNote={selectNote} />)

    fireEvent.keyDown(screen.getByRole("button", { name: "n1" }), { key: "ArrowDown", altKey: true })
    await waitFor(() => expect(selectNote).toHaveBeenCalledWith("n2"))
  })

  it("focuses the rich editor once it becomes editable after the note opens", async () => {
    let api: ReturnType<typeof useNotesKeyboardNavigation> | null = null
    const selectNote = vi.fn(async () => {
      // The editor turns editable a frame after the first focus attempt.
      window.requestAnimationFrame(() => {
        window.requestAnimationFrame(() => {
          screen.getByTestId("rich-editor").setAttribute("contenteditable", "true")
        })
      })
      return true
    })
    render(<Harness selectNote={selectNote} onReady={(value) => { api = value }} />)
    screen.getByRole("button", { name: "n1" }).focus()

    await act(async () => {
      await api!.openNoteAndFocusEditor("n2")
    })
    await nextFrames(4)

    expect(screen.getByTestId("rich-editor")).toHaveFocus()
  })
})
