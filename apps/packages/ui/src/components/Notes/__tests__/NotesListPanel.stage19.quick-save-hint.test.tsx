import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import NotesListPanel from "../NotesListPanel"

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

describe("NotesListPanel stage 19 quick-save discoverability hint", () => {
  it("shows chat quick-save guidance in the active-notes empty state", async () => {
    render(
      <NotesListPanel
        listMode="active"
        searchQuery=""
        isOnline
        isFetching={false}
        demoEnabled={false}
        capsLoading={false}
        capabilities={{ hasNotes: true } as any}
        notes={[]}
        total={0}
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
      await screen.findByText("You can also create notes directly from chat messages using quick save.")
    ).toBeInTheDocument()
  })

  it("starts a new note from the active-notes empty-state primary action", async () => {
    const onCreateNote = vi.fn()
    const onResetEditor = vi.fn()

    render(
      <NotesListPanel
        listMode="active"
        searchQuery=""
        isOnline
        isFetching={false}
        demoEnabled={false}
        capsLoading={false}
        capabilities={{ hasNotes: true } as any}
        notes={[]}
        total={0}
        page={1}
        pageSize={20}
        selectedId={null}
        onSelectNote={vi.fn()}
        onChangePage={vi.fn()}
        onCreateNote={onCreateNote}
        onResetEditor={onResetEditor}
        onOpenSettings={vi.fn()}
        onOpenHealth={vi.fn()}
        onRestoreNote={vi.fn()}
        onExportAllMd={vi.fn()}
        onExportAllCsv={vi.fn()}
        onExportAllJson={vi.fn()}
      />
    )

    fireEvent.click(await screen.findByRole("button", { name: "Create note" }))

    expect(onCreateNote).toHaveBeenCalledTimes(1)
    expect(onResetEditor).not.toHaveBeenCalled()
  })

  it("keeps the trash empty-state primary action scoped to returning to active notes", async () => {
    const onCreateNote = vi.fn()
    const onResetEditor = vi.fn()

    render(
      <NotesListPanel
        listMode="trash"
        searchQuery=""
        isOnline
        isFetching={false}
        demoEnabled={false}
        capsLoading={false}
        capabilities={{ hasNotes: true } as any}
        notes={[]}
        total={0}
        page={1}
        pageSize={20}
        selectedId={null}
        onSelectNote={vi.fn()}
        onChangePage={vi.fn()}
        onCreateNote={onCreateNote}
        onResetEditor={onResetEditor}
        onOpenSettings={vi.fn()}
        onOpenHealth={vi.fn()}
        onRestoreNote={vi.fn()}
        onExportAllMd={vi.fn()}
        onExportAllCsv={vi.fn()}
        onExportAllJson={vi.fn()}
      />
    )

    fireEvent.click(await screen.findByRole("button", { name: "Back to notes" }))

    expect(onResetEditor).toHaveBeenCalledTimes(1)
    expect(onCreateNote).not.toHaveBeenCalled()
  })
})
