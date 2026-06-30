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

const buildNotes = (): NoteListItem[] => [
  {
    id: "n1",
    title: "Open Source Model Benchmark",
    content: "The model performs well in open source datasets.",
    updated_at: new Date().toISOString(),
    deleted: false,
    keywords: ["research"]
  }
]

const renderPanel = (searchQuery?: string) =>
  render(
    <NotesListPanel
      listMode="active"
      searchQuery={searchQuery}
      isOnline
      isFetching={false}
      demoEnabled={false}
      capsLoading={false}
      capabilities={{ hasNotes: true } as any}
      notes={buildNotes()}
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

describe("NotesListPanel stage 2 search highlighting", () => {
  it("highlights phrase and token matches in title and preview", () => {
    renderPanel('"Open Source" model')

    expect(screen.getAllByText(/Open Source/i, { selector: "mark" }).length).toBeGreaterThan(0)
    expect(screen.getAllByText(/Model/i, { selector: "mark" }).length).toBeGreaterThan(0)
  })

  it("renders plain text without marks when no query is present", () => {
    const { container } = renderPanel("")
    expect(container.querySelectorAll("mark").length).toBe(0)
  })
})
