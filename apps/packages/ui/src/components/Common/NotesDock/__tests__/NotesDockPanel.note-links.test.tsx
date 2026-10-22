import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { NotesDockPanel } from "../NotesDockPanel"
import { useNotesDockStore } from "@/store/notes-dock"

const {
  mockBgRequest,
  mockNavigate,
  stableTranslate,
  stableMessageApi
} = vi.hoisted(() => ({
  mockBgRequest: vi.fn(),
  mockNavigate: vi.fn(),
  stableTranslate: (
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
  },
  stableMessageApi: {
    error: vi.fn(),
    success: vi.fn(),
    warning: vi.fn(),
    info: vi.fn()
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: stableTranslate
  })
}))

vi.mock("react-router-dom", () => ({
  useNavigate: () => mockNavigate
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: { hasNotes: true },
    loading: false
  })
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: mockBgRequest
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => stableMessageApi
}))

class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

describe("NotesDockPanel note links (XP-05)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubGlobal("ResizeObserver", ResizeObserverMock as any)
    mockBgRequest.mockResolvedValue({ items: [], pagination: { total_items: 0, total_pages: 1 } })
  })

  afterEach(() => {
    useNotesDockStore.setState({ isOpen: false, notes: [], activeNoteId: null })
    vi.unstubAllGlobals()
  })

  const openDockWith = (note: { localId: string; id?: number; title: string }) =>
    useNotesDockStore.setState({
      isOpen: true,
      position: { x: 24, y: 80 },
      size: { width: 640, height: 520 },
      notes: [
        {
          ...note,
          content: "Body",
          keywords: [],
          version: note.id ? 1 : null,
          snapshot: null,
          isDirty: false
        }
      ],
      activeNoteId: note.localId
    })

  it("opens the dock's saved note on the Notes page", () => {
    openDockWith({ localId: "local-1", id: 101, title: "Saved dock note" })
    render(<NotesDockPanel />)

    fireEvent.click(screen.getByRole("button", { name: "Open Notes page" }))

    expect(mockNavigate).toHaveBeenCalledWith("/notes?note=101")
  })

  it("opens the Notes page itself while the dock note has never been saved", () => {
    openDockWith({ localId: "local-2", title: "Unsaved dock note" })
    render(<NotesDockPanel />)

    fireEvent.click(screen.getByRole("button", { name: "Open Notes page" }))

    expect(mockNavigate).toHaveBeenCalledWith("/notes")
  })
})
