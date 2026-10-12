import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { NotesDockPanel } from "../NotesDockPanel"
import { useNotesDockStore } from "@/store/notes-dock"
import { getQueryClient } from "@/services/query-client"

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

describe("NotesDockPanel stage 4 cross-surface cache sync", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.stubGlobal("ResizeObserver", ResizeObserverMock as any)

    mockBgRequest.mockImplementation(async (request: { path?: string; method?: string }) => {
      const path = String(request.path || "")
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?page=1")) {
        return { items: [], pagination: { total_items: 0, total_pages: 1 } }
      }
      if (path === "/api/v1/notes/101" && method === "PUT") {
        return {
          id: 101,
          title: "Dock note",
          content: "Updated dock content",
          keywords: ["research"],
          version: 3
        }
      }
      return {}
    })

    useNotesDockStore.setState({
      isOpen: true,
      position: { x: 24, y: 80 },
      size: { width: 640, height: 520 },
      notes: [
        {
          localId: "local-1",
          id: 101,
          title: "Dock note",
          content: "Updated dock content",
          keywords: ["research"],
          version: 2,
          snapshot: {
            title: "Dock note",
            content: "Old content",
            keywords: ["research"],
            version: 2
          },
          isDirty: true
        }
      ],
      activeNoteId: "local-1"
    })
  })

  afterEach(() => {
    useNotesDockStore.setState({
      isOpen: false,
      notes: [],
      activeNoteId: null
    })
    vi.unstubAllGlobals()
  })

  it("invalidates notes queries after successful dock save", async () => {
    const invalidateSpy = vi
      .spyOn(getQueryClient(), "invalidateQueries")
      .mockResolvedValue(undefined as any)

    render(<NotesDockPanel />)
    fireEvent.click(screen.getByRole("button", { name: "Update" }))

    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ["notes"] })
    })
    invalidateSpy.mockRestore()
  })

  it("shows and clears a sync indicator while cache invalidation is in flight", async () => {
    let resolveInvalidate: (() => void) | null = null
    const invalidateSpy = vi
      .spyOn(getQueryClient(), "invalidateQueries")
      .mockImplementation(
        () =>
          new Promise<void>((resolve) => {
            resolveInvalidate = resolve
          }) as any
      )

    render(<NotesDockPanel />)
    fireEvent.click(screen.getByRole("button", { name: "Update" }))

    expect(await screen.findByTestId("notes-dock-sync-indicator")).toHaveTextContent(
      "Syncing notes list..."
    )

    resolveInvalidate?.()
    await waitFor(() => {
      expect(screen.queryByTestId("notes-dock-sync-indicator")).not.toBeInTheDocument()
    })
    invalidateSpy.mockRestore()
  })
  it('opens mixed source through the real archive owner and keeps title-only save exact', async () => {
    const source = ' <>&\r\n雪\n '
    const writes: Record<string,unknown>[] = []
    mockBgRequest.mockImplementation(async request => {
      if (request.path.startsWith('/api/v1/notes/?page=1')) return {items:[{id:202,title:'Mixed source',content:source}],pagination:{total_items:1}}
      if (request.method === 'PUT') {writes.push(request.body);return {...request.body,id:202,version:2}}
      if (request.path === '/api/v1/notes/202') return {id:202,title:'Mixed source',content:source,content_language:null,version:1}
      return {}
    })
    render(<NotesDockPanel />)
    fireEvent.click(await screen.findByRole('button',{name:/Mixed source/}))
    const body = await screen.findByRole('textbox',{name:'Note content'})
    expect(body.textContent).toBe(source);expect(body).toHaveAttribute('aria-readonly','true')
    fireEvent.change(screen.getByPlaceholderText('Note title...'),{target:{value:'Title only'}})
    fireEvent.click(screen.getByRole('button',{name:'Update'}))
    await waitFor(()=>expect(writes).toHaveLength(1))
    expect(writes[0].content).toBe(source);expect(writes[0]).not.toHaveProperty('content_language')
  })
  it.each(['json','yaml','text','future'])('opens %s from the archive in the canonical source editor', async language => {
    mockBgRequest.mockImplementation(async request => request.path.startsWith('/api/v1/notes/?page=1')
      ? {items:[{id:202,title:'Structured source',content:'{}'}],pagination:{total_items:1}}
      : request.path === '/api/v1/notes/202' ? {id:202,title:'Structured source',content:'{}',content_language:language,version:1} : {})
    render(<NotesDockPanel />)
    fireEvent.click(await screen.findByRole('button',{name:/Structured source/}))
    await waitFor(()=>expect(mockNavigate).toHaveBeenCalledWith('/notes?source_ref_id=202'))
    expect(useNotesDockStore.getState().notes.some(note=>note.id===202)).toBe(false)
  })

})
