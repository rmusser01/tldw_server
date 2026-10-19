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

const requestScope = {
  config: { serverUrl: "https://notes.example.test", authMode: "multi-user" as const, accessToken: "fixture-token" },
  userId: 1
}
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {
  initialize: vi.fn(async () => undefined),
  ensureConfigForRequest: vi.fn(async () => ({ serverUrl: "https://notes.example.test", authMode: "multi-user", accessToken: "fixture-token" }))
} }))
vi.mock("@/services/tldw/TldwAuth", () => ({ tldwAuth: {
  getCurrentUser: vi.fn(async () => ({ id: 1, is_active: true }))
} }))
vi.mock("@/services/tldw/deployment-mode", () => ({ isHostedTldwDeployment: () => false }))

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
      const pathname = path.split("?")[0]
      const method = String(request.method || "GET").toUpperCase()
      if (path.startsWith("/api/v1/notes/?page=1")) {
        return { items: [], pagination: { total_items: 0, total_pages: 1 } }
      }
      if (pathname === "/api/v1/notes/101" && method === "PUT") {
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
          requestScope,
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
      expect(mockBgRequest).toHaveBeenCalledWith(expect.objectContaining({
        path: "/api/v1/notes/101?expected_version=2",
        method: "PUT",
        headers: expect.objectContaining({ "X-TLDW-Expected-User-ID": "1" })
      }))
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
})
