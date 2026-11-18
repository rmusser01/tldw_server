import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { StudyPackCreateDrawer } from "../StudyPackCreateDrawer"
import { tldwClient } from "@/services/tldw"
import type { StudyPackIntent } from "@/services/tldw/study-pack-handoff"

const navigateMock = vi.hoisted(() => vi.fn())
const createStudyPackJobMock = vi.hoisted(() => vi.fn())
const useStudyPackJobQueryMock = vi.hoisted(() => vi.fn())
const messageSpies = vi.hoisted(() => ({
  success: vi.fn(),
  error: vi.fn(),
  info: vi.fn(),
  warning: vi.fn(),
  loading: vi.fn(),
  open: vi.fn(),
  destroy: vi.fn()
}))

vi.mock("react-router-dom", () => ({
  useNavigate: () => navigateMock
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) {
        return defaultValueOrOptions.defaultValue.replace(
          /\{\{(\w+)\}\}/g,
          (_match, token: string) =>
            String((defaultValueOrOptions as Record<string, unknown>)[token] ?? `{{${token}}}`)
        )
      }
      return key
    }
  })
}))

vi.mock("@/hooks/useAntdMessage", () => ({
  useAntdMessage: () => messageSpies
}))

vi.mock("../../hooks", async () => {
  const actual = await vi.importActual<typeof import("../../hooks")>("../../hooks")
  return {
    ...actual,
    useStudyPackCreateMutation: () => ({
      mutateAsync: createStudyPackJobMock,
      isPending: false
    }),
    useStudyPackJobQuery: useStudyPackJobQueryMock
  }
})

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

if (typeof window !== "undefined" && typeof window.matchMedia !== "function") {
  Object.defineProperty(window, "matchMedia", {
    writable: true,
    value: vi.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: vi.fn(),
      removeListener: vi.fn(),
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
      dispatchEvent: vi.fn()
    }))
  })
}

const buildCompletedResponse = () => ({
  job: {
    id: 91,
    status: "completed",
    domain: "study_packs",
    queue: "default",
    job_type: "study_pack_generate"
  },
  study_pack: {
    id: 31,
    workspace_id: null,
    title: "Networks",
    deck_id: 8,
    source_bundle_json: {},
    generation_options_json: { deck_mode: "new" },
    status: "active",
    superseded_by_pack_id: null,
    created_at: "2026-04-02T18:00:00Z",
    last_modified: "2026-04-02T18:00:00Z",
    deleted: false,
    client_id: "study-pack-tests",
    version: 1
  },
  error: null
})

const renderDrawer = (drawer: React.ReactElement) => {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } }
  })
  return render(<QueryClientProvider client={client}>{drawer}</QueryClientProvider>)
}

describe("StudyPackCreateDrawer", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useStudyPackJobQueryMock.mockReturnValue({
      data: null,
      isFetching: false
    })
  })

  it("browses and searches the media library, then submits the chosen media ID", async () => {
    vi.spyOn(tldwClient, "listMedia").mockResolvedValue({
      items: [{ id: 7, title: "Recent lecture", type: "video" }]
    })
    vi.spyOn(tldwClient, "searchMedia").mockResolvedValue({
      results: [{ id: 42, title: "Routing protocols", type: "pdf" }]
    })
    createStudyPackJobMock.mockResolvedValue({ job: { id: 91 } })

    renderDrawer(<StudyPackCreateDrawer open onClose={vi.fn()} />)

    const picker = screen.getByRole("combobox", { name: "Search media library" })
    fireEvent.mouseDown(picker)
    expect(await screen.findByText("Recent lecture")).toBeInTheDocument()

    fireEvent.change(picker, { target: { value: "Routing" } })
    fireEvent.click(await screen.findByText("Routing protocols"))
    expect(screen.getByTestId("study-pack-source-0")).toHaveTextContent("Routing protocols")
    expect(tldwClient.searchMedia).toHaveBeenCalledWith(
      { query: "Routing" },
      { page: 1, results_per_page: 50 },
      expect.objectContaining({ signal: expect.any(AbortSignal) })
    )

    fireEvent.change(screen.getByLabelText("Title"), { target: { value: "Networks" } })
    fireEvent.click(screen.getByRole("button", { name: "Create study pack" }))
    await waitFor(() => {
      expect(createStudyPackJobMock).toHaveBeenCalledWith({
        title: "Networks",
        source_items: [{ source_type: "media", source_id: "42", source_title: "Routing protocols" }],
        deck_mode: "new"
      })
    })
  })

  it("shows a media load failure and can retry into an empty library", async () => {
    vi.spyOn(tldwClient, "listMedia")
      .mockRejectedValueOnce(new Error("request failed"))
      .mockResolvedValueOnce({ items: [] })

    renderDrawer(<StudyPackCreateDrawer open onClose={vi.fn()} />)
    const picker = screen.getByRole("combobox", { name: "Search media library" })
    fireEvent.mouseDown(picker)

    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load media.")
    fireEvent.click(screen.getByRole("button", { name: "Retry" }))
    await waitFor(() => expect(screen.queryByRole("alert")).not.toBeInTheDocument())
    if (picker.getAttribute("aria-expanded") === "false") fireEvent.mouseDown(picker)
    expect(await screen.findByText("Your media library is empty.")).toBeInTheDocument()
  })

  it("requires a title and at least one source before enabling submit", () => {
    renderDrawer(<StudyPackCreateDrawer open onClose={vi.fn()} />)

    expect(screen.getByRole("dialog", { name: "Create study pack" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Create study pack" })).toBeDisabled()

    fireEvent.change(screen.getByLabelText("Title"), {
      target: { value: "Networks" }
    })

    expect(screen.getByRole("button", { name: "Create study pack" })).toBeDisabled()

    fireEvent.change(screen.getByLabelText(/Source ID/), {
      target: { value: "42" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Add source" }))

    // An untitled manual source is named by its type and ID.
    expect(screen.getByTestId("study-pack-source-0")).toHaveTextContent("Media 42")
    expect(screen.getByRole("button", { name: "Create study pack" })).toBeEnabled()
  })

  it("prefills from an intent and navigates to review after a completed job", async () => {
    const intent: StudyPackIntent = {
      title: "Networks",
      sourceItems: [
        {
          sourceType: "media",
          sourceId: "42",
          sourceTitle: "Lecture 5"
        }
      ]
    }

    createStudyPackJobMock.mockResolvedValue({
      job: {
        id: 91,
        status: "queued",
        domain: "study_packs",
        queue: "default",
        job_type: "study_pack_generate"
      }
    })
    useStudyPackJobQueryMock.mockImplementation((jobId: number | null | undefined) =>
      jobId === 91
        ? {
            data: buildCompletedResponse(),
            isFetching: false
          }
        : {
            data: null,
            isFetching: false
          }
    )

    const onClose = vi.fn()

    renderDrawer(<StudyPackCreateDrawer open onClose={onClose} initialIntent={intent} />)

    expect(screen.getByDisplayValue("Networks")).toBeInTheDocument()
    expect(screen.getByText("Lecture 5")).toBeInTheDocument()
    expect(screen.getByTestId("study-pack-source-0")).toHaveTextContent("Media")

    fireEvent.click(screen.getByRole("button", { name: "Create study pack" }))

    await waitFor(() => {
      expect(createStudyPackJobMock).toHaveBeenCalledWith({
        title: "Networks",
        source_items: [
          {
            source_type: "media",
            source_id: "42",
            source_title: "Lecture 5"
          }
        ],
        deck_mode: "new"
      })
    })

    await waitFor(() => {
      expect(onClose).toHaveBeenCalledTimes(1)
      expect(navigateMock).toHaveBeenCalledWith("/flashcards?tab=review&deck_id=8", {
        replace: true
      })
    })
  })

  it("shows a friendly error when job creation fails", async () => {
    createStudyPackJobMock.mockRejectedValue(new Error("request failed"))

    renderDrawer(<StudyPackCreateDrawer open onClose={vi.fn()} />)

    fireEvent.change(screen.getByLabelText("Title"), {
      target: { value: "Networks" }
    })
    fireEvent.change(screen.getByLabelText(/Source ID/), {
      target: { value: "42" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Add source" }))
    fireEvent.click(screen.getByRole("button", { name: "Create study pack" }))

    await waitFor(() => {
      expect(messageSpies.error).toHaveBeenCalledWith("Study pack generation failed.")
    })
    expect(navigateMock).not.toHaveBeenCalled()
  })

  it("names a source from Notes by its title and type and tucks the ID entry under Advanced (NE-08)", () => {
    const noteId = "cc02b050-2725-4d05-9793-f7070ac43150"
    renderDrawer(
      <StudyPackCreateDrawer
        open
        onClose={vi.fn()}
        initialIntent={{
          title: "My first note",
          sourceItems: [{ sourceType: "note", sourceId: noteId, sourceTitle: "My first note" }]
        }}
      />
    )

    const row = screen.getByTestId("study-pack-source-0")
    expect(row).toHaveTextContent("My first note")
    expect(row).toHaveTextContent("Note")
    // The raw ID is noise when the source already has a title.
    expect(row).not.toHaveTextContent(noteId)

    const advanced = screen.getByRole("button", { name: "Add a source by ID" })
    expect(advanced).toHaveAttribute("aria-expanded", "false")
    // aria-controls must name an element that exists while collapsed.
    const controlledId = advanced.getAttribute("aria-controls")
    expect(controlledId).toBeTruthy()
    const controlled = document.getElementById(controlledId!)
    expect(controlled).not.toBeNull()
    expect(controlled).not.toBeVisible()
    expect(screen.queryByLabelText("Source ID (advanced/manual)")).not.toBeInTheDocument()
    expect(screen.queryByTestId("study-pack-source-id-guidance")).not.toBeInTheDocument()

    fireEvent.click(advanced)
    expect(advanced).toHaveAttribute("aria-expanded", "true")
    expect(screen.getByLabelText("Source ID (advanced/manual)")).toBeInTheDocument()
    expect(screen.getByTestId("study-pack-source-id-guidance")).toBeInTheDocument()
  })

  it("labels source IDs as an advanced manual reference with supported source guidance", () => {
    renderDrawer(<StudyPackCreateDrawer open onClose={vi.fn()} />)

    expect(screen.getByLabelText("Source ID (advanced/manual)")).toBeInTheDocument()
    expect(screen.getByTestId("study-pack-source-id-guidance")).toHaveTextContent(
      "Media, Note, or Message"
    )
    expect(screen.getByText(/supported source/i)).toBeInTheDocument()
  })
})
