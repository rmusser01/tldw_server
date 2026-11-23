import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { StudyPackCreateDrawer } from "../StudyPackCreateDrawer"
import { bgRequest } from "@/services/background-proxy"
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

vi.mock("@/services/background-proxy", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/services/background-proxy")>(),
  bgRequest: vi.fn()
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
    vi.spyOn(tldwClient, "searchMedia")
      .mockResolvedValueOnce({ results: [] })
      .mockResolvedValueOnce({ results: [{ id: 42, title: "Routing protocols", type: "pdf" }] })
    createStudyPackJobMock.mockResolvedValue({ job: { id: 91 } })

    renderDrawer(<StudyPackCreateDrawer open onClose={vi.fn()} />)

    const picker = screen.getByRole("combobox", { name: "Search media library" })
    fireEvent.mouseDown(picker)
    expect(await screen.findByText("Recent lecture")).toBeInTheDocument()

    fireEvent.change(picker, { target: { value: "NoSuchTitle" } })
    expect(await screen.findByText("No matching media.")).toBeInTheDocument()
    expect(screen.queryByText("Recent lecture")).not.toBeInTheDocument()

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

  it("submits only the checked sections and blocks an empty selection", async () => {
    vi.mocked(bgRequest).mockResolvedValue({
      revision: "current-media",
      sections: [
        { index: 0, title: "Chapter one", character_count: 5000 },
        { index: 1, title: "Chapter two", character_count: 5600 }
      ]
    })
    createStudyPackJobMock.mockResolvedValue({ job: { id: 91 } })
    const user = userEvent.setup()
    renderDrawer(
      <StudyPackCreateDrawer
        open
        onClose={vi.fn()}
        initialIntent={{
          title: "Networks",
          sourceItems: [{ sourceType: "media", sourceId: "42", sourceTitle: "Networking textbook" }]
        }}
      />
    )

    const create = screen.getByRole("button", { name: "Create study pack" })
    // The plan loads as soon as the source is chosen; expanding only shows it.
    await waitFor(() => expect(bgRequest).toHaveBeenCalledTimes(1))
    await user.click(screen.getByRole("button", { name: "Review sections" }))
    const first = await screen.findByRole("checkbox", { name: "Chapter one" })
    const second = screen.getByRole("checkbox", { name: "Chapter two" })
    expect(first).toBeChecked()
    expect(second).toBeChecked()
    expect(bgRequest).toHaveBeenCalledWith({
      method: "GET",
      path: "/api/v1/flashcards/study-packs/media/42/sections",
      abortSignal: expect.any(AbortSignal)
    })

    await user.click(first)
    await user.click(second)
    expect(create).toBeDisabled()
    await user.click(second)
    await user.click(screen.getByRole("button", { name: "Review sections" }))
    await user.click(create)
    expect(createStudyPackJobMock).toHaveBeenCalledWith({
      title: "Networks",
      source_items: [{
        source_type: "media",
        source_id: "42",
        source_title: "Networking textbook",
        locator: { section_indices: [1], section_revision: "current-media" }
      }],
      deck_mode: "new"
    })
  })

  it("offers section review only for a whole media source", () => {
    renderDrawer(
      <StudyPackCreateDrawer
        open
        onClose={vi.fn()}
        initialIntent={{
          title: "Networks",
          sourceItems: [
            { sourceType: "media", sourceId: "1", locator: { timestamp_seconds: 0 } },
            { sourceType: "media", sourceId: "2", locator: { chunk_index: 0 } },
            { sourceType: "media", sourceId: "3", excerptText: "An extract" },
            { sourceType: "note", sourceId: "4" },
            { sourceType: "message", sourceId: "5" },
            { sourceType: "media", sourceId: "6" }
          ]
        }}
      />
    )

    expect(screen.getAllByRole("button", { name: "Review sections" })).toHaveLength(1)
    expect(screen.getByTestId("study-pack-source-5")).toContainElement(
      screen.getByRole("button", { name: "Review sections" })
    )
  })

  it("requires a fresh section choice when the media revision changes", async () => {
    const sections = [
      { index: 0, title: "Chapter one", character_count: 5000 },
      { index: 1, title: "Chapter two", character_count: 5600 }
    ]
    vi.mocked(bgRequest)
      .mockResolvedValueOnce({ revision: "old-media", sections })
      .mockResolvedValueOnce({
        revision: "new-media",
        sections: [sections[0], { ...sections[1], title: "Revised chapter" }]
      })
    createStudyPackJobMock.mockResolvedValue({ job: { id: 91 } })
    const user = userEvent.setup()
    renderDrawer(
      <StudyPackCreateDrawer
        open
        onClose={vi.fn()}
        initialIntent={{
          title: "Networks",
          sourceItems: [{ sourceType: "media", sourceId: "42" }]
        }}
      />
    )

    const review = screen.getByRole("button", { name: "Review sections" })
    await user.click(review)
    await user.click(await screen.findByRole("checkbox", { name: "Chapter one" }))
    await user.click(review)
    await user.click(review)

    const revised = await screen.findByRole("checkbox", { name: "Revised chapter" })
    expect(revised).not.toBeChecked()
    expect(screen.getByRole("checkbox", { name: "Chapter one" })).not.toBeChecked()
    expect(screen.getByText("This media changed. Choose its sections again.")).toBeVisible()
    const create = screen.getByRole("button", { name: "Create study pack" })
    expect(create).toBeDisabled()
    await user.click(revised)
    await user.click(create)
    expect(createStudyPackJobMock).toHaveBeenCalledWith(expect.objectContaining({
      source_items: [expect.objectContaining({
        locator: { section_indices: [1], section_revision: "new-media" }
      })]
    }))
  })

  it("requires an explicit choice to use the whole source when its sections disappear", async () => {
    vi.mocked(bgRequest)
      .mockResolvedValueOnce({
        revision: "old-media",
        sections: [
          { index: 0, title: "Chapter one", character_count: 5000 },
          { index: 1, title: "Chapter two", character_count: 5600 }
        ]
      })
      .mockResolvedValueOnce({ revision: "short-media", sections: [] })
    createStudyPackJobMock.mockResolvedValue({ job: { id: 91 } })
    const user = userEvent.setup()
    renderDrawer(
      <StudyPackCreateDrawer
        open
        onClose={vi.fn()}
        initialIntent={{
          title: "Networks",
          sourceItems: [{ sourceType: "media", sourceId: "42" }]
        }}
      />
    )

    const review = screen.getByRole("button", { name: "Review sections" })
    await user.click(review)
    await user.click(await screen.findByRole("checkbox", { name: "Chapter one" }))
    await user.click(review)
    await user.click(review)
    const useWhole = await screen.findByRole("button", { name: "Use whole source" })
    const create = screen.getByRole("button", { name: "Create study pack" })
    expect(create).toBeDisabled()
    await user.click(useWhole)
    await user.click(create)
    expect(createStudyPackJobMock).toHaveBeenCalledWith({
      title: "Networks",
      source_items: [{ source_type: "media", source_id: "42", locator: undefined }],
      deck_mode: "new"
    })
  })

  it("requires narrowing the selection when previewed sources exceed the pack pass budget", async () => {
    vi.mocked(bgRequest)
      .mockResolvedValueOnce({
        revision: "long-media",
        max_generation_passes: 3,
        sections: [
          { index: 0, title: "Chapter one", character_count: 5000, generation_passes: 2 },
          { index: 1, title: "Chapter two", character_count: 5600, generation_passes: 1 }
        ]
      })
      .mockResolvedValueOnce({ revision: "short-media", max_generation_passes: 3, sections: [] })
    createStudyPackJobMock.mockResolvedValue({ job: { id: 91 } })
    const user = userEvent.setup()
    renderDrawer(
      <StudyPackCreateDrawer
        open
        onClose={vi.fn()}
        initialIntent={{
          title: "Networks",
          sourceItems: [
            { sourceType: "media", sourceId: "42" },
            { sourceType: "media", sourceId: "7" }
          ]
        }}
      />
    )

    const longSource = within(screen.getByTestId("study-pack-source-0"))
    const shortSource = within(screen.getByTestId("study-pack-source-1"))
    await user.click(longSource.getByRole("button", { name: "Review sections" }))
    const chapterTwo = await longSource.findByRole("checkbox", { name: "Chapter two" })
    expect(chapterTwo).toBeChecked()
    expect(await screen.findByText(/needs 4 generation passes.*limit is 3.*Uncheck sections/)).toBeVisible()
    await user.click(shortSource.getByRole("button", { name: "Review sections" }))
    expect(await screen.findByText(/needs 4 generation passes.*limit is 3.*Uncheck sections/)).toBeVisible()
    const create = screen.getByRole("button", { name: "Create study pack" })
    expect(create).toBeDisabled()

    await user.click(chapterTwo)
    await user.click(create)
    expect(createStudyPackJobMock).toHaveBeenCalledWith({
      title: "Networks",
      source_items: [
        { source_type: "media", source_id: "42", locator: { section_indices: [0], section_revision: "long-media" } },
        { source_type: "media", source_id: "7", locator: undefined }
      ],
      deck_mode: "new"
    })
  })

  it("warns about the pass budget before any source is expanded, from the section sizes", async () => {
    vi.mocked(bgRequest).mockResolvedValue({
      revision: "long-media",
      max_generation_passes: 3,
      pass_chars: 1000,
      sections: [
        { index: 0, title: "Chapter one", character_count: 1500, generation_passes: 2 },
        { index: 1, title: "Chapter two", character_count: 2600, generation_passes: 3 }
      ]
    })
    renderDrawer(
      <StudyPackCreateDrawer
        open
        onClose={vi.fn()}
        initialIntent={{ title: "Networks", sourceItems: [{ sourceType: "media", sourceId: "42" }] }}
      />
    )

    expect(await screen.findByText(/needs 5 generation passes.*limit is 3/)).toBeVisible()
    expect(screen.getByRole("button", { name: "Create study pack" })).toBeDisabled()
  })

  it("aborts a section preview when the authenticated account changes", async () => {
    vi.mocked(bgRequest).mockImplementationOnce(() => new Promise(() => {}))
    const onClose = vi.fn()
    renderDrawer(
      <StudyPackCreateDrawer
        open
        onClose={onClose}
        initialIntent={{
          title: "Networks",
          sourceItems: [{ sourceType: "media", sourceId: "42" }]
        }}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Review sections" }))
    await waitFor(() => expect(bgRequest).toHaveBeenCalledTimes(1))
    const signal = vi.mocked(bgRequest).mock.calls[0][0].abortSignal
    expect(signal?.aborted).toBe(false)
    fireEvent(window, new Event("tldw:auth-principal-changed"))

    expect(onClose).toHaveBeenCalledTimes(1)
    expect(signal?.aborted).toBe(true)
    expect(screen.queryByRole("button", { name: "Review sections" })).not.toBeInTheDocument()
  })

  it("keeps a short media source whole when the preview has no sections", async () => {
    vi.mocked(bgRequest).mockResolvedValueOnce({ revision: "short-media", sections: [] })
    createStudyPackJobMock.mockResolvedValue({ job: { id: 91 } })
    renderDrawer(
      <StudyPackCreateDrawer
        open
        onClose={vi.fn()}
        initialIntent={{
          title: "Networks",
          sourceItems: [{ sourceType: "media", sourceId: "42" }]
        }}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Review sections" }))
    expect(await screen.findByText("This media will be used as a single source.")).toBeVisible()
    fireEvent.click(screen.getByRole("button", { name: "Create study pack" }))
    expect(createStudyPackJobMock).toHaveBeenCalledWith({
      title: "Networks",
      source_items: [{ source_type: "media", source_id: "42", locator: undefined }],
      deck_mode: "new"
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

  it("keeps the search term after a failed search and retries the same request", async () => {
    vi.spyOn(tldwClient, "listMedia").mockResolvedValue({
      items: [{ id: 7, title: "Recent lecture", type: "video" }]
    })
    const searchMedia = vi.spyOn(tldwClient, "searchMedia")
      .mockRejectedValueOnce(new Error("search failed"))
      .mockResolvedValueOnce({ results: [{ id: 42, title: "Routing protocols", type: "pdf" }] })
    const user = userEvent.setup()

    renderDrawer(<StudyPackCreateDrawer open onClose={vi.fn()} />)
    const picker = screen.getByRole("combobox", { name: "Search media library" })
    await user.click(picker)
    expect(await screen.findByText("Recent lecture")).toBeInTheDocument()

    await user.type(picker, "Routing")
    expect(await screen.findByRole("alert")).toHaveTextContent("Could not load media.")
    expect(picker).toHaveValue("Routing")

    await user.click(screen.getByRole("button", { name: "Retry" }))
    await waitFor(() => expect(searchMedia).toHaveBeenCalledTimes(2))
    expect(searchMedia.mock.calls.map(([request]) => request.query)).toEqual(["Routing", "Routing"])
    expect(picker).toHaveValue("Routing")
    if (picker.getAttribute("aria-expanded") === "false") await user.click(picker)
    expect(await screen.findByText("Routing protocols")).toBeInTheDocument()
  })

  it("distinguishes same-title media and discloses a truncated result set", async () => {
    vi.spyOn(tldwClient, "listMedia").mockResolvedValue({
      items: [{ id: 7, title: "Lecture" }, { id: 8, title: "Lecture" }],
      pagination: { total_items: 54 }
    })
    createStudyPackJobMock.mockResolvedValue({ job: { id: 91 } })
    renderDrawer(<StudyPackCreateDrawer open onClose={vi.fn()} />)
    fireEvent.mouseDown(screen.getByRole("combobox", { name: "Search media library" }))

    expect(await screen.findByText("Showing 2 of 54 media items. Try a more specific search.")).toBeVisible()
    expect(screen.getByText("Lecture (ID 7)")).toBeInTheDocument()
    fireEvent.click(screen.getByText("Lecture (ID 8)"))
    fireEvent.change(screen.getByLabelText("Title"), { target: { value: "Networks" } })
    fireEvent.click(screen.getByRole("button", { name: "Create study pack" }))
    await waitFor(() => expect(createStudyPackJobMock).toHaveBeenCalledWith({
      title: "Networks",
      source_items: [{ source_type: "media", source_id: "8", source_title: "Lecture" }],
      deck_mode: "new"
    }))
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
