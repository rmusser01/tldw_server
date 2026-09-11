import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { OsceStationEditor } from "../OsceStationEditor"
import { createOsceStation, getOsceStation, updateOsceStation } from "@/services/osce"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, value?: string | { defaultValue?: string }) =>
      typeof value === "string" ? value : value?.defaultValue ?? _key
  })
}))

vi.mock("@/services/osce", async () => {
  const actual = await vi.importActual<typeof import("@/services/osce")>("@/services/osce")
  return {
    ...actual,
    createOsceStation: vi.fn(),
    getOsceStation: vi.fn(),
    updateOsceStation: vi.fn()
  }
})

const station = {
  id: 9,
  quiz_id: 7,
  content: {
    schema_version: "osce.station.v1" as const,
    title: "Anticoagulant counselling",
    candidate_instructions: "Speak with a simulated patient.",
    candidate_task: "Explain safe medicine use.",
    patient_context: { text: "A fictional adult recently started treatment.", citations: [] },
    recommended_duration_seconds: 480,
    checklist_items: [
      { id: "11111111-1111-4111-8111-111111111111", label: "Explain purpose", rationale: "Supports adherence.", citations: [] },
      { id: "22222222-2222-4222-8222-222222222222", label: "Discuss monitoring", rationale: "Supports safety.", citations: [] }
    ],
    rubric_domains: [{
      id: "33333333-3333-4333-8333-333333333333",
      label: "Communication",
      levels: [
        { id: "44444444-4444-4444-8444-444444444444", label: "Needs development", description: "Incomplete." },
        { id: "55555555-5555-4555-8555-555555555555", label: "Effective", description: "Clear." }
      ]
    }],
    expected_key_points: [{ id: "66666666-6666-4666-8666-666666666666", text: "Discuss warning signs.", citations: [] }]
  },
  order_index: 0,
  version: 4,
  origin: "manual" as const,
  provenance: null,
  source_bundle: [],
  verification_state: "manually_authored" as const,
  verification_timestamp: null,
  verification_summary: null,
  deleted: false,
  created_at: "2026-09-11T00:00:00Z",
  updated_at: "2026-09-11T00:00:00Z"
}

const renderEditor = (onDirtyStateChange = vi.fn()) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={client}>
      <OsceStationEditor quizId={7} station={station} onDirtyStateChange={onDirtyStateChange} />
    </QueryClientProvider>
  )
  return onDirtyStateChange
}

describe("OsceStationEditor", () => {
  beforeEach(() => vi.clearAllMocks())

  it("renders the complete inline authoring surface with accessible reorder controls", () => {
    renderEditor()

    expect(screen.getByLabelText("Station title")).toHaveValue("Anticoagulant counselling")
    expect(screen.getByLabelText("Candidate instructions")).toBeInTheDocument()
    expect(screen.getByLabelText("Candidate task")).toBeInTheDocument()
    expect(screen.getByLabelText("Patient context")).toBeInTheDocument()
    expect(screen.getByRole("spinbutton", { name: "Recommended duration in seconds" })).toHaveValue("480")
    expect(screen.getByText("Checklist items")).toBeInTheDocument()
    expect(screen.getByText("Rubric domains")).toBeInTheDocument()
    expect(screen.getByText("Expected key points")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Move checklist item 2 up" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Remove rubric level 1 from domain 1" })).toBeInTheDocument()
    expect(screen.getByText(/Do not enter real patient information/i)).toBeInTheDocument()
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it("tracks dirty state from the last acknowledged payload and saves explicitly", async () => {
    const onDirty = renderEditor()
    vi.mocked(updateOsceStation).mockResolvedValue({ ...station, version: 5, content: { ...station.content, title: "Updated title" } })

    fireEvent.change(screen.getByLabelText("Station title"), { target: { value: "Updated title" } })
    expect(onDirty).toHaveBeenLastCalledWith(true)
    expect(updateOsceStation).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole("button", { name: "Save station" }))

    await waitFor(() => expect(updateOsceStation).toHaveBeenCalledWith(
      7,
      9,
      expect.objectContaining({ expected_version: 4, content: expect.objectContaining({ title: "Updated title" }) })
    ))
    await waitFor(() => expect(onDirty).toHaveBeenLastCalledWith(false))
  })

  it("allows only one custom create request during rapid double activation", async () => {
    let resolveCreate: ((value: typeof station) => void) | undefined
    const onCreate = vi.fn(() => new Promise<typeof station>((resolve) => {
      resolveCreate = resolve
    }))
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <OsceStationEditor initialContent={station.content} onCreate={onCreate} />
      </QueryClientProvider>
    )
    fireEvent.change(screen.getByLabelText("Station title"), { target: { value: "Local draft" } })
    const saveButton = screen.getByRole("button", { name: "Save station" })

    act(() => {
      saveButton.click()
      saveButton.click()
    })

    expect(onCreate).toHaveBeenCalledTimes(1)
    expect(saveButton).toBeDisabled()
    await act(async () => {
      resolveCreate?.({ ...station, content: { ...station.content, title: "Local draft" } })
    })
  })

  it.each([
    ["missing status", new TypeError("Failed to fetch")],
    ["status zero", Object.assign(new Error("offline"), { status: 0 })],
    ["timeout", Object.assign(new Error("timeout"), { status: 408 })],
    ["server error", Object.assign(new Error("unavailable"), { status: 503 })]
  ])("fails closed after ambiguous direct station creation: %s", async (_label, failure) => {
    vi.mocked(createOsceStation).mockRejectedValue(failure)
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <OsceStationEditor quizId={7} initialContent={station.content} />
      </QueryClientProvider>
    )
    fireEvent.change(screen.getByLabelText("Station title"), { target: { value: "Preserved draft" } })

    fireEvent.click(screen.getByRole("button", { name: "Save station" }))

    expect(await screen.findByText("Station creation status is unknown.")).toBeInTheDocument()
    expect(screen.getByText(/inspect Manage or reload the station list/i)).toBeInTheDocument()
    expect(screen.getByLabelText("Station title")).toHaveValue("Preserved draft")
    expect(screen.getByRole("button", { name: "Save station" })).toBeDisabled()
    fireEvent.click(screen.getByRole("button", { name: "Save station" }))
    expect(createOsceStation).toHaveBeenCalledTimes(1)
  })

  it("keeps a definitive direct station rejection retryable", async () => {
    vi.mocked(createOsceStation)
      .mockRejectedValueOnce(Object.assign(new Error("invalid station"), { status: 422 }))
      .mockResolvedValueOnce({ ...station, content: { ...station.content, title: "Retried draft" } })
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <OsceStationEditor quizId={7} initialContent={station.content} />
      </QueryClientProvider>
    )
    fireEvent.change(screen.getByLabelText("Station title"), { target: { value: "Retried draft" } })

    fireEvent.click(screen.getByRole("button", { name: "Save station" }))
    await waitFor(() => expect(createOsceStation).toHaveBeenCalledTimes(1))
    expect(screen.getByRole("button", { name: "Save station" })).toBeEnabled()
    fireEvent.click(screen.getByRole("button", { name: "Save station" }))

    await waitFor(() => expect(createOsceStation).toHaveBeenCalledTimes(2))
  })

  it("keeps citation row identity and focus while the source ID is typed", async () => {
    const user = userEvent.setup()
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <OsceStationEditor quizId={7} station={station} />
      </QueryClientProvider>
    )
    await user.click(screen.getAllByRole("button", { name: "Add citation" })[0])
    const sourceId = screen.getByLabelText("Patient context citation 1 source ID")

    await user.type(sourceId, "source-123")

    expect(sourceId).toHaveValue("source-123")
    expect(sourceId).toHaveFocus()
  })

  it("preserves a dirty draft across same-station prop refresh and reaches conflict recovery", async () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const view = render(
      <QueryClientProvider client={client}>
        <OsceStationEditor quizId={7} station={station} />
      </QueryClientProvider>
    )
    vi.mocked(updateOsceStation).mockRejectedValue(
      Object.assign(new Error("conflict"), { status: 409 })
    )
    const serverRefresh = {
      ...station,
      version: 5,
      content: { ...station.content, title: "Server refresh" }
    }
    vi.mocked(getOsceStation).mockResolvedValue(serverRefresh)
    fireEvent.change(screen.getByLabelText("Station title"), { target: { value: "Local draft" } })

    view.rerender(
      <QueryClientProvider client={client}>
        <OsceStationEditor quizId={7} station={serverRefresh} />
      </QueryClientProvider>
    )

    expect(screen.getByLabelText("Station title")).toHaveValue("Local draft")
    fireEvent.click(screen.getByRole("button", { name: "Save station" }))
    await waitFor(() => expect(updateOsceStation).toHaveBeenCalledWith(
      7,
      9,
      expect.objectContaining({
        expected_version: 4,
        content: expect.objectContaining({ title: "Local draft" })
      })
    ))
    expect(await screen.findByText("This station changed on the server.")).toBeInTheDocument()
  })

  it("adopts server content when clean or when station identity changes", () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const view = render(
      <QueryClientProvider client={client}>
        <OsceStationEditor quizId={7} station={station} />
      </QueryClientProvider>
    )
    const cleanRefresh = {
      ...station,
      version: 5,
      content: { ...station.content, title: "Clean server refresh" }
    }
    view.rerender(
      <QueryClientProvider client={client}>
        <OsceStationEditor quizId={7} station={cleanRefresh} />
      </QueryClientProvider>
    )
    expect(screen.getByLabelText("Station title")).toHaveValue("Clean server refresh")

    fireEvent.change(screen.getByLabelText("Station title"), { target: { value: "Dirty first station" } })
    view.rerender(
      <QueryClientProvider client={client}>
        <OsceStationEditor
          quizId={7}
          station={{
            ...cleanRefresh,
            id: 10,
            content: { ...station.content, title: "Different station" }
          }}
        />
      </QueryClientProvider>
    )
    expect(screen.getByLabelText("Station title")).toHaveValue("Different station")
  })

  it("preserves a local draft on 409 and requires confirmation before retrying against the latest version", async () => {
    renderEditor()
    vi.mocked(updateOsceStation)
      .mockRejectedValueOnce(Object.assign(new Error("conflict"), { status: 409 }))
      .mockResolvedValueOnce({ ...station, version: 7, content: { ...station.content, title: "Local draft" } })
    vi.mocked(getOsceStation).mockResolvedValue({ ...station, version: 6, order_index: 3, content: { ...station.content, title: "Server title" } })

    fireEvent.change(screen.getByLabelText("Station title"), { target: { value: "Local draft" } })
    fireEvent.click(screen.getByRole("button", { name: "Save station" }))

    expect(await screen.findByText("This station changed on the server.")).toBeInTheDocument()
    expect(screen.getByLabelText("Station title")).toHaveValue("Local draft")
    fireEvent.click(screen.getByRole("button", { name: "Keep local draft" }))
    fireEvent.click(screen.getByRole("button", { name: "Save station" }))
    const confirmButton = await screen.findByRole("button", { name: "Confirm overwrite" })
    fireEvent.click(confirmButton)

    await waitFor(() => expect(updateOsceStation).toHaveBeenLastCalledWith(
      7,
      9,
      expect.objectContaining({
        expected_version: 6,
        order_index: 3,
        content: expect.objectContaining({ title: "Local draft" })
      })
    ))
  })

  it("requires absolute HTTP(S) citation URLs and unique rubric labels", () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <OsceStationEditor
          quizId={7}
          initialContent={{
            ...station.content,
            patient_context: {
              ...station.content.patient_context,
              citations: [{ source_type: "url", source_id: "source-1", source_url: "/relative" }]
            },
            rubric_domains: [{
              ...station.content.rubric_domains[0],
              levels: [
                station.content.rubric_domains[0].levels[0],
                { ...station.content.rubric_domains[0].levels[1], label: "needs DEVELOPMENT" }
              ]
            }]
          }}
        />
      </QueryClientProvider>
    )

    fireEvent.change(screen.getByLabelText("Station title"), { target: { value: "Changed title" } })

    expect(screen.getByText("URL citations require an absolute HTTP(S) URL.")).toBeInTheDocument()
    expect(screen.getByText("Rubric level labels must be unique within each domain.")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save station" })).toBeDisabled()
  })

  it("matches backend casefold behavior for rubric level labels", () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <OsceStationEditor
          quizId={7}
          initialContent={{
            ...station.content,
            rubric_domains: [{
              ...station.content.rubric_domains[0],
              levels: [
                { ...station.content.rubric_domains[0].levels[0], label: "Straße" },
                { ...station.content.rubric_domains[0].levels[1], label: "STRASSE" }
              ]
            }]
          }}
        />
      </QueryClientProvider>
    )

    fireEvent.change(screen.getByLabelText("Station title"), { target: { value: "Changed title" } })

    expect(screen.getByText("Rubric level labels must be unique within each domain.")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save station" })).toBeDisabled()
  })
})
