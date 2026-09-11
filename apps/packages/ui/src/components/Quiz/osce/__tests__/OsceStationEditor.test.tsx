import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { OsceStationEditor } from "../OsceStationEditor"
import { getOsceStation, updateOsceStation } from "@/services/osce"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, value?: string | { defaultValue?: string }) =>
      typeof value === "string" ? value : value?.defaultValue ?? _key
  })
}))

vi.mock("@/services/osce", async () => {
  const actual = await vi.importActual<typeof import("@/services/osce")>("@/services/osce")
  return { ...actual, getOsceStation: vi.fn(), updateOsceStation: vi.fn() }
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

  it("preserves a local draft on 409 and requires confirmation before retrying against the latest version", async () => {
    renderEditor()
    vi.mocked(updateOsceStation)
      .mockRejectedValueOnce(Object.assign(new Error("conflict"), { status: 409 }))
      .mockResolvedValueOnce({ ...station, version: 7, content: { ...station.content, title: "Local draft" } })
    vi.mocked(getOsceStation).mockResolvedValue({ ...station, version: 6, content: { ...station.content, title: "Server title" } })

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
      expect.objectContaining({ expected_version: 6, content: expect.objectContaining({ title: "Local draft" }) })
    ))
  })
})
