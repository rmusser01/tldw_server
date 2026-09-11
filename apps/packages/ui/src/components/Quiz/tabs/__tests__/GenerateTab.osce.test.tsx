import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { GenerateTab } from "../GenerateTab"
import { CreateTab } from "../CreateTab"
import {
  useCreateOsceStationMutation,
  useCreateQuizMutation,
  useGenerateQuizMutation
} from "../../hooks"
import { listQuizGenerationProfiles } from "@/services/quizzes"
import { createOsceStation } from "@/services/osce"
import { tldwClient } from "@/services/tldw"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, value?: string | ({ defaultValue?: string } & Record<string, unknown>)) => {
      const template = typeof value === "string" ? value : value?.defaultValue ?? _key
      return Object.entries(typeof value === "object" && value ? value : {}).reduce(
        (text, [key, replacement]) => text.replaceAll(`{{${key}}}`, String(replacement)),
        template
      )
    }
  })
}))
vi.mock("../../hooks", () => ({
  useGenerateQuizMutation: vi.fn(),
  useCreateQuizMutation: vi.fn(),
  useCreateOsceStationMutation: vi.fn(),
  useCreateQuestionMutation: vi.fn(() => ({ mutateAsync: vi.fn(), isPending: false }))
}))
vi.mock("@/services/quizzes", async () => ({
  ...(await vi.importActual<typeof import("@/services/quizzes")>("@/services/quizzes")),
  listQuizGenerationProfiles: vi.fn()
}))
vi.mock("@/services/osce", async () => ({
  ...(await vi.importActual<typeof import("@/services/osce")>("@/services/osce")),
  createOsceStation: vi.fn()
}))
vi.mock("@/services/tldw", () => ({
  tldwClient: {
    listMedia: vi.fn(), searchMedia: vi.fn(), getMediaDetails: vi.fn(),
    listNotes: vi.fn(), searchNotes: vi.fn()
  }
}))
vi.mock("@/services/flashcards", () => ({
  generateFlashcards: vi.fn(), createDeck: vi.fn(), createFlashcard: vi.fn(),
  listDecks: vi.fn(async () => []), listFlashcards: vi.fn(async () => ({ items: [], count: 0 }))
}))
vi.mock("react-router-dom", () => ({ useNavigate: () => vi.fn() }))

const renderGenerate = (onNavigateToManage = vi.fn()) => {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(<QueryClientProvider client={client}><GenerateTab onNavigateToTake={() => {}} onNavigateToManage={onNavigateToManage} /></QueryClientProvider>)
  return onNavigateToManage
}

const fillOsceCreateForm = () => {
  fireEvent.click(screen.getByRole("radio", { name: "OSCE" }))
  fireEvent.change(screen.getByLabelText("Quiz Name"), { target: { value: "Safe prescribing OSCE" } })
  fireEvent.change(screen.getByLabelText("Station title"), { target: { value: "Anticoagulant counselling" } })
  fireEvent.change(screen.getByLabelText("Candidate instructions"), { target: { value: "Speak with the patient." } })
  fireEvent.change(screen.getByLabelText("Candidate task"), { target: { value: "Explain safe medicine use." } })
  fireEvent.change(screen.getByLabelText("Patient context"), { target: { value: "A fictional adult started treatment." } })
  fireEvent.change(screen.getByLabelText("Checklist item 1"), { target: { value: "Explains monitoring" } })
  fireEvent.change(screen.getByLabelText("Rubric domain 1"), { target: { value: "Communication" } })
  fireEvent.change(screen.getByLabelText("Rubric level 1 description in domain 1"), { target: { value: "The explanation is incomplete." } })
  fireEvent.change(screen.getByLabelText("Rubric level 2 description in domain 1"), { target: { value: "The explanation is clear." } })
  fireEvent.change(screen.getByLabelText("Expected key point 1"), { target: { value: "Discuss monitoring and warning signs." } })
}

describe("OSCE Generate and Create controls", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(tldwClient.listMedia).mockResolvedValue({ items: [{ id: 10, title: "Clinical notes", type: "pdf" }], pagination: { total_items: 1 } } as never)
    vi.mocked(tldwClient.getMediaDetails).mockResolvedValue({} as never)
    vi.mocked(tldwClient.listNotes).mockResolvedValue({ items: [] } as never)
    vi.mocked(tldwClient.searchNotes).mockResolvedValue({ items: [] } as never)
    vi.mocked(listQuizGenerationProfiles).mockResolvedValue([
      {
        id: "standard_recall", label: "Standard Recall", description: "Questions", status: "available",
        output_kind: "questions", default_num_stations: null, default_num_questions: 10,
        default_difficulty: "mixed", default_question_types: ["multiple_choice"]
      },
      {
        id: "osce_scenario", label: "OSCE Scenario", description: "Clinical practice station", status: "available",
        output_kind: "osce_stations", default_num_stations: 1, default_num_questions: 1,
        default_difficulty: "mixed", default_question_types: []
      }
    ] as never)
    vi.mocked(useGenerateQuizMutation).mockReturnValue({
      mutateAsync: vi.fn(async () => ({
        output_kind: "osce_stations", quiz: { id: 77, name: "Generated OSCE" }, questions: [], osce_stations: [{ id: 9 }]
      })), isPending: false
    } as never)
    vi.mocked(useCreateQuizMutation).mockReturnValue({ mutateAsync: vi.fn(async () => ({ id: 80 })), isPending: false } as never)
    vi.mocked(useCreateOsceStationMutation).mockReturnValue({
      mutateAsync: vi.fn(async () => ({ id: 81, quiz_id: 80 })),
      isPending: false
    } as never)
    vi.mocked(createOsceStation).mockResolvedValue({ id: 81, quiz_id: 80 } as never)
  })

  it("shows OSCE station count 1-10 and suppresses question-only controls", async () => {
    renderGenerate()
    const profile = await screen.findByTestId("generate-profile-select")
    fireEvent.mouseDown(profile.querySelector(".ant-select-selector") ?? profile)
    fireEvent.click(await screen.findByText("OSCE Scenario", { selector: ".ant-select-item-option-content" }))

    expect(await screen.findByRole("spinbutton", { name: "Stations" })).toHaveValue("1")
    expect(screen.getByRole("spinbutton", { name: "Stations" })).toHaveAttribute("aria-valuemin", "1")
    expect(screen.getByRole("spinbutton", { name: "Stations" })).toHaveAttribute("aria-valuemax", "10")
    expect(screen.queryByTestId("generate-question-plan")).not.toBeInTheDocument()
    expect(screen.queryByText("Passing Score (%)")).not.toBeInTheDocument()
    expect(screen.queryByTestId("generate-study-materials-toggle")).not.toBeInTheDocument()
  })

  it("shows the available fallback OSCE profile when the catalog is unavailable", async () => {
    vi.mocked(listQuizGenerationProfiles).mockRejectedValueOnce(new Error("catalog unavailable"))

    renderGenerate()
    const profile = await screen.findByTestId("generate-profile-select")
    fireEvent.mouseDown(profile.querySelector(".ant-select-selector") ?? profile)

    expect(await screen.findByText("Standard Recall", { selector: ".ant-select-item-option-content" })).toBeInTheDocument()
    expect(await screen.findByText("OSCE Scenario", { selector: ".ant-select-item-option-content" })).toBeInTheDocument()
  })

  it("sends num_stations and navigates successful OSCE generation to Manage", async () => {
    const navigateManage = renderGenerate()
    const profile = await screen.findByTestId("generate-profile-select")
    fireEvent.mouseDown(profile.querySelector(".ant-select-selector") ?? profile)
    fireEvent.click(await screen.findByText("OSCE Scenario", { selector: ".ant-select-item-option-content" }))
    await screen.findByRole("spinbutton", { name: "Stations" })
    await waitFor(() => expect(screen.getByText("1 media items available")).toBeInTheDocument())
    fireEvent.mouseDown(screen.getAllByRole("combobox")[0])
    fireEvent.click(await screen.findByText("Clinical notes (pdf)"))
    fireEvent.click(screen.getByRole("button", { name: "Generate OSCE" }))

    await waitFor(() => expect(useGenerateQuizMutation().mutateAsync).toHaveBeenCalledWith(expect.objectContaining({
      request: expect.objectContaining({ generation_profile: "osce_scenario", num_stations: 1 })
    })))
    await waitFor(() => expect(navigateManage).toHaveBeenCalledTimes(1))
  })

  it("uses a Quiz or OSCE segmented activity control and hides incompatible Create fields", () => {
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(<QueryClientProvider client={client}><CreateTab onNavigateToTake={() => {}} /></QueryClientProvider>)

    fireEvent.click(screen.getByRole("radio", { name: "OSCE" }))
    expect(screen.queryByText("Time Limit (minutes)")).not.toBeInTheDocument()
    expect(screen.queryByText("Passing Score (%)")).not.toBeInTheDocument()
    expect(screen.queryByText(/Questions \(/)).not.toBeInTheDocument()
    expect(screen.getByText("Station authoring")).toBeInTheDocument()
  })

  it("creates the OSCE shell and station explicitly before navigating to Manage", async () => {
    const navigateManage = vi.fn()
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(
      <QueryClientProvider client={client}>
        <CreateTab onNavigateToTake={() => {}} onNavigateToManage={navigateManage} />
      </QueryClientProvider>
    )

    fillOsceCreateForm()
    fireEvent.click(screen.getByRole("button", { name: "Save station" }))

    await waitFor(() => expect(useCreateQuizMutation().mutateAsync).toHaveBeenCalledWith({
      name: "Safe prescribing OSCE",
      description: undefined,
      activity_type: "osce"
    }))
    await waitFor(() => expect(useCreateOsceStationMutation().mutateAsync).toHaveBeenCalledWith(
      expect.objectContaining({
        quizId: 80,
        request: expect.objectContaining({
        order_index: 0,
        content: expect.objectContaining({ title: "Anticoagulant counselling" })
        })
      })
    ))
    expect(navigateManage).toHaveBeenCalledTimes(1)
    await waitFor(() => expect(screen.getByLabelText("Station title")).toHaveValue(""))
  })

  it("reuses the created OSCE quiz shell after a definitive station rejection", async () => {
    const rejected = Object.assign(new Error("station rejected"), { status: 422 })
    const stationCreate = vi.fn()
      .mockRejectedValueOnce(rejected)
      .mockResolvedValueOnce({ id: 81, quiz_id: 80 })
    vi.mocked(useCreateOsceStationMutation).mockReturnValue({
      mutateAsync: stationCreate,
      isPending: false
    } as never)
    const navigateManage = vi.fn()
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(<QueryClientProvider client={client}>
      <CreateTab onNavigateToTake={() => {}} onNavigateToManage={navigateManage} />
    </QueryClientProvider>)
    fillOsceCreateForm()

    const saveButton = screen.getByRole("button", { name: "Save station" })
    fireEvent.click(saveButton)
    await waitFor(() => expect(stationCreate).toHaveBeenCalledTimes(1))
    expect(screen.getByLabelText("Station title")).toHaveValue("Anticoagulant counselling")
    expect(screen.getByRole("radio", { name: "OSCE" })).toBeDisabled()
    expect(screen.getByLabelText("Quiz Name")).toBeDisabled()
    await waitFor(() => expect(saveButton).toBeEnabled())
    fireEvent.click(saveButton)

    await waitFor(() => expect(stationCreate).toHaveBeenCalledTimes(2))
    expect(useCreateQuizMutation().mutateAsync).toHaveBeenCalledTimes(1)
    expect(navigateManage).toHaveBeenCalledTimes(1)
  })

  it.each([
    ["network failure", new TypeError("Failed to fetch")],
    ["production transport failure", Object.assign(new Error("network unavailable"), { status: 0 })],
    ["request timeout", Object.assign(new Error("timed out"), { status: 408 })],
    ["server failure", Object.assign(new Error("unavailable"), { status: 503 })]
  ])("fails closed after ambiguous station creation: %s", async (_label, failure) => {
    const stationCreate = vi.fn().mockRejectedValue(failure)
    vi.mocked(useCreateOsceStationMutation).mockReturnValue({
      mutateAsync: stationCreate,
      isPending: false
    } as never)
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(<QueryClientProvider client={client}>
      <CreateTab onNavigateToTake={() => {}} onNavigateToManage={() => {}} />
    </QueryClientProvider>)
    fillOsceCreateForm()

    fireEvent.click(screen.getByRole("button", { name: "Save station" }))

    expect(await screen.findByText(/Station creation status is unknown/i)).toBeInTheDocument()
    expect(screen.getByText(/inspect Manage/i)).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save station" })).toBeDisabled()
    expect(stationCreate).toHaveBeenCalledTimes(1)
    expect(useCreateQuizMutation().mutateAsync).toHaveBeenCalledTimes(1)
  })

  it("does not recreate an OSCE shell after the production status-zero transport error", async () => {
    vi.mocked(useCreateQuizMutation).mockReturnValue({
      mutateAsync: vi.fn(async () => {
        throw Object.assign(new Error("network unavailable"), { status: 0 })
      }),
      isPending: false
    } as never)
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    render(<QueryClientProvider client={client}>
      <CreateTab onNavigateToTake={() => {}} />
    </QueryClientProvider>)
    fillOsceCreateForm()

    fireEvent.click(screen.getByRole("button", { name: "Save station" }))
    expect(await screen.findByText(/Quiz creation status is unknown/i)).toBeInTheDocument()
    expect(screen.queryByText("Station creation status is unknown.")).not.toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Save station" }))

    expect(useCreateQuizMutation().mutateAsync).toHaveBeenCalledTimes(1)
    expect(useCreateOsceStationMutation().mutateAsync).not.toHaveBeenCalled()
  })
})
