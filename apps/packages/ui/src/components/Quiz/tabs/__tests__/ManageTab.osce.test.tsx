import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ManageTab } from "../ManageTab"
import {
  useCreateQuestionMutation, useCreateQuizMutation, useDeleteQuestionMutation,
  useDeleteQuizMutation, useQuestionsQuery, useQuizzesQuery,
  useUpdateQuestionMutation, useUpdateQuizMutation
} from "../../hooks"
import { useAllOsceStationsQuery, useOsceStationQuery } from "../../hooks/useOsceQueries"
import { importQuizzesJson } from "@/services/quizzes"
import { getOsceStation, listAllOsceStations } from "@/services/osce"

vi.mock("react-i18next", () => ({ useTranslation: () => ({
  t: (_key: string, value?: string | { defaultValue?: string }) => typeof value === "string" ? value : value?.defaultValue ?? _key
}) }))
vi.mock("../../hooks", () => ({
  useCreateQuestionMutation: vi.fn(), useCreateQuizMutation: vi.fn(), useDeleteQuestionMutation: vi.fn(),
  useDeleteQuizMutation: vi.fn(), useQuestionsQuery: vi.fn(), useQuizzesQuery: vi.fn(),
  useUpdateQuestionMutation: vi.fn(), useUpdateQuizMutation: vi.fn()
}))
vi.mock("../../hooks/useOsceQueries", () => ({
  useAllOsceStationsQuery: vi.fn(),
  useOsceStationQuery: vi.fn(),
  useCreateOsceStationMutation: vi.fn(),
  useUpdateOsceStationMutation: vi.fn()
}))
vi.mock("@/services/tldw", () => ({
  tldwClient: { getConfig: vi.fn(async () => ({ authMode: "single-user" })), getMediaDetails: vi.fn() },
  tldwAuth: { getCurrentUser: vi.fn() }
}))
vi.mock("@/services/quizzes", async () => ({
  ...(await vi.importActual<typeof import("@/services/quizzes")>("@/services/quizzes")),
  importQuizzesJson: vi.fn()
}))
vi.mock("@/services/osce", async () => ({
  ...(await vi.importActual<typeof import("@/services/osce")>("@/services/osce")),
  getOsceStation: vi.fn(),
  listAllOsceStations: vi.fn()
}))

describe("ManageTab OSCE authoring", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(useQuizzesQuery).mockReturnValue({
      data: { items: [{
        id: 8, name: "Clinical communication", description: "Practice", activity_type: "osce",
        total_questions: 0, total_stations: 1, passing_score: null, time_limit_seconds: null,
        media_id: null, deleted: false, client_id: "test", version: 2
      }], count: 1 }, isLoading: false, refetch: vi.fn()
    } as never)
    vi.mocked(useQuestionsQuery).mockReturnValue({ data: { items: [], count: 0 }, isLoading: false } as never)
    vi.mocked(useOsceStationQuery).mockReturnValue({ data: undefined, isLoading: false } as never)
    vi.mocked(useAllOsceStationsQuery).mockReturnValue({
      data: [{
        id: 9, quiz_id: 8, title: "Explain anticoagulant safety", recommended_duration_seconds: 480,
        order_index: 0, version: 2, checklist_count: 4, rubric_domain_count: 2,
        verification_state: "source_verified", created_at: "2026-09-11", updated_at: "2026-09-11"
      }], isLoading: false
    } as never)
    vi.mocked(importQuizzesJson).mockResolvedValue({
      imported_quizzes: 1, failed_quizzes: 0,
      imported_questions: 0, failed_questions: 0,
      imported_stations: 1, failed_stations: 0,
      items: [], errors: []
    })
    vi.mocked(listAllOsceStations).mockResolvedValue([])
    vi.mocked(getOsceStation).mockImplementation(async (quizId, stationId) => ({
      id: stationId,
      quiz_id: quizId,
      content: { schema_version: "osce.station.v1", title: `Station ${stationId}` },
      order_index: stationId,
      version: 1
    }) as never)
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => "blob:osce-export")
    })
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: vi.fn()
    })
    const idle = { mutateAsync: vi.fn(), isPending: false } as never
    vi.mocked(useCreateQuizMutation).mockReturnValue(idle)
    vi.mocked(useDeleteQuizMutation).mockReturnValue(idle)
    vi.mocked(useUpdateQuizMutation).mockReturnValue(idle)
    vi.mocked(useCreateQuestionMutation).mockReturnValue(idle)
    vi.mocked(useUpdateQuestionMutation).mockReturnValue(idle)
    vi.mocked(useDeleteQuestionMutation).mockReturnValue(idle)
  })

  it("shows compact station summaries and a non-color-only verification label", async () => {
    render(<ManageTab onNavigateToCreate={() => {}} onNavigateToGenerate={() => {}} onStartQuiz={() => {}} />)

    expect(screen.getByText("1 station")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Manage stations" }))
    expect(await screen.findByText("Explain anticoagulant safety")).toBeInTheDocument()
    expect(screen.getByText("8 min")).toBeInTheDocument()
    expect(screen.getByText("4 checklist items")).toBeInTheDocument()
    expect(screen.getByText("2 rubric domains")).toBeInTheDocument()
    expect(screen.getByText("Source verified")).toBeInTheDocument()
  })

  it("suppresses ordinary quiz actions that are incompatible with OSCE", async () => {
    render(<ManageTab onNavigateToCreate={() => {}} onNavigateToGenerate={() => {}} onStartQuiz={() => {}} />)
    await waitFor(() => expect(screen.getByText("Clinical communication")).toBeInTheDocument())

    expect(screen.queryByRole("button", { name: "Start" })).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Share" })).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Duplicate" })).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Print" })).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Manage stations" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Export" })).toBeInTheDocument()
  })

  it("passes a v2 OSCE import envelope to the Task 7 endpoint unchanged", async () => {
    render(<ManageTab onNavigateToCreate={() => {}} onNavigateToGenerate={() => {}} onStartQuiz={() => {}} />)
    const payload = {
      export_format: "tldw.quiz.export.v2",
      exported_at: "2026-09-11T00:00:00Z",
      quizzes: [{
        activity_type: "osce",
        quiz: { name: "Imported OSCE", activity_type: "osce" },
        stations: []
      }]
    }
    const file = new File([JSON.stringify(payload)], "osce-import.json", { type: "application/json" })

    fireEvent.change(screen.getByTestId("manage-import-input"), {
      target: { files: [file] }
    })

    await waitFor(() => expect(importQuizzesJson).toHaveBeenCalledWith(payload))
  })

  it("exports every OSCE station returned by complete pagination", async () => {
    const summaries = Array.from({ length: 201 }, (_, index) => ({
      id: index + 1,
      quiz_id: 8,
      title: `Station ${index + 1}`,
      recommended_duration_seconds: 480,
      order_index: index,
      version: 1,
      checklist_count: 1,
      rubric_domain_count: 1,
      verification_state: "manually_authored" as const,
      created_at: "2026-09-11",
      updated_at: "2026-09-11"
    }))
    vi.mocked(listAllOsceStations).mockResolvedValue(summaries)
    render(<ManageTab onNavigateToCreate={() => {}} onNavigateToGenerate={() => {}} onStartQuiz={() => {}} />)

    fireEvent.click(screen.getByRole("button", { name: "Export" }))

    await waitFor(() => expect(getOsceStation).toHaveBeenCalledTimes(201))
    expect(listAllOsceStations).toHaveBeenCalledWith(8)
    expect(URL.createObjectURL).toHaveBeenCalledTimes(1)
  })
})
