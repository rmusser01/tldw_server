import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ManageTab } from "../ManageTab"
import {
  useCreateQuestionMutation, useCreateQuizMutation, useDeleteQuestionMutation,
  useDeleteQuizMutation, useQuestionsQuery, useQuizzesQuery,
  useUpdateQuestionMutation, useUpdateQuizMutation
} from "../../hooks"
import {
  useAllOsceStationsQuery,
  useDeleteOsceStationMutation,
  useOsceStationQuery
} from "../../hooks/useOsceQueries"
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
  useDeleteOsceStationMutation: vi.fn(),
  useUpdateOsceStationMutation: vi.fn()
}))
vi.mock("../../osce/OsceStationEditor", () => ({
  OsceStationEditor: ({
    createStatusUncertain,
    onCreateStatusUncertainChange,
    orderIndex,
    onDirtyStateChange,
    station
  }: {
    createStatusUncertain?: boolean
    onCreateStatusUncertainChange?: (uncertain: boolean) => void
    orderIndex?: number
    onDirtyStateChange?: (dirty: boolean) => void
    station?: { content?: { title?: string } }
  }) => {
    const [title, setTitle] = React.useState(station?.content?.title ?? "")
    return (
      <div data-testid="osce-station-editor">
        <span data-testid="osce-station-order-index">{orderIndex}</span>
        <input
          aria-label="Managed station draft"
          value={title}
          onChange={(event) => {
            setTitle(event.target.value)
            onDirtyStateChange?.(true)
          }}
        />
        <button type="button" onClick={() => onDirtyStateChange?.(true)}>Mark station dirty</button>
        {!station ? (
          <button type="button" onClick={() => onCreateStatusUncertainChange?.(true)}>
            Simulate ambiguous station create
          </button>
        ) : null}
        {createStatusUncertain ? <span>Managed create is blocked</span> : null}
      </div>
    )
  }
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
  const quizRefetch = vi.fn()
  const stationListRefetch = vi.fn()
  const stationDetailRefetch = vi.fn()
  const deleteStation = vi.fn()
  const deleteQuiz = vi.fn()
  let stationVersion = 2
  let stationTitle = "Explain anticoagulant safety"

  beforeEach(() => {
    vi.clearAllMocks()
    stationVersion = 2
    stationTitle = "Explain anticoagulant safety"
    vi.mocked(useQuizzesQuery).mockReturnValue({
      data: { items: [{
        id: 8, name: "Clinical communication", description: "Practice", activity_type: "osce",
        total_questions: 0, total_stations: 1, passing_score: null, time_limit_seconds: null,
        media_id: null, deleted: false, client_id: "test", version: 2
      }], count: 1 }, isLoading: false, refetch: quizRefetch
    } as never)
    vi.mocked(useQuestionsQuery).mockReturnValue({ data: { items: [], count: 0 }, isLoading: false } as never)
    vi.mocked(useOsceStationQuery).mockImplementation((_quizId, stationId) => ({
      data: stationId == null ? undefined : {
        id: stationId,
        quiz_id: 8,
        content: { schema_version: "osce.station.v1", title: stationTitle },
        order_index: 0,
        version: stationVersion
      },
      isLoading: false,
      refetch: stationDetailRefetch
    } as never))
    vi.mocked(useAllOsceStationsQuery).mockReturnValue({
      data: [{
        id: 9, quiz_id: 8, title: "Explain anticoagulant safety", recommended_duration_seconds: 480,
        order_index: 0, version: 2, checklist_count: 4, rubric_domain_count: 2,
        verification_state: "source_verified", created_at: "2026-09-11", updated_at: "2026-09-11"
      }], isLoading: false, refetch: stationListRefetch
    } as never)
    stationListRefetch.mockResolvedValue({ data: [] })
    stationDetailRefetch.mockResolvedValue({ data: undefined })
    quizRefetch.mockResolvedValue({ data: undefined })
    vi.mocked(useDeleteOsceStationMutation).mockReturnValue({
      mutateAsync: deleteStation,
      isPending: false
    } as never)
    deleteStation.mockResolvedValue(undefined)
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
    vi.mocked(useDeleteQuizMutation).mockReturnValue({
      mutateAsync: deleteQuiz,
      isPending: false
    } as never)
    deleteQuiz.mockResolvedValue(undefined)
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

  it("cancels confirmed station deletion without changing selection", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(false)
    render(<ManageTab onNavigateToCreate={() => {}} onNavigateToGenerate={() => {}} onStartQuiz={() => {}} />)
    fireEvent.click(screen.getByRole("button", { name: "Manage stations" }))
    fireEvent.click(screen.getByRole("button", { name: "Edit station Explain anticoagulant safety" }))
    expect(await screen.findByTestId("osce-station-editor")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Delete station Explain anticoagulant safety" }))

    expect(deleteStation).not.toHaveBeenCalled()
    expect(screen.getByTestId("osce-station-editor")).toBeInTheDocument()
  })

  it("deletes the selected station after dirty-draft and delete confirmations", async () => {
    vi.spyOn(window, "confirm").mockReturnValueOnce(true).mockReturnValueOnce(true)
    render(<ManageTab onNavigateToCreate={() => {}} onNavigateToGenerate={() => {}} onStartQuiz={() => {}} />)
    fireEvent.click(screen.getByRole("button", { name: "Manage stations" }))
    fireEvent.click(screen.getByRole("button", { name: "Edit station Explain anticoagulant safety" }))
    fireEvent.click(await screen.findByRole("button", { name: "Mark station dirty" }))

    fireEvent.click(screen.getByRole("button", { name: "Delete station Explain anticoagulant safety" }))

    await waitFor(() => expect(deleteStation).toHaveBeenCalledWith({
      quizId: 8,
      stationId: 9,
      expectedVersion: 2
    }))
    expect(window.confirm).toHaveBeenNthCalledWith(1, "Discard unsaved station changes?")
    expect(window.confirm).toHaveBeenNthCalledWith(
      2,
      'Delete station "Explain anticoagulant safety"? This cannot be undone.'
    )
    await waitFor(() => expect(screen.queryByTestId("osce-station-editor")).not.toBeInTheDocument())
    expect(stationListRefetch).toHaveBeenCalled()
    expect(quizRefetch).toHaveBeenCalled()
  })

  it("preserves the selected station when deletion fails", async () => {
    deleteStation.mockRejectedValueOnce(Object.assign(new Error("delete failed"), { status: 422 }))
    vi.spyOn(window, "confirm").mockReturnValue(true)
    render(<ManageTab onNavigateToCreate={() => {}} onNavigateToGenerate={() => {}} onStartQuiz={() => {}} />)
    fireEvent.click(screen.getByRole("button", { name: "Manage stations" }))
    fireEvent.click(screen.getByRole("button", { name: "Edit station Explain anticoagulant safety" }))

    fireEvent.click(screen.getByRole("button", { name: "Delete station Explain anticoagulant safety" }))

    expect(await screen.findByText("Failed to delete station.")).toBeInTheDocument()
    expect(screen.getByTestId("osce-station-editor")).toBeInTheDocument()
    expect(stationListRefetch).not.toHaveBeenCalled()
  })

  it("reconciles an ambiguous station delete that committed on the server", async () => {
    deleteStation.mockRejectedValueOnce(Object.assign(new Error("transport lost"), { status: 0 }))
    vi.mocked(listAllOsceStations).mockResolvedValueOnce([])
    vi.mocked(getOsceStation).mockRejectedValueOnce(Object.assign(new Error("not found"), { status: 404 }))
    vi.spyOn(window, "confirm").mockReturnValue(true)
    render(<ManageTab onNavigateToCreate={() => {}} onNavigateToGenerate={() => {}} onStartQuiz={() => {}} />)
    fireEvent.click(screen.getByRole("button", { name: "Manage stations" }))
    fireEvent.click(screen.getByRole("button", { name: "Edit station Explain anticoagulant safety" }))

    fireEvent.click(screen.getByRole("button", { name: "Delete station Explain anticoagulant safety" }))

    await waitFor(() => expect(listAllOsceStations).toHaveBeenCalledWith(8))
    expect(getOsceStation).toHaveBeenCalledWith(8, 9)
    await waitFor(() => expect(screen.queryByTestId("osce-station-editor")).not.toBeInTheDocument())
    expect(stationListRefetch).toHaveBeenCalled()
    expect(stationDetailRefetch).toHaveBeenCalled()
    expect(quizRefetch).toHaveBeenCalled()
  })

  it("keeps a noncommitted ambiguous delete actionable and treats a 404 retry as deleted", async () => {
    deleteStation
      .mockRejectedValueOnce(Object.assign(new Error("transport lost"), { status: 0 }))
      .mockRejectedValueOnce(Object.assign(new Error("already deleted"), { status: 404 }))
    vi.mocked(listAllOsceStations).mockResolvedValueOnce([{
      id: 9,
      quiz_id: 8,
      title: "Explain anticoagulant safety",
      recommended_duration_seconds: 480,
      order_index: 0,
      version: 2,
      checklist_count: 4,
      rubric_domain_count: 2,
      verification_state: "source_verified",
      created_at: "2026-09-11",
      updated_at: "2026-09-11"
    }])
    vi.mocked(getOsceStation).mockResolvedValueOnce({
      id: 9,
      quiz_id: 8,
      content: { schema_version: "osce.station.v1", title: "Explain anticoagulant safety" },
      order_index: 0,
      version: 2
    } as never)
    vi.spyOn(window, "confirm").mockReturnValue(true)
    render(<ManageTab onNavigateToCreate={() => {}} onNavigateToGenerate={() => {}} onStartQuiz={() => {}} />)
    fireEvent.click(screen.getByRole("button", { name: "Manage stations" }))
    fireEvent.click(screen.getByRole("button", { name: "Edit station Explain anticoagulant safety" }))

    fireEvent.click(screen.getByRole("button", { name: "Delete station Explain anticoagulant safety" }))

    expect(await screen.findByText("Delete status could not be confirmed. Station remains available.")).toBeInTheDocument()
    expect(screen.getByTestId("osce-station-editor")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Delete station Explain anticoagulant safety" })).toBeEnabled()
    expect(stationListRefetch).toHaveBeenCalled()
    expect(stationDetailRefetch).toHaveBeenCalled()

    fireEvent.click(screen.getByRole("button", { name: "Delete station Explain anticoagulant safety" }))

    await waitFor(() => expect(deleteStation).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(screen.queryByTestId("osce-station-editor")).not.toBeInTheDocument())
  })

  it("uses max existing order index plus one for a new station", async () => {
    vi.mocked(useAllOsceStationsQuery).mockReturnValue({
      data: [
        { id: 9, quiz_id: 8, title: "First", order_index: 0 },
        { id: 10, quiz_id: 8, title: "Imported A", order_index: 5 },
        { id: 11, quiz_id: 8, title: "Imported B", order_index: 5 }
      ],
      isLoading: false,
      refetch: stationListRefetch
    } as never)
    render(<ManageTab onNavigateToCreate={() => {}} onNavigateToGenerate={() => {}} onStartQuiz={() => {}} />)
    fireEvent.click(screen.getByRole("button", { name: "Manage stations" }))

    fireEvent.click(screen.getByRole("button", { name: "Add station" }))

    expect(await screen.findByTestId("osce-station-order-index")).toHaveTextContent("6")
  })

  it("keeps ambiguous create blocked for its quiz across editor selection and manager close", async () => {
    render(<ManageTab onNavigateToCreate={() => {}} onNavigateToGenerate={() => {}} onStartQuiz={() => {}} />)
    fireEvent.click(screen.getByRole("button", { name: "Manage stations" }))
    fireEvent.click(screen.getByRole("button", { name: "Add station" }))
    fireEvent.click(await screen.findByRole("button", { name: "Simulate ambiguous station create" }))

    expect(await screen.findByText("Station creation status is unknown.")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Add station" })).toBeDisabled()

    fireEvent.click(screen.getByRole("button", { name: "Edit station Explain anticoagulant safety" }))
    expect(screen.getByText("Station creation status is unknown.")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Add station" })).toBeDisabled()

    fireEvent.click(screen.getByRole("button", { name: "Close" }))
    fireEvent.click(screen.getByRole("button", { name: "Manage stations" }))

    expect(screen.getByText("Station creation status is unknown.")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Add station" })).toBeDisabled()
  })

  it("keeps create blocked after failed reconciliation and clears it after a fresh list succeeds", async () => {
    stationListRefetch
      .mockRejectedValueOnce(Object.assign(new Error("offline"), { status: 0 }))
      .mockResolvedValueOnce({ data: [] })
    render(<ManageTab onNavigateToCreate={() => {}} onNavigateToGenerate={() => {}} onStartQuiz={() => {}} />)
    fireEvent.click(screen.getByRole("button", { name: "Manage stations" }))
    fireEvent.click(screen.getByRole("button", { name: "Add station" }))
    fireEvent.click(await screen.findByRole("button", { name: "Simulate ambiguous station create" }))

    fireEvent.click(await screen.findByRole("button", { name: "Reload station list" }))

    expect(await screen.findByText("Could not refresh stations. Creation remains blocked.")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Add station" })).toBeDisabled()

    fireEvent.click(screen.getByRole("button", { name: "Reload station list" }))

    await waitFor(() => expect(screen.queryByText("Station creation status is unknown.")).not.toBeInTheDocument())
    expect(screen.getByRole("button", { name: "Add station" })).toBeEnabled()
    expect(stationListRefetch).toHaveBeenNthCalledWith(1, { throwOnError: true })
    expect(stationListRefetch).toHaveBeenNthCalledWith(2, { throwOnError: true })
    expect(quizRefetch).toHaveBeenCalled()
  })

  it("does not remount a dirty station editor when the same station version refreshes", async () => {
    const view = render(<ManageTab onNavigateToCreate={() => {}} onNavigateToGenerate={() => {}} onStartQuiz={() => {}} />)
    fireEvent.click(screen.getByRole("button", { name: "Manage stations" }))
    fireEvent.click(screen.getByRole("button", { name: "Edit station Explain anticoagulant safety" }))
    fireEvent.change(await screen.findByLabelText("Managed station draft"), {
      target: { value: "Local draft" }
    })

    stationVersion = 3
    stationTitle = "Server refresh"
    view.rerender(<ManageTab onNavigateToCreate={() => {}} onNavigateToGenerate={() => {}} onStartQuiz={() => {}} />)

    expect(screen.getByLabelText("Managed station draft")).toHaveValue("Local draft")
  })

  it("cancels deleting the managed quiz when its station draft is dirty", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(false)
    render(<ManageTab onNavigateToCreate={() => {}} onNavigateToGenerate={() => {}} onStartQuiz={() => {}} />)
    fireEvent.click(screen.getByRole("button", { name: "Manage stations" }))
    fireEvent.click(screen.getByRole("button", { name: "Edit station Explain anticoagulant safety" }))
    fireEvent.change(await screen.findByLabelText("Managed station draft"), {
      target: { value: "Local draft" }
    })

    fireEvent.click(screen.getByRole("button", { name: "Delete quiz Clinical communication" }))

    expect(window.confirm).toHaveBeenCalledWith("Discard unsaved station changes?")
    expect(screen.getByRole("button", { name: "Delete quiz Clinical communication" })).toBeInTheDocument()
    expect(screen.getByLabelText("Managed station draft")).toHaveValue("Local draft")
    expect(deleteQuiz).not.toHaveBeenCalled()
  })

  it("proceeds with managed quiz deletion after discarding the dirty station draft", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true)
    render(<ManageTab onNavigateToCreate={() => {}} onNavigateToGenerate={() => {}} onStartQuiz={() => {}} />)
    fireEvent.click(screen.getByRole("button", { name: "Manage stations" }))
    fireEvent.click(screen.getByRole("button", { name: "Edit station Explain anticoagulant safety" }))
    fireEvent.change(await screen.findByLabelText("Managed station draft"), {
      target: { value: "Local draft" }
    })

    fireEvent.click(screen.getByRole("button", { name: "Delete quiz Clinical communication" }))

    expect(screen.queryByTestId("osce-station-editor")).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Delete quiz Clinical communication" })).not.toBeInTheDocument()
  })

  it("cancels bulk deletion when it includes the managed quiz with a dirty station draft", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(false)
    render(<ManageTab onNavigateToCreate={() => {}} onNavigateToGenerate={() => {}} onStartQuiz={() => {}} />)
    fireEvent.click(screen.getByRole("button", { name: "Manage stations" }))
    fireEvent.click(screen.getByRole("button", { name: "Edit station Explain anticoagulant safety" }))
    fireEvent.change(await screen.findByLabelText("Managed station draft"), {
      target: { value: "Local draft" }
    })
    fireEvent.click(screen.getByRole("checkbox", { name: "Select quiz {{name}}" }))
    fireEvent.click(screen.getByTestId("manage-bulk-delete"))
    const popconfirm = (await screen.findByText("Delete selected quizzes?")).closest(".ant-popover")
    expect(popconfirm).not.toBeNull()
    fireEvent.click(within(popconfirm as HTMLElement).getByRole("button", { name: /^Delete$/i }))

    expect(deleteQuiz).not.toHaveBeenCalled()
    expect(screen.getByLabelText("Managed station draft")).toHaveValue("Local draft")
    expect(screen.getByRole("button", { name: "Delete quiz Clinical communication" })).toBeInTheDocument()
  })

  it("proceeds with bulk deletion after discarding the managed quiz station draft", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true)
    render(<ManageTab onNavigateToCreate={() => {}} onNavigateToGenerate={() => {}} onStartQuiz={() => {}} />)
    fireEvent.click(screen.getByRole("button", { name: "Manage stations" }))
    fireEvent.click(screen.getByRole("button", { name: "Edit station Explain anticoagulant safety" }))
    fireEvent.change(await screen.findByLabelText("Managed station draft"), {
      target: { value: "Local draft" }
    })
    fireEvent.click(screen.getByRole("checkbox", { name: "Select quiz {{name}}" }))
    fireEvent.click(screen.getByTestId("manage-bulk-delete"))
    const popconfirm = (await screen.findByText("Delete selected quizzes?")).closest(".ant-popover")
    expect(popconfirm).not.toBeNull()
    fireEvent.click(within(popconfirm as HTMLElement).getByRole("button", { name: /^Delete$/i }))

    await waitFor(() => expect(deleteQuiz).toHaveBeenCalledWith({ quizId: 8, version: 2 }))
    expect(screen.queryByTestId("osce-station-editor")).not.toBeInTheDocument()
  })
})
