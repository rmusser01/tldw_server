import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"

import { TakeQuizTab } from "../TakeQuizTab"
import {
  useAttemptsQuery,
  useQuizzesQuery,
  useQuizQuery,
  useStartAttemptMutation,
  useSubmitAttemptMutation
} from "../../hooks"
import {
  useActiveOsceAttemptsQuery,
  useAllOsceStationsQuery,
  useStartOsceAttemptMutation
} from "../../hooks/useOsceQueries"

const authMocks = vi.hoisted(() => ({ getCurrentUser: vi.fn() }))
const clientMocks = vi.hoisted(() => ({ getConfig: vi.fn() }))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, value?: string | { defaultValue?: string }) =>
      typeof value === "string" ? value : value?.defaultValue ?? _key
  })
}))

vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: { getCurrentUser: authMocks.getCurrentUser }
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { getConfig: clientMocks.getConfig }
}))

vi.mock("@/hooks/useServerOnline", () => ({ useServerOnline: () => true }))
vi.mock("../../hooks/useQuizTimer", () => ({ useQuizTimer: () => null }))
vi.mock("../../hooks/useQuizAutoSave", () => ({
  useQuizAutoSave: () => ({
    storageUnavailable: false,
    restoreSavedAnswers: vi.fn(async () => false),
    clearSavedProgress: vi.fn(async () => {}),
    hasSavedProgress: vi.fn(async () => false),
    getSavedProgress: vi.fn(async () => null),
    forceSave: vi.fn(async () => {})
  })
}))

vi.mock("../../hooks", () => ({
  useAttemptsQuery: vi.fn(),
  useQuizzesQuery: vi.fn(),
  useQuizQuery: vi.fn(),
  useStartAttemptMutation: vi.fn(),
  useSubmitAttemptMutation: vi.fn()
}))

vi.mock("../../hooks/useOsceQueries", () => ({
  useActiveOsceAttemptsQuery: vi.fn(),
  useAllOsceStationsQuery: vi.fn(),
  useStartOsceAttemptMutation: vi.fn(),
  selectMostRecentlyModifiedOsceAttempt: (attempts: Array<{ id: number; last_modified_at: string }>) =>
    [...attempts].sort((left, right) =>
      right.last_modified_at.localeCompare(left.last_modified_at) || right.id - left.id
    )[0] ?? null
}))

vi.mock("../../osce/OscePracticePanel", () => ({
  OscePracticePanel: ({ attemptId, userScope }: { attemptId: number; userScope: string }) => (
    <div data-testid="osce-practice-panel" data-user-scope={userScope}>Attempt {attemptId}</div>
  )
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

const osceQuiz = {
  id: 9,
  name: "Clinical communication OSCE",
  description: "Practice counselling",
  activity_type: "osce" as const,
  total_questions: 0,
  total_stations: 2,
  passing_score: null,
  time_limit_seconds: null,
  deleted: false,
  client_id: "test",
  version: 1,
  created_at: "2026-09-11T00:00:00Z"
}

const activeAttempts = [
  {
    id: 21,
    quiz_id: 9,
    station_id: 90,
    station_title: "Deleted station snapshot",
    client_attempt_id: "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
    state: "in_progress" as const,
    version: 2,
    started_at: "2026-09-11T09:00:00Z",
    self_assessment_started_at: null,
    completed_at: null,
    last_modified_at: "2026-09-11T10:00:00Z",
    elapsed_seconds: null,
    checklist_met_count: null,
    checklist_total: null,
    rubric_results: []
  },
  {
    id: 22,
    quiz_id: 9,
    station_id: 12,
    station_title: "Most recent station",
    client_attempt_id: "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
    state: "self_assessment" as const,
    version: 5,
    started_at: "2026-09-11T09:30:00Z",
    self_assessment_started_at: "2026-09-11T09:38:00Z",
    completed_at: null,
    last_modified_at: "2026-09-11T10:05:00Z",
    elapsed_seconds: 480,
    checklist_met_count: null,
    checklist_total: null,
    rubric_results: []
  }
]

describe("TakeQuizTab OSCE routing", () => {
  const startQuestion = vi.fn()
  const startOsce = vi.fn()

  beforeEach(() => {
    vi.clearAllMocks()
    authMocks.getCurrentUser.mockResolvedValue({ id: 42 })
    clientMocks.getConfig.mockResolvedValue({
      serverUrl: "https://server-a.example.test",
      authMode: "multi-user",
      orgId: 7,
      accessToken: null,
      apiKey: null
    })
    vi.mocked(useAttemptsQuery).mockReturnValue({ data: { items: [], count: 0 } } as any)
    vi.mocked(useQuizzesQuery).mockReturnValue({ data: { items: [osceQuiz], count: 1 }, isLoading: false } as any)
    vi.mocked(useQuizQuery).mockImplementation((id: number | null) => ({ data: id === 9 ? osceQuiz : null } as any))
    vi.mocked(useStartAttemptMutation).mockReturnValue({ mutateAsync: startQuestion, isPending: false } as any)
    vi.mocked(useSubmitAttemptMutation).mockReturnValue({ mutateAsync: vi.fn(), isPending: false } as any)
    vi.mocked(useActiveOsceAttemptsQuery).mockReturnValue({
      data: {
        items: activeAttempts,
        count: 2,
        has_more: false,
        next_offset: null,
        pagination: { mode: "offset", total: 2, offset: 0, limit: 200, has_more: false, next_offset: null }
      },
      isLoading: false
    } as any)
    vi.mocked(useAllOsceStationsQuery).mockReturnValue({
      data: [
        { id: 11, quiz_id: 9, title: "First station", order_index: 0 },
        { id: 12, quiz_id: 9, title: "Most recent station", order_index: 1 }
      ],
      isLoading: false
    } as any)
    startOsce.mockResolvedValue({ id: 33, quiz_id: 9, station_id: 11, state: "in_progress" })
    vi.mocked(useStartOsceAttemptMutation).mockReturnValue({ mutateAsync: startOsce, isPending: false } as any)
  })

  const renderTab = (props: Record<string, unknown> = {}) => render(
    <MemoryRouter>
      <TakeQuizTab onNavigateToGenerate={() => {}} onNavigateToCreate={() => {}} {...props} />
    </MemoryRouter>
  )

  it("resumes the most recently modified attempt and keeps every active attempt selectable", async () => {
    renderTab()

    const card = screen.getByTestId("take-quiz-card-9")
    fireEvent.click(within(card).getByRole("button", { name: /Practice station/ }))

    expect(await screen.findByTestId("osce-practice-panel")).toHaveTextContent("Attempt 22")
    expect(screen.getByRole("combobox", { name: "Active OSCE attempts" })).toBeInTheDocument()
    fireEvent.mouseDown(screen.getByRole("combobox", { name: "Active OSCE attempts" }))
    expect(await screen.findByRole("option", { name: "Deleted station snapshot · In progress" })).toBeInTheDocument()
    expect(screen.getByRole("option", { name: "Most recent station · Self-assessment" })).toBeInTheDocument()
    expect(screen.getByTestId("osce-practice-panel").getAttribute("data-user-scope")).toMatch(
      /^server:[^:]+:auth:multi-user:org:7:user:42$/
    )
    expect(startQuestion).not.toHaveBeenCalled()
  })

  it("shows a multi-station picker and starts the selected station", async () => {
    vi.mocked(useActiveOsceAttemptsQuery).mockReturnValue({
      data: {
        items: [], count: 0, has_more: false, next_offset: null,
        pagination: { mode: "offset", total: 0, offset: 0, limit: 200, has_more: false, next_offset: null }
      },
      isLoading: false
    } as any)
    renderTab()

    fireEvent.click(screen.getByRole("button", { name: /Practice station/ }))
    const picker = await screen.findByRole("combobox", { name: "Choose a station" })
    fireEvent.mouseDown(picker)
    fireEvent.click(await screen.findByText("Most recent station"))
    fireEvent.click(screen.getByRole("button", { name: "Start new practice" }))

    await waitFor(() => expect(startOsce).toHaveBeenCalledWith(expect.objectContaining({ stationId: 12 })))
    expect(await screen.findByTestId("osce-practice-panel")).toHaveTextContent("Attempt 33")
  })

  it("routes a direct OSCE start without invoking ordinary attempts", async () => {
    renderTab({ startQuizId: 9, onStartHandled: vi.fn() })

    expect(await screen.findByTestId("osce-practice-panel")).toHaveTextContent("Attempt 22")
    expect(startQuestion).not.toHaveBeenCalled()
  })

  it("keeps ordinary quiz start behavior unchanged", async () => {
    const ordinaryQuiz = { ...osceQuiz, id: 7, name: "Biology", activity_type: "questions" as const, total_questions: 1, total_stations: 0 }
    vi.mocked(useQuizzesQuery).mockReturnValue({ data: { items: [ordinaryQuiz], count: 1 }, isLoading: false } as any)
    vi.mocked(useQuizQuery).mockReturnValue({ data: ordinaryQuiz } as any)
    startQuestion.mockResolvedValue({
      id: 100,
      quiz_id: 7,
      started_at: "2026-09-11T10:00:00Z",
      total_possible: 1,
      answers: [],
      questions: [{ id: 1, quiz_id: 7, question_type: "true_false", question_text: "Cells are alive.", points: 1, order_index: 0 }]
    })
    renderTab()

    fireEvent.click(screen.getByRole("button", { name: /Start Quiz/ }))
    fireEvent.click(screen.getByRole("button", { name: "Begin Quiz" }))

    await waitFor(() => expect(startQuestion).toHaveBeenCalledWith(7))
  })
})
