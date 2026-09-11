import React from "react"
import { renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"

const mockBgRequest = vi.hoisted(() => vi.fn())

vi.mock("@/services/background-proxy", () => ({ bgRequest: mockBgRequest }))

import {
  beginOsceSelfAssessment,
  completeOsceAttempt,
  createOsceStation,
  deleteOsceStation,
  getOsceAttempt,
  getOsceStation,
  listAllOsceStations,
  listOsceAttempts,
  listOsceStations,
  patchOsceAttempt,
  startOsceAttempt,
  updateOsceStation,
  type OsceStationCreateContent
} from "@/services/osce"
import {
  osceKeys,
  useCreateOsceStationMutation,
  useUpdateOsceStationMutation
} from "@/components/Quiz/hooks/useOsceQueries"
import { buildQuizExport } from "@/components/Quiz/osce/oscePortability"

const content: OsceStationCreateContent = {
  schema_version: "osce.station.v1",
  title: "Discuss safe anticoagulant use",
  candidate_instructions: "Speak with a simulated patient.",
  candidate_task: "Explain key safety advice.",
  patient_context: { text: "A fictional adult recently started warfarin.", citations: [] },
  recommended_duration_seconds: 480,
  checklist_items: [{ label: "Explains the treatment purpose", rationale: "Supports safe use.", citations: [] }],
  rubric_domains: [{
    label: "Communication",
    levels: [
      { label: "Needs development", description: "The explanation is incomplete." },
      { label: "Effective", description: "The explanation is clear." }
    ]
  }],
  expected_key_points: [{ text: "Discuss monitoring and warning signs.", citations: [] }]
}

describe("OSCE service wire contract", () => {
  beforeEach(() => {
    mockBgRequest.mockReset()
    mockBgRequest.mockResolvedValue({})
  })

  it("encodes compact station routes and expected-version mutations", async () => {
    await listOsceStations(7, { limit: 10, offset: 20 })
    await getOsceStation(7, 9)
    await createOsceStation(7, { content, order_index: 2 })
    await updateOsceStation(7, 9, { expected_version: 4, content, order_index: 1 })
    await deleteOsceStation(7, 9, 5)

    expect(mockBgRequest.mock.calls.map(([request]) => request)).toEqual([
      expect.objectContaining({ path: "/api/v1/quizzes/7/osce-stations?limit=10&offset=20", method: "GET" }),
      expect.objectContaining({ path: "/api/v1/quizzes/7/osce-stations/9", method: "GET" }),
      expect.objectContaining({ path: "/api/v1/quizzes/7/osce-stations", method: "POST", body: { content, order_index: 2 } }),
      expect.objectContaining({ path: "/api/v1/quizzes/7/osce-stations/9", method: "PATCH", body: { expected_version: 4, content, order_index: 1 } }),
      expect.objectContaining({ path: "/api/v1/quizzes/7/osce-stations/9?expected_version=5", method: "DELETE" })
    ])
  })

  it("encodes repeated attempt states and all expected-version bodies", async () => {
    await startOsceAttempt(9, "4a0ea5d1-2df4-45a4-a4d7-211e35b1c405")
    await listOsceAttempts({
      quiz_id: 7,
      station_id: 9,
      states: ["in_progress", "self_assessment"],
      limit: 25,
      offset: 50
    })
    await getOsceAttempt(11)
    await patchOsceAttempt(11, { expected_version: 2, notes: "Private notes" })
    await beginOsceSelfAssessment(11, 3)
    await completeOsceAttempt(11, 4)

    expect(mockBgRequest.mock.calls.map(([request]) => request)).toEqual([
      expect.objectContaining({
        path: "/api/v1/quizzes/osce-stations/9/attempts",
        method: "POST",
        body: { client_attempt_id: "4a0ea5d1-2df4-45a4-a4d7-211e35b1c405" }
      }),
      expect.objectContaining({
        path: "/api/v1/quizzes/osce-attempts?quiz_id=7&station_id=9&state=in_progress&state=self_assessment&limit=25&offset=50",
        method: "GET"
      }),
      expect.objectContaining({ path: "/api/v1/quizzes/osce-attempts/11", method: "GET" }),
      expect.objectContaining({ path: "/api/v1/quizzes/osce-attempts/11", method: "PATCH", body: { expected_version: 2, notes: "Private notes" } }),
      expect.objectContaining({ path: "/api/v1/quizzes/osce-attempts/11/begin-self-assessment", method: "POST", body: { expected_version: 3 } }),
      expect.objectContaining({ path: "/api/v1/quizzes/osce-attempts/11/complete", method: "POST", body: { expected_version: 4 } })
    ])
  })

  it("propagates request errors unchanged", async () => {
    const conflict = Object.assign(new Error("Station was modified"), { status: 409 })
    mockBgRequest.mockRejectedValueOnce(conflict)

    await expect(getOsceStation(7, 9)).rejects.toBe(conflict)
  })

  it("loads every station page without silently truncating", async () => {
    mockBgRequest
      .mockResolvedValueOnce({
        items: [{ id: 1 }, { id: 2 }], count: 2, has_more: true, next_offset: 2,
        pagination: { mode: "offset", total: 3, offset: 0, limit: 2, has_more: true, next_offset: 2 }
      })
      .mockResolvedValueOnce({
        items: [{ id: 3 }], count: 1, has_more: false, next_offset: null,
        pagination: { mode: "offset", total: 3, offset: 2, limit: 2, has_more: false, next_offset: null }
      })

    await expect(listAllOsceStations(7, { pageSize: 2 })).resolves.toEqual([
      { id: 1 }, { id: 2 }, { id: 3 }
    ])
    expect(mockBgRequest.mock.calls.map(([request]) => request.path)).toEqual([
      "/api/v1/quizzes/7/osce-stations?limit=2&offset=0",
      "/api/v1/quizzes/7/osce-stations?limit=2&offset=2"
    ])
  })

  it("fails closed when station pagination does not advance", async () => {
    mockBgRequest.mockResolvedValueOnce({
      items: [{ id: 1 }], count: 1, has_more: true, next_offset: 0,
      pagination: { mode: "offset", total: null, offset: 0, limit: 2, has_more: true, next_offset: 0 }
    })

    await expect(listAllOsceStations(7, { pageSize: 2 })).rejects.toThrow(
      "OSCE station pagination did not advance"
    )
    expect(mockBgRequest).toHaveBeenCalledTimes(1)
  })
})

describe("OSCE query keys and invalidation", () => {
  beforeEach(() => {
    mockBgRequest.mockReset()
    mockBgRequest.mockResolvedValue({})
  })

  it("keeps stable station and repeated-state attempt keys", () => {
    expect(osceKeys.stations(7, { limit: 10, offset: 0 })).toEqual([
      "quizzes", "osce", "stations", 7, { limit: 10, offset: 0 }
    ])
    expect(osceKeys.attempts({ states: ["in_progress", "self_assessment"] })).toEqual([
      "quizzes", "osce", "attempts", { states: ["in_progress", "self_assessment"] }
    ])
  })

  it("invalidates quiz and station caches after a station update", async () => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const invalidate = vi.spyOn(queryClient, "invalidateQueries")
    mockBgRequest.mockResolvedValueOnce({ id: 9, quiz_id: 7, content, version: 5 })
    const wrapper = ({ children }: { children: React.ReactNode }) =>
      React.createElement(QueryClientProvider, { client: queryClient }, children)
    const { result } = renderHook(() => useUpdateOsceStationMutation(), { wrapper })

    await result.current.mutateAsync({ quizId: 7, stationId: 9, request: { expected_version: 4, content } })

    await waitFor(() => {
      expect(invalidate).toHaveBeenCalledWith({ queryKey: ["quizzes:detail", 7] })
      expect(invalidate).toHaveBeenCalledWith({ queryKey: ["quizzes", "osce", "stations", 7] })
    })
  })

  it("invalidates quiz and station caches after station creation", async () => {
    const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    const invalidate = vi.spyOn(queryClient, "invalidateQueries")
    mockBgRequest.mockResolvedValueOnce({ id: 9, quiz_id: 7, content, version: 1 })
    const wrapper = ({ children }: { children: React.ReactNode }) =>
      React.createElement(QueryClientProvider, { client: queryClient }, children)
    const { result } = renderHook(() => useCreateOsceStationMutation(), { wrapper })

    await result.current.mutateAsync({ quizId: 7, request: { content, order_index: 0 } })

    await waitFor(() => {
      expect(invalidate).toHaveBeenCalledWith({ queryKey: ["quizzes:list"] })
      expect(invalidate).toHaveBeenCalledWith({ queryKey: ["quizzes:detail", 7] })
      expect(invalidate).toHaveBeenCalledWith({ queryKey: ["quizzes", "osce", "stations", 7] })
      expect(queryClient.getQueryData(osceKeys.station(7, 9))).toEqual(
        expect.objectContaining({ id: 9, quiz_id: 7 })
      )
    })
  })
})

describe("quiz portability", () => {
  const questionQuiz = {
    activity_type: "questions" as const,
    quiz: { id: 1, name: "Recall", activity_type: "questions" as const },
    questions: [{ id: 2, question_type: "true_false" as const, question_text: "Safe?", correct_answer: "true", points: 1, order_index: 0 }]
  }
  const osceQuiz = {
    activity_type: "osce" as const,
    quiz: { id: 3, name: "Counselling", activity_type: "osce" as const, total_stations: 1 },
    stations: [{
      id: 9,
      quiz_id: 3,
      content,
      order_index: 0,
      version: 2,
      origin: "generated" as const,
      provenance: { provider: "local", api_key: "must-not-export" },
      source_bundle: [],
      verification_state: "source_verified" as const,
      verification_timestamp: "2026-09-11T00:00:00Z",
      verification_summary: "Verified against selected sources",
      deleted: false,
      created_at: "2026-09-11T00:00:00Z",
      updated_at: "2026-09-11T00:00:00Z",
      attempts: [{ notes: "must-not-export" }],
      candidate_notes: "must-not-export"
    }]
  }

  it("uses v1 only for question quizzes and v2 when OSCE is present", () => {
    expect(buildQuizExport([questionQuiz], "2026-09-11T01:00:00Z").export_format).toBe("tldw.quiz.export.v1")
    expect(buildQuizExport([questionQuiz, osceQuiz], "2026-09-11T01:00:00Z").export_format).toBe("tldw.quiz.export.v2")
  })

  it("exports editable OSCE content and safe provenance without attempts or notes", () => {
    const exported = buildQuizExport([osceQuiz], "2026-09-11T01:00:00Z")
    const serialized = JSON.stringify(exported)

    expect(serialized).toContain("Discuss safe anticoagulant use")
    expect(serialized).toContain("source_verified")
    expect(serialized).toContain("local")
    expect(serialized).not.toContain("must-not-export")
    expect(serialized).not.toContain("candidate_notes")
    expect(serialized).not.toContain("attempts")
  })
})
