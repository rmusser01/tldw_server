// @vitest-environment jsdom
import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useStudySuggestions } from "../useStudySuggestions"
import {
  getStudySuggestionAnchorStatus,
  getStudySuggestionSnapshot,
  refreshStudySuggestionSnapshot
} from "@/services/studySuggestions"

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/services/studySuggestions", () => ({
  getStudySuggestionAnchorStatus: vi.fn(),
  getStudySuggestionSnapshot: vi.fn(),
  refreshStudySuggestionSnapshot: vi.fn(),
  performStudySuggestionAction: vi.fn()
}))

const buildWrapper = (queryClient: QueryClient) => {
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
}

const buildSnapshot = (id: number, label: string) => ({
  snapshot: {
    id,
    service: "quiz",
    activity_type: "quiz_attempt",
    anchor_type: "quiz_attempt",
    anchor_id: 101,
    suggestion_type: "study_suggestions",
    status: "active",
    payload: {
      summary: {
        score: 7,
        correct_count: 2,
        total_count: 4
      },
      topics: [
        {
          id: "topic-1",
          display_label: label,
          type: "weakness",
          status: "weakness",
          selected: true
        }
      ]
    },
    user_selection: {
      selected_topic_ids: ["topic-1"]
    },
    refreshed_from_snapshot_id: null,
    created_at: "2026-04-05T18:00:00Z",
    last_modified: "2026-04-05T18:00:00Z"
  },
  live_evidence: {
    "topic-1": {
      source_available: true,
      source_type: "note",
      source_id: "note-7"
    }
  }
})

const buildSnapshotV2 = (id: number) => ({
  snapshot: {
    id,
    service: "quiz",
    activity_type: "quiz_attempt",
    anchor_type: "quiz_attempt",
    anchor_id: 101,
    suggestion_type: "study_suggestions",
    status: "active",
    payload: {
      summary: {
        score: 8,
        correct_count: 3,
        total_count: 4
      },
      topics: [
        {
          id: "topic-local-1",
          display_label: "Cardiovascular review",
          topic_key: "cardio-001",
          canonical_label: "Cardiovascular review",
          evidence_reasons: ["topic_gap", "incorrect_answer"],
          type: "grounded",
          status: "weakness",
          selected: true
        }
      ]
    },
    user_selection: {
      selected_topic_ids: ["cardio-001"]
    },
    refreshed_from_snapshot_id: null,
    created_at: "2026-04-05T18:00:00Z",
    last_modified: "2026-04-05T18:00:00Z"
  },
  live_evidence: {
    "cardio-001": {
      source_available: true,
      source_type: "note",
      source_id: "note-9"
    }
  }
})

describe("useStudySuggestions", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.useRealTimers()
  })

  it("polls the anchor until the status becomes ready", async () => {
    vi.mocked(getStudySuggestionAnchorStatus)
      .mockResolvedValueOnce({
        anchor_type: "quiz_attempt",
        anchor_id: 101,
        status: "pending",
        job_id: 44,
        snapshot_id: null
      })
      .mockResolvedValueOnce({
        anchor_type: "quiz_attempt",
        anchor_id: 101,
        status: "ready",
        job_id: null,
        snapshot_id: 88
      })
    vi.mocked(getStudySuggestionSnapshot).mockResolvedValue(buildSnapshot(88, "Renal basics"))

    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false }
      }
    })

    const { result } = renderHook(() => useStudySuggestions("quiz_attempt", 101), {
      wrapper: buildWrapper(queryClient)
    })

    await waitFor(() => {
      expect(result.current.status).toBe("pending")
    })

    expect(getStudySuggestionAnchorStatus).toHaveBeenCalledTimes(1)

    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 1700))
    })

    await waitFor(() => {
      expect(result.current.status).toBe("ready")
    })

    await waitFor(() => {
      expect(getStudySuggestionAnchorStatus).toHaveBeenCalledTimes(2)
      expect(getStudySuggestionSnapshot).toHaveBeenCalledWith(
        88,
        expect.objectContaining({
          signal: expect.any(AbortSignal)
        })
      )
    })
    await waitFor(() => {
      expect(result.current.snapshot?.snapshot.id).toBe(88)
    })
  })

  it("keeps the old snapshot visible while a refresh resolves a replacement snapshot", async () => {
    let resolveReplacementSnapshot: ((value: ReturnType<typeof buildSnapshot>) => void) | null = null

    vi.mocked(getStudySuggestionAnchorStatus)
      .mockResolvedValueOnce({
        anchor_type: "flashcard_review_session",
        anchor_id: 202,
        status: "ready",
        job_id: null,
        snapshot_id: 88
      })
      .mockResolvedValueOnce({
        anchor_type: "flashcard_review_session",
        anchor_id: 202,
        status: "pending",
        job_id: 55,
        snapshot_id: null
      })
      .mockResolvedValueOnce({
        anchor_type: "flashcard_review_session",
        anchor_id: 202,
        status: "ready",
        job_id: null,
        snapshot_id: 99
      })

    vi.mocked(getStudySuggestionSnapshot).mockImplementation(async (snapshotId: number) => {
      if (snapshotId === 88) {
        return buildSnapshot(88, "Initial topic")
      }
      if (snapshotId === 99) {
        return await new Promise((resolve) => {
          resolveReplacementSnapshot = resolve
        })
      }
      throw new Error(`Unexpected snapshot id: ${snapshotId}`)
    })

    vi.mocked(refreshStudySuggestionSnapshot).mockResolvedValue({
      job: {
        id: 501,
        status: "queued"
      }
    })

    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })

    const { result } = renderHook(() => useStudySuggestions("flashcard_review_session", 202), {
      wrapper: buildWrapper(queryClient)
    })

    await waitFor(() => {
      expect(result.current.snapshot?.snapshot.id).toBe(88)
    })

    await act(async () => {
      await result.current.refresh()
    })

    await waitFor(() => {
      expect(result.current.status).toBe("pending")
    })

    expect(result.current.snapshot?.snapshot.id).toBe(88)

    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 1700))
    })

    await waitFor(() => {
      expect(result.current.status).toBe("ready")
    })

    expect(result.current.snapshot?.snapshot.id).toBe(88)

    await act(async () => {
      resolveReplacementSnapshot?.(buildSnapshot(99, "Replacement topic"))
    })

    await waitFor(() => {
      expect(result.current.snapshot?.snapshot.id).toBe(99)
    })
  })

  it("stops polling after a failed terminal status without fetching a snapshot", async () => {
    vi.mocked(getStudySuggestionAnchorStatus).mockResolvedValue({
      anchor_type: "quiz_attempt",
      anchor_id: 303,
      status: "failed",
      job_id: 77,
      snapshot_id: null
    })

    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false }
      }
    })

    const { result } = renderHook(() => useStudySuggestions("quiz_attempt", 303), {
      wrapper: buildWrapper(queryClient)
    })

    await waitFor(() => {
      expect(result.current.status).toBe("failed")
    })

    expect(result.current.snapshot).toBeNull()
    expect(getStudySuggestionSnapshot).not.toHaveBeenCalled()
    expect(getStudySuggestionAnchorStatus).toHaveBeenCalledTimes(1)

    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 1700))
    })

    expect(getStudySuggestionAnchorStatus).toHaveBeenCalledTimes(1)
    expect(getStudySuggestionSnapshot).not.toHaveBeenCalled()
  })

  it("preserves V2 snapshot topic fields in the hook response", async () => {
    vi.mocked(getStudySuggestionAnchorStatus).mockResolvedValue({
      anchor_type: "quiz_attempt",
      anchor_id: 404,
      status: "ready",
      job_id: null,
      snapshot_id: 91
    })
    vi.mocked(getStudySuggestionSnapshot).mockResolvedValue(buildSnapshotV2(91))

    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false }
      }
    })

    const { result } = renderHook(() => useStudySuggestions("quiz_attempt", 404), {
      wrapper: buildWrapper(queryClient)
    })

    await waitFor(() => {
      expect(result.current.snapshot?.snapshot.id).toBe(91)
    })

    const snapshot = result.current.snapshot
    const topic =
      snapshot &&
      snapshot.snapshot.payload &&
      typeof snapshot.snapshot.payload === "object" &&
      !Array.isArray(snapshot.snapshot.payload)
        ? (snapshot.snapshot.payload as { topics?: Array<Record<string, unknown>> }).topics?.[0]
        : null

    expect(topic).toMatchObject({
      id: "topic-local-1",
      topic_key: "cardio-001",
      canonical_label: "Cardiovascular review",
      evidence_reasons: ["topic_gap", "incorrect_answer"]
    })
  })
})
