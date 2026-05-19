import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  useStudyPackCreateMutation,
  useStudyPackJobQuery,
  useStudyPackQuery,
  isTerminalStudyPackJobStatus
} from "../useStudyPackQueries"
import {
  createStudyPackJob,
  getStudyPackJob,
  getStudyPack,
  type StudyPackCreateJobRequest,
  type StudyPackJobStatusResponse
} from "@/services/flashcards"

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: { hasFlashcards: true },
    loading: false
  })
}))

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/services/flashcards", async () => {
  const actual = await vi.importActual<typeof import("@/services/flashcards")>(
    "@/services/flashcards"
  )
  return {
    ...actual,
    createStudyPackJob: vi.fn(),
    getStudyPackJob: vi.fn(),
    getStudyPack: vi.fn()
  }
})

const buildWrapper = (queryClient: QueryClient) => {
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
}

const createRequest: StudyPackCreateJobRequest = {
  title: "Networks",
  workspace_id: "ws-1",
  source_items: [
    {
      source_type: "media",
      source_id: "42",
      source_title: "Lecture 5"
    }
  ]
}

const queuedJob = (): StudyPackJobStatusResponse => ({
  job: {
    id: 91,
    status: "queued",
    domain: "study_packs",
    queue: "default",
    job_type: "study_pack_generate"
  },
  study_pack: null,
  error: null
})

const completedJob = (): StudyPackJobStatusResponse => ({
  job: {
    id: 91,
    status: "completed",
    domain: "study_packs",
    queue: "default",
    job_type: "study_pack_generate"
  },
  study_pack: {
    id: 31,
    workspace_id: "ws-1",
    title: "Networks",
    deck_id: 8,
    source_bundle_json: {
      items: [
        {
          source_type: "media",
          source_id: "42",
          label: "Lecture 5",
          locator: {
            media_id: 42
          }
        }
      ]
    },
    generation_options_json: {
      deck_mode: "new"
    },
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

describe("study pack query hooks", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.useRealTimers()
  })

  it("posts study-pack create-job requests", async () => {
    vi.mocked(createStudyPackJob).mockResolvedValue({
      job: {
        id: 91,
        status: "queued",
        domain: "study_packs",
        queue: "default",
        job_type: "study_pack_generate"
      }
    })

    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false },
        mutations: { retry: false }
      }
    })

    const { result } = renderHook(() => useStudyPackCreateMutation(), {
      wrapper: buildWrapper(queryClient)
    })

    await act(async () => {
      await result.current.mutateAsync(createRequest)
    })

    expect(createStudyPackJob).toHaveBeenCalledWith(createRequest)
  })

  it("polls study-pack jobs until they become terminal", async () => {
    vi.mocked(getStudyPackJob).mockResolvedValue(queuedJob())

    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false }
      }
    })

    const { result } = renderHook(() => useStudyPackJobQuery(91), {
      wrapper: buildWrapper(queryClient)
    })

    await waitFor(() => {
      expect(result.current.data?.job.status).toBe("queued")
    })

    expect(vi.mocked(getStudyPackJob)).toHaveBeenCalledTimes(1)

    const query = queryClient.getQueryCache().find({
      queryKey: ["flashcards:study-packs:job", 91]
    })
    const refetchInterval = query?.options.refetchInterval

    expect(typeof refetchInterval).toBe("function")
    expect(refetchInterval?.(query!)).toBe(1500)

    queryClient.setQueryData(["flashcards:study-packs:job", 91], completedJob())

    expect(refetchInterval?.(query!)).toBe(false)
  })

  it("fetches a study pack by id", async () => {
    vi.mocked(getStudyPack).mockResolvedValue(completedJob().study_pack!)

    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false }
      }
    })

    const { result } = renderHook(() => useStudyPackQuery(31), {
      wrapper: buildWrapper(queryClient)
    })

    await waitFor(() => {
      expect(result.current.data?.id).toBe(31)
    })

    expect(getStudyPack).toHaveBeenCalledWith(31)
  })

  it("respects explicit disabled flags for job and pack queries", async () => {
    const queryClient = new QueryClient({
      defaultOptions: {
        queries: { retry: false }
      }
    })

    renderHook(() => useStudyPackJobQuery(91, { enabled: false }), {
      wrapper: buildWrapper(queryClient)
    })
    renderHook(() => useStudyPackQuery(31, { enabled: false }), {
      wrapper: buildWrapper(queryClient)
    })

    await waitFor(() => {
      expect(vi.mocked(getStudyPackJob)).not.toHaveBeenCalled()
      expect(vi.mocked(getStudyPack)).not.toHaveBeenCalled()
    })
  })

  it("treats completed, failed, and cancelled jobs as terminal", () => {
    expect(isTerminalStudyPackJobStatus("completed")).toBe(true)
    expect(isTerminalStudyPackJobStatus("failed")).toBe(true)
    expect(isTerminalStudyPackJobStatus("cancelled")).toBe(true)
    expect(isTerminalStudyPackJobStatus("running")).toBe(false)
  })
})
