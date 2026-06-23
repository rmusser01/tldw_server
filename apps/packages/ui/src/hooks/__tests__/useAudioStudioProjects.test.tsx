import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"

const serviceMocks = vi.hoisted(() => ({
  listAudioStudioProjects: vi.fn(),
  createAudioStudioProject: vi.fn()
}))

vi.mock("@/services/audio-studio", () => ({
  listAudioStudioProjects: (...args: unknown[]) =>
    serviceMocks.listAudioStudioProjects(...args),
  createAudioStudioProject: (...args: unknown[]) =>
    serviceMocks.createAudioStudioProject(...args)
}))

import {
  audioStudioProjectQueryKeys,
  useAudioStudioProjects,
  useCreateAudioStudioProject
} from "@/hooks/useAudioStudioProjects"
import { useAudioStudioStore } from "@/store/audio-studio"

const buildWrapper = (queryClient: QueryClient) => {
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
}

describe("useAudioStudioProjects", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    useAudioStudioStore.getState().resetAudioStudio()
  })

  it("loads server projects for the selected workflow and hydrates the store", async () => {
    serviceMocks.listAudioStudioProjects.mockResolvedValueOnce([
      {
        project_id: "brief-1",
        title: "Daily Brief",
        workflow: "briefing",
        status: "draft",
        revision_id: "rev-1",
        updated_at: "2026-06-23T12:00:00Z"
      }
    ])
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })

    const { result } = renderHook(
      () => useAudioStudioProjects({ workflow: "briefing" }),
      { wrapper: buildWrapper(queryClient) }
    )

    await waitFor(() => {
      expect(result.current.isSuccess).toBe(true)
    })

    expect(serviceMocks.listAudioStudioProjects).toHaveBeenCalledWith({
      workflow: "briefing"
    })
    expect(useAudioStudioStore.getState().projects[0].title).toBe("Daily Brief")
  })

  it("invalidates project queries after creating a project", async () => {
    serviceMocks.createAudioStudioProject.mockResolvedValueOnce({
      project_id: "pod-1",
      title: "Interview",
      workflow: "podcast",
      status: "draft",
      revision_id: "rev-1",
      updated_at: "2026-06-23T12:00:00Z"
    })
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries")

    const { result } = renderHook(() => useCreateAudioStudioProject(), {
      wrapper: buildWrapper(queryClient)
    })

    await act(async () => {
      await result.current.mutateAsync({
        title: "Interview",
        workflow: "podcast"
      })
    })

    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: audioStudioProjectQueryKeys.projects()
    })
    expect(useAudioStudioStore.getState().projects[0].workflow).toBe("podcast")
  })
})
