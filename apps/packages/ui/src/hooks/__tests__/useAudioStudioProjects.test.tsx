import React from "react"
import { act, renderHook, waitFor } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { beforeEach, describe, expect, it, vi } from "vitest"

const serviceMocks = vi.hoisted(() => ({
  listAudioStudioProjects: vi.fn(),
  createAudioStudioProject: vi.fn(),
  updateAudioStudioProject: vi.fn(),
  getAudioStudioProject: vi.fn(),
  upsertAudioStudioSection: vi.fn()
}))

vi.mock("@/services/audio-studio", () => ({
  getAudioStudioProject: (...args: unknown[]) => serviceMocks.getAudioStudioProject(...args),
  upsertAudioStudioSection: (...args: unknown[]) => serviceMocks.upsertAudioStudioSection(...args),
  listAudioStudioProjects: (...args: unknown[]) =>
    serviceMocks.listAudioStudioProjects(...args),
  createAudioStudioProject: (...args: unknown[]) =>
    serviceMocks.createAudioStudioProject(...args),
  updateAudioStudioProject: (...args: unknown[]) =>
    serviceMocks.updateAudioStudioProject(...args)
}))

import {
  audioStudioProjectQueryKeys,
  useAudioStudioProjects,
  useAudioStudioProjectDetail,
  useCreateAudioStudioProject,
  useUpdateAudioStudioProject
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
    }, undefined)
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

  it("updates a project, upserts the server response, and marks it clean", async () => {
    useAudioStudioStore.getState().setProjects([
      {
        project_id: "brief-1",
        title: "Daily Brief",
        workflow: "briefing",
        status: "draft",
        revision_id: "rev-1",
        updated_at: "2026-06-23T12:00:00Z"
      }
    ])
    useAudioStudioStore.getState().setActiveProjectId("brief-1")
    useAudioStudioStore.getState().markProjectDirty("brief-1")
    serviceMocks.updateAudioStudioProject.mockResolvedValueOnce({
      project_id: "brief-1",
      title: "Daily Brief Updated",
      workflow: "briefing",
      status: "draft",
      revision_id: "rev-2",
      updated_at: "2026-06-23T12:05:00Z"
    })
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } }
    })
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries")

    const { result } = renderHook(
      () => useUpdateAudioStudioProject("brief-1"),
      { wrapper: buildWrapper(queryClient) }
    )

    await act(async () => {
      await result.current.mutateAsync({
        title: "Daily Brief Updated",
        base_revision_id: "rev-1"
      })
    })

    expect(serviceMocks.updateAudioStudioProject).toHaveBeenCalledWith(
      "brief-1",
      {
        title: "Daily Brief Updated",
        base_revision_id: "rev-1"
      },
      undefined
    )
    expect(invalidateSpy).toHaveBeenCalledWith({
      queryKey: audioStudioProjectQueryKeys.projects()
    })
    expect(useAudioStudioStore.getState().activeProject?.revision_id).toBe("rev-2")
    expect(
      useAudioStudioStore.getState().dirtyProjectIds["brief-1"]
    ).toBeUndefined()
  })
  it("B06 reopening hydrates saved sections from project detail", async () => {
    useAudioStudioStore.getState().setProjects([{ project_id: "p", title: "Saved", workflow: "narration", status: "draft" }])
    useAudioStudioStore.getState().setActiveProjectId("p")
    serviceMocks.getAudioStudioProject.mockResolvedValue({ project_id: "p", title: "Saved", workflow: "narration", status: "draft", current_revision_id: "r2", sections: [{ section_id: "s", body_text: "Retained", order_index: 2 }] })
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
    renderHook(() => useAudioStudioProjectDetail("p"), { wrapper: buildWrapper(client) })
    await waitFor(() => expect(useAudioStudioStore.getState().activeProject?.sections[0]?.body_text).toBe("Retained"))
    expect(useAudioStudioStore.getState().activeProject?.sections[0]?.order).toBe(2)
  })

  it("B06 saves ordered sections using each returned project revision", async () => {
    const saved = { project_id: "p", title: "Saved", workflow: "narration" as const, status: "draft", current_revision_id: "r1", sections: [{ section_id: "b", title: "Second", body_text: "Second", workflow: "narration" as const, order: 1 }, { section_id: "a", title: "First", body_text: "First", workflow: "narration" as const, order: 0 }] }
    useAudioStudioStore.getState().setProjects([saved])
    useAudioStudioStore.getState().setActiveProjectId("p")
    serviceMocks.updateAudioStudioProject.mockResolvedValue({ ...saved, current_revision_id: "r2" })
    serviceMocks.upsertAudioStudioSection.mockResolvedValueOnce({ current_revision_id: "r3" }).mockResolvedValueOnce({ current_revision_id: "r4" })
    serviceMocks.getAudioStudioProject.mockResolvedValue({ ...saved, current_revision_id: "r4" })
    const client = new QueryClient()
    const { result } = renderHook(() => useUpdateAudioStudioProject("p"), { wrapper: buildWrapper(client) })
    await act(async () => { await result.current.mutateAsync({ title: "Saved", base_revision_id: "r1" }) })
    expect(serviceMocks.upsertAudioStudioSection.mock.calls.map(([id, section, payload]) => [id, section, payload.base_revision_id])).toEqual([["p", "a", "r2"], ["p", "b", "r3"]])
    expect(useAudioStudioStore.getState().activeProject?.current_revision_id).toBe("r4")
  })

  it("B06 keeps newer edits while adopting its acknowledged save revision", async () => {
    const project = { project_id: "p", title: "Draft", workflow: "narration" as const, status: "draft", current_revision_id: "r1", sections: [{ section_id: "s", title: "Section", body_text: "Submitted", order: 0, workflow: "narration" as const }] }
    const store = useAudioStudioStore.getState(); store.setProjects([project]); store.setActiveProjectId("p"); store.markProjectDirty("p")
    let finish!: (value: unknown) => void
    serviceMocks.updateAudioStudioProject.mockReturnValueOnce(new Promise((resolve) => { finish = resolve }))
    serviceMocks.upsertAudioStudioSection.mockResolvedValueOnce({ current_revision_id: "r3" })
    serviceMocks.getAudioStudioProject.mockResolvedValueOnce({ ...project, current_revision_id: "r3" })
    const { result } = renderHook(() => useUpdateAudioStudioProject("p"), { wrapper: buildWrapper(new QueryClient()) })
    let saving!: Promise<unknown>
    act(() => { saving = result.current.mutateAsync({ title: "Draft", base_revision_id: "r1" }) })
    await waitFor(() => expect(serviceMocks.updateAudioStudioProject).toHaveBeenCalled())
    act(() => store.updateProjectLocal("p", { sections: [{ ...project.sections[0], body_text: "Newer unsaved edit" }] }))
    await act(async () => { finish({ ...project, current_revision_id: "r2" }); await saving })
    const current = useAudioStudioStore.getState()
    expect(current.activeProject?.sections[0].body_text).toBe("Newer unsaved edit")
    expect(current.activeProject?.current_revision_id).toBe("r3")
    expect(current.dirtyProjectIds.p).toBe(true)
  })

  it("B06 partial section save retains the draft at its last acknowledged revision", async () => {
    const project = { project_id: "p", title: "Draft", workflow: "narration" as const, status: "draft", current_revision_id: "r1", sections: [{ section_id: "a", title: "A", body_text: "First", order: 0, workflow: "narration" as const }, { section_id: "b", title: "B", body_text: "Second", order: 1, workflow: "narration" as const }] }
    const store = useAudioStudioStore.getState(); store.setProjects([project]); store.setActiveProjectId("p"); store.markProjectDirty("p")
    serviceMocks.updateAudioStudioProject.mockResolvedValueOnce({ ...project, current_revision_id: "r2" })
    serviceMocks.upsertAudioStudioSection.mockResolvedValueOnce({ current_revision_id: "r3" }).mockRejectedValueOnce(new Error("connection lost"))
    const { result } = renderHook(() => useUpdateAudioStudioProject("p"), { wrapper: buildWrapper(new QueryClient()) })
    await act(async () => { await expect(result.current.mutateAsync({ title: "Draft", base_revision_id: "r1" })).rejects.toThrow("connection lost") })
    expect(useAudioStudioStore.getState().activeProject?.current_revision_id).toBe("r3")
    expect(useAudioStudioStore.getState().activeProject?.sections).toEqual(project.sections)
    expect(useAudioStudioStore.getState().dirtyProjectIds.p).toBe(true)
  })

})
