import React from "react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  createManuscriptAnnotation,
  deleteManuscriptAnnotation,
  listManuscriptAnnotations,
  updateManuscriptAnnotation,
  type ManuscriptAnnotationListResponse
} from "@/services/writing-playground"
import {
  buildWritingAnnotationsQueryKey,
  useWritingAnnotations
} from "../hooks/useWritingAnnotations"
import { resolveWritingAnnotationTargetContext } from "../writing-annotation-types"

vi.mock("@/services/writing-playground", () => ({
  createManuscriptAnnotation: vi.fn(),
  deleteManuscriptAnnotation: vi.fn(),
  listManuscriptAnnotations: vi.fn(),
  updateManuscriptAnnotation: vi.fn(),
  reviewManuscriptSelection: vi.fn(),
  reviewManuscriptScene: vi.fn()
}))

const emptyList: ManuscriptAnnotationListResponse = {
  annotations: [],
  total: 0,
  limit: 50,
  offset: 0,
  has_more: false,
  next_offset: null,
  pagination: {
    mode: "offset",
    limit: 50,
    offset: 0,
    total: 0,
    has_more: false,
    next_offset: null
  }
}

const createQueryClient = () =>
  new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: Infinity
      },
      mutations: {
        retry: false
      }
    }
  })

function renderAnnotationsHook(
  props: Parameters<typeof useWritingAnnotations>[0],
  queryClient = createQueryClient()
) {
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )

  return {
    ...renderHook(() => useWritingAnnotations(props), { wrapper }),
    queryClient
  }
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(listManuscriptAnnotations).mockResolvedValue(emptyList)
  vi.mocked(createManuscriptAnnotation).mockResolvedValue({
    id: "annotation-1",
    project_id: "project-1",
    target_type: "scene",
    target_id: "scene-1",
    status: "open",
    category: "other",
    tags: [],
    source: "user",
    body: "Created",
    metadata: {},
    anchor_status: "scene_level",
    scene_level: true,
    created_at: "2026-06-25T00:00:00Z",
    last_modified: "2026-06-25T00:00:00Z",
    deleted: false,
    client_id: "test",
    version: 1
  })
  vi.mocked(updateManuscriptAnnotation).mockResolvedValue({
    id: "annotation-1",
    project_id: "project-1",
    target_type: "scene",
    target_id: "scene-1",
    status: "resolved",
    category: "other",
    tags: [],
    source: "user",
    body: "Updated",
    metadata: {},
    anchor_status: "scene_level",
    scene_level: true,
    created_at: "2026-06-25T00:00:00Z",
    last_modified: "2026-06-25T00:00:00Z",
    deleted: false,
    client_id: "test",
    version: 2
  })
  vi.mocked(deleteManuscriptAnnotation).mockResolvedValue()
})

describe("useWritingAnnotations", () => {
  it("keeps scene annotation context on the selected scene id while scene binding loads", () => {
    expect(
      resolveWritingAnnotationTargetContext({
        projectId: "project-1",
        activeNodeType: "scene",
        activeNodeId: "scene-1",
        activeSceneId: null
      })
    ).toEqual({ targetType: "scene", targetId: "scene-1" })

    expect(
      resolveWritingAnnotationTargetContext({
        projectId: "project-1",
        activeNodeType: "scene",
        activeNodeId: null,
        activeSceneId: null
      })
    ).toBeNull()
  })

  it("builds a query key with project id, target context, and filters", async () => {
    const filters = { status: "open" as const, category: "clarity" as const }
    const { queryClient } = renderAnnotationsHook({
      projectId: "project-1",
      targetContext: { targetType: "scene", targetId: "scene-1" },
      filters
    })

    await waitFor(() => {
      expect(listManuscriptAnnotations).toHaveBeenCalledWith("project-1", {
        target_type: "scene",
        target_id: "scene-1",
        status: "open",
        category: "clarity",
        limit: 50,
        offset: 0
      })
    })

    expect(
      queryClient.getQueryCache().find({
        queryKey: buildWritingAnnotationsQueryKey({
          projectId: "project-1",
          targetContext: { targetType: "scene", targetId: "scene-1" },
          filters
        })
      })
    ).toBeDefined()
  })

  it("keeps annotation queries without data and does not list when disabled", () => {
    const { result } = renderAnnotationsHook({
      projectId: "project-1",
      targetContext: { targetType: "project", targetId: "project-1" },
      enabled: false
    })

    expect(listManuscriptAnnotations).not.toHaveBeenCalled()
    expect(result.current.annotations).toEqual([])
  })

  it("invalidates the exact active annotation query key after mutations", async () => {
    const queryClient = createQueryClient()
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries")
    const props = {
      projectId: "project-1",
      targetContext: { targetType: "chapter" as const, targetId: "chapter-1" },
      filters: { status: "open" as const }
    }
    const { result } = renderAnnotationsHook(props, queryClient)
    const expectedKey = buildWritingAnnotationsQueryKey(props)

    await act(async () => {
      await result.current.createAnnotation({
        target_type: "chapter",
        target_id: "chapter-1",
        category: "other",
        body: "Chapter note"
      })
      await result.current.updateAnnotation("annotation-1", { status: "resolved" }, 1)
      await result.current.deleteAnnotation("annotation-1", 2)
    })

    expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: expectedKey })
    expect(invalidateSpy).toHaveBeenCalledTimes(3)
  })
})
