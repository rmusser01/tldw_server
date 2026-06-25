import { useCallback, useMemo } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  createManuscriptAnnotation,
  deleteManuscriptAnnotation,
  listManuscriptAnnotations,
  reviewManuscriptScene,
  reviewManuscriptSelection,
  updateManuscriptAnnotation,
  type ManuscriptAnnotationCreateInput,
  type ManuscriptAnnotationListFilters,
  type ManuscriptAnnotationResponse,
  type ManuscriptAnnotationUpdateInput,
  type ManuscriptSceneAnnotationReviewJobResponse,
  type ManuscriptSceneAnnotationReviewRequest,
  type ManuscriptSelectedTextAnnotationReviewRequest
} from "@/services/writing-playground"
import type { WritingAnnotationTargetContext } from "../writing-annotation-types"

const DEFAULT_LIMIT = 50

type UseWritingAnnotationsProps = {
  projectId?: string | null
  targetContext?: WritingAnnotationTargetContext | null
  filters?: ManuscriptAnnotationListFilters
  enabled?: boolean
}

export type BuildWritingAnnotationsQueryKeyInput = {
  projectId?: string | null
  targetContext?: WritingAnnotationTargetContext | null
  filters?: ManuscriptAnnotationListFilters
}

const normalizeFilters = ({
  targetContext,
  filters
}: {
  targetContext?: WritingAnnotationTargetContext | null
  filters?: ManuscriptAnnotationListFilters
}): ManuscriptAnnotationListFilters => ({
  target_type: targetContext?.targetType,
  target_id: targetContext?.targetId,
  status: "open",
  limit: DEFAULT_LIMIT,
  offset: 0,
  ...filters
})

export const buildWritingAnnotationsQueryKey = ({
  projectId,
  targetContext,
  filters
}: BuildWritingAnnotationsQueryKeyInput) => [
  "writing-manuscript-annotations",
  projectId ?? null,
  targetContext
    ? {
        targetType: targetContext.targetType,
        targetId: targetContext.targetId
      }
    : null,
  normalizeFilters({ targetContext, filters })
]

export type UseWritingAnnotationsResult = {
  annotations: ManuscriptAnnotationResponse[]
  isLoading: boolean
  isFetching: boolean
  error: Error | null
  createAnnotation: (
    input: ManuscriptAnnotationCreateInput
  ) => Promise<ManuscriptAnnotationResponse>
  updateAnnotation: (
    annotationId: string,
    input: ManuscriptAnnotationUpdateInput,
    version: number
  ) => Promise<ManuscriptAnnotationResponse>
  deleteAnnotation: (annotationId: string, version: number) => Promise<void>
  reviewSelection: (
    sceneId: string,
    input: ManuscriptSelectedTextAnnotationReviewRequest
  ) => Promise<ManuscriptAnnotationResponse>
  reviewScene: (
    sceneId: string,
    input: ManuscriptSceneAnnotationReviewRequest
  ) => Promise<ManuscriptSceneAnnotationReviewJobResponse>
  isCreating: boolean
  isUpdating: boolean
  isDeleting: boolean
  isReviewingSelection: boolean
  isReviewingScene: boolean
}

export function useWritingAnnotations({
  projectId,
  targetContext,
  filters,
  enabled = true
}: UseWritingAnnotationsProps): UseWritingAnnotationsResult {
  const queryClient = useQueryClient()
  const queryFilters = useMemo(
    () => normalizeFilters({ targetContext, filters }),
    [filters, targetContext]
  )
  const queryKey = useMemo(
    () => buildWritingAnnotationsQueryKey({ projectId, targetContext, filters }),
    [filters, projectId, targetContext]
  )
  const queryEnabled =
    enabled &&
    Boolean(projectId) &&
    Boolean(targetContext?.targetType) &&
    Boolean(targetContext?.targetId)

  const annotationsQuery = useQuery({
    queryKey,
    queryFn: () => listManuscriptAnnotations(projectId!, queryFilters),
    enabled: queryEnabled
  })

  const invalidateActiveAnnotations = useCallback(async () => {
    await queryClient.invalidateQueries({ queryKey })
  }, [queryClient, queryKey])

  const createMutation = useMutation({
    mutationFn: (input: ManuscriptAnnotationCreateInput) =>
      createManuscriptAnnotation(input),
    onSuccess: invalidateActiveAnnotations
  })

  const updateMutation = useMutation({
    mutationFn: ({
      annotationId,
      input,
      version
    }: {
      annotationId: string
      input: ManuscriptAnnotationUpdateInput
      version: number
    }) => updateManuscriptAnnotation(annotationId, input, version),
    onSuccess: invalidateActiveAnnotations
  })

  const deleteMutation = useMutation({
    mutationFn: ({
      annotationId,
      version
    }: {
      annotationId: string
      version: number
    }) => deleteManuscriptAnnotation(annotationId, version),
    onSuccess: invalidateActiveAnnotations
  })

  const reviewSelectionMutation = useMutation({
    mutationFn: ({
      sceneId,
      input
    }: {
      sceneId: string
      input: ManuscriptSelectedTextAnnotationReviewRequest
    }) => reviewManuscriptSelection(sceneId, input),
    onSuccess: invalidateActiveAnnotations
  })

  const reviewSceneMutation = useMutation({
    mutationFn: ({
      sceneId,
      input
    }: {
      sceneId: string
      input: ManuscriptSceneAnnotationReviewRequest
    }) => reviewManuscriptScene(sceneId, input),
    onSuccess: invalidateActiveAnnotations
  })

  return {
    annotations: annotationsQuery.data?.annotations ?? [],
    isLoading: annotationsQuery.isLoading,
    isFetching: annotationsQuery.isFetching,
    error: annotationsQuery.error as Error | null,
    createAnnotation: createMutation.mutateAsync,
    updateAnnotation: (annotationId, input, version) =>
      updateMutation.mutateAsync({ annotationId, input, version }),
    deleteAnnotation: (annotationId, version) =>
      deleteMutation.mutateAsync({ annotationId, version }),
    reviewSelection: (sceneId, input) =>
      reviewSelectionMutation.mutateAsync({ sceneId, input }),
    reviewScene: (sceneId, input) =>
      reviewSceneMutation.mutateAsync({ sceneId, input }),
    isCreating: createMutation.isPending,
    isUpdating: updateMutation.isPending,
    isDeleting: deleteMutation.isPending,
    isReviewingSelection: reviewSelectionMutation.isPending,
    isReviewingScene: reviewSceneMutation.isPending
  }
}
