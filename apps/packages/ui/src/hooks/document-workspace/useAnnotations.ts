import { useEffect } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { tldwClient } from "@/services/tldw"
import type { QuickIngestOperation } from "@/services/tldw/quick-ingest-authority"
import { createServicePromptScopeChangedError } from "@/services/tldw/service-prompt-scope-error"
import { useConnectionStore } from "@/store/connection"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"
import type {
  Annotation,
  AnnotationColor,
  AnnotationType
} from "@/components/DocumentWorkspace/types"
import { shouldRetryDocumentWorkspaceQuery } from "./request-retry"
import { useDocumentOperation } from "./useDocumentOperation"

export interface AnnotationResponse {
  id: string
  media_id: number
  location: string
  text: string
  color: AnnotationColor
  note?: string
  annotation_type: AnnotationType
  chapter_title?: string
  percentage?: number
  created_at: string
  updated_at: string
}
export interface AnnotationsListResponse {
  media_id: number
  annotations: AnnotationResponse[]
  total_count: number
}

export function toAnnotation(response: AnnotationResponse): Annotation {
  return {
    id: response.id,
    documentId: response.media_id,
    location: isNaN(Number(response.location))
      ? response.location
      : Number(response.location),
    text: response.text,
    color: response.color,
    note: response.note,
    annotationType: response.annotation_type,
    chapterTitle: response.chapter_title,
    percentage: response.percentage,
    createdAt: new Date(response.created_at),
    updatedAt: new Date(response.updated_at)
  }
}
export const annotationQueryKey = (
  operation: QuickIngestOperation | null,
  mediaId: number | null
) =>
  [
    "document-annotations",
    operation?.authorityKey,
    operation?.authorityRevision,
    mediaId
  ] as const

/** Publish a read only for the still-active document and verified account. */
export function useAnnotations(mediaId: number | null) {
  const operation = useDocumentOperation()
  const available = useConnectionStore(
    (s) => s.state.isConnected && s.state.mode !== "demo"
  )
  const query = useQuery<AnnotationsListResponse | null>({
    queryKey: annotationQueryKey(operation, mediaId),
    queryFn: async ({ signal }) => {
      if (mediaId === null || !operation) return null
      operation.assertCurrent()
      const response = await tldwClient.listAnnotations(mediaId, {
        requestScope: operation.requestScope,
        signal: AbortSignal.any([operation.signal, signal])
      })
      operation.assertCurrent()
      return response
    },
    enabled: mediaId !== null && available && Boolean(operation),
    staleTime: 5 * 60 * 1000,
    retry: (count, error) => shouldRetryDocumentWorkspaceQuery(count, error, 2),
    refetchOnWindowFocus: false
  })
  useEffect(() => {
    if (
      !operation?.isCurrent() ||
      mediaId === null ||
      useDocumentWorkspaceStore.getState().activeDocumentId !== mediaId
    )
      return
    const store = useDocumentWorkspaceStore.getState()
    if (query.data) {
      store.setAnnotations(query.data.annotations.map(toAnnotation), mediaId)
      store.setAnnotationsHealth("ok")
    } else if (query.isError) {
      store.setAnnotationsHealth("error")
    }
  }, [query.data, query.isError, mediaId, operation])
  return query
}

/** Bind mutation variables before React Query can await any lifecycle callback. */
function useOwnedAnnotationMutation<V extends { mediaId: number }, R>(
  send: (variables: V, operation: QuickIngestOperation) => Promise<R>,
  complete: (response: R, variables: V) => void
) {
  const operation = useDocumentOperation()
  const queryClient = useQueryClient()
  type Owned = V & { operation: QuickIngestOperation }
  const mutation = useMutation<R, Error, Owned>({
    mutationFn: async (variables) => {
      const owner = variables.operation
      owner.assertCurrent()
      await queryClient.cancelQueries({
        queryKey: annotationQueryKey(owner, variables.mediaId)
      })
      owner.assertCurrent()
      const response = await send(variables, owner)
      owner.assertCurrent()
      return response
    },
    onSuccess: (response, variables) => {
      if (!variables.operation.isCurrent()) return
      complete(response, variables)
      void queryClient.invalidateQueries({
        queryKey: annotationQueryKey(variables.operation, variables.mediaId)
      })
    },
    onError: (_error, variables) => {
      if (variables.operation.isCurrent())
        useDocumentWorkspaceStore.getState().setAnnotationSyncStatus("error")
    }
  })
  const bind = (variables: V): Owned => {
    if (!operation) throw createServicePromptScopeChangedError()
    operation.assertCurrent()
    return { ...variables, operation }
  }
  return {
    ...mutation,
    mutate: (variables: V, options?: Parameters<typeof mutation.mutate>[1]) =>
      mutation.mutate(bind(variables), options),
    mutateAsync: (
      variables: V,
      options?: Parameters<typeof mutation.mutateAsync>[1]
    ) => mutation.mutateAsync(bind(variables), options)
  }
}

export function useCreateAnnotation() {
  return useOwnedAnnotationMutation(
    (
      v: {
        mediaId: number
        annotation: {
          location: string
          text: string
          color?: AnnotationColor
          note?: string
          annotation_type?: AnnotationType
        }
      },
      owner
    ) => tldwClient.createAnnotation(v.mediaId, v.annotation, owner),
    (response) =>
      useDocumentWorkspaceStore
        .getState()
        .acknowledgeAnnotations([], [toAnnotation(response)])
  )
}
export function useUpdateAnnotation() {
  return useOwnedAnnotationMutation(
    (
      v: {
        mediaId: number
        annotationId: string
        updates: { text?: string; color?: AnnotationColor; note?: string }
      },
      owner
    ) =>
      tldwClient.updateAnnotation(v.mediaId, v.annotationId, v.updates, owner),
    (response) =>
      useDocumentWorkspaceStore
        .getState()
        .acknowledgeAnnotations([], [toAnnotation(response)])
  )
}
export function useDeleteAnnotation() {
  return useOwnedAnnotationMutation(
    async (v: { mediaId: number; annotationId: string }, owner) => {
      await tldwClient.deleteAnnotation(v.mediaId, v.annotationId, owner)
      return v.annotationId
    },
    (id) => useDocumentWorkspaceStore.getState().removeAnnotation(id)
  )
}
