import { useEffect, useRef, useCallback } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { tldwClient } from "@/services/tldw"
import type { QuickIngestOperation } from "@/services/tldw/quick-ingest-authority"
import { useConnectionStore } from "@/store/connection"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"
import type { Annotation } from "@/components/DocumentWorkspace/types"
import { annotationQueryKey, toAnnotation } from "./useAnnotations"
import { useDocumentOperation } from "./useDocumentOperation"
import { createServicePromptScopeChangedError } from "@/services/tldw/service-prompt-scope-error"
import { getErrorStatus } from "./request-retry"

import { resolveBrowserRequestTransport, resolveTldwRequestHeaders } from "@/services/tldw/request-core"
import type { AllowedPath } from "@/services/tldw/openapi-guard"
import { requestScopeFields } from "@/services/tldw/domains/service-prompts"
import { buildMediaNavigationScopeKey } from "@/utils/media-navigation-scope"

const MAX_RETRY_ATTEMPTS = 3

const MAX_UNLOAD_PAYLOAD_BYTES = 60_000

const syncPath = (mediaId: number): AllowedPath => `/api/v1/media/${mediaId}/annotations/sync`

const pendingStorageKey = (scopeKey: string, mediaId: number) =>
  `tldw:document-workspace:pending-annotations:${scopeKey}:${mediaId}`

const toSyncPayload = (annotations: Annotation[]) => ({
  annotations: annotations.map((ann) => ({
    location: String(ann.location),
    text: ann.text,
    color: ann.color,
    note: ann.note,
    annotation_type: ann.annotationType ?? "highlight",
    chapter_title: ann.chapterTitle,
    percentage: ann.percentage
  })),
  client_ids: annotations.map((ann) => ann.id)
})

function readPersistedAnnotations(key: string, mediaId: number): Annotation[] {
  const stored = localStorage.getItem(key)
  if (stored === null) return []
  const parsed: unknown = JSON.parse(stored)
  if (
    !Array.isArray(parsed) ||
    parsed.some(annotation => !annotation || annotation.documentId !== mediaId ||
      typeof annotation.id !== "string" || !annotation.id.trim())
  ) throw new Error("Persisted annotation document identity is invalid")
  return parsed
}

/** Merge pending annotations into the persisted set (replay is idempotent on client id). */
function persistPendingAnnotations(key: string, pending: Annotation[], mediaId: number): Annotation[] {
  const pendingIds = new Set(pending.map((ann) => ann.id))
  const merged = [
    ...readPersistedAnnotations(key, mediaId).filter((ann) => !pendingIds.has(ann.id)),
    ...pending
  ]
  try {
    localStorage.setItem(key, JSON.stringify(merged))
  } catch (error) {
    console.error("Persisting pending annotations failed:", error)
  }
  return merged
}

function acknowledgeSavedAnnotations(
  response: Awaited<ReturnType<typeof tldwClient.syncAnnotations>>,
  submitted: Annotation[],
  mediaId: number
) {
  if (
    response.media_id !== mediaId ||
    response.synced_count !== submitted.length ||
    response.annotations.length !== submitted.length ||
    new Set(response.annotations.map((annotation) => annotation.id))
      .size !== submitted.length ||
    new Set(
      submitted.map((annotation) => response.id_mapping?.[annotation.id])
    ).size !== submitted.length ||
    submitted.some((annotation) => {
      const id = response.id_mapping?.[annotation.id]
      return (
        !id ||
        !response.annotations.some(
          (saved) => saved.id === id && saved.media_id === mediaId
        )
      )
    })
  )
    throw new Error("Annotation save acknowledgement was incomplete")
  useDocumentWorkspaceStore
    .getState()
    .acknowledgeAnnotations(
      submitted,
      response.annotations.map(toAnnotation),
      response.id_mapping
    )
}

type Sync = {
  mediaId: number
  submitted: Annotation[]
  operation: QuickIngestOperation
}

/** Serialize captured creation batches; ACKs retire only their exact snapshots. */
export function useAnnotationSync(mediaId: number | null, debounceMs = 2000) {
  const operation = useDocumentOperation()
  const queryClient = useQueryClient()
  const available = useConnectionStore(
    (s) => s.state.isConnected && s.state.mode !== "demo"
  )
  const pending = useDocumentWorkspaceStore((s) => s.pendingAnnotations)
  const syncStatus = useDocumentWorkspaceStore((s) => s.annotationSyncStatus)
  const inflight = useRef<{
    promise: Promise<unknown>
    operation: QuickIngestOperation
  } | null>(null)
  const mutation = useMutation({
    mutationFn: async ({ mediaId, submitted, operation: owner }: Sync) => {
      owner.assertCurrent()
      await queryClient.cancelQueries({
        queryKey: annotationQueryKey(owner, mediaId)
      })
      owner.assertCurrent()
      const response = await tldwClient.syncAnnotations(
        mediaId,
        submitted.map((annotation) => ({
          location: String(annotation.location),
          text: annotation.text,
          color: annotation.color,
          note: annotation.note,
          annotation_type: annotation.annotationType ?? "highlight",
          chapter_title: annotation.chapterTitle,
          percentage: annotation.percentage
        })),
        submitted.map((annotation) => annotation.id),
        owner
      )
      owner.assertCurrent()
      return response
    },
    retry: (count, error) => {
      const status = getErrorStatus(error)
      return (
        Boolean(inflight.current?.operation.isCurrent()) &&
        !(status && status >= 400 && status < 500) &&
        count < MAX_RETRY_ATTEMPTS - 1
      )
    },
    retryDelay: (count) => Math.min(1000 * 2 ** count, 8000),
    onSuccess: (response, { submitted, mediaId, operation: owner }) => {
      if (!owner.isCurrent()) return
      acknowledgeSavedAnnotations(response, submitted, mediaId)
      void queryClient.invalidateQueries({
        queryKey: annotationQueryKey(owner, mediaId)
      })
    },
    onError: (_error, { operation: owner }) => {
      if (owner.isCurrent())
        useDocumentWorkspaceStore.getState().setAnnotationSyncStatus("error")
    }
  })
  const mutateAsync = mutation.mutateAsync
  const forceSync = useCallback(
    async (targetMediaId = mediaId) => {
      if (targetMediaId === null) return
      if (!available || !operation?.isCurrent())
        throw createServicePromptScopeChangedError()
      while (inflight.current) {
        try {
          await inflight.current.promise
        } catch {
          /* The failed document keeps its queue. */
        }
        operation.assertCurrent()
      }
      const submitted = useDocumentWorkspaceStore
        .getState()
        .pendingAnnotations.filter(
          (annotation) => annotation.documentId === targetMediaId
        )
      if (!submitted.length) return
      useDocumentWorkspaceStore.getState().setAnnotationSyncStatus("pending")
      const task = mutateAsync({ mediaId: targetMediaId, submitted, operation })
      inflight.current = { promise: task, operation }
      try {
        await task
      } finally {
        if (inflight.current?.promise === task) inflight.current = null
      }
    },
    [mediaId, available, operation, mutateAsync]
  )
  useEffect(() => {
    if (
      syncStatus !== "pending" ||
      !pending.some((annotation) => annotation.documentId === mediaId) ||
      !available ||
      mutation.isPending
    )
      return
    const timer = setTimeout(() => {
      void forceSync().catch(() => {})
    }, debounceMs)
    return () => clearTimeout(timer)
  }, [
    pending,
    syncStatus,
    mediaId,
    available,
    mutation.isPending,
    debounceMs,
    forceSync
  ])
  const retrySync = useCallback(() => {
    void forceSync().catch(() => {})
  }, [forceSync])
  return {
    isSyncing: mutation.isPending,
    syncStatus,
    forceSync,
    retrySync,
    retryCount: mutation.failureCount,
    maxRetries: MAX_RETRY_ATTEMPTS,
    error: mutation.error
  }
}

/** Persist before unload; a queued request never substitutes for an exact ACK. */
export function useAnnotationSyncOnClose(
  mediaId: number | null,
  forceSync: () => void | Promise<void>
) {
  const operation = useDocumentOperation()
  const queryClient = useQueryClient()
  const prepared = useRef<{
    operation: QuickIngestOperation
    headers: Record<string, string> | null
  } | null>(null)
  const current = useRef({ mediaId, forceSync, operation })
  useEffect(() => {
    const previous = current.current
    current.current = { mediaId, forceSync, operation }
    if (previous.mediaId !== mediaId)
      void Promise.resolve().then(previous.forceSync).catch(() => {})
  }, [mediaId, forceSync, operation])
  useEffect(() => {
    if (mediaId === null || !operation?.isCurrent()) return
    let cancelled = false
    const isCurrent = () => !cancelled && operation.isCurrent()
    const scopeKey = buildMediaNavigationScopeKey({
      ...operation.requestScope.config, userId: operation.requestScope.userId
    })
    const key = pendingStorageKey(scopeKey, mediaId)
    void (async () => {
      const headers = await resolveTldwRequestHeaders(
        {
          path: syncPath(mediaId), method: "POST", body: {},
          headers: requestScopeFields(operation.requestScope).headers
        },
        { getConfig: async () => operation.requestScope.config, useRuntimeAuthOverride: false }
      )
      if (!isCurrent()) return
      prepared.current = { operation, headers }
      const stored = localStorage.getItem(key)
      const persisted = readPersistedAnnotations(key, mediaId)
      if (!persisted.length) return
      const { annotations, client_ids } = toSyncPayload(persisted)
      const response = await tldwClient.syncAnnotations(mediaId, annotations, client_ids, operation)
      if (!isCurrent()) return
      acknowledgeSavedAnnotations(response, persisted, mediaId)
      void queryClient.invalidateQueries({ queryKey: annotationQueryKey(operation, mediaId) })
      if (localStorage.getItem(key) === stored) localStorage.removeItem(key)
    })().catch(() => {
      if (isCurrent()) console.warn("Annotation unload preparation or replay failed; pending writes retained")
    })
    return () => {
      cancelled = true
      if (prepared.current?.operation === operation) prepared.current = null
    }
  }, [mediaId, operation, queryClient])
  useEffect(() => {
    const beforeUnload = (event: BeforeUnloadEvent) => {
      const pending = useDocumentWorkspaceStore.getState().pendingAnnotations
      if (!pending.length) return
      event.preventDefault()
      event.returnValue = ""
      const context = prepared.current
      const owner = current.current.operation
      if (!context || context.operation !== owner || !owner?.isCurrent()) return
      const scopeKey = buildMediaNavigationScopeKey({
        ...owner.requestScope.config, userId: owner.requestScope.userId
      })
      let deferred = false
      for (const documentId of new Set(pending.map(annotation => annotation.documentId))) {
        if (!owner.isCurrent()) break
        try {
          const toSend = persistPendingAnnotations(
            pendingStorageKey(scopeKey, documentId),
            pending.filter(annotation => annotation.documentId === documentId),
            documentId
          )
          const body = JSON.stringify(toSyncPayload(toSend))
          if (!context.headers || new Blob([body]).size > MAX_UNLOAD_PAYLOAD_BYTES) {
            deferred = true
            continue
          }
          const url = resolveBrowserRequestTransport({
            config: owner.requestScope.config, path: syncPath(documentId)
          }).url
          void fetch(url, {
            method: "POST", keepalive: true, headers: context.headers, body
          }).catch(() => console.warn("Annotation keepalive failed; pending writes retained"))
        } catch {
          deferred = true
          console.warn("Annotation unload save failed; pending writes retained")
        }
      }
      if (deferred) void Promise.resolve(current.current.forceSync()).catch(() => {})
    }
    window.addEventListener("beforeunload", beforeUnload)
    return () => {
      window.removeEventListener("beforeunload", beforeUnload)
      void Promise.resolve().then(() => current.current.forceSync()).catch(() => {})
    }
  }, [])
}
