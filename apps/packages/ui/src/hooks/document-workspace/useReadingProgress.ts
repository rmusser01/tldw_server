import { useEffect, useRef, useCallback } from "react"
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query"
import { tldwClient } from "@/services/tldw"
import type { QuickIngestOperation } from "@/services/tldw/quick-ingest-authority"
import { createServicePromptScopeChangedError } from "@/services/tldw/service-prompt-scope-error"
import { useConnectionStore } from "@/store/connection"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"
import type { ViewMode } from "@/components/DocumentWorkspace/types"
import {
  isNotFoundError,
  shouldRetryDocumentWorkspaceQuery
} from "./request-retry"
import { useDocumentOperation } from "./useDocumentOperation"

export interface ReadingProgress {
  media_id: number
  current_page: number
  total_pages: number
  zoom_level: number
  view_mode: ViewMode
  percent_complete: number
  cfi?: string
  last_read_at: string
}
export interface ReadingProgressNotFound {
  media_id: number
  has_progress: false
}
type Progress = {
  current_page: number
  total_pages: number
  zoom_level?: number
  view_mode?: ViewMode
  cfi?: string
  percentage?: number
}
const progressKey = (
  owner: QuickIngestOperation | null,
  mediaId: number | null
) =>
  [
    "reading-progress",
    owner?.authorityKey,
    owner?.authorityRevision,
    mediaId
  ] as const

/** A cold read belongs to its captured account and still-active document. */
export function useReadingProgress(mediaId: number | null) {
  const owner = useDocumentOperation()
  const available = useConnectionStore(
    (s) => s.state.isConnected && s.state.mode !== "demo"
  )
  const applied = useRef(new Set<string>())
  const query = useQuery({
    queryKey: progressKey(owner, mediaId),
    queryFn: async ({ signal }) => {
      if (!owner || mediaId === null) return null
      owner.assertCurrent()
      const response = await tldwClient.getReadingProgress(mediaId, {
        requestScope: owner.requestScope,
        signal: AbortSignal.any([owner.signal, signal])
      })
      owner.assertCurrent()
      if (response.media_id !== mediaId)
        throw new Error("Reading position belongs to another document")
      return response
    },
    enabled: Boolean(owner) && mediaId !== null && available,
    staleTime: 60000,
    retry: (count, error) =>
      Boolean(owner?.isCurrent()) &&
      shouldRetryDocumentWorkspaceQuery(count, error, 1),
    refetchOnWindowFocus: false
  })
  useEffect(() => {
    const state = useDocumentWorkspaceStore.getState()
    if (
      !owner?.isCurrent() ||
      mediaId === null ||
      state.activeDocumentId !== mediaId
    )
      return
    if (query.isError) {
      state.setProgressHealth("error")
      return
    }
    const response = query.data
    if (!response) return
    state.setProgressHealth("ok")
    const key = JSON.stringify(progressKey(owner, mediaId))
    if (response.has_progress === false || applied.current.has(key)) return
    const local = state.openDocuments.find(
      (doc) => doc.id === mediaId
    )?.viewerState
    if (
      local &&
      (local.totalPages > 0 ||
        local.currentPage !== state.currentPage ||
        local.zoomLevel !== state.zoomLevel ||
        local.viewMode !== state.viewMode ||
        local.currentCfi !== state.currentCfi)
    ) {
      applied.current.add(key)
      return
    }
    if (response.current_page && response.zoom_level && response.view_mode) {
      state.setCurrentPage(response.current_page)
      state.setZoomLevel(response.zoom_level)
      state.setViewMode(response.view_mode)
      if (response.cfi) {
        state.setCurrentCfi(response.cfi)
        state.setCurrentPercentage(response.percent_complete ?? 0)
      }
      applied.current.add(key)
    }
  }, [owner, mediaId, query.data, query.isError])
  return query
}

/** Bind writes before the first asynchronous boundary. */
export function useUpdateReadingProgress() {
  const owner = useDocumentOperation()
  const queryClient = useQueryClient()
  type Variables = { mediaId: number; progress: Progress }
  const mutation = useMutation({
    mutationFn: async (
      variables: Variables & { owner: QuickIngestOperation }
    ) => {
      variables.owner.assertCurrent()
      const saved = await tldwClient.updateReadingProgress(
        variables.mediaId,
        variables.progress,
        variables.owner
      )
      variables.owner.assertCurrent()
      return saved
    },
    onSuccess: (_saved, variables) => {
      if (variables.owner.isCurrent())
        void queryClient.invalidateQueries({
          queryKey: progressKey(variables.owner, variables.mediaId)
        })
    }
  })
  const send = mutation.mutateAsync
  const mutateAsync = useCallback(
    (variables: Variables) => {
      if (!owner) return Promise.reject(createServicePromptScopeChangedError())
      return send({ ...variables, owner })
    },
    [owner, send]
  )
  return {
    ...mutation,
    mutate: (variables: Variables, options?: Parameters<typeof mutation.mutate>[1]) => {
      if (!owner) throw createServicePromptScopeChangedError()
      owner.assertCurrent()
      mutation.mutate({ ...variables, owner }, options)
    },
    mutateAsync
  }
}

/** Debounce exact document snapshots; a completed old save cannot ACK a new one. */
export function useReadingProgressAutoSave(
  mediaId: number | null,
  debounceMs = 5000,
  readReady = true
) {
  const owner = useDocumentOperation()
  const available = useConnectionStore(
    (s) => s.state.isConnected && s.state.mode !== "demo"
  )
  const currentPage = useDocumentWorkspaceStore((s) => s.currentPage)
  const totalPages = useDocumentWorkspaceStore((s) => s.totalPages)
  const zoomLevel = useDocumentWorkspaceStore((s) => s.zoomLevel)
  const viewMode = useDocumentWorkspaceStore((s) => s.viewMode)
  const cfi = useDocumentWorkspaceStore((s) => s.currentCfi)
  const percentage = useDocumentWorkspaceStore((s) => s.currentPercentage)
  const { mutateAsync, isPending, error } = useUpdateReadingProgress()
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const saved = useRef(new Map<string, string>())
  const missing = useRef(new Set<string>())
  const inflight = useRef<Promise<unknown> | null>(null)
  const forceSave = useCallback(async () => {
    if (timer.current) clearTimeout(timer.current)
    if (
      !readReady ||
      !owner?.isCurrent() ||
      !available ||
      !mediaId ||
      totalPages === 0
    )
      return
    while (inflight.current) {
      try {
        await inflight.current
      } catch {
        /* Failed writes remain eligible for explicit recovery. */
      }
      if (!owner.isCurrent()) return
    }
    const key = JSON.stringify(progressKey(owner, mediaId))
    const progress: Progress = {
      current_page: currentPage,
      total_pages: totalPages,
      zoom_level: zoomLevel,
      view_mode: viewMode,
      ...(cfi ? { cfi, percentage } : {})
    }
    const snapshot = JSON.stringify(progress)
    if (saved.current.get(key) === snapshot || missing.current.has(key)) return
    const task = mutateAsync({ mediaId, progress })
    inflight.current = task
    try {
      await task
      if (owner.isCurrent()) saved.current.set(key, snapshot)
    } catch (error) {
      if (!owner.isCurrent()) return
      if (isNotFoundError(error)) missing.current.add(key)
      else useDocumentWorkspaceStore.getState().setProgressHealth("error")
    } finally {
      if (inflight.current === task) inflight.current = null
    }
  }, [
    readReady,
    owner,
    available,
    mediaId,
    currentPage,
    totalPages,
    zoomLevel,
    viewMode,
    cfi,
    percentage,
    mutateAsync
  ])
  useEffect(() => {
    timer.current = setTimeout(() => {
      void forceSave()
    }, debounceMs)
    return () => {
      if (timer.current) clearTimeout(timer.current)
    }
  }, [forceSave, debounceMs])
  return { isSaving: isPending, forceSave, error }
}

/** Flush the previous document callback; unload delivery remains best effort. */
export function useReadingProgressSaveOnClose(
  mediaId: number | null,
  forceSave: () => void | Promise<void>
) {
  const current = useRef({ mediaId, forceSave })
  useEffect(() => {
    const previous = current.current
    current.current = { mediaId, forceSave }
    if (previous.mediaId !== mediaId)
      void Promise.resolve()
        .then(previous.forceSave)
        .catch(() => {})
  }, [mediaId, forceSave])
  useEffect(() => {
    const finish = () => {
      void Promise.resolve()
        .then(() => current.current.forceSave())
        .catch(() => {})
    }
    window.addEventListener("beforeunload", finish)
    return () => {
      window.removeEventListener("beforeunload", finish)
      finish()
    }
  }, [])
}
