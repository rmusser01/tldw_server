import { useEffect, useRef, useCallback, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { tldwClient } from "@/services/tldw"
import {
  resolveBrowserRequestTransport,
  resolveTldwRequestHeaders
} from "@/services/tldw/request-core"
import { useConnectionStore } from "@/store/connection"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"
import { buildMediaNavigationScopeKey } from "@/utils/media-navigation-scope"
import type { Annotation, AnnotationColor } from "@/components/DocumentWorkspace/types"

// Retry configuration
const MAX_RETRY_ATTEMPTS = 3
const INITIAL_RETRY_DELAY_MS = 1000 // 1 second
const MAX_RETRY_DELAY_MS = 8000 // 8 seconds

// The Fetch spec caps in-flight keepalive bodies at 64 KiB, so larger unload
// payloads are only persisted and replayed on the next load.
const MAX_UNLOAD_PAYLOAD_BYTES = 60_000

const syncPath = (mediaId: number) => `/api/v1/media/${mediaId}/annotations/sync`

const pendingStorageKey = (scopeKey: string, mediaId: number) =>
  `tldw:document-workspace:pending-annotations:${scopeKey}:${mediaId}`

const toSyncPayload = (annotations: Annotation[]) => ({
  annotations: annotations.map((ann) => ({
    location: String(ann.location),
    text: ann.text,
    color: ann.color,
    note: ann.note,
    annotation_type: ann.annotationType ?? "highlight"
  })),
  client_ids: annotations.map((ann) => ann.id)
})

/**
 * Prepared ahead of time by useAnnotationSyncOnClose, because beforeunload
 * cannot await the async config that the auth headers and the per-user
 * storage scope come from. Headers are null when credentials are unavailable.
 */
const unloadContextRef: {
  current: { scopeKey: string; headers: Record<string, string> | null } | null
} = { current: null }

function readPersistedAnnotations(key: string): Annotation[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(key) || "[]")
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

/** Merge pending annotations into the persisted set (replay is idempotent on client id). */
function persistPendingAnnotations(key: string, pending: Annotation[]): Annotation[] {
  const pendingIds = new Set(pending.map((ann) => ann.id))
  const merged = [
    ...readPersistedAnnotations(key).filter((ann) => !pendingIds.has(ann.id)),
    ...pending
  ]
  try {
    localStorage.setItem(key, JSON.stringify(merged))
  } catch (error) {
    console.error("Persisting pending annotations failed:", error)
  }
  return merged
}

async function replayPersistedAnnotations(key: string, mediaId: number): Promise<boolean> {
  const stored = localStorage.getItem(key)
  const persisted = readPersistedAnnotations(key)
  if (persisted.length === 0) return false
  const { annotations, client_ids } = toSyncPayload(persisted)
  await tldwClient.syncAnnotations(mediaId, annotations, client_ids)
  // An unload during the replay may have persisted more; keep those for next time.
  if (localStorage.getItem(key) === stored) localStorage.removeItem(key)
  return true
}

/**
 * Sync pending annotations on page unload. They are persisted locally first
 * (replayed on the next load), then sent with an authenticated keepalive fetch.
 *
 * @returns "sent" when the keepalive request was queued; "deferred" when no
 * request was made (payload over 60KB, credentials or context not ready).
 */
export function syncAnnotationsOnUnload(
  serverUrl: string | null,
  mediaId: number,
  pending: Annotation[]
): "sent" | "deferred" {
  const context = unloadContextRef.current
  if (!serverUrl || pending.length === 0 || !context) {
    return "deferred"
  }

  try {
    const toSend = persistPendingAnnotations(
      pendingStorageKey(context.scopeKey, mediaId),
      pending
    )
    const body = JSON.stringify(toSyncPayload(toSend))
    if (!context.headers || new Blob([body]).size > MAX_UNLOAD_PAYLOAD_BYTES) {
      return "deferred"
    }
    const url = resolveBrowserRequestTransport({
      config: { serverUrl },
      path: syncPath(mediaId)
    }).url
    fetch(url, {
      method: "POST",
      keepalive: true,
      headers: context.headers,
      body
    }).catch((err) => console.error("keepalive fetch for annotations failed:", err))
    return "sent"
  } catch (error) {
    console.error("syncAnnotationsOnUnload failed:", error)
    return "deferred"
  }
}

/**
 * Hook to automatically sync pending annotations to the backend.
 *
 * This hook watches the store's pendingAnnotations and syncs them
 * to the backend when connection is available. Includes automatic
 * retry with exponential backoff on failure.
 *
 * @param mediaId - The active document's media ID
 * @param debounceMs - Debounce time in milliseconds (default: 2000)
 */
export function useAnnotationSync(
  mediaId: number | null,
  debounceMs: number = 2000
) {
  const queryClient = useQueryClient()
  const isConnected = useConnectionStore((s) => s.state.isConnected)
  const mode = useConnectionStore((s) => s.state.mode)
  const isServerAvailable = isConnected && mode !== "demo"

  const pendingAnnotations = useDocumentWorkspaceStore(
    (s) => s.pendingAnnotations
  )
  const annotationSyncStatus = useDocumentWorkspaceStore(
    (s) => s.annotationSyncStatus
  )
  const setAnnotationSyncStatus = useDocumentWorkspaceStore(
    (s) => s.setAnnotationSyncStatus
  )

  const syncTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const retryTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const [retryCount, setRetryCount] = useState(0)

  // Clear retry timeout on cleanup
  useEffect(() => {
    return () => {
      if (retryTimeoutRef.current) {
        clearTimeout(retryTimeoutRef.current)
      }
    }
  }, [])

  // Reset retry count when sync succeeds or annotations change
  useEffect(() => {
    if (annotationSyncStatus === "synced") {
      setRetryCount(0)
    }
  }, [annotationSyncStatus])

  // Mutation for batch sync
  const syncMutation = useMutation({
    mutationFn: async ({
      mediaId,
      annotations,
      clientIds
    }: {
      mediaId: number
      annotations: Array<{
        location: string
        text: string
        color?: AnnotationColor
        note?: string
        annotation_type?: "highlight" | "page_note"
      }>
      clientIds: string[]
    }) => {
      return await tldwClient.syncAnnotations(
        mediaId,
        annotations,
        clientIds
      )
    },
    onSuccess: (_data, { mediaId }) => {
      queryClient.invalidateQueries({ queryKey: ["document-annotations", mediaId] })
      setAnnotationSyncStatus("synced")
      setRetryCount(0)
    },
    onError: () => {
      // Schedule retry with exponential backoff if under max attempts
      if (retryCount < MAX_RETRY_ATTEMPTS) {
        const delay = Math.min(
          INITIAL_RETRY_DELAY_MS * Math.pow(2, retryCount),
          MAX_RETRY_DELAY_MS
        )
        setRetryCount((prev) => prev + 1)
        setAnnotationSyncStatus("pending") // Keep as pending to trigger retry

        retryTimeoutRef.current = setTimeout(() => {
          // The debounced effect will pick this up
        }, delay)

        console.warn(
          `Annotation sync failed, will retry in ${delay}ms (attempt ${retryCount + 1}/${MAX_RETRY_ATTEMPTS})`
        )
      } else {
        // Max retries exceeded, set error status
        setAnnotationSyncStatus("error")
        console.error(
          `Annotation sync failed after ${MAX_RETRY_ATTEMPTS} attempts`
        )
      }
    }
  })

  // Sync function
  const syncPendingAnnotations = useCallback(async () => {
    if (
      !mediaId ||
      !isServerAvailable ||
      pendingAnnotations.length === 0 ||
      syncMutation.isPending
    ) {
      return
    }

    const annotationsToSync = pendingAnnotations.map((ann) => ({
      location: String(ann.location),
      text: ann.text,
      color: ann.color,
      note: ann.note,
      annotation_type: ann.annotationType ?? "highlight"
    }))

    const clientIds = pendingAnnotations.map((ann) => ann.id)

    try {
      await syncMutation.mutateAsync({
        mediaId,
        annotations: annotationsToSync,
        clientIds
      })
    } catch (error) {
      console.error("Failed to sync annotations:", error)
    }
  }, [mediaId, isServerAvailable, pendingAnnotations, syncMutation])

  // Debounced sync effect
  useEffect(() => {
    if (annotationSyncStatus !== "pending" || !isServerAvailable) {
      return
    }

    // Clear existing timeout
    if (syncTimeoutRef.current) {
      clearTimeout(syncTimeoutRef.current)
    }

    // Set new debounced sync
    syncTimeoutRef.current = setTimeout(() => {
      syncPendingAnnotations()
    }, debounceMs)

    return () => {
      if (syncTimeoutRef.current) {
        clearTimeout(syncTimeoutRef.current)
      }
    }
  }, [annotationSyncStatus, isServerAvailable, debounceMs, syncPendingAnnotations])

  // Force sync (useful for immediate save)
  const forceSync = useCallback(() => {
    if (syncTimeoutRef.current) {
      clearTimeout(syncTimeoutRef.current)
    }
    syncPendingAnnotations()
  }, [syncPendingAnnotations])

  // Manual retry function for user-initiated retries
  const retrySync = useCallback(() => {
    if (annotationSyncStatus === "error") {
      setRetryCount(0)
      setAnnotationSyncStatus("pending")
    }
  }, [annotationSyncStatus, setAnnotationSyncStatus])

  return {
    isSyncing: syncMutation.isPending,
    syncStatus: annotationSyncStatus,
    forceSync,
    retrySync,
    retryCount,
    maxRetries: MAX_RETRY_ATTEMPTS,
    error: syncMutation.error
  }
}

/**
 * Hook that syncs annotations when the document is closed or changed.
 * Call this from the document workspace page component and pass the
 * forceSync function returned by useAnnotationSync.
 *
 * On page unload, pending annotations are persisted locally and sent with an
 * authenticated keepalive fetch; persisted annotations are replayed on the
 * next load (the sync endpoint is idempotent on client id).
 */
export function useAnnotationSyncOnClose(
  mediaId: number | null,
  forceSync: () => void
) {
  const queryClient = useQueryClient()
  const annotationSyncStatus = useDocumentWorkspaceStore(
    (s) => s.annotationSyncStatus
  )
  const pendingAnnotations = useDocumentWorkspaceStore(
    (s) => s.pendingAnnotations
  )
  const serverUrl = useConnectionStore((s) => s.state.serverUrl)
  const configUpdatedAt = useConnectionStore((s) => s.state.lastConfigUpdatedAt)
  const replayedKeyRef = useRef<string | null>(null)
  const previousMediaIdRef = useRef<number | null>(mediaId)
  const forceSyncRef = useRef(forceSync)
  const statusRef = useRef(annotationSyncStatus)
  const pendingRef = useRef(pendingAnnotations)
  const serverUrlRef = useRef(serverUrl)
  const mediaIdRef = useRef(mediaId)

  useEffect(() => {
    forceSyncRef.current = forceSync
  }, [forceSync])

  useEffect(() => {
    statusRef.current = annotationSyncStatus
  }, [annotationSyncStatus])

  useEffect(() => {
    pendingRef.current = pendingAnnotations
  }, [pendingAnnotations])

  useEffect(() => {
    serverUrlRef.current = serverUrl
  }, [serverUrl])

  useEffect(() => {
    mediaIdRef.current = mediaId
  }, [mediaId])

  // Prepare unload headers and storage scope whenever the pending set or auth
  // config changes; on first preparation for a document, replay what a
  // previous page persisted.
  useEffect(() => {
    if (mediaId === null) return
    let cancelled = false
    void (async () => {
      const cfg = await tldwClient.getConfig()
      if (!cfg) return
      const headers = await resolveTldwRequestHeaders(
        { path: syncPath(mediaId), method: "POST", body: {} },
        { getConfig: async () => cfg }
      )
      if (cancelled) return
      const scopeKey = buildMediaNavigationScopeKey(cfg)
      unloadContextRef.current = { scopeKey, headers }
      const key = pendingStorageKey(scopeKey, mediaId)
      if (replayedKeyRef.current === key) return
      replayedKeyRef.current = key
      if (await replayPersistedAnnotations(key, mediaId)) {
        queryClient.invalidateQueries({ queryKey: ["document-annotations", mediaId] })
      }
    })().catch((error) => {
      console.warn("Annotation unload preparation or replay failed:", error)
    })
    return () => {
      cancelled = true
    }
  }, [mediaId, serverUrl, configUpdatedAt, pendingAnnotations, queryClient])

  // Sync when document changes (normal async sync is fine here)
  useEffect(() => {
    if (previousMediaIdRef.current !== mediaId && statusRef.current === "pending") {
      forceSyncRef.current()
    }
    previousMediaIdRef.current = mediaId
  }, [mediaId])

  // Sync on unmount / page close
  useEffect(() => {
    const handleBeforeUnload = () => {
      if (statusRef.current === "pending" && mediaIdRef.current !== null) {
        const result = syncAnnotationsOnUnload(
          serverUrlRef.current,
          mediaIdRef.current,
          pendingRef.current
        )
        // Best effort only (may not complete); the persisted copy is replayed next load.
        if (result === "deferred") {
          forceSyncRef.current()
        }
      }
    }

    window.addEventListener("beforeunload", handleBeforeUnload)
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload)
      // On unmount (not page close), regular async sync is fine
      if (statusRef.current === "pending") {
        forceSyncRef.current()
      }
    }
  }, [])
}
