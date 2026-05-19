import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useEffect, useState } from "react"

import {
  getStudySuggestionAnchorStatus,
  getStudySuggestionSnapshot,
  performStudySuggestionAction,
  refreshStudySuggestionSnapshot,
  type SuggestionAnchorType,
  type SuggestionStatus,
  type StudySuggestionActionRequest,
  type StudySuggestionActionResponse,
  type StudySuggestionSnapshotResponse,
  type StudySuggestionStatusResponse
} from "@/services/studySuggestions"
import { useServerOnline } from "@/hooks/useServerOnline"

const STUDY_SUGGESTIONS_POLL_INTERVAL_MS = 1500

export type UseStudySuggestionsOptions = {
  enabled?: boolean
}

export type UseStudySuggestionsResult = {
  status: SuggestionStatus
  statusQuery: ReturnType<typeof useQuery<StudySuggestionStatusResponse, Error>>
  snapshot: StudySuggestionSnapshotResponse | null
  activeSnapshotId: number | null
  isLoading: boolean
  isRefreshing: boolean
  refresh: (reason?: string | null) => Promise<unknown>
  performAction: (
    request: StudySuggestionActionRequest
  ) => Promise<StudySuggestionActionResponse>
}

const buildStatusQueryKey = (
  anchorType: SuggestionAnchorType | null | undefined,
  anchorId: number | null | undefined
) => ["study-suggestions:anchor-status", anchorType ?? null, anchorId ?? null]

const buildSnapshotQueryKey = (snapshotId: number | null) => [
  "study-suggestions:snapshot",
  snapshotId
]

export function useStudySuggestions(
  anchorType: SuggestionAnchorType | null | undefined,
  anchorId: number | null | undefined,
  options?: UseStudySuggestionsOptions
): UseStudySuggestionsResult {
  const isOnline = useServerOnline()
  const queryClient = useQueryClient()
  const [activeSnapshotId, setActiveSnapshotId] = useState<number | null>(null)
  const [visibleSnapshot, setVisibleSnapshot] =
    useState<StudySuggestionSnapshotResponse | null>(null)

  const statusQuery = useQuery({
    queryKey: buildStatusQueryKey(anchorType, anchorId),
    queryFn: async ({ signal }) => {
      if (anchorType == null || anchorId == null) {
        throw new Error("Study suggestions anchor is required")
      }
      return await getStudySuggestionAnchorStatus(anchorType, anchorId, { signal })
    },
    enabled:
      (options?.enabled ?? true) &&
      isOnline &&
      anchorType != null &&
      anchorId != null,
    retry: false,
    refetchIntervalInBackground: true,
    refetchInterval: (query) => {
      const status = query.state.data?.status
      return status === "ready" || status === "failed"
        ? false
        : STUDY_SUGGESTIONS_POLL_INTERVAL_MS
    }
  })

  // Reset snapshot state when anchor changes to prevent stale data
  useEffect(() => {
    setActiveSnapshotId(null)
    setVisibleSnapshot(null)
  }, [anchorType, anchorId])

  useEffect(() => {
    const nextSnapshotId = statusQuery.data?.snapshot_id ?? null
    if (statusQuery.data?.status === "ready" && nextSnapshotId != null) {
      setActiveSnapshotId(nextSnapshotId)
      return
    }
    if (
      statusQuery.data?.status === "none" ||
      statusQuery.data?.status === "failed"
    ) {
      setActiveSnapshotId(null)
      setVisibleSnapshot(null)
    }
  }, [statusQuery.data])

  const snapshotQuery = useQuery({
    queryKey: buildSnapshotQueryKey(activeSnapshotId),
    queryFn: async ({ signal }) => {
      if (activeSnapshotId == null) {
        throw new Error("Study suggestion snapshot is unavailable")
      }
      return await getStudySuggestionSnapshot(activeSnapshotId, { signal })
    },
    enabled:
      (options?.enabled ?? true) &&
      isOnline &&
      activeSnapshotId != null,
    retry: false
  })

  useEffect(() => {
    if (snapshotQuery.data) {
      setVisibleSnapshot(snapshotQuery.data)
    }
  }, [snapshotQuery.data])

  const refreshMutation = useMutation({
    mutationKey: ["study-suggestions:refresh"],
    mutationFn: async (reason?: string | null) => {
      const snapshotId = activeSnapshotId ?? statusQuery.data?.snapshot_id ?? null
      if (snapshotId == null) {
        throw new Error("Study suggestion snapshot is unavailable")
      }
      return await refreshStudySuggestionSnapshot(snapshotId, { reason })
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: buildStatusQueryKey(anchorType, anchorId)
      })
    }
  })

  const actionMutation = useMutation({
    mutationKey: ["study-suggestions:action"],
    mutationFn: async (request: StudySuggestionActionRequest) => {
      const snapshotId = activeSnapshotId ?? statusQuery.data?.snapshot_id ?? null
      if (snapshotId == null) {
        throw new Error("Study suggestion snapshot is unavailable")
      }
      return await performStudySuggestionAction(snapshotId, request)
    },
    onSuccess: async () => {
      await queryClient.invalidateQueries({
        queryKey: buildStatusQueryKey(anchorType, anchorId)
      })
    }
  })

  return {
    status: statusQuery.data?.status ?? "none",
    statusQuery,
    snapshot: visibleSnapshot,
    activeSnapshotId,
    isLoading: statusQuery.isLoading || snapshotQuery.isLoading,
    isRefreshing: refreshMutation.isPending,
    refresh: refreshMutation.mutateAsync,
    performAction: actionMutation.mutateAsync
  }
}
