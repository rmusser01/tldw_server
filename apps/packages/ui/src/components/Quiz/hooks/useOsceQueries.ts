import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  beginOsceSelfAssessment,
  completeOsceAttempt,
  createOsceStation,
  deleteOsceStation,
  getOsceAttempt,
  getOsceStation,
  listAllOsceStations,
  listOsceAttempts,
  listOsceStations,
  patchOsceAttempt,
  startOsceAttempt,
  updateOsceStation,
  type OsceAttemptFilters,
  type OsceAttemptSummary,
  type OsceAttemptPatch,
  type OsceStationCreateRequest,
  type OsceStationListParams,
  type OsceStationPatchRequest
} from "@/services/osce"

const ACTIVE_OSCE_STATES = ["in_progress", "self_assessment"] as const

export const selectMostRecentlyModifiedOsceAttempt = (
  attempts: OsceAttemptSummary[]
): OsceAttemptSummary | null => {
  if (attempts.length === 0) return null
  return [...attempts].sort((left, right) => {
    const leftTime = Date.parse(left.last_modified_at)
    const rightTime = Date.parse(right.last_modified_at)
    const normalizedLeft = Number.isFinite(leftTime) ? leftTime : Number.NEGATIVE_INFINITY
    const normalizedRight = Number.isFinite(rightTime) ? rightTime : Number.NEGATIVE_INFINITY
    return normalizedRight - normalizedLeft || right.id - left.id
  })[0]
}

export const osceKeys = {
  all: ["quizzes", "osce"] as const,
  stationsRoot: (quizId: number) => [...osceKeys.all, "stations", quizId] as const,
  stations: (quizId: number, params: OsceStationListParams = {}) =>
    [...osceKeys.stationsRoot(quizId), params] as const,
  allStations: (quizId: number) => [...osceKeys.stationsRoot(quizId), "all"] as const,
  station: (quizId: number, stationId: number) =>
    [...osceKeys.stationsRoot(quizId), "detail", stationId] as const,
  attemptsRoot: () => [...osceKeys.all, "attempts"] as const,
  attempts: (filters: OsceAttemptFilters = {}) =>
    [...osceKeys.attemptsRoot(), filters] as const,
  attempt: (attemptId: number) => [...osceKeys.all, "attempt", attemptId] as const
}

const invalidateStationCaches = (queryClient: ReturnType<typeof useQueryClient>, quizId: number) => {
  queryClient.invalidateQueries({ queryKey: ["quizzes:list"] })
  queryClient.invalidateQueries({ queryKey: ["quizzes:detail", quizId] })
  queryClient.invalidateQueries({ queryKey: osceKeys.stationsRoot(quizId) })
}

const invalidateAttemptCaches = (
  queryClient: ReturnType<typeof useQueryClient>,
  attemptId?: number,
  quizId?: number
) => {
  queryClient.invalidateQueries({ queryKey: osceKeys.attemptsRoot() })
  if (attemptId !== undefined) queryClient.invalidateQueries({ queryKey: osceKeys.attempt(attemptId) })
  if (quizId !== undefined) queryClient.invalidateQueries({ queryKey: ["quizzes:detail", quizId] })
}

export const useOsceStationsQuery = (
  quizId: number | null | undefined,
  params: OsceStationListParams = {},
  options?: { enabled?: boolean }
) => useQuery({
  queryKey: osceKeys.stations(quizId ?? 0, params),
  queryFn: ({ signal }) => listOsceStations(quizId!, params, { signal }),
  enabled: (options?.enabled ?? true) && quizId != null,
  staleTime: 30_000,
  refetchOnWindowFocus: false
})

export const useAllOsceStationsQuery = (
  quizId: number | null | undefined,
  options?: { enabled?: boolean }
) => useQuery({
  queryKey: osceKeys.allStations(quizId ?? 0),
  queryFn: ({ signal }) => listAllOsceStations(quizId!, { signal }),
  enabled: (options?.enabled ?? true) && quizId != null,
  staleTime: 30_000,
  refetchOnWindowFocus: false
})

export const useOsceStationQuery = (
  quizId: number | null | undefined,
  stationId: number | null | undefined,
  options?: { enabled?: boolean }
) => useQuery({
  queryKey: osceKeys.station(quizId ?? 0, stationId ?? 0),
  queryFn: ({ signal }) => getOsceStation(quizId!, stationId!, { signal }),
  enabled: (options?.enabled ?? true) && quizId != null && stationId != null,
  staleTime: 30_000,
  refetchOnWindowFocus: false
})

export const useCreateOsceStationMutation = () => {
  const queryClient = useQueryClient()
  return useMutation({
    mutationKey: [...osceKeys.all, "station", "create"],
    mutationFn: ({ quizId, request }: { quizId: number; request: OsceStationCreateRequest }) =>
      createOsceStation(quizId, request),
    onSuccess: (station) => {
      queryClient.setQueryData(osceKeys.station(station.quiz_id, station.id), station)
      invalidateStationCaches(queryClient, station.quiz_id)
    }
  })
}

export const useUpdateOsceStationMutation = () => {
  const queryClient = useQueryClient()
  return useMutation({
    mutationKey: [...osceKeys.all, "station", "update"],
    mutationFn: ({ quizId, stationId, request }: {
      quizId: number
      stationId: number
      request: OsceStationPatchRequest
    }) => updateOsceStation(quizId, stationId, request),
    onSuccess: (station) => {
      queryClient.setQueryData(osceKeys.station(station.quiz_id, station.id), station)
      invalidateStationCaches(queryClient, station.quiz_id)
    }
  })
}

export const useDeleteOsceStationMutation = () => {
  const queryClient = useQueryClient()
  return useMutation({
    mutationKey: [...osceKeys.all, "station", "delete"],
    mutationFn: ({ quizId, stationId, expectedVersion }: {
      quizId: number
      stationId: number
      expectedVersion?: number
    }) => deleteOsceStation(quizId, stationId, expectedVersion),
    onSuccess: (_response, variables) => invalidateStationCaches(queryClient, variables.quizId)
  })
}

export const useOsceAttemptsQuery = (
  filters: OsceAttemptFilters = {},
  options?: { enabled?: boolean }
) => useQuery({
  queryKey: osceKeys.attempts(filters),
  queryFn: ({ signal }) => listOsceAttempts(filters, { signal }),
  enabled: options?.enabled ?? true,
  staleTime: 30_000,
  refetchOnWindowFocus: false
})

export const useActiveOsceAttemptsQuery = (
  quizId?: number | null,
  options?: { enabled?: boolean; limit?: number }
) => useOsceAttemptsQuery({
  quiz_id: quizId ?? undefined,
  states: [...ACTIVE_OSCE_STATES],
  limit: options?.limit ?? 200,
  offset: 0
}, options)

export const useCompletedOsceAttemptsQuery = (
  filters: Omit<OsceAttemptFilters, "states"> = {},
  options?: { enabled?: boolean }
) => useOsceAttemptsQuery({ ...filters, states: ["completed"] }, options)

export const useOsceAttemptQuery = (
  attemptId: number | null | undefined,
  options?: { enabled?: boolean }
) => useQuery({
  queryKey: osceKeys.attempt(attemptId ?? 0),
  queryFn: ({ signal }) => getOsceAttempt(attemptId!, { signal }),
  enabled: (options?.enabled ?? true) && attemptId != null,
  staleTime: 30_000,
  refetchOnWindowFocus: false
})

export const useStartOsceAttemptMutation = () => {
  const queryClient = useQueryClient()
  return useMutation({
    mutationKey: [...osceKeys.all, "attempt", "start"],
    mutationFn: ({ stationId, clientAttemptId }: { stationId: number; clientAttemptId: string }) =>
      startOsceAttempt(stationId, clientAttemptId),
    onSuccess: (attempt) => {
      queryClient.setQueryData(osceKeys.attempt(attempt.id), attempt)
      invalidateAttemptCaches(queryClient, attempt.id, attempt.quiz_id)
    }
  })
}

export const usePatchOsceAttemptMutation = () => {
  const queryClient = useQueryClient()
  return useMutation({
    mutationKey: [...osceKeys.all, "attempt", "patch"],
    mutationFn: ({ attemptId, patch }: { attemptId: number; patch: OsceAttemptPatch }) =>
      patchOsceAttempt(attemptId, patch),
    onSuccess: (attempt) => {
      queryClient.setQueryData(osceKeys.attempt(attempt.id), attempt)
      invalidateAttemptCaches(queryClient, undefined, attempt.quiz_id)
    }
  })
}

const useTransitionMutation = (
  operation: typeof beginOsceSelfAssessment | typeof completeOsceAttempt,
  suffix: string
) => {
  const queryClient = useQueryClient()
  return useMutation({
    mutationKey: [...osceKeys.all, "attempt", suffix],
    mutationFn: ({ attemptId, expectedVersion }: { attemptId: number; expectedVersion: number }) =>
      operation(attemptId, expectedVersion),
    onSuccess: (attempt) => {
      queryClient.setQueryData(osceKeys.attempt(attempt.id), attempt)
      invalidateAttemptCaches(queryClient, undefined, attempt.quiz_id)
    }
  })
}

export const useBeginOsceSelfAssessmentMutation = () =>
  useTransitionMutation(beginOsceSelfAssessment, "begin-self-assessment")

export const useCompleteOsceAttemptMutation = () =>
  useTransitionMutation(completeOsceAttempt, "complete")
