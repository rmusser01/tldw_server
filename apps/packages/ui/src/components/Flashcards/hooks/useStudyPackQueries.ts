import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"

import {
  createStudyPackJob,
  getStudyPack,
  getStudyPackJob,
  regenerateStudyPackJob,
  type StudyPackCreateJobRequest,
  type StudyPackJobApiStatus
} from "@/services/flashcards"
import { useFlashcardsEnabled } from "./useFlashcardQueries"

const STUDY_PACK_JOB_POLL_INTERVAL_MS = 1500

export type UseStudyPackQueryOptions = {
  enabled?: boolean
}

export const isTerminalStudyPackJobStatus = (
  status: StudyPackJobApiStatus | null | undefined
): boolean => {
  return status === "completed" || status === "failed" || status === "cancelled"
}

const invalidateStudyPackQueries = async (queryClient: ReturnType<typeof useQueryClient>) => {
  await queryClient.invalidateQueries({
    predicate: (query) =>
      Array.isArray(query.queryKey) &&
      typeof query.queryKey[0] === "string" &&
      query.queryKey[0].startsWith("flashcards:study-packs")
  })
}

export function useStudyPackCreateMutation() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationKey: ["flashcards:study-packs:create"],
    mutationFn: (request: StudyPackCreateJobRequest) => createStudyPackJob(request),
    onSuccess: async () => {
      await invalidateStudyPackQueries(queryClient)
    }
  })
}

export function useStudyPackJobQuery(
  jobId: number | null | undefined,
  options?: UseStudyPackQueryOptions
) {
  const { flashcardsEnabled } = useFlashcardsEnabled()

  return useQuery({
    queryKey: ["flashcards:study-packs:job", jobId ?? null],
    queryFn: async () => {
      if (jobId == null) {
        return null
      }
      return await getStudyPackJob(jobId)
    },
    enabled: (options?.enabled ?? true) && flashcardsEnabled && jobId != null,
    retry: false,
    refetchIntervalInBackground: true,
    refetchInterval: (query) => {
      const status = query.state.data?.job?.status
      return isTerminalStudyPackJobStatus(status) ? false : STUDY_PACK_JOB_POLL_INTERVAL_MS
    }
  })
}

export function useStudyPackQuery(
  packId: number | null | undefined,
  options?: UseStudyPackQueryOptions
) {
  const { flashcardsEnabled } = useFlashcardsEnabled()

  return useQuery({
    queryKey: ["flashcards:study-packs:pack", packId ?? null],
    queryFn: async () => {
      if (packId == null) {
        return null
      }
      return await getStudyPack(packId)
    },
    enabled: (options?.enabled ?? true) && flashcardsEnabled && packId != null,
    retry: false
  })
}

export function useStudyPackRegenerateMutation() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationKey: ["flashcards:study-packs:regenerate"],
    mutationFn: (packId: number) => regenerateStudyPackJob(packId),
    onSuccess: async () => {
      await invalidateStudyPackQueries(queryClient)
    }
  })
}
