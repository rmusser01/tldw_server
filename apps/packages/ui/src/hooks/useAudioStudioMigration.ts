import { useMutation, useQueryClient } from "@tanstack/react-query"
import {
  commitAudiobookMigration,
  previewAudiobookMigration,
  type AudiobookMigrationCommitRequest,
  type AudiobookMigrationPreviewRequest
} from "@/services/audio-studio"
import { audioStudioProjectQueryKeys } from "@/hooks/useAudioStudioProjects"

export const usePreviewAudiobookMigration = () =>
  useMutation({
    mutationFn: (payload: AudiobookMigrationPreviewRequest) =>
      previewAudiobookMigration(payload)
  })

export const useCommitAudiobookMigration = () => {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (payload: AudiobookMigrationCommitRequest) =>
      commitAudiobookMigration(payload),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: audioStudioProjectQueryKeys.projects()
      })
    }
  })
}
