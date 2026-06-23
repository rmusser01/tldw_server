import { useMutation, useQueryClient } from "@tanstack/react-query"
import {
  createAudioStudioExport,
  createAudioStudioGeneration,
  createAudioStudioRender,
  type AudioStudioExportCreateRequest,
  type AudioStudioGenerationCreateRequest,
  type AudioStudioRenderCreateRequest
} from "@/services/audio-studio"
import { audioStudioProjectQueryKeys } from "@/hooks/useAudioStudioProjects"

const invalidateProjectQueries = (
  queryClient: ReturnType<typeof useQueryClient>
) => {
  queryClient.invalidateQueries({
    queryKey: audioStudioProjectQueryKeys.projects()
  })
}

export const useCreateAudioStudioGeneration = (projectId: string | null) => {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (payload: AudioStudioGenerationCreateRequest) => {
      if (!projectId) throw new Error("Audio Studio project is required")
      return createAudioStudioGeneration(projectId, payload)
    },
    onSuccess: () => invalidateProjectQueries(queryClient)
  })
}

export const useCreateAudioStudioRender = (projectId: string | null) => {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (payload: AudioStudioRenderCreateRequest) => {
      if (!projectId) throw new Error("Audio Studio project is required")
      return createAudioStudioRender(projectId, payload)
    },
    onSuccess: () => invalidateProjectQueries(queryClient)
  })
}

export const useCreateAudioStudioExport = (projectId: string | null) => {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (payload: AudioStudioExportCreateRequest) => {
      if (!projectId) throw new Error("Audio Studio project is required")
      return createAudioStudioExport(projectId, payload)
    },
    onSuccess: () => invalidateProjectQueries(queryClient)
  })
}
