import { useEffect } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  createAudioStudioProject,
  listAudioStudioProjects,
  updateAudioStudioProject,
  upsertAudioStudioClip,
  upsertAudioStudioSection,
  type AudioStudioClipUpsertRequest,
  type AudioStudioSectionUpsertRequest,
  type CreateAudioStudioProjectRequest,
  type ListAudioStudioProjectsParams,
  type UpdateAudioStudioProjectRequest
} from "@/services/audio-studio"
import { useAudioStudioStore } from "@/store/audio-studio"

export const audioStudioProjectQueryKeys = {
  all: ["audio-studio"] as const,
  projects: () => [...audioStudioProjectQueryKeys.all, "projects"] as const,
  projectList: (params: ListAudioStudioProjectsParams = {}) =>
    [...audioStudioProjectQueryKeys.projects(), params] as const
}

export const useAudioStudioProjects = (
  params: ListAudioStudioProjectsParams = {}
) => {
  const setProjects = useAudioStudioStore((state) => state.setProjects)
  const query = useQuery({
    queryKey: audioStudioProjectQueryKeys.projectList(params),
    queryFn: () => listAudioStudioProjects(params)
  })

  useEffect(() => {
    if (query.data) {
      setProjects(query.data)
    }
  }, [query.data, setProjects])

  return query
}

export const useCreateAudioStudioProject = () => {
  const queryClient = useQueryClient()
  const upsertProjectFromServer = useAudioStudioStore(
    (state) => state.upsertProjectFromServer
  )
  const setActiveProjectId = useAudioStudioStore((state) => state.setActiveProjectId)

  return useMutation({
    mutationFn: (payload: CreateAudioStudioProjectRequest) =>
      createAudioStudioProject(payload),
    onSuccess: (project) => {
      upsertProjectFromServer(project)
      setActiveProjectId(project.project_id)
      queryClient.invalidateQueries({
        queryKey: audioStudioProjectQueryKeys.projects()
      })
    }
  })
}

export const useUpdateAudioStudioProject = (projectId: string | null) => {
  const queryClient = useQueryClient()
  const upsertProjectFromServer = useAudioStudioStore(
    (state) => state.upsertProjectFromServer
  )
  const markProjectClean = useAudioStudioStore((state) => state.markProjectClean)

  return useMutation({
    mutationFn: (payload: UpdateAudioStudioProjectRequest) => {
      if (!projectId) throw new Error("Audio Studio project is required")
      return updateAudioStudioProject(projectId, payload)
    },
    onSuccess: (project) => {
      markProjectClean(project.project_id)
      upsertProjectFromServer(project)
      queryClient.invalidateQueries({
        queryKey: audioStudioProjectQueryKeys.projects()
      })
    }
  })
}

export const useUpsertAudioStudioSection = (projectId: string | null) => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({
      sectionId,
      payload
    }: {
      sectionId: string
      payload: AudioStudioSectionUpsertRequest
    }) => {
      if (!projectId) throw new Error("Audio Studio project is required")
      return upsertAudioStudioSection(projectId, sectionId, payload)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: audioStudioProjectQueryKeys.projects()
      })
    }
  })
}

export const useUpsertAudioStudioClip = (projectId: string | null) => {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({
      clipId,
      payload
    }: {
      clipId: string
      payload: AudioStudioClipUpsertRequest
    }) => {
      if (!projectId) throw new Error("Audio Studio project is required")
      return upsertAudioStudioClip(projectId, clipId, payload)
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: audioStudioProjectQueryKeys.projects()
      })
    }
  })
}
