import { useEffect } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import {
  createAudioStudioProject,
  listAudioStudioProjects,
  type CreateAudioStudioProjectRequest,
  type ListAudioStudioProjectsParams
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
