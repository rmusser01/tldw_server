import { useEffect } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as audioStudioService from "@/services/audio-studio";
import { useAudioStudioStore } from "@/store/audio-studio";

export const audioStudioProjectQueryKeys = {
  all: ["audio-studio"] as const,
  projects: () => [...audioStudioProjectQueryKeys.all, "projects"] as const,
  projectList: (
    params: audioStudioService.ListAudioStudioProjectsParams = {},
  ) => [...audioStudioProjectQueryKeys.projects(), params] as const,
  artifacts: (projectId: string | null) =>
    [...audioStudioProjectQueryKeys.all, "artifacts", projectId] as const,
};

export const useAudioStudioProjects = (
  params: audioStudioService.ListAudioStudioProjectsParams = {},
) => {
  const setProjects = useAudioStudioStore((state) => state.setProjects);
  const query = useQuery({
    queryKey: audioStudioProjectQueryKeys.projectList(params),
    queryFn: () => audioStudioService.listAudioStudioProjects(params),
  });

  useEffect(() => {
    if (query.data) {
      setProjects(query.data);
    }
  }, [query.data, setProjects]);

  return query;
};

export const useAudioStudioArtifacts = (projectId: string | null) =>
  useQuery({
    queryKey: audioStudioProjectQueryKeys.artifacts(projectId),
    queryFn: () => {
      if (!projectId) return Promise.resolve([]);
      return audioStudioService.listAudioStudioArtifacts(projectId);
    },
    enabled: Boolean(projectId),
    initialData: [],
  });

export const useCreateAudioStudioProject = () => {
  const queryClient = useQueryClient();
  const upsertProjectFromServer = useAudioStudioStore(
    (state) => state.upsertProjectFromServer,
  );
  const setActiveProjectId = useAudioStudioStore(
    (state) => state.setActiveProjectId,
  );

  return useMutation({
    mutationFn: (payload: audioStudioService.CreateAudioStudioProjectRequest) =>
      audioStudioService.createAudioStudioProject(payload),
    onSuccess: (project) => {
      upsertProjectFromServer(project);
      setActiveProjectId(project.project_id);
      queryClient.invalidateQueries({
        queryKey: audioStudioProjectQueryKeys.projects(),
      });
    },
  });
};

export const useUpdateAudioStudioProject = (projectId: string | null) => {
  const queryClient = useQueryClient();
  const upsertProjectFromServer = useAudioStudioStore(
    (state) => state.upsertProjectFromServer,
  );
  const markProjectClean = useAudioStudioStore(
    (state) => state.markProjectClean,
  );

  return useMutation({
    mutationFn: (
      payload: audioStudioService.UpdateAudioStudioProjectRequest,
    ) => {
      if (!projectId) throw new Error("Audio Studio project is required");
      return audioStudioService.updateAudioStudioProject(projectId, payload);
    },
    onSuccess: (project) => {
      markProjectClean(project.project_id);
      upsertProjectFromServer(project);
      queryClient.invalidateQueries({
        queryKey: audioStudioProjectQueryKeys.projects(),
      });
    },
  });
};

export const useUpsertAudioStudioSection = (projectId: string | null) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      sectionId,
      payload,
    }: {
      sectionId: string;
      payload: audioStudioService.AudioStudioSectionUpsertRequest;
    }) => {
      if (!projectId) throw new Error("Audio Studio project is required");
      return audioStudioService.upsertAudioStudioSection(
        projectId,
        sectionId,
        payload,
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: audioStudioProjectQueryKeys.projects(),
      });
    },
  });
};

export const useUpsertAudioStudioClip = (projectId: string | null) => {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      clipId,
      payload,
    }: {
      clipId: string;
      payload: audioStudioService.AudioStudioClipUpsertRequest;
    }) => {
      if (!projectId) throw new Error("Audio Studio project is required");
      return audioStudioService.upsertAudioStudioClip(
        projectId,
        clipId,
        payload,
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: audioStudioProjectQueryKeys.projects(),
      });
    },
  });
};
