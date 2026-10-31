import { useEffect } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import * as audioStudioService from "@/services/audio-studio";
import { useAudioStudioSession } from "@/components/Option/AudioStudio/AudioStudioSession";
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
  const session = useAudioStudioSession();
  const setProjects = useAudioStudioStore((state) => state.setProjects);
  const query = useQuery({
    queryKey: [...audioStudioProjectQueryKeys.projectList(params), session?.ownerKey],
    queryFn: () => audioStudioService.listAudioStudioProjects(params, session?.options),
  });

  useEffect(() => {
    if (query.data) {
      session?.options.assertCurrent?.();
      setProjects(query.data);
    }
  }, [query.data, setProjects, session]);

  return query;
};

export const useAudioStudioProjectDetail = (projectId: string | null) => {
  const session = useAudioStudioSession();
  const query = useQuery({
    queryKey: [...audioStudioProjectQueryKeys.projects(), "detail", projectId, session?.ownerKey],
    queryFn: () => audioStudioService.getAudioStudioProject(projectId!, session?.options),
    enabled: Boolean(projectId),
  });
  useEffect(() => {
    if (query.data) {
      session?.options.assertCurrent?.();
      useAudioStudioStore.getState().upsertProjectFromServer(query.data);
    }
  }, [query.data, session]);
  return query;
};

export const useAudioStudioArtifacts = (projectId: string | null) => {
  const session = useAudioStudioSession();
  return useQuery({
    queryKey: [...audioStudioProjectQueryKeys.artifacts(projectId), session?.ownerKey],
    queryFn: () => audioStudioService.listAudioStudioArtifacts(projectId!, session?.options),
    enabled: Boolean(projectId),
    initialData: [],
    refetchInterval: projectId ? 2000 : false,
  });
};

export const useCreateAudioStudioProject = () => {
  const session = useAudioStudioSession();
  const queryClient = useQueryClient();
  const upsertProjectFromServer = useAudioStudioStore(
    (state) => state.upsertProjectFromServer,
  );
  const setActiveProjectId = useAudioStudioStore(
    (state) => state.setActiveProjectId,
  );

  return useMutation({
    mutationFn: (payload: audioStudioService.CreateAudioStudioProjectRequest) =>
      audioStudioService.createAudioStudioProject(payload, session?.options),
    onSuccess: (project) => {
      session?.options.assertCurrent?.();
      upsertProjectFromServer(project);
      setActiveProjectId(project.project_id);
      queryClient.invalidateQueries({
        queryKey: audioStudioProjectQueryKeys.projects(),
      });
    },
  });
};

export const useUpdateAudioStudioProject = (projectId: string | null) => {
  const session = useAudioStudioSession();
  const queryClient = useQueryClient();
  const upsertProjectFromServer = useAudioStudioStore(
    (state) => state.upsertProjectFromServer,
  );
  const markProjectClean = useAudioStudioStore(
    (state) => state.markProjectClean,
  );

  return useMutation({
    mutationFn: async (payload: audioStudioService.UpdateAudioStudioProjectRequest) => {
      if (!projectId) throw new Error("Audio Studio project is required");
      const local = useAudioStudioStore.getState().projects.find((p) => p.project_id === projectId);
      let expectedLocal = local;
      let newerEdits = false;
      const acknowledge = (revision: string | undefined) => {
        session?.options.assertCurrent?.();
        if (!revision) throw new Error("The server did not return a saved revision. Reload before retrying.");
        const store = useAudioStudioStore.getState();
        const current = store.projects.find((p) => p.project_id === projectId);
        if (current !== expectedLocal) newerEdits = true;
        if (current) {
          store.updateProjectLocal(projectId, { current_revision_id: revision, revision_id: revision });
          expectedLocal = useAudioStudioStore.getState().projects.find((p) => p.project_id === projectId);
        }
      };
      let project = await audioStudioService.updateAudioStudioProject(projectId, payload, session?.options);
      let revision = project.current_revision_id ?? project.revision_id;
      acknowledge(revision);
      try {
        for (const section of [...(local?.sections ?? [])].sort((a, b) => a.order - b.order)) {
          if (!revision) throw new Error("The project revision is missing. Reload before saving.");
          const saved = await audioStudioService.upsertAudioStudioSection(projectId, section.section_id, {
            base_revision_id: revision, title: section.title, body_text: section.body_text,
            speaker_id: section.speaker_id, order_index: section.order, settings: section.settings,
          }, session?.options);
          revision = saved.current_revision_id ?? undefined;
          acknowledge(revision);
        }
        if (local?.sections.length) project = await audioStudioService.getAudioStudioProject(projectId, session?.options);
        session?.options.assertCurrent?.();
        const current = useAudioStudioStore.getState().projects.find((p) => p.project_id === projectId);
        if (!newerEdits && current === expectedLocal) {
          markProjectClean(projectId);
          upsertProjectFromServer(project);
        }
      } catch (error) {
        // A partial save retains the draft; detail refresh records the new server revision as a conflict.
        void queryClient.invalidateQueries({ queryKey: audioStudioProjectQueryKeys.projects() });
        throw error;
      }
      return project;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: audioStudioProjectQueryKeys.projects() });
    },
  });
};

export const useUpsertAudioStudioSection = (projectId: string | null) => {
  const session = useAudioStudioSession();
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
        session?.options,
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
  const session = useAudioStudioSession();
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
        session?.options,
      );
    },
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: audioStudioProjectQueryKeys.projects(),
      });
    },
  });
};
