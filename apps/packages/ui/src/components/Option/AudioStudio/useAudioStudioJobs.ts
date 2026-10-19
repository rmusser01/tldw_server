import { useEffect } from 'react';
import { useAudioStudioStore } from '@/store/audio-studio';
import { isTerminalAudioStudioJob, saveAudioStudioIntent } from './audioStudioAdmission';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { cancelAudioStudioJob, listAudioStudioJobs } from '@/services/audio-studio';
import { audioStudioProjectQueryKeys } from '@/hooks/useAudioStudioProjects';
import { useAudioStudioSession } from './AudioStudioSession';
export const isAudioStudioJobActive = (status: string) =>
  ['queued', 'pending', 'scheduled', 'running', 'processing', 'leased', 'retrying'].includes(
    status,
  );
export const useAudioStudioJobs = (projectId: string | null) => {
  const session = useAudioStudioSession();
  const client = useQueryClient();
  const intent = useAudioStudioStore((state) =>
    projectId ? state.jobIntents[projectId] : undefined,
  );
  const query = useQuery({
    queryKey: [...audioStudioProjectQueryKeys.all, 'jobs', projectId, session?.ownerKey],
    queryFn: () => listAudioStudioJobs(projectId!, session?.options),
    enabled: Boolean(projectId),
    refetchInterval: projectId ? 2000 : false,
  });
  useEffect(() => {
    if (!session || !projectId || !intent?.job_id) return;
    const job = query.data?.find((candidate) => candidate.job_id === intent.job_id);
    if (!job || !isTerminalAudioStudioJob(job.status)) return;
    let cancelled = false;
    void client
      .invalidateQueries(
        { queryKey: audioStudioProjectQueryKeys.artifacts(projectId) },
        { cancelRefetch: false, throwOnError: true },
      )
      .then(() => {
        if (
          cancelled ||
          useAudioStudioStore.getState().jobIntents[projectId]?.job_id !== intent.job_id
        )
          return;
        session.options.assertCurrent?.();
        saveAudioStudioIntent(session.ownerKey, projectId);
      })
      .catch(() => {
        /* Retain the intent until both artifacts and browser storage are available. */
      });
    return () => {
      cancelled = true;
    };
  }, [session, projectId, intent, query.data, query.dataUpdatedAt, client]);
  return query;
};
export const useCancelAudioStudioJob = (projectId: string | null) => {
  const session = useAudioStudioSession();
  const client = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => {
      if (!projectId) throw new Error('Select a project first.');
      return cancelAudioStudioJob(projectId, jobId, session?.options);
    },
    onSettled: () => {
      void client.invalidateQueries({ queryKey: audioStudioProjectQueryKeys.all });
    },
  });
};
