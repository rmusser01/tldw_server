import { useMutation, useQueryClient } from '@tanstack/react-query';
import type {
  AudioStudioAdmissionKind,
  AudioStudioAdmissionBody,
  AudioStudioGenerationCreateRequest,
  AudioStudioRenderCreateRequest,
  AudioStudioExportCreateRequest,
} from '@/services/audio-studio';
import { useAudioStudioSession } from '@/components/Option/AudioStudio/AudioStudioSession';
import { admitAudioStudioJob } from '@/components/Option/AudioStudio/audioStudioAdmission';
import { useAudioStudioStore } from '@/store/audio-studio';
import { audioStudioProjectQueryKeys } from '@/hooks/useAudioStudioProjects';

const useAudioStudioAdmission = <T extends AudioStudioAdmissionBody>(
  kind: AudioStudioAdmissionKind,
  projectId: string | null,
  retryStored = false,
) => {
  const session = useAudioStudioSession();
  const client = useQueryClient();
  return useMutation({
    mutationFn: (payload: T) => {
      if (!projectId) throw new Error('Audio Studio project is required');
      if (!session) throw new Error('A verified Audio Studio account is required.');
      return admitAudioStudioJob(
        session.ownerKey,
        projectId,
        kind,
        payload,
        session.options,
        retryStored,
      );
    },
    onSettled: () => {
      void client.invalidateQueries({ queryKey: audioStudioProjectQueryKeys.all });
    },
  });
};
export const useCreateAudioStudioGeneration = (projectId: string | null) =>
  useAudioStudioAdmission<AudioStudioGenerationCreateRequest>('generation', projectId);
export const useCreateAudioStudioRender = (projectId: string | null) =>
  useAudioStudioAdmission<AudioStudioRenderCreateRequest>('render', projectId);
export const useCreateAudioStudioExport = (projectId: string | null) =>
  useAudioStudioAdmission<AudioStudioExportCreateRequest>('export', projectId);
export const useRetryAudioStudioAdmission = (projectId: string | null) => {
  const intent = useAudioStudioStore((state) =>
    projectId ? state.jobIntents[projectId] : undefined,
  );
  const mutation = useAudioStudioAdmission(intent?.kind ?? 'generation', projectId, true);
  return { ...mutation, intent };
};
