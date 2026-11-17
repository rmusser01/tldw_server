import { useCreateAudioStudioGeneration } from '@/hooks/useAudioStudioGeneration';
import { useAudioStudioStore } from '@/store/audio-studio';
import {
  createAudioStudioIdempotencyKey,
  getFirstSectionTargetId,
  getMusicTrackTargetId,
  getSpeechOptions,
  getProjectRevisionId,
} from './generationPayload';

type MusicOptions = {
  prompt: string;
  lyrics: string;
  style: string;
  duration: number;
};

export const useAudioStudioGenerationActions = () => {
  const activeWorkflow = useAudioStudioStore((state) => state.activeWorkflow);
  const activeProject = useAudioStudioStore((state) => state.activeProject);
  const generation = useCreateAudioStudioGeneration(activeProject?.project_id ?? null);
  const intent = useAudioStudioStore((state) =>
    activeProject ? state.jobIntents[activeProject.project_id] : undefined,
  );
  const isDirty = useAudioStudioStore((state) =>
    activeProject ? Boolean(state.dirtyProjectIds[activeProject.project_id]) : false,
  );
  const revisionId = getProjectRevisionId(activeProject);
  const sectionTargetId = getFirstSectionTargetId(activeProject);
  const workflowMismatch = activeProject !== null && activeProject.workflow !== activeWorkflow;

  const projectDisabledReason = !activeProject
    ? 'Select a project before queuing generation.'
    : workflowMismatch
      ? 'Select a project for this workflow before queuing generation.'
      : intent
        ? 'Resolve the pending project job before queuing more work.'
        : isDirty
          ? 'Save your changes before queuing generation.'
          : !revisionId
            ? 'Save the project before queuing generation.'
            : undefined;
  let speechValidation: string | undefined;
  if (activeProject && sectionTargetId) {
    const section = activeProject.sections.find((s) => s.section_id === sectionTargetId)!;
    try {
      getSpeechOptions(activeProject, section);
    } catch (error) {
      speechValidation = (error as Error).message;
    }
    if (!section.body_text?.trim())
      speechValidation = 'Enter and save section text before generating.';
  }
  const speechDisabledReason =
    projectDisabledReason ??
    speechValidation ??
    (!sectionTargetId ? 'Add a section before queuing speech generation.' : undefined);
  const musicDisabledReason = projectDisabledReason;

  const queueMusicGeneration = (options: MusicOptions, provider = 'ace_step') => {
    if (!activeProject || projectDisabledReason || !revisionId) return;

    return generation.mutateAsync({
      kind: 'music',
      provider,
      idempotency_key: createAudioStudioIdempotencyKey('music', activeProject.project_id),
      target_resource_kind: 'track',
      target_resource_id: getMusicTrackTargetId(activeProject),
      target_revision_id: revisionId,
      options,
    });
  };

  const queueSpeechGeneration = () => {
    return queueSpeechGenerationForSection(sectionTargetId ?? undefined);
  };

  const queueSpeechGenerationForSection = (sectionId?: string, targetRevisionId = revisionId) => {
    if (!activeProject || projectDisabledReason || !targetRevisionId || !sectionId) return;
    const section = activeProject.sections.find((s) => s.section_id === sectionId);
    if (!section?.body_text?.trim())
      throw new Error('Enter and save section text before generating.');
    const options = getSpeechOptions(activeProject, section);

    return generation.mutateAsync({
      kind: 'speech',
      provider: 'tts',
      idempotency_key: createAudioStudioIdempotencyKey('speech', activeProject.project_id),
      target_resource_kind: 'section',
      target_resource_id: sectionId,
      target_revision_id: targetRevisionId,
      options: {
        ...options,
        workflow: activeWorkflow,
      },
    });
  };

  return {
    isPending: generation.isPending,
    hasPendingIntent: Boolean(intent),
    error: generation.error,
    musicDisabledReason,
    queueMusicGeneration,
    queueSpeechGeneration,
    queueSpeechGenerationForSection,
    speechDraftDisabledReason: projectDisabledReason,
    speechDisabledReason,
  };
};
