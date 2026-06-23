import { useCreateAudioStudioGeneration } from "@/hooks/useAudioStudioGeneration"
import { useAudioStudioStore } from "@/store/audio-studio"
import {
  createAudioStudioIdempotencyKey,
  getFirstSectionTargetId,
  getMusicTrackTargetId,
  getProjectRevisionId
} from "./generationPayload"

type MusicOptions = {
  prompt: string
  lyrics: string
  style: string
  duration: number
}

export const useAudioStudioGenerationActions = () => {
  const activeWorkflow = useAudioStudioStore((state) => state.activeWorkflow)
  const activeProject = useAudioStudioStore((state) => state.activeProject)
  const generation = useCreateAudioStudioGeneration(
    activeProject?.project_id ?? null
  )
  const revisionId = getProjectRevisionId(activeProject)
  const sectionTargetId = getFirstSectionTargetId(activeProject)
  const workflowMismatch =
    activeProject !== null && activeProject.workflow !== activeWorkflow

  const projectDisabledReason = !activeProject
    ? "Select a project before queuing generation."
    : workflowMismatch
      ? "Select a project for this workflow before queuing generation."
    : !revisionId
      ? "Save the project before queuing generation."
      : undefined
  const speechDisabledReason =
    projectDisabledReason ??
    (!sectionTargetId
      ? "Add a section before queuing speech generation."
      : undefined)
  const musicDisabledReason = projectDisabledReason

  const queueMusicGeneration = (
    options: MusicOptions,
    provider = "ace_step"
  ) => {
    if (!activeProject || workflowMismatch || !revisionId) return

    void generation.mutateAsync({
      kind: "music",
      provider,
      idempotency_key: createAudioStudioIdempotencyKey(
        "music",
        activeProject.project_id
      ),
      target_resource_kind: "track",
      target_resource_id: getMusicTrackTargetId(activeProject),
      target_revision_id: revisionId,
      options
    })
  }

  const queueSpeechGeneration = () => {
    if (!activeProject || workflowMismatch || !revisionId || !sectionTargetId) return

    void generation.mutateAsync({
      kind: "speech",
      provider: "tts",
      idempotency_key: createAudioStudioIdempotencyKey(
        "speech",
        activeProject.project_id
      ),
      target_resource_kind: "section",
      target_resource_id: sectionTargetId,
      target_revision_id: revisionId,
      options: {
        workflow: activeWorkflow
      }
    })
  }

  return {
    isPending: generation.isPending,
    musicDisabledReason,
    queueMusicGeneration,
    queueSpeechGeneration,
    speechDisabledReason
  }
}
