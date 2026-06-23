import React from "react"
import { Button, Typography } from "antd"
import { Play } from "lucide-react"
import { useCreateAudioStudioGeneration } from "@/hooks/useAudioStudioGeneration"
import { useAudioStudioStore } from "@/store/audio-studio"
import {
  createAudioStudioIdempotencyKey,
  getFirstSectionTargetId,
  getMusicTrackTargetId,
  getProjectRevisionId
} from "./generationPayload"

const { Text } = Typography

export const GenerationPanel: React.FC = () => {
  const activeWorkflow = useAudioStudioStore((state) => state.activeWorkflow)
  const activeProject = useAudioStudioStore((state) => state.activeProject)
  const generation = useCreateAudioStudioGeneration(
    activeProject?.project_id ?? null
  )
  const isMusic = activeWorkflow === "music"
  const revisionId = getProjectRevisionId(activeProject)
  const sectionTargetId = getFirstSectionTargetId(activeProject)
  const disabledReason = !activeProject
    ? "Select a project before queuing generation."
    : !revisionId
      ? "Save the project before queuing generation."
      : !isMusic && !sectionTargetId
        ? "Add a section before queuing speech generation."
        : undefined

  const queueGeneration = () => {
    if (!activeProject || !revisionId) return

    if (isMusic) {
      void generation.mutateAsync({
        kind: "music",
        provider: "ace_step",
        idempotency_key: createAudioStudioIdempotencyKey(
          "music",
          activeProject.project_id
        ),
        target_resource_kind: "track",
        target_resource_id: getMusicTrackTargetId(activeProject),
        target_revision_id: revisionId,
        options: {
          prompt: "",
          lyrics: "",
          style: "",
          duration: 45
        }
      })
      return
    }

    if (!sectionTargetId) return

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

  return (
    <section className="rounded-md border border-border bg-surface p-3">
      <Text strong className="block">
        Generation
      </Text>
      <Text type="secondary" className="mt-1 block text-xs">
        {isMusic
          ? "Create music cues through server-side provider adapters."
          : "Create speech assets through server-managed TTS jobs."}
      </Text>
      <Button
        className="mt-3"
        type="primary"
        block
        icon={<Play className="h-4 w-4" />}
        disabled={Boolean(disabledReason) || generation.isPending}
        loading={generation.isPending}
        title={disabledReason}
        onClick={queueGeneration}
      >
        {isMusic ? "Queue music generation" : "Queue speech generation"}
      </Button>
    </section>
  )
}
