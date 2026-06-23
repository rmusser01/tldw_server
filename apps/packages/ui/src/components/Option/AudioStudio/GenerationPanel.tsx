import React from "react"
import { Button, Typography } from "antd"
import { Play } from "lucide-react"
import { useAudioStudioStore } from "@/store/audio-studio"
import { useAudioStudioGenerationActions } from "./useAudioStudioGenerationActions"

const { Text } = Typography

export const GenerationPanel: React.FC = () => {
  const activeWorkflow = useAudioStudioStore((state) => state.activeWorkflow)
  const {
    isPending,
    musicDisabledReason,
    queueMusicGeneration,
    queueSpeechGeneration,
    speechDisabledReason
  } = useAudioStudioGenerationActions()
  const isMusic = activeWorkflow === "music"
  const disabledReason = isMusic ? musicDisabledReason : speechDisabledReason

  const queueGeneration = () => {
    if (isMusic) {
      queueMusicGeneration(
        {
          prompt: "",
          lyrics: "",
          style: "",
          duration: 45
        },
        "ace_step"
      )
      return
    }

    queueSpeechGeneration()
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
        disabled={Boolean(disabledReason) || isPending}
        loading={isPending}
        title={disabledReason}
        onClick={queueGeneration}
      >
        {isMusic ? "Queue music generation" : "Queue speech generation"}
      </Button>
    </section>
  )
}
