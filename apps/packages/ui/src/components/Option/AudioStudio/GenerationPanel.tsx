import React from "react"
import { Button, Typography } from "antd"
import { Play } from "lucide-react"
import { useAudioStudioStore } from "@/store/audio-studio"

const { Text } = Typography

export const GenerationPanel: React.FC = () => {
  const activeWorkflow = useAudioStudioStore((state) => state.activeWorkflow)
  const activeProject = useAudioStudioStore((state) => state.activeProject)
  const isMusic = activeWorkflow === "music"

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
        disabled={!activeProject}
      >
        {isMusic ? "Queue music generation" : "Queue speech generation"}
      </Button>
    </section>
  )
}
