import React from "react"
import { Button, Input, Typography } from "antd"
import { useAudioStudioGenerationActions } from "./useAudioStudioGenerationActions"

const { TextArea } = Input
const { Text } = Typography

export const PodcastWorkflow: React.FC = () => {
  const {
    isPending,
    queueSpeechGeneration,
    speechDisabledReason
  } = useAudioStudioGenerationActions()

  return (
    <section className="min-w-0 rounded-md border border-border bg-surface p-4">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_220px]">
        <div>
          <Text strong className="mb-2 block">
            Podcast script
          </Text>
          <TextArea
            aria-label="Podcast script"
            placeholder="Host: Welcome to the episode..."
            autoSize={{ minRows: 14, maxRows: 24 }}
          />
        </div>
        <div className="space-y-3">
          <Text strong className="block">
            Speakers
          </Text>
          <Input aria-label="Host speaker" placeholder="Host voice" />
          <Input aria-label="Guest speaker" placeholder="Guest voice" />
          <Button
            block
            disabled={Boolean(speechDisabledReason) || isPending}
            loading={isPending}
            title={speechDisabledReason}
            onClick={queueSpeechGeneration}
          >
            Generate segment speech
          </Button>
        </div>
      </div>
    </section>
  )
}
