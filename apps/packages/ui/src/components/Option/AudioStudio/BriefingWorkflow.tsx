import React from "react"
import { Button, Input, Typography } from "antd"
import { useAudioStudioGenerationActions } from "./useAudioStudioGenerationActions"

const { TextArea } = Input
const { Text } = Typography

export const BriefingWorkflow: React.FC = () => {
  const {
    isPending,
    queueSpeechGeneration,
    speechDisabledReason
  } = useAudioStudioGenerationActions()

  return (
    <section className="min-w-0 rounded-md border border-border bg-surface p-4">
      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_260px]">
        <div>
          <Text strong className="mb-2 block">
            Briefing outline
          </Text>
          <TextArea
            aria-label="Briefing outline"
            placeholder="Top story, context, implications, next actions..."
            autoSize={{ minRows: 14, maxRows: 24 }}
          />
        </div>
        <div className="space-y-3">
          <Text strong className="block">
            Source notes
          </Text>
          <TextArea
            aria-label="Source notes"
            placeholder="Paste source refs or analyst notes"
            autoSize={{ minRows: 8, maxRows: 12 }}
          />
          <Button
            block
            disabled={Boolean(speechDisabledReason) || isPending}
            loading={isPending}
            title={speechDisabledReason}
            onClick={queueSpeechGeneration}
          >
            Generate briefing sections
          </Button>
        </div>
      </div>
    </section>
  )
}
