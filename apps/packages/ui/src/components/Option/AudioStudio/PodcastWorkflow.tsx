import React from "react"
import { Button, Input, Typography } from "antd"
import { useUpsertAudioStudioSection } from "@/hooks/useAudioStudioProjects"
import { useAudioStudioStore } from "@/store/audio-studio"
import { getProjectRevisionId } from "./generationPayload"
import { useAudioStudioGenerationActions } from "./useAudioStudioGenerationActions"

const { TextArea } = Input
const { Text } = Typography
const PODCAST_SECTION_ID = "section_podcast_script"

export const PodcastWorkflow: React.FC = () => {
  const activeProject = useAudioStudioStore((state) => state.activeProject)
  const [script, setScript] = React.useState("")
  const [hostSpeaker, setHostSpeaker] = React.useState("")
  const [guestSpeaker, setGuestSpeaker] = React.useState("")
  const upsertSection = useUpsertAudioStudioSection(activeProject?.project_id ?? null)
  const {
    isPending,
    queueSpeechGenerationForSection,
    speechDraftDisabledReason,
    speechDisabledReason
  } = useAudioStudioGenerationActions()
  const revisionId = getProjectRevisionId(activeProject)
  const disabledReason =
    speechDraftDisabledReason ??
    (!script.trim() ? "Enter a podcast script before queuing speech." : undefined)

  const handleGenerate = async () => {
    if (!activeProject || !revisionId || disabledReason) return
    const section = await upsertSection.mutateAsync({
      sectionId: PODCAST_SECTION_ID,
      payload: {
        base_revision_id: revisionId,
        title: "Podcast script",
        body_text: script,
        order_index: 0,
        settings: {
          hostSpeaker,
          guestSpeaker
        }
      }
    })
    queueSpeechGenerationForSection(
      PODCAST_SECTION_ID,
      section.current_revision_id ?? revisionId
    )
  }

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
            value={script}
            onChange={(event) => setScript(event.target.value)}
            autoSize={{ minRows: 14, maxRows: 24 }}
          />
        </div>
        <div className="space-y-3">
          <Text strong className="block">
            Speakers
          </Text>
          <Input
            aria-label="Host speaker"
            placeholder="Host voice"
            value={hostSpeaker}
            onChange={(event) => setHostSpeaker(event.target.value)}
          />
          <Input
            aria-label="Guest speaker"
            placeholder="Guest voice"
            value={guestSpeaker}
            onChange={(event) => setGuestSpeaker(event.target.value)}
          />
          <Button
            block
            disabled={Boolean(disabledReason) || isPending || upsertSection.isPending}
            loading={isPending || upsertSection.isPending}
            title={disabledReason ?? speechDisabledReason}
            onClick={() => void handleGenerate()}
          >
            Generate segment speech
          </Button>
        </div>
      </div>
    </section>
  )
}
