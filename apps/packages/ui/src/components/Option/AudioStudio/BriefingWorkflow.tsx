import React from "react"
import { Button, Input, Typography } from "antd"
import { useUpsertAudioStudioSection } from "@/hooks/useAudioStudioProjects"
import { useAudioStudioStore } from "@/store/audio-studio"
import { getProjectRevisionId } from "./generationPayload"
import { useAudioStudioGenerationActions } from "./useAudioStudioGenerationActions"

const { TextArea } = Input
const { Text } = Typography
const BRIEFING_SECTION_ID = "section_briefing_outline"

export const BriefingWorkflow: React.FC = () => {
  const activeProject = useAudioStudioStore((state) => state.activeProject)
  const [outline, setOutline] = React.useState("")
  const [sourceNotes, setSourceNotes] = React.useState("")
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
    (!outline.trim() ? "Enter a briefing outline before queuing speech." : undefined)

  const handleGenerate = async () => {
    if (!activeProject || !revisionId || disabledReason) return
    const section = await upsertSection.mutateAsync({
      sectionId: BRIEFING_SECTION_ID,
      payload: {
        base_revision_id: revisionId,
        title: "Briefing outline",
        body_text: outline,
        order_index: 0,
        settings: {
          sourceNotes
        }
      }
    })
    queueSpeechGenerationForSection(
      BRIEFING_SECTION_ID,
      section.current_revision_id ?? revisionId
    )
  }

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
            value={outline}
            onChange={(event) => setOutline(event.target.value)}
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
            value={sourceNotes}
            onChange={(event) => setSourceNotes(event.target.value)}
            autoSize={{ minRows: 8, maxRows: 12 }}
          />
          <Button
            block
            disabled={Boolean(disabledReason) || isPending || upsertSection.isPending}
            loading={isPending || upsertSection.isPending}
            title={disabledReason ?? speechDisabledReason}
            onClick={() => void handleGenerate()}
          >
            Generate briefing sections
          </Button>
        </div>
      </div>
    </section>
  )
}
