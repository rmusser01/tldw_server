import React, { useEffect } from "react"
import { Typography } from "antd"
import { PageShell } from "@/components/Common/PageShell"
import { useLocation } from "react-router-dom"
import { useAudioStudioProjects } from "@/hooks/useAudioStudioProjects"
import {
  AUDIO_STUDIO_WORKFLOWS,
  useAudioStudioStore,
  type AudioStudioWorkflow
} from "@/store/audio-studio"
import { BriefingWorkflow } from "./BriefingWorkflow"
import { GenerationPanel } from "./GenerationPanel"
import { MigrationBanner } from "./MigrationBanner"
import { MusicWorkflow } from "./MusicWorkflow"
import { NarrationWorkflow } from "./NarrationWorkflow"
import { PodcastWorkflow } from "./PodcastWorkflow"
import { ProjectHeader } from "./ProjectHeader"
import { ProjectSidebar } from "./ProjectSidebar"
import { RenderExportPanel } from "./RenderExportPanel"
import { WorkflowSwitcher } from "./WorkflowSwitcher"

const { Text } = Typography

const WORKFLOW_IDS = new Set<AudioStudioWorkflow>(
  AUDIO_STUDIO_WORKFLOWS.map((workflow) => workflow.id)
)

const getWorkflowFromSearch = (search: string): AudioStudioWorkflow | null => {
  const workflow = new URLSearchParams(search).get("workflow")
  return WORKFLOW_IDS.has(workflow as AudioStudioWorkflow)
    ? (workflow as AudioStudioWorkflow)
    : null
}

const WorkflowEditor: React.FC<{ workflow: AudioStudioWorkflow }> = ({
  workflow
}) => {
  if (workflow === "podcast") return <PodcastWorkflow />
  if (workflow === "briefing") return <BriefingWorkflow />
  if (workflow === "music") return <MusicWorkflow />
  return <NarrationWorkflow />
}

export const AudioStudioPage: React.FC = () => {
  const location = useLocation()
  const activeWorkflow = useAudioStudioStore((state) => state.activeWorkflow)
  const setActiveWorkflow = useAudioStudioStore((state) => state.setActiveWorkflow)
  const projectsQuery = useAudioStudioProjects({
    workflow: activeWorkflow,
    includeArchived: false
  })

  useEffect(() => {
    const workflow = getWorkflowFromSearch(location.search)
    if (workflow && workflow !== activeWorkflow) {
      setActiveWorkflow(workflow)
    }
  }, [activeWorkflow, location.search, setActiveWorkflow])

  return (
    <PageShell maxWidthClassName="max-w-7xl" className="py-4">
      <div className="space-y-4">
        <ProjectHeader />
        <WorkflowSwitcher
          activeWorkflow={activeWorkflow}
          onChange={setActiveWorkflow}
        />
        {projectsQuery.isLoading ? (
          <Text type="secondary" className="block text-xs">
            Loading Audio Studio projects...
          </Text>
        ) : null}
        {projectsQuery.isError ? (
          <Text type="danger" className="block text-xs">
            Audio Studio projects could not load.
          </Text>
        ) : null}
        <MigrationBanner />
        <div className="grid gap-4 lg:grid-cols-[280px_minmax(0,1fr)_320px]">
          <ProjectSidebar />
          <WorkflowEditor workflow={activeWorkflow} />
          <div className="space-y-4">
            <GenerationPanel />
            <RenderExportPanel />
          </div>
        </div>
      </div>
    </PageShell>
  )
}
