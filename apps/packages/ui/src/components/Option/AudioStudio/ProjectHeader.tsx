import React from "react"
import { Button, Input, Typography } from "antd"
import { Save, Settings } from "lucide-react"
import { useAudioStudioStore } from "@/store/audio-studio"
import { useUpdateAudioStudioProject } from "@/hooks/useAudioStudioProjects"
import { getProjectRevisionId } from "./generationPayload"

const { Text } = Typography

export const ProjectHeader: React.FC = () => {
  const activeProject = useAudioStudioStore((state) => state.activeProject)
  const activeWorkflow = useAudioStudioStore((state) => state.activeWorkflow)
  const updateProjectLocal = useAudioStudioStore((state) => state.updateProjectLocal)
  const updateProjectMutation = useUpdateAudioStudioProject(
    activeProject?.project_id ?? null
  )
  const title = activeProject?.title ?? "Untitled Audio Project"
  const revisionId = getProjectRevisionId(activeProject)
  const workflowMismatch =
    activeProject !== null && activeProject.workflow !== activeWorkflow
  const saveDisabledReason = !activeProject
    ? "Select a project before saving."
    : workflowMismatch
      ? "Select a project for this workflow before saving."
    : !revisionId
      ? "Save requires a server-backed project revision."
      : undefined

  const saveProject = () => {
    if (!activeProject || workflowMismatch || !revisionId) return

    void updateProjectMutation.mutateAsync({
      title: activeProject.title,
      description: activeProject.description,
      settings: activeProject.settings,
      base_revision_id: revisionId
    })
  }

  return (
    <div className="flex flex-col gap-3 border-b border-border pb-3 lg:flex-row lg:items-center lg:justify-between">
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-2">
          <h1 className="m-0 text-xl font-semibold text-text">Audio Studio</h1>
          <Text className="text-xs uppercase tracking-wide text-text-muted">
            {activeWorkflow}
          </Text>
        </div>
        <Input
          aria-label="Project title"
          value={title}
          onChange={(event) => {
            if (activeProject) {
              updateProjectLocal(activeProject.project_id, {
                title: event.target.value
              })
            }
          }}
          className="mt-2 max-w-xl"
        />
      </div>
      <div className="flex items-center gap-2">
        <Button icon={<Settings className="h-4 w-4" />}>Settings</Button>
        <Button
          type="primary"
          icon={<Save className="h-4 w-4" />}
          disabled={Boolean(saveDisabledReason) || updateProjectMutation.isPending}
          loading={updateProjectMutation.isPending}
          title={saveDisabledReason}
          onClick={saveProject}
        >
          Save
        </Button>
      </div>
    </div>
  )
}
