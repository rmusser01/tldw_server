import React from "react"
import { Button, Input, Typography } from "antd"
import { Save, Settings } from "lucide-react"
import { useAudioStudioStore } from "@/store/audio-studio"

const { Text } = Typography

export const ProjectHeader: React.FC = () => {
  const activeProject = useAudioStudioStore((state) => state.activeProject)
  const activeWorkflow = useAudioStudioStore((state) => state.activeWorkflow)
  const updateProjectLocal = useAudioStudioStore((state) => state.updateProjectLocal)
  const title = activeProject?.title ?? "Untitled Audio Project"

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
        <Button type="primary" icon={<Save className="h-4 w-4" />}>
          Save
        </Button>
      </div>
    </div>
  )
}
