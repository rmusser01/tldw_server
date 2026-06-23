import React from "react"
import { Button, Empty, Typography } from "antd"
import { Plus } from "lucide-react"
import {
  AUDIO_STUDIO_WORKFLOWS,
  useAudioStudioStore,
  type AudioStudioProject
} from "@/store/audio-studio"

const { Text } = Typography

const createDraftProject = (workflow: AudioStudioProject["workflow"]): AudioStudioProject => ({
  project_id: crypto.randomUUID(),
  title: `Untitled ${AUDIO_STUDIO_WORKFLOWS.find((item) => item.id === workflow)?.label ?? "Audio"} Project`,
  workflow,
  status: "draft",
  revision_id: "local-draft",
  updated_at: new Date().toISOString(),
  sections: [],
  tracks: [],
  clips: [],
  settings: {}
})

export const ProjectSidebar: React.FC = () => {
  const projects = useAudioStudioStore((state) => state.projects)
  const activeProjectId = useAudioStudioStore((state) => state.activeProjectId)
  const activeWorkflow = useAudioStudioStore((state) => state.activeWorkflow)
  const setProjects = useAudioStudioStore((state) => state.setProjects)
  const setActiveProjectId = useAudioStudioStore((state) => state.setActiveProjectId)
  const visibleProjects = projects.filter(
    (project) => project.workflow === activeWorkflow
  )

  const createProject = () => {
    const project = createDraftProject(activeWorkflow)
    setProjects([...projects, project])
    setActiveProjectId(project.project_id)
  }

  return (
    <aside className="min-h-[360px] rounded-md border border-border bg-surface p-3">
      <div className="mb-3 flex items-center justify-between gap-2">
        <Text strong>Projects</Text>
        <Button
          size="small"
          type="text"
          aria-label="New Audio Studio project"
          icon={<Plus className="h-4 w-4" />}
          onClick={createProject}
        />
      </div>
      {visibleProjects.length === 0 ? (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description="No projects for this workflow"
        />
      ) : (
        <div className="space-y-2">
          {visibleProjects.map((project) => (
            <button
              key={project.project_id}
              type="button"
              onClick={() => setActiveProjectId(project.project_id)}
              className={`w-full rounded-md border px-3 py-2 text-left text-sm ${
                project.project_id === activeProjectId
                  ? "border-primary bg-primary/10"
                  : "border-border bg-surface2 hover:border-primary/60"
              }`}
            >
              <span className="block truncate font-medium">{project.title}</span>
              <span className="text-xs text-text-muted">{project.status}</span>
            </button>
          ))}
        </div>
      )}
    </aside>
  )
}
