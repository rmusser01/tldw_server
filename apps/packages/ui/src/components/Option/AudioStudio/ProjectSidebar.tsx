import { useAudioStudioSession } from "./AudioStudioSession"
import React from "react"
import { useNavigate } from "react-router-dom"
import { Button, Empty, Typography } from "antd"
import { Plus } from "lucide-react"
import {
  AUDIO_STUDIO_WORKFLOWS,
  useAudioStudioStore
} from "@/store/audio-studio"
import { useCreateAudioStudioProject } from "@/hooks/useAudioStudioProjects"

const { Text } = Typography

export const ProjectSidebar: React.FC = () => {
  const session = useAudioStudioSession()
  const navigate = useNavigate()
  const hasDrafts = useAudioStudioStore((state) => Object.keys(state.dirtyProjectIds).length > 0)
  const pending = useAudioStudioStore((state) => Object.keys(state.jobIntents).length > 0)
  const projects = useAudioStudioStore((state) => state.projects)
  const activeProjectId = useAudioStudioStore((state) => state.activeProjectId)
  const activeWorkflow = useAudioStudioStore((state) => state.activeWorkflow)
  const createProjectMutation = useCreateAudioStudioProject()
  const visibleProjects = projects.filter(
    (project) => project.workflow === activeWorkflow
  )

  const createProject = () => {
    if (hasDrafts || pending || session?.readOnlyReason) return
    const workflowLabel =
      AUDIO_STUDIO_WORKFLOWS.find((item) => item.id === activeWorkflow)?.label ??
      "Audio"

    void createProjectMutation.mutateAsync({
      title: `Untitled ${workflowLabel} Project`,
      workflow: activeWorkflow
    }).then((project) => { if (project) navigate(`?workflow=${project.workflow}&project=${encodeURIComponent(project.project_id)}`) }).catch(() => {})
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
          disabled={createProjectMutation.isPending || hasDrafts || pending || Boolean(session?.readOnlyReason)}
          title={hasDrafts ? "Save your project drafts before creating another project." : pending ? "Resolve pending project work first." : undefined}
          loading={createProjectMutation.isPending}
          onClick={createProject}
        />
      </div>
      {createProjectMutation.isError ? <p role="alert">Project could not be created. Check your connection and try again.</p> : null}
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
              onClick={() => { const state = useAudioStudioStore.getState(); if (!Object.keys(state.dirtyProjectIds).length) state.setActiveProjectId(project.project_id); navigate(`?workflow=${project.workflow}&project=${encodeURIComponent(project.project_id)}`); }}
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
