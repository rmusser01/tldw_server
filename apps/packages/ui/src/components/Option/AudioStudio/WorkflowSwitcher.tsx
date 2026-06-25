import React from "react"
import { AUDIO_STUDIO_WORKFLOWS, type AudioStudioWorkflow } from "@/store/audio-studio"

type WorkflowSwitcherProps = {
  activeWorkflow: AudioStudioWorkflow
  onChange: (workflow: AudioStudioWorkflow) => void
}

const classNames = (...classes: Array<string | false | null | undefined>) =>
  classes.filter(Boolean).join(" ")

const PANEL_ID = "audio-studio-workflow-panel"

export const WorkflowSwitcher: React.FC<WorkflowSwitcherProps> = ({
  activeWorkflow,
  onChange
}) => {
  const handleKeyDown = (
    event: React.KeyboardEvent<HTMLButtonElement>,
    workflow: AudioStudioWorkflow
  ) => {
    const currentIndex = AUDIO_STUDIO_WORKFLOWS.findIndex(
      (candidate) => candidate.id === workflow
    )
    if (currentIndex < 0) return
    const lastIndex = AUDIO_STUDIO_WORKFLOWS.length - 1
    let nextIndex: number | null = null

    if (event.key === "ArrowRight" || event.key === "ArrowDown") {
      nextIndex = currentIndex === lastIndex ? 0 : currentIndex + 1
    } else if (event.key === "ArrowLeft" || event.key === "ArrowUp") {
      nextIndex = currentIndex === 0 ? lastIndex : currentIndex - 1
    } else if (event.key === "Home") {
      nextIndex = 0
    } else if (event.key === "End") {
      nextIndex = lastIndex
    }

    if (nextIndex === null) return
    event.preventDefault()
    onChange(AUDIO_STUDIO_WORKFLOWS[nextIndex].id)
  }

  return (
    <div
      className="grid gap-2 sm:grid-cols-2 lg:grid-cols-4"
      role="tablist"
      aria-label="Audio Studio workflows"
    >
      {AUDIO_STUDIO_WORKFLOWS.map((workflow) => {
        const selected = workflow.id === activeWorkflow
        return (
          <button
            key={workflow.id}
            type="button"
            id={`audio-studio-workflow-tab-${workflow.id}`}
            role="tab"
            aria-selected={selected}
            aria-controls={PANEL_ID}
            tabIndex={selected ? 0 : -1}
            onClick={() => onChange(workflow.id)}
            onKeyDown={(event) => handleKeyDown(event, workflow.id)}
            className={classNames(
              "min-h-[76px] rounded-md border px-3 py-2 text-left transition focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-focus",
              selected
                ? "border-primary bg-primary/10 text-text"
                : "border-border bg-surface hover:border-primary/60"
            )}
          >
            <span className="block text-sm font-semibold">{workflow.label}</span>
            <span className="mt-1 block text-xs leading-5 text-text-muted">
              {workflow.description}
            </span>
          </button>
        )
      })}
    </div>
  )
}
