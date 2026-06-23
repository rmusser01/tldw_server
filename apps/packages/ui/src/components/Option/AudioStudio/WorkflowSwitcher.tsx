import React from "react"
import { AUDIO_STUDIO_WORKFLOWS, type AudioStudioWorkflow } from "@/store/audio-studio"

type WorkflowSwitcherProps = {
  activeWorkflow: AudioStudioWorkflow
  onChange: (workflow: AudioStudioWorkflow) => void
}

const classNames = (...classes: Array<string | false | null | undefined>) =>
  classes.filter(Boolean).join(" ")

export const WorkflowSwitcher: React.FC<WorkflowSwitcherProps> = ({
  activeWorkflow,
  onChange
}) => {
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
            role="tab"
            aria-selected={selected}
            onClick={() => onChange(workflow.id)}
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
