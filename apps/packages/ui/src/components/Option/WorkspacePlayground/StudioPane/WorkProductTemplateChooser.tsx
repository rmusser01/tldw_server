import React from "react"
import { Tooltip } from "antd"
import { FileText } from "lucide-react"
import type { WorkProductTemplateId } from "@/workspace-templates/types"
import {
  WORK_PRODUCT_TEMPLATES,
  type WorkProductTemplate
} from "@/workspace-templates/work-product-templates"

type WorkProductTemplateChooserProps = {
  selectedTemplateId: WorkProductTemplateId
  selectedSourceCount: number
  onSelectTemplate: (templateId: WorkProductTemplateId) => void
  disabled?: boolean
}

const isActionableTemplate = (template: WorkProductTemplate) =>
  template.id === "executive_brief"

export const WorkProductTemplateChooser: React.FC<
  WorkProductTemplateChooserProps
> = ({
  selectedTemplateId,
  selectedSourceCount,
  onSelectTemplate,
  disabled = false
}) => {
  return (
    <section aria-label="Work product templates" className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <p className="text-xs font-semibold uppercase tracking-wide text-text-muted">
          Work Products
        </p>
        <span className="text-[11px] text-text-muted">
          {selectedSourceCount} selected
        </span>
      </div>
      <div className="grid gap-2">
        {WORK_PRODUCT_TEMPLATES.map((template) => {
          const sourceRequirementMet =
            selectedSourceCount >= template.minSelectedSources
          const actionable = isActionableTemplate(template)
          const unavailable = disabled || !actionable || !sourceRequirementMet
          const selected = selectedTemplateId === template.id
          let unavailableReason = template.description
          if (disabled) {
            unavailableReason = "Generating..."
          } else if (!sourceRequirementMet) {
            unavailableReason = `Requires ${template.minSelectedSources} selected source${
              template.minSelectedSources === 1 ? "" : "s"
            }.`
          } else if (!actionable) {
            unavailableReason = "Planned"
          }

          return (
            <Tooltip
              key={template.id}
              title={unavailable ? unavailableReason : template.description}
            >
              <span className="block">
                <button
                  type="button"
                  disabled={unavailable}
                  aria-disabled={unavailable ? "true" : undefined}
                  aria-pressed={selected}
                  aria-label={template.label}
                  onClick={() => {
                    if (unavailable) return
                    onSelectTemplate(template.id)
                  }}
                  className={`flex w-full items-start gap-2 rounded-md border px-3 py-2 text-left transition-colors ${
                    selected
                      ? "border-primary/60 bg-primary/5"
                      : "border-border bg-surface2/30"
                  } ${
                    unavailable
                      ? "cursor-not-allowed opacity-65"
                      : "hover:border-primary/50 hover:bg-primary/5"
                  }`}
                >
                  <FileText className="mt-0.5 h-4 w-4 flex-none text-text-muted" />
                  <span className="min-w-0 flex-1">
                    <span className="flex flex-wrap items-center gap-1.5">
                      <span className="text-xs font-medium text-text">
                        {template.label}
                      </span>
                      {!actionable && (
                        <span className="rounded border border-border bg-surface px-1.5 py-0.5 text-[10px] font-medium text-text-muted">
                          Planned
                        </span>
                      )}
                    </span>
                    <span className="mt-1 block text-[11px] leading-snug text-text-muted">
                      {template.description}
                    </span>
                  </span>
                </button>
              </span>
            </Tooltip>
          )
        })}
      </div>
    </section>
  )
}
