import React from "react"
import { Modal } from "antd"
import { ModalFooter } from "@/components/ui/layout"
import {
  describeRolePlaySetupPreview,
  isRolePlayRelevantBundle,
  type StartupTemplateBundle
} from "./startup-template-bundles"
import type { ParameterPreset } from "./ParameterPresets"
import { toText } from "./hooks"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface PlaygroundStartupTemplateModalProps {
  preview: StartupTemplateBundle | null
  onClose: () => void
  onDelete: (id: string) => void
  onApply: () => void
  promptDescription: string | null
  promptResolution: { source?: string } | null
  preset: ParameterPreset | undefined
  t: (key: string, defaultValue?: string, options?: any) => any
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export const PlaygroundStartupTemplateModal: React.FC<PlaygroundStartupTemplateModalProps> =
  React.memo(function PlaygroundStartupTemplateModal(props) {
    const {
      preview,
      onClose,
      onDelete,
      onApply,
      promptDescription,
      promptResolution,
      preset,
      t
    } = props
    const rolePlayPreview =
      preview && isRolePlayRelevantBundle(preview)
        ? describeRolePlaySetupPreview(preview)
        : null

    return (
      <Modal
        open={Boolean(preview)}
        onCancel={onClose}
        title={t(
          "playground:composer.startupTemplatePreviewTitle",
          "Launch startup template"
        )}
        destroyOnHidden
        data-testid="startup-template-preview-modal"
        footer={
          <ModalFooter
            align="between"
            data-testid="startup-template-preview-modal-footer"
            leftActions={[
              {
                label: t(
                  "playground:composer.startupTemplateDelete",
                  "Delete template"
                ),
                onClick: () => {
                  if (!preview) return
                  onDelete(preview.id)
                },
                disabled: !preview,
                danger: true
              }
            ]}
            onCancel={onClose}
            cancelLabel={t("common:cancel", "Cancel")}
            primaryAction={{
              label: t(
                "playground:composer.startupTemplateApply",
                "Apply template"
              ),
              onClick: onApply,
              disabled: !preview
            }}
          />
        }
      >
        {preview ? (
          <div className="space-y-3">
            <p className="text-sm text-text-muted">
              {t(
                "playground:composer.startupTemplatePreviewBody",
                "Review active context that will be applied before your next send."
              )}
            </p>
            {rolePlayPreview ? (
              <div className="grid gap-2 text-xs text-text sm:grid-cols-2">
                <div className="rounded-md border border-border bg-surface px-2 py-2">
                  <div className="text-[10px] font-semibold uppercase tracking-wide text-text-muted">
                    {t("playground:composer.context.character", "Character")}
                  </div>
                  <div className="mt-1">{rolePlayPreview.identity}</div>
                </div>
                <div className="rounded-md border border-border bg-surface px-2 py-2">
                  <div className="text-[10px] font-semibold uppercase tracking-wide text-text-muted">
                    {t("playground:composer.context.behavior", "Behavior")}
                  </div>
                  <div className="mt-1">{rolePlayPreview.behavior}</div>
                </div>
                <div className="rounded-md border border-border bg-surface px-2 py-2">
                  <div className="text-[10px] font-semibold uppercase tracking-wide text-text-muted">
                    {t("playground:composer.context.scene", "Scene")}
                  </div>
                  <div className="mt-1">{rolePlayPreview.scene}</div>
                </div>
                <div className="rounded-md border border-border bg-surface px-2 py-2">
                  <div className="text-[10px] font-semibold uppercase tracking-wide text-text-muted">
                    {t(
                      "playground:composer.context.generationStyle",
                      "Generation style"
                    )}
                  </div>
                  <div className="mt-1">{rolePlayPreview.generation}</div>
                </div>
              </div>
            ) : (
              <div className="grid gap-2 text-xs text-text sm:grid-cols-2">
                <div className="rounded-md border border-border bg-surface px-2 py-2">
                  <div className="text-[10px] font-semibold uppercase tracking-wide text-text-muted">
                    {t("playground:composer.context.model", "Model")}
                  </div>
                  <div className="mt-1">
                    {preview.selectedModel || t("common:none", "None")}
                  </div>
                </div>
                <div className="rounded-md border border-border bg-surface px-2 py-2">
                  <div className="text-[10px] font-semibold uppercase tracking-wide text-text-muted">
                    {t("playground:composer.context.prompt", "Prompt")}
                  </div>
                  <div className="mt-1">{promptDescription}</div>
                </div>
                <div className="rounded-md border border-border bg-surface px-2 py-2">
                  <div className="text-[10px] font-semibold uppercase tracking-wide text-text-muted">
                    {t("playground:composer.context.preset", "Preset")}
                  </div>
                  <div className="mt-1">
                    {preset
                      ? t(
                          `playground:presets.${preset.key}.label`,
                          preset.label
                        )
                      : t("common:none", "None")}
                  </div>
                </div>
                <div className="rounded-md border border-border bg-surface px-2 py-2">
                  <div className="text-[10px] font-semibold uppercase tracking-wide text-text-muted">
                    {t("playground:composer.context.character", "Character")}
                  </div>
                  <div className="mt-1">
                    {preview.character?.name || t("common:none", "None")}
                  </div>
                </div>
              </div>
            )}
            <div className="rounded-md border border-border bg-surface px-2 py-2 text-xs text-text">
              <div className="text-[10px] font-semibold uppercase tracking-wide text-text-muted">
                {t("playground:composer.context.pinnedSources", "Pinned")}
              </div>
              <div className="mt-1">
                {rolePlayPreview
                  ? rolePlayPreview.context
                  : toText(
                      t("playground:composer.context.pinnedCount", {
                        defaultValue: "{{count}} sources",
                        count: preview.ragPinnedResults.length
                      } as any)
                    )}
              </div>
              {promptResolution?.source === "prompt-studio" && (
                <div className="mt-1 text-[11px] text-text-muted">
                  {t(
                    "playground:composer.startupTemplatePromptStudioApplied",
                    "Prompt Studio mapping will be reapplied if available."
                  )}
                </div>
              )}
            </div>
          </div>
        ) : null}
      </Modal>
    )
  })
