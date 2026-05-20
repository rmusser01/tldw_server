import React from "react"
import { Button, Input } from "antd"

import {
  describeRolePlaySetupPreview,
  isRolePlayRelevantBundle,
  type StartupTemplateBundle
} from "./startup-template-bundles"

type SavedRolePlaySetupsPanelProps = {
  setups: StartupTemplateBundle[]
  draftName: string
  nameFallback: string
  onDraftNameChange: (name: string) => void
  onSaveCurrent: () => void
  onPreviewSetup: (id: string) => void
  onApplySetup: (setup: StartupTemplateBundle) => void | Promise<void>
  onRenameSetup: (id: string, name: string) => void
  onDeleteSetup: (id: string) => void
  t: (...args: any[]) => any
}

export const SavedRolePlaySetupsPanel: React.FC<SavedRolePlaySetupsPanelProps> = ({
  setups,
  draftName,
  nameFallback,
  onDraftNameChange,
  onSaveCurrent,
  onPreviewSetup,
  onApplySetup,
  onRenameSetup,
  onDeleteSetup,
  t
}) => {
  const rolePlaySetups = React.useMemo(
    () => setups.filter((setup) => isRolePlayRelevantBundle(setup)),
    [setups]
  )
  const [renameDrafts, setRenameDrafts] = React.useState<Record<string, string>>({})
  const [pendingDeleteId, setPendingDeleteId] = React.useState<string | null>(null)
  const confirmDeleteButtonRefs = React.useRef<Record<string, HTMLElement | null>>({})

  const getRenameDraft = React.useCallback(
    (setup: StartupTemplateBundle) => renameDrafts[setup.id] ?? setup.name,
    [renameDrafts]
  )

  const updateRenameDraft = React.useCallback((id: string, name: string) => {
    setRenameDrafts((current) => ({
      ...current,
      [id]: name
    }))
  }, [])

  React.useEffect(() => {
    if (!pendingDeleteId) return
    if (rolePlaySetups.some((setup) => setup.id === pendingDeleteId)) return
    setPendingDeleteId(null)
  }, [pendingDeleteId, rolePlaySetups])

  React.useEffect(() => {
    if (!pendingDeleteId) return
    confirmDeleteButtonRefs.current[pendingDeleteId]?.focus()
  }, [pendingDeleteId])

  const confirmDelete = React.useCallback(
    (id: string) => {
      onDeleteSetup(id)
      setPendingDeleteId(null)
    },
    [onDeleteSetup]
  )

  return (
    <section
      aria-label={t(
        "playground:composer.savedRolePlaySetups",
        "Saved role-play setups"
      )}
      className="space-y-3 rounded-md border border-border bg-surface p-3">
      <div className="space-y-1">
        <h3 className="text-sm font-semibold text-text">
          {t(
            "playground:composer.savedRolePlaySetups",
            "Saved role-play setups"
          )}
        </h3>
        <p className="text-xs text-text-muted">
          {t(
            "playground:composer.savedRolePlaySetupsHint",
            "Save and reuse character, behavior, scene, generation, and pinned context."
          )}
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Input
          aria-label={t(
            "playground:composer.savedRolePlaySetupName",
            "Saved setup name"
          )}
          value={draftName}
          onChange={(event) => onDraftNameChange(event.target.value)}
          placeholder={nameFallback}
          className="min-w-[180px] flex-1"
        />
        <Button onClick={onSaveCurrent}>
          {t("playground:composer.saveRolePlaySetup", "Save setup")}
        </Button>
      </div>

      {rolePlaySetups.length === 0 ? (
        <p className="text-xs text-text-muted">
          {t(
            "playground:composer.noSavedRolePlaySetups",
            "No saved role-play setups yet."
          )}
        </p>
      ) : (
        <ul
          aria-label={t(
            "playground:composer.savedRolePlaySetupList",
            "Saved role-play setup list"
          )}
          className="space-y-2">
          {rolePlaySetups.map((setup) => {
            const preview = describeRolePlaySetupPreview(setup)
            const renameDraft = getRenameDraft(setup)
            const deletePending = pendingDeleteId === setup.id
            return (
              <li
                key={setup.id}
                aria-label={t(
                  "playground:composer.savedRolePlaySetupListItem",
                  "{{name}} role-play setup",
                  { name: setup.name }
                )}
                className="space-y-2 rounded-md border border-border bg-surface2 p-2">
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div className="min-w-0">
                    <div className="truncate text-sm font-medium text-text">
                      {setup.name}
                    </div>
                    <div className="mt-1 flex flex-wrap gap-1 text-[11px] text-text-muted">
                      <span>{preview.identity}</span>
                      <span aria-hidden="true">/</span>
                      <span>{preview.behavior}</span>
                      <span aria-hidden="true">/</span>
                      <span>{preview.generation}</span>
                      <span aria-hidden="true">/</span>
                      <span>{preview.context}</span>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    <Button
                      size="small"
                      aria-label={t(
                        "playground:composer.previewRolePlaySetup",
                        "Preview {{name}}",
                        { name: setup.name }
                      )}
                      onClick={() => onPreviewSetup(setup.id)}>
                      {t("common:preview", "Preview")}
                    </Button>
                    <Button
                      size="small"
                      type="primary"
                      aria-label={t(
                        "playground:composer.applyRolePlaySetup",
                        "Apply {{name}}",
                        { name: setup.name }
                      )}
                      onClick={() => void onApplySetup(setup)}>
                      {t("common:apply", "Apply")}
                    </Button>
                    <Button
                      size="small"
                      danger
                      aria-label={t(
                        "playground:composer.deleteRolePlaySetup",
                        "Delete {{name}}",
                        { name: setup.name }
                      )}
                      onClick={() => setPendingDeleteId(setup.id)}>
                      {t("common:delete", "Delete")}
                    </Button>
                  </div>
                </div>
                {deletePending ? (
                  <div
                    role="alert"
                    className="flex flex-wrap items-center justify-between gap-2 rounded-md border border-danger/40 bg-danger/10 p-2 text-xs text-danger">
                    <span>
                      {t(
                        "playground:composer.confirmDeleteRolePlaySetup",
                        "Delete {{name}}?",
                        { name: setup.name }
                      )}
                    </span>
                    <div className="flex flex-wrap gap-1">
                      <Button
                        ref={(node) => {
                          confirmDeleteButtonRefs.current[setup.id] = node
                        }}
                        size="small"
                        danger
                        aria-label={t(
                          "playground:composer.confirmDeleteRolePlaySetupAction",
                          "Confirm delete {{name}}",
                          { name: setup.name }
                        )}
                        onClick={() => confirmDelete(setup.id)}>
                        {t("common:confirm", "Confirm")}
                      </Button>
                      <Button
                        size="small"
                        aria-label={t(
                          "playground:composer.cancelDeleteRolePlaySetup",
                          "Cancel delete {{name}}",
                          { name: setup.name }
                        )}
                        onClick={() => setPendingDeleteId(null)}>
                        {t("common:cancel", "Cancel")}
                      </Button>
                    </div>
                  </div>
                ) : null}
                <div className="flex flex-wrap items-center gap-2">
                  <Input
                    aria-label={t(
                      "playground:composer.renameRolePlaySetupField",
                      "Rename field for {{name}}",
                      { name: setup.name }
                    )}
                    value={renameDraft}
                    onChange={(event) =>
                      updateRenameDraft(setup.id, event.target.value)
                    }
                    className="min-w-[180px] flex-1"
                  />
                  <Button
                    size="small"
                    aria-label={t(
                      "playground:composer.renameRolePlaySetup",
                      "Rename {{name}}",
                      { name: setup.name }
                    )}
                    onClick={() => onRenameSetup(setup.id, renameDraft)}>
                    {t("common:rename", "Rename")}
                  </Button>
                </div>
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}
