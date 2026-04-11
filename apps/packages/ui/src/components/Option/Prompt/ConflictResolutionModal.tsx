import React from "react"
import { Alert, Modal, Tag } from "antd"
import { useTranslation } from "react-i18next"
import type { ConflictInfo, ConflictResolution } from "@/services/prompt-sync"

interface ConflictResolutionModalProps {
  open: boolean
  loading?: boolean
  conflictInfo: ConflictInfo | null
  onClose: () => void
  onResolve: (resolution: ConflictResolution) => void
}

const formatDateTime = (value: string | number | null | undefined): string => {
  if (!value) return "-"
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return "-"
  return date.toLocaleString()
}

const getLocalName = (info: ConflictInfo) =>
  info.localPrompt.name || info.localPrompt.title || "Untitled prompt"

const getServerName = (info: ConflictInfo) =>
  info.serverPrompt.name || "Untitled prompt"

const getLocalSystemPrompt = (info: ConflictInfo) => {
  const explicit = info.localPrompt.system_prompt || ""
  if (explicit.trim().length > 0) return explicit
  if (info.localPrompt.is_system) return info.localPrompt.content || ""
  return ""
}

const getLocalUserPrompt = (info: ConflictInfo) => {
  const explicit = info.localPrompt.user_prompt || ""
  if (explicit.trim().length > 0) return explicit
  if (!info.localPrompt.is_system) return info.localPrompt.content || ""
  return ""
}

const getServerSystemPrompt = (info: ConflictInfo) =>
  info.serverPrompt.system_prompt || ""

const getServerUserPrompt = (info: ConflictInfo) =>
  info.serverPrompt.user_prompt || ""

const renderPromptField = (
  label: string,
  value: string,
  isChanged: boolean
) => (
  <div className="space-y-1">
    <div className="flex items-center gap-2">
      <p className="text-xs font-medium text-text-muted uppercase tracking-wide">
        {label}
      </p>
      {isChanged ? (
        <Tag color="gold" className="text-[10px]">
          Changed
        </Tag>
      ) : null}
    </div>
    <pre className="max-h-40 overflow-auto rounded border border-border bg-surface2 p-2 text-xs whitespace-pre-wrap break-words">
      {value.trim().length > 0 ? value : "—"}
    </pre>
  </div>
)

export const ConflictResolutionModal: React.FC<ConflictResolutionModalProps> = ({
  open,
  loading = false,
  conflictInfo,
  onClose,
  onResolve
}) => {
  const { t } = useTranslation(["settings", "common"])

  const localName = conflictInfo ? getLocalName(conflictInfo) : ""
  const serverName = conflictInfo ? getServerName(conflictInfo) : ""
  const localSystem = conflictInfo ? getLocalSystemPrompt(conflictInfo) : ""
  const serverSystem = conflictInfo ? getServerSystemPrompt(conflictInfo) : ""
  const localUser = conflictInfo ? getLocalUserPrompt(conflictInfo) : ""
  const serverUser = conflictInfo ? getServerUserPrompt(conflictInfo) : ""

  const hasNamedDifferences = Boolean(
    conflictInfo &&
      (localName !== serverName ||
        localSystem !== serverSystem ||
        localUser !== serverUser)
  )

  return (
    <Modal
      title={t("managePrompts.sync.resolveConflict", {
        defaultValue: "Resolve conflict"
      })}
      open={open}
      onCancel={onClose}
      width={960}
      footer={
        <div className="flex flex-wrap items-center justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="inline-flex items-center justify-center rounded-md border border-border px-3 py-1.5 text-sm text-text hover:bg-surface2"
            disabled={loading}
          >
            {t("common:cancel", { defaultValue: "Cancel" })}
          </button>
          <button
            type="button"
            onClick={() => onResolve("keep_server")}
            className="inline-flex items-center justify-center rounded-md border border-border px-3 py-1.5 text-sm text-text hover:bg-surface2 disabled:opacity-50"
            disabled={loading || !conflictInfo}
          >
            {t("managePrompts.sync.keepServer", { defaultValue: "Keep server" })}
          </button>
          <button
            type="button"
            onClick={() => onResolve("keep_both")}
            className="inline-flex items-center justify-center rounded-md border border-primary/40 px-3 py-1.5 text-sm text-primary hover:bg-primary/10 disabled:opacity-50"
            disabled={loading || !conflictInfo}
          >
            {t("managePrompts.sync.keepBoth", { defaultValue: "Keep both" })}
          </button>
          <button
            type="button"
            onClick={() => onResolve("keep_local")}
            className="inline-flex items-center justify-center rounded-md border border-transparent bg-primary px-3 py-1.5 text-sm text-white hover:bg-primaryStrong disabled:opacity-50"
            disabled={loading || !conflictInfo}
          >
            {t("managePrompts.sync.keepMine", { defaultValue: "Keep mine" })}
          </button>
        </div>
      }
    >
      {loading && !conflictInfo ? (
        <p className="text-sm text-text-muted">
          {t("managePrompts.sync.loadingConflict", {
            defaultValue: "Loading conflict details..."
          })}
        </p>
      ) : null}

      {!loading && !conflictInfo ? (
        <Alert
          type="warning"
          showIcon
          message={t("managePrompts.sync.conflictUnavailable", {
            defaultValue: "Conflict details unavailable"
          })}
          description={t("managePrompts.sync.conflictUnavailableDesc", {
            defaultValue:
              "We couldn't retrieve local and server versions for comparison."
          })}
        />
      ) : null}

      {conflictInfo ? (
        <div className="space-y-3">
          <Alert
            type={hasNamedDifferences ? "warning" : "info"}
            showIcon
            message={
              hasNamedDifferences
                ? t("managePrompts.sync.conflictDiffDetected", {
                    defaultValue: "Differences detected between local and server versions."
                  })
                : t("managePrompts.sync.conflictNoTextDiff", {
                    defaultValue: "No text differences detected. This conflict may be metadata-only."
                  })
            }
          />

          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <section className="rounded border border-border p-3 space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-medium">
                  {t("managePrompts.sync.localVersion", {
                    defaultValue: "Local version"
                  })}
                </h4>
                <span className="text-xs text-text-muted">
                  {formatDateTime(conflictInfo.localUpdatedAt)}
                </span>
              </div>
              {renderPromptField(
                t("managePrompts.columns.title", { defaultValue: "Title" }),
                localName,
                localName !== serverName
              )}
              {renderPromptField(
                t("managePrompts.form.systemPrompt.shortLabel", {
                  defaultValue: "System"
                }),
                localSystem,
                localSystem !== serverSystem
              )}
              {renderPromptField(
                t("managePrompts.form.userPrompt.shortLabel", {
                  defaultValue: "User"
                }),
                localUser,
                localUser !== serverUser
              )}
            </section>

            <section className="rounded border border-border p-3 space-y-3">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-medium">
                  {t("managePrompts.sync.serverVersion", {
                    defaultValue: "Server version"
                  })}
                </h4>
                <span className="text-xs text-text-muted">
                  {formatDateTime(conflictInfo.serverUpdatedAt)}
                </span>
              </div>
              {renderPromptField(
                t("managePrompts.columns.title", { defaultValue: "Title" }),
                serverName,
                localName !== serverName
              )}
              {renderPromptField(
                t("managePrompts.form.systemPrompt.shortLabel", {
                  defaultValue: "System"
                }),
                serverSystem,
                localSystem !== serverSystem
              )}
              {renderPromptField(
                t("managePrompts.form.userPrompt.shortLabel", {
                  defaultValue: "User"
                }),
                serverUser,
                localUser !== serverUser
              )}
            </section>
          </div>
        </div>
      ) : null}
    </Modal>
  )
}

export default ConflictResolutionModal
