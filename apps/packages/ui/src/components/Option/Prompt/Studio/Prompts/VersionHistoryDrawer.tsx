import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Drawer, Timeline, Skeleton, notification, Tag, Empty } from "antd"
import { History, Undo2, CheckCircle2 } from "lucide-react"
import React from "react"
import { useTranslation } from "react-i18next"
import {
  getPromptHistory,
  revertPrompt,
  getPrompt,
  type Prompt,
  type PromptVersion
} from "@/services/prompt-studio"
import { useConfirmDanger } from "@/components/Common/confirm-danger"
import { Button } from "@/components/Common/Button"
import { usePromptStudioStore } from "@/store/prompt-studio"

type VersionHistoryDrawerProps = {
  open: boolean
  promptId: number | null
  onClose: () => void
  onRevertSuccess?: (context: {
    promptId: number
    revertedVersion: number
  }) => void
}

export const VersionHistoryDrawer: React.FC<VersionHistoryDrawerProps> = ({
  open,
  promptId,
  onClose,
  onRevertSuccess
}) => {
  const { t } = useTranslation(["settings", "common"])
  const queryClient = useQueryClient()
  const confirmDanger = useConfirmDanger()
  const [selectedVersionId, setSelectedVersionId] = React.useState<number | null>(
    null
  )

  const selectedProjectId = usePromptStudioStore((s) => s.selectedProjectId)

  const { data: promptResponse } = useQuery({
    queryKey: ["prompt-studio", "prompt", promptId],
    queryFn: () => getPrompt(promptId!),
    enabled: open && promptId !== null
  })

  const currentPrompt: Prompt | null = (promptResponse as any)?.data?.data ?? null
  const currentVersion = currentPrompt?.version_number

  const { data: historyResponse, status: historyStatus } = useQuery({
    queryKey: ["prompt-studio", "prompt-history", promptId],
    queryFn: () => getPromptHistory(promptId!),
    enabled: open && promptId !== null
  })

  const versions: PromptVersion[] = (historyResponse as any)?.data?.data ?? []
  const sortedVersions = React.useMemo(
    () => [...versions].sort((a, b) => b.version_number - a.version_number),
    [versions]
  )

  React.useEffect(() => {
    if (!open) {
      setSelectedVersionId(null)
      return
    }
    if (sortedVersions.length === 0) {
      return
    }

    setSelectedVersionId((current) => {
      const existingSelection =
        current !== null
          ? sortedVersions.find((version) => version.id === current) ?? null
          : null

      if (
        existingSelection &&
        existingSelection.version_number !== currentVersion
      ) {
        return current
      }
      const preferredVersion =
        sortedVersions.find(
          (version) => version.version_number !== currentVersion
        ) ?? sortedVersions[0]
      return preferredVersion.id
    })
  }, [open, sortedVersions, currentVersion])

  const selectedVersionPromptId =
    selectedVersionId !== null && selectedVersionId !== promptId
      ? selectedVersionId
      : null

  const {
    data: selectedVersionResponse,
    status: selectedVersionStatus,
    error: selectedVersionError
  } = useQuery({
    queryKey: ["prompt-studio", "prompt-version-preview", selectedVersionPromptId],
    queryFn: () => getPrompt(selectedVersionPromptId!),
    enabled: open && selectedVersionPromptId !== null
  })

  const selectedVersionPrompt: Prompt | null =
    selectedVersionId === promptId
      ? currentPrompt
      : ((selectedVersionResponse as any)?.data?.data ?? null)

  const selectedVersionMeta =
    sortedVersions.find((version) => version.id === selectedVersionId) ?? null

  const revertMutation = useMutation({
    mutationFn: (version: number) => revertPrompt(promptId!, version),
    onSuccess: (_result, revertedVersion) => {
      queryClient.invalidateQueries({
        queryKey: ["prompt-studio", "prompts", selectedProjectId]
      })
      queryClient.invalidateQueries({
        queryKey: ["prompt-studio", "prompt", promptId]
      })
      queryClient.invalidateQueries({
        queryKey: ["prompt-studio", "prompt-history", promptId]
      })
      notification.success({
        message: t("managePrompts.studio.prompts.revertSuccess", {
          defaultValue: "Prompt reverted"
        }),
        description: t("managePrompts.studio.prompts.revertSuccessDesc", {
          defaultValue: "A new version has been created from the selected version."
        })
      })
      if (promptId !== null) {
        onRevertSuccess?.({ promptId, revertedVersion })
      }
    },
    onError: (error: any) => {
      notification.error({
        message: t("common:error", { defaultValue: "Error" }),
        description: error?.message || t("common:unknownError")
      })
    }
  })

  const handleRevert = async (version: PromptVersion) => {
    const ok = await confirmDanger({
      title: t("managePrompts.studio.prompts.revertConfirmTitle", {
        defaultValue: "Revert to version {{version}}?",
        version: version.version_number
      }),
      content: t("managePrompts.studio.prompts.revertConfirmContent", {
        defaultValue:
          "This will create a new version with the content from version {{version}}. The current version will be preserved in history.",
        version: version.version_number
      }),
      okText: t("managePrompts.studio.prompts.revertBtn", {
        defaultValue: "Revert"
      }),
      cancelText: t("common:cancel", { defaultValue: "Cancel" })
    })

    if (ok) {
      revertMutation.mutate(version.version_number)
    }
  }

  const formatDate = (dateStr?: string) => {
    if (!dateStr) return "-"
    return new Date(dateStr).toLocaleString()
  }

  const renderPromptPreviewCard = (
    label: string,
    prompt: Prompt | null,
    versionLabel: string,
    accentClassName: string
  ) => {
    const sections: Array<{ label: string; content: string }> = []

    if (prompt?.system_prompt?.trim()) {
      sections.push({
        label: t("managePrompts.form.systemPrompt.shortLabel", {
          defaultValue: "System"
        }),
        content: prompt.system_prompt.trim()
      })
    }

    if (prompt?.user_prompt?.trim()) {
      sections.push({
        label: t("managePrompts.form.userPrompt.shortLabel", {
          defaultValue: "User"
        }),
        content: prompt.user_prompt.trim()
      })
    }

    if (prompt?.prompt_format === "structured" && prompt.prompt_definition) {
      sections.push({
        label: t("managePrompts.studio.prompts.structuredDefinition", {
          defaultValue: "Structured definition"
        }),
        content: JSON.stringify(prompt.prompt_definition, null, 2)
      })
    }

    if (sections.length === 0) {
      sections.push({
        label: t("managePrompts.studio.prompts.previewLabel", {
          defaultValue: "Preview"
        }),
        content: t("managePrompts.studio.prompts.previewEmpty", {
          defaultValue: "No prompt content is stored for this version."
        })
      })
    }

    return (
      <div className={`rounded-md border p-3 ${accentClassName}`}>
        <div className="mb-3 flex items-center justify-between gap-2">
          <div>
            <p className="text-sm font-medium text-text">{label}</p>
            <p className="text-xs text-text-muted">{versionLabel}</p>
          </div>
          {prompt?.change_description && <Tag>{prompt.change_description}</Tag>}
        </div>
        <div className="space-y-3">
          {sections.map((section) => (
            <div key={`${label}-${section.label}`}>
              <p className="mb-1 text-xs font-medium uppercase tracking-wide text-text-muted">
                {section.label}
              </p>
              <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded bg-bg p-2 text-xs text-text">
                {section.content}
              </pre>
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <Drawer
      open={open}
      onClose={onClose}
      title={
        <span className="flex items-center gap-2">
          <History className="size-5" />
          {t("managePrompts.studio.prompts.versionHistoryTitle", {
            defaultValue: "Version History"
          })}
        </span>
      }
      styles={{ wrapper: { width: 480 } }}
      destroyOnHidden
    >
      {historyStatus === "pending" && <Skeleton paragraph={{ rows: 6 }} />}

      {historyStatus === "success" && versions.length === 0 && (
        <Empty
          description={t("managePrompts.studio.prompts.noVersionHistory", {
            defaultValue: "No version history available"
          })}
        />
      )}

      {historyStatus === "success" && versions.length > 0 && (
        <div className="space-y-6">
          <Timeline
            items={sortedVersions.map((version) => {
              const isCurrent = version.version_number === currentVersion
              const isSelected = version.id === selectedVersionId

              return {
                color: isCurrent ? "green" : "gray",
                dot: isCurrent ? (
                  <CheckCircle2 className="size-4 text-success" />
                ) : undefined,
                children: (
                  <div
                    className={`rounded-md border p-3 transition ${
                      isCurrent
                        ? "border-success/30 bg-success/5"
                        : "border-border bg-surface2/50"
                    } ${isSelected ? "ring-2 ring-primary/20" : ""}`}
                    role="button"
                    tabIndex={0}
                    onClick={() => setSelectedVersionId(version.id)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" || event.key === " ") {
                        event.preventDefault()
                        setSelectedVersionId(version.id)
                      }
                    }}
                  >
                    <div className="mb-2 flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Tag color={isCurrent ? "green" : "default"}>
                          v{version.version_number}
                        </Tag>
                        {isCurrent && (
                          <span className="text-xs font-medium text-success">
                            {t("managePrompts.studio.prompts.currentVersion", {
                              defaultValue: "Current"
                            })}
                          </span>
                        )}
                      </div>
                      {!isCurrent && (
                        <Button
                          type="ghost"
                          size="sm"
                          onClick={(event) => {
                            event.stopPropagation()
                            void handleRevert(version)
                          }}
                          loading={revertMutation.isPending}
                        >
                          <Undo2 className="mr-1 size-3" />
                          {t("managePrompts.studio.prompts.revertBtn", {
                            defaultValue: "Revert"
                          })}
                        </Button>
                      )}
                    </div>

                    <div className="space-y-1">
                      <p className="text-sm font-medium">{version.name}</p>
                      {version.change_description && (
                        <p className="text-sm text-text-muted">
                          {version.change_description}
                        </p>
                      )}
                      <p className="text-xs text-text-muted">
                        {formatDate(version.created_at)}
                      </p>
                      {isSelected && (
                        <p className="text-xs text-primary">
                          {t("managePrompts.studio.prompts.previewSelectedHint", {
                            defaultValue: "Previewing this version below"
                          })}
                        </p>
                      )}
                    </div>
                  </div>
                )
              }
            })}
          />

          {selectedVersionMeta && (
            <div className="space-y-3 border-t border-border pt-4">
              <div>
                <p className="text-sm font-medium text-text">
                  {t("managePrompts.studio.prompts.previewTitle", {
                    defaultValue: "Version preview"
                  })}
                </p>
                <p className="text-xs text-text-muted">
                  {t("managePrompts.studio.prompts.previewDescription", {
                    defaultValue:
                      "Compare the selected version with the current prompt before restoring it."
                  })}
                </p>
              </div>

              {selectedVersionStatus === "pending" &&
                selectedVersionPromptId !== null && (
                  <Skeleton paragraph={{ rows: 4 }} />
                )}

              {selectedVersionStatus === "error" &&
                selectedVersionPromptId !== null && (
                  <Empty
                    description={
                      selectedVersionError instanceof Error
                        ? selectedVersionError.message
                        : t("managePrompts.studio.prompts.previewErrorTitle", {
                            defaultValue: "Unable to load selected version preview"
                          })
                    }
                  />
                )}

              {selectedVersionPrompt && currentPrompt && (
                <div className="grid gap-3">
                  {renderPromptPreviewCard(
                    t("managePrompts.studio.prompts.previewSelectedVersion", {
                      defaultValue: "Selected version"
                    }),
                    selectedVersionPrompt,
                    `v${selectedVersionMeta.version_number}`,
                    "border-primary/20 bg-primary/5"
                  )}
                  {renderPromptPreviewCard(
                    t("managePrompts.studio.prompts.previewCurrentVersion", {
                      defaultValue: "Current version"
                    }),
                    currentPrompt,
                    `v${currentVersion ?? "-"}`,
                    "border-border bg-surface2/50"
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </Drawer>
  )
}
