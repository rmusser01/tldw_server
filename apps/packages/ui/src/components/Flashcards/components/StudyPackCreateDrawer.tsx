import React from "react"
import { Button, Drawer, Input, Select, Space, Typography } from "antd"
import { ChevronDown, ChevronRight, FileText, Film, MessageSquare, Trash2 } from "lucide-react"
import { useNavigate } from "react-router-dom"
import { useTranslation } from "react-i18next"

import { useAntdMessage } from "@/hooks/useAntdMessage"
import type {
  StudyPackSourceSelection,
  StudyPackSourceType,
  StudyPackSummaryResponse
} from "@/services/flashcards"
import type { StudyPackIntent } from "@/services/tldw/study-pack-handoff"
import { useStudyPackCreateMutation, useStudyPackJobQuery } from "../hooks"

const { Text } = Typography

type StudyPackCreateDrawerProps = {
  open: boolean
  onClose: () => void
  initialIntent?: StudyPackIntent | null
  onCreated?: (pack: StudyPackSummaryResponse) => void
}

const DEFAULT_SOURCE_TYPE: StudyPackSourceType = "media"

const SOURCE_TYPE_ICONS: Record<StudyPackSourceType, React.ComponentType<{ className?: string }>> = {
  media: Film,
  note: FileText,
  message: MessageSquare
}

const normalizeIntentSources = (
  intent: StudyPackIntent | null | undefined
): StudyPackSourceSelection[] =>
  Array.isArray(intent?.sourceItems)
    ? intent.sourceItems.map((item) => ({
        source_type: item.sourceType,
        source_id: item.sourceId,
        source_title: item.sourceTitle,
        excerpt_text: item.excerptText,
        locator: item.locator
      }))
    : []

export const StudyPackCreateDrawer: React.FC<StudyPackCreateDrawerProps> = ({
  open,
  onClose,
  initialIntent,
  onCreated
}) => {
  const { t } = useTranslation(["option", "common"])
  const navigate = useNavigate()
  const message = useAntdMessage()
  const createMutation = useStudyPackCreateMutation()
  const handledTerminalJobIdRef = React.useRef<number | null>(null)

  const [title, setTitle] = React.useState("")
  const [sourceItems, setSourceItems] = React.useState<StudyPackSourceSelection[]>([])
  const [sourceType, setSourceType] = React.useState<StudyPackSourceType>(DEFAULT_SOURCE_TYPE)
  const [sourceId, setSourceId] = React.useState("")
  const [sourceTitle, setSourceTitle] = React.useState("")
  const [jobId, setJobId] = React.useState<number | null>(null)
  // Typing a raw ID is the fallback. Hide it when the drawer opens with
  // sources already chosen elsewhere (NE-08).
  const [manualEntryOpen, setManualEntryOpen] = React.useState(true)

  const jobQuery = useStudyPackJobQuery(jobId, {
    enabled: open && jobId != null
  })
  const sourceTypeOptions = React.useMemo<Array<{ label: string; value: StudyPackSourceType }>>(
    () => [
      {
        label: t("option:flashcards.studyPackSourceType.media", {
          defaultValue: "Media"
        }),
        value: "media"
      },
      {
        label: t("option:flashcards.studyPackSourceType.note", {
          defaultValue: "Note"
        }),
        value: "note"
      },
      {
        label: t("option:flashcards.studyPackSourceType.message", {
          defaultValue: "Message"
        }),
        value: "message"
      }
    ],
    [t]
  )
  const sourceTypeLabel = React.useCallback(
    (type: StudyPackSourceType) =>
      sourceTypeOptions.find((option) => option.value === type)?.label ?? type,
    [sourceTypeOptions]
  )

  React.useEffect(() => {
    if (!open) return
    setTitle(initialIntent?.title ?? "")
    const intentSources = normalizeIntentSources(initialIntent)
    setSourceItems(intentSources)
    setManualEntryOpen(intentSources.length === 0)
    setSourceType(DEFAULT_SOURCE_TYPE)
    setSourceId("")
    setSourceTitle("")
    setJobId(null)
    handledTerminalJobIdRef.current = null
  }, [initialIntent, open])

  React.useEffect(() => {
    const response = jobQuery.data
    if (!response) return
    if (handledTerminalJobIdRef.current === response.job.id) return

    if (response.job.status === "completed") {
      handledTerminalJobIdRef.current = response.job.id
      const pack = response.study_pack
      if (!pack?.deck_id) {
        message.error(
          t("option:flashcards.studyPackResultUnavailable", {
            defaultValue: "Study pack completed, but its review deck is unavailable."
          })
        )
        setJobId(null)
        return
      }
      onCreated?.(pack)
      onClose()
      navigate(`/flashcards?tab=review&deck_id=${pack.deck_id}`, {
        replace: true
      })
      return
    }

    if (response.job.status === "failed" || response.job.status === "cancelled") {
      handledTerminalJobIdRef.current = response.job.id
      message.error(
        response.error ||
          t("option:flashcards.studyPackCreateFailed", {
            defaultValue: "Study pack generation failed."
          })
      )
      setJobId(null)
    }
  }, [jobQuery.data, message, navigate, onClose, onCreated, t])

  const trimmedTitle = title.trim()
  const trimmedSourceId = sourceId.trim()
  const trimmedSourceTitle = sourceTitle.trim()
  const hasPendingJob = jobId != null
  const canSubmit =
    trimmedTitle.length > 0 &&
    sourceItems.length > 0 &&
    !createMutation.isPending &&
    !hasPendingJob
  const pendingJobMessage = jobQuery.isError
    ? t("option:flashcards.studyPackJobStatusUnavailable", {
        defaultValue: "Unable to check study pack progress. Checking again shortly."
      })
    : jobQuery.data?.job.status === "running"
      ? t("option:flashcards.studyPackJobRunning", {
          defaultValue: "Creating your study pack."
        })
      : jobQuery.data?.job.status === "queued"
        ? t("option:flashcards.studyPackJobQueued", {
            defaultValue: "Study pack queued. Waiting for generation to start."
          })
        : t("option:flashcards.studyPackJobAccepted", {
            defaultValue: "Study pack request accepted. Waiting for a status update."
          })

  const handleAddSource = React.useCallback(() => {
    if (!trimmedSourceId) return

    setSourceItems((current) => [
      ...current,
      {
        source_type: sourceType,
        source_id: trimmedSourceId,
        source_title: trimmedSourceTitle || undefined
      }
    ])
    setSourceId("")
    setSourceTitle("")
  }, [sourceType, trimmedSourceId, trimmedSourceTitle])

  const handleRemoveSource = React.useCallback((index: number) => {
    setSourceItems((current) => current.filter((_, currentIndex) => currentIndex !== index))
  }, [])

  const handleSubmit = React.useCallback(async () => {
    if (!canSubmit) return

    try {
      const accepted = await createMutation.mutateAsync({
        title: trimmedTitle,
        source_items: sourceItems,
        deck_mode: "new"
      })
      setJobId(accepted.job.id)
    } catch (error) {
      const detail =
        typeof error === "object" &&
        error !== null &&
        "response" in error &&
        typeof (error as { response?: { data?: { detail?: unknown } } }).response?.data?.detail === "string"
          ? (error as { response?: { data?: { detail?: string } } }).response?.data?.detail
          : null
      message.error(
        detail ||
          t("option:flashcards.studyPackCreateFailed", {
            defaultValue: "Study pack generation failed."
          })
      )
    }
  }, [canSubmit, createMutation, message, sourceItems, t, trimmedTitle])

  const drawerTitle = t("option:flashcards.studyPackDrawerTitle", {
    defaultValue: "Create study pack"
  })

  return (
    <Drawer
      open={open}
      title={drawerTitle}
      onClose={onClose}
      destroyOnClose
      size="large"
      footer={
        <Space>
          <Button onClick={onClose}>
            {t("common:cancel", { defaultValue: "Cancel" })}
          </Button>
          <Button
            type="primary"
            onClick={() => {
              void handleSubmit()
            }}
            disabled={!canSubmit}
            loading={createMutation.isPending || hasPendingJob}
          >
            {t("option:flashcards.studyPackCreateButton", {
              defaultValue: "Create study pack"
            })}
          </Button>
        </Space>
      }
    >
      <div className="space-y-4">
        {hasPendingJob ? (
          <Text role="status" aria-live="polite" aria-atomic="true" className="block">
            {pendingJobMessage}
          </Text>
        ) : null}
        <div className="space-y-1">
          <Text strong>Title</Text>
          <Input
            aria-label="Title"
            value={title}
            onChange={(event) => setTitle(event.target.value)}
            placeholder={t("option:flashcards.studyPackTitlePlaceholder", {
              defaultValue: "Study pack title"
            })}
          />
        </div>

        <div className="rounded border border-border bg-surface p-3">
          <div className="mb-3 flex items-center justify-between gap-2">
            <Text strong>
              {t("option:flashcards.studyPackSourcesLabel", {
                defaultValue: "Sources"
              })}
            </Text>
            <Text type="secondary">
              {t("option:flashcards.studyPackSourcesCount", {
                defaultValue: "{{count}} selected",
                count: sourceItems.length
              })}
            </Text>
          </div>

          <button
            type="button"
            className="mb-2 inline-flex items-center gap-1 text-sm text-text-muted hover:text-text"
            aria-expanded={manualEntryOpen}
            aria-controls="study-pack-manual-source"
            onClick={() => setManualEntryOpen((current) => !current)}
            data-testid="study-pack-manual-source-toggle"
          >
            {manualEntryOpen ? (
              <ChevronDown className="size-4" aria-hidden="true" />
            ) : (
              <ChevronRight className="size-4" aria-hidden="true" />
            )}
            {t("option:flashcards.studyPackManualSourceToggle", {
              defaultValue: "Add a source by ID"
            })}
          </button>
          {manualEntryOpen ? (
            <div id="study-pack-manual-source">
              <Space.Compact className="w-full">
                <Select
                  aria-label="Source Type"
                  value={sourceType}
                  onChange={(value) => setSourceType(value)}
                  options={sourceTypeOptions}
                  className="min-w-[120px]"
                />
                <Input
                  aria-label={t("option:flashcards.studyPackSourceIdAdvancedLabel", {
                    defaultValue: "Source ID (advanced/manual)"
                  })}
                  value={sourceId}
                  onChange={(event) => setSourceId(event.target.value)}
                  placeholder={t("option:flashcards.studyPackSourceIdAdvancedPlaceholder", {
                    defaultValue: "Source ID (advanced/manual)"
                  })}
                />
                <Input
                  aria-label="Source Title"
                  value={sourceTitle}
                  onChange={(event) => setSourceTitle(event.target.value)}
                  placeholder={t("option:flashcards.studyPackSourceTitlePlaceholder", {
                    defaultValue: "Optional title"
                  })}
                />
                <Button onClick={handleAddSource}>
                  {t("common:add", { defaultValue: "Add source" })}
                </Button>
              </Space.Compact>
              <Text
                type="secondary"
                className="mt-2 block text-xs"
                data-testid="study-pack-source-id-guidance"
              >
                {t("option:flashcards.studyPackSourceIdGuidance", {
                  defaultValue:
                    "Advanced/manual source reference. Choose Media, Note, or Message, then paste the item ID from that source."
                })}
              </Text>
            </div>
          ) : null}

          <div className="mt-3 space-y-2">
            {sourceItems.length === 0 ? (
              <Text type="secondary">
                {t("option:flashcards.studyPackSourcesEmpty", {
                  defaultValue:
                    "Add at least one supported source: Media, Note, or Message."
                })}
              </Text>
            ) : (
              sourceItems.map((item, index) => {
                const TypeIcon = SOURCE_TYPE_ICONS[item.source_type] ?? FileText
                const typeLabel = sourceTypeLabel(item.source_type)
                const title = item.source_title?.trim()
                return (
                  <div
                    key={`${item.source_type}-${item.source_id}-${index}`}
                    className="flex items-start justify-between gap-3 rounded border border-border bg-background p-2"
                    data-testid={`study-pack-source-${index}`}
                  >
                    <div className="flex min-w-0 items-start gap-2">
                      <TypeIcon className="mt-0.5 size-4 shrink-0 text-text-muted" aria-hidden="true" />
                      <div className="min-w-0">
                        {/* Name a source by its title; fall back to the ID
                            only when nothing else identifies it (NE-08). */}
                        <Text className="block truncate">
                          {title || `${typeLabel} ${item.source_id}`}
                        </Text>
                        {title ? (
                          <Text type="secondary" className="block text-xs">
                            {typeLabel}
                          </Text>
                        ) : null}
                      </div>
                    </div>
                    <Button
                      type="text"
                      aria-label={t("common:remove", { defaultValue: "Remove" })}
                      icon={<Trash2 className="size-4" />}
                      onClick={() => handleRemoveSource(index)}
                    />
                  </div>
                )
              })
            )}
          </div>
        </div>
      </div>
    </Drawer>
  )
}

export default StudyPackCreateDrawer
