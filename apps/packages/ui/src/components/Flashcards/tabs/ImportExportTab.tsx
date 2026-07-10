import React from "react"
import { Card, Segmented, Typography } from "antd"
import { useTranslation } from "react-i18next"

import type { FlashcardsGenerateIntent } from "@/services/tldw/flashcards-generate-handoff"
import type { StudyPackIntent } from "@/services/tldw/study-pack-handoff"
import type { SourceReviewFlashcardsIntent } from "@/services/tldw/source-review-handoff"

import { StudyPackCreateDrawer } from "../components/StudyPackCreateDrawer"
import { useImportLimitsQuery } from "../hooks"
import { ImageOcclusionTransferPanel } from "./ImageOcclusionTransferPanel"
import { ExportPanel } from "./ImportExport/ExportPanel"
import { GeneratePanel } from "./ImportExport/GeneratePanel"
import { ImportPanel } from "./ImportExport/ImportPanel"
import { StudyPackPanel } from "./ImportExport/StudyPackPanel"
import {
  guardInterpolatedText,
  normalizeImportLimits,
  type TransferActionSummary,
  type TransferActionSummaryInput
} from "./ImportExport/shared"

const { Text, Title } = Typography
export type TransferTaskKey = "create" | "import" | "export"

const getGenerateIntentToken = (
  intent: FlashcardsGenerateIntent | null | undefined
): string | null => {
  if (!intent) return null
  return JSON.stringify([
    intent.text,
    intent.sourceType ?? "",
    intent.sourceId ?? "",
    intent.sourceTitle ?? "",
    intent.conversationId ?? "",
    intent.messageId ?? ""
  ])
}

const getStudyPackIntentToken = (
  intent: StudyPackIntent | null | undefined
): string | null => {
  if (!intent) return null
  return JSON.stringify([
    intent.title,
    intent.sourceItems.map((item) => [
      item.sourceType,
      item.sourceId,
      item.sourceTitle ?? "",
      item.excerptText ?? "",
      item.locator ? JSON.stringify(item.locator) : ""
    ])
  ])
}

const getExportHandoffToken = (
  deckId: number | null | undefined,
  handoffKey: string | null | undefined
): string | null => {
  if (deckId == null || !handoffKey) return null
  return `${deckId}:${handoffKey}`
}

/**
 * Import/Export tab for flashcards.
 */
type ImportExportTabProps = {
  generateIntent?: FlashcardsGenerateIntent | null
  sourceReviewIntent?: SourceReviewFlashcardsIntent | null
  studyPackIntent?: StudyPackIntent | null
  initialTask?: TransferTaskKey | null
  initialTaskHandoffKey?: string | null
  initialExportDeckId?: number | null
  initialExportDeckHandoffKey?: string | null
  onSourceReviewIntentExit?: () => void
}

export const ImportExportTab: React.FC<ImportExportTabProps> = ({
  generateIntent,
  sourceReviewIntent,
  studyPackIntent,
  initialTask = null,
  initialTaskHandoffKey = null,
  initialExportDeckId = null,
  initialExportDeckHandoffKey = null,
  onSourceReviewIntentExit
}) => {
  const { t } = useTranslation(["option", "common"])
  const limitsQuery = useImportLimitsQuery()
  const [activeTask, setActiveTask] = React.useState<TransferTaskKey>(() =>
    initialTask ?? (initialExportDeckId != null && initialExportDeckHandoffKey ? "export" : "create")
  )
  const [lastTransferAction, setLastTransferAction] =
    React.useState<TransferActionSummary | null>(null)
  const [studyPackDrawerOpen, setStudyPackDrawerOpen] = React.useState(false)
  const seenGenerateIntentTokenRef = React.useRef<string | null>(null)
  const seenStudyPackIntentTokenRef = React.useRef<string | null>(null)
  const seenTaskHandoffTokenRef = React.useRef<string | null>(null)
  const seenExportHandoffTokenRef = React.useRef<string | null>(null)
  const importLimitsText = React.useMemo(() => {
    const importLimits = normalizeImportLimits(limitsQuery.data)
    if (!importLimits) return null
    const fallback = [
      `${importLimits.maxLines.toLocaleString()} lines`,
      `${importLimits.maxLineLengthBytes.toLocaleString()} bytes per line`,
      `${importLimits.maxFieldLengthBytes.toLocaleString()} bytes per field`
    ].join(" / ")
    return guardInterpolatedText(
      t("option:flashcards.transferSummaryLimitsValue", {
        defaultValue:
          "{{lines}} lines / {{lineBytes}} bytes per line / {{fieldBytes}} bytes per field",
        lines: importLimits.maxLines.toLocaleString(),
        lineBytes: importLimits.maxLineLengthBytes.toLocaleString(),
        fieldBytes: importLimits.maxFieldLengthBytes.toLocaleString()
      }),
      fallback
    )
  }, [limitsQuery.data, t])

  React.useEffect(() => {
    const token = getStudyPackIntentToken(studyPackIntent)
    if (!token) {
      seenStudyPackIntentTokenRef.current = null
      return
    }
    if (token === seenStudyPackIntentTokenRef.current) return

    seenStudyPackIntentTokenRef.current = token
    setActiveTask("create")
    setStudyPackDrawerOpen(true)
  }, [studyPackIntent])

  React.useEffect(() => {
    const token = getGenerateIntentToken(generateIntent)
    if (!token) {
      seenGenerateIntentTokenRef.current = null
      return
    }
    if (token === seenGenerateIntentTokenRef.current) return

    seenGenerateIntentTokenRef.current = token
    setActiveTask("create")
  }, [generateIntent])

  React.useEffect(() => {
    if (sourceReviewIntent) setActiveTask("create")
  }, [sourceReviewIntent])

  const handleTaskChange = React.useCallback(
    (value: string | number) => {
      const nextTask = value as TransferTaskKey
      if (
        activeTask === "create" &&
        nextTask !== "create" &&
        sourceReviewIntent
      ) {
        onSourceReviewIntentExit?.()
      }
      setActiveTask(nextTask)
    },
    [activeTask, onSourceReviewIntentExit, sourceReviewIntent]
  )

  React.useEffect(() => {
    if (!initialTask || !initialTaskHandoffKey) {
      seenTaskHandoffTokenRef.current = null
      return
    }
    const token = `${initialTask}:${initialTaskHandoffKey}`
    if (token === seenTaskHandoffTokenRef.current) return

    seenTaskHandoffTokenRef.current = token
    setActiveTask(initialTask)
  }, [initialTask, initialTaskHandoffKey])

  React.useEffect(() => {
    const token = getExportHandoffToken(initialExportDeckId, initialExportDeckHandoffKey)
    if (!token) {
      seenExportHandoffTokenRef.current = null
      return
    }
    if (token === seenExportHandoffTokenRef.current) return

    seenExportHandoffTokenRef.current = token
    setActiveTask("export")
  }, [initialExportDeckHandoffKey, initialExportDeckId])

  const handleTransferAction = React.useCallback((summary: TransferActionSummaryInput) => {
    setLastTransferAction({
      ...summary,
      at: new Date().toISOString()
    })
  }, [])

  const transferTaskOptions = React.useMemo(
    () => [
      {
        label: t("option:flashcards.transferTaskCreate", {
          defaultValue: "Create cards"
        }),
        value: "create" as const
      },
      {
        label: t("option:flashcards.transferTaskImport", {
          defaultValue: "Import file"
        }),
        value: "import" as const
      },
      {
        label: t("option:flashcards.transferTaskExport", {
          defaultValue: "Export backup"
        }),
        value: "export" as const
      }
    ],
    [t]
  )

  const lastTransferActionText = React.useMemo(() => {
    if (!lastTransferAction) {
      return t("option:flashcards.transferSummaryNoAction", {
        defaultValue: "No import/export actions yet in this session."
      })
    }
    const areaLabel =
      lastTransferAction.area === "import"
        ? t("option:flashcards.importTitle", { defaultValue: "Import Flashcards" })
        : lastTransferAction.area === "export"
          ? t("option:flashcards.exportTitle", { defaultValue: "Export Flashcards" })
          : lastTransferAction.area === "occlusion"
            ? t("option:flashcards.occlusionTitle", {
                defaultValue: "Image Occlusion"
              })
            : t("option:flashcards.generateTitle", {
                defaultValue: "Generate Flashcards"
              })
    const actionTime = new Date(lastTransferAction.at).toLocaleTimeString()
    const fallback = `${areaLabel} · ${lastTransferAction.message} · ${actionTime}`
    return guardInterpolatedText(
      t("option:flashcards.transferSummaryLastAction", {
        defaultValue: "{{area}} · {{message}} · {{time}}",
        area: areaLabel,
        message: lastTransferAction.message,
        time: actionTime
      }),
      fallback
    )
  }, [lastTransferAction, t])

  return (
    <div className="flex flex-col gap-4">
      <Card
        title={t("option:flashcards.transferTaskSwitcherTitle", {
          defaultValue: "Task"
        })}
        data-testid="flashcards-transfer-task-switcher"
      >
        <Segmented
          block
          aria-label={t("option:flashcards.transferTaskSwitcherAria", {
            defaultValue: "Create, import, or export task"
          })}
          options={transferTaskOptions}
          value={activeTask}
          onChange={handleTaskChange}
        />
      </Card>
      <section
        aria-labelledby="flashcards-create-generate-heading"
        data-testid="flashcards-create-generate-section"
        role="region"
      >
        <div className="mb-2">
          <Title id="flashcards-create-generate-heading" level={5} className="!mb-0">
            {t("option:flashcards.createGenerateSectionTitle", {
              defaultValue: "Create and generate"
            })}
          </Title>
          <Text type="secondary">
            {t("option:flashcards.createGenerateSectionDescription", {
              defaultValue: "Start new study material from study packs, source text, or images."
            })}
          </Text>
        </div>
        <section
          className={
            activeTask === "create" ? "grid gap-4 grid-cols-1 xl:grid-cols-2" : "hidden"
          }
          data-testid="flashcards-create-task-panel"
        >
          <Card
            className="xl:col-span-2"
            title={t("option:flashcards.studyPackLauncherTitle", {
              defaultValue: "Study packs"
            })}
            data-testid="flashcards-study-pack-launcher"
          >
            <StudyPackPanel onLaunch={() => setStudyPackDrawerOpen(true)} />
          </Card>
          <Card
            title={t("option:flashcards.generateTitle", {
              defaultValue: "Generate Flashcards"
            })}
          >
            <GeneratePanel
              initialIntent={generateIntent || null}
              sourceReviewIntent={sourceReviewIntent || null}
              onTransferAction={handleTransferAction}
            />
          </Card>
          <Card
            title={t("option:flashcards.occlusionTitle", {
              defaultValue: "Image Occlusion"
            })}
          >
            <ImageOcclusionTransferPanel onTransferAction={handleTransferAction} />
          </Card>
        </section>
      </section>
      <section
        aria-labelledby="flashcards-import-export-heading"
        data-testid="flashcards-import-export-section"
        role="region"
      >
        <div className="mb-2">
          <Title id="flashcards-import-export-heading" level={5} className="!mb-0">
            {t("option:flashcards.importExportSectionTitle", {
              defaultValue: "Import and export"
            })}
          </Title>
          <Text type="secondary">
            {t("option:flashcards.importExportSectionDescription", {
              defaultValue: "Move cards in and out without changing the flashcards route."
            })}
          </Text>
        </div>
        <Card
          title={t("option:flashcards.transferSummaryTitle", {
            defaultValue: "Import/export summary"
          })}
          data-testid="flashcards-transfer-summary"
          size="small"
          className="mb-4"
        >
          <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
            <div data-testid="flashcards-transfer-summary-formats">
              <Text strong className="block">
                {t("option:flashcards.transferSummaryFormatsLabel", {
                  defaultValue: "Supported formats"
                })}
              </Text>
              <Text type="secondary">
                {t("option:flashcards.transferSummaryFormatsValue", {
                  defaultValue:
                    "Import: CSV, TSV, JSON, JSONL, Structured Q&A, APKG · Author: Generate, Image Occlusion · Export: TSV, CSV, JSON, APKG"
                })}
              </Text>
            </div>
            <div data-testid="flashcards-transfer-summary-limits">
              <Text strong className="block">
                {t("option:flashcards.transferSummaryLimitsLabel", {
                  defaultValue: "Current import limits"
                })}
              </Text>
              <Text type="secondary">
                {importLimitsText
                  ? importLimitsText
                  : t("option:flashcards.transferSummaryLimitsUnknown", {
                      defaultValue: "Limits unavailable"
                    })}
              </Text>
            </div>
            <div data-testid="flashcards-transfer-summary-last-action">
              <Text strong className="block">
                {t("option:flashcards.transferSummaryLastActionLabel", {
                  defaultValue: "Last action"
                })}
              </Text>
              <Text
                type={
                  lastTransferAction?.status === "error"
                    ? "danger"
                    : lastTransferAction?.status === "warning"
                      ? "warning"
                      : "secondary"
                }
              >
                {lastTransferActionText}
              </Text>
            </div>
          </div>
        </Card>
        <section
          className={activeTask === "import" ? "" : "hidden"}
          data-testid="flashcards-import-task-panel"
        >
          <Card
            title={t("option:flashcards.importTitle", {
              defaultValue: "Import Flashcards"
            })}
          >
            <ImportPanel onTransferAction={handleTransferAction} />
          </Card>
        </section>
        <section
          className={activeTask === "export" ? "" : "hidden"}
          data-testid="flashcards-export-task-panel"
        >
          <Card
            title={t("option:flashcards.exportTitle", {
              defaultValue: "Export Flashcards"
            })}
          >
            <ExportPanel
              onTransferAction={handleTransferAction}
              initialDeckId={initialExportDeckId}
              initialDeckHandoffKey={initialExportDeckHandoffKey}
            />
          </Card>
        </section>
      </section>
      <StudyPackCreateDrawer
        open={studyPackDrawerOpen}
        onClose={() => setStudyPackDrawerOpen(false)}
        initialIntent={studyPackIntent || null}
      />
    </div>
  )
}

export default ImportExportTab
